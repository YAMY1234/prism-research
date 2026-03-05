# Prism-Research 架构设计文档

## 项目概述

Prism 是一个多模型 LLM 服务框架，通过对 SGLang 进行最小侵入式的 monkey patch，实现：
- **多模型共享 GPU**：多个模型可以在同一 GPU 上时分复用
- **弹性内存管理**：通过 kvcached 实现 KV cache 的动态分配/回收
- **智能调度**：GPU Scheduler 根据 SLO 和资源状况调度请求

**运行环境**：Docker 容器中 pip 安装的 SGLang + 挂载的 Prism 代码（patch）

## 目录结构

```
python/prism/
├── __init__.py                 # 入口，自动应用 patches
├── io_struct.py                # Prism 数据结构（扩展 SGLang 的 GenerateReqInput 等）
├── launch.py                   # 启动脚本入口
│
├── patches/                    # SGLang Monkey Patches
│   ├── __init__.py             # Patch 加载器（import 时自动执行）
│   ├── scheduler_patch.py      # Scheduler 扩展（Redis 轮询、activate/deactivate）
│   ├── server_args_patch.py    # ServerArgs 转换（MultiModelServerArgs → ServerArgs）
│   ├── tp_worker_patch.py      # TpModelWorker 扩展（模型激活/停用）
│   ├── model_runner_patch.py   # ModelRunner 扩展（弹性内存）
│   └── memory_pool_patch.py    # MHATokenToKVPoolElastic（kvcached KV cache）
│
├── multi_model/                # 多模型服务核心
│   ├── server.py               # 主服务启动逻辑（进程编排）
│   ├── server_args.py          # MultiModelServerArgs（多模型配置）
│   ├── managers.py             # Scheduler/Detokenizer 子进程管理
│   ├── port_args.py            # PrismPortArgs（IPC 通道配置）
│   ├── request_handler.py      # HTTP 请求处理（路由到 Redis）
│   ├── engine.py               # 引擎管理
│   │
│   └── scheduling/             # 调度系统
│       ├── gpu/
│       │   ├── gpu_scheduler.py    # GPU Scheduler（请求分发核心）
│       │   ├── request_queue.py    # 请求队列（SLO 优先级堆）
│       │   └── resource_manager.py # 资源管理（内存追踪）
│       ├── controller_global.py    # 全局控制器
│       ├── action.py               # 调度动作（Activate/Deactivate/Resize）
│       └── state.py                # 模型实例状态
│
├── model_runner/               # WorkerPool 模式
│   └── worker_pool_runner.py
│
└── utils/
    └── redis_utils.py          # Redis 客户端（同步 + 异步）
```

---

## 完整调用链路

### 阶段一：启动 (Startup)

```
python -m prism.launch --model-config-file ... --enable-elastic-memory --enable-gpu-scheduler
    │
    ▼
launch.py:main()
    │
    ├─ import prism          # 触发 prism/__init__.py
    │   └─ apply_patches()   # 触发 prism/patches/__init__.py
    │       ├─ apply_scheduler_patch()    # Scheduler 类的方法被替换/新增
    │       │   ├─ Scheduler.event_loop_normal = patched_event_loop_normal
    │       │   ├─ Scheduler.init_request_dispatcher = patched_init_request_dispatcher
    │       │   ├─ Scheduler._prism_recv_generation_requests = ...   (新增)
    │       │   ├─ Scheduler._prism_handle_raw_generate_request = ... (新增)
    │       │   ├─ Scheduler._prism_handle_activate_request = ...    (新增)
    │       │   └─ Scheduler._prism_handle_deactivate_request = ...  (新增)
    │       ├─ apply_server_args_patch()   # ServerArgs.from_multi_model_server_args 新增
    │       ├─ apply_tp_worker_patch()     # TpModelWorker.activate/deactivate 新增
    │       └─ apply_model_runner_patch()  # ModelRunner 初始化逻辑扩展
    │
    ├─ prepare_server_args(sys.argv)  # 解析 CLI → MultiModelServerArgs
    │
    └─ launch_multi_model_server(args)  # server.py 主入口
```

### 阶段二：进程创建 (Process Orchestration)

`launch_multi_model_server()` 在 **主进程** 中编排所有子进程：

```
launch_multi_model_server(args)
    │
    ├─ 1. _launch_model_engines()   为每个 model x placement 创建引擎
    │     │
    │     ├─ 对每个 (model_name, instance_config):
    │     │   ├─ ServerArgs.from_multi_model_server_args()  [patched]
    │     │   │   ├─ 移除 Prism 特有字段
    │     │   │   ├─ 设置 served_model_name = "model_1" (逻辑名)
    │     │   │   ├─ 设置 redis_host, redis_port, redis_db
    │     │   │   ├─ 设置 backend_generate_request_key_prefix
    │     │   │   ├─ 设置 enable_elastic_memory, on, max_memory_pool_size
    │     │   │   └─ 返回标准 SGLang ServerArgs (附加了 Prism 属性)
    │     │   │
    │     │   └─ _launch_single_engine()
    │     │       ├─ PrismPortArgs.init_new()  创建 IPC 文件名
    │     │       └─ mp.Process(target=run_scheduler_process, ...)  启动子进程
    │     │
    │     └─ 返回 engine_info_dict, port_args_dict, ...
    │
    ├─ 2. launch_request_handler()   启动 HTTP → Redis 桥梁
    │     ├─ 检查 Redis 连接
    │     ├─ 清空 Redis 队列
    │     ├─ 关闭 Redis 连接（fork 安全！）
    │     └─ 创建 RequestHandler 实例
    │
    ├─ 3. _launch_gpu_scheduler()    启动 GPU 调度器
    │     └─ mp.Process(target=run_gpu_scheduler_process, ...)
    │
    └─ 4. uvicorn.run()              启动 HTTP Server
```

### 阶段三：Scheduler 子进程初始化

每个模型实例一个 Scheduler 子进程，在 `managers.py:run_scheduler_process()` 中：

```
run_scheduler_process(server_args, port_args, gpu_id, tp_rank=0, ...)
    │
    ├─ import prism.patches          # 子进程中重新 apply patches
    │
    ├─ 转换 PrismPortArgs → SGLang PortArgs
    │   sglang_port_args = SGLangPortArgs(
    │       tokenizer_ipc_name = port_args.request_handler_ipc_name,
    │       scheduler_input_ipc_name = port_args.scheduler_input_ipc_name,
    │       ...
    │   )
    │
    ├─ scheduler = Scheduler(server_args, sglang_port_args, gpu_id, tp_rank, ...)
    │   │
    │   │  SGLang Scheduler.__init__ 内部：
    │   ├─ 创建 ZMQ sockets (recv_from_tokenizer, send_to_detokenizer)
    │   ├─ init_request_dispatcher()  [已被 patched!]
    │   │   ├─ 先调原生: 注册 SGLang 原有请求类型 (FlushCacheReq, AbortReq, ...)
    │   │   └─ 再扩展: 注册 Prism 请求类型
    │   │       ├─ ActivateReqInput  → _prism_handle_activate_request
    │   │       ├─ DeactivateReqInput → _prism_handle_deactivate_request
    │   │       ├─ GenerateReqInput  → _prism_handle_raw_generate_request
    │   │       └─ GetMemPoolSizeReq / GetMemoryUsageReq / ResizeMemPoolReqInput
    │   ├─ 创建 TpModelWorker [已被 patched: 支持弹性内存]
    │   │   └─ ModelRunner.__init__ [已被 patched: 创建 MHATokenToKVPoolElastic]
    │   └─ _activated = False (SGLang 默认)
    │
    ├─ 手动设置 Prism 属性（SGLang 原生没有）：
    │   ├─ scheduler.redis_client = RedisClient(host, port, db)  ← 每个子进程独立连接
    │   ├─ scheduler.model_name = "model_1"
    │   ├─ scheduler._activated = True    ← 覆盖 SGLang 默认
    │   ├─ scheduler._prism_memory_usage_shm = 共享内存
    │   └─ scheduler.idle_sleeper = None  ← 禁用（防止 ZMQ 阻塞）
    │
    └─ scheduler.event_loop_normal()  [已被 patched!]  ← 进入主循环
```

### 阶段四：请求处理主循环 (Event Loop)

**patched_event_loop_normal** 是每个 Scheduler 子进程的主循环：

```
patched_event_loop_normal(self)           # scheduler_patch.py
    │
    while True:
    │
    ├─ Step 1: self.recv_requests()       # SGLang 原生 ZMQ 接收
    │   │  从 tokenizer/request_handler 接收控制消息
    │   │  使用 zmq.NOBLOCK → 非阻塞
    │   └─ 返回: [FlushCacheReq, AbortReq, ActivateReqInput, ...]
    │
    ├─ Step 2: self._prism_recv_generation_requests()  # Prism 新增
    │   │  从 Redis backend queue 读取生成请求
    │   ├─ 检查: tp_rank == 0? _activated? redis_client 存在?
    │   ├─ key = "backend_generate_request_{uuid}:{model_name}"
    │   ├─ redis_client.recv_pyobj_non_block(key, count=32)
    │   │   └─ 底层: redis.lpop(key, count=32)  ← 非阻塞
    │   └─ 返回: [GenerateReqInput, ...]
    │
    │   对每个 Redis 请求调用:
    │   └─ self._prism_handle_raw_generate_request(req)
    │       ├─ 取出 text/input_ids
    │       ├─ 用 self.tokenizer 分词
    │       ├─ 创建 TokenizedGenerateReqInput
    │       └─ self.handle_generate_request(tokenized_req)
    │           └─ 加入 SGLang waiting_queue
    │
    ├─ Step 3: self.process_input_requests(recv_reqs)  # SGLang 原生
    │   │  使用 TypeBasedDispatcher 分发请求
    │   ├─ FlushCacheReq → flush_cache()
    │   ├─ AbortReq → abort_request()
    │   ├─ ActivateReqInput → _prism_handle_activate_request()   [Prism 新增]
    │   ├─ DeactivateReqInput → _prism_handle_deactivate_request() [Prism 新增]
    │   └─ ...
    │
    ├─ Step 4: 如果 _activated:
    │   ├─ batch = self.get_next_batch_to_run()      # SGLang 原生
    │   │   └─ 从 waiting_queue 取请求，组装 batch
    │   │
    │   ├─ 如果有 batch:
    │   │   ├─ result = self.run_batch(batch)          # SGLang → TpModelWorker → ModelRunner
    │   │   ├─ self.process_batch_result(batch, result) # 结果 → detokenizer → HTTP response
    │   │   └─ (可选) 连续 decode 多步
    │   │
    │   └─ 如果没有 batch:
    │       └─ sleep(0.001) + check_memory()
    │
    └─ 如果 not _activated:
        └─ sleep(0.001)  等待激活命令
```

### 阶段五：端到端请求流 (Request Flow)

一个完整的 HTTP 请求从进入到返回：

```
╔══════════════════════════════════════════════════════════════════╗
║ 客户端: POST /generate {"model":"model_1", "text":"Hello"}      ║
╚══════════════════════════════════╦═══════════════════════════════╝
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. HTTP Server (FastAPI/Uvicorn)                     [主进程]     │
│    route: /generate → generate_request(obj)                      │
│    obj = GenerateReqInput(model="model_1", text="Hello", ...)    │
│    obj.arrival_time = time.time()                                │
│    obj.normalize_batch_and_arguments()                           │
└──────────────────────────────────┬───────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. RequestHandler._send_single_request()             [主进程]     │
│    await redis_client.send_pyobj(                                │
│        key="frontend_generate_request_{uuid}:model_1",           │
│        obj=single_request_obj                                    │
│    )                                                             │
│    → 请求序列化后 RPUSH 到 Redis                                  │
│    → 同时 send_to_controller 通知全局控制器                        │
└──────────────────────────────────┬───────────────────────────────┘
                                   ▼  (Redis)
┌──────────────────────────────────────────────────────────────────┐
│ 3. GPU Scheduler                                    [独立进程]    │
│    _recv_requests_from_frontend()                                │
│    → redis_client.recv_pyobj_non_block("frontend_...:model_1")   │
│    → RequestQueue.add_requests([req])                            │
│    → 计算 SLO 优先级                                              │
│    → _admit_requests(): 检查资源，决定放行                         │
│    → _send_to_backend_queue():                                   │
│      redis_client.send_pyobj(                                    │
│          key="backend_generate_request_{uuid}:model_1",          │
│          obj=req                                                 │
│      )                                                           │
└──────────────────────────────────┬───────────────────────────────┘
                                   ▼  (Redis)
┌──────────────────────────────────────────────────────────────────┐
│ 4. Scheduler (model_1 子进程)                       [子进程]      │
│    patched_event_loop_normal() 每轮循环:                          │
│    │                                                             │
│    ├─ _prism_recv_generation_requests()                           │
│    │  redis_client.recv_pyobj_non_block(                          │
│    │      "backend_generate_request_{uuid}:model_1", count=32)   │
│    │  → 得到 [GenerateReqInput]                                   │
│    │                                                             │
│    ├─ _prism_handle_raw_generate_request(req)                     │
│    │  ├─ tokenizer.encode("Hello") → [15496]                     │
│    │  ├─ 创建 TokenizedGenerateReqInput(input_ids=[15496], ...)  │
│    │  └─ handle_generate_request(tokenized_req)                  │
│    │      └─ 加入 self.waiting_queue                             │
│    │                                                             │
│    ├─ get_next_batch_to_run()                                    │
│    │  └─ 从 waiting_queue 组装 ScheduleBatch                     │
│    │                                                             │
│    ├─ run_batch(batch)                                           │
│    │  └─ TpModelWorker.forward_batch()                           │
│    │      └─ ModelRunner.forward()  → GPU 推理                    │
│    │                                                             │
│    └─ process_batch_result(batch, result)                        │
│       └─ send_to_detokenizer → detokenize → 结果写入 rid_state   │
└──────────────────────────────────┬───────────────────────────────┘
                                   ▼  (ZMQ → Detokenizer → ZMQ)
┌──────────────────────────────────────────────────────────────────┐
│ 5. RequestHandler._handle_loop()                     [主进程]     │
│    recv_from_detokenizer → 匹配 rid_to_state → event.set()       │
│    → HTTP response 返回客户端                                     │
╚══════════════════════════════════════════════════════════════════╝
```

### 进程全景图

```
┌─ 主进程 (python -m prism.launch)
│   ├─ FastAPI HTTP Server (:30000)
│   ├─ RequestHandler (async, Redis + ZMQ)
│   │
│   fork ──▶ Scheduler 子进程 (model_1)
│   │         ├─ patched_event_loop_normal [while True 循环]
│   │         ├─ ZMQ: recv_requests()
│   │         ├─ Redis: _prism_recv_generation_requests()
│   │         ├─ redis_client (独立连接)
│   │         └─ 共享内存: ipc_0_model_1_xxx
│   │
│   fork ──▶ Scheduler 子进程 (model_2)
│   │         ├─ patched_event_loop_normal [while True 循环]
│   │         ├─ ZMQ: recv_requests()
│   │         ├─ Redis: _prism_recv_generation_requests()
│   │         ├─ redis_client (独立连接)
│   │         └─ 共享内存: ipc_0_model_2_xxx
│   │
│   fork ──▶ Detokenizer 子进程
│   │
│   fork ──▶ GPU Scheduler 子进程 (GPU 0)
│              ├─ 从 frontend Redis queue 读取
│              ├─ 调度决策（SLO 优先级）
│              ├─ 写入 backend Redis queue
│              └─ ZMQ: 发送 activate/deactivate 到 Scheduler
│
└─ 外部: Redis Server (另一个 Docker 容器, network=host)
```

---

## 通信通道总结

| 通道 | 协议 | 方向 | 用途 |
|------|------|------|------|
| `frontend_generate_request_{uuid}:{model}` | Redis | RequestHandler → GPU Scheduler | 生成请求入队 |
| `backend_generate_request_{uuid}:{model}` | Redis | GPU Scheduler → Scheduler | 已调度请求 |
| `engine_to_gpu_scheduler_{uuid}:{gpu_id}` | Redis | Scheduler → GPU Scheduler | 激活/停用响应 |
| `scheduler_input_ipc_name` | ZMQ (IPC) | RequestHandler → Scheduler | 控制消息(flush/abort/activate/deactivate) |
| `detokenizer_ipc_name` | ZMQ (IPC) | Scheduler → Detokenizer | batch 结果 |
| `request_handler_ipc_name` | ZMQ (IPC) | Detokenizer → RequestHandler | 解码后文本 |
| `ipc_{gpu}_{model}_{user}` | 共享内存 | Scheduler → GPU Scheduler | 内存使用量 |

---

## Patch 对 SGLang 类的改动一览

### Scheduler (scheduler_patch.py)

| 方法 | 类型 | 说明 |
|------|------|------|
| `event_loop_normal` | **替换** | 原生只有 ZMQ，patch 后增加 Redis 轮询 |
| `init_request_dispatcher` | **替换** | 先调原生，再注册 Prism 请求类型 |
| `_prism_recv_generation_requests` | 新增 | Redis backend queue 非阻塞读取 |
| `_prism_handle_raw_generate_request` | 新增 | GenerateReqInput → TokenizedGenerateReqInput |
| `_prism_handle_activate_request` | 新增 | 模型激活 |
| `_prism_handle_deactivate_request` | 新增 | 模型停用 |
| `_prism_get_memory_usage` | 新增 | 内存使用量查询 |
| `_prism_update_memory_usage_shm` | 新增 | 更新共享内存 |

### ServerArgs (server_args_patch.py)

| 方法 | 类型 | 说明 |
|------|------|------|
| `from_multi_model_server_args` | 新增 (classmethod) | MultiModelServerArgs → ServerArgs，附加 Prism 属性 |

### TpModelWorker (tp_worker_patch.py)

| 方法 | 类型 | 说明 |
|------|------|------|
| `activate_model_runner` | 新增 | 加载权重、分配 KV cache |
| `deactivate_model_runner` | 新增 | 释放 KV cache |

### ModelRunner (model_runner_patch.py)

| 方法 | 类型 | 说明 |
|------|------|------|
| `__init__` | **扩展** | 检测 enable_elastic_memory，使用 MHATokenToKVPoolElastic |

---

## 关键设计决策

### 1. 为什么用 Redis 传递生成请求？
- prism-old 就是这个设计：ZMQ 用于控制消息（activate/deactivate），Redis 用于生成请求
- GPU Scheduler 作为中间层做调度决策，需要一个异步队列
- Redis 支持多进程并发读写，天然适合多模型场景

### 2. 为什么必须 patch event_loop_normal？
- pip 安装的 SGLang 原生只从 ZMQ 接收请求（`recv_requests`）
- 没有 `recv_generation_requests` 方法，不知道 Redis 的存在
- 必须替换 event loop 来添加 Redis 轮询

### 3. 为什么每个子进程创建独立的 Redis 连接？
- Python `multiprocessing.Process` fork 后共享的 Redis 连接会导致阻塞
- 与 prism-old 设计一致：每个 Scheduler 在 `__init__` 中创建自己的 `redis_client`

### 4. 为什么关闭主进程的 Redis 连接后再 fork？
- `redis-py` 的连接在 fork 后可能共享 socket 状态
- 主进程（RequestHandler）使用 AsyncRedisClient，子进程使用同步 RedisClient
- fork 前关闭防止连接状态泄漏

---

## 配置示例

### model_configs/1_gpu_2_model_our.json

```json
{
  "models": [
    {
      "model_name": "model_1",
      "model_path": "meta-llama/Llama-3.2-3B",
      "placements": [{"gpu_id": 0, "on": true}],
      "max_memory_pool_size": 100
    },
    {
      "model_name": "model_2", 
      "model_path": "meta-llama/Llama-3.2-3B",
      "placements": [{"gpu_id": 0, "on": true}],
      "max_memory_pool_size": 100
    }
  ]
}
```

### 启动命令

```bash
python -m prism.launch \
    --model-config-file ./model_configs/1_gpu_2_model_our.json \
    --port 30000 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --enable-elastic-memory \
    --enable-gpu-scheduler
```

## 与 prism-old 的区别

| 方面 | prism-old | prism-research |
|------|-----------|----------------|
| SGLang 版本 | 内置修改版（源码级改动） | pip 安装 + monkey patches |
| 代码侵入性 | 直接修改 SGLang 源码 | 零侵入，只 patch |
| Scheduler.__init__ | 内置 redis_client | managers.py 手动添加 |
| event_loop_normal | 内置 Redis 支持 | patch 替换添加 |
| 维护性 | 需要跟踪 SGLang 每次更新 | 只需更新 patches |

## 当前状态与待验证项

- Patch 机制：完成
- Redis 通信链路：完成
- 弹性内存（kvcached）集成：完成
- multi-model 端到端测试：待验证
- WorkerPool 模式：待验证

## 调试技巧

```bash
# 查看 Redis 队列长度
redis-cli LLEN backend_generate_request_xxx:model_1

# 查看日志
tail -f debug_log.txt | grep -E "Prism|Redis|recv"

# 检查共享内存
ls /dev/shm/ipc_*

# 检查 Scheduler 签名（确认 pip 版本）
python -c "from sglang.srt.managers.scheduler import Scheduler; import inspect; print(inspect.signature(Scheduler.__init__))"

# 检查 patch 是否生效
python -c "from sglang.srt.managers.scheduler import Scheduler; print(hasattr(Scheduler, '_prism_recv_generation_requests'))"
```
