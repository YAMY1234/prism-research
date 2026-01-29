# Prism + SGLang Redis 阻塞问题调试总结

## 问题描述

在将 Prism 项目与新版 SGLang 集成时，遇到了 **Redis 读取阻塞**问题：
- 第一个启动的 scheduler 进程能够正常访问 Redis
- 第二个启动的 scheduler 进程在调用 `redis_client.recv_pyobj_non_block()` 时会阻塞

## 环境信息

- **代码位置**: `/home/yangminl/files/prism-research` (新版), `/home/yangminl/files/prism-old` (参考版)
- **测试环境**: Docker 容器 `sgl`
- **SGLang**: 新版（与 prism-old 使用的旧版不同）

## 核心架构差异

### prism-old 架构
| 组件 | 实现方式 |
|------|----------|
| Scheduler | 继承/扩展 SGLang 的 Scheduler 类 |
| Redis client | 在 `Scheduler.__init__` 中创建 |
| event_loop_normal | 完全自定义实现 |
| recv_requests | 使用独立的 `recv_generation_requests()` 从 Redis 读取 |

### prism-research 架构 (我们的 patch)
| 组件 | 实现方式 |
|------|----------|
| Scheduler | Patch 新版 SGLang 的 Scheduler 类 |
| Redis client | 在 `managers.py` 中创建（Scheduler 初始化后）|
| event_loop_normal | Patch 版本，调用原始方法 |
| recv_requests | Patch 后调用 `_original_recv_requests()` + Redis |

## 已尝试的解决方案

### 1. 禁用 idle_sleeper
**假设**: SGLang 的 `idle_sleeper` 使用 ZMQ 阻塞等待，可能干扰 Redis 接收

**修改**:
```python
# managers.py
if hasattr(scheduler, 'idle_sleeper') and scheduler.idle_sleeper is not None:
    scheduler.idle_sleeper = None
```

**结果**: ❌ 未解决问题

### 2. 完全替换 event_loop_normal
**假设**: 原始 event_loop 中有导致阻塞的逻辑

**修改**: 在 `scheduler_patch.py` 中完全替换 `event_loop_normal`，不调用原始方法

**结果**: ❌ 未解决问题，但有助于隔离问题

### 3. 分离 Redis 调用
**假设**: 在 `patched_recv_requests` 中调用 Redis 可能与 ZMQ 操作冲突

**修改**: 
- `patched_recv_requests` 只调用 `_original_recv_requests()`（ZMQ）
- 新增 `_prism_recv_generation_requests()` 方法单独处理 Redis
- 在 `patched_event_loop_normal` 中分别调用两者

**结果**: ⚠️ 部分改善，两个 model 都能启动 event loop，但请求未被正确接收

### 4. 添加 Redis 超时参数
**假设**: 默认的 Redis 连接没有超时设置

**修改**:
```python
self.client = redis.Redis(
    host=host, port=port, db=db,
    socket_timeout=5,
    socket_connect_timeout=5
)
```

**结果**: ❌ 未解决问题

### 5. 使用独立连接池
**假设**: redis-py 的默认连接池在多进程环境下有问题

**修改**:
```python
self._pool = redis.ConnectionPool(host=host, port=port, db=db)
self.client = redis.Redis(connection_pool=self._pool)
```

**结果**: ❌ 未解决问题

### 6. Fork-safe 连接管理
**假设**: fork 后子进程继承了父进程的 Redis 连接状态

**修改**:
```python
def _check_pid(self):
    if os.getpid() != self._pid:
        self._pool = redis.ConnectionPool(...)
        self.client = redis.Redis(connection_pool=self._pool)
```

**结果**: ❌ 未解决问题（已 revert）

### 7. 主进程 Redis 连接清理
**假设**: 主进程在 fork 前创建的 Redis 连接影响子进程

**修改**:
```python
# server.py
redis_client.close()
del redis_client
```

**结果**: ❌ 未解决问题

## 关键日志分析

### 阻塞场景（修复前）
```
[model_1 GPU=0 TP0] [DEBUG] model_1 recv #1 - calling Redis recv
[model_1 GPU=0 TP0] [DEBUG] model_1 recv #1 - Redis recv done, got 0  ✓
...
[model_2 GPU=0 TP0] [DEBUG] model_2 recv #1 - calling Redis recv
# 卡住，没有后续日志
```

### 最新测试（2026-01-22 08:22）
```
# model_1 正常运行 3 个循环
[model_1 GPU=0 TP0] model_1 loop #1 - before recv_requests()
[model_1 GPU=0 TP0] model_1 loop #1 - after recv_requests(), got 0 ZMQ reqs
[model_1 GPU=0 TP0] model_1 loop #1 - before _prism_recv_generation_requests()
[model_1 GPU=0 TP0] model_1 loop #1 - after Redis recv, got 0 reqs  ✓
... loop #2, #3 同样正常

# model_2 卡在 Redis
[model_2 GPU=0 TP0] model_2 loop #1 - before recv_requests()
[model_2 GPU=0 TP0] model_2 loop #1 - after recv_requests(), got 0 ZMQ reqs
[model_2 GPU=0 TP0] model_2 loop #1 - before _prism_recv_generation_requests()
# ❌ 没有 "after Redis recv" 日志 - 卡在 Redis 操作中！
```

**关键发现**：问题确实是 **Redis 阻塞**，第二个 scheduler 进程在调用 Redis `lpop` 或 `llen` 时会阻塞。

### 最新测试（2026-01-22 08:52 - 每次创建新连接）
```
# model_2 成功创建了新的 Redis 连接，但卡在 lpop 操作本身！
[model_2] model_2 [pid=90395] creating fresh Redis connection to localhost:6379  ✓
[model_2] model_2 [pid=90395] calling lpop on key=...  ✓
# ❌ 没有 "lpop returned" - 卡在 lpop 操作本身！
```

**重要发现**：
- 问题**不在 redis-py 客户端连接管理**
- model_2 能成功创建新连接
- 但 **`lpop` 操作本身阻塞了**
- 这说明问题可能在 **Redis 服务器层面**

## 待解决问题

1. **Event loop 正常启动**：两个 model 的 event loop 都能启动（不再卡住）
2. **请求未被接收**：GPU Scheduler 发送的请求没有被 scheduler 接收到
3. **可能原因**:
   - `_prism_recv_generation_requests` 返回空列表
   - Redis key 不匹配
   - event loop 在其他地方阻塞

## 与 prism-old 的关键对比

prism-old 没有这个问题，可能的原因：
1. **进程启动时机**: prism-old 的 scheduler 可能有更大的启动间隔
2. **架构差异**: prism-old 不调用新版 SGLang 的任何方法
3. **Redis 使用模式**: prism-old 在 `Scheduler.__init__` 中创建 Redis client

## 下一步建议

### 当前最高优先级：排查 Redis 服务器问题

1. **检查 Redis 服务器状态**：
   ```bash
   redis-cli INFO
   redis-cli CLIENT LIST
   redis-cli CONFIG GET maxclients
   ```

2. **检查是否有阻塞操作**：
   ```bash
   redis-cli SLOWLOG GET 10
   redis-cli DEBUG SLEEP 0  # 测试 Redis 是否响应
   ```

3. **尝试用 redis-cli 手动测试**：
   ```bash
   # 在一个终端启动 model_1 和 model_2 后，在另一个终端测试
   redis-cli LPOP backend_generate_request_xxx:model_1
   redis-cli LPOP backend_generate_request_xxx:model_2
   ```

4. **检查 prism-old 使用的 Redis 版本/配置**

### 其他方向

5. 完全模仿 prism-old 的架构，不使用 patch
6. 尝试使用不同的 Redis 操作（如 BRPOPLPUSH 替代 LPOP）

## 测试脚本

### 基本测试（非 WorkerPool 模式）

```bash
# 进入 Docker 容器
docker exec -it sgl bash

# 运行测试（45秒超时）
cd /sgl-workspace/files/prism-research/benchmark/multi-model && \
PYTHONUNBUFFERED=1 timeout 45 python -m prism.launch \
    --model-config-file ./model_configs/1_gpu_2_model_our.json \
    --port 30060 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --enable-elastic-memory \
    --enable-gpu-scheduler
```

### WorkerPool 模式测试

```bash
cd /sgl-workspace/files/prism-research/benchmark/multi-model && \
PYTHONUNBUFFERED=1 timeout 60 python -m prism.launch \
    --model-config-file ./model_configs/1_gpu_2_model_our.json \
    --port 30060 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --enable-elastic-memory \
    --enable-gpu-scheduler \
    --enable-worker-pool \
    --workers-per-gpu 2
```

### Redis 调试命令

```bash
# 在 Docker 容器内检查 Redis 状态
redis-cli KEYS "*"

# 检查特定队列长度
redis-cli LLEN "backend_generate_request_<prefix>:model_1"
redis-cli LLEN "backend_generate_request_<prefix>:model_2"

# 查看队列内容（不删除）
redis-cli LRANGE "backend_generate_request_<prefix>:model_1" 0 -1

# 清空 Redis 数据库
redis-cli FLUSHDB
```

### 杀死残留进程

```bash
# 杀死 prism 相关进程
pkill -9 -f 'python -m prism'

# 杀死 sglang 相关进程
pkill -9 -f sglang
```

### 日志保存测试

```bash
cd /sgl-workspace/files/prism-research/benchmark/multi-model && \
PYTHONUNBUFFERED=1 timeout 45 python -m prism.launch \
    --model-config-file ./model_configs/1_gpu_2_model_our.json \
    --port 30060 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --enable-elastic-memory \
    --enable-gpu-scheduler 2>&1 | tee /sgl-workspace/files/prism_test.log
```

## 相关文件

- `/home/yangminl/files/prism-research/python/prism/patches/scheduler_patch.py` - Scheduler patch
- `/home/yangminl/files/prism-research/python/prism/multi_model/managers.py` - 进程启动逻辑
- `/home/yangminl/files/prism-research/python/prism/utils/redis_utils.py` - Redis 客户端
- `/home/yangminl/files/prism-old/python/sglang/srt/managers/scheduler.py` - 参考实现
