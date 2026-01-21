# Install Prism

Prism 是一个多模型 LLM 推理系统，通过灵活的 GPU 共享实现 >2× 成本节省和 3.3× 更高的 SLO 达成率。

本指南介绍如何安装和配置 Prism。

## 前置要求

- Python >= 3.10
- NVIDIA GPU with CUDA >= 12.1
- Redis Server

## 方法 1: 使用 pip 安装 (推荐)

### 1. 安装 SGLang

首先安装 SGLang 基础框架：

```bash
# 使用 uv 加速安装 (推荐)
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]" --prerelease=allow

# 或者从源码安装
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python[all]"
```

### 2. 安装 Prism

```bash
cd prism-research/python/prism
pip install -e .
```

### 3. 安装 Redis

使用 Docker 启动 Redis 服务：

```bash
docker run --name prism-redis -p 6379:6379 -d redis
```

或者本地安装：

```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis
brew services start redis
```

验证 Redis 运行正常：

```python
>>> import redis
>>> r = redis.Redis(host='localhost', port=6379, db=0)
>>> r.set('test', 'ok')
True
>>> r.get('test')
b'ok'
```

### 4. (可选) 安装 kvcached 支持弹性内存

```bash
git clone https://github.com/ovg-project/kvcached.git
cd kvcached
pip install -e .
```

## 方法 2: 使用 Docker

### 1. 启动开发容器

```bash
docker run -dit --gpus all --ipc=host --network=host \
    -v `pwd`/prism-research:/workspace/prism \
    -v ~/.cache/huggingface/:/root/.cache/huggingface \
    --name prism-dev \
    lmsysorg/sglang:latest bash
```

### 2. 在容器中安装 Prism

```bash
docker exec -it prism-dev bash

# 在容器内
cd /workspace/prism/python/prism
pip install -e .
```

## 验证安装

```python
>>> import prism
>>> prism.__version__
'1.0.0'
>>> prism.is_patches_applied()
True
>>> from prism import MultiModelServerArgs, ActivateReqInput
>>> print("Prism installed successfully!")
```

## 快速开始

### 1. 创建模型配置文件

创建 `models.json`：

```json
[
  {
    "model_name": "llama-7b",
    "model_path": "meta-llama/Llama-2-7b-chat-hf",
    "init_placements": [
      {"gpu_ids": [0], "on": true, "max_memory_pool_size": 16}
    ]
  }
]
```

### 2. 启动单模型服务

```bash
python -m prism.launch \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --port 30000 \
    --disable-cuda-graph \
    --disable-radix-cache
```

### 3. 启动多模型服务

```bash
python -m prism.launch \
    --model-config-file models.json \
    --port 30000 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --enable-elastic-memory \
    --enable-controller
```

### 4. 启动 Worker Pool 模式 (推荐用于多模型)

```bash
python -m prism.launch \
    --model-config-file models.json \
    --port 30000 \
    --enable-worker-pool \
    --workers-per-gpu 4 \
    --num-gpus 8 \
    --enable-elastic-memory \
    --enable-gpu-scheduler \
    --enable-controller \
    --policy simple-global \
    --disable-cuda-graph \
    --disable-radix-cache
```

## 运行测试

### 基础功能测试

```bash
cd prism-research/python
python run_test.py
```

### 单模型 Benchmark

```bash
cd benchmark/multi-model

# 启动服务
python -m prism.launch \
    --model-config-file ./model_configs/model_config_single.json \
    --port 30000 \
    --disable-cuda-graph \
    --disable-radix-cache

# 运行测试
python benchmark.py -n 1 --base-url http://127.0.0.1:30000
```

### 多模型 E2E Benchmark

```bash
# 启动服务 (8 GPU, 18 模型)
python -m prism.launch \
    --model-config-file model_configs/8_gpu_18_model_our.json \
    --port 30333 \
    --enable-worker-pool \
    --workers-per-gpu 4 \
    --num-gpus 8 \
    --enable-elastic-memory \
    --enable-gpu-scheduler \
    --enable-controller \
    --policy simple-global \
    --enable-model-service \
    --num-model-service-workers 4 \
    --enable-cpu-share-memory \
    --disable-cuda-graph \
    --disable-radix-cache \
    --use-kvcached-v0 \
    --max-mem-usage 67.28 \
    --log-file server.log

# 运行 benchmark
python benchmark.py \
    --base-url http://127.0.0.1:30333 \
    --real-trace ./real_trace.pkl \
    --num-models 18 \
    --num-gpus 8 \
    --exp-name prism_test \
    --e2e-benchmark \
    --time-scale 1 \
    --replication 1
```

## 配置说明

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-config-file` | 模型配置 JSON 文件路径 | - |
| `--enable-worker-pool` | 启用 Worker Pool 模式 | False |
| `--workers-per-gpu` | 每个 GPU 的 worker 数量 | 1 |
| `--num-gpus` | GPU 数量 | 1 |
| `--enable-elastic-memory` | 启用弹性内存 (需要 kvcached) | False |
| `--enable-controller` | 启用多模型调度控制器 | False |
| `--enable-gpu-scheduler` | 启用 GPU 级调度器 | False |
| `--policy` | 调度策略 | simple-global |
| `--max-mem-usage` | 最大 GPU 内存使用量 (GB) | - |

### 模型配置文件格式

```json
[
  {
    "model_name": "unique_model_name",
    "model_path": "path/to/model",
    "tokenizer_path": "path/to/tokenizer",  // 可选，默认同 model_path
    "tp_size": 1,  // Tensor Parallelism 大小
    "init_placements": [
      {
        "gpu_ids": [0],           // GPU ID 列表 (支持 TP)
        "on": true,               // 是否初始加载到 GPU
        "max_memory_pool_size": 16  // 最大内存池大小 (GB)
      }
    ]
  }
]
```

## 常见问题

### FlashInfer 相关错误

如果遇到 FlashInfer 问题，使用其他后端：

```bash
python -m prism.launch ... --attention-backend triton --sampling-backend pytorch
```

### CUDA 内存不足

1. 减小 `max_memory_pool_size`
2. 减少每 GPU 的 worker 数量
3. 使用 `--load-format dummy` 进行调试

### Redis 连接失败

确保 Redis 服务正在运行：

```bash
redis-cli ping  # 应返回 PONG
```

## 项目结构

```
prism/
├── __init__.py              # 入口，自动应用 patches
├── io_struct.py             # Prism 专用数据结构
├── launch.py                # 启动脚本
├── patches/                 # Monkey patches
│   ├── scheduler_patch.py   # 扩展 TypeBasedDispatcher
│   └── memory_pool_patch.py # MHATokenToKVPoolElastic
├── model_runner/            # Worker Pool 模型加载
├── multi_model/             # 多模型服务核心
│   ├── server.py
│   ├── server_args.py
│   ├── scheduling/          # 调度系统
│   └── utils/
└── utils/
    └── redis_utils.py
```

## 许可证

Apache License 2.0
