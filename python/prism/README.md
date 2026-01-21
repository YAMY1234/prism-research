# Prism: Cost-Efficient Multi-LLM Inference

Prism is a multi-LLM serving system that achieves >2× cost savings and 3.3× more SLO attainment through flexible GPU sharing.

## Architecture

Prism is designed as an **independent plugin** for SGLang, using minimal-invasive monkey patching to extend SGLang's capabilities without modifying its core code.

```
prism/
├── __init__.py              # Entry point, auto-applies patches
├── io_struct.py             # Prism-specific data structures
├── patches/                 # Monkey patches for SGLang
│   ├── __init__.py
│   ├── scheduler_patch.py   # Extends TypeBasedDispatcher for new request types
│   └── memory_pool_patch.py # MHATokenToKVPoolElastic using kvcached
├── model_runner/            # Worker pool model runner
│   ├── __init__.py
│   └── worker_pool_runner.py
├── multi_model/             # Multi-model server components
│   ├── __init__.py
│   ├── server_args.py       # MultiModelServerArgs
│   └── scheduling/          # GPU scheduling
│       ├── __init__.py
│       └── state.py         # Scheduler state management
└── utils/                   # Utilities
    ├── __init__.py
    └── redis_utils.py       # Redis client for IPC
```

## How It Works

### 1. Minimal-Invasive Patching

Prism uses SGLang's `TypeBasedDispatcher` mechanism to extend request handling:

```python
# SGLang's TypeBasedDispatcher allows adding new request handlers
self._request_dispatcher._mapping[ActivateReqInput] = self._prism_handle_activate_request
self._request_dispatcher._mapping[DeactivateReqInput] = self._prism_handle_deactivate_request
```

This means:
- No modifications to existing SGLang code paths
- New request types are handled by new methods
- Easy to maintain across SGLang version upgrades

### 2. Elastic Memory via kvcached

`MHATokenToKVPoolElastic` is an **independent class** (not inheriting from SGLang's `MHATokenToKVPool`) that uses kvcached for dynamic memory allocation:

```python
from prism.patches.memory_pool_patch import MHATokenToKVPoolElastic

pool = MHATokenToKVPoolElastic(
    size=max_tokens,
    dtype=torch.float16,
    head_num=32,
    head_dim=128,
    layer_num=32,
    device="cuda",
    gpu_id=0,
    enable_elastic_memory=True,
)
```

### 3. Worker Pool Model Runner

`PrismWorkerPoolModelRunner` supports:
- CPU model preloading for fast activation
- On-demand GPU activation/deactivation
- Worker pool mode for multiple models per GPU

## Usage

```python
import prism  # Automatically applies patches

from prism import MultiModelServerArgs, ActivateReqInput

# Configure multi-model server
args = MultiModelServerArgs(
    model_config_file="models.json",
    enable_elastic_memory=True,
    enable_worker_pool=True,
    workers_per_gpu=2,
    num_gpus=4,
)

# Create activation request
activate_req = ActivateReqInput(
    model_name="llama-7b",
    gpu_id=0,
    memory_pool_size=10.0,  # GB
)
```

## Environment Variables

- `PRISM_NO_AUTO_PATCH=1`: Disable automatic patch application on import

## Dependencies

- `sglang>=0.4.0`: Base LLM serving framework
- `kvcached>=1.0.0`: Elastic memory management
- `redis>=4.0.0`: Optional, for multi-process coordination

## License

Apache License 2.0
