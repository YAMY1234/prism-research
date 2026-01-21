# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Prism: Cost-Efficient Multi-LLM Inference

Prism is a multi-LLM serving system that achieves >2× cost savings and 3.3× more
SLO attainment through flexible GPU sharing.

This package provides minimal-invasive patches to extend SGLang with multi-model
serving capabilities.

Architecture:
    prism/
    ├── __init__.py              # Entry point, auto-applies patches
    ├── io_struct.py             # Prism-specific data structures
    ├── patches/                 # Monkey patches for SGLang
    │   ├── scheduler_patch.py   # Extends TypeBasedDispatcher
    │   └── memory_pool_patch.py # MHATokenToKVPoolElastic
    ├── model_runner/            # Worker pool model runner
    │   └── worker_pool_runner.py
    ├── multi_model/             # Multi-model server
    │   ├── server_args.py       # MultiModelServerArgs
    │   └── scheduling/          # GPU scheduling
    └── utils/                   # Utilities (Redis, etc.)

Usage:
    import prism  # Automatically applies patches to SGLang
    
    from prism.multi_model import MultiModelServerArgs
    
    args = MultiModelServerArgs(
        model_config_file="models.json",
        enable_elastic_memory=True,
    )
"""

__version__ = "1.0.0"

import logging

logger = logging.getLogger(__name__)

# Lazy patch application
_patches_applied = False


def apply_patches():
    """
    Apply Prism patches to SGLang.
    
    This is called automatically on import, but can be called manually
    if you need to control when patches are applied.
    """
    global _patches_applied
    if _patches_applied:
        return
    
    try:
        from prism.patches import apply_all_patches
        apply_all_patches()
        _patches_applied = True
        logger.info("Prism patches applied to SGLang")
    except ImportError as e:
        logger.warning(f"Could not apply Prism patches: {e}")
        logger.warning("Some Prism features may not work correctly")


def is_patches_applied() -> bool:
    """Check if patches have been applied."""
    return _patches_applied


# Apply patches on import (can be disabled by setting PRISM_NO_AUTO_PATCH=1)
import os
if not os.environ.get("PRISM_NO_AUTO_PATCH"):
    apply_patches()


# Public API - Data Structures (always available)
from prism.io_struct import (
    ActivateReqInput,
    ActivateReqOutput,
    DeactivateReqInput,
    DeactivateReqOutput,
    MemoryUsage,
    PreemptMode,
    GetMemPoolSizeReq,
    GetMemPoolSizeReqOutput,
    GetMemoryUsageReq,
    GetMemoryUsageReqOutput,
    ResizeMemPoolReqInput,
)

# Public API - Server Args (always available)
from prism.multi_model.server_args import (
    MultiModelServerArgs,
    ModelConfig,
    InstanceConfig,
    Placement,
)

# Lazy imports for modules that require full sglang
def __getattr__(name):
    """Lazy load heavy modules that require full sglang installation."""
    if name == "PrismWorkerPoolModelRunner":
        from prism.model_runner import PrismWorkerPoolModelRunner
        return PrismWorkerPoolModelRunner
    elif name == "MHATokenToKVPoolElastic":
        from prism.patches.memory_pool_patch import MHATokenToKVPoolElastic
        return MHATokenToKVPoolElastic
    elif name == "launch_multi_model_server":
        from prism.multi_model.server import launch_multi_model_server
        return launch_multi_model_server
    raise AttributeError(f"module 'prism' has no attribute '{name}'")


__all__ = [
    # Version
    "__version__",
    
    # Functions
    "apply_patches",
    "is_patches_applied",
    
    # Data Structures
    "ActivateReqInput",
    "ActivateReqOutput",
    "DeactivateReqInput",
    "DeactivateReqOutput",
    "MemoryUsage",
    "PreemptMode",
    "GetMemPoolSizeReq",
    "GetMemPoolSizeReqOutput",
    "GetMemoryUsageReq",
    "GetMemoryUsageReqOutput",
    "ResizeMemPoolReqInput",
    
    # Server Args
    "MultiModelServerArgs",
    "ModelConfig",
    "InstanceConfig",
    "Placement",
    
    # Lazy loaded
    "PrismWorkerPoolModelRunner",
    "MHATokenToKVPoolElastic",
    "launch_multi_model_server",
]
