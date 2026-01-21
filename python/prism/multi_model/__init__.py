# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Prism Multi-Model Serving Module.

This module provides:
- MultiModelServerArgs: Configuration for multi-model serving
- MultiModelServer: Main server class
- GPU scheduling and resource management
- Worker pool coordination
"""

from prism.multi_model.server_args import (
    ModelConfig,
    InstanceConfig,
    MultiModelServerArgs,
    Placement,
    load_model_configs,
    prepare_server_args,
)


# Lazy import for server module (requires full sglang)
def __getattr__(name):
    if name == "launch_multi_model_server":
        from prism.multi_model.server import launch_multi_model_server
        return launch_multi_model_server
    elif name == "EngineInfo":
        from prism.multi_model.server import EngineInfo
        return EngineInfo
    raise AttributeError(f"module 'prism.multi_model' has no attribute '{name}'")


__all__ = [
    # Server Args (always available)
    "ModelConfig",
    "InstanceConfig", 
    "MultiModelServerArgs",
    "Placement",
    "load_model_configs",
    "prepare_server_args",
    # Server (lazy loaded)
    "launch_multi_model_server",
    "EngineInfo",
]
