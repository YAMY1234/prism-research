# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Server args patch for Prism multi-model serving.

This module patches SGLang's ServerArgs to support multi-model configuration.
"""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from prism.multi_model.server_args import MultiModelServerArgs, InstanceConfig

logger = logging.getLogger(__name__)

_patch_applied = False


def apply_server_args_patch():
    """
    Apply Prism patches to SGLang's ServerArgs.
    
    Adds the from_multi_model_server_args method for converting
    MultiModelServerArgs to ServerArgs.
    """
    global _patch_applied
    
    if _patch_applied:
        logger.debug("Server args patch already applied.")
        return
    
    from sglang.srt.server_args import ServerArgs
    
    @staticmethod
    def from_multi_model_server_args(
        multi_model_server_args: "MultiModelServerArgs",
        instance_config: Optional["InstanceConfig"] = None,
        worker_id: Optional[int] = None,
    ):
        """
        Create ServerArgs from MultiModelServerArgs.
        
        Args:
            multi_model_server_args: The multi-model server configuration
            instance_config: Optional instance configuration for a specific model
            worker_id: Optional worker ID for worker pool mode
            
        Returns:
            ServerArgs instance configured for the specified model/worker
        """
        # Keys to remove (Prism-specific, not in base ServerArgs)
        keys_to_remove = {
            "model_config_file",
            "model_configs",
            "model_name",
            "model_path",
            "tokenizer_path",
            "max_memory_pool_size",
            "tp_size",
            "enable_controller",
            "enable_cpu_share_memory",
            "enable_gpu_scheduler",
            "policy",
            "enable_model_service",
            "num_model_service_workers",
            "enable_elastic_memory",
            "use_kvcached_v0",
            "abort_exceed_slos",
            "async_loading",
            "queue_id",
            "frontend_generate_request_key_prefix",
            "backend_generate_request_key_prefix",
            "engine_to_gpu_scheduler_key_prefix",
        }
        
        # Copy args dict and remove Prism-specific keys
        args_dict = vars(multi_model_server_args).copy()
        for key in keys_to_remove:
            args_dict.pop(key, None)
        
        # Also remove any keys not in ServerArgs
        import dataclasses
        server_args_fields = {f.name for f in dataclasses.fields(ServerArgs)}
        args_dict = {k: v for k, v in args_dict.items() if k in server_args_fields}
        
        if instance_config is None:
            # Worker pool mode - use worker_id
            if worker_id is not None:
                args_dict['worker_id'] = worker_id
            return ServerArgs(**args_dict)
        else:
            # Instance mode - use instance_config values
            return ServerArgs(
                model_path=instance_config.model_path,
                tokenizer_path=instance_config.tokenizer_path,
                tp_size=instance_config.tp_size,
                **args_dict,
            )
    
    # Apply patch
    ServerArgs.from_multi_model_server_args = from_multi_model_server_args
    
    _patch_applied = True
    logger.info("Prism server args patch applied successfully.")


def is_patch_applied() -> bool:
    """Check if the server args patch has been applied."""
    return _patch_applied


__all__ = ["apply_server_args_patch", "is_patch_applied"]
