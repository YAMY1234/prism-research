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
        # But we'll store some of these as extra attributes after creation
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
            "virtual_memory_size_gb",
            "abort_exceed_slos",
            "async_loading",
            "queue_id",
            "frontend_generate_request_key_prefix",
            "backend_generate_request_key_prefix",
            "engine_to_gpu_scheduler_key_prefix",
        }
        
        # Save Prism-specific values before removing
        prism_extra_attrs = {
            'enable_elastic_memory': getattr(multi_model_server_args, 'enable_elastic_memory', False),
            'use_kvcached_v0': getattr(multi_model_server_args, 'use_kvcached_v0', True),
            'virtual_memory_size_gb': getattr(multi_model_server_args, 'virtual_memory_size_gb', 50.0),
            'enable_worker_pool': getattr(multi_model_server_args, 'enable_worker_pool', False),
            # Redis connection parameters
            'redis_host': getattr(multi_model_server_args, 'redis_host', 'localhost'),
            'redis_port': getattr(multi_model_server_args, 'redis_port', 6379),
            'redis_db': getattr(multi_model_server_args, 'redis_db', 0),
            # Redis queue key prefixes for GPU scheduler communication
            'backend_generate_request_key_prefix': getattr(multi_model_server_args, 'backend_generate_request_key_prefix', None),
            'frontend_generate_request_key_prefix': getattr(multi_model_server_args, 'frontend_generate_request_key_prefix', None),
            'engine_to_gpu_scheduler_key_prefix': getattr(multi_model_server_args, 'engine_to_gpu_scheduler_key_prefix', None),
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
            # Worker pool mode - need a default model_path
            # Use the first model from configs if available
            if 'model_path' not in args_dict:
                model_configs = getattr(multi_model_server_args, 'model_configs', [])
                if model_configs:
                    args_dict['model_path'] = model_configs[0].model_path
                else:
                    args_dict['model_path'] = getattr(multi_model_server_args, 'model_path', 'meta-llama/Llama-3.2-1B')
            
            # Note: With kvcached enabled, workers can over-allocate via virtual memory
            # So we don't need to reduce mem_fraction_static
            
            server_args = ServerArgs(**args_dict)
            # Store worker_id as extra attribute (not a ServerArgs field in new sglang)
            if worker_id is not None:
                server_args.worker_id = worker_id
        else:
            # Instance mode - use instance_config values
            # Calculate mem_fraction_static from max_memory_pool_size if specified
            if instance_config.max_memory_pool_size is not None:
                # max_memory_pool_size is in GB, convert to fraction
                # Assume ~280GB GPU memory, add buffer for model weights (~15GB for 3B model)
                import torch
                if torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                else:
                    gpu_mem_gb = 280  # Default estimate
                # mem_fraction_static = (model_weights + kv_cache) / total_gpu_mem
                # For small models, estimate model weights as ~5GB per billion params
                # max_memory_pool_size is the KV cache size
                estimated_model_size = 15  # Conservative estimate in GB
                target_memory = instance_config.max_memory_pool_size + estimated_model_size
                args_dict['mem_fraction_static'] = min(0.95, target_memory / gpu_mem_gb)
                logger.info(f"Setting mem_fraction_static={args_dict['mem_fraction_static']:.3f} "
                           f"for max_memory_pool_size={instance_config.max_memory_pool_size}GB")
            
            # IMPORTANT: Set served_model_name to the logical model name (not model_path)
            # This is critical for Prism multi-model routing via Redis queues
            served_model_name = getattr(instance_config, 'model_name', None) or instance_config.model_path
            
            # Remove served_model_name from args_dict if present to avoid duplicate
            args_dict.pop('served_model_name', None)
            
            server_args = ServerArgs(
                model_path=instance_config.model_path,
                tokenizer_path=instance_config.tokenizer_path,
                tp_size=instance_config.tp_size,
                served_model_name=served_model_name,
                **args_dict,
            )
        
        # Add Prism-specific attributes to server_args instance
        # These will be accessible by TpModelWorker patch
        for attr_name, attr_value in prism_extra_attrs.items():
            setattr(server_args, attr_name, attr_value)
        
        # Store instance-specific attributes for elastic memory
        if instance_config is not None:
            setattr(server_args, 'max_memory_pool_size', instance_config.max_memory_pool_size)
            setattr(server_args, 'on', instance_config.on)
        
        return server_args
    
    # Apply patch
    ServerArgs.from_multi_model_server_args = from_multi_model_server_args
    
    _patch_applied = True
    logger.info("Prism server args patch applied successfully.")


def is_patch_applied() -> bool:
    """Check if the server args patch has been applied."""
    return _patch_applied


__all__ = ["apply_server_args_patch", "is_patch_applied"]
