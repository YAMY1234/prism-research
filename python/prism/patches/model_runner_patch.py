# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
ModelRunner patch for Prism elastic memory via kvcached.

This module patches SGLang's ModelRunner to use kvcached for KV cache allocation
when elastic memory is enabled, allowing over-allocation via virtual memory.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_patch_applied = False
_kvcached_initialized = False
_kvcached_ops = None


def init_kvcached_global(virtual_memory_size_gb: float = 50.0, gpu_id: int = 0):
    """
    Initialize kvcached globally before any model runner is created.
    
    Args:
        virtual_memory_size_gb: Size of virtual memory pool in GB
        gpu_id: GPU device ID
    """
    global _kvcached_initialized, _kvcached_ops
    
    if _kvcached_initialized:
        logger.debug("kvcached already initialized globally")
        return True
    
    try:
        from kvcached import ops as kvcached_ops
        
        logger.info(f"Initializing kvcached with {virtual_memory_size_gb}GB virtual memory on GPU {gpu_id}")
        
        kvcached_ops.init_kvcached(
            gpu_id=gpu_id,
            virtual_mem_size_gb=int(virtual_memory_size_gb),
            reserve_virtual_mem=True,
        )
        
        _kvcached_ops = kvcached_ops
        _kvcached_initialized = True
        
        logger.info("kvcached initialized successfully")
        return True
        
    except ImportError:
        logger.warning(
            "kvcached not available. Elastic memory disabled. "
            "Install with: pip install kvcached"
        )
        return False
    except Exception as e:
        logger.error(f"Failed to initialize kvcached: {e}")
        return False


def apply_model_runner_patch():
    """
    Apply Prism patches to SGLang's ModelRunner for elastic memory support.
    
    This patches the init_memory_pool method to use MHATokenToKVPoolElastic
    when elastic memory is enabled (via server_args.enable_elastic_memory).
    """
    global _patch_applied
    
    if _patch_applied:
        logger.debug("ModelRunner patch already applied.")
        return
    
    try:
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import ModelRunnerKVCacheMixin
        from sglang.srt.model_executor.model_runner import ModelRunner
        
        # Save original methods
        _original_init_memory_pool = ModelRunnerKVCacheMixin.init_memory_pool
        _original_profile_max_num_token = ModelRunnerKVCacheMixin.profile_max_num_token
        
        def patched_init_memory_pool(self: ModelRunner, total_gpu_memory: int):
            """
            Patched init_memory_pool that uses MHATokenToKVPoolElastic when
            elastic memory is enabled.
            """
            # Check if elastic memory is enabled
            enable_elastic = getattr(self.server_args, 'enable_elastic_memory', False)
            
            if not enable_elastic:
                # Use original implementation
                return _original_init_memory_pool(self, total_gpu_memory)
            
            logger.info("Prism: Using elastic memory pool with kvcached")
            
            # Initialize kvcached if not already done
            virtual_mem_gb = getattr(self.server_args, 'virtual_memory_size_gb', 50.0)
            gpu_id = getattr(self, 'gpu_id', 0)
            
            if not _kvcached_initialized:
                if not init_kvcached_global(virtual_mem_gb, gpu_id):
                    logger.warning("Prism: kvcached init failed, falling back to standard memory pool")
                    return _original_init_memory_pool(self, total_gpu_memory)
            
            # Import required modules
            from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
            from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator, TokenToKVPoolAllocator
            from sglang.srt.layers.dp_attention import get_attention_tp_size
            from sglang.srt.utils import get_available_gpu_memory
            
            try:
                from prism.patches.memory_pool_patch import MHATokenToKVPoolElastic
            except ImportError as e:
                logger.warning(f"Prism: Could not import elastic pool: {e}, falling back")
                return _original_init_memory_pool(self, total_gpu_memory)
            
            # ========================================
            # Elastic Memory: custom init_memory_pool
            # ========================================
            
            max_num_reqs = self.server_args.max_running_requests
            max_total_tokens = self.server_args.max_total_tokens
            
            # Get the max_memory_pool_size from server_args (in GB)
            # This is set per-model in the Prism config
            max_memory_pool_size_gb = getattr(self.server_args, 'max_memory_pool_size', None)
            
            if max_memory_pool_size_gb is not None:
                # Calculate max_total_num_tokens based on config
                # Each token needs: 2 * layer_num * head_num * head_dim * dtype_size bytes (K+V)
                head_num = self.model_config.get_num_kv_heads(get_attention_tp_size())
                head_dim = self.model_config.head_dim
                layer_num = self.num_effective_layers
                dtype_size = 2  # bfloat16
                
                bytes_per_token = 2 * layer_num * head_num * head_dim * dtype_size
                max_bytes = max_memory_pool_size_gb * 1024 * 1024 * 1024
                self.max_total_num_tokens = int(max_bytes / bytes_per_token)
                
                logger.info(f"Prism: Using configured max_memory_pool_size={max_memory_pool_size_gb}GB "
                           f"-> max_total_num_tokens={self.max_total_num_tokens}")
            else:
                # Fall back to profiling (but with a larger virtual limit)
                self.max_total_num_tokens = _original_profile_max_num_token(self, total_gpu_memory)
            
            # Apply max_total_tokens limit if specified
            if max_total_tokens is not None:
                self.max_total_num_tokens = min(self.max_total_num_tokens, max_total_tokens)
            
            # Align to page size
            self.max_total_num_tokens = (
                self.max_total_num_tokens
                // self.server_args.page_size
                * self.server_args.page_size
            )
            
            if self.max_total_num_tokens <= 0:
                raise RuntimeError(
                    f"Prism: max_total_num_tokens is {self.max_total_num_tokens}. "
                    f"Please check your configuration."
                )
            
            # Calculate max_num_reqs
            if max_num_reqs is None:
                max_num_reqs = min(
                    max(
                        int(self.max_total_num_tokens / self.model_config.context_len * 512),
                        2048,
                    ),
                    4096,
                )
            
            # Initialize req_to_token_pool
            if self.req_to_token_pool is None:
                extra_max_context_len = 4
                if self.server_args.speculative_num_draft_tokens is not None:
                    extra_max_context_len += self.server_args.speculative_num_draft_tokens
                
                self.req_to_token_pool = ReqToTokenPool(
                    size=max_num_reqs,
                    max_context_len=self.model_config.context_len + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                )
            
            # Initialize token_to_kv_pool with elastic pool
            head_num = self.model_config.get_num_kv_heads(get_attention_tp_size())
            head_dim = self.model_config.head_dim
            layer_num = self.num_effective_layers
            
            logger.info(f"Prism: Creating elastic KV pool "
                       f"(size={self.max_total_num_tokens}, heads={head_num}, layers={layer_num})")
            
            self.token_to_kv_pool = MHATokenToKVPoolElastic(
                size=self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=layer_num,
                device=self.device,
                gpu_id=gpu_id,
                model_name=getattr(self.server_args, 'served_model_name', 'unknown'),
                use_kvcached_v0=getattr(self.server_args, 'use_kvcached_v0', True),
                enable_worker_pool=getattr(self.server_args, 'enable_worker_pool', False),
            )
            
            # Initialize token_to_kv_pool_allocator
            need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
            if self.token_to_kv_pool_allocator is None:
                if self.page_size == 1:
                    self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
            
            logger.info(
                f"Prism: Elastic memory pool initialized. "
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
            )
        
        # Apply the patch
        ModelRunnerKVCacheMixin.init_memory_pool = patched_init_memory_pool
        
        _patch_applied = True
        logger.info("Prism ModelRunner patch applied successfully.")
        
    except Exception as e:
        logger.error(f"Failed to apply ModelRunner patch: {e}")
        import traceback
        traceback.print_exc()


def is_patch_applied() -> bool:
    """Check if the ModelRunner patch has been applied."""
    return _patch_applied


def is_kvcached_initialized() -> bool:
    """Check if kvcached has been initialized."""
    return _kvcached_initialized


__all__ = [
    "apply_model_runner_patch", 
    "is_patch_applied",
    "init_kvcached_global",
    "is_kvcached_initialized",
]
