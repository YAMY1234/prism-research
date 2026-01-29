# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
TpModelWorker patch for Prism multi-model serving.

This module patches SGLang's TpModelWorker to support:
- Model activation (activate_model_runner)
- Model deactivation (deactivate_model_runner)
- Memory pool resizing
- Elastic memory via kvcached
"""

import gc
import logging
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_patch_applied = False


def apply_tp_worker_patch():
    """
    Apply Prism patches to SGLang's TpModelWorker.
    
    Adds methods for:
    - activate_model_runner: Activate model and allocate resources
    - deactivate_model_runner: Release model resources
    - resize_memory_pool: Resize KV cache memory pool
    """
    global _patch_applied
    
    if _patch_applied:
        logger.debug("TpModelWorker patch already applied.")
        return
    
    from sglang.srt.managers.tp_worker import TpModelWorker
    
    def activate_model_runner(
        self,
        memory_pool_size: Optional[float] = None,
        gpu_id: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        """
        Activate the model runner and allocate GPU resources.
        
        For Prism multi-model serving, this is called when a model needs
        to be activated on a GPU that may be shared with other models.
        
        Args:
            memory_pool_size: Size of KV cache memory pool in GB (optional)
            gpu_id: GPU ID for verification
            model_name: Model name for logging
        """
        tic = time.perf_counter()
        model_name = model_name or getattr(self.server_args, 'served_model_name', 'unknown')
        logger.info(f"Prism: Activating model runner for {model_name}...")
        
        # Check if already activated
        if getattr(self, '_prism_activated', False):
            logger.warning(f"Model {model_name} is already activated")
            return True
        
        try:
            # If model_runner was deactivated (weights on CPU), reload to GPU
            if getattr(self, '_prism_weights_on_cpu', False):
                self._prism_load_weights_to_gpu()
            
            # Initialize or resize KV cache pool if needed
            if memory_pool_size is not None:
                self._prism_resize_kv_cache(memory_pool_size)
            
            # Try to initialize kvcached for elastic memory
            self._prism_init_kvcached()
            
            self._prism_activated = True
            
            elapsed = time.perf_counter() - tic
            logger.info(f"Prism: Model {model_name} activated in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Prism: Failed to activate model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def deactivate_model_runner(self):
        """
        Deactivate the model runner and release GPU resources.
        
        This releases GPU memory for other models while optionally
        keeping model weights in CPU memory for fast reactivation.
        """
        tic = time.perf_counter()
        model_name = getattr(self.server_args, 'served_model_name', 'unknown')
        logger.info(f"Prism: Deactivating model runner for {model_name}...")
        
        if not getattr(self, '_prism_activated', True):
            logger.warning(f"Model {model_name} is already deactivated")
            return True
        
        try:
            # Release KV cache memory
            self._prism_release_kv_cache()
            
            # Optionally move model weights to CPU (for fast reactivation)
            # For now, just clear CUDA cache
            self._prism_clear_gpu_memory()
            
            self._prism_activated = False
            
            elapsed = time.perf_counter() - tic
            logger.info(f"Prism: Model {model_name} deactivated in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Prism: Failed to deactivate model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def resize_memory_pool(self, new_size_gb: float) -> bool:
        """
        Resize the KV cache memory pool.
        
        Args:
            new_size_gb: New size in GB
            
        Returns:
            True if successful
        """
        logger.info(f"Prism: Resizing memory pool to {new_size_gb}GB")
        try:
            self._prism_resize_kv_cache(new_size_gb)
            return True
        except Exception as e:
            logger.error(f"Prism: Failed to resize memory pool: {e}")
            return False
    
    def _prism_init_kvcached(self):
        """Initialize kvcached for elastic memory management."""
        # Check if kvcached should be enabled
        enable_elastic = getattr(self.server_args, 'enable_elastic_memory', False)
        if not enable_elastic:
            logger.debug("Prism: Elastic memory not enabled, skipping kvcached init")
            return
        
        if getattr(self, '_prism_kvcached_initialized', False):
            logger.debug("Prism: kvcached already initialized")
            return
        
        try:
            from kvcached import ops as kvcached_ops
            
            virtual_mem_size_gb = getattr(self.server_args, 'virtual_memory_size_gb', 50.0)
            
            # Initialize kvcached runtime
            kvcached_ops.init_kvcached(
                virtual_mem_size_gb=virtual_mem_size_gb,
                reserve_virtual_mem=True,
            )
            
            self._prism_kvcached_ops = kvcached_ops
            self._prism_kvcached_initialized = True
            
            logger.info(f"Prism: kvcached initialized with {virtual_mem_size_gb}GB virtual memory")
            
        except ImportError:
            logger.warning(
                "Prism: kvcached not available. Elastic memory features disabled. "
                "Install with: pip install kvcached"
            )
        except Exception as e:
            logger.warning(f"Prism: Failed to initialize kvcached: {e}")
    
    def _prism_release_kv_cache(self):
        """Release KV cache memory."""
        try:
            model_runner = self.model_runner
            
            # Release token_to_kv_pool
            if hasattr(model_runner, 'token_to_kv_pool') and model_runner.token_to_kv_pool is not None:
                pool = model_runner.token_to_kv_pool
                
                # Check for elastic pool with kvcached
                if hasattr(pool, 'release'):
                    pool.release()
                    logger.info("Prism: Elastic KV cache pool released")
                elif hasattr(pool, 'free'):
                    pool.free()
                    logger.info("Prism: KV cache pool freed")
                else:
                    # Standard pool - just clear references
                    if hasattr(pool, 'kv_buffer'):
                        del pool.kv_buffer
                    logger.info("Prism: KV cache buffer cleared")
            
            # Release kvcached tensors if available
            if hasattr(self, '_prism_kvcached_ops') and self._prism_kvcached_ops is not None:
                try:
                    self._prism_kvcached_ops.free_kv_cached_tensors()
                    logger.info("Prism: kvcached tensors freed")
                except Exception as e:
                    logger.warning(f"Prism: Error freeing kvcached tensors: {e}")
                    
        except Exception as e:
            logger.error(f"Prism: Error releasing KV cache: {e}")
    
    def _prism_resize_kv_cache(self, new_size_gb: float):
        """Resize KV cache to new size."""
        try:
            model_runner = self.model_runner
            
            if hasattr(model_runner, 'token_to_kv_pool') and model_runner.token_to_kv_pool is not None:
                pool = model_runner.token_to_kv_pool
                
                # Check for elastic pool
                if hasattr(pool, 'resize'):
                    # Convert GB to tokens (rough estimate)
                    # This is model-dependent; using a conservative estimate
                    bytes_per_token = getattr(pool, 'bytes_per_token', 2048)  # Default estimate
                    new_num_tokens = int(new_size_gb * 1024 * 1024 * 1024 / bytes_per_token)
                    pool.resize(new_num_tokens)
                    logger.info(f"Prism: KV cache resized to {new_num_tokens} tokens")
                else:
                    logger.warning("Prism: KV cache pool does not support resize")
                    
        except Exception as e:
            logger.error(f"Prism: Error resizing KV cache: {e}")
    
    def _prism_clear_gpu_memory(self):
        """Clear GPU memory cache."""
        try:
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("Prism: GPU memory cache cleared")
        except Exception as e:
            logger.warning(f"Prism: Error clearing GPU memory: {e}")
    
    def _prism_load_weights_to_gpu(self):
        """Reload model weights from CPU to GPU."""
        logger.info("Prism: Reloading weights to GPU...")
        
        if not getattr(self, '_prism_cpu_state_dict', None):
            logger.warning("Prism: No CPU state dict available for reload")
            return
        
        try:
            model = self.model_runner.model
            device = f"cuda:{self.gpu_id}"
            
            # Load state dict back to GPU
            state_dict = {k: v.to(device) for k, v in self._prism_cpu_state_dict.items()}
            model.load_state_dict(state_dict, assign=True)
            
            self._prism_weights_on_cpu = False
            logger.info("Prism: Weights reloaded to GPU")
            
        except Exception as e:
            logger.error(f"Prism: Error reloading weights: {e}")
    
    def _prism_move_weights_to_cpu(self):
        """Move model weights to CPU to free GPU memory."""
        logger.info("Prism: Moving weights to CPU...")
        
        try:
            model = self.model_runner.model
            
            # Save state dict to CPU
            self._prism_cpu_state_dict = {
                k: v.cpu() for k, v in model.state_dict().items()
            }
            
            # Clear model on GPU
            for param in model.parameters():
                param.data = torch.empty(0, device='cpu')
            
            self._prism_weights_on_cpu = True
            torch.cuda.empty_cache()
            
            logger.info("Prism: Weights moved to CPU")
            
        except Exception as e:
            logger.error(f"Prism: Error moving weights to CPU: {e}")
    
    # Apply patches to TpModelWorker class
    TpModelWorker.activate_model_runner = activate_model_runner
    TpModelWorker.deactivate_model_runner = deactivate_model_runner
    TpModelWorker.resize_memory_pool = resize_memory_pool
    TpModelWorker._prism_init_kvcached = _prism_init_kvcached
    TpModelWorker._prism_release_kv_cache = _prism_release_kv_cache
    TpModelWorker._prism_resize_kv_cache = _prism_resize_kv_cache
    TpModelWorker._prism_clear_gpu_memory = _prism_clear_gpu_memory
    TpModelWorker._prism_load_weights_to_gpu = _prism_load_weights_to_gpu
    TpModelWorker._prism_move_weights_to_cpu = _prism_move_weights_to_cpu
    
    # Initialize default state attributes
    TpModelWorker._prism_activated = True  # Models start activated
    TpModelWorker._prism_kvcached_initialized = False
    TpModelWorker._prism_kvcached_ops = None
    TpModelWorker._prism_weights_on_cpu = False
    TpModelWorker._prism_cpu_state_dict = None
    
    _patch_applied = True
    logger.info("Prism TpModelWorker patch applied successfully.")


def is_patch_applied() -> bool:
    """Check if the TpModelWorker patch has been applied."""
    return _patch_applied


__all__ = ["apply_tp_worker_patch", "is_patch_applied"]
