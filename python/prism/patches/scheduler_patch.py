# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Scheduler patch for Prism multi-model serving.

This module patches SGLang's Scheduler to support:
- Model activation requests (ActivateReqInput)
- Model deactivation requests (DeactivateReqInput)
- Memory usage queries
- Memory pool resizing

The patch works by extending the TypeBasedDispatcher with new request handlers.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_patch_applied = False


def apply_scheduler_patch():
    """
    Apply Prism patches to SGLang's Scheduler.
    
    This function:
    1. Extends the request dispatcher with new request types
    2. Adds handler methods for activation/deactivation
    3. Adds state tracking for Prism features
    
    This is idempotent - calling multiple times has no effect.
    """
    global _patch_applied
    
    if _patch_applied:
        logger.debug("Scheduler patch already applied.")
        return
    
    from sglang.srt.managers.scheduler import Scheduler
    
    # Import Prism data structures
    from prism.io_struct import (
        ActivateReqInput,
        ActivateReqOutput,
        DeactivateReqInput,
        DeactivateReqOutput,
        GetMemoryUsageReq,
        GetMemoryUsageReqOutput,
        GetMemPoolSizeReq,
        GetMemPoolSizeReqOutput,
        MemoryUsage,
        ResizeMemPoolReqInput,
    )
    
    # Save original init_request_dispatcher
    _original_init_request_dispatcher = Scheduler.init_request_dispatcher
    
    def patched_init_request_dispatcher(self):
        """Extended init_request_dispatcher with Prism request types."""
        # Call original implementation
        _original_init_request_dispatcher(self)
        
        # Extend dispatcher with Prism request types
        # TypeBasedDispatcher uses _mapping which is an OrderedDict
        self._request_dispatcher._mapping[ActivateReqInput] = self._prism_handle_activate_request
        self._request_dispatcher._mapping[DeactivateReqInput] = self._prism_handle_deactivate_request
        self._request_dispatcher._mapping[GetMemPoolSizeReq] = self._prism_handle_get_mem_pool_size
        self._request_dispatcher._mapping[GetMemoryUsageReq] = self._prism_handle_get_memory_usage
        self._request_dispatcher._mapping[ResizeMemPoolReqInput] = self._prism_handle_resize_mem_pool
        
        logger.debug("Prism request handlers registered to Scheduler dispatcher.")
    
    def _prism_handle_activate_request(self, recv_req: ActivateReqInput):
        """
        Handle model activation request.
        
        This activates a model on the specified GPU, loading weights and
        initializing the KV cache.
        """
        logger.info(f"Prism: Handling activate request for model={recv_req.model_name}, gpu={recv_req.gpu_id}")
        
        # Check if already activated
        if getattr(self, '_prism_activated', True):
            logger.warning(f"Model already activated, ignoring request rid={recv_req.rid}")
            return ActivateReqOutput(
                rid=recv_req.rid,
                success=False,
                memory_usage=self._prism_get_memory_usage(),
                model_name=recv_req.model_name,
                instance_idx=recv_req.instance_idx,
                gpu_id=recv_req.gpu_id,
            )
        
        start_time = time.perf_counter()
        
        try:
            # Activate model runner if available
            if hasattr(self, 'tp_worker') and hasattr(self.tp_worker, 'activate_model_runner'):
                self.tp_worker.activate_model_runner(
                    memory_pool_size=recv_req.memory_pool_size,
                    gpu_id=recv_req.gpu_id,
                    model_name=recv_req.model_name,
                )
            
            # Update state
            self._prism_activated = True
            
            elapsed = time.perf_counter() - start_time
            logger.info(f"Prism: Model {recv_req.model_name} activated in {elapsed:.2f}s")
            
            return ActivateReqOutput(
                rid=recv_req.rid,
                success=True,
                memory_usage=self._prism_get_memory_usage(),
                model_name=recv_req.model_name,
                instance_idx=recv_req.instance_idx,
                gpu_id=recv_req.gpu_id,
            )
            
        except Exception as e:
            logger.error(f"Prism: Failed to activate model: {e}")
            return ActivateReqOutput(
                rid=recv_req.rid,
                success=False,
                memory_usage=self._prism_get_memory_usage(),
                model_name=recv_req.model_name,
                instance_idx=recv_req.instance_idx,
                gpu_id=recv_req.gpu_id,
            )
    
    def _prism_handle_deactivate_request(self, recv_req: DeactivateReqInput):
        """
        Handle model deactivation request.
        
        This deactivates a model, releasing GPU memory for other models.
        """
        logger.info(f"Prism: Handling deactivate request for model={recv_req.model_name}")
        
        # Check if already deactivated
        if not getattr(self, '_prism_activated', True):
            logger.warning(f"Model already deactivated, ignoring request rid={recv_req.rid}")
            return DeactivateReqOutput(
                rid=recv_req.rid,
                success=False,
                memory_usage=self._prism_get_memory_usage(),
                model_name=recv_req.model_name,
                instance_idx=recv_req.instance_idx,
                gpu_id=recv_req.gpu_id,
            )
        
        start_time = time.perf_counter()
        
        try:
            # Handle preemption if requested
            if recv_req.preempt:
                self._prism_handle_preemption(recv_req.preempt_mode)
            else:
                # Wait for running requests to complete
                self._prism_run_to_completion()
            
            # Deactivate model runner if available
            if hasattr(self, 'tp_worker') and hasattr(self.tp_worker, 'deactivate_model_runner'):
                self.tp_worker.deactivate_model_runner()
            
            # Update state
            self._prism_activated = False
            
            elapsed = time.perf_counter() - start_time
            logger.info(f"Prism: Model {recv_req.model_name} deactivated in {elapsed:.2f}s")
            
            return DeactivateReqOutput(
                rid=recv_req.rid,
                success=True,
                memory_usage=self._prism_get_memory_usage(),
                model_name=recv_req.model_name,
                instance_idx=recv_req.instance_idx,
                gpu_id=recv_req.gpu_id,
            )
            
        except Exception as e:
            logger.error(f"Prism: Failed to deactivate model: {e}")
            return DeactivateReqOutput(
                rid=recv_req.rid,
                success=False,
                memory_usage=self._prism_get_memory_usage(),
                model_name=recv_req.model_name,
                instance_idx=recv_req.instance_idx,
                gpu_id=recv_req.gpu_id,
            )
    
    def _prism_handle_get_mem_pool_size(self, recv_req: GetMemPoolSizeReq):
        """Handle request to get memory pool size."""
        size = getattr(self, 'max_total_num_tokens', 0)
        return GetMemPoolSizeReqOutput(size=size, rid=recv_req.rid)
    
    def _prism_handle_get_memory_usage(self, recv_req: GetMemoryUsageReq):
        """Handle request to get memory usage."""
        return GetMemoryUsageReqOutput(
            rid=recv_req.rid,
            memory_usage=self._prism_get_memory_usage(),
        )
    
    def _prism_handle_resize_mem_pool(self, recv_req: ResizeMemPoolReqInput):
        """Handle request to resize memory pool."""
        logger.info(f"Prism: Resizing memory pool to {recv_req.memory_pool_size}")
        
        if hasattr(self, 'tp_worker') and hasattr(self.tp_worker, 'resize_memory_pool'):
            success = self.tp_worker.resize_memory_pool(recv_req.memory_pool_size)
            if success:
                logger.info(f"Prism: Memory pool resized successfully")
            else:
                logger.warning(f"Prism: Failed to resize memory pool")
        
        return None  # No response needed
    
    def _prism_get_memory_usage(self) -> MemoryUsage:
        """Get current memory usage."""
        try:
            import torch
            
            # Get GPU memory info
            if torch.cuda.is_available():
                gpu_id = getattr(self, 'gpu_id', 0)
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            else:
                total_memory = 0.0
                allocated = 0.0
            
            # Estimate component sizes
            model_weights = getattr(self, '_prism_model_weights_memory', 0.0)
            
            return MemoryUsage(
                total_used_memory=allocated,
                model_weights_memory=model_weights,
                memory_pool_memory=allocated - model_weights,
                req_to_token_pool_memory=0.0,  # Would need actual tracking
                token_to_kv_pool_memory=0.0,   # Would need actual tracking
            )
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return MemoryUsage(
                total_used_memory=0.0,
                model_weights_memory=0.0,
                memory_pool_memory=0.0,
                req_to_token_pool_memory=0.0,
                token_to_kv_pool_memory=0.0,
            )
    
    def _prism_handle_preemption(self, preempt_mode):
        """Handle request preemption during deactivation."""
        from prism.io_struct import PreemptMode
        
        if preempt_mode == PreemptMode.RETURN:
            # Return partial results for running requests
            logger.debug("Prism: Preemption mode RETURN - returning partial results")
            # Implementation depends on scheduler internals
            pass
        elif preempt_mode == PreemptMode.RECOMPUTE:
            # Mark requests for recomputation
            logger.debug("Prism: Preemption mode RECOMPUTE - marking for recompute")
            pass
        elif preempt_mode == PreemptMode.ABORT:
            # Abort running requests
            logger.debug("Prism: Preemption mode ABORT - aborting requests")
            pass
    
    def _prism_run_to_completion(self):
        """Run all current requests to completion before deactivation."""
        logger.debug("Prism: Running requests to completion...")
        # This would integrate with scheduler's batch processing
        # For now, just log
        pass
    
    # Apply patches to Scheduler class
    Scheduler.init_request_dispatcher = patched_init_request_dispatcher
    Scheduler._prism_handle_activate_request = _prism_handle_activate_request
    Scheduler._prism_handle_deactivate_request = _prism_handle_deactivate_request
    Scheduler._prism_handle_get_mem_pool_size = _prism_handle_get_mem_pool_size
    Scheduler._prism_handle_get_memory_usage = _prism_handle_get_memory_usage
    Scheduler._prism_handle_resize_mem_pool = _prism_handle_resize_mem_pool
    Scheduler._prism_get_memory_usage = _prism_get_memory_usage
    Scheduler._prism_handle_preemption = _prism_handle_preemption
    Scheduler._prism_run_to_completion = _prism_run_to_completion
    
    # Initialize Prism state attributes (will be set per-instance)
    # Note: These are class-level defaults, instances may override
    Scheduler._prism_activated = True  # Default to activated
    Scheduler._prism_model_weights_memory = 0.0
    
    _patch_applied = True
    logger.info("Prism scheduler patch applied successfully.")


def is_patch_applied() -> bool:
    """Check if the scheduler patch has been applied."""
    return _patch_applied


__all__ = ["apply_scheduler_patch", "is_patch_applied"]
