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
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    
    # Import Prism data structures
    from prism.io_struct import (
        ActivateReqInput,
        ActivateReqOutput,
        DeactivateReqInput,
        DeactivateReqOutput,
        GenerateReqInput,
        GetMemoryUsageReq,
        GetMemoryUsageReqOutput,
        GetMemPoolSizeReq,
        GetMemPoolSizeReqOutput,
        MemoryUsage,
        ResizeMemPoolReqInput,
    )
    
    # Save original methods (only what we actually patch)
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
        # Handle raw GenerateReqInput from GPU scheduler (needs tokenization)
        self._request_dispatcher._mapping[GenerateReqInput] = self._prism_handle_raw_generate_request
        
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
    
    def _prism_handle_raw_generate_request(self, recv_req: GenerateReqInput):
        """
        Handle raw GenerateReqInput from GPU scheduler.
        
        This method tokenizes the request and converts it to TokenizedGenerateReqInput
        before passing it to the standard generate request handler.
        """
        logger.info(f"[DEBUG] Scheduler received raw GenerateReqInput rid={recv_req.rid}, model={getattr(recv_req, 'model', None)}")
        
        try:
            logger.info(f"[DEBUG] Request has text={recv_req.text is not None}, input_ids={recv_req.input_ids is not None}")
            # Get text or input_ids
            if recv_req.input_ids is not None:
                if isinstance(recv_req.input_ids, list) and len(recv_req.input_ids) > 0:
                    if isinstance(recv_req.input_ids[0], int):
                        input_ids = recv_req.input_ids
                    else:
                        input_ids = recv_req.input_ids[0]
                else:
                    input_ids = recv_req.input_ids
            elif recv_req.text is not None:
                # Tokenize text using scheduler's tokenizer
                text = recv_req.text if isinstance(recv_req.text, str) else recv_req.text[0]
                if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                    input_ids = self.tokenizer.encode(text)
                else:
                    logger.error("No tokenizer available for text tokenization")
                    return
            else:
                logger.error("No text or input_ids in request")
                return
            
            # Get sampling params
            sampling_params = recv_req.sampling_params
            if isinstance(sampling_params, list):
                sampling_params = sampling_params[0] if len(sampling_params) > 0 else {}
            if isinstance(sampling_params, dict):
                from sglang.srt.sampling.sampling_params import SamplingParams
                sampling_params = SamplingParams.from_dict(sampling_params)
            
            # Create TokenizedGenerateReqInput
            tokenized_req = TokenizedGenerateReqInput(
                rid=recv_req.rid,
                input_text=recv_req.text if isinstance(recv_req.text, str) else (recv_req.text[0] if recv_req.text else ""),
                input_ids=input_ids,
                mm_inputs={},  # No multimodal inputs for now
                sampling_params=sampling_params,
                return_logprob=getattr(recv_req, 'return_logprob', False),
                logprob_start_len=getattr(recv_req, 'logprob_start_len', 0),
                top_logprobs_num=getattr(recv_req, 'top_logprobs_num', 0),
                stream=getattr(recv_req, 'stream', False),
                lora_path=getattr(recv_req, 'lora_path', None),
            )
            
            # Call the standard generate request handler
            logger.info(f"[DEBUG] Scheduler calling handle_generate_request for rid={recv_req.rid}")
            self.handle_generate_request(tokenized_req)
            logger.info(f"[DEBUG] Scheduler finished handle_generate_request for rid={recv_req.rid}")
            
        except Exception as e:
            logger.error(f"[DEBUG] Prism: Failed to handle raw generate request rid={recv_req.rid}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _prism_get_memory_usage(self) -> MemoryUsage:
        """Get current memory usage and update shared memory for GPU scheduler."""
        try:
            import torch
            
            # Get GPU memory info
            if torch.cuda.is_available():
                gpu_id = getattr(self, 'gpu_id', 0)
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                allocated_bytes = torch.cuda.memory_allocated(gpu_id)
            else:
                total_memory = 0.0
                allocated = 0.0
                allocated_bytes = 0
            
            # Update shared memory for GPU scheduler
            self._prism_update_memory_usage_shm(allocated_bytes)
            
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
    
    def _prism_update_memory_usage_shm(self, memory_bytes: int):
        """Update shared memory with current memory usage for GPU scheduler."""
        try:
            mem_array = getattr(self, '_prism_memory_usage_array', None)
            if mem_array is not None:
                mem_array[0] = memory_bytes
        except Exception as e:
            logger.debug(f"Failed to update memory usage shared memory: {e}")
    
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
    
    def _prism_recv_generation_requests(self):
        """
        Receive generation requests from Redis backend queue.
        
        This is the Prism extension - pip-installed SGLang doesn't have this.
        GPU scheduler sends requests to Redis, scheduler reads from here.
        """
        recv_reqs = []
        
        # Only tp_rank == 0 should read from Redis
        if getattr(self, 'tp_rank', 0) != 0:
            return recv_reqs
        
        # Check if activated
        if not getattr(self, '_activated', True):
            return recv_reqs
        
        # Get Redis client (created in managers.py)
        redis_client = getattr(self, 'redis_client', None)
        if redis_client is None:
            return recv_reqs
        
        # Get model name and backend key prefix
        model_name = getattr(self, 'model_name', None)
        server_args = getattr(self, 'server_args', None)
        if not model_name or not server_args:
            return recv_reqs
        
        backend_key_prefix = getattr(server_args, 'backend_generate_request_key_prefix', None)
        if not backend_key_prefix:
            return recv_reqs
        
        key = f"{backend_key_prefix}:{model_name}"
        
        try:
            # Non-blocking read from Redis queue
            recv_reqs = redis_client.recv_pyobj_non_block(key=key, count=32)
            if recv_reqs:
                logger.info(f"Prism: Received {len(recv_reqs)} generation requests from Redis for {model_name}")
        except Exception as e:
            logger.warning(f"Prism: Redis recv error for {model_name}: {e}")
        
        return recv_reqs
    
    # Save original event_loop_normal
    _original_event_loop_normal = Scheduler.event_loop_normal
    
    def patched_event_loop_normal(self):
        """
        Patched event loop that adds Redis support for Prism multi-model serving.
        
        pip-installed SGLang doesn't have recv_generation_requests(), so we add it here.
        This polls both ZMQ (for control messages) and Redis (for generation requests).
        """
        import time as time_module
        
        model_name = getattr(self, 'model_name', 'unknown')
        logger.info(f"Prism: {model_name} event_loop_normal STARTED")
        
        self.last_batch = None
        
        while True:
            # Step 1: Receive requests from tokenizer (ZMQ) - uses zmq.NOBLOCK internally
            recv_reqs = []
            if hasattr(self, 'recv_requests'):
                recv_reqs = self.recv_requests()
            
            # Step 2: Receive generation requests from Redis (Prism extension)
            redis_reqs = self._prism_recv_generation_requests()
            if redis_reqs:
                # Process each raw request through our handler
                for req in redis_reqs:
                    self._prism_handle_raw_generate_request(req)
            
            # Step 3: Process ZMQ requests
            if recv_reqs:
                self.process_input_requests(recv_reqs)
            
            # Step 4: Run batch if activated
            if getattr(self, '_activated', True):
                batch = self.get_next_batch_to_run()
                
                if batch:
                    result = self.run_batch(batch)
                    self.process_batch_result(batch, result)
                    
                    # Decode multiple steps
                    if hasattr(batch, 'forward_mode') and batch.forward_mode.is_decode():
                        num_steps = getattr(self.server_args, 'num_continuous_decode_steps', 1) - 1
                        for _ in range(num_steps):
                            if not self.running_batch:
                                break
                            self.update_running_batch()
                            if not self.running_batch:
                                break
                            result = self.run_batch(batch)
                            self.process_batch_result(batch, result)
                else:
                    # No batch to run - brief sleep to avoid busy loop
                    time_module.sleep(0.001)
                    if hasattr(self, 'check_memory'):
                        self.check_memory()
                
                self.last_batch = batch
            else:
                # Not activated - sleep
                time_module.sleep(0.001)
    
    # Apply patches to Scheduler class
    # IMPORTANT: We MUST patch event_loop_normal because pip-installed SGLang
    # doesn't have recv_generation_requests() for Redis support.
    Scheduler.init_request_dispatcher = patched_init_request_dispatcher
    Scheduler.event_loop_normal = patched_event_loop_normal
    Scheduler._prism_recv_generation_requests = _prism_recv_generation_requests
    Scheduler._prism_handle_activate_request = _prism_handle_activate_request
    Scheduler._prism_handle_deactivate_request = _prism_handle_deactivate_request
    Scheduler._prism_handle_get_mem_pool_size = _prism_handle_get_mem_pool_size
    Scheduler._prism_handle_get_memory_usage = _prism_handle_get_memory_usage
    Scheduler._prism_handle_resize_mem_pool = _prism_handle_resize_mem_pool
    Scheduler._prism_handle_raw_generate_request = _prism_handle_raw_generate_request
    Scheduler._prism_get_memory_usage = _prism_get_memory_usage
    Scheduler._prism_update_memory_usage_shm = _prism_update_memory_usage_shm
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
