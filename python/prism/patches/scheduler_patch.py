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
        
        In WorkerPool mode, this dynamically binds model_name, tokenizer, etc.
        In non-WorkerPool mode, just activates the already-loaded model.
        Sends response to both detokenizer and GPU Scheduler (via Redis).
        """
        logger.info(f"Prism: Handling activate request for model={recv_req.model_name}, gpu={recv_req.gpu_id}")
        
        if getattr(self, '_activated', False):
            logger.warning(f"Model already activated, ignoring request rid={recv_req.rid}")
            output = ActivateReqOutput(
                rid=recv_req.rid, success=False,
                memory_usage=self._prism_get_memory_usage(),
                model_name=recv_req.model_name,
                instance_idx=recv_req.instance_idx, gpu_id=recv_req.gpu_id,
            )
            self._prism_send_activate_response(output)
            return output
        
        start_time = time.perf_counter()
        gpu_id = recv_req.gpu_id if recv_req.gpu_id is not None else getattr(self, 'gpu_id', 0)
        
        try:
            # Activate model runner
            if hasattr(self, 'tp_worker') and hasattr(self.tp_worker, 'activate_model_runner'):
                self.tp_worker.activate_model_runner(
                    memory_pool_size=recv_req.memory_pool_size,
                    gpu_id=gpu_id,
                    model_name=recv_req.model_name,
                )
            
            # --- WorkerPool dynamic binding (matching prism-old) ---
            # Bind model_name so _prism_recv_generation_requests reads the correct Redis key
            self.model_name = recv_req.model_name
            
            # Get tokenizer — needed for _prism_handle_raw_generate_request
            if getattr(self, 'enable_worker_pool', False) or not hasattr(self, 'tokenizer') or self.tokenizer is None:
                # WorkerPool or missing tokenizer: get from tp_worker or load fresh
                if hasattr(self.tp_worker, 'get_tokenizer'):
                    self.tokenizer = self.tp_worker.get_tokenizer()
                elif hasattr(self, 'model_names_to_model_paths') and self.model_names_to_model_paths:
                    model_path = self.model_names_to_model_paths.get(recv_req.model_name)
                    if model_path:
                        from sglang.srt.utils.hf_transformers_utils import get_tokenizer
                        self.tokenizer = get_tokenizer(
                            model_path,
                            tokenizer_mode=getattr(self.server_args, 'tokenizer_mode', 'auto'),
                            trust_remote_code=getattr(self.server_args, 'trust_remote_code', True),
                        )
                        logger.info(f"Prism: Loaded tokenizer for {recv_req.model_name}")
            
            # Get model_config if available
            if hasattr(self.tp_worker, 'get_model_config'):
                self.model_config = self.tp_worker.get_model_config()
            
            # Update pad_input_ids_func on first activate
            if getattr(self, 'first_time_activate', True):
                if hasattr(self.tp_worker, 'get_pad_input_ids_func'):
                    self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
                self.first_time_activate = False
            
            # Set activated
            self._activated = True
            self._prism_activated = True
            
            elapsed = time.perf_counter() - start_time
            logger.info(f"Prism: Model {recv_req.model_name} activated in {elapsed:.2f}s")
            
            output = ActivateReqOutput(
                rid=recv_req.rid, success=True,
                memory_usage=self._prism_get_memory_usage(),
                model_name=recv_req.model_name,
                instance_idx=recv_req.instance_idx, gpu_id=gpu_id,
            )
            self._prism_send_activate_response(output)
            return output
            
        except Exception as e:
            logger.error(f"Prism: Failed to activate model: {e}")
            import traceback as tb
            logger.error(tb.format_exc())
            output = ActivateReqOutput(
                rid=recv_req.rid, success=False,
                memory_usage=self._prism_get_memory_usage(),
                model_name=recv_req.model_name,
                instance_idx=recv_req.instance_idx, gpu_id=gpu_id,
            )
            self._prism_send_activate_response(output)
            return output
    
    def _prism_send_activate_response(self, output: ActivateReqOutput):
        """Send activate response to detokenizer (ZMQ) and GPU Scheduler (Redis)."""
        # Send to detokenizer → request_handler
        if hasattr(self, 'send_to_detokenizer'):
            try:
                self.send_to_detokenizer.send_pyobj(output)
            except Exception as e:
                logger.warning(f"Prism: Failed to send activate response to detokenizer: {e}")
        
        # Send to GPU Scheduler via Redis
        redis_client = getattr(self, 'redis_client', None)
        gpu_id = getattr(self, 'gpu_id', 0)
        engine_key_prefix = getattr(self.server_args, 'engine_to_gpu_scheduler_key_prefix', None)
        if redis_client and engine_key_prefix:
            try:
                redis_client.send_pyobj(
                    key=f"{engine_key_prefix}:{gpu_id}",
                    obj=output,
                )
            except Exception as e:
                logger.warning(f"Prism: Failed to send activate response to GPU Scheduler: {e}")
    
    def _prism_handle_deactivate_request(self, recv_req: DeactivateReqInput):
        """
        Handle model deactivation request.
        
        Deactivates the model, releases GPU memory.
        Sends response to both detokenizer and GPU Scheduler.
        """
        logger.info(f"Prism: Handling deactivate request for model={recv_req.model_name}")
        
        gpu_id = recv_req.gpu_id if recv_req.gpu_id is not None else getattr(self, 'gpu_id', 0)
        
        if not getattr(self, '_activated', False):
            logger.warning(f"Model already deactivated, ignoring request rid={recv_req.rid}")
            output = DeactivateReqOutput(
                rid=recv_req.rid, success=False,
                memory_usage=self._prism_get_memory_usage(),
                model_name=recv_req.model_name,
                instance_idx=recv_req.instance_idx, gpu_id=gpu_id,
            )
            self._prism_send_deactivate_response(output)
            return output
        
        start_time = time.perf_counter()
        
        try:
            if recv_req.preempt:
                self._prism_handle_preemption(recv_req.preempt_mode)
            else:
                self._prism_run_to_completion()
            
            # Deactivate model runner
            if hasattr(self, 'tp_worker') and hasattr(self.tp_worker, 'deactivate_model_runner'):
                self.tp_worker.deactivate_model_runner()
            
            # Set deactivated
            self._activated = False
            self._prism_activated = False
            
            elapsed = time.perf_counter() - start_time
            logger.info(f"Prism: Model {recv_req.model_name} deactivated in {elapsed:.2f}s")
            
            output = DeactivateReqOutput(
                rid=recv_req.rid, success=True,
                memory_usage=self._prism_get_memory_usage(),
                model_name=recv_req.model_name,
                instance_idx=recv_req.instance_idx, gpu_id=gpu_id,
            )
            self._prism_send_deactivate_response(output)
            return output
            
        except Exception as e:
            logger.error(f"Prism: Failed to deactivate model: {e}")
            import traceback as tb
            logger.error(tb.format_exc())
            output = DeactivateReqOutput(
                rid=recv_req.rid, success=False,
                memory_usage=self._prism_get_memory_usage(),
                model_name=recv_req.model_name,
                instance_idx=recv_req.instance_idx, gpu_id=gpu_id,
            )
            self._prism_send_deactivate_response(output)
            return output
    
    def _prism_send_deactivate_response(self, output: DeactivateReqOutput):
        """Send deactivate response to detokenizer (ZMQ) and GPU Scheduler (Redis)."""
        if hasattr(self, 'send_to_detokenizer'):
            try:
                self.send_to_detokenizer.send_pyobj(output)
            except Exception as e:
                logger.warning(f"Prism: Failed to send deactivate response to detokenizer: {e}")
        
        redis_client = getattr(self, 'redis_client', None)
        gpu_id = getattr(self, 'gpu_id', 0)
        engine_key_prefix = getattr(self.server_args, 'engine_to_gpu_scheduler_key_prefix', None)
        if redis_client and engine_key_prefix:
            try:
                redis_client.send_pyobj(
                    key=f"{engine_key_prefix}:{gpu_id}",
                    obj=output,
                )
            except Exception as e:
                logger.warning(f"Prism: Failed to send deactivate response to GPU Scheduler: {e}")
    
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
    
    import zmq as _zmq_module
    _ZMQ_NOBLOCK = _zmq_module.NOBLOCK
    _ZMQ_ERROR = _zmq_module.ZMQError
    
    def _prism_recv_gpu_scheduler_requests(self):
        """Receive activate/deactivate requests from GPU Scheduler via ZMQ (non-blocking)."""
        recv_reqs = []
        sock = getattr(self, '_prism_recv_from_gpu_scheduler', None)
        if sock is None:
            return recv_reqs
        try:
            while True:
                try:
                    req = sock.recv_pyobj(_ZMQ_NOBLOCK)
                    recv_reqs.append(req)
                except _ZMQ_ERROR:
                    break
        except Exception as e:
            logger.error(f"Prism: _prism_recv_gpu_scheduler_requests error: {e}")
        return recv_reqs
    
    def patched_event_loop_normal(self):
        import time as _t
        _name = getattr(self, 'model_name', None) or 'unassigned'
        logger.info(f"Prism: {_name} event_loop_normal STARTED")
        self.last_batch = None
        _n = 0
        _tl = _t.time()
        import os as _os
        _pid = _os.getpid()
        _dbg_path = f"/tmp/prism_worker_{_pid}.log"
        def _dbg(msg):
            with open(_dbg_path, "a") as f:
                f.write(f"{_t.time():.3f} [{_name}] {msg}\n")
        _dbg(f"entering while True, activated={getattr(self,'_activated','?')}")
        while True:
            _n += 1
            try:
                _tn = _t.time()
                if _tn - _tl >= 3.0:
                    _tl = _tn
                    print(f"[{_name}] loop#{_n} act={getattr(self,'_activated','?')}", flush=True)
                if _n <= 5 or _n % 1000 == 0:
                    _dbg(f"loop#{_n} begin")
                gs = self._prism_recv_gpu_scheduler_requests()
                if _n <= 5:
                    _dbg(f"loop#{_n} gpu_sched={len(gs)}")
                if gs:
                    logger.info(f"Prism: Got {len(gs)} GPU Scheduler reqs")
                    self.process_input_requests(gs)
                if getattr(self, '_activated', False):
                    if hasattr(self, 'recv_requests'):
                        rr = self.recv_requests()
                        if rr:
                            self.process_input_requests(rr)
                    rq = self._prism_recv_generation_requests()
                    if rq:
                        for r in rq:
                            self._prism_handle_raw_generate_request(r)
                    b = self.get_next_batch_to_run()
                    if b:
                        res = self.run_batch(b)
                        self.process_batch_result(b, res)
                        if hasattr(b, 'forward_mode') and b.forward_mode.is_decode():
                            for _ in range(getattr(self.server_args, 'num_continuous_decode_steps', 1) - 1):
                                if not self.running_batch:
                                    break
                                self.update_running_batch()
                                if not self.running_batch:
                                    break
                                res = self.run_batch(b)
                                self.process_batch_result(b, res)
                    else:
                        _t.sleep(0.001)
                        if hasattr(self, 'check_memory'):
                            self.check_memory()
                    self.last_batch = b
                else:
                    _t.sleep(0.001)
            except Exception as e:
                print(f"[{_name}] LOOP ERR #{_n}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                _t.sleep(1.0)
    
    # Apply patches to Scheduler class
    # IMPORTANT: We MUST patch event_loop_normal because pip-installed SGLang
    # doesn't have recv_generation_requests() for Redis support.
    Scheduler.init_request_dispatcher = patched_init_request_dispatcher
    Scheduler.event_loop_normal = patched_event_loop_normal
    Scheduler._prism_recv_generation_requests = _prism_recv_generation_requests
    Scheduler._prism_recv_gpu_scheduler_requests = _prism_recv_gpu_scheduler_requests
    Scheduler._prism_handle_activate_request = _prism_handle_activate_request
    Scheduler._prism_handle_deactivate_request = _prism_handle_deactivate_request
    Scheduler._prism_send_activate_response = _prism_send_activate_response
    Scheduler._prism_send_deactivate_response = _prism_send_deactivate_response
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
