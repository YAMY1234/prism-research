# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Prism manager processes for multi-model serving.

This module provides custom run_scheduler_process and run_detokenizer_process
functions that wrap SGLang's managers with Prism's multi-model extensions.
"""

# IMPORTANT: Import prism.patches FIRST to apply scheduler patch before Scheduler is imported
import prism.patches  # noqa: F401 - side effect: applies Prism patches to SGLang

import atexit
import getpass
import logging
import os
import signal
from multiprocessing import shared_memory
from typing import Dict, List, Optional

import numpy as np
import psutil
import setproctitle
import torch
import torch.nn as nn

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    configure_logger,
    kill_process_tree,
    kill_itself_when_parent_died,
    suppress_other_loggers,
)
from sglang.utils import get_exception_traceback

from prism.multi_model.port_args import PrismPortArgs

logger = logging.getLogger(__name__)


def create_memory_usage_shm(gpu_id: int, model_name: str) -> shared_memory.SharedMemory:
    """
    Create shared memory for reporting memory usage to GPU scheduler.
    
    The shared memory stores an int64 value representing memory usage in bytes.
    Name format: ipc_{gpu_id}_{model_name}_{username}
    """
    username = getpass.getuser()
    shm_name = f"ipc_{gpu_id}_{model_name}_{username}"
    
    # Try to clean up any existing shared memory with the same name
    try:
        old_shm = shared_memory.SharedMemory(name=shm_name)
        old_shm.close()
        old_shm.unlink()
    except FileNotFoundError:
        pass
    
    # Create new shared memory (8 bytes for int64)
    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=8)
    
    # Initialize to 0
    mem_array = np.ndarray((1,), dtype=np.int64, buffer=shm.buf)
    mem_array[0] = 0
    
    logger.info(f"Created shared memory {shm_name} for memory usage tracking")
    return shm


def cleanup_memory_usage_shm(shm: shared_memory.SharedMemory):
    """Clean up shared memory on exit."""
    try:
        shm.close()
        shm.unlink()
    except Exception as e:
        logger.debug(f"Failed to cleanup shared memory: {e}")


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PrismPortArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    shared_cpu_models: Optional[Dict[str, List[nn.Module]]] = None,
    model_names_to_model_paths: Optional[Dict[str, str]] = None,
    engine_id: Optional[str] = None,
    input_queue: Optional[torch.multiprocessing.Queue] = None,
    output_queue: Optional[torch.multiprocessing.Queue] = None,
):
    """
    Prism's custom scheduler process launcher.
    
    This extends SGLang's scheduler with multi-model support parameters.
    """
    kill_itself_when_parent_died()
    
    # Debug: check if setattr'd Prism attrs survived pickle across mp.Process
    _bkp = getattr(server_args, 'backend_generate_request_key_prefix', 'MISSING_AFTER_PICKLE')
    _em = getattr(server_args, 'enable_elastic_memory', 'MISSING_AFTER_PICKLE')
    import sys
    sys.stderr.write(f"[run_scheduler_process] backend_key_prefix={_bkp}, enable_elastic_memory={_em}\n")
    sys.stderr.flush()
    
    # Set process title for identification
    model_name = getattr(server_args, 'served_model_name', None) or getattr(server_args, 'model_name', 'unknown')
    setproctitle.setproctitle(f"sglang::scheduler::{model_name}")
    
    # Configure logging
    if dp_rank is None:
        if getattr(server_args, 'enable_worker_pool', False):
            worker_id = getattr(server_args, 'worker_id', 0)
            configure_logger(
                server_args,
                prefix=f" GPU={gpu_id} Worker={worker_id} TP{tp_rank}",
            )
        else:
            configure_logger(
                server_args,
                prefix=f" {model_name} GPU={gpu_id} TP{tp_rank}",
            )
    else:
        configure_logger(
            server_args,
            prefix=f" {model_name} GPU={gpu_id} DP{dp_rank} TP{tp_rank}",
        )

    suppress_other_loggers()

    try:
        # Convert PrismPortArgs to SGLang's PortArgs format
        # pip-installed SGLang PortArgs: (tokenizer_ipc_name, scheduler_input_ipc_name, detokenizer_ipc_name, 
        #                                  nccl_port, rpc_ipc_name, metrics_ipc_name, tokenizer_worker_ipc_name)
        from sglang.srt.server_args import PortArgs as SGLangPortArgs
        import tempfile
        
        sglang_port_args = SGLangPortArgs(
            tokenizer_ipc_name=f"ipc://{port_args.request_handler_ipc_name}",
            scheduler_input_ipc_name=f"ipc://{port_args.scheduler_input_ipc_name}",
            detokenizer_ipc_name=f"ipc://{port_args.detokenizer_ipc_name}",
            nccl_port=port_args.nccl_port,
            rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            tokenizer_worker_ipc_name=None,
        )
        
        # Initialize scheduler with pip-installed SGLang parameters
        # Latest SGLang Scheduler signature: (server_args, port_args, gpu_id, tp_rank, moe_ep_rank, pp_rank, attn_cp_rank, moe_dp_rank, dp_rank)
        import inspect
        scheduler_params = inspect.signature(Scheduler.__init__).parameters
        scheduler_kwargs = {
            "server_args": server_args,
            "port_args": sglang_port_args,
            "gpu_id": gpu_id,
            "tp_rank": tp_rank,
            "dp_rank": dp_rank,
        }
        for param_name in ["moe_ep_rank", "pp_rank", "attn_cp_rank", "moe_dp_rank"]:
            if param_name in scheduler_params:
                scheduler_kwargs[param_name] = 0
        scheduler = Scheduler(**scheduler_kwargs)
        
        # Store WorkerPool parameters as attributes (prism-old style)
        scheduler.model_names_to_model_paths = model_names_to_model_paths
        scheduler.engine_id = engine_id
        scheduler.input_queue = input_queue
        scheduler.output_queue = output_queue
        
        # Create shared memory for memory usage tracking (for GPU scheduler)
        # In WorkerPool mode, use worker_id instead of model_name (model_name may contain '/')
        is_worker_pool = getattr(server_args, 'enable_worker_pool', False)
        shm_label = str(getattr(server_args, 'worker_id', 0)) if is_worker_pool else model_name
        memory_usage_shm = create_memory_usage_shm(gpu_id, shm_label)
        atexit.register(cleanup_memory_usage_shm, memory_usage_shm)
        
        # Store prism-specific data in scheduler for patch access
        scheduler._prism_shared_cpu_models = shared_cpu_models
        scheduler._prism_model_names_to_paths = model_names_to_model_paths
        scheduler._prism_engine_id = engine_id
        scheduler._prism_input_queue = input_queue
        scheduler._prism_output_queue = output_queue
        scheduler._prism_gpu_id = gpu_id
        scheduler._prism_model_name = model_name
        scheduler._prism_memory_usage_shm = memory_usage_shm
        scheduler._prism_memory_usage_array = np.ndarray((1,), dtype=np.int64, buffer=memory_usage_shm.buf)
        
        # pip-installed SGLang doesn't have redis_client - we need to create it
        from prism.utils.redis_utils import RedisClient
        backend_key = getattr(server_args, 'backend_generate_request_key_prefix', None)
        
        if tp_rank == 0:
            redis_host = getattr(server_args, 'redis_host', 'localhost')
            redis_port = getattr(server_args, 'redis_port', 6379)
            redis_db = getattr(server_args, 'redis_db', 0)
            scheduler.redis_client = RedisClient(redis_host, redis_port, redis_db)
            logger.info(f"Prism: Created Redis client, backend_key_prefix={backend_key}")
        else:
            scheduler.redis_client = None
        
        # Create ZMQ channel for receiving activate/deactivate from GPU Scheduler
        # In prism-old, Scheduler binds to ipc://gpu_scheduler_{gpu_id}_to_worker_{worker_id}
        # WorkerPool or GPU Scheduler connects to this channel to send commands
        import zmq
        worker_id = getattr(server_args, 'worker_id', 0)
        gpu_sched_ipc = f"gpu_scheduler_{gpu_id}_to_worker_{worker_id}"
        scheduler._prism_zmq_ctx = zmq.Context(1)  # Store ref to prevent GC
        scheduler._prism_recv_from_gpu_scheduler = scheduler._prism_zmq_ctx.socket(zmq.PULL)
        scheduler._prism_recv_from_gpu_scheduler.bind(f"ipc://{gpu_sched_ipc}")
        logger.info(f"Prism: Bound to {gpu_sched_ipc} for GPU Scheduler commands")
        
        is_worker_pool = getattr(server_args, 'enable_worker_pool', False)
        
        if is_worker_pool:
            # WorkerPool mode: start deactivated, model_name not yet known
            scheduler._activated = False
            scheduler.model_name = None
            logger.info(f"Prism: Worker {worker_id} on GPU {gpu_id} started (WorkerPool, deactivated)")
        else:
            # Non-WorkerPool: start activated with known model
            scheduler._activated = True
            scheduler.model_name = model_name
            logger.info(f"Prism: Activated scheduler for {model_name}")
        
        # Disable idle_sleeper if present
        if hasattr(scheduler, 'idle_sleeper') and scheduler.idle_sleeper is not None:
            scheduler.idle_sleeper = None
            logger.info(f"Prism: Disabled idle_sleeper")
        
        # Send memory usage back through pipe
        # Use Prism's patched method to get memory usage
        if hasattr(scheduler, '_prism_get_memory_usage'):
            mem_usage = scheduler._prism_get_memory_usage()
        else:
            mem_usage = None
        pipe_writer.send(mem_usage)
        
        # Start event loop
        logger.info(f"Prism: Starting scheduler event loop for {model_name}, overlap={getattr(server_args, 'enable_overlap_schedule', False)}")
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        if getattr(server_args, 'enable_overlap_schedule', False):
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
        # This should never be reached
        logger.error(f"Prism: Scheduler event loop exited unexpectedly for {model_name}!")
            
    except Exception:
        msg = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {msg}")
        parent_process = psutil.Process().parent()
        parent_process.send_signal(signal.SIGQUIT)


def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PrismPortArgs,
    model_names_to_model_paths: Optional[Dict[str, str]] = None,
):
    """
    Prism's custom detokenizer process launcher.
    
    This extends SGLang's detokenizer with multi-model support.
    """
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        # Convert PrismPortArgs to standard PortArgs format
        from sglang.srt.server_args import PortArgs as SGLangPortArgs
        import tempfile
        
        sglang_port_args = SGLangPortArgs(
            tokenizer_ipc_name=f"ipc://{port_args.request_handler_ipc_name}",
            scheduler_input_ipc_name=f"ipc://{port_args.scheduler_input_ipc_name}",
            detokenizer_ipc_name=f"ipc://{port_args.detokenizer_ipc_name}",
            nccl_port=port_args.nccl_port,
            rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            tokenizer_worker_ipc_name=None,
        )
        
        manager = DetokenizerManager(server_args, sglang_port_args)
        
        # Store prism-specific data
        manager._prism_model_names_to_paths = model_names_to_model_paths
        
        # Start event loop
        if getattr(server_args, 'tokenizer_worker_num', 1) == 1:
            manager.event_loop()
        else:
            manager.multi_http_worker_event_loop()
            
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        if 'manager' in locals():
            manager.maybe_clear_socket_mapping()
        parent_process.send_signal(signal.SIGQUIT)
