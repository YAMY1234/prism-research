# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Prism manager processes for multi-model serving.

This module provides custom run_scheduler_process and run_detokenizer_process
functions that wrap SGLang's managers with Prism's multi-model extensions.
"""

import logging
import os
import signal
from typing import Dict, List, Optional

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
        # Convert PrismPortArgs to standard PortArgs format expected by new SGLang
        from sglang.srt.server_args import PortArgs as SGLangPortArgs
        import tempfile
        
        # Create a compatible PortArgs for new SGLang
        sglang_port_args = SGLangPortArgs(
            tokenizer_ipc_name=f"ipc://{port_args.request_handler_ipc_name}",
            scheduler_input_ipc_name=f"ipc://{port_args.scheduler_input_ipc_name}",
            detokenizer_ipc_name=f"ipc://{port_args.detokenizer_ipc_name}",
            nccl_port=port_args.nccl_port,
            rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            tokenizer_worker_ipc_name=None,
        )
        
        # Initialize scheduler with standard SGLang parameters
        # Note: Prism's extra parameters (shared_cpu_models, etc.) are handled via patches
        scheduler = Scheduler(
            server_args,
            sglang_port_args,
            gpu_id,
            tp_rank,
            0,  # moe_ep_rank
            0,  # pp_rank
            dp_rank,
        )
        
        # Store prism-specific data in scheduler for patch access
        scheduler._prism_shared_cpu_models = shared_cpu_models
        scheduler._prism_model_names_to_paths = model_names_to_model_paths
        scheduler._prism_engine_id = engine_id
        scheduler._prism_input_queue = input_queue
        scheduler._prism_output_queue = output_queue
        
        # Send memory usage back through pipe
        if hasattr(scheduler, 'get_memory_usage'):
            mem_usage = scheduler.get_memory_usage()
        else:
            mem_usage = None
        pipe_writer.send(mem_usage)
        
        # Start event loop
        if getattr(server_args, 'enable_overlap_schedule', False):
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
            
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
