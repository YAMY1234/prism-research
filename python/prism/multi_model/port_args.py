# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Custom PortArgs for Prism multi-model support.
This provides compatibility with SGLang while adding Prism-specific fields.
"""

import dataclasses
import tempfile
from typing import Optional

from sglang.srt.utils import is_port_available


@dataclasses.dataclass
class PrismPortArgs:
    """Port arguments for Prism multi-model server IPC communication."""
    
    # The ipc filename for request_handler to receive inputs from detokenizer (zmq)
    request_handler_ipc_name: str
    # The ipc filename for scheduler (rank 0) to receive inputs from controller/request_handler (zmq)
    scheduler_input_ipc_name: str
    # The ipc filename for detokenizer to receive inputs from scheduler (zmq)
    detokenizer_ipc_name: str
    # The port for nccl initialization (torch.dist)
    nccl_port: int
    # The ipc filename for controller to receive inputs from scheduler (zmq)
    controller_ipc_name: Optional[str] = None
    # The ipc filename for GPU scheduler to send requests to scheduler (zmq)
    # Scheduler binds, GPU scheduler connects
    gpu_scheduler_ipc_name: Optional[str] = None

    @staticmethod
    def init_new(server_args) -> "PrismPortArgs":
        """Initialize new port args from server args."""
        port = server_args.port + 1
        while True:
            if is_port_available(port):
                break
            port += 1

        return PrismPortArgs(
            request_handler_ipc_name=tempfile.NamedTemporaryFile(delete=False).name,
            scheduler_input_ipc_name=tempfile.NamedTemporaryFile(delete=False).name,
            detokenizer_ipc_name=tempfile.NamedTemporaryFile(delete=False).name,
            nccl_port=port,
            gpu_scheduler_ipc_name=tempfile.NamedTemporaryFile(delete=False).name,
        )

    @staticmethod
    def init_with_request_handler_ipc_name(
        start_port: int,
        request_handler_ipc_name: str,
        controller_ipc_name: Optional[str] = None,
    ) -> "PrismPortArgs":
        """Initialize port args with a specific request handler IPC name."""
        port = start_port + 1
        while True:
            if is_port_available(port):
                break
            port += 1

        return PrismPortArgs(
            request_handler_ipc_name=request_handler_ipc_name,
            scheduler_input_ipc_name=tempfile.NamedTemporaryFile(delete=False).name,
            detokenizer_ipc_name=tempfile.NamedTemporaryFile(delete=False).name,
            controller_ipc_name=controller_ipc_name,
            nccl_port=port,
            gpu_scheduler_ipc_name=tempfile.NamedTemporaryFile(delete=False).name,
        )
