# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Prism utility functions.
"""

import logging
import os
from typing import Any, Dict, Optional, Set

from prism.utils.redis_utils import RedisClient, AsyncRedisClient


def cleanup_zmq_ipc(
    zmq_sockets: Dict[str, Any] = None,
    ipc_files: Set[str] = None,
    component_name: str = "Component",
    gpu_id: Optional[int] = None,
    rank: Optional[int] = None,
):
    """Clean up ZeroMQ sockets and IPC files.

    Args:
        zmq_sockets: Dictionary of socket objects to close
        ipc_files: Set of IPC file paths to remove
        component_name: Name of the component being cleaned up (for logging)
        gpu_id: GPU ID (for logging)
        rank: Process rank (for logging)
    """
    logger = logging.getLogger(__name__)
    location_info = ""
    if gpu_id is not None:
        location_info += f" GPU {gpu_id}"
    if rank is not None:
        location_info += f" rank {rank}"

    try:
        logger.info(
            f"Cleaning up {component_name}{location_info} ZMQ sockets and IPC files"
        )

        # Close sockets
        if zmq_sockets:
            for name, socket in zmq_sockets.items():
                try:
                    if socket:
                        socket.close()
                        logger.debug(f"Closed {name} socket")
                except Exception as e:
                    logger.warning(f"Error closing {name} socket: {e}")

        # Remove IPC files
        if ipc_files:
            for ipc_file in ipc_files:
                try:
                    if ipc_file and os.path.exists(ipc_file):
                        os.unlink(ipc_file)
                        logger.debug(f"Removed IPC file: {ipc_file}")
                except Exception as e:
                    logger.warning(f"Error removing IPC file {ipc_file}: {e}")

        logger.info(f"{component_name}{location_info} cleanup complete")
    except Exception as e:
        logger.error(f"Error during {component_name}{location_info} cleanup: {e}")


def prepare_model_and_tokenizer(model_path: str, tokenizer_path: str):
    """Prepare model and tokenizer paths, handling modelscope downloads if needed.
    
    Args:
        model_path: Path to the model or modelscope model ID
        tokenizer_path: Path to the tokenizer or modelscope model ID
        
    Returns:
        Tuple of (model_path, tokenizer_path) after potential download
    """
    if "SGLANG_USE_MODELSCOPE" in os.environ:
        if not os.path.exists(model_path):
            from modelscope import snapshot_download

            model_path = snapshot_download(model_path)
            tokenizer_path = snapshot_download(
                tokenizer_path, ignore_patterns=["*.bin", "*.safetensors"]
            )
    return model_path, tokenizer_path


__all__ = ["RedisClient", "AsyncRedisClient", "cleanup_zmq_ipc", "prepare_model_and_tokenizer"]
