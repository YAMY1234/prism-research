# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Multi-Model Server Arguments for Prism.

This module defines configuration classes and argument parsing for
Prism's multi-model serving system.
"""

import argparse
import dataclasses
import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Lazy import from sglang for compatibility
def is_flashinfer_available():
    """Check if flashinfer is available."""
    try:
        from sglang.srt.utils import is_flashinfer_available as _is_flashinfer_available
        return _is_flashinfer_available()
    except ImportError:
        return False

def is_ipv6(host: str) -> bool:
    """Check if host is IPv6."""
    try:
        from sglang.srt.utils import is_ipv6 as _is_ipv6
        return _is_ipv6(host)
    except ImportError:
        return ":" in host and not host.startswith("[")

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Placement:
    """Placement configuration for a model instance."""
    
    gpu_ids: List[int]  # For TP > 1, list of GPU IDs
    on: bool = True     # Whether the instance should be activated initially
    max_memory_pool_size: Optional[float] = None  # Max memory in GB


@dataclasses.dataclass
class ModelConfig:
    """Configuration for a single model."""
    
    model_name: str  # Unique identifier for the model
    model_path: str  # Path to model weights
    tokenizer_path: Optional[str] = None
    tp_size: int = 1
    init_placements: List[Placement] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

    def get_instance_configs(self) -> List["InstanceConfig"]:
        """Get instance configurations from placements."""
        return [
            InstanceConfig(
                model_name=self.model_name,
                model_path=self.model_path,
                tokenizer_path=self.tokenizer_path,
                gpu_ids=placement["gpu_ids"] if isinstance(placement, dict) else placement.gpu_ids,
                tp_size=self.tp_size,
                on=placement.get("on", True) if isinstance(placement, dict) else placement.on,
                max_memory_pool_size=placement.get("max_memory_pool_size") if isinstance(placement, dict) else placement.max_memory_pool_size,
            )
            for placement in self.init_placements
        ]


@dataclasses.dataclass
class InstanceConfig:
    """Configuration for a model instance."""
    
    model_name: str
    model_path: str
    tokenizer_path: Optional[str] = None
    gpu_ids: List[int] = field(default_factory=list)
    tp_size: int = 1
    on: bool = True
    max_memory_pool_size: Optional[float] = None


def load_model_configs(file_path: str) -> List[ModelConfig]:
    """Load model configurations from a JSON file."""
    with open(file_path, "r") as f:
        config_data = json.load(f)
        model_configs = [ModelConfig(**model) for model in config_data]
    return model_configs


@dataclasses.dataclass
class MultiModelServerArgs:
    """Arguments for the multi-model server."""
    
    # Model and tokenizer
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    tokenizer_path: Optional[str] = None
    model_config_file: Optional[str] = None
    model_configs: Optional[List[ModelConfig]] = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    load_format: str = "auto"
    trust_remote_code: bool = True
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    quantization: Optional[str] = None
    context_length: Optional[int] = None
    device: str = "cuda"
    served_model_name: Optional[str] = None
    chat_template: Optional[str] = None
    is_embedding: bool = False

    # Port
    host: str = "127.0.0.1"
    port: int = 30000

    # Worker pool
    enable_worker_pool: bool = False
    workers_per_gpu: int = 1
    num_gpus: int = 1

    # Memory and scheduling
    mem_fraction_static: Optional[float] = None
    max_running_requests: Optional[int] = None
    max_total_tokens: Optional[int] = None
    max_mem_usage: Optional[float] = None
    max_memory_pool_size: Optional[float] = None
    chunked_prefill_size: int = 8192
    max_prefill_tokens: int = 16384
    schedule_policy: str = "lpm"
    schedule_conservativeness: float = 1.0

    # Other runtime options
    tp_size: int = 1
    stream_interval: int = 1
    random_seed: Optional[int] = None
    constrained_json_whitespace_pattern: Optional[str] = None

    # Logging
    log_level: str = "info"
    log_level_http: Optional[str] = None
    log_requests: bool = False
    show_time_cost: bool = False
    log_file: Optional[str] = None

    # Other
    api_key: Optional[str] = None
    file_storage_pth: str = "SGLang_storage"
    enable_cache_report: bool = False

    # Data parallelism
    dp_size: int = 1
    load_balance_method: str = "round_robin"

    # Distributed args
    dist_init_addr: Optional[str] = None
    nnodes: int = 1
    node_rank: int = 0

    # Model override args
    json_model_override_args: str = "{}"

    # Redis args for multi-model serving
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    enable_controller: bool = False
    enable_gpu_scheduler: bool = False
    policy: str = "simple-global"
    queue_id: str = field(default_factory=lambda: str(uuid.uuid4().hex))

    # Async loading
    async_loading: bool = False

    # Prism-specific options
    enable_elastic_memory: bool = False
    use_kvcached_v0: bool = True
    enable_cpu_share_memory: bool = False
    enable_model_service: bool = False
    num_model_service_workers: int = 1
    abort_exceed_slos: bool = False

    # Optimization options (inherited from sglang)
    attention_backend: Optional[str] = None
    sampling_backend: Optional[str] = None
    disable_flashinfer: bool = False
    disable_flashinfer_sampling: bool = False
    disable_radix_cache: bool = False
    disable_regex_jump_forward: bool = False
    disable_cuda_graph: bool = False
    disable_cuda_graph_padding: bool = False
    disable_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    disable_mla: bool = False
    disable_penalizer: bool = False
    disable_nan_detection: bool = False
    enable_overlap_schedule: bool = False
    enable_mixed_chunk: bool = False
    enable_torch_compile: bool = False
    max_torch_compile_bs: int = 32
    torchao_config: str = ""
    enable_p2p_check: bool = False
    triton_attention_reduce_in_fp32: bool = False
    num_continuous_decode_steps: int = 1

    # LoRA
    lora_paths: Optional[List[str]] = None
    max_loras_per_batch: int = 8

    # Double Sparsity
    enable_double_sparsity: bool = False
    ds_channel_config_path: Optional[str] = None
    ds_heavy_channel_num: int = 32
    ds_heavy_token_num: int = 256
    ds_heavy_channel_type: str = "qk"
    ds_sparse_decode_threshold: int = 4096

    def __post_init__(self):
        """Initialize derived values and validate configuration."""
        if self.model_path is not None:
            if self.tokenizer_path is None:
                self.tokenizer_path = self.model_path
            if self.served_model_name is None:
                self.served_model_name = self.model_path
            if self.model_name is None:
                self.model_name = self.model_path

        # Mem fraction depends on tensor parallelism size
        if self.mem_fraction_static is None:
            if self.tp_size >= 16:
                self.mem_fraction_static = 0.79
            elif self.tp_size >= 8:
                self.mem_fraction_static = 0.83
            elif self.tp_size >= 4:
                self.mem_fraction_static = 0.85
            elif self.tp_size >= 2:
                self.mem_fraction_static = 0.87
            else:
                self.mem_fraction_static = 0.88

        # Load model configs
        if self.model_configs is None:
            if self.model_config_file:
                self.model_configs = load_model_configs(self.model_config_file)
            elif self.model_path:
                model_config = ModelConfig(
                    model_name=self.model_name,
                    model_path=self.model_path,
                    tokenizer_path=self.tokenizer_path,
                    init_placements=[{"gpu_ids": [0], "on": True}],
                )
                self.model_configs = [model_config]
            else:
                raise ValueError(
                    "model_config_file or model_path is required "
                    "when model_configs is not provided"
                )

        if self.chunked_prefill_size <= 0:
            self.chunked_prefill_size = None

        if self.random_seed is None:
            self.random_seed = random.randint(0, 1 << 30)

        # Handle deprecations
        if self.disable_flashinfer:
            logger.warning(
                "The option '--disable-flashinfer' is deprecated. "
                "Please use '--attention-backend triton' instead."
            )
            self.attention_backend = "triton"
            
        if self.disable_flashinfer_sampling:
            logger.warning(
                "The option '--disable-flashinfer-sampling' is deprecated. "
                "Please use '--sampling-backend pytorch' instead."
            )
            self.sampling_backend = "pytorch"

        if not is_flashinfer_available():
            self.attention_backend = "triton"
            self.sampling_backend = "pytorch"

        # Default backends
        if self.attention_backend is None:
            self.attention_backend = "flashinfer"
        if self.sampling_backend is None:
            self.sampling_backend = "flashinfer"

        # Setup Redis queue keys
        if self.queue_id is None:
            self.queue_id = str(uuid.uuid4().hex)
            
        if self.enable_gpu_scheduler:
            self.frontend_generate_request_key_prefix = (
                f"frontend_generate_request_{self.queue_id}"
            )
            self.backend_generate_request_key_prefix = (
                f"backend_generate_request_{self.queue_id}"
            )
            self.engine_to_gpu_scheduler_key_prefix = (
                f"engine_to_gpu_scheduler_{self.queue_id}"
            )
        else:
            self.frontend_generate_request_key_prefix = (
                f"generate_request_{self.queue_id}"
            )
            self.backend_generate_request_key_prefix = (
                f"generate_request_{self.queue_id}"
            )
            self.engine_to_gpu_scheduler_key_prefix = (
                f"engine_to_gpu_scheduler_{self.queue_id}"
            )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        """Add CLI arguments for multi-model server."""
        # Core model args
        parser.add_argument("--model-path", type=str, help="Path to model weights")
        parser.add_argument("--model-name", type=str, help="Unique model identifier")
        parser.add_argument("--tokenizer-path", type=str, help="Path to tokenizer")
        parser.add_argument("--model-config-file", type=str, help="Path to model config JSON")
        parser.add_argument("--load-format", type=str, default="auto", help="Model loading format")
        
        # Server args
        parser.add_argument("--host", type=str, default="127.0.0.1")
        parser.add_argument("--port", type=int, default=30000)
        
        # Worker pool args
        parser.add_argument("--enable-worker-pool", action="store_true")
        parser.add_argument("--workers-per-gpu", type=int, default=1)
        parser.add_argument("--num-gpus", type=int, default=1)
        
        # Memory args
        parser.add_argument("--mem-fraction-static", type=float)
        parser.add_argument("--max-memory-pool-size", type=float)
        parser.add_argument("--max-mem-usage", type=float, help="Max GPU memory usage in GB")
        parser.add_argument("--enable-elastic-memory", action="store_true")
        parser.add_argument("--use-kvcached-v0", action="store_true")
        parser.add_argument("--enable-cpu-share-memory", action="store_true")
        
        # Redis args
        parser.add_argument("--redis-host", type=str, default="localhost")
        parser.add_argument("--redis-port", type=int, default=6379)
        parser.add_argument("--redis-db", type=int, default=0)
        parser.add_argument("--enable-controller", action="store_true")
        parser.add_argument("--enable-gpu-scheduler", action="store_true")
        parser.add_argument("--policy", type=str, default="simple-global")
        
        # Model service args
        parser.add_argument("--enable-model-service", action="store_true")
        parser.add_argument("--num-model-service-workers", type=int, default=1)
        
        # Parallelism
        parser.add_argument("--tp-size", "--tensor-parallel-size", type=int, default=1)
        parser.add_argument("--dp-size", "--data-parallel-size", type=int, default=1)
        
        # Logging
        parser.add_argument("--log-level", type=str, default="info")
        parser.add_argument("--log-file", type=str, help="Path to log file")
        
        # Optimization options
        parser.add_argument("--trust-remote-code", action="store_true")
        parser.add_argument("--dtype", type=str, default="auto")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--disable-cuda-graph", action="store_true")
        parser.add_argument("--disable-radix-cache", action="store_true")
        parser.add_argument("--attention-backend", type=str, choices=["flashinfer", "triton"])
        parser.add_argument("--sampling-backend", type=str, choices=["flashinfer", "pytorch"])
        
        # Async loading
        parser.add_argument("--async-loading", action="store_true")

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        """Create instance from CLI arguments."""
        if hasattr(args, 'tensor_parallel_size'):
            args.tp_size = args.tensor_parallel_size
        if hasattr(args, 'data_parallel_size'):
            args.dp_size = args.data_parallel_size
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr, None) for attr in attrs if hasattr(args, attr)})

    def url(self) -> str:
        """Get server URL."""
        if is_ipv6(self.host):
            return f"http://[{self.host}]:{self.port}"
        return f"http://{self.host}:{self.port}"

    def check_server_args(self):
        """Validate server arguments."""
        assert self.tp_size % self.nnodes == 0, \
            "tp_size must be divisible by number of nodes"
        assert not (self.dp_size > 1 and self.nnodes != 1), \
            "multi-node data parallel is not supported"


def prepare_server_args(argv: List[str]) -> MultiModelServerArgs:
    """Parse CLI arguments and create server args."""
    parser = argparse.ArgumentParser(description="Prism Multi-Model Server")
    MultiModelServerArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    return MultiModelServerArgs.from_cli_args(raw_args)


__all__ = [
    "Placement",
    "ModelConfig",
    "InstanceConfig",
    "MultiModelServerArgs",
    "load_model_configs",
    "prepare_server_args",
]
