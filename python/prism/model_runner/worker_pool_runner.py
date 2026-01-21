# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Prism Worker Pool Model Runner.

This module provides PrismWorkerPoolModelRunner for multi-model serving with:
- CPU model preloading for fast activation
- On-demand GPU model activation/deactivation
- Elastic memory management via kvcached
- Support for worker pool mode with multiple workers per GPU
"""

import gc
import io
import logging
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PrismWorkerPoolModelRunner:
    """
    Model runner optimized for multi-model serving in worker pool mode.
    
    This class supports:
    - Preloading multiple model configs
    - Fast model switching via CPU->GPU transfer
    - Elastic memory via kvcached
    - Worker pool coordination
    
    Unlike standard ModelRunner, this supports activate/deactivate lifecycle.
    """
    
    def __init__(
        self,
        model_configs: Dict[str, Any],
        shared_cpu_models: Dict[Tuple[str, int], List[nn.Module]],
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        worker_id: int,
        enable_elastic_memory: bool = True,
        use_kvcached_v0: bool = True,
        virtual_memory_size_gb: float = 25.0,
        min_reserve_mem: float = 1.0,
    ):
        """
        Initialize the worker pool model runner.
        
        Args:
            model_configs: Dict mapping model_name -> ModelConfig
            shared_cpu_models: Dict mapping (model_path, tp_size) -> list of CPU models per rank
            gpu_id: GPU device ID
            tp_rank: Tensor parallel rank
            tp_size: Tensor parallel size
            worker_id: Worker ID within the pool
            enable_elastic_memory: Use kvcached for elastic memory
            use_kvcached_v0: Use kvcached v0 API
            virtual_memory_size_gb: Virtual memory size for kvcached
            min_reserve_mem: Minimum memory to reserve (GB)
        """
        self.model_configs = model_configs
        self.shared_cpu_models = shared_cpu_models
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.worker_id = worker_id
        self.enable_elastic_memory = enable_elastic_memory
        self.use_kvcached_v0 = use_kvcached_v0
        self.virtual_memory_size_gb = virtual_memory_size_gb
        self.min_reserve_mem = min_reserve_mem
        
        # Current state
        self.is_active = False
        self.current_model_name = None
        self.model = None
        self.model_config = None
        self.token_to_kv_pool = None
        self.req_to_token_pool = None
        
        # Device
        self.device = "cuda"
        
        # Initialize kvcached if enabled
        if self.enable_elastic_memory:
            self._init_kvcached()
        
        logger.info(
            f"PrismWorkerPoolModelRunner initialized: "
            f"gpu_id={gpu_id}, worker_id={worker_id}, "
            f"elastic_memory={enable_elastic_memory}"
        )
    
    def _init_kvcached(self):
        """Initialize kvcached runtime."""
        try:
            from kvcached import ops as kvcached_ops
            
            kvcached_ops.init_kvcached(
                virtual_mem_size_gb=self.virtual_memory_size_gb,
                reserve_virtual_mem=True,
            )
            self.kvcached_ops = kvcached_ops
            logger.info(f"kvcached initialized with {self.virtual_memory_size_gb}GB virtual memory")
        except ImportError as e:
            raise ImportError(
                "kvcached is required for elastic memory. "
                "Please install it with: pip install kvcached"
            ) from e
    
    def activate(
        self,
        model_name: str,
        memory_pool_size: Optional[float] = None,
        gpu_id: Optional[int] = None,
    ):
        """
        Activate a model on the GPU.
        
        Args:
            model_name: Name of the model to activate
            memory_pool_size: Size of memory pool (GB), or None for auto
            gpu_id: GPU ID (should match self.gpu_id)
        """
        if self.is_active:
            raise RuntimeError(
                f"Model runner is already active with {self.current_model_name}. "
                "Deactivate first."
            )
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        if gpu_id is not None and gpu_id != self.gpu_id:
            raise ValueError(
                f"GPU ID mismatch: expected {self.gpu_id}, got {gpu_id}"
            )
        
        tic = time.perf_counter()
        logger.info(f"Activating model {model_name}...")
        
        # Set model config
        self.model_config = self.model_configs[model_name]
        self.current_model_name = model_name
        
        # Load model to GPU
        self._load_gpu_model(model_name)
        
        # Initialize KV cache pool
        self._init_token_to_kv_pool(memory_pool_size)
        
        self.is_active = True
        
        elapsed = time.perf_counter() - tic
        logger.info(f"Model {model_name} activated in {elapsed:.2f}s")
    
    def deactivate(self):
        """
        Deactivate the current model and release GPU memory.
        """
        if not self.is_active:
            logger.warning("Model runner is not active, nothing to deactivate")
            return
        
        tic = time.perf_counter()
        logger.info(f"Deactivating model {self.current_model_name}...")
        
        # Release KV cache
        if self.token_to_kv_pool is not None:
            self.token_to_kv_pool.release()
            self.token_to_kv_pool = None
        
        # Release model weights
        if self.model is not None:
            del self.model
            self.model = None
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()
        
        self.is_active = False
        model_name = self.current_model_name
        self.current_model_name = None
        self.model_config = None
        
        elapsed = time.perf_counter() - tic
        logger.info(f"Model {model_name} deactivated in {elapsed:.2f}s")
    
    def _load_gpu_model(self, model_name: str):
        """Load model weights from CPU to GPU."""
        model_path = self.model_config.path if hasattr(self.model_config, 'path') else str(self.model_config)
        model_key = (model_path, self.tp_size)
        
        # Check for preloaded CPU model
        if model_key in self.shared_cpu_models:
            cpu_model_ref = self.shared_cpu_models[model_key][self.tp_rank]
            self._load_from_cpu_model(cpu_model_ref)
        else:
            # Fall back to loading from disk
            logger.warning(
                f"No preloaded CPU model for {model_name}, "
                "loading from disk (slower)"
            )
            self._load_from_disk(model_name)
    
    def _load_from_cpu_model(self, cpu_model_ref: nn.Module):
        """Load model by copying from preloaded CPU model."""
        tic = time.perf_counter()
        
        # Serialize and deserialize to create a new model instance
        # This is faster than deepcopy for large models
        buf = io.BytesIO()
        pickle.dump(cpu_model_ref, buf, protocol=pickle.HIGHEST_PROTOCOL)
        buf.seek(0)
        self.model = pickle.loads(buf.getvalue())
        
        # Transfer state dict to GPU
        try:
            from tensordict import TensorDict
            
            state_dict_host = TensorDict(self.model.state_dict())
            state_dict_device = state_dict_host.to(
                f"{self.device}:{self.gpu_id}",
                non_blocking=True,
            )
            self.model.load_state_dict(state_dict_device, assign=True)
        except ImportError:
            # Fallback without TensorDict
            state_dict = self.model.state_dict()
            for key in state_dict:
                state_dict[key] = state_dict[key].to(
                    f"{self.device}:{self.gpu_id}",
                    non_blocking=True,
                )
            self.model.load_state_dict(state_dict, assign=True)
        
        elapsed = time.perf_counter() - tic
        logger.info(f"Model loaded from CPU in {elapsed:.2f}s")
    
    def _load_from_disk(self, model_name: str):
        """Load model from disk (fallback)."""
        raise NotImplementedError(
            "Direct disk loading not implemented. "
            "Please preload models to CPU first."
        )
    
    def _init_token_to_kv_pool(self, memory_pool_size: Optional[float] = None):
        """Initialize KV cache memory pool."""
        if self.enable_elastic_memory:
            from prism.patches.memory_pool_patch import MHATokenToKVPoolElastic
            
            # Calculate max tokens based on memory
            if memory_pool_size is None:
                # Auto-calculate based on available memory
                available = torch.cuda.get_device_properties(self.gpu_id).total_memory
                available_gb = available / (1024**3) - self.min_reserve_mem
                # Rough estimation
                max_tokens = int(available_gb * 1024 * 1024 / self._get_kv_cache_cell_size())
            else:
                max_tokens = int(memory_pool_size * 1024 * 1024 / self._get_kv_cache_cell_size())
            
            self.token_to_kv_pool = MHATokenToKVPoolElastic(
                size=max_tokens,
                dtype=self._get_kv_cache_dtype(),
                head_num=self._get_num_kv_heads(),
                head_dim=self._get_head_dim(),
                layer_num=self._get_num_layers(),
                device=self.device,
                gpu_id=self.gpu_id,
                model_name=self.current_model_name,
                use_kvcached_v0=self.use_kvcached_v0,
                enable_worker_pool=True,
            )
        else:
            raise NotImplementedError(
                "Non-elastic memory pool not supported in worker pool mode"
            )
    
    def _get_kv_cache_cell_size(self) -> int:
        """Get size of one KV cache cell in bytes."""
        dtype_size = torch.finfo(self._get_kv_cache_dtype()).bits // 8
        return (
            self._get_num_kv_heads()
            * self._get_head_dim()
            * self._get_num_layers()
            * 2  # K and V
            * dtype_size
        )
    
    def _get_kv_cache_dtype(self) -> torch.dtype:
        """Get KV cache data type."""
        if self.model_config is not None and hasattr(self.model_config, 'dtype'):
            return self.model_config.dtype
        return torch.float16
    
    def _get_num_kv_heads(self) -> int:
        """Get number of KV heads."""
        if self.model_config is not None:
            if hasattr(self.model_config, 'get_num_kv_heads'):
                return self.model_config.get_num_kv_heads(self.tp_size)
            if hasattr(self.model_config, 'num_key_value_heads'):
                return self.model_config.num_key_value_heads // self.tp_size
        return 32  # Default
    
    def _get_head_dim(self) -> int:
        """Get head dimension."""
        if self.model_config is not None and hasattr(self.model_config, 'head_dim'):
            return self.model_config.head_dim
        return 128  # Default
    
    def _get_num_layers(self) -> int:
        """Get number of transformer layers."""
        if self.model_config is not None:
            if hasattr(self.model_config, 'num_hidden_layers'):
                return self.model_config.num_hidden_layers
        return 32  # Default
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB."""
        if not self.is_active:
            return {
                "total_used_memory": 0.0,
                "model_weights_memory": 0.0,
                "kv_cache_memory": 0.0,
            }
        
        allocated = torch.cuda.memory_allocated(self.gpu_id) / (1024**3)
        
        return {
            "total_used_memory": allocated,
            "model_weights_memory": 0.0,  # Would need tracking
            "kv_cache_memory": 0.0,  # Would need tracking
        }


__all__ = ["PrismWorkerPoolModelRunner"]
