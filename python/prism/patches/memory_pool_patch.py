# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Prism elastic memory pool implementation using kvcached.

This provides MHATokenToKVPoolElastic, which uses kvcached for dynamic
memory allocation, enabling flexible GPU sharing between multiple models.

Note: This is an independent implementation that works alongside SGLang's
memory pool, not a modification of it.
"""

import logging
import time
from typing import List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

# Check interval for physical memory
PHYSICAL_MEM_CHECK_FREQ = 0.01


class MHATokenToKVPoolElastic:
    """
    Elastic KV cache memory pool using kvcached.
    
    This class provides dynamic memory allocation for KV cache, enabling:
    - On-demand memory allocation
    - Memory sharing between multiple models
    - Fast model activation/deactivation
    
    Note: This is independent of SGLang's MHATokenToKVPool and should be used
    when enable_elastic_memory=True in Prism's multi-model serving.
    """

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        gpu_id: int,
        model_name: str = "default",
        use_kvcached_v0: bool = True,
        enable_worker_pool: bool = False,
        min_reserve_mem: float = 0.0,
        enable_overlap: bool = False,
        shm=None,
        ipc_name: Optional[str] = None,
    ):
        """
        Initialize elastic KV cache pool.
        
        Args:
            size: Maximum number of tokens in the pool
            dtype: Data type for KV cache
            head_num: Number of attention heads
            head_dim: Dimension per head
            layer_num: Number of transformer layers
            device: Device type (e.g., "cuda")
            gpu_id: GPU device ID
            model_name: Name of the model (for logging)
            use_kvcached_v0: Use kvcached v0 API
            enable_worker_pool: Whether running in worker pool mode
            min_reserve_mem: Minimum memory to reserve (GB)
            enable_overlap: Enable overlapped operations
            shm: Shared memory handle (for worker pool)
            ipc_name: IPC name for communication
        """
        self.size = size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.device = device
        self.gpu_id = gpu_id
        self.model_name = model_name
        self.use_kvcached_v0 = use_kvcached_v0
        self.enable_worker_pool = enable_worker_pool
        self.min_reserve_mem = min_reserve_mem
        self.enable_overlap = enable_overlap
        self.shm = shm
        self.ipc_name = ipc_name or f"prism_kv_{gpu_id}_{model_name}"
        
        # Store dtype handling
        if dtype == torch.float8_e5m2:
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        
        # State tracking
        self.last_available_size = None
        self.k_buffer = None
        self.v_buffer = None
        self.kv_allocator = None
        self.kvcached_ops = None
        
        # Free slots tracking (for non-elastic fallback)
        self.free_slots = None
        
        # Initialize kvcached
        self._init_kvcached()
    
    def _init_kvcached(self):
        """Initialize kvcached backend."""
        try:
            from kvcached import ops as kvcached_ops
            from kvcached.slab_allocator import KVCacheManager
            
            self.kvcached_ops = kvcached_ops
            self.KVCacheManager = KVCacheManager
            
            # Initialize kvcached runtime (only if not in worker pool mode)
            if not self.enable_worker_pool:
                self.kvcached_ops.init_kvcached(self.gpu_id)
            
            # Allocate KV cache buffers
            self._init_kv_allocator()
            
            logger.info(
                f"MHATokenToKVPoolElastic initialized for {self.model_name}: "
                f"size={self.size}, layers={self.layer_num}, "
                f"heads={self.head_num}, head_dim={self.head_dim}"
            )
            
        except ImportError as e:
            raise ImportError(
                "kvcached package is required for elastic memory. "
                "Please install it with: pip install kvcached"
            ) from e
    
    def _init_kv_allocator(self):
        """Initialize KV cache allocator."""
        # Allocate KV cache buffers via kvcached
        k_buffer, v_buffer = self.kvcached_ops.sgl_alloc_kv_cache(
            self.size,
            self.head_num,
            self.head_dim,
            self.dtype,
            f"{self.device}:{self.gpu_id}",
            self.layer_num,
        )
        self.k_buffer = k_buffer
        self.v_buffer = v_buffer
        
        # Calculate cell size for allocator
        self.cell_size = self.head_num * self.head_dim * self.dtype.itemsize
        
        # Initialize allocator
        if self.use_kvcached_v0:
            self.kv_allocator = self.KVCacheManager(
                self.size,
                1,
                self.cell_size,
                num_layers=self.layer_num,
                shm=self.shm,
            )
            logger.debug("Elastic memory: kv_cache_manager_v0 initialized")
        else:
            self.kv_allocator = self.KVCacheManager(
                self.size,
                1,
                self.cell_size,
                num_layers=self.layer_num,
                enable_overlap=self.enable_overlap,
                ipc_name=self.ipc_name,
            )
            logger.debug("Elastic memory: kv_cache_manager initialized")
    
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """
        Allocate KV cache slots.
        
        Args:
            need_size: Number of slots to allocate
            
        Returns:
            Tensor of allocated indices, or None if allocation fails
        """
        indices = self.kv_allocator.alloc(need_size)
        
        if indices is None:
            return None
        
        if self.use_kvcached_v0:
            if isinstance(indices, list):
                indices = torch.tensor(
                    indices,
                    dtype=torch.int32,
                    device=f"{self.device}:{self.gpu_id}",
                )
            elif indices is not None:
                indices = indices.to(
                    f"{self.device}:{self.gpu_id}", non_blocking=True
                )
        else:
            indices = indices.to(
                f"{self.device}:{self.gpu_id}", non_blocking=True
            )
        
        return indices
    
    def free(self, free_index: torch.Tensor):
        """
        Free KV cache slots.
        
        Args:
            free_index: Tensor of indices to free
        """
        if self.use_kvcached_v0:
            self.kv_allocator.free(free_index.cpu().numpy())
        else:
            self.kv_allocator.free(free_index.cpu())
    
    def available_size(self) -> int:
        """
        Get available slots in the pool.
        
        Returns:
            Number of available slots
        """
        if self.use_kvcached_v0:
            return self.kv_allocator.available_size()
        
        # Check physical memory periodically
        if self.last_available_size is None:
            free_size = self._physical_free_size(0.5)
            check_time = time.perf_counter()
            self.last_available_size = (free_size, check_time)
        elif (
            time.perf_counter() - self.last_available_size[1]
            > PHYSICAL_MEM_CHECK_FREQ
        ):
            free_size = self._physical_free_size(0.5)
            self.last_available_size = (free_size, time.perf_counter())
        else:
            free_size, _ = self.last_available_size
        
        return min(free_size, self.kv_allocator.available_size())
    
    def _physical_free_size(self, min_reserve_mem_gb: float) -> int:
        """Calculate available physical memory in terms of KV cache slots."""
        avail_phy_mem_size, _ = torch.cuda.mem_get_info()
        avail_phy_mem_size -= int(min_reserve_mem_gb * (1 << 30))
        
        from kvcached.slab_allocator import PAGE_SIZE
        
        avail_phy_pages = avail_phy_mem_size // PAGE_SIZE
        # Each layer needs K and V tensors
        avail_phy_blocks = (avail_phy_pages // self.layer_num // 2) * (
            PAGE_SIZE // self.kv_allocator.block_mem_size
        )
        return avail_phy_blocks
    
    def update_size(self, new_size: int) -> bool:
        """
        Resize the memory pool.
        
        Args:
            new_size: New size in number of slots
            
        Returns:
            True if resize succeeded
        """
        return self.kv_allocator.resize(new_size)
    
    def try_to_reserve(self, need_size: int) -> bool:
        """
        Try to reserve memory for future allocation.
        
        Args:
            need_size: Number of slots to reserve
            
        Returns:
            True if reservation succeeded
        """
        return self.kv_allocator.try_to_reserve(need_size)
    
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        """Get key buffer for a specific layer."""
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id].view(self.dtype)
        return self.k_buffer[layer_id]
    
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        """Get value buffer for a specific layer."""
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id].view(self.dtype)
        return self.v_buffer[layer_id]
    
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get both key and value buffers for a layer."""
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)
    
    def set_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        """
        Set KV cache values at specified locations.
        
        Args:
            layer_id: Layer index
            loc: Location indices
            cache_k: Key cache values
            cache_v: Value cache values
        """
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        
        if self.store_dtype != self.dtype:
            self.k_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
            self.v_buffer[layer_id][loc] = cache_v.view(self.store_dtype)
        else:
            self.k_buffer[layer_id][loc] = cache_k
            self.v_buffer[layer_id][loc] = cache_v
    
    def release(self):
        """Release memory (for deactivation)."""
        if self.use_kvcached_v0:
            self.kv_allocator.trim()
            if self.enable_worker_pool:
                del self.kv_allocator
                self.kvcached_ops.free_kv_cached_tensors()
        else:
            self.kv_allocator.clear()
    
    def shutdown(self):
        """Shutdown kvcached and release all resources."""
        if self.kvcached_ops is not None:
            self.kvcached_ops.shutdown_kvcached()
        if self.kv_allocator is not None:
            del self.kv_allocator
            self.kv_allocator = None
        self.k_buffer = None
        self.v_buffer = None
    
    def clear(self):
        """Clear all allocations."""
        if self.kv_allocator is not None:
            self.kv_allocator.clear()
        self.last_available_size = None


# Export
__all__ = ["MHATokenToKVPoolElastic"]
