# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Prism-specific data structures for multi-model serving.

These are NEW data structures that extend SGLang's io_struct.
They are used for model activation/deactivation and multi-model scheduling.
"""

import dataclasses
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

from sglang.srt.managers.io_struct import GenerateReqInput as SGLangGenerateReqInput
from sglang.srt.managers.io_struct import EmbeddingReqInput as SGLangEmbeddingReqInput


@dataclass
class GenerateReqInput(SGLangGenerateReqInput):
    """Extended GenerateReqInput with Prism multi-model support."""
    
    # Model name for multi-model routing (required for Prism)
    model: Optional[str] = None
    # SLO deadline for scheduling (seconds from arrival_time)
    slo: Optional[float] = None
    # Whether this is a warmup request
    is_warmup: bool = False
    # Request arrival time for scheduling
    arrival_time: Optional[float] = None
    
    def get_prompt_len(self) -> int:
        """Estimate prompt length from text or input_ids."""
        if self.input_ids is not None:
            if isinstance(self.input_ids, list):
                if len(self.input_ids) > 0 and isinstance(self.input_ids[0], int):
                    return len(self.input_ids)
                elif len(self.input_ids) > 0:
                    return len(self.input_ids[0])
            return 0
        elif self.text is not None:
            if isinstance(self.text, str):
                # Rough estimate: ~4 chars per token
                return len(self.text) // 4
            elif isinstance(self.text, list) and len(self.text) > 0:
                return len(self.text[0]) // 4
        return 0


@dataclass  
class EmbeddingReqInput(SGLangEmbeddingReqInput):
    """Extended EmbeddingReqInput with Prism multi-model support."""
    
    # Model name for multi-model routing
    model: Optional[str] = None
    # SLO deadline for scheduling
    slo: Optional[float] = None
    # Whether this is a warmup request
    is_warmup: bool = False


@dataclass
class MemoryUsage:
    """Memory usage information in GB."""
    
    total_used_memory: float  # Total used memory (higher than sum due to allocator cache)
    model_weights_memory: float
    memory_pool_memory: float
    req_to_token_pool_memory: float
    token_to_kv_pool_memory: float

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "MemoryUsage":
        return MemoryUsage(**d)


class PreemptMode(Enum):
    """Mode for preempting ongoing requests during deactivation."""
    RECOMPUTE = 1  # Recompute from scratch
    SWAP = 2       # Swap to CPU (not implemented)
    RETURN = 3     # Return partial results
    ABORT = 4      # Abort requests


@dataclass
class ActivateReqInput:
    """Request to activate a model on a GPU."""
    
    model_name: str
    gpu_id: int  # For TP case, it's gpu_id for rank0
    instance_idx: Optional[int] = 0
    rid: Optional[str] = None
    memory_pool_size: Optional[float] = None  # in GB

    def __post_init__(self):
        if self.rid is None:
            self.rid = uuid.uuid4().hex


@dataclass
class ActivateReqOutput:
    """Response from model activation."""
    
    rid: str
    success: bool
    memory_usage: MemoryUsage  # in GB
    model_name: str
    instance_idx: Optional[int] = 0
    gpu_id: Optional[int] = None


@dataclass
class DeactivateReqInput:
    """Request to deactivate a model."""
    
    model_name: str
    instance_idx: Optional[int] = 0
    evict_waiting_requests: Optional[bool] = True
    preempt: bool = False  # Whether to preempt ongoing requests
    preempt_mode: str = PreemptMode.RETURN.name
    gpu_id: Optional[int] = None
    rid: Optional[str] = None

    def __post_init__(self):
        if self.rid is None:
            self.rid = uuid.uuid4().hex
        if isinstance(self.preempt_mode, str):
            if self.preempt_mode not in PreemptMode.__members__:
                raise ValueError(
                    f"Invalid preempt mode: {self.preempt_mode}. "
                    f"Should be one of {list(PreemptMode.__members__.keys())}"
                )
            self.preempt_mode = PreemptMode[self.preempt_mode]


@dataclass
class DeactivateReqOutput:
    """Response from model deactivation."""
    
    rid: str
    success: bool
    memory_usage: MemoryUsage
    model_name: str
    instance_idx: Optional[int] = 0
    gpu_id: Optional[int] = None


@dataclass
class GetMemPoolSizeReq:
    """Request to get memory pool size."""
    
    model_name: str
    instance_idx: Optional[int] = 0
    rid: Optional[str] = None

    def __post_init__(self):
        if self.rid is None:
            self.rid = uuid.uuid4().hex


@dataclass
class GetMemPoolSizeReqOutput:
    """Response with memory pool size."""
    
    size: int
    rid: str


@dataclass
class GetMemoryUsageReq:
    """Request to get memory usage."""
    
    model_name: str
    instance_idx: Optional[int] = 0
    rid: Optional[str] = None

    def __post_init__(self):
        if self.rid is None:
            self.rid = uuid.uuid4().hex


@dataclass
class GetMemoryUsageReqOutput:
    """Response with memory usage."""
    
    rid: str
    memory_usage: MemoryUsage


@dataclass
class ResizeMemPoolReqInput:
    """Request to resize memory pool."""
    
    model_name: str
    memory_pool_size: Optional[float] = None  # in GB
    instance_idx: Optional[int] = 0


@dataclass
class FinishReq:
    """Notification that a request has finished."""
    
    rid: str
    model: str
    finish_time: float
    is_warmup: bool = False
    gpu_ids: Optional[List[int]] = None


@dataclass
class BatchRetractDecodeReq:
    """Request to retract decode batch (for preemption)."""
    
    rids: List[str]
    len_output_ids: List[int]
    model: str
    retract_time: float


@dataclass
class BatchRunReq:
    """Notification that a batch is running."""
    
    rids: List[str]
    model: str
    run_time: float
    gpu_id: Optional[int] = None


@dataclass
class UpdateModelTput:
    """Update model throughput statistics."""
    
    model_name: str
    instance_idx: int = 0
    latest_token_tput: float = 0.0
    prefill_token_tput: float = 0.0
    decode_token_tput: float = 0.0
    token_count: int = 0
    prefill_token_count: int = 0
    decode_token_count: int = 0


# Export all
__all__ = [
    "GenerateReqInput",
    "EmbeddingReqInput",
    "MemoryUsage",
    "PreemptMode",
    "ActivateReqInput",
    "ActivateReqOutput",
    "DeactivateReqInput",
    "DeactivateReqOutput",
    "GetMemPoolSizeReq",
    "GetMemPoolSizeReqOutput",
    "GetMemoryUsageReq",
    "GetMemoryUsageReqOutput",
    "ResizeMemPoolReqInput",
    "FinishReq",
    "BatchRetractDecodeReq",
    "BatchRunReq",
    "UpdateModelTput",
]
