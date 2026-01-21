# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Prism Model Runner extensions.

Provides PrismWorkerPoolModelRunner for multi-model serving with:
- CPU model preloading
- On-demand activation/deactivation
- Elastic memory management via kvcached
"""

from prism.model_runner.worker_pool_runner import PrismWorkerPoolModelRunner

__all__ = ["PrismWorkerPoolModelRunner"]
