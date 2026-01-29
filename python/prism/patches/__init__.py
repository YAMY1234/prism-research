# Copyright 2024 Prism Team
# Licensed under the Apache License, Version 2.0

"""
Prism patches for SGLang.

This module provides minimal-invasive patches to extend SGLang with:
- Model activation/deactivation support
- Elastic memory management via kvcached
- Multi-model scheduling capabilities

Patches are applied in a specific order to ensure compatibility.
"""

import logging

logger = logging.getLogger(__name__)

_patches_applied = False


def apply_all_patches():
    """
    Apply all Prism patches to SGLang.
    
    This function is idempotent - calling it multiple times has no effect.
    Patches are applied in the following order:
    1. scheduler patches (request handling extension)
    2. server_args patches (multi-model configuration support)
    3. memory pool patches (elastic memory support - optional)
    """
    global _patches_applied
    
    if _patches_applied:
        logger.debug("Prism patches already applied, skipping.")
        return
    
    logger.info("Applying Prism patches to SGLang...")
    
    try:
        # Phase 1: Apply scheduler patches (extends TypeBasedDispatcher)
        from prism.patches.scheduler_patch import apply_scheduler_patch
        apply_scheduler_patch()
        logger.debug("Scheduler patch applied.")
        
        # Phase 2: Apply server_args patches (adds from_multi_model_server_args)
        from prism.patches.server_args_patch import apply_server_args_patch
        apply_server_args_patch()
        logger.debug("Server args patch applied.")
        
        # Phase 3: Apply TpModelWorker patches (activate/deactivate support)
        from prism.patches.tp_worker_patch import apply_tp_worker_patch
        apply_tp_worker_patch()
        logger.debug("TpModelWorker patch applied.")
        
        # Phase 4: Apply ModelRunner patches (elastic memory via kvcached)
        from prism.patches.model_runner_patch import apply_model_runner_patch
        apply_model_runner_patch()
        logger.debug("ModelRunner patch applied.")
        
        # Phase 5: Memory pool patches are optional (used via inheritance)
        # MHATokenToKVPoolElastic is available as a new class
        
        _patches_applied = True
        logger.info("Prism patches applied successfully!")
        
    except Exception as e:
        logger.error(f"Failed to apply Prism patches: {e}")
        raise


def is_patches_applied() -> bool:
    """Check if patches have been applied."""
    return _patches_applied


# Auto-apply patches on import
# This ensures patches are applied when any module imports prism.patches
apply_all_patches()
