#!/usr/bin/env python3
"""
Direct test script for Prism patch mechanism.
Run with: python run_test.py
"""

import sys
import os

# CRITICAL: Set up paths BEFORE any imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_sglang_main = os.path.join(os.path.dirname(os.path.dirname(_script_dir)), "python")

# Main sglang MUST be first
sys.path.insert(0, _sglang_main)
sys.path.insert(1, _script_dir)

print(f"Python path setup:")
print(f"  1. Main sglang: {_sglang_main}")
print(f"  2. Prism: {_script_dir}")
print()

# Disable auto-patch for controlled testing
os.environ["PRISM_NO_AUTO_PATCH"] = "1"


def test_type_based_dispatcher():
    """Test TypeBasedDispatcher directly"""
    print("=" * 60)
    print("Test: TypeBasedDispatcher and Patch Mechanism")
    print("=" * 60)
    
    # Import TypeBasedDispatcher
    from sglang.utils import TypeBasedDispatcher
    print(f"✅ TypeBasedDispatcher imported from: {TypeBasedDispatcher.__module__}")
    
    # Create dispatcher
    dispatcher = TypeBasedDispatcher([])
    print(f"✅ Created dispatcher with _mapping: {type(dispatcher._mapping).__name__}")
    
    # Import Prism data structures
    from prism.io_struct import ActivateReqInput, DeactivateReqInput
    print(f"✅ Imported Prism data structures")
    
    # Simulate patch: add handlers
    call_log = []
    
    def handle_activate(req):
        call_log.append(('activate', req.model_name, req.gpu_id))
        return f"Activated {req.model_name} on GPU {req.gpu_id}"
    
    def handle_deactivate(req):
        call_log.append(('deactivate', req.model_name))
        return f"Deactivated {req.model_name}"
    
    # This is the key mechanism our patch uses!
    dispatcher._mapping[ActivateReqInput] = handle_activate
    dispatcher._mapping[DeactivateReqInput] = handle_deactivate
    print(f"✅ Registered handlers to dispatcher._mapping")
    print(f"   Registered types: {[t.__name__ for t in dispatcher._mapping.keys()]}")
    
    # Test activate
    req1 = ActivateReqInput(model_name="llama-7b", gpu_id=0)
    result1 = dispatcher(req1)
    print(f"✅ Activate: {result1}")
    
    # Test deactivate
    req2 = DeactivateReqInput(model_name="llama-7b")
    result2 = dispatcher(req2)
    print(f"✅ Deactivate: {result2}")
    
    # Verify
    expected = [('activate', 'llama-7b', 0), ('deactivate', 'llama-7b')]
    assert call_log == expected, f"Expected {expected}, got {call_log}"
    print(f"✅ Call sequence verified: {call_log}")
    
    return True


def test_prism_imports():
    """Test Prism package imports"""
    print("\n" + "=" * 60)
    print("Test: Prism Package Imports")
    print("=" * 60)
    
    from prism.io_struct import (
        ActivateReqInput, ActivateReqOutput,
        DeactivateReqInput, DeactivateReqOutput,
        MemoryUsage, PreemptMode,
    )
    print("✅ io_struct imports OK")
    
    from prism.multi_model import MultiModelServerArgs, ModelConfig
    print("✅ multi_model imports OK")
    
    from prism.patches.memory_pool_patch import MHATokenToKVPoolElastic
    print("✅ MHATokenToKVPoolElastic import OK")
    
    from prism.utils.redis_utils import RedisClient, AsyncRedisClient
    print("✅ redis_utils imports OK")
    
    # Test creating objects
    req = ActivateReqInput(model_name="test", gpu_id=0)
    print(f"✅ Created ActivateReqInput: rid={req.rid[:8]}...")
    
    args = MultiModelServerArgs(model_path="test/model")
    print(f"✅ Created MultiModelServerArgs: model_name={args.model_name}")
    
    return True


def main():
    print("\n" + "=" * 60)
    print("  PRISM PATCH MECHANISM TEST")
    print("=" * 60 + "\n")
    
    results = []
    
    try:
        results.append(("TypeBasedDispatcher", test_type_based_dispatcher()))
    except Exception as e:
        print(f"❌ TypeBasedDispatcher test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("TypeBasedDispatcher", False))
    
    try:
        results.append(("Prism imports", test_prism_imports()))
    except Exception as e:
        print(f"❌ Prism imports test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Prism imports", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:30s} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Patch mechanism is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
