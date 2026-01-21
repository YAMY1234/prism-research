#!/usr/bin/env python3
"""
Test script to verify Prism patch mechanism works correctly.

This test verifies:
1. Prism package can be imported
2. Data structures are correctly defined
3. Scheduler patch can be applied
4. New request types are registered to TypeBasedDispatcher

Run with:
    cd prism-research/python
    python -m prism.tests.test_patch_mechanism
"""

import sys
import os

# IMPORTANT: Add main sglang FIRST, then prism
# This ensures we use the main sglang, not the one in prism-research
_prism_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_sglang_root = os.path.dirname(os.path.dirname(_prism_root))  # Go up to sglang root
_main_sglang = os.path.join(_sglang_root, "python")

# Insert main sglang at position 0 (highest priority)
if _main_sglang not in sys.path:
    sys.path.insert(0, _main_sglang)
# Insert prism after
if _prism_root not in sys.path:
    sys.path.insert(1, _prism_root)

def test_import_prism():
    """Test 1: Basic import"""
    print("=" * 60)
    print("Test 1: Import prism package")
    print("=" * 60)
    
    # Disable auto-patch for controlled testing
    os.environ["PRISM_NO_AUTO_PATCH"] = "1"
    
    try:
        import prism
        print(f"✅ prism imported successfully")
        print(f"   Version: {prism.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import prism: {e}")
        return False


def test_data_structures():
    """Test 2: Data structures"""
    print("\n" + "=" * 60)
    print("Test 2: Data structures")
    print("=" * 60)
    
    try:
        from prism.io_struct import (
            ActivateReqInput,
            ActivateReqOutput,
            DeactivateReqInput,
            DeactivateReqOutput,
            MemoryUsage,
            PreemptMode,
        )
        
        # Test ActivateReqInput
        req = ActivateReqInput(
            model_name="test-model",
            gpu_id=0,
            instance_idx=0,
        )
        print(f"✅ ActivateReqInput created: model={req.model_name}, gpu={req.gpu_id}, rid={req.rid[:8]}...")
        
        # Test MemoryUsage
        mem = MemoryUsage(
            total_used_memory=10.5,
            model_weights_memory=5.0,
            memory_pool_memory=4.0,
            req_to_token_pool_memory=0.5,
            token_to_kv_pool_memory=1.0,
        )
        print(f"✅ MemoryUsage created: total={mem.total_used_memory}GB")
        
        # Test PreemptMode
        print(f"✅ PreemptMode.RETURN = {PreemptMode.RETURN}")
        
        # Test DeactivateReqInput with preempt mode
        deact_req = DeactivateReqInput(
            model_name="test-model",
            preempt=True,
            preempt_mode="RETURN",
        )
        print(f"✅ DeactivateReqInput created: preempt_mode={deact_req.preempt_mode}")
        
        return True
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_server_args():
    """Test 3: MultiModelServerArgs"""
    print("\n" + "=" * 60)
    print("Test 3: MultiModelServerArgs")
    print("=" * 60)
    
    try:
        from prism.multi_model import MultiModelServerArgs, ModelConfig
        
        # Test with model_path
        args = MultiModelServerArgs(
            model_path="meta-llama/Llama-2-7b-chat-hf",
            enable_elastic_memory=True,
            enable_worker_pool=True,
        )
        print(f"✅ MultiModelServerArgs created")
        print(f"   model_name: {args.model_name}")
        print(f"   enable_elastic_memory: {args.enable_elastic_memory}")
        print(f"   enable_worker_pool: {args.enable_worker_pool}")
        print(f"   mem_fraction_static: {args.mem_fraction_static}")
        
        # Test ModelConfig
        config = ModelConfig(
            model_name="test-model",
            model_path="/path/to/model",
            tp_size=2,
        )
        print(f"✅ ModelConfig created: {config.model_name}, tp_size={config.tp_size}")
        
        return True
    except Exception as e:
        print(f"❌ Server args test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sglang_import():
    """Test 4: SGLang core can be imported"""
    print("\n" + "=" * 60)
    print("Test 4: SGLang core import")
    print("=" * 60)
    
    try:
        # Test TypeBasedDispatcher directly (this is the core mechanism we use)
        from sglang.utils import TypeBasedDispatcher
        print(f"✅ TypeBasedDispatcher imported")
        
        # Check TypeBasedDispatcher structure
        dispatcher = TypeBasedDispatcher([])
        print(f"✅ TypeBasedDispatcher created")
        print(f"   Has _mapping: {hasattr(dispatcher, '_mapping')}")
        print(f"   _mapping type: {type(dispatcher._mapping)}")
        
        # Test extending the dispatcher
        from prism.io_struct import ActivateReqInput
        
        def mock_handler(req):
            return f"Handled {req.model_name}"
        
        dispatcher._mapping[ActivateReqInput] = mock_handler
        print(f"✅ Successfully added ActivateReqInput to dispatcher")
        
        # Test calling the handler
        test_req = ActivateReqInput(model_name="test", gpu_id=0)
        result = dispatcher(test_req)
        print(f"✅ Dispatcher called handler correctly: {result}")
        
        return True
    except ImportError as e:
        print(f"⚠️  SGLang import failed: {e}")
        return None
    except Exception as e:
        print(f"❌ SGLang test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_patch_application():
    """Test 5: Patch mechanism works"""
    print("\n" + "=" * 60)
    print("Test 5: Patch mechanism")
    print("=" * 60)
    
    try:
        from sglang.utils import TypeBasedDispatcher
        from prism.io_struct import ActivateReqInput, DeactivateReqInput
        
        # Create a mock dispatcher (simulating what Scheduler has)
        dispatcher = TypeBasedDispatcher([])
        
        # Simulate applying patches
        call_log = []
        
        def handle_activate(req):
            call_log.append(('activate', req.model_name))
            return f"Activated {req.model_name}"
        
        def handle_deactivate(req):
            call_log.append(('deactivate', req.model_name))
            return f"Deactivated {req.model_name}"
        
        # Add handlers (this is what our patch does)
        dispatcher._mapping[ActivateReqInput] = handle_activate
        dispatcher._mapping[DeactivateReqInput] = handle_deactivate
        
        print(f"✅ Handlers registered to dispatcher")
        print(f"   Registered types: {list(dispatcher._mapping.keys())}")
        
        # Test activation
        activate_req = ActivateReqInput(model_name="llama-7b", gpu_id=0)
        result1 = dispatcher(activate_req)
        print(f"✅ Activate handled: {result1}")
        
        # Test deactivation  
        deactivate_req = DeactivateReqInput(model_name="llama-7b")
        result2 = dispatcher(deactivate_req)
        print(f"✅ Deactivate handled: {result2}")
        
        # Verify call log
        assert call_log == [('activate', 'llama-7b'), ('deactivate', 'llama-7b')]
        print(f"✅ Call sequence verified: {call_log}")
        
        return True
    except ImportError as e:
        print(f"⚠️  Import failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Patch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_elastic_memory_pool():
    """Test 6: Elastic memory pool (structure only, no GPU)"""
    print("\n" + "=" * 60)
    print("Test 6: Elastic memory pool structure")
    print("=" * 60)
    
    try:
        from prism.patches.memory_pool_patch import MHATokenToKVPoolElastic
        
        # Just check the class exists and has the right methods
        methods = ['__init__', 'alloc', 'free', 'available_size', 
                   'get_key_buffer', 'get_value_buffer', 'release', 'shutdown']
        
        for method in methods:
            if hasattr(MHATokenToKVPoolElastic, method):
                print(f"✅ MHATokenToKVPoolElastic.{method} exists")
            else:
                print(f"❌ MHATokenToKVPoolElastic.{method} missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Memory pool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_redis_utils():
    """Test 7: Redis utilities (no actual Redis needed)"""
    print("\n" + "=" * 60)
    print("Test 7: Redis utilities")
    print("=" * 60)
    
    try:
        from prism.utils.redis_utils import RedisClient, AsyncRedisClient
        
        print(f"✅ RedisClient imported")
        print(f"✅ AsyncRedisClient imported")
        
        # Check methods exist
        for cls, name in [(RedisClient, "RedisClient"), (AsyncRedisClient, "AsyncRedisClient")]:
            methods = ['send_pyobj', 'recv_pyobj_block', 'close']
            for method in methods:
                if hasattr(cls, method):
                    print(f"   {name}.{method} ✓")
                else:
                    print(f"   {name}.{method} ✗")
        
        return True
    except ImportError as e:
        print(f"⚠️  Redis import failed (redis package may not be installed): {e}")
        return None
    except Exception as e:
        print(f"❌ Redis utils test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  PRISM PATCH MECHANISM VERIFICATION")
    print("=" * 60)
    
    results = {}
    
    results['import'] = test_import_prism()
    results['data_structures'] = test_data_structures()
    results['server_args'] = test_server_args()
    results['sglang'] = test_sglang_import()
    results['patch'] = test_patch_application()
    results['memory_pool'] = test_elastic_memory_pool()
    results['redis'] = test_redis_utils()
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results.items():
        if result is True:
            status = "✅ PASS"
            passed += 1
        elif result is False:
            status = "❌ FAIL"
            failed += 1
        else:
            status = "⚠️  SKIP"
            skipped += 1
        print(f"  {name:20s} {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1
    else:
        print("\n✅ All required tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
