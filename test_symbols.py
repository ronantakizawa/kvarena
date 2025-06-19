#!/usr/bin/env python3
import ctypes
import os

try:
    # Try to load the library
    lib_path = "./libarena_kv_cache.dylib"
    if not os.path.exists(lib_path):
        print(f"❌ Library not found: {lib_path}")
        exit(1)
    
    print(f"📚 Loading library: {lib_path}")
    lib = ctypes.CDLL(lib_path)
    print("✅ Library loaded successfully")
    
    # Test if our functions are available
    functions_to_test = [
        "arena_cache_manager_new",
        "arena_cache_manager_free", 
        "arena_create_sequence",
        "arena_sequence_free",
        "arena_allocate_kv_tensor",
        "arena_get_stats"
    ]
    
    for func_name in functions_to_test:
        try:
            func = getattr(lib, func_name)
            print(f"✅ Found function: {func_name}")
        except AttributeError:
            print(f"❌ Missing function: {func_name}")
    
    print("\n🎉 Symbol test completed!")
    
except Exception as e:
    print(f"❌ Error loading library: {e}")
    import traceback
    traceback.print_exc()
