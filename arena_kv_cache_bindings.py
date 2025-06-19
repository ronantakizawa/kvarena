
"""
Auto-generated ctypes bindings for Arena KV-Cache library.
"""

import ctypes
import os
from pathlib import Path

# Load the library
_lib_path = Path(__file__).parent / "libarena_kv_cache.dylib"
_lib = ctypes.CDLL(str(_lib_path))

# Error codes
ARENA_SUCCESS = 0
ARENA_ERROR_ALLOC = -1
ARENA_ERROR_INVALID_PARAM = -2

# Function signatures
_lib.arena_cache_manager_new.argtypes = [ctypes.c_size_t]
_lib.arena_cache_manager_new.restype = ctypes.c_void_p

_lib.arena_cache_manager_free.argtypes = [ctypes.c_void_p]
_lib.arena_cache_manager_free.restype = None

_lib.arena_create_sequence.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
_lib.arena_create_sequence.restype = ctypes.c_int

_lib.arena_sequence_free.argtypes = [ctypes.c_void_p]
_lib.arena_sequence_free.restype = None

_lib.arena_allocate_kv_tensor.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, 
    ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), 
    ctypes.POINTER(ctypes.c_size_t)
]
_lib.arena_allocate_kv_tensor.restype = ctypes.c_int

_lib.arena_get_tensor_ptr.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
]
_lib.arena_get_tensor_ptr.restype = ctypes.c_void_p

_lib.arena_get_stats.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double)
]
_lib.arena_get_stats.restype = ctypes.c_int

_lib.arena_benchmark_allocation.argtypes = [
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_size_t)
]
_lib.arena_benchmark_allocation.restype = ctypes.c_int

class ArenaKVCacheManager:
    """Python wrapper for Rust KV cache manager"""
    
    def __init__(self, page_size=256 * 1024):
        self._ptr = _lib.arena_cache_manager_new(page_size)
        if not self._ptr:
            raise RuntimeError("Failed to create KV cache manager")
    
    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            _lib.arena_cache_manager_free(self._ptr)
    
    def create_sequence_arena(self):
        """Create a new sequence arena"""
        arena_ptr = ctypes.c_void_p()
        result = _lib.arena_create_sequence(self._ptr, ctypes.byref(arena_ptr))
        if result != ARENA_SUCCESS:
            raise RuntimeError(f"Failed to create sequence arena: {result}")
        return SequenceArena(arena_ptr.value)

class SequenceArena:
    """Python wrapper for Rust sequence arena"""
    
    def __init__(self, ptr):
        self._ptr = ptr
    
    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            _lib.arena_sequence_free(self._ptr)
    
    def allocate_kv_tensor(self, seq_len, hidden_dim, num_heads, dtype_size):
        """Allocate a KV tensor and return (offset, size)"""
        offset = ctypes.c_size_t()
        size = ctypes.c_size_t()
        
        result = _lib.arena_allocate_kv_tensor(
            self._ptr, seq_len, hidden_dim, num_heads, dtype_size,
            ctypes.byref(offset), ctypes.byref(size)
        )
        
        if result != ARENA_SUCCESS:
            raise RuntimeError(f"Failed to allocate KV tensor: {result}")
        
        return offset.value, size.value
    
    def get_tensor_ptr(self, offset, size, seq_len, hidden_dim, num_heads):
        """Get raw pointer to tensor data"""
        ptr = _lib.arena_get_tensor_ptr(self._ptr, offset, size, seq_len, hidden_dim, num_heads)
        if not ptr:
            raise RuntimeError("Failed to get tensor pointer")
        return ptr
    
    def get_stats(self):
        """Get arena statistics"""
        total_allocated = ctypes.c_size_t()
        num_pages = ctypes.c_size_t()
        utilization = ctypes.c_double()
        
        result = _lib.arena_get_stats(
            self._ptr, ctypes.byref(total_allocated),
            ctypes.byref(num_pages), ctypes.byref(utilization)
        )
        
        if result != ARENA_SUCCESS:
            raise RuntimeError(f"Failed to get stats: {result}")
        
        return {
            'total_allocated': total_allocated.value,
            'num_pages': num_pages.value,
            'utilization': utilization.value
        }

def benchmark_allocation(page_size, num_sequences, avg_seq_len, hidden_dim, num_heads, dtype_size):
    """Benchmark allocation performance"""
    time_ns = ctypes.c_uint64()
    memory_bytes = ctypes.c_size_t()
    
    result = _lib.arena_benchmark_allocation(
        page_size, num_sequences, avg_seq_len, hidden_dim, num_heads, dtype_size,
        ctypes.byref(time_ns), ctypes.byref(memory_bytes)
    )
    
    if result != ARENA_SUCCESS:
        raise RuntimeError(f"Benchmark failed: {result}")
    
    return {
        'time_ns': time_ns.value,
        'memory_bytes': memory_bytes.value,
        'time_ms': time_ns.value / 1_000_000,
        'memory_mb': memory_bytes.value / 1024 / 1024
    }
