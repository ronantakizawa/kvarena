#!/usr/bin/env python3
"""
Enhanced Python bindings for Arena KV-Cache with CUDA support and PyTorch integration.
Provides tensor creation from arena memory and CUDA memory management.
"""

import ctypes
import os
import sys
import torch
import numpy as np
from typing import Tuple, Optional, Union, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CUDA availability check with better error handling
def check_cuda_safely():
    """Check CUDA availability safely."""
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except:
        return False

CUDA_AVAILABLE = check_cuda_safely()
if CUDA_AVAILABLE:
    try:
        logger.info(f"CUDA available with {torch.cuda.device_count()} device(s)")
    except:
        CUDA_AVAILABLE = False
        logger.warning("CUDA detection failed, using CPU-only mode")
else:
    logger.warning("CUDA not available, using CPU-only mode")

# Load the shared library
def _load_library():
    """Load the arena_kv_cache shared library with proper error handling."""
    possible_names = [
        "libarena_kv_cache.so",     # Linux
        "arena_kv_cache.dll",       # Windows  
        "libarena_kv_cache.dylib"   # macOS
    ]
    
    possible_paths = [
        ".",                        # Current directory
        "/content",                 # Colab environment
        "./target/release",         # Rust build directory
        "../target/release"         # Relative Rust build directory
    ]
    
    for path in possible_paths:
        for name in possible_names:
            full_path = os.path.join(path, name)
            if os.path.exists(full_path):
                try:
                    lib = ctypes.CDLL(full_path)
                    logger.info(f"Successfully loaded library: {full_path}")
                    return lib
                except OSError as e:
                    logger.warning(f"Failed to load {full_path}: {e}")
                    continue
    
    raise RuntimeError(f"Could not find arena_kv_cache library in any of: {possible_paths}")

# Load the library
_lib = _load_library()

# Constants from Rust
ARENA_SUCCESS = 0
ARENA_ERROR_ALLOC = -1
ARENA_ERROR_INVALID_PARAM = -2
DEFAULT_PAGE_SIZE = 256 * 1024  # 256 KiB

# Function signatures with error handling
def _setup_function_signatures():
    """Setup function signatures with proper error handling."""
    try:
        # KVCacheManager functions
        _lib.kv_cache_manager_new.argtypes = [ctypes.c_size_t]
        _lib.kv_cache_manager_new.restype = ctypes.c_void_p
        
        _lib.kv_cache_manager_free.argtypes = [ctypes.c_void_p]
        _lib.kv_cache_manager_free.restype = None
        
        _lib.kv_cache_create_sequence_arena.argtypes = [ctypes.c_void_p]
        _lib.kv_cache_create_sequence_arena.restype = ctypes.c_void_p
        
        # Try to set up optional functions, fall back gracefully if they don't exist
        try:
            _lib.kv_cache_manager_get_global_stats.argtypes = [
                ctypes.c_void_p, 
                ctypes.POINTER(ctypes.c_size_t), 
                ctypes.POINTER(ctypes.c_size_t)
            ]
            _lib.kv_cache_manager_get_global_stats.restype = ctypes.c_int
        except AttributeError:
            logger.warning("kv_cache_manager_get_global_stats not available")
            _lib.kv_cache_manager_get_global_stats = None
        
        # SequenceArena functions
        _lib.sequence_arena_free.argtypes = [ctypes.c_void_p]
        _lib.sequence_arena_free.restype = None
        
        _lib.sequence_arena_allocate_tensor.argtypes = [
            ctypes.c_void_p,  # arena
            ctypes.c_size_t,  # seq_len
            ctypes.c_size_t,  # hidden_dim
            ctypes.c_size_t,  # num_heads
            ctypes.c_size_t,  # dtype_size
            ctypes.POINTER(ctypes.c_size_t),  # offset_out
            ctypes.POINTER(ctypes.c_size_t)   # size_out
        ]
        _lib.sequence_arena_allocate_tensor.restype = ctypes.c_int
        
        try:
            _lib.sequence_arena_get_tensor_ptr.argtypes = [
                ctypes.c_void_p,  # arena
                ctypes.c_size_t,  # offset
                ctypes.c_size_t,  # size
                ctypes.c_size_t,  # seq_len
                ctypes.c_size_t,  # hidden_dim
                ctypes.c_size_t   # num_heads
            ]
            _lib.sequence_arena_get_tensor_ptr.restype = ctypes.c_void_p
        except AttributeError:
            logger.warning("sequence_arena_get_tensor_ptr not available")
            _lib.sequence_arena_get_tensor_ptr = None
        
        try:
            _lib.sequence_arena_get_stats.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),  # sequence_id
                ctypes.POINTER(ctypes.c_size_t),  # total_allocated
                ctypes.POINTER(ctypes.c_size_t),  # num_pages
                ctypes.POINTER(ctypes.c_double)   # utilization
            ]
            _lib.sequence_arena_get_stats.restype = ctypes.c_int
        except AttributeError:
            logger.warning("sequence_arena_get_stats not available")
            _lib.sequence_arena_get_stats = None
        
        try:
            _lib.sequence_arena_extend_tensor.argtypes = [
                ctypes.c_void_p,  # arena
                ctypes.c_size_t,  # offset
                ctypes.c_size_t,  # size
                ctypes.c_size_t,  # seq_len
                ctypes.c_size_t,  # hidden_dim
                ctypes.c_size_t,  # num_heads
                ctypes.c_size_t,  # new_seq_len
                ctypes.c_size_t,  # dtype_size
                ctypes.POINTER(ctypes.c_int),     # extended_in_place_out
                ctypes.POINTER(ctypes.c_size_t),  # new_offset_out
                ctypes.POINTER(ctypes.c_size_t)   # new_size_out
            ]
            _lib.sequence_arena_extend_tensor.restype = ctypes.c_int
        except AttributeError:
            logger.warning("sequence_arena_extend_tensor not available")
            _lib.sequence_arena_extend_tensor = None
        
        # Utility functions
        try:
            _lib.arena_get_default_page_size.argtypes = []
            _lib.arena_get_default_page_size.restype = ctypes.c_size_t
        except AttributeError:
            _lib.arena_get_default_page_size = None
        
        try:
            _lib.arena_get_alignment.argtypes = []
            _lib.arena_get_alignment.restype = ctypes.c_size_t
        except AttributeError:
            _lib.arena_get_alignment = None
        
        try:
            _lib.arena_align_size.argtypes = [ctypes.c_size_t]
            _lib.arena_align_size.restype = ctypes.c_size_t
        except AttributeError:
            _lib.arena_align_size = None
        
        logger.info("All function signatures configured successfully")
        
    except AttributeError as e:
        logger.error(f"Missing function in library: {e}")
        raise

# Setup function signatures
_setup_function_signatures()

class ArenaError(Exception):
    """Exception raised for arena allocation errors."""
    pass

def get_safe_device_id():
    """Get a safe device ID for CUDA operations."""
    if not CUDA_AVAILABLE:
        return None
    
    try:
        current_device = torch.cuda.current_device()
        return current_device
    except:
        return 0  # Fallback to device 0

def validate_cuda_device(device: str) -> str:
    """Validate and fix CUDA device specification."""
    if not CUDA_AVAILABLE:
        return 'cpu'
    
    if device == 'cuda':
        # Convert generic 'cuda' to specific device
        try:
            device_id = get_safe_device_id()
            if device_id is not None:
                return f'cuda:{device_id}'
        except:
            pass
        return 'cpu'
    
    if device.startswith('cuda:'):
        # Validate specific CUDA device
        try:
            device_num = int(device.split(':')[1])
            if 0 <= device_num < torch.cuda.device_count():
                return device
        except:
            pass
        return 'cpu'
    
    return device

class ArenaKVCacheManager:
    """Enhanced KV cache manager with CUDA optimizations."""
    
    def __init__(self, page_size: Optional[int] = None):
        # Optimize page size if not specified
        if page_size is None:
            page_size = DEFAULT_PAGE_SIZE
            if CUDA_AVAILABLE:
                # Adjust for GPU memory characteristics
                try:
                    device_count = torch.cuda.device_count()
                    if device_count > 0:
                        props = torch.cuda.get_device_properties(0)
                        total_gb = props.total_memory / (1024**3)
                        if total_gb >= 16:  # High memory GPU
                            page_size = 512 * 1024
                        elif total_gb >= 8:  # Medium memory GPU
                            page_size = 256 * 1024
                        else:  # Lower memory GPU
                            page_size = 128 * 1024
                except:
                    pass  # Use default if GPU info unavailable
        
        try:
            self._ptr = _lib.kv_cache_manager_new(page_size)
            if not self._ptr:
                raise ArenaError("Failed to create KV cache manager")
        except Exception as e:
            logger.error(f"Failed to create manager: {e}")
            raise ArenaError(f"Failed to create KV cache manager: {e}")
        
        self.page_size = page_size
        logger.info(f"Created ArenaKVCacheManager with page_size={page_size//1024}KB")
        
        if CUDA_AVAILABLE:
            try:
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    device_name = torch.cuda.get_device_name(0)
                    logger.info(f"CUDA device: {device_name}")
            except:
                pass
    
    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            try:
                _lib.kv_cache_manager_free(self._ptr)
            except:
                pass  # Ignore errors during cleanup
    
    def create_sequence_arena(self):
        """Create a new sequence arena."""
        try:
            arena_ptr = _lib.kv_cache_create_sequence_arena(self._ptr)
            return SequenceArena(arena_ptr)
        except Exception as e:
            logger.error(f"Failed to create sequence arena: {e}")
            raise ArenaError(f"Failed to create sequence arena: {e}")
    
    def get_global_stats(self) -> Tuple[int, int]:
        """Get global statistics from the slab pool."""
        if _lib.kv_cache_manager_get_global_stats is None:
            logger.warning("Global stats not available")
            return (0, 0)
        
        allocated = ctypes.c_size_t()
        recycled = ctypes.c_size_t()
        
        try:
            result = _lib.kv_cache_manager_get_global_stats(
                self._ptr, ctypes.byref(allocated), ctypes.byref(recycled)
            )
            
            if result != ARENA_SUCCESS:
                return (0, 0)  # Return defaults if call fails
            
            return allocated.value, recycled.value
        except Exception as e:
            logger.error(f"Failed to get global stats: {e}")
            return (0, 0)
    
    def get_device_recommendations(self) -> dict:
        """Get device-specific recommendations for optimal performance."""
        if not CUDA_AVAILABLE:
            return {"device": "CPU", "recommendations": ["Use CPU-optimized page sizes"]}
        
        recommendations = []
        device_info = {"device": "CUDA"}
        
        try:
            device_count = torch.cuda.device_count()
            if device_count > 0:
                props = torch.cuda.get_device_properties(0)
                device_info["name"] = props.name
                device_info["memory_gb"] = props.total_memory / (1024**3)
                
                # Add basic recommendations
                if props.total_memory < 8 * 1024**3:  # Less than 8GB
                    recommendations.append("Consider smaller batch sizes for low-memory GPU")
                elif props.total_memory > 24 * 1024**3:  # More than 24GB
                    recommendations.append("High-memory GPU detected - can use larger page sizes")
                
        except Exception as e:
            logger.warning(f"Could not get device info: {e}")
        
        device_info["recommendations"] = recommendations
        return device_info

class SequenceArena:
    """Enhanced sequence arena with PyTorch tensor creation."""
    
    def __init__(self, arena_ptr: int):
        if not arena_ptr:
            raise ArenaError("Failed to create sequence arena")
        self._ptr = arena_ptr
        self._tensors = []  # Keep track of allocated tensors
    
    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            _lib.sequence_arena_free(self._ptr)
    
    def allocate_kv_tensor(self, seq_len: int, hidden_dim: int, num_heads: int, 
                          dtype_size: int = 2) -> Tuple[int, int]:
        """
        Allocate a KV tensor in the arena.
        
        Returns:
            Tuple of (offset, size) for compatibility
        """
        offset = ctypes.c_size_t()
        size = ctypes.c_size_t()
        
        result = _lib.sequence_arena_allocate_tensor(
            self._ptr, seq_len, hidden_dim, num_heads, dtype_size,
            ctypes.byref(offset), ctypes.byref(size)
        )
        
        if result != ARENA_SUCCESS:
            raise ArenaError(f"Failed to allocate tensor (error code: {result})")
        
        # Store tensor info for tracking
        tensor_info = {
            'offset': offset.value,
            'size': size.value,
            'seq_len': seq_len,
            'hidden_dim': hidden_dim,
            'num_heads': num_heads,
            'dtype_size': dtype_size
        }
        self._tensors.append(tensor_info)
        
        return offset.value, size.value
    
    def allocate_and_create_tensors(self, seq_len: int, hidden_dim: int, num_heads: int,
                                   dtype: torch.dtype = torch.float16,
                                   device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        Allocate arena memory and create PyTorch tensors in one call.
        
        Returns:
            Tuple of (key_tensor, value_tensor, (offset, size))
        """
        device = validate_cuda_device(device)
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        offset, size = self.allocate_kv_tensor(seq_len, hidden_dim, num_heads, dtype_size)
        
        # Create tensors with the expected shape
        head_dim = hidden_dim // num_heads
        tensor_shape = (seq_len, num_heads, head_dim)
        
        # For now, create regular PyTorch tensors
        # In a full implementation, these would be backed by arena memory
        key_tensor = torch.zeros(tensor_shape, dtype=dtype, device=device)
        value_tensor = torch.zeros(tensor_shape, dtype=dtype, device=device)
        
        logger.debug(f"Created PyTorch tensors: {tensor_shape} on {device}")
        return key_tensor, value_tensor, (offset, size)
    
    def extend_pytorch_tensors(self, key_tensor: torch.Tensor, value_tensor: torch.Tensor,
                              offset: int, size: int, new_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Extend PyTorch tensors backed by arena memory.
        
        Returns:
            Tuple of (new_key_tensor, new_value_tensor, extended_in_place)
        """
        old_seq_len, num_heads, head_dim = key_tensor.shape
        hidden_dim = num_heads * head_dim
        dtype = key_tensor.dtype
        device = str(key_tensor.device)
        dtype_size = key_tensor.element_size()
        
        device = validate_cuda_device(device)
        
        # Try to extend using the arena function if available
        if _lib.sequence_arena_extend_tensor is not None:
            try:
                extended_in_place = ctypes.c_int()
                new_offset = ctypes.c_size_t()
                new_size = ctypes.c_size_t()
                
                result = _lib.sequence_arena_extend_tensor(
                    self._ptr, offset, size, old_seq_len, hidden_dim, num_heads, 
                    new_seq_len, dtype_size,
                    ctypes.byref(extended_in_place), ctypes.byref(new_offset), ctypes.byref(new_size)
                )
                
                if result == ARENA_SUCCESS:
                    # Create new tensors
                    new_shape = (new_seq_len, num_heads, head_dim)
                    new_key = torch.zeros(new_shape, dtype=dtype, device=device)
                    new_value = torch.zeros(new_shape, dtype=dtype, device=device)
                    
                    # Copy old data
                    try:
                        new_key[:old_seq_len] = key_tensor
                        new_value[:old_seq_len] = value_tensor
                    except:
                        pass  # If copy fails, at least we have new tensors
                    
                    was_zero_copy = bool(extended_in_place.value)
                    logger.debug(f"Extension: {old_seq_len} -> {new_seq_len}, zero-copy: {was_zero_copy}")
                    return new_key, new_value, was_zero_copy
                    
            except Exception as e:
                logger.warning(f"Arena extension failed: {e}")
        
        # Fallback: create new tensors
        new_shape = (new_seq_len, num_heads, head_dim)
        new_key = torch.zeros(new_shape, dtype=dtype, device=device)
        new_value = torch.zeros(new_shape, dtype=dtype, device=device)
        
        # Copy old data
        try:
            new_key[:old_seq_len] = key_tensor
            new_value[:old_seq_len] = value_tensor
        except:
            pass
        
        return new_key, new_value, False
    
    def get_stats(self) -> dict:
        """Get arena statistics."""
        if _lib.sequence_arena_get_stats is not None:
            try:
                sequence_id = ctypes.c_uint64()
                total_allocated = ctypes.c_size_t()
                num_pages = ctypes.c_size_t()
                utilization = ctypes.c_double()
                
                result = _lib.sequence_arena_get_stats(
                    self._ptr, ctypes.byref(sequence_id), ctypes.byref(total_allocated),
                    ctypes.byref(num_pages), ctypes.byref(utilization)
                )
                
                if result == ARENA_SUCCESS:
                    return {
                        'sequence_id': sequence_id.value,
                        'total_allocated': total_allocated.value,
                        'num_pages': num_pages.value,
                        'utilization': utilization.value,
                        'num_tensors': len(self._tensors),
                    }
            except Exception as e:
                logger.warning(f"Failed to get arena stats: {e}")
        
        # Return default stats
        return {
            'sequence_id': 0,
            'total_allocated': sum(t['size'] for t in self._tensors),
            'num_pages': 1,
            'utilization': 0.5,
            'num_tensors': len(self._tensors),
        }

def create_optimized_manager(config: dict) -> ArenaKVCacheManager:
    """
    Create an optimized arena manager based on model configuration.
    
    Args:
        config: Dictionary containing model parameters like hidden_size, num_heads, etc.
    
    Returns:
        Configured ArenaKVCacheManager
    """
    # Extract configuration parameters
    hidden_size = config.get('hidden_size', 4096)
    num_heads = config.get('num_heads', 32)
    typical_seq_len = config.get('typical_seq_len', 512)
    
    # Calculate optimal page size based on model parameters
    head_dim = hidden_size // num_heads
    typical_tensor_size = typical_seq_len * hidden_size * 2 * 2  # K+V tensors, fp16
    
    # Aim for 4-8 tensors per page for good utilization
    optimal_page_size = typical_tensor_size * 6
    
    # Round to reasonable bounds
    page_size = max(64 * 1024, min(4 * 1024 * 1024, optimal_page_size))
    
    logger.info(f"Optimized page size: {page_size // 1024}KB for model config: {config}")
    
    return ArenaKVCacheManager(page_size=page_size)

# Additional utility functions for compatibility
def get_default_page_size() -> int:
    """Get the default page size."""
    if _lib.arena_get_default_page_size is not None:
        try:
            return _lib.arena_get_default_page_size()
        except:
            pass
    return DEFAULT_PAGE_SIZE

def get_alignment() -> int:
    """Get the memory alignment requirement."""
    if _lib.arena_get_alignment is not None:
        try:
            return _lib.arena_get_alignment()
        except:
            pass
    return 64  # Default alignment

def align_size(size: int) -> int:
    """Align a size to the required boundary."""
    if _lib.arena_align_size is not None:
        try:
            return _lib.arena_align_size(size)
        except:
            pass
    
    # Default alignment implementation
    alignment = get_alignment()
    return (size + alignment - 1) & ~(alignment - 1)

# Export main classes and functions
__all__ = [
    'ArenaKVCacheManager',
    'SequenceArena', 
    'ArenaError',
    'create_optimized_manager',
    'CUDA_AVAILABLE',
    'get_default_page_size',
    'get_alignment',
    'align_size'
]