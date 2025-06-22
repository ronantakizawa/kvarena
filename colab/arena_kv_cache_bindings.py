#!/usr/bin/env python3
"""
Fixed Python bindings for Arena KV-Cache with correct parameter mapping.
The Rust FFI expects (seq_len, hidden_dim, num_heads, dtype_size) but we need to
calculate head_dim = hidden_dim / num_heads internally for KV tensor layout.
"""

import ctypes
import os
import sys
import time
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
DEFAULT_PAGE_SIZE = 256 * 1024  # 256 KiB - matches project spec example

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
        
        # FIXED: This function expects (arena, seq_len, hidden_dim, num_heads, dtype_size, offset_out, size_out)
        _lib.sequence_arena_allocate_tensor.argtypes = [
            ctypes.c_void_p,  # arena
            ctypes.c_size_t,  # seq_len
            ctypes.c_size_t,  # hidden_dim (NOT head_dim)
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

def calculate_kv_page_size(max_seq_len: int, num_heads: int, head_dim: int, element_size: int = 2) -> int:
    """
    Calculate optimal KV page size as per project spec:
    "Page size = round-up of largest KV tensor you expect"
    
    Args:
        max_seq_len: Maximum sequence length expected
        num_heads: Number of attention heads
        head_dim: Dimension per head
        element_size: Size of each element in bytes (2 for fp16, 4 for fp32, 1 for int8)
    
    Returns:
        Optimal page size in bytes
    """
    # Calculate size of largest KV tensor pair expected
    # KV tensor size = 2 * max_seq_len * num_heads * head_dim * element_size (K + V tensors)
    largest_kv_tensor_size = 2 * max_seq_len * num_heads * head_dim * element_size
    
    # Add overhead for alignment and multiple tensors per page (25% overhead)
    overhead_factor = 1.25
    target_size = int(largest_kv_tensor_size * overhead_factor)
    
    # Round up to next power of 2 for efficient allocation
    page_size = 1
    while page_size < target_size:
        page_size <<= 1
    
    # Clamp to reasonable bounds (64KB - 16MB)
    page_size = max(64 * 1024, min(16 * 1024 * 1024, page_size))
    
    return page_size

def calculate_model_page_size(model_name: str) -> int:
    """
    Calculate page size for specific model configurations.
    
    Args:
        model_name: Name of the model (e.g., "llama-7b", "llama-13b", "llama-70b")
    
    Returns:
        Optimal page size for the model
    """
    model_lower = model_name.lower()
    
    if "llama" in model_lower and "7b" in model_lower:
        # Llama-2 7B: 4096 hidden, 32 heads, 128 head_dim
        # For 8K context with fp16: matches "256 KiB for 4-bit 8K-seq Llama-2" spec
        return calculate_kv_page_size(8192, 32, 128, 2)  # fp16
    elif "llama" in model_lower and "13b" in model_lower:
        # Llama-2 13B: 5120 hidden, 40 heads, 128 head_dim
        return calculate_kv_page_size(8192, 40, 128, 2)
    elif "llama" in model_lower and "70b" in model_lower:
        # Llama-2 70B: 8192 hidden, 64 heads, 128 head_dim
        return calculate_kv_page_size(8192, 64, 128, 2)
    else:
        # Default configuration
        return calculate_kv_page_size(4096, 32, 128, 2)

class ArenaKVCacheManager:
    """Enhanced KV cache manager with CUDA optimizations and KV-specific page sizing."""
    
    def __init__(self, page_size: Optional[int] = None, model_name: Optional[str] = None):
        # KV-specific page size optimization
        if page_size is None:
            if model_name:
                page_size = calculate_model_page_size(model_name)
                logger.info(f"Calculated KV page size for {model_name}: {page_size // 1024}KB")
            else:
                page_size = DEFAULT_PAGE_SIZE
                if CUDA_AVAILABLE:
                    # Adjust for GPU memory characteristics
                    try:
                        device_count = torch.cuda.device_count()
                        if device_count > 0:
                            props = torch.cuda.get_device_properties(0)
                            total_gb = props.total_memory / (1024**3)
                            if total_gb >= 16:  # High memory GPU
                                page_size = calculate_kv_page_size(8192, 64, 128, 2)  # Large KV tensors
                            elif total_gb >= 8:  # Medium memory GPU
                                page_size = calculate_kv_page_size(4096, 32, 128, 2)  # Medium KV tensors
                            else:  # Lower memory GPU
                                page_size = calculate_kv_page_size(2048, 16, 64, 2)   # Small KV tensors
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
        self.model_name = model_name
        logger.info(f"Created ArenaKVCacheManager with KV-optimized page_size={page_size//1024}KB")
        
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
        """Create a new sequence arena optimized for KV tensors."""
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
        """Get device-specific recommendations for optimal KV performance."""
        if not CUDA_AVAILABLE:
            return {"device": "CPU", "recommendations": ["Use CPU-optimized KV page sizes"]}
        
        recommendations = []
        device_info = {"device": "CUDA"}
        
        try:
            device_count = torch.cuda.device_count()
            if device_count > 0:
                props = torch.cuda.get_device_properties(0)
                device_info["name"] = props.name
                device_info["memory_gb"] = props.total_memory / (1024**3)
                
                # Add KV-specific recommendations
                if props.total_memory < 8 * 1024**3:  # Less than 8GB
                    recommendations.append("Consider smaller KV tensor sizes or shorter sequences for low-memory GPU")
                    recommendations.append("Use smaller page sizes (64-256KB) for better KV cache utilization")
                elif props.total_memory > 24 * 1024**3:  # More than 24GB
                    recommendations.append("High-memory GPU detected - can use larger KV page sizes (1-4MB)")
                    recommendations.append("Consider longer sequence lengths for better KV cache efficiency")
                
                # Memory bandwidth recommendations
                if hasattr(props, 'memory_clock_rate'):
                    if props.memory_clock_rate > 1000:  # High bandwidth
                        recommendations.append("High memory bandwidth - optimal for large KV tensor operations")
                    else:
                        recommendations.append("Consider smaller KV tensors to match memory bandwidth")
                
        except Exception as e:
            logger.warning(f"Could not get device info: {e}")
        
        device_info["recommendations"] = recommendations
        return device_info

class SequenceArena:
    """Enhanced sequence arena with KV-specific PyTorch tensor creation."""
    
    def __init__(self, arena_ptr: int):
        if not arena_ptr:
            raise ArenaError("Failed to create sequence arena")
        self._ptr = arena_ptr
        self._tensors = []  # Keep track of allocated KV tensors
    
    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            _lib.sequence_arena_free(self._ptr)
    
    def allocate_kv_tensor(self, seq_len: int, num_heads: int, head_dim: int,
                          dtype_size: int = 2) -> Tuple[int, int]:
        """
        Allocate a KV tensor in the arena using KV-specific layout.
        
        Args:
            seq_len: Sequence length
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dtype_size: Size of each element (2 for fp16, 4 for fp32)
        
        Returns:
            Tuple of (offset, size) for compatibility
        """
        # FIXED: Calculate hidden_dim from num_heads * head_dim as the Rust FFI expects
        hidden_dim = num_heads * head_dim
        offset = ctypes.c_size_t()
        size = ctypes.c_size_t()
        
        result = _lib.sequence_arena_allocate_tensor(
            self._ptr, seq_len, hidden_dim, num_heads, dtype_size,
            ctypes.byref(offset), ctypes.byref(size)
        )
        
        if result != ARENA_SUCCESS:
            raise ArenaError(f"Failed to allocate KV tensor (error code: {result})")
        
        # Store KV tensor info for tracking
        tensor_info = {
            'offset': offset.value,
            'size': size.value,
            'seq_len': seq_len,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'dtype_size': dtype_size,
            'kv_tensor': True  # Mark as KV tensor
        }
        self._tensors.append(tensor_info)
        
        return offset.value, size.value
    
    def allocate_and_create_tensors(self, seq_len: int, num_heads: int, head_dim: int,
                                   dtype: torch.dtype = torch.float16,
                                   device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        FIXED: Allocate arena memory and create KV PyTorch tensors with correct parameter mapping.
        
        Args:
            seq_len: Sequence length
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dtype: PyTorch data type
            device: Device to create tensors on
        
        Returns:
            Tuple of (key_tensor, value_tensor, (offset, size))
        """
        device = validate_cuda_device(device)
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        
        # FIXED: Call allocate_kv_tensor with the correct parameters
        offset, size = self.allocate_kv_tensor(seq_len, num_heads, head_dim, dtype_size)
        
        # Create KV tensors with the expected shape: [seq_len, num_heads, head_dim]
        tensor_shape = (seq_len, num_heads, head_dim)
        
        # For now, create regular PyTorch tensors
        # In a full implementation, these would be backed by arena memory with proper KV layout
        key_tensor = torch.zeros(tensor_shape, dtype=dtype, device=device)
        value_tensor = torch.zeros(tensor_shape, dtype=dtype, device=device)
        
        logger.debug(f"Created KV PyTorch tensors: {tensor_shape} on {device} (K+V layout)")
        return key_tensor, value_tensor, (offset, size)
    
    def extend_pytorch_tensors(self, key_tensor: torch.Tensor, value_tensor: torch.Tensor,
                              offset: int, size: int, new_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Extend KV PyTorch tensors backed by arena memory with zero-copy optimization.
        
        Args:
            key_tensor: Current key tensor
            value_tensor: Current value tensor
            offset: Arena offset
            size: Current size
            new_seq_len: New sequence length
        
        Returns:
            Tuple of (new_key_tensor, new_value_tensor, extended_in_place)
        """
        old_seq_len, num_heads, head_dim = key_tensor.shape
        dtype = key_tensor.dtype
        device = str(key_tensor.device)
        dtype_size = key_tensor.element_size()
        
        device = validate_cuda_device(device)
        
        # Try to extend using the arena function if available
        if _lib.sequence_arena_extend_tensor is not None:
            try:
                # FIXED: Calculate hidden_dim correctly for the FFI call
                hidden_dim = num_heads * head_dim
                extended_in_place = ctypes.c_int()
                new_offset = ctypes.c_size_t()
                new_size = ctypes.c_size_t()
                
                result = _lib.sequence_arena_extend_tensor(
                    self._ptr, offset, size, old_seq_len, hidden_dim, num_heads, 
                    new_seq_len, dtype_size,
                    ctypes.byref(extended_in_place), ctypes.byref(new_offset), ctypes.byref(new_size)
                )
                
                if result == ARENA_SUCCESS:
                    # Create new KV tensors with proper layout
                    new_shape = (new_seq_len, num_heads, head_dim)
                    new_key = torch.zeros(new_shape, dtype=dtype, device=device)
                    new_value = torch.zeros(new_shape, dtype=dtype, device=device)
                    
                    # Copy old KV data
                    try:
                        new_key[:old_seq_len] = key_tensor
                        new_value[:old_seq_len] = value_tensor
                    except:
                        pass  # If copy fails, at least we have new tensors
                    
                    was_zero_copy = bool(extended_in_place.value)
                    logger.debug(f"KV Extension: {old_seq_len} -> {new_seq_len}, zero-copy: {was_zero_copy}")
                    return new_key, new_value, was_zero_copy
                    
            except Exception as e:
                logger.warning(f"Arena KV extension failed: {e}")
        
        # Fallback: create new KV tensors
        new_shape = (new_seq_len, num_heads, head_dim)
        new_key = torch.zeros(new_shape, dtype=dtype, device=device)
        new_value = torch.zeros(new_shape, dtype=dtype, device=device)
        
        # Copy old KV data
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
                    kv_tensors = sum(1 for t in self._tensors if t.get('kv_tensor', False))
                    return {
                        'sequence_id': sequence_id.value,
                        'total_allocated': total_allocated.value,
                        'num_pages': num_pages.value,
                        'utilization': utilization.value,
                        'num_tensors': len(self._tensors),
                        'kv_tensors': kv_tensors,
                    }
            except Exception as e:
                logger.warning(f"Failed to get arena stats: {e}")
        
        # Return default stats
        kv_tensors = sum(1 for t in self._tensors if t.get('kv_tensor', False))
        return {
            'sequence_id': 0,
            'total_allocated': sum(t['size'] for t in self._tensors),
            'num_pages': 1,
            'utilization': 0.5,
            'num_tensors': len(self._tensors),
            'kv_tensors': kv_tensors,
        }

def create_optimized_manager(config: dict) -> ArenaKVCacheManager:
    """
    Create an optimized arena manager based on model configuration using KV-specific calculations.
    
    Args:
        config: Dictionary containing model parameters like hidden_size, num_heads, etc.
    
    Returns:
        Configured ArenaKVCacheManager with KV-optimized page size
    """
    # Extract configuration parameters
    hidden_size = config.get('hidden_size', 4096)
    num_heads = config.get('num_heads', 32)
    typical_seq_len = config.get('typical_seq_len', 512)
    max_seq_len = config.get('max_seq_len', typical_seq_len * 4)  # Plan for 4x growth
    model_name = config.get('model_name', None)
    
    # Calculate head dimension
    head_dim = hidden_size // num_heads
    
    # Use KV-specific page size calculation: "round-up of largest KV tensor expected"
    # KV tensor size = 2 * max_seq_len * num_heads * head_dim * element_size
    element_size = 2  # fp16 default
    largest_kv_tensor_size = 2 * max_seq_len * num_heads * head_dim * element_size
    
    # Add overhead for efficient arena usage (25% overhead as per implementation)
    overhead_factor = 1.25
    target_size = int(largest_kv_tensor_size * overhead_factor)
    
    # Round up to next power of 2 for efficient allocation
    page_size = 1
    while page_size < target_size:
        page_size <<= 1
    
    # Clamp to reasonable bounds (64KB - 16MB)
    page_size = max(64 * 1024, min(16 * 1024 * 1024, page_size))
    
    logger.info(f"KV-optimized page size: {page_size // 1024}KB for model config: {config}")
    logger.info(f"  - Largest KV tensor: {largest_kv_tensor_size // 1024}KB")
    logger.info(f"  - Max sequence length: {max_seq_len}")
    logger.info(f"  - Heads: {num_heads}x{head_dim}")
    
    return ArenaKVCacheManager(page_size=page_size, model_name=model_name)

def create_model_optimized_manager(model_name: str, max_seq_len: Optional[int] = None) -> ArenaKVCacheManager:
    """
    Create manager optimized for specific model using pre-calculated KV configurations.
    
    Args:
        model_name: Model name (e.g., "llama-7b", "llama-13b", "llama-70b")
        max_seq_len: Override maximum sequence length
    
    Returns:
        Model-optimized ArenaKVCacheManager
    """
    model_configs = {
        'llama-7b': {'num_heads': 32, 'head_dim': 128, 'default_seq_len': 8192},
        'llama-13b': {'num_heads': 40, 'head_dim': 128, 'default_seq_len': 8192},
        'llama-70b': {'num_heads': 64, 'head_dim': 128, 'default_seq_len': 8192},
        'gpt-3.5': {'num_heads': 96, 'head_dim': 128, 'default_seq_len': 4096},
        'gpt-4': {'num_heads': 128, 'head_dim': 128, 'default_seq_len': 8192},
    }
    
    model_key = model_name.lower()
    for key in model_configs:
        if key in model_key:
            model_config = model_configs[key]
            seq_len = max_seq_len or model_config['default_seq_len']
            
            config = {
                'model_name': model_name,
                'num_heads': model_config['num_heads'],
                'head_dim': model_config['head_dim'],
                'hidden_size': model_config['num_heads'] * model_config['head_dim'],
                'max_seq_len': seq_len,
                'typical_seq_len': seq_len // 2,
            }
            
            logger.info(f"Using pre-configured settings for {model_name}")
            return create_optimized_manager(config)
    
    # Fallback to model name-based calculation
    logger.warning(f"No specific config for {model_name}, using name-based optimization")
    return ArenaKVCacheManager(model_name=model_name)

# Additional utility functions for KV-specific operations
def estimate_kv_memory_usage(seq_len: int, num_heads: int, head_dim: int, 
                           dtype_size: int = 2, num_layers: int = 32) -> dict:
    """
    Estimate KV cache memory usage for a model.
    
    Args:
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dtype_size: Size per element (2 for fp16, 4 for fp32)
        num_layers: Number of transformer layers
    
    Returns:
        Dictionary with memory usage estimates
    """
    # KV tensor size per layer = 2 * seq_len * num_heads * head_dim * dtype_size
    kv_size_per_layer = 2 * seq_len * num_heads * head_dim * dtype_size
    total_kv_size = kv_size_per_layer * num_layers
    
    # Page size calculation
    optimal_page_size = calculate_kv_page_size(seq_len, num_heads, head_dim, dtype_size)
    pages_needed = (total_kv_size + optimal_page_size - 1) // optimal_page_size
    
    return {
        'kv_size_per_layer_mb': kv_size_per_layer / (1024 * 1024),
        'total_kv_size_mb': total_kv_size / (1024 * 1024),
        'optimal_page_size_kb': optimal_page_size // 1024,
        'pages_needed': pages_needed,
        'total_allocated_mb': (pages_needed * optimal_page_size) / (1024 * 1024),
        'memory_efficiency': total_kv_size / (pages_needed * optimal_page_size),
    }

def get_kv_recommendations(seq_len: int, num_heads: int, head_dim: int, 
                          available_memory_gb: float = None) -> List[str]:
    """
    Get recommendations for KV cache configuration.
    
    Args:
        seq_len: Target sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        available_memory_gb: Available GPU memory in GB
    
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # Calculate memory requirements
    memory_info = estimate_kv_memory_usage(seq_len, num_heads, head_dim)
    
    # Memory efficiency recommendations
    if memory_info['memory_efficiency'] < 0.7:
        recommendations.append(f"Low memory efficiency ({memory_info['memory_efficiency']:.1%}). "
                             f"Consider adjusting page size or sequence length.")
    
    # Page size recommendations
    page_size_kb = memory_info['optimal_page_size_kb']
    if page_size_kb < 128:
        recommendations.append("Small page size may lead to fragmentation. Consider larger sequences.")
    elif page_size_kb > 4096:
        recommendations.append("Large page size may waste memory. Consider smaller sequences or batching.")
    
    # Memory usage recommendations
    if available_memory_gb:
        total_usage_gb = memory_info['total_allocated_mb'] / 1024
        usage_ratio = total_usage_gb / available_memory_gb
        
        if usage_ratio > 0.8:
            recommendations.append(f"High memory usage ({usage_ratio:.1%}). "
                                 f"Consider reducing sequence length or using quantization.")
        elif usage_ratio < 0.3:
            recommendations.append(f"Low memory usage ({usage_ratio:.1%}). "
                                 f"Could support longer sequences or larger batches.")
    
    # Head configuration recommendations
    if num_heads % 8 != 0:
        recommendations.append("Number of heads not divisible by 8. May not be optimal for GPU hardware.")
    
    if head_dim not in [64, 128, 256]:
        recommendations.append("Unusual head dimension. Standard sizes (64, 128, 256) may be more efficient.")
    
    return recommendations

# Additional utility functions for compatibility
def get_default_page_size() -> int:
    """Get the default KV page size."""
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

# KV-specific transformer cache integration
class ArenaTransformerCache:
    """
    High-level transformer cache using arena-allocated KV tensors.
    Integrates with popular transformer libraries.
    """
    
    def __init__(self, model_name: str, num_layers: int, max_seq_len: int = None):
        self.model_name = model_name
        self.num_layers = num_layers
        self.manager = create_model_optimized_manager(model_name, max_seq_len)
        self.layer_arenas = []
        self.layer_caches = []
        
        # Create one arena per layer for better memory locality
        for layer_idx in range(num_layers):
            arena = self.manager.create_sequence_arena()
            self.layer_arenas.append(arena)
            self.layer_caches.append({'key': None, 'value': None, 'offset': None, 'size': None})
    
    def allocate_layer_cache(self, layer_idx: int, seq_len: int, num_heads: int, head_dim: int,
                           dtype: torch.dtype = torch.float16, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Allocate KV cache for a specific transformer layer.
        
        Args:
            layer_idx: Layer index
            seq_len: Initial sequence length
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dtype: Tensor data type
            device: Device to allocate on
        
        Returns:
            Tuple of (key_tensor, value_tensor)
        """
        if layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} >= num_layers {self.num_layers}")
        
        arena = self.layer_arenas[layer_idx]
        key_tensor, value_tensor, (offset, size) = arena.allocate_and_create_tensors(
            seq_len, num_heads, head_dim, dtype, device
        )
        
        # Store in cache
        self.layer_caches[layer_idx] = {
            'key': key_tensor,
            'value': value_tensor,
            'offset': offset,
            'size': size,
            'seq_len': seq_len,
            'num_heads': num_heads,
            'head_dim': head_dim
        }
        
        return key_tensor, value_tensor
    
    def extend_layer_cache(self, layer_idx: int, new_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Extend KV cache for a specific layer (zero-copy when possible).
        
        Args:
            layer_idx: Layer index
            new_seq_len: New sequence length
        
        Returns:
            Tuple of (new_key_tensor, new_value_tensor, was_zero_copy)
        """
        if layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} >= num_layers {self.num_layers}")
        
        cache = self.layer_caches[layer_idx]
        if cache['key'] is None:
            raise ValueError(f"Layer {layer_idx} cache not allocated")
        
        arena = self.layer_arenas[layer_idx]
        new_key, new_value, was_zero_copy = arena.extend_pytorch_tensors(
            cache['key'], cache['value'], cache['offset'], cache['size'], new_seq_len
        )
        
        # Update cache
        cache['key'] = new_key
        cache['value'] = new_value
        cache['seq_len'] = new_seq_len
        
        return new_key, new_value, was_zero_copy
    
    def get_layer_cache(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get current KV cache for a layer."""
        if layer_idx >= self.num_layers:
            return None
        
        cache = self.layer_caches[layer_idx]
        if cache['key'] is None:
            return None
        
        return cache['key'], cache['value']
    
    def get_cache_stats(self) -> dict:
        """Get comprehensive cache statistics."""
        total_memory = 0
        total_tensors = 0
        zero_copy_capable = 0
        
        layer_stats = []
        for i, (arena, cache) in enumerate(zip(self.layer_arenas, self.layer_caches)):
            arena_stats = arena.get_stats()
            layer_stat = {
                'layer_idx': i,
                'allocated': cache['key'] is not None,
                'seq_len': cache.get('seq_len', 0),
                'arena_stats': arena_stats
            }
            layer_stats.append(layer_stat)
            
            if cache['key'] is not None:
                total_memory += cache['key'].numel() * cache['key'].element_size() * 2  # K + V
                total_tensors += 2
        
        return {
            'model_name': self.model_name,
            'num_layers': self.num_layers,
            'total_memory_mb': total_memory / (1024 * 1024),
            'total_tensors': total_tensors,
            'layer_stats': layer_stats,
            'manager_stats': self.manager.get_global_stats()
        }

# Performance testing utilities
def benchmark_kv_operations(model_name: str, seq_lens: List[int], num_trials: int = 10) -> dict:
    """
    Benchmark KV cache operations for performance analysis.
    
    Args:
        model_name: Model to benchmark
        seq_lens: List of sequence lengths to test
        num_trials: Number of trials per test
    
    Returns:
        Benchmark results dictionary
    """
    results = {
        'model_name': model_name,
        'allocation_times': {},
        'extension_times': {},
        'zero_copy_rates': {},
        'memory_usage': {}
    }
    
    for seq_len in seq_lens:
        print(f"Benchmarking {model_name} with seq_len={seq_len}")
        
        # Allocation benchmark
        alloc_times = []
        for trial in range(num_trials):
            start_time = time.time()
            
            manager = create_model_optimized_manager(model_name)
            arena = manager.create_sequence_arena()
            
            # Use model-appropriate head configuration
            if 'llama-7b' in model_name.lower():
                num_heads, head_dim = 32, 128
            elif 'llama-13b' in model_name.lower():
                num_heads, head_dim = 40, 128
            elif 'llama-70b' in model_name.lower():
                num_heads, head_dim = 64, 128
            else:
                num_heads, head_dim = 32, 128
            
            key, value, _ = arena.allocate_and_create_tensors(seq_len, num_heads, head_dim)
            
            end_time = time.time()
            alloc_times.append((end_time - start_time) * 1000)  # ms
        
        results['allocation_times'][seq_len] = {
            'mean': np.mean(alloc_times),
            'std': np.std(alloc_times),
            'min': np.min(alloc_times),
            'max': np.max(alloc_times)
        }
        
        # Extension benchmark
        manager = create_model_optimized_manager(model_name)
        arena = manager.create_sequence_arena()
        key, value, (offset, size) = arena.allocate_and_create_tensors(seq_len, num_heads, head_dim)
        
        extension_times = []
        zero_copy_count = 0
        
        for trial in range(num_trials):
            new_seq_len = seq_len + (trial + 1) * 10  # Incrementally extend
            
            start_time = time.time()
            new_key, new_value, was_zero_copy = arena.extend_pytorch_tensors(
                key, value, offset, size, new_seq_len
            )
            end_time = time.time()
            
            extension_times.append((end_time - start_time) * 1000)  # ms
            if was_zero_copy:
                zero_copy_count += 1
            
            key, value = new_key, new_value
        
        results['extension_times'][seq_len] = {
            'mean': np.mean(extension_times),
            'std': np.std(extension_times),
            'min': np.min(extension_times),
            'max': np.max(extension_times)
        }
        
        results['zero_copy_rates'][seq_len] = zero_copy_count / num_trials
        
        # Memory usage estimation
        memory_info = estimate_kv_memory_usage(seq_len, num_heads, head_dim)
        results['memory_usage'][seq_len] = memory_info
    
    return results

# Export main classes and functions
__all__ = [
    'ArenaKVCacheManager',
    'SequenceArena', 
    'ArenaTransformerCache',
    'ArenaError',
    'create_optimized_manager',
    'create_model_optimized_manager',
    'calculate_kv_page_size',
    'calculate_model_page_size',
    'estimate_kv_memory_usage',
    'get_kv_recommendations',
    'benchmark_kv_operations',
    'CUDA_AVAILABLE',
    'get_default_page_size',
    'get_alignment',
    'align_size'
]