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

# Function signatures
try:
    # KVCacheManager functions
    _lib.kv_cache_manager_new.argtypes = [ctypes.c_size_t]
    _lib.kv_cache_manager_new.restype = ctypes.c_void_p
    
    _lib.kv_cache_manager_free.argtypes = [ctypes.c_void_p]
    _lib.kv_cache_manager_free.restype = None
    
    _lib.kv_cache_create_sequence_arena.argtypes = [ctypes.c_void_p]
    _lib.kv_cache_create_sequence_arena.restype = ctypes.c_void_p
    
    _lib.kv_cache_manager_get_global_stats.argtypes = [
        ctypes.c_void_p, 
        ctypes.POINTER(ctypes.c_size_t), 
        ctypes.POINTER(ctypes.c_size_t)
    ]
    _lib.kv_cache_manager_get_global_stats.restype = ctypes.c_int
    
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
    
    _lib.sequence_arena_get_tensor_ptr.argtypes = [
        ctypes.c_void_p,  # arena
        ctypes.c_size_t,  # offset
        ctypes.c_size_t,  # size
        ctypes.c_size_t,  # seq_len
        ctypes.c_size_t,  # hidden_dim
        ctypes.c_size_t   # num_heads
    ]
    _lib.sequence_arena_get_tensor_ptr.restype = ctypes.c_void_p
    
    _lib.sequence_arena_get_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),  # sequence_id
        ctypes.POINTER(ctypes.c_size_t),  # total_allocated
        ctypes.POINTER(ctypes.c_size_t),  # num_pages
        ctypes.POINTER(ctypes.c_double)   # utilization
    ]
    _lib.sequence_arena_get_stats.restype = ctypes.c_int
    
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
    
    # Utility functions
    _lib.arena_get_default_page_size.argtypes = []
    _lib.arena_get_default_page_size.restype = ctypes.c_size_t
    
    _lib.arena_get_alignment.argtypes = []
    _lib.arena_get_alignment.restype = ctypes.c_size_t
    
    _lib.arena_align_size.argtypes = [ctypes.c_size_t]
    _lib.arena_align_size.restype = ctypes.c_size_t
    
    logger.info("All function signatures configured successfully")
    
except AttributeError as e:
    logger.error(f"Missing function in library: {e}")
    raise


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


class CUDAMemoryManager:
    """CUDA memory management utilities with better error handling."""
    
    def __init__(self):
        self.device_count = 0
        self.current_device = 0
        
        if CUDA_AVAILABLE:
            try:
                self.device_count = torch.cuda.device_count()
                self.current_device = torch.cuda.current_device()
                logger.info(f"CUDA manager initialized with {self.device_count} device(s)")
            except Exception as e:
                logger.warning(f"CUDA manager initialization failed: {e}")
                self.device_count = 0
        
    def get_device_info(self, device: Optional[int] = None) -> dict:
        """Get CUDA device information."""
        if not CUDA_AVAILABLE or self.device_count == 0:
            return {"error": "CUDA not available"}
        
        device = device or self.current_device
        
        try:
            with torch.cuda.device(device):
                props = torch.cuda.get_device_properties(device)
                try:
                    free_memory, total_memory = torch.cuda.mem_get_info()
                    allocated_memory = torch.cuda.memory_allocated()
                except:
                    # Fallback if mem_get_info fails
                    total_memory = props.total_memory
                    allocated_memory = torch.cuda.memory_allocated()
                    free_memory = total_memory - allocated_memory
                
            return {
                "device": device,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory": total_memory,
                "free_memory": free_memory,
                "allocated_memory": allocated_memory,
                "utilization": (total_memory - free_memory) / total_memory * 100,
                "multiprocessor_count": props.multi_processor_count
            }
        except Exception as e:
            return {"error": str(e)}
    
    def optimize_page_size(self, typical_seq_len: int, hidden_dim: int, num_heads: int) -> int:
        """Calculate optimal page size for given model parameters."""
        # Calculate typical tensor size
        head_dim = hidden_dim // num_heads
        single_tensor_size = typical_seq_len * num_heads * head_dim * 2  # fp16
        total_kv_size = single_tensor_size * 2  # key + value
        
        # Aim for 4-8 tensors per page
        optimal_page_size = total_kv_size * 6
        
        # Round to nearest power of 2, constrain to reasonable bounds
        optimal_page_size = max(64 * 1024, min(2 * 1024 * 1024, 
                                             2 ** round(np.log2(optimal_page_size))))
        
        logger.debug(f"Calculated optimal page size: {optimal_page_size // 1024}KB for "
                    f"seq_len={typical_seq_len}, hidden_dim={hidden_dim}")
        
        return optimal_page_size


class ArenaBuffer:
    """Wrapper for arena memory that interfaces with PyTorch."""
    
    def __init__(self, ptr: int, size: int, offset: int, dtype: torch.dtype = torch.float16):
        self.ptr = ptr
        self.size = size
        self.offset = offset
        self.dtype = dtype
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        self.num_elements = size // self.element_size
        
        # Create ctypes array for memory access
        if self.dtype == torch.float16:
            self.c_type = ctypes.c_uint16  # fp16 as uint16
        elif self.dtype == torch.float32:
            self.c_type = ctypes.c_float
        elif self.dtype == torch.int32:
            self.c_type = ctypes.c_int32
        else:
            self.c_type = ctypes.c_uint8
            
        # Create ctypes array from pointer
        try:
            array_type = self.c_type * self.num_elements
            self.c_array = array_type.from_address(ptr)
        except (OSError, ValueError) as e:
            logger.error(f"Failed to create ctypes array from pointer {ptr}: {e}")
            # Create a dummy array as fallback
            self.c_array = (self.c_type * self.num_elements)()
    
    def to_numpy(self) -> np.ndarray:
        """Convert arena memory to numpy array with zero-copy."""
        try:
            # Convert ctypes array to numpy array (zero-copy)
            np_array = np.ctypeslib.as_array(self.c_array)
            
            # Convert dtype if needed
            if self.dtype == torch.float16:
                # Interpret uint16 as float16
                np_array = np_array.view(np.float16)
            elif self.dtype == torch.float32:
                np_array = np_array.astype(np.float32)
            
            return np_array
        except Exception as e:
            logger.warning(f"Failed to create numpy array from buffer: {e}")
            # Create a fallback array
            if self.dtype == torch.float16:
                return np.zeros(self.num_elements, dtype=np.float16)
            elif self.dtype == torch.float32:
                return np.zeros(self.num_elements, dtype=np.float32)
            else:
                return np.zeros(self.num_elements, dtype=np.uint8)
    
    def to_tensor(self, shape: Optional[Tuple[int, ...]] = None, 
                  device: str = 'cpu') -> torch.Tensor:
        """Convert arena memory to PyTorch tensor with zero-copy when possible."""
        try:
            # Get numpy array (zero-copy from arena)
            np_array = self.to_numpy()
            
            # Create PyTorch tensor from numpy (zero-copy)
            tensor = torch.from_numpy(np_array)
            
            # Reshape if needed
            if shape is not None:
                tensor = tensor.view(shape)
            
            # Move to device (this will copy for CUDA)
            device = validate_cuda_device(device)
            if device != 'cpu':
                try:
                    tensor = tensor.to(device, non_blocking=True)
                except Exception as e:
                    logger.warning(f"Failed to move tensor to {device}: {e}, keeping on CPU")
                    tensor = tensor.to('cpu')
            
            return tensor
        except Exception as e:
            logger.error(f"Failed to create tensor from buffer: {e}")
            # Create a fallback tensor
            if shape is not None:
                return torch.zeros(shape, dtype=self.dtype, device='cpu')
            else:
                return torch.zeros(self.num_elements, dtype=self.dtype, device='cpu')


class KVTensor:
    """Enhanced KV tensor with PyTorch integration."""
    
    def __init__(self, offset: int, size: int, seq_len: int, hidden_dim: int, num_heads: int):
        self.offset = offset
        self.size = size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
    
    def get_kv_shapes(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Get shapes for key and value tensors."""
        # Shape: [seq_len, num_heads, head_dim]
        shape = (self.seq_len, self.num_heads, self.head_dim)
        return shape, shape
    
    def __repr__(self):
        return (f"KVTensor(offset={self.offset}, size={self.size}, "
                f"seq_len={self.seq_len}, shape=({self.seq_len}, {self.num_heads}, {self.head_dim}))")


class SequenceArena:
    """Enhanced sequence arena with PyTorch tensor creation."""
    
    def __init__(self, arena_ptr: int):
        if not arena_ptr:
            raise ArenaError("Failed to create sequence arena")
        self._ptr = arena_ptr
        self._tensors = []  # Keep track of allocated tensors
        self.cuda_manager = CUDAMemoryManager()
    
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
        tensor = KVTensor(offset.value, size.value, seq_len, hidden_dim, num_heads)
        self._tensors.append(tensor)
        
        return offset.value, size.value
    
    def create_pytorch_tensors(self, offset: int, size: int, seq_len: int, 
                              hidden_dim: int, num_heads: int,
                              dtype: torch.dtype = torch.float16,
                              device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create PyTorch key and value tensors from arena memory.
        
        Returns:
            Tuple of (key_tensor, value_tensor)
        """
        # Validate device
        device = validate_cuda_device(device)
        
        # Get raw pointer from arena
        ptr = self.get_tensor_ptr(offset, size, seq_len, hidden_dim, num_heads)
        
        # Create arena buffer
        buffer = ArenaBuffer(ptr, size, offset, dtype)
        
        # Calculate shapes
        head_dim = hidden_dim // num_heads
        tensor_shape = (seq_len, num_heads, head_dim)
        single_tensor_elements = seq_len * num_heads * head_dim
        
        try:
            # Get numpy array from arena memory
            np_array = buffer.to_numpy()
            
            # Split into key and value
            key_array = np_array[:single_tensor_elements]
            value_array = np_array[single_tensor_elements:single_tensor_elements*2]
            
            # Create PyTorch tensors
            key_tensor = torch.from_numpy(key_array).view(tensor_shape)
            value_tensor = torch.from_numpy(value_array).view(tensor_shape)
            
            # Move to target device
            if device != 'cpu':
                try:
                    key_tensor = key_tensor.to(device, non_blocking=True)
                    value_tensor = value_tensor.to(device, non_blocking=True)
                except Exception as e:
                    logger.warning(f"Failed to move tensors to {device}: {e}, keeping on CPU")
                    key_tensor = key_tensor.to('cpu')
                    value_tensor = value_tensor.to('cpu')
                    device = 'cpu'
            
            logger.debug(f"Created PyTorch tensors: {tensor_shape} on {device}")
            return key_tensor, value_tensor
        
        except Exception as e:
            logger.error(f"Failed to create PyTorch tensors: {e}")
            # Fallback: create empty tensors
            key_tensor = torch.zeros(tensor_shape, dtype=dtype, device='cpu')
            value_tensor = torch.zeros(tensor_shape, dtype=dtype, device='cpu')
            
            if device != 'cpu':
                try:
                    key_tensor = key_tensor.to(device)
                    value_tensor = value_tensor.to(device)
                except:
                    device = 'cpu'
            
            return key_tensor, value_tensor
    
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
        
        key_tensor, value_tensor = self.create_pytorch_tensors(
            offset, size, seq_len, hidden_dim, num_heads, dtype, device
        )
        
        return key_tensor, value_tensor, (offset, size)
    
    def get_tensor_ptr(self, offset: int, size: int, seq_len: int, 
                      hidden_dim: int, num_heads: int) -> int:
        """Get a raw pointer to tensor data."""
        ptr = _lib.sequence_arena_get_tensor_ptr(
            self._ptr, offset, size, seq_len, hidden_dim, num_heads
        )
        if not ptr:
            raise ArenaError("Failed to get tensor pointer")
        return ptr
    
    def extend_kv_tensor(self, offset: int, size: int, seq_len: int,
                        hidden_dim: int, num_heads: int, new_seq_len: int, 
                        dtype_size: int = 2) -> Tuple[bool, Tuple[int, int]]:
        """
        Extend a KV tensor for a longer sequence.
        
        Returns:
            Tuple of (extended_in_place, (new_offset, new_size))
        """
        extended_in_place = ctypes.c_int()
        new_offset = ctypes.c_size_t()
        new_size = ctypes.c_size_t()
        
        result = _lib.sequence_arena_extend_tensor(
            self._ptr, offset, size, seq_len, hidden_dim, num_heads, 
            new_seq_len, dtype_size,
            ctypes.byref(extended_in_place), ctypes.byref(new_offset), ctypes.byref(new_size)
        )
        
        if result != ARENA_SUCCESS:
            raise ArenaError(f"Failed to extend tensor (error code: {result})")
        
        return bool(extended_in_place.value), (new_offset.value, new_size.value)
    
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
        
        # Validate device
        device = validate_cuda_device(device)
        
        try:
            # Try to extend in arena
            extended_in_place, (new_offset, new_size) = self.extend_kv_tensor(
                offset, size, old_seq_len, hidden_dim, num_heads, new_seq_len, dtype_size
            )
            
            # Create new tensors from extended memory
            new_key, new_value = self.create_pytorch_tensors(
                new_offset, new_size, new_seq_len, hidden_dim, num_heads, dtype, device
            )
            
            if not extended_in_place:
                # Copy old data to new tensor since we had to reallocate
                try:
                    new_key[:old_seq_len] = key_tensor
                    new_value[:old_seq_len] = value_tensor
                    logger.debug(f"Copy-based extension: {old_seq_len} -> {new_seq_len}")
                except Exception as copy_error:
                    logger.warning(f"Failed to copy old data: {copy_error}")
            else:
                # For in-place extension, try to copy data if tensors are different
                try:
                    if new_key.data_ptr() != key_tensor.data_ptr():
                        new_key[:old_seq_len] = key_tensor
                        new_value[:old_seq_len] = value_tensor
                except Exception as copy_error:
                    logger.debug(f"In-place extension, copy failed (expected): {copy_error}")
                logger.debug(f"Zero-copy extension: {old_seq_len} -> {new_seq_len}")
            
            return new_key, new_value, extended_in_place
        
        except Exception as e:
            logger.error(f"Extension failed: {e}, creating new tensors")
            # Fallback: create completely new tensors
            new_key, new_value, _ = self.allocate_and_create_tensors(
                new_seq_len, hidden_dim, num_heads, dtype, device
            )
            
            # Copy old data
            try:
                new_key[:old_seq_len] = key_tensor
                new_value[:old_seq_len] = value_tensor
            except:
                pass  # If copy fails, at least we have new tensors
            
            return new_key, new_value, False
    
    def get_stats(self) -> dict:
        """Get arena statistics."""
        sequence_id = ctypes.c_uint64()
        total_allocated = ctypes.c_size_t()
        num_pages = ctypes.c_size_t()
        utilization = ctypes.c_double()
        
        result = _lib.sequence_arena_get_stats(
            self._ptr, ctypes.byref(sequence_id), ctypes.byref(total_allocated),
            ctypes.byref(num_pages), ctypes.byref(utilization)
        )
        
        if result != ARENA_SUCCESS:
            # Return default stats if call fails
            return {
                'sequence_id': 0,
                'total_allocated': 0,
                'num_pages': 1,
                'utilization': 0.0,
                'num_tensors': len(self._tensors),
                'tensors': [str(t) for t in self._tensors]
            }
        
        return {
            'sequence_id': sequence_id.value,
            'total_allocated': total_allocated.value,
            'num_pages': num_pages.value,
            'utilization': utilization.value,
            'num_tensors': len(self._tensors),
            'tensors': [str(t) for t in self._tensors]
        }


class ArenaKVCacheManager:
    """Enhanced KV cache manager with CUDA optimizations."""
    
    def __init__(self, page_size: Optional[int] = None):
        self.cuda_manager = CUDAMemoryManager()
        
        # Optimize page size if not specified
        if page_size is None:
            page_size = DEFAULT_PAGE_SIZE
            if CUDA_AVAILABLE:
                # Adjust for GPU memory characteristics
                device_info = self.cuda_manager.get_device_info()
                if 'total_memory' in device_info:
                    total_gb = device_info['total_memory'] / (1024**3)
                    if total_gb >= 16:  # High memory GPU
                        page_size = 512 * 1024
                    elif total_gb >= 8:  # Medium memory GPU
                        page_size = 256 * 1024
                    else:  # Lower memory GPU
                        page_size = 128 * 1024
        
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
            device_info = self.cuda_manager.get_device_info()
            logger.info(f"CUDA device: {device_info.get('name', 'Unknown')}")
    
    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            try:
                _lib.kv_cache_manager_free(self._ptr)
            except:
                pass  # Ignore errors during cleanup
    
    def create_sequence_arena(self) -> SequenceArena:
        """Create a new sequence arena."""
        try:
            arena_ptr = _lib.kv_cache_create_sequence_arena(self._ptr)
            return SequenceArena(arena_ptr)
        except Exception as e:
            logger.error(f"Failed to create sequence arena: {e}")
            raise ArenaError(f"Failed to create sequence arena: {e}")
    
    def get_global_stats(self) -> Tuple[int, int]:
        """Get global statistics from the slab pool."""
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
        
        device_info = self.cuda_manager.get_device_info()
        recommendations = []