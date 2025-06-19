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

# CUDA availability check
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    logger.info(f"CUDA available with {torch.cuda.device_count()} device(s)")
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


class CUDAMemoryManager:
    """CUDA memory management utilities."""
    
    def __init__(self):
        self.device_count = 0
        self.current_device = 0
        
        if CUDA_AVAILABLE:
            self.device_count = torch.cuda.device_count()
            self.current_device = torch.cuda.current_device()
            logger.info(f"CUDA manager initialized with {self.device_count} device(s)")
        
    def get_device_info(self, device: Optional[int] = None) -> dict:
        """Get CUDA device information."""
        if not CUDA_AVAILABLE:
            return {"error": "CUDA not available"}
        
        device = device or self.current_device
        
        try:
            with torch.cuda.device(device):
                props = torch.cuda.get_device_properties(device)
                free_memory, total_memory = torch.cuda.mem_get_info()
                allocated_memory = torch.cuda.memory_allocated()
                
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
        array_type = self.c_type * self.num_elements
        self.c_array = array_type.from_address(ptr)
    
    def to_numpy(self) -> np.ndarray:
        """Convert arena memory to numpy array with zero-copy."""
        # Convert ctypes array to numpy array (zero-copy)
        np_array = np.ctypeslib.as_array(self.c_array)
        
        # Convert dtype if needed
        if self.dtype == torch.float16:
            # Interpret uint16 as float16
            np_array = np_array.view(np.float16)
        elif self.dtype == torch.float32:
            np_array = np_array.astype(np.float32)
        
        return np_array
    
    def to_tensor(self, shape: Optional[Tuple[int, ...]] = None, 
                  device: str = 'cpu') -> torch.Tensor:
        """Convert arena memory to PyTorch tensor with zero-copy when possible."""
        # Get numpy array (zero-copy from arena)
        np_array = self.to_numpy()
        
        # Create PyTorch tensor from numpy (zero-copy)
        tensor = torch.from_numpy(np_array)
        
        # Reshape if needed
        if shape is not None:
            tensor = tensor.view(shape)
        
        # Move to device (this will copy for CUDA)
        if device != 'cpu':
            tensor = tensor.to(device, non_blocking=True)
        
        return tensor


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
        # Get raw pointer from arena
        ptr = self.get_tensor_ptr(offset, size, seq_len, hidden_dim, num_heads)
        
        # Create arena buffer
        buffer = ArenaBuffer(ptr, size, offset, dtype)
        
        # Calculate shapes
        head_dim = hidden_dim // num_heads
        tensor_shape = (seq_len, num_heads, head_dim)
        single_tensor_elements = seq_len * num_heads * head_dim
        
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
            key_tensor = key_tensor.to(device, non_blocking=True)
            value_tensor = value_tensor.to(device, non_blocking=True)
        
        logger.debug(f"Created PyTorch tensors: {tensor_shape} on {device}")
        return key_tensor, value_tensor
    
    def allocate_and_create_tensors(self, seq_len: int, hidden_dim: int, num_heads: int,
                                   dtype: torch.dtype = torch.float16,
                                   device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        Allocate arena memory and create PyTorch tensors in one call.
        
        Returns:
            Tuple of (key_tensor, value_tensor, (offset, size))
        """
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
            new_key[:old_seq_len] = key_tensor
            new_value[:old_seq_len] = value_tensor
            logger.debug(f"Copy-based extension: {old_seq_len} -> {new_seq_len}")
        else:
            # For in-place extension, the old data is already in the right place
            # but we may need to copy it to the new tensor view
            try:
                # Try to copy data if tensors are different
                if new_key.data_ptr() != key_tensor.data_ptr():
                    new_key[:old_seq_len] = key_tensor
                    new_value[:old_seq_len] = value_tensor
            except:
                # If copying fails, the tensors might already share memory
                pass
            logger.debug(f"Zero-copy extension: {old_seq_len} -> {new_seq_len}")
        
        return new_key, new_value, extended_in_place
    
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
            raise ArenaError(f"Failed to get stats (error code: {result})")
        
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
        
        self._ptr = _lib.kv_cache_manager_new(page_size)
        if not self._ptr:
            raise ArenaError("Failed to create KV cache manager")
        
        self.page_size = page_size
        logger.info(f"Created ArenaKVCacheManager with page_size={page_size//1024}KB")
        
        if CUDA_AVAILABLE:
            device_info = self.cuda_manager.get_device_info()
            logger.info(f"CUDA device: {device_info.get('name', 'Unknown')}")
    
    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            _lib.kv_cache_manager_free(self._ptr)
    
    def create_sequence_arena(self) -> SequenceArena:
        """Create a new sequence arena."""
        arena_ptr = _lib.kv_cache_create_sequence_arena(self._ptr)
        return SequenceArena(arena_ptr)
    
    def get_global_stats(self) -> Tuple[int, int]:
        """Get global statistics from the slab pool."""
        allocated = ctypes.c_size_t()
        recycled = ctypes.c_size_t()
        
        result = _lib.kv_cache_manager_get_global_stats(
            self._ptr, ctypes.byref(allocated), ctypes.byref(recycled)
        )
        
        if result != ARENA_SUCCESS:
            raise ArenaError(f"Failed to get global stats (error code: {result})")
        
        return allocated.value, recycled.value
    
    def get_device_recommendations(self) -> dict:
        """Get device-specific recommendations for optimal performance."""
        if not CUDA_AVAILABLE:
            return {"device": "CPU", "recommendations": ["Use CPU-optimized page sizes"]}
        
        device_info = self.cuda_manager.get_device_info()
        recommendations = []
        
        if 'total_memory' in device_info:
            total_gb = device_info['total_memory'] / (1024**3)
            
            if total_gb >= 24:
                recommendations.extend([
                    "Use 512KB-1MB page sizes for large models",
                    "Enable memory pooling for batch processing",
                    "Consider multi-GPU distribution for very large models"
                ])
            elif total_gb >= 12:
                recommendations.extend([
                    "Use 256KB-512KB page sizes",
                    "Monitor memory usage for large batches",
                    "Enable gradient checkpointing if needed"
                ])
            else:
                recommendations.extend([
                    "Use 128KB-256KB page sizes",
                    "Limit batch sizes to avoid OOM",
                    "Consider model sharding for large models"
                ])
        
        if 'compute_capability' in device_info:
            cc = device_info['compute_capability']
            if cc >= "8.0":
                recommendations.append("Enable Tensor Core optimizations")
            elif cc >= "7.0":
                recommendations.append("Use mixed precision for better performance")
        
        return {
            "device_info": device_info,
            "recommendations": recommendations,
            "optimal_page_size": self.cuda_manager.optimize_page_size(512, 4096, 32)
        }


# Utility functions
def get_default_page_size() -> int:
    """Get the default page size."""
    return _lib.arena_get_default_page_size()


def get_alignment() -> int:
    """Get the memory alignment requirement."""
    return _lib.arena_get_alignment()


def align_size(size: int) -> int:
    """Align a size to the required boundary."""
    return _lib.arena_align_size(size)


def create_optimized_manager(model_config: Optional[dict] = None) -> ArenaKVCacheManager:
    """
    Create an optimized arena manager based on model configuration.
    
    Args:
        model_config: Dict with keys like 'hidden_size', 'num_heads', 'typical_seq_len'
    """
    if model_config is None:
        return ArenaKVCacheManager()
    
    # Calculate optimal page size based on model
    hidden_size = model_config.get('hidden_size', 4096)
    num_heads = model_config.get('num_heads', 32)
    typical_seq_len = model_config.get('typical_seq_len', 512)
    
    cuda_manager = CUDAMemoryManager()
    optimal_page_size = cuda_manager.optimize_page_size(typical_seq_len, hidden_size, num_heads)
    
    logger.info(f"Creating optimized manager for model: hidden_size={hidden_size}, "
               f"num_heads={num_heads}, typical_seq_len={typical_seq_len}")
    logger.info(f"Optimal page size: {optimal_page_size//1024}KB")
    
    return ArenaKVCacheManager(optimal_page_size)


if __name__ == "__main__":
    # Example usage and testing
    print("🚀 Enhanced Arena KV-Cache with CUDA Support")
    print("=" * 50)
    
    try:
        # Test basic functionality
        manager = ArenaKVCacheManager()
        arena = manager.create_sequence_arena()
        
        device = 'cuda' if CUDA_AVAILABLE else 'cpu'
        print(f"Using device: {device}")
        
        # Test tensor creation
        key, value, (offset, size) = arena.allocate_and_create_tensors(
            seq_len=128, hidden_dim=512, num_heads=8, 
            dtype=torch.float16, device=device
        )
        
        print(f"✅ Created tensors: key={key.shape}, value={value.shape}")
        print(f"   Device: {key.device}, dtype: {key.dtype}")
        print(f"   Arena allocation: offset={offset}, size={size}")
        
        # Test extension
        new_key, new_value, extended = arena.extend_pytorch_tensors(
            key, value, offset, size, new_seq_len=256
        )
        print(f"✅ Extended tensors: {key.shape} -> {new_key.shape}")
        print(f"   Zero-copy extension: {extended}")
        
        # Get stats
        stats = arena.get_stats()
        print(f"✅ Arena stats: {stats}")
        
        # Device recommendations
        recommendations = manager.get_device_recommendations()
        print(f"\n💡 Device recommendations:")
        for rec in recommendations['recommendations']:
            print(f"   • {rec}")
        
        print("\n✅ Enhanced Arena KV-Cache ready for production!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()