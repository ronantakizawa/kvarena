// src/ffi.rs - C FFI bindings for Python integration
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Arc;
use crate::{KVCacheManager, SequenceArena, ArenaStats, AllocError};

// Opaque pointers for C FFI
pub struct CKVCacheManager(KVCacheManager);
pub struct CSequenceArena(SequenceArena);

// Error codes
pub const ARENA_SUCCESS: i32 = 0;
pub const ARENA_ERROR_ALLOC: i32 = -1;
pub const ARENA_ERROR_INVALID_PARAM: i32 = -2;

/// Create a new KV cache manager
#[no_mangle]
pub extern "C" fn arena_cache_manager_new(page_size: usize) -> *mut CKVCacheManager {
    let manager = KVCacheManager::new(page_size);
    Box::into_raw(Box::new(CKVCacheManager(manager)))
}

/// Destroy a KV cache manager
#[no_mangle]
pub extern "C" fn arena_cache_manager_free(manager: *mut CKVCacheManager) {
    if !manager.is_null() {
        unsafe {
            let _ = Box::from_raw(manager);
        }
    }
}

/// Create a new sequence arena
#[no_mangle]
pub extern "C" fn arena_create_sequence(
    manager: *mut CKVCacheManager,
    arena_out: *mut *mut CSequenceArena
) -> i32 {
    if manager.is_null() || arena_out.is_null() {
        return ARENA_ERROR_INVALID_PARAM;
    }
    
    let manager = unsafe { &mut (*manager).0 };
    
    match manager.create_sequence_arena() {
        Ok(arena) => {
            unsafe {
                *arena_out = Box::into_raw(Box::new(CSequenceArena(arena)));
            }
            ARENA_SUCCESS
        }
        Err(_) => ARENA_ERROR_ALLOC,
    }
}

/// Destroy a sequence arena
#[no_mangle]
pub extern "C" fn arena_sequence_free(arena: *mut CSequenceArena) {
    if !arena.is_null() {
        unsafe {
            let _ = Box::from_raw(arena);
        }
    }
}

/// Allocate a KV tensor
#[no_mangle]
pub extern "C" fn arena_allocate_kv_tensor(
    arena: *mut CSequenceArena,
    seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
    dtype_size: usize,
    offset_out: *mut usize,
    size_out: *mut usize
) -> i32 {
    if arena.is_null() || offset_out.is_null() || size_out.is_null() {
        return ARENA_ERROR_INVALID_PARAM;
    }
    
    let arena = unsafe { &mut (*arena).0 };
    
    match arena.allocate_kv_tensor(seq_len, hidden_dim, num_heads, dtype_size) {
        Ok(tensor) => {
            unsafe {
                *offset_out = tensor.offset();
                *size_out = tensor.size();
            }
            ARENA_SUCCESS
        }
        Err(_) => ARENA_ERROR_ALLOC,
    }
}

/// Get tensor pointer
#[no_mangle]
pub extern "C" fn arena_get_tensor_ptr(
    arena: *mut CSequenceArena,
    offset: usize,
    size: usize,
    seq_len: usize,
    hidden_dim: usize,
    num_heads: usize
) -> *mut c_void {
    if arena.is_null() {
        return std::ptr::null_mut();
    }
    
    let arena = unsafe { &(*arena).0 };
    let tensor = crate::KVTensor::new(offset, size, seq_len, hidden_dim, num_heads);
    arena.get_tensor_ptr(&tensor) as *mut c_void
}

/// Get arena statistics
#[no_mangle] 
pub extern "C" fn arena_get_stats(
    arena: *mut CSequenceArena,
    total_allocated: *mut usize,
    num_pages: *mut usize,
    utilization: *mut f64
) -> i32 {
    if arena.is_null() || total_allocated.is_null() || num_pages.is_null() || utilization.is_null() {
        return ARENA_ERROR_INVALID_PARAM;
    }
    
    let arena = unsafe { &(*arena).0 };
    let stats = arena.stats();
    
    unsafe {
        *total_allocated = stats.total_allocated;
        *num_pages = stats.num_pages;
        *utilization = stats.current_page_utilization;
    }
    
    ARENA_SUCCESS
}

// Python bindings using PyO3 (optional feature)
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyclass(name = "ArenaKVCacheManager")]
pub struct PyArenaKVCacheManager {
    inner: KVCacheManager,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyArenaKVCacheManager {
    #[new]
    fn new(page_size: Option<usize>) -> Self {
        Self {
            inner: KVCacheManager::new(page_size.unwrap_or(256 * 1024)),
        }
    }
    
    fn create_sequence_arena(&self) -> PyResult<PySequenceArena> {
        match self.inner.create_sequence_arena() {
            Ok(arena) => Ok(PySequenceArena { inner: Some(arena) }),
            Err(_) => Err(pyo3::exceptions::PyMemoryError::new_err("Failed to allocate arena")),
        }
    }
    
    fn global_stats(&self) -> (usize, usize) {
        self.inner.global_stats()
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "ArenaSequenceArena")]
pub struct PySequenceArena {
    inner: Option<SequenceArena>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySequenceArena {
    fn allocate_kv_tensor(
        &mut self,
        seq_len: usize,
        hidden_dim: usize,
        num_heads: usize,
        dtype_size: usize,
    ) -> PyResult<(usize, usize)> {
        match &mut self.inner {
            Some(arena) => {
                match arena.allocate_kv_tensor(seq_len, hidden_dim, num_heads, dtype_size) {
                    Ok(tensor) => Ok((tensor.offset(), tensor.size())),
                    Err(_) => Err(pyo3::exceptions::PyMemoryError::new_err("Allocation failed")),
                }
            }
            None => Err(pyo3::exceptions::PyRuntimeError::new_err("Arena not initialized")),
        }
    }
    
    fn get_tensor_ptr(&self, offset: usize, size: usize, seq_len: usize, hidden_dim: usize, num_heads: usize) -> PyResult<usize> {
        match &self.inner {
            Some(arena) => {
                let tensor = crate::KVTensor::new(offset, size, seq_len, hidden_dim, num_heads);
                Ok(arena.get_tensor_ptr(&tensor) as usize)
            }
            None => Err(pyo3::exceptions::PyRuntimeError::new_err("Arena not initialized")),
        }
    }
    
    fn stats(&self) -> PyResult<(u64, usize, usize, f64)> {
        match &self.inner {
            Some(arena) => {
                let stats = arena.stats();
                Ok((
                    stats.sequence_id,
                    stats.total_allocated,
                    stats.num_pages,
                    stats.current_page_utilization,
                ))
            }
            None => Err(pyo3::exceptions::PyRuntimeError::new_err("Arena not initialized")),
        }
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn arena_kv_cache(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyArenaKVCacheManager>()?;
    m.add_class::<PySequenceArena>()?;
    Ok(())
}

// Benchmark functions
#[no_mangle]
pub extern "C" fn arena_benchmark_allocation(
    page_size: usize,
    num_sequences: usize,
    avg_seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
    dtype_size: usize,
    time_ns_out: *mut u64,
    memory_bytes_out: *mut usize
) -> i32 {
    use std::time::Instant;
    
    if time_ns_out.is_null() || memory_bytes_out.is_null() {
        return ARENA_ERROR_INVALID_PARAM;
    }
    
    let manager = KVCacheManager::new(page_size);
    let start = Instant::now();
    let mut total_memory = 0;
    
    for i in 0..num_sequences {
        let seq_len = avg_seq_len + (i % 100); // Vary sequence length
        
        match manager.create_sequence_arena() {
            Ok(mut arena) => {
                match arena.allocate_kv_tensor(seq_len, hidden_dim, num_heads, dtype_size) {
                    Ok(tensor) => {
                        total_memory += tensor.size();
                    }
                    Err(_) => return ARENA_ERROR_ALLOC,
                }
            }
            Err(_) => return ARENA_ERROR_ALLOC,
        }
    }
    
    let duration = start.elapsed();
    
    unsafe {
        *time_ns_out = duration.as_nanos() as u64;
        *memory_bytes_out = total_memory;
    }
    
    ARENA_SUCCESS
}

// CUDA integration (optional)
#[cfg(feature = "cuda")]
pub mod cuda {
    use super::*;
    use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};
    
    #[no_mangle]
    pub extern "C" fn arena_cuda_allocate_tensor(
        arena: *mut CSequenceArena,
        seq_len: usize,
        hidden_dim: usize,
        num_heads: usize,
        dtype_size: usize,
        device_ptr_out: *mut *mut c_void
    ) -> i32 {
        if arena.is_null() || device_ptr_out.is_null() {
            return ARENA_ERROR_INVALID_PARAM;
        }
        
        // CUDA allocation would go here
        // This is a simplified example
        ARENA_SUCCESS
    }
    
    #[no_mangle]
    pub extern "C" fn arena_cuda_copy_to_device(
        src: *const c_void,
        dst: *mut c_void,
        size: usize
    ) -> i32 {
        // CUDA memcpy would go here
        ARENA_SUCCESS
    }
}