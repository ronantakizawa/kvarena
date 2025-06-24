// src/cuda/raw.rs - Direct low-level CUDA API wrappers
use super::bindings::*;
use super::error::CudaError;
use std::ffi::c_void;

/// Set CUDA device
pub fn set_device(device_id: i32) -> Result<(), CudaError> {
    #[cfg(cuda_available)]
    unsafe {
        let result = cudaSetDevice(device_id);
        if result != CUDA_SUCCESS {
            Err(CudaError(result))
        } else {
            Ok(())
        }
    }
    
    #[cfg(not(cuda_available))]
    {
        Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
    }
}

/// Get current CUDA device
pub fn get_device() -> Result<i32, CudaError> {
    #[cfg(cuda_available)]
    unsafe {
        let mut device = 0;
        let result = cudaGetDevice(&mut device);
        if result != CUDA_SUCCESS {
            Err(CudaError(result))
        } else {
            Ok(device)
        }
    }
    
    #[cfg(not(cuda_available))]
    {
        Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
    }
}

/// Get device count
pub fn get_device_count() -> Result<i32, CudaError> {
    #[cfg(cuda_available)]
    unsafe {
        let mut count = 0;
        let result = cudaGetDeviceCount(&mut count);
        if result != CUDA_SUCCESS {
            Err(CudaError(result))
        } else {
            Ok(count)
        }
    }
    
    #[cfg(not(cuda_available))]
    {
        Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
    }
}

/// Raw memory allocation
pub fn malloc(size: usize) -> Result<*mut c_void, CudaError> {
    #[cfg(cuda_available)]
    unsafe {
        let mut ptr = std::ptr::null_mut();
        let result = cudaMalloc(&mut ptr, size);
        if result != CUDA_SUCCESS {
            Err(CudaError(result))
        } else {
            Ok(ptr)
        }
    }
    
    #[cfg(not(cuda_available))]
    {
        Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
    }
}

/// Raw memory deallocation
pub fn free(ptr: *mut c_void) -> Result<(), CudaError> {
    #[cfg(cuda_available)]
    unsafe {
        let result = cudaFree(ptr);
        if result != CUDA_SUCCESS {
            Err(CudaError(result))
        } else {
            Ok(())
        }
    }
    
    #[cfg(not(cuda_available))]
    {
        Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
    }
}

/// Raw memory copy
pub fn memcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> Result<(), CudaError> {
    #[cfg(cuda_available)]
    unsafe {
        let result = cudaMemcpy(dst, src, count, kind);
        if result != CUDA_SUCCESS {
            Err(CudaError(result))
        } else {
            Ok(())
        }
    }
    
    #[cfg(not(cuda_available))]
    {
        Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
    }
}

/// Raw memory set
pub fn memset(ptr: *mut c_void, value: i32, count: usize) -> Result<(), CudaError> {
    #[cfg(cuda_available)]
    unsafe {
        let result = cudaMemset(ptr, value, count);
        if result != CUDA_SUCCESS {
            Err(CudaError(result))
        } else {
            Ok(())
        }
    }
    
    #[cfg(not(cuda_available))]
    {
        Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
    }
}

/// Device synchronize
pub fn device_synchronize() -> Result<(), CudaError> {
    #[cfg(cuda_available)]
    unsafe {
        let result = cudaDeviceSynchronize();
        if result != CUDA_SUCCESS {
            Err(CudaError(result))
        } else {
            Ok(())
        }
    }
    
    #[cfg(not(cuda_available))]
    {
        Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
    }
}

/// Get memory info
pub fn mem_get_info() -> Result<(usize, usize), CudaError> {
    #[cfg(cuda_available)]
    unsafe {
        let mut free = 0;
        let mut total = 0;
        let result = cudaMemGetInfo(&mut free, &mut total);
        if result != CUDA_SUCCESS {
            Err(CudaError(result))
        } else {
            Ok((free, total))
        }
    }
    
    #[cfg(not(cuda_available))]
    {
        Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
    }
}

/// Get last error
pub fn get_last_error() -> CudaError {
    #[cfg(cuda_available)]
    unsafe {
        CudaError(cudaGetLastError())
    }
    
    #[cfg(not(cuda_available))]
    {
        CudaError(CUDA_ERROR_NOT_INITIALIZED)
    }
}