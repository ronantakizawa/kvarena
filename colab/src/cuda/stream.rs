// src/cuda/stream.rs - CUDA stream management
use super::bindings::*;
use super::error::CudaError;
use std::ptr::NonNull;
use std::ffi::c_void;

/// CUDA stream for asynchronous operations
#[derive(Debug)]
pub struct CudaStream {
    stream: NonNull<c_void>,
    device_id: i32,
}

impl CudaStream {
    /// Create a new CUDA stream
    pub fn new(device_id: i32, non_blocking: bool) -> Result<Self, CudaError> {
        #[cfg(cuda_available)]
        unsafe {
            let result = cudaSetDevice(device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            let mut stream = std::ptr::null_mut();
            let result = if non_blocking {
                cudaStreamCreateWithFlags(&mut stream, CUDA_STREAM_NON_BLOCKING)
            } else {
                cudaStreamCreate(&mut stream)
            };

            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            Ok(CudaStream {
                stream: NonNull::new(stream).unwrap(),
                device_id,
            })
        }
        
        #[cfg(not(cuda_available))]
        {
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.stream.as_ptr()
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        #[cfg(cuda_available)]
        unsafe {
            let result = cudaStreamSynchronize(self.stream.as_ptr());
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

    pub fn is_complete(&self) -> Result<bool, CudaError> {
        #[cfg(cuda_available)]
        unsafe {
            let result = cudaStreamQuery(self.stream.as_ptr());
            match result {
                CUDA_SUCCESS => Ok(true),
                CUDA_ERROR_NOT_READY => Ok(false),
                _ => Err(CudaError(result)),
            }
        }
        
        #[cfg(not(cuda_available))]
        {
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }
    
    /// Get device ID for this stream
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        #[cfg(cuda_available)]
        unsafe {
            let _ = cudaSetDevice(self.device_id);
            let _ = cudaStreamDestroy(self.stream.as_ptr());
        }
    }
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}