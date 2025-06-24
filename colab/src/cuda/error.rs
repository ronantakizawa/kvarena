// src/cuda/error.rs - CUDA error handling
use super::bindings::*;

#[derive(Debug, Clone, Copy)]
pub struct CudaError(pub i32);

impl CudaError {
    pub fn from_code(code: i32) -> Self {
        CudaError(code)
    }
    
    pub fn code(&self) -> i32 {
        self.0
    }
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(cuda_available)]
        unsafe {
            let c_str = cudaGetErrorString(self.0);
            if c_str.is_null() {
                write!(f, "CUDA Error {}: Unknown error", self.0)
            } else {
                let cstr = std::ffi::CStr::from_ptr(c_str);
                write!(f, "CUDA Error {}: {}", self.0, cstr.to_string_lossy())
            }
        }
        
        #[cfg(not(cuda_available))]
        write!(f, "CUDA Error {}: CUDA not available", self.0)
    }
}

impl std::error::Error for CudaError {}

/// Utility function for CUDA operations
pub fn check_cuda_error(operation: &str) -> Result<(), CudaError> {
    #[cfg(cuda_available)]
    unsafe {
        let error = cudaGetLastError();
        if error != CUDA_SUCCESS {
            log::error!("CUDA error in {}: {}", operation, CudaError(error));
            Err(CudaError(error))
        } else {
            Ok(())
        }
    }
    
    #[cfg(not(cuda_available))]
    {
        Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
    }
}