// src/cuda/tensor.rs - CUDA tensor operations
use super::bindings::*;
use super::error::CudaError;
use super::allocator::CudaPage;
use std::sync::Arc;
use std::ptr::NonNull;
use std::ffi::c_void;

/// CUDA tensor that directly references device memory with real operations
#[derive(Debug)]
pub struct CudaTensor {
    /// Direct pointer to device memory (no copies)
    device_ptr: NonNull<u8>,
    /// Tensor dimensions
    shape: Vec<usize>,
    /// Element size in bytes
    element_size: usize,
    /// Device ID
    device_id: i32,
    /// Reference to parent page (keeps it alive)
    _page_ref: Option<Arc<CudaPage>>,
}

impl CudaTensor {
    /// Create tensor view directly from device memory - NO ALLOCATION
    pub fn from_page(
        page: &Arc<CudaPage>,
        offset: usize,
        shape: Vec<usize>,
        element_size: usize,
    ) -> Result<Self, CudaError> {
        let total_elements: usize = shape.iter().product();
        let size_bytes = total_elements * element_size;
        
        if offset + size_bytes > page.size() {
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }

        // Get direct pointer to device memory
        let device_ptr = unsafe {
            let base_ptr = page.device_ptr() as *mut u8;
            NonNull::new(base_ptr.add(offset)).ok_or(CudaError(CUDA_ERROR_INVALID_VALUE))?
        };

        Ok(CudaTensor {
            device_ptr,
            shape,
            element_size,
            device_id: page.device_id(),
            _page_ref: Some(Arc::clone(page)),
        })
    }

    /// Create tensor from raw device pointer (DANGEROUS)
    pub unsafe fn from_raw_ptr(
        device_ptr: *mut u8,
        shape: Vec<usize>,
        element_size: usize,
        device_id: i32,
    ) -> Result<Self, CudaError> {
        let device_ptr = NonNull::new(device_ptr).ok_or(CudaError(CUDA_ERROR_INVALID_VALUE))?;
        
        Ok(CudaTensor {
            device_ptr,
            shape,
            element_size,
            device_id,
            _page_ref: None,
        })
    }

    /// Get raw device pointer
    pub fn device_ptr(&self) -> *mut c_void {
        self.device_ptr.as_ptr() as *mut c_void
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Get element size
    pub fn element_size(&self) -> usize {
        self.element_size
    }

    /// Zero-copy reshape (just update metadata)
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), CudaError> {
        let new_elements: usize = new_shape.iter().product();
        let old_elements: usize = self.shape.iter().product();
        
        if new_elements != old_elements {
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }

        self.shape = new_shape;
        Ok(())
    }

    /// Zero-copy slice (just adjust pointer and shape)
    pub fn slice(&self, start: usize, end: usize) -> Result<CudaTensor, CudaError> {
        if start >= end || end > self.shape[0] {
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }

        let slice_len = end - start;
        let elements_per_row: usize = self.shape[1..].iter().product();
        let slice_offset = start * elements_per_row * self.element_size;
        
        let mut new_shape = self.shape.clone();
        new_shape[0] = slice_len;

        let new_device_ptr = unsafe {
            NonNull::new(self.device_ptr.as_ptr().add(slice_offset))
                .ok_or(CudaError(CUDA_ERROR_INVALID_VALUE))?
        };

        Ok(CudaTensor {
            device_ptr: new_device_ptr,
            shape: new_shape,
            element_size: self.element_size,
            device_id: self.device_id,
            _page_ref: self._page_ref.clone(),
        })
    }

    /// Copy data from host to this tensor (REAL CUDA memcpy)
    pub fn copy_from_host(&self, host_data: *const c_void) -> Result<(), CudaError> {
        let size_bytes = self.shape.iter().product::<usize>() * self.element_size;
        
        #[cfg(cuda_available)]
        unsafe {
            let result = cudaSetDevice(self.device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            let result = cudaMemcpy(
                self.device_ptr() as *mut c_void,
                host_data,
                size_bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );

            if result != CUDA_SUCCESS {
                Err(CudaError(result))
            } else {
                log::trace!("CUDA tensor copy from host: {} bytes", size_bytes);
                Ok(())
            }
        }
        
        #[cfg(not(cuda_available))]
        {
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }
    
    /// Copy data from this tensor to host (REAL CUDA memcpy)
    pub fn copy_to_host(&self, host_data: *mut c_void) -> Result<(), CudaError> {
        let size_bytes = self.shape.iter().product::<usize>() * self.element_size;
        
        #[cfg(cuda_available)]
        unsafe {
            let result = cudaSetDevice(self.device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            let result = cudaMemcpy(
                host_data,
                self.device_ptr() as *const c_void,
                size_bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );

            if result != CUDA_SUCCESS {
                Err(CudaError(result))
            } else {
                log::trace!("CUDA tensor copy to host: {} bytes", size_bytes);
                Ok(())
            }
        }
        
        #[cfg(not(cuda_available))]
        {
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }

    /// Copy from another tensor (device-to-device)
    pub fn copy_from_tensor(&self, src: &CudaTensor) -> Result<(), CudaError> {
        if self.size_bytes() != src.size_bytes() {
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }

        #[cfg(cuda_available)]
        unsafe {
            let result = cudaSetDevice(self.device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            let result = cudaMemcpy(
                self.device_ptr() as *mut c_void,
                src.device_ptr() as *const c_void,
                self.size_bytes(),
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
            );

            if result != CUDA_SUCCESS {
                Err(CudaError(result))
            } else {
                log::trace!("CUDA tensor device-to-device copy: {} bytes", self.size_bytes());
                Ok(())
            }
        }
        
        #[cfg(not(cuda_available))]
        {
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }

    /// Synchronize tensor operations
    pub fn synchronize(&self) -> Result<(), CudaError> {
        #[cfg(cuda_available)]
        unsafe {
            let result = cudaSetDevice(self.device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }
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
    
    /// Get tensor size in bytes
    pub fn size_bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * self.element_size
    }
    
    /// Get number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
    
    /// Get tensor dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    
    /// Get page reference if available
    pub fn page_ref(&self) -> Option<&Arc<CudaPage>> {
        self._page_ref.as_ref()
    }
}