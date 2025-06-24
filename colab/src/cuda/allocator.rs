// src/cuda/allocator.rs - Bump allocator and page management
use super::bindings::*;
use super::error::CudaError;
use super::stream::CudaStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ptr::NonNull;
use std::ffi::c_void;

/// True bump allocator with REAL CUDA device memory
#[derive(Debug)]
pub struct BumpAllocator {
    /// Current allocation offset (atomic for thread safety)
    current_offset: AtomicUsize,
    /// REAL device memory pointer from cudaMalloc
    device_ptr: NonNull<u8>,
    /// Total page size
    page_size: usize,
    /// Device ID for this allocator
    device_id: i32,
    /// Optional CUDA stream for async operations
    stream: Option<Arc<CudaStream>>,
}

impl BumpAllocator {
    /// Create new bump allocator with REAL CUDA memory allocation
    pub fn new(page_size: usize, device_id: i32) -> Result<Self, CudaError> {
        #[cfg(cuda_available)]
        unsafe {
            // Set device context
            let result = cudaSetDevice(device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            // REAL CUDA ALLOCATION - this actually calls cudaMalloc
            let mut device_ptr = std::ptr::null_mut();
            let result = cudaMalloc(&mut device_ptr, page_size);
            if result != CUDA_SUCCESS {
                log::error!("cudaMalloc failed for {} bytes on device {}: {}", 
                           page_size, device_id, CudaError(result));
                return Err(CudaError(result));
            }

            // Zero initialize the memory asynchronously if possible
            let stream = match CudaStream::new(device_id, true) {
                Ok(stream) => {
                    let stream_ptr = stream.as_ptr();
                    let memset_result = cudaMemsetAsync(device_ptr, 0, page_size, stream_ptr);
                    if memset_result == CUDA_SUCCESS {
                        Some(Arc::new(stream))
                    } else {
                        // Fall back to synchronous memset
                        let _ = cudaMemset(device_ptr, 0, page_size);
                        Some(Arc::new(stream))
                    }
                }
                Err(_) => {
                    // No stream, use synchronous memset
                    let _ = cudaMemset(device_ptr, 0, page_size);
                    None
                }
            };

            log::info!("REAL CUDA allocation: {} bytes on device {}, ptr={:p}", 
                      page_size, device_id, device_ptr);

            Ok(BumpAllocator {
                current_offset: AtomicUsize::new(0),
                device_ptr: NonNull::new(device_ptr as *mut u8).unwrap(),
                page_size,
                device_id,
                stream,
            })
        }
        
        #[cfg(not(cuda_available))]
        {
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }

    /// Pure bump allocation: offset += align(size)
    pub fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        let aligned_size = (size + align - 1) & !(align - 1);
        
        // Atomic bump allocation
        let old_offset = self.current_offset.fetch_add(aligned_size, Ordering::Relaxed);
        
        if old_offset + aligned_size <= self.page_size {
            // Return pointer to allocated region
            let ptr = unsafe { self.device_ptr.as_ptr().add(old_offset) };
            log::trace!("Bump allocated {} bytes at offset {} (device ptr {:p})", 
                       aligned_size, old_offset, ptr);
            Some(NonNull::new(ptr).unwrap())
        } else {
            // Page full - allocation failed
            log::debug!("Bump allocation failed: {} bytes requested, {} available", 
                       aligned_size, self.page_size - old_offset);
            None
        }
    }

    /// Get device pointer at specific offset
    pub fn device_ptr_at(&self, offset: usize) -> Option<*mut c_void> {
        if offset < self.page_size {
            Some(unsafe { self.device_ptr.as_ptr().add(offset) as *mut c_void })
        } else {
            None
        }
    }

    /// Copy data from host to device at offset (REAL CUDA memcpy)
    pub fn copy_from_host(&self, offset: usize, host_data: *const c_void, size: usize) -> Result<(), CudaError> {
        if offset + size > self.page_size {
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }

        #[cfg(cuda_available)]
        unsafe {
            let result = cudaSetDevice(self.device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            let dst = self.device_ptr.as_ptr().add(offset) as *mut c_void;
            
            let result = if let Some(stream) = &self.stream {
                cudaMemcpyAsync(dst, host_data, size, CUDA_MEMCPY_HOST_TO_DEVICE, stream.as_ptr())
            } else {
                cudaMemcpy(dst, host_data, size, CUDA_MEMCPY_HOST_TO_DEVICE)
            };

            if result != CUDA_SUCCESS {
                log::error!("CUDA memcpy failed: {} bytes from host to device offset {}", size, offset);
                Err(CudaError(result))
            } else {
                log::trace!("CUDA memcpy: {} bytes from host to device offset {}", size, offset);
                Ok(())
            }
        }
        
        #[cfg(not(cuda_available))]
        {
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }

    /// Copy data from device to host at offset (REAL CUDA memcpy)
    pub fn copy_to_host(&self, offset: usize, host_data: *mut c_void, size: usize) -> Result<(), CudaError> {
        if offset + size > self.page_size {
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }

        #[cfg(cuda_available)]
        unsafe {
            let result = cudaSetDevice(self.device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            let src = self.device_ptr.as_ptr().add(offset) as *const c_void;
            
            let result = if let Some(stream) = &self.stream {
                cudaMemcpyAsync(host_data, src, size, CUDA_MEMCPY_DEVICE_TO_HOST, stream.as_ptr())
            } else {
                cudaMemcpy(host_data, src, size, CUDA_MEMCPY_DEVICE_TO_HOST)
            };

            if result != CUDA_SUCCESS {
                log::error!("CUDA memcpy failed: {} bytes from device offset {} to host", size, offset);
                Err(CudaError(result))
            } else {
                log::trace!("CUDA memcpy: {} bytes from device offset {} to host", size, offset);
                Ok(())
            }
        }
        
        #[cfg(not(cuda_available))]
        {
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }

    /// Zero-copy device-to-device move within page (REAL CUDA memcpy)
    pub fn device_to_device_copy(&self, src_offset: usize, dst_offset: usize, size: usize) -> Result<(), CudaError> {
        if src_offset + size > self.page_size || dst_offset + size > self.page_size {
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }

        #[cfg(cuda_available)]
        unsafe {
            let result = cudaSetDevice(self.device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            let src = self.device_ptr.as_ptr().add(src_offset) as *const c_void;
            let dst = self.device_ptr.as_ptr().add(dst_offset) as *mut c_void;
            
            let result = if let Some(stream) = &self.stream {
                cudaMemcpyAsync(dst, src, size, CUDA_MEMCPY_DEVICE_TO_DEVICE, stream.as_ptr())
            } else {
                cudaMemcpy(dst, src, size, CUDA_MEMCPY_DEVICE_TO_DEVICE)
            };

            if result != CUDA_SUCCESS {
                log::error!("CUDA device-to-device copy failed: {} bytes", size);
                Err(CudaError(result))
            } else {
                log::debug!("Zero-copy device move: {} bytes from offset {} to {}", size, src_offset, dst_offset);
                Ok(())
            }
        }
        
        #[cfg(not(cuda_available))]
        {
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }

    /// Get current allocation offset
    pub fn current_offset(&self) -> usize {
        self.current_offset.load(Ordering::Relaxed)
    }

    /// Get available space
    pub fn available_space(&self) -> usize {
        let used = self.current_offset();
        self.page_size.saturating_sub(used)
    }

    /// Get utilization ratio
    pub fn utilization(&self) -> f64 {
        self.current_offset() as f64 / self.page_size as f64
    }

    /// Synchronize all operations on this allocator
    pub fn synchronize(&self) -> Result<(), CudaError> {
        if let Some(stream) = &self.stream {
            stream.synchronize()
        } else {
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
    }

    /// Reset allocator (for page reuse)
    pub fn reset(&self) {
        self.current_offset.store(0, Ordering::Relaxed);
        // Optionally zero the memory again
        if let Some(stream) = &self.stream {
            #[cfg(cuda_available)]
            unsafe {
                let _ = cudaMemsetAsync(
                    self.device_ptr.as_ptr() as *mut c_void, 
                    0, 
                    self.page_size, 
                    stream.as_ptr()
                );
            }
        }
    }

    /// Get basic info
    pub fn device_id(&self) -> i32 { self.device_id }
    pub fn page_size(&self) -> usize { self.page_size }
    pub fn device_ptr(&self) -> *mut c_void { self.device_ptr.as_ptr() as *mut c_void }
    
    /// Get stream reference
    pub fn stream(&self) -> Option<&Arc<CudaStream>> {
        self.stream.as_ref()
    }
}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
        #[cfg(cuda_available)]
        unsafe {
            // Set device context
            let _ = cudaSetDevice(self.device_id);
            
            // Synchronize before freeing
            if let Some(stream) = &self.stream {
                let _ = stream.synchronize();
            } else {
                let _ = cudaDeviceSynchronize();
            }
            
            // REAL CUDA FREE - this actually calls cudaFree
            let result = cudaFree(self.device_ptr.as_ptr() as *mut c_void);
            if result != CUDA_SUCCESS {
                log::error!("Failed to free CUDA memory: {}", CudaError(result));
            } else {
                log::info!("Freed CUDA page: {} bytes on device {}, ptr={:p}", 
                          self.page_size, self.device_id, self.device_ptr.as_ptr());
            }
        }
    }
}

unsafe impl Send for BumpAllocator {}
unsafe impl Sync for BumpAllocator {}

/// CUDA page backed by real device memory
#[derive(Debug)]
pub struct CudaPage {
    allocator: BumpAllocator,
    allocation_id: u64,
}

impl CudaPage {
    /// Create new CUDA page with real device memory
    pub fn new(size: usize, device_id: i32) -> Result<Self, CudaError> {
        let allocator = BumpAllocator::new(size, device_id)?;
        let allocation_id = super::NEXT_ALLOCATION_ID.fetch_add(1, Ordering::Relaxed) as u64;
        
        log::debug!("Created CUDA page {}: {} bytes on device {}", allocation_id, size, device_id);
        
        Ok(CudaPage {
            allocator,
            allocation_id,
        })
    }

    /// Bump allocate within this page
    pub fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        self.allocator.allocate(size, align)
    }

    /// Get device pointer at offset
    pub fn device_ptr_at_offset(&self, offset: usize) -> *mut c_void {
        self.allocator.device_ptr_at(offset).unwrap_or(std::ptr::null_mut())
    }

    /// Copy from host to device
    pub fn copy_from_host(&self, host_data: *const c_void, size: usize, offset: usize) -> Result<(), CudaError> {
        self.allocator.copy_from_host(offset, host_data, size)
    }

    /// Copy from device to host
    pub fn copy_to_host(&self, host_data: *mut c_void, size: usize, offset: usize) -> Result<(), CudaError> {
        self.allocator.copy_to_host(offset, host_data, size)
    }

    /// Zero-copy device-to-device copy
    pub fn copy_device_to_device(&self, src_offset: usize, dst_offset: usize, size: usize) -> Result<(), CudaError> {
        self.allocator.device_to_device_copy(src_offset, dst_offset, size)
    }

    /// Synchronize all operations
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.allocator.synchronize()
    }

    /// Reset for reuse
    pub fn reset(&self) {
        self.allocator.reset()
    }

    // Getters
    pub fn size(&self) -> usize { self.allocator.page_size() }
    pub fn device_id(&self) -> i32 { self.allocator.device_id() }
    pub fn device_ptr(&self) -> *mut c_void { self.allocator.device_ptr() }
    pub fn current_offset(&self) -> usize { self.allocator.current_offset() }
    pub fn available_space(&self) -> usize { self.allocator.available_space() }
    pub fn utilization(&self) -> f64 { self.allocator.utilization() }
    pub fn allocation_id(&self) -> u64 { self.allocation_id }
    
    /// Get allocator reference
    pub fn allocator(&self) -> &BumpAllocator {
        &self.allocator
    }
    
    pub fn is_ready(&self) -> Result<bool, CudaError> { 
        // Check if any async operations are complete
        self.synchronize()?;
        Ok(true)
    }
}