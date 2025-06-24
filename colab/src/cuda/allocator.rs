// src/cuda/allocator.rs - Fixed bump allocator with proper alignment and memory safety
use super::bindings::*;
use super::error::CudaError;
use super::stream::CudaStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ptr::NonNull;
use std::ffi::c_void;

/// Proper alignment constants for CUDA memory
const CUDA_MEMORY_ALIGNMENT: usize = 256; // 256-byte alignment for optimal GPU performance
const MIN_ALLOCATION_SIZE: usize = 32;    // Minimum allocation size to prevent tiny allocations

/// Safe alignment function that prevents overflow and ensures CUDA compatibility
fn align_size(size: usize, alignment: usize) -> Result<usize, CudaError> {
    // CRITICAL: Ensure alignment is a power of 2 and at least 256 for CUDA
    let cuda_alignment = alignment.max(256).next_power_of_two();
    
    if size == 0 {
        return Ok(cuda_alignment);
    }
    
    // Check for potential overflow
    if size > usize::MAX - cuda_alignment {
        return Err(CudaError(CUDA_ERROR_OUT_OF_MEMORY));
    }
    
    // Perform alignment calculation
    let aligned = (size + cuda_alignment - 1) & !(cuda_alignment - 1);
    
    // Ensure minimum allocation size and verify alignment
    let final_size = aligned.max(MIN_ALLOCATION_SIZE);
    
    // Verify the result is properly aligned
    if final_size % cuda_alignment != 0 {
        return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
    }
    
    Ok(final_size)
}

/// True bump allocator with REAL CUDA device memory and proper alignment
#[derive(Debug)]
pub struct BumpAllocator {
    /// Current allocation offset (atomic for thread safety)
    current_offset: AtomicUsize,
    /// REAL device memory pointer from cudaMalloc
    device_ptr: NonNull<u8>,
    /// Total page size (aligned)
    page_size: usize,
    /// Device ID for this allocator
    device_id: i32,
    /// Optional CUDA stream for async operations
    stream: Option<Arc<CudaStream>>,
    /// Track if this allocator has been properly initialized
    initialized: bool,
}

impl BumpAllocator {
    /// Create new bump allocator with REAL CUDA memory allocation and proper alignment
    pub fn new(page_size: usize, device_id: i32) -> Result<Self, CudaError> {
        // Align page size to CUDA memory alignment
        let aligned_page_size = align_size(page_size, CUDA_MEMORY_ALIGNMENT)?;
        
        #[cfg(cuda_available)]
        unsafe {
            // Set device context first
            let result = cudaSetDevice(device_id);
            if result != CUDA_SUCCESS {
                log::error!("Failed to set CUDA device {}: {}", device_id, CudaError(result));
                return Err(CudaError(result));
            }

            // REAL CUDA ALLOCATION - this actually calls cudaMalloc with proper alignment
            let mut device_ptr = std::ptr::null_mut();
            let result = cudaMalloc(&mut device_ptr, aligned_page_size);
            if result != CUDA_SUCCESS {
                log::error!("cudaMalloc failed for {} bytes on device {}: {}", 
                           aligned_page_size, device_id, CudaError(result));
                return Err(CudaError(result));
            }

            // Verify pointer alignment
            let ptr_addr = device_ptr as usize;
            if ptr_addr % CUDA_MEMORY_ALIGNMENT != 0 {
                log::warn!("CUDA allocated pointer {:p} is not properly aligned", device_ptr);
                // Continue anyway as CUDA should handle this
            }

            // Zero initialize the memory safely
            let memset_result = cudaMemset(device_ptr, 0, aligned_page_size);
            if memset_result != CUDA_SUCCESS {
                log::warn!("Failed to zero-initialize CUDA memory: {}", CudaError(memset_result));
                // Continue without zeroing - not critical
            }

            // Try to create stream for async operations (optional)
            let stream = match CudaStream::new(device_id, true) {
                Ok(stream) => {
                    log::debug!("Created async stream for CUDA allocator on device {}", device_id);
                    Some(Arc::new(stream))
                }
                Err(e) => {
                    log::debug!("No async stream for device {} (using sync operations): {}", device_id, e);
                    None
                }
            };

            log::info!("✓ CUDA allocation successful: {} bytes (aligned from {}) on device {}, ptr={:p}", 
                      aligned_page_size, page_size, device_id, device_ptr);

            Ok(BumpAllocator {
                current_offset: AtomicUsize::new(0),
                device_ptr: NonNull::new(device_ptr as *mut u8).unwrap(),
                page_size: aligned_page_size,
                device_id,
                stream,
                initialized: true,
            })
        }
        
        #[cfg(not(cuda_available))]
        {
            log::error!("CUDA not available - cannot create allocator");
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }

    /// Pure bump allocation with proper alignment: offset += align(size)
    pub fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        if !self.initialized {
            log::error!("Allocator not properly initialized");
            return None;
        }

        // CRITICAL: Force minimum 256-byte alignment for CUDA
        let cuda_align = align.max(CUDA_MEMORY_ALIGNMENT);
        
        // Use safe alignment function
        let aligned_size = match align_size(size, cuda_align) {
            Ok(size) => size,
            Err(e) => {
                log::error!("Alignment calculation failed: {}", e);
                return None;
            }
        };
        
        // Atomic bump allocation with overflow check
        let old_offset = self.current_offset.fetch_add(aligned_size, Ordering::Relaxed);
        
        // Check for overflow and bounds
        if old_offset > self.page_size || aligned_size > self.page_size - old_offset {
            // Revert the offset increment
            self.current_offset.fetch_sub(aligned_size, Ordering::Relaxed);
            log::debug!("Bump allocation failed: {} bytes requested, {} available (page size: {})", 
                       aligned_size, self.page_size.saturating_sub(old_offset), self.page_size);
            return None;
        }
        
        // Calculate aligned pointer with verification
        let ptr = unsafe { self.device_ptr.as_ptr().add(old_offset) };
        
        // CRITICAL: Verify both base alignment and final alignment
        let base_addr = self.device_ptr.as_ptr() as usize;
        let ptr_addr = ptr as usize;
        
        // Check base pointer alignment (should be aligned from cudaMalloc)
        if base_addr % CUDA_MEMORY_ALIGNMENT != 0 {
            log::error!("Base device pointer not aligned: {:p} (expected {} bytes)", 
                       self.device_ptr.as_ptr(), CUDA_MEMORY_ALIGNMENT);
            self.current_offset.fetch_sub(aligned_size, Ordering::Relaxed);
            return None;
        }
        
        // Check final pointer alignment
        if ptr_addr % cuda_align != 0 {
            log::error!("Allocated pointer not aligned: {:p} (expected {} bytes)", ptr, cuda_align);
            // Revert allocation
            self.current_offset.fetch_sub(aligned_size, Ordering::Relaxed);
            return None;
        }
        
        log::trace!("✓ Bump allocated {} bytes at offset {} (device ptr {:p}, aligned to {})", 
                   aligned_size, old_offset, ptr, cuda_align);
        
        unsafe { Some(NonNull::new_unchecked(ptr)) }
    }

    /// Get device pointer at specific offset with bounds checking
    pub fn device_ptr_at(&self, offset: usize) -> Option<*mut c_void> {
        if offset < self.page_size {
            Some(unsafe { self.device_ptr.as_ptr().add(offset) as *mut c_void })
        } else {
            log::error!("Offset {} exceeds page size {}", offset, self.page_size);
            None
        }
    }

    /// Copy data from host to device at offset with bounds checking (REAL CUDA memcpy)
    pub fn copy_from_host(&self, offset: usize, host_data: *const c_void, size: usize) -> Result<(), CudaError> {
        if !self.initialized {
            return Err(CudaError(CUDA_ERROR_NOT_INITIALIZED));
        }

        if host_data.is_null() {
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }

        if offset.saturating_add(size) > self.page_size {
            log::error!("Copy would exceed page bounds: offset={}, size={}, page_size={}", 
                       offset, size, self.page_size);
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
                log::error!("CUDA memcpy failed: {} bytes from host to device offset {}: {}", 
                           size, offset, CudaError(result));
                Err(CudaError(result))
            } else {
                log::trace!("✓ CUDA memcpy: {} bytes from host to device offset {}", size, offset);
                Ok(())
            }
        }
        
        #[cfg(not(cuda_available))]
        {
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }

    /// Copy data from device to host at offset with bounds checking (REAL CUDA memcpy)
    pub fn copy_to_host(&self, offset: usize, host_data: *mut c_void, size: usize) -> Result<(), CudaError> {
        if !self.initialized {
            return Err(CudaError(CUDA_ERROR_NOT_INITIALIZED));
        }

        if host_data.is_null() {
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }

        if offset.saturating_add(size) > self.page_size {
            log::error!("Copy would exceed page bounds: offset={}, size={}, page_size={}", 
                       offset, size, self.page_size);
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
                log::error!("CUDA memcpy failed: {} bytes from device offset {} to host: {}", 
                           size, offset, CudaError(result));
                Err(CudaError(result))
            } else {
                log::trace!("✓ CUDA memcpy: {} bytes from device offset {} to host", size, offset);
                Ok(())
            }
        }
        
        #[cfg(not(cuda_available))]
        {
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }

    /// Zero-copy device-to-device move within page with bounds checking (REAL CUDA memcpy)
    pub fn device_to_device_copy(&self, src_offset: usize, dst_offset: usize, size: usize) -> Result<(), CudaError> {
        if !self.initialized {
            return Err(CudaError(CUDA_ERROR_NOT_INITIALIZED));
        }

        if src_offset.saturating_add(size) > self.page_size || 
           dst_offset.saturating_add(size) > self.page_size {
            log::error!("Device-to-device copy would exceed page bounds");
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
                log::error!("CUDA device-to-device copy failed: {} bytes: {}", size, CudaError(result));
                Err(CudaError(result))
            } else {
                log::debug!("✓ Zero-copy device move: {} bytes from offset {} to {}", 
                           size, src_offset, dst_offset);
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
        if self.page_size == 0 {
            0.0
        } else {
            self.current_offset() as f64 / self.page_size as f64
        }
    }

    /// Synchronize all operations on this allocator
    pub fn synchronize(&self) -> Result<(), CudaError> {
        if !self.initialized {
            return Err(CudaError(CUDA_ERROR_NOT_INITIALIZED));
        }

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

    /// Reset allocator (for page reuse) with proper cleanup
    pub fn reset(&self) {
        self.current_offset.store(0, Ordering::Relaxed);
        
        // Optionally zero the memory again for clean reuse
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
        
        log::debug!("✓ Reset allocator on device {}", self.device_id);
    }

    /// Get basic info
    pub fn device_id(&self) -> i32 { self.device_id }
    pub fn page_size(&self) -> usize { self.page_size }
    pub fn device_ptr(&self) -> *mut c_void { self.device_ptr.as_ptr() as *mut c_void }
    pub fn is_initialized(&self) -> bool { self.initialized }
    
    /// Get stream reference
    pub fn stream(&self) -> Option<&Arc<CudaStream>> {
        self.stream.as_ref()
    }
}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
        if !self.initialized {
            return;
        }

        #[cfg(cuda_available)]
        unsafe {
            // Set device context
            let device_result = cudaSetDevice(self.device_id);
            if device_result != CUDA_SUCCESS {
                log::error!("Failed to set device {} for cleanup: {}", 
                           self.device_id, CudaError(device_result));
                // Continue with cleanup anyway
            }
            
            // Synchronize before freeing to ensure all operations complete
            if let Some(stream) = &self.stream {
                if let Err(e) = stream.synchronize() {
                    log::warn!("Failed to synchronize stream during cleanup: {}", e);
                }
            } else {
                let sync_result = cudaDeviceSynchronize();
                if sync_result != CUDA_SUCCESS {
                    log::warn!("Failed to synchronize device during cleanup: {}", CudaError(sync_result));
                }
            }
            
            // REAL CUDA FREE - this actually calls cudaFree
            let free_result = cudaFree(self.device_ptr.as_ptr() as *mut c_void);
            if free_result != CUDA_SUCCESS {
                log::error!("Failed to free CUDA memory: {}", CudaError(free_result));
            } else {
                log::info!("✓ Freed CUDA page: {} bytes on device {}, ptr={:p}", 
                          self.page_size, self.device_id, self.device_ptr.as_ptr());
            }
        }
        
        self.initialized = false;
    }
}

unsafe impl Send for BumpAllocator {}
unsafe impl Sync for BumpAllocator {}

/// CUDA page backed by real device memory with proper alignment
#[derive(Debug)]
pub struct CudaPage {
    allocator: BumpAllocator,
    allocation_id: u64,
}

impl CudaPage {
    /// Create new CUDA page with real device memory and proper alignment
    pub fn new(size: usize, device_id: i32) -> Result<Self, CudaError> {
        let allocator = BumpAllocator::new(size, device_id)?;
        let allocation_id = super::NEXT_ALLOCATION_ID.fetch_add(1, Ordering::Relaxed) as u64;
        
        log::debug!("✓ Created CUDA page {}: {} bytes on device {}", 
                   allocation_id, allocator.page_size(), device_id);
        
        Ok(CudaPage {
            allocator,
            allocation_id,
        })
    }

    /// Bump allocate within this page with proper alignment
    pub fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        let alignment = align.max(CUDA_MEMORY_ALIGNMENT);
        self.allocator.allocate(size, alignment)
    }

    /// Get device pointer at offset with bounds checking
    pub fn device_ptr_at_offset(&self, offset: usize) -> *mut c_void {
        self.allocator.device_ptr_at(offset).unwrap_or(std::ptr::null_mut())
    }

    /// Copy from host to device with validation
    pub fn copy_from_host(&self, host_data: *const c_void, size: usize, offset: usize) -> Result<(), CudaError> {
        self.allocator.copy_from_host(offset, host_data, size)
    }

    /// Copy from device to host with validation
    pub fn copy_to_host(&self, host_data: *mut c_void, size: usize, offset: usize) -> Result<(), CudaError> {
        self.allocator.copy_to_host(offset, host_data, size)
    }

    /// Zero-copy device-to-device copy with validation
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

    // Safe getters with validation
    pub fn size(&self) -> usize { self.allocator.page_size() }
    pub fn device_id(&self) -> i32 { self.allocator.device_id() }
    pub fn device_ptr(&self) -> *mut c_void { self.allocator.device_ptr() }
    pub fn current_offset(&self) -> usize { self.allocator.current_offset() }
    pub fn available_space(&self) -> usize { self.allocator.available_space() }
    pub fn utilization(&self) -> f64 { self.allocator.utilization() }
    pub fn allocation_id(&self) -> u64 { self.allocation_id }
    pub fn is_initialized(&self) -> bool { self.allocator.is_initialized() }
    
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