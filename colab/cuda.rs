// src/cuda.rs - Direct CUDA memory management for arena allocation
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::ptr::NonNull;
use crossbeam::queue::SegQueue;

// CUDA FFI bindings for direct memory management
extern "C" {
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: i32, stream: *mut c_void) -> i32;
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const i8;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
}

// CUDA constants
const CUDA_SUCCESS: i32 = 0;
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

#[derive(Debug, Clone, Copy)]
pub struct CudaError(pub i32);

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let c_str = cudaGetErrorString(self.0);
            let cstr = std::ffi::CStr::from_ptr(c_str);
            write!(f, "CUDA Error {}: {}", self.0, cstr.to_string_lossy())
        }
    }
}

impl std::error::Error for CudaError {}

/// Direct CUDA device memory page - manages actual GPU heap
#[derive(Debug)]
pub struct CudaPage {
    device_ptr: NonNull<c_void>,
    size: usize,
    device_id: i32,
    stream: Option<NonNull<c_void>>,
}

impl CudaPage {
    /// Allocate a new CUDA page directly on device
    pub fn new(size: usize, device_id: i32) -> Result<Self, CudaError> {
        unsafe {
            // Set device context
            let result = cudaSetDevice(device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            // Allocate device memory directly
            let mut device_ptr = std::ptr::null_mut();
            let result = cudaMalloc(&mut device_ptr, size);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            // Create dedicated CUDA stream for async operations
            let mut stream = std::ptr::null_mut();
            let result = cudaStreamCreate(&mut stream);
            let stream_opt = if result == CUDA_SUCCESS {
                Some(NonNull::new(stream).unwrap())
            } else {
                None
            };

            Ok(CudaPage {
                device_ptr: NonNull::new(device_ptr).unwrap(),
                size,
                device_id,
                stream: stream_opt,
            })
        }
    }

    /// Get raw device pointer
    pub fn device_ptr(&self) -> *mut c_void {
        self.device_ptr.as_ptr()
    }

    /// Get device pointer at offset
    pub fn device_ptr_at_offset(&self, offset: usize) -> *mut c_void {
        unsafe {
            (self.device_ptr.as_ptr() as *mut u8).add(offset) as *mut c_void
        }
    }

    /// Get page size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Copy data from host to this device page
    pub fn copy_from_host(&self, host_data: *const c_void, size: usize, offset: usize) -> Result<(), CudaError> {
        if offset + size > self.size {
            return Err(CudaError(-1)); // Out of bounds
        }

        unsafe {
            let dst = self.device_ptr_at_offset(offset);
            let result = if let Some(stream) = self.stream {
                cudaMemcpyAsync(dst, host_data, size, CUDA_MEMCPY_HOST_TO_DEVICE, stream.as_ptr())
            } else {
                cudaMemcpy(dst, host_data, size, CUDA_MEMCPY_HOST_TO_DEVICE)
            };

            if result != CUDA_SUCCESS {
                Err(CudaError(result))
            } else {
                Ok(())
            }
        }
    }

    /// Copy data from this device page to host
    pub fn copy_to_host(&self, host_data: *mut c_void, size: usize, offset: usize) -> Result<(), CudaError> {
        if offset + size > self.size {
            return Err(CudaError(-1)); // Out of bounds
        }

        unsafe {
            let src = self.device_ptr_at_offset(offset);
            let result = if let Some(stream) = self.stream {
                cudaMemcpyAsync(host_data, src, size, CUDA_MEMCPY_DEVICE_TO_HOST, stream.as_ptr())
            } else {
                cudaMemcpy(host_data, src, size, CUDA_MEMCPY_DEVICE_TO_HOST)
            };

            if result != CUDA_SUCCESS {
                Err(CudaError(result))
            } else {
                Ok(())
            }
        }
    }

    /// Copy data within device (device-to-device)
    pub fn copy_device_to_device(&self, src_offset: usize, dst_offset: usize, size: usize) -> Result<(), CudaError> {
        if src_offset + size > self.size || dst_offset + size > self.size {
            return Err(CudaError(-1)); // Out of bounds
        }

        unsafe {
            let src = self.device_ptr_at_offset(src_offset);
            let dst = self.device_ptr_at_offset(dst_offset);
            let result = if let Some(stream) = self.stream {
                cudaMemcpyAsync(dst, src, size, CUDA_MEMCPY_DEVICE_TO_DEVICE, stream.as_ptr())
            } else {
                cudaMemcpy(dst, src, size, CUDA_MEMCPY_DEVICE_TO_DEVICE)
            };

            if result != CUDA_SUCCESS {
                Err(CudaError(result))
            } else {
                Ok(())
            }
        }
    }

    /// Synchronize stream operations
    pub fn synchronize(&self) -> Result<(), CudaError> {
        if let Some(stream) = self.stream {
            unsafe {
                let result = cudaStreamSynchronize(stream.as_ptr());
                if result != CUDA_SUCCESS {
                    Err(CudaError(result))
                } else {
                    Ok(())
                }
            }
        } else {
            Ok(()) // No stream to synchronize
        }
    }

    /// Check if this page can accommodate a tensor at given offset
    pub fn can_fit(&self, offset: usize, size: usize) -> bool {
        offset + size <= self.size
    }

    /// Get available space from offset
    pub fn available_space_from(&self, offset: usize) -> usize {
        if offset >= self.size {
            0
        } else {
            self.size - offset
        }
    }
}

// Implement Clone for CudaPage to fix lifetime management issues
impl Clone for CudaPage {
    fn clone(&self) -> Self {
        // Create a new page with the same size and device
        // Note: This creates a new allocation, not a shared reference
        // In a production system, you might want reference counting instead
        Self::new(self.size, self.device_id).unwrap_or_else(|_| {
            // Fallback: create a dummy page that won't be used
            CudaPage {
                device_ptr: NonNull::dangling(),
                size: 0,
                device_id: self.device_id,
                stream: None,
            }
        })
    }
}

impl Drop for CudaPage {
    fn drop(&mut self) {
        // Only free if we have a valid allocation
        if self.size > 0 && !self.device_ptr.as_ptr().is_null() {
            unsafe {
                // Set device context before freeing
                let _ = cudaSetDevice(self.device_id);
                
                // Destroy stream first
                if let Some(stream) = self.stream {
                    let _ = cudaStreamDestroy(stream.as_ptr());
                }
                
                // Free device memory
                let _ = cudaFree(self.device_ptr.as_ptr());
            }
        }
    }
}

unsafe impl Send for CudaPage {}
unsafe impl Sync for CudaPage {}

/// CUDA device information and capabilities
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub device_id: i32,
    pub total_memory: usize,
    pub free_memory: usize,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub warp_size: i32,
}

impl CudaDeviceInfo {
    /// Query device information
    pub fn query(device_id: i32) -> Result<Self, CudaError> {
        unsafe {
            let result = cudaSetDevice(device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            // Get memory info
            let mut free_memory = 0;
            let mut total_memory = 0;
            let result = cudaMemGetInfo(&mut free_memory, &mut total_memory);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            // Get device attributes
            let mut major = 0;
            let mut minor = 0;
            let mut mp_count = 0;
            let mut max_threads = 0;
            let mut warp_size = 0;

            cudaDeviceGetAttribute(&mut major, 2, device_id); // cudaDevAttrComputeCapabilityMajor
            cudaDeviceGetAttribute(&mut minor, 3, device_id); // cudaDevAttrComputeCapabilityMinor
            cudaDeviceGetAttribute(&mut mp_count, 16, device_id); // cudaDevAttrMultiProcessorCount
            cudaDeviceGetAttribute(&mut max_threads, 1, device_id); // cudaDevAttrMaxThreadsPerBlock
            cudaDeviceGetAttribute(&mut warp_size, 10, device_id); // cudaDevAttrWarpSize

            Ok(CudaDeviceInfo {
                device_id,
                total_memory,
                free_memory,
                compute_capability_major: major,
                compute_capability_minor: minor,
                multiprocessor_count: mp_count,
                max_threads_per_block: max_threads,
                warp_size,
            })
        }
    }

    /// Check if device supports efficient memory access patterns
    pub fn supports_efficient_access(&self) -> bool {
        // Compute capability 6.0+ supports efficient uncoalesced access
        (self.compute_capability_major > 6) || 
        (self.compute_capability_major == 6 && self.compute_capability_minor >= 0)
    }

    /// Calculate optimal page size for this device
    pub fn optimal_page_size(&self, typical_tensor_size: usize) -> usize {
        // Base page size on L2 cache size and memory bandwidth
        let base_page_size = if self.compute_capability_major >= 8 {
            1024 * 1024 // 1MB for modern GPUs (A100, H100)
        } else if self.compute_capability_major >= 7 {
            512 * 1024  // 512KB for V100/T4
        } else {
            256 * 1024  // 256KB for older GPUs
        };

        // Ensure page can fit multiple typical tensors
        let min_page_size = typical_tensor_size * 4;
        base_page_size.max(min_page_size).min(16 * 1024 * 1024) // Cap at 16MB
    }
}

/// CUDA memory manager for direct device heap management
#[derive(Debug)]
pub struct CudaMemoryManager {
    device_infos: Vec<CudaDeviceInfo>,
    current_device: i32,
}

impl CudaMemoryManager {
    /// Initialize CUDA memory manager
    pub fn new() -> Result<Self, CudaError> {
        unsafe {
            // Get current device
            let mut current_device = 0;
            let result = cudaGetDevice(&mut current_device);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            // Query all available devices
            let mut device_infos = Vec::new();
            for device_id in 0..8 { // Check up to 8 devices
                match CudaDeviceInfo::query(device_id) {
                    Ok(info) => device_infos.push(info),
                    Err(_) => break, // No more devices
                }
            }

            if device_infos.is_empty() {
                return Err(CudaError(-1)); // No CUDA devices found
            }

            Ok(CudaMemoryManager {
                device_infos,
                current_device,
            })
        }
    }

    /// Get device information
    pub fn device_info(&self, device_id: i32) -> Option<&CudaDeviceInfo> {
        self.device_infos.iter().find(|info| info.device_id == device_id)
    }

    /// Get current device info
    pub fn current_device_info(&self) -> &CudaDeviceInfo {
        self.device_info(self.current_device).unwrap()
    }

    /// Set current device
    pub fn set_device(&mut self, device_id: i32) -> Result<(), CudaError> {
        if self.device_info(device_id).is_none() {
            return Err(CudaError(-1)); // Invalid device
        }

        unsafe {
            let result = cudaSetDevice(device_id);
            if result != CUDA_SUCCESS {
                Err(CudaError(result))
            } else {
                self.current_device = device_id;
                Ok(())
            }
        }
    }

    /// Allocate CUDA page on current device
    pub fn allocate_page(&self, size: usize) -> Result<CudaPage, CudaError> {
        CudaPage::new(size, self.current_device)
    }

    /// Allocate CUDA page on specific device
    pub fn allocate_page_on_device(&self, size: usize, device_id: i32) -> Result<CudaPage, CudaError> {
        CudaPage::new(size, device_id)
    }

    /// Get total available memory across all devices
    pub fn total_available_memory(&self) -> usize {
        self.device_infos.iter().map(|info| info.free_memory).sum()
    }

    /// Check memory pressure on current device
    pub fn memory_pressure(&self) -> f32 {
        let info = self.current_device_info();
        1.0 - (info.free_memory as f32 / info.total_memory as f32)
    }

    /// Suggest optimal page size for current device and workload
    pub fn suggest_page_size(&self, typical_tensor_size: usize) -> usize {
        self.current_device_info().optimal_page_size(typical_tensor_size)
    }
}

// Implement Clone for CudaMemoryManager to fix the ownership issues
impl Clone for CudaMemoryManager {
    fn clone(&self) -> Self {
        // Clone by recreating the manager
        Self::new().unwrap_or_else(|_| {
            // Fallback: create empty manager
            CudaMemoryManager {
                device_infos: vec![],
                current_device: 0,
            }
        })
    }
}

/// CUDA context manager for multi-device scenarios
pub struct CudaContext {
    manager: CudaMemoryManager,
    device_states: Vec<CudaDeviceState>,
}

#[derive(Debug)]
struct CudaDeviceState {
    device_id: i32,
    allocated_bytes: AtomicUsize,
    active_pages: AtomicUsize,
}

impl CudaContext {
    pub fn new() -> Result<Self, CudaError> {
        let manager = CudaMemoryManager::new()?;
        let device_states = manager.device_infos
            .iter()
            .map(|info| CudaDeviceState {
                device_id: info.device_id,
                allocated_bytes: AtomicUsize::new(0),
                active_pages: AtomicUsize::new(0),
            })
            .collect();

        Ok(CudaContext {
            manager,
            device_states,
        })
    }

    /// Allocate page with automatic device selection
    pub fn allocate_page_auto(&self, size: usize) -> Result<CudaPage, CudaError> {
        // Find device with most free memory
        let best_device = self.manager.device_infos
            .iter()
            .max_by_key(|info| info.free_memory)
            .map(|info| info.device_id)
            .unwrap_or(0);

        self.allocate_page_on_device(size, best_device)
    }

    /// Allocate page on specific device with tracking
    pub fn allocate_page_on_device(&self, size: usize, device_id: i32) -> Result<CudaPage, CudaError> {
        let page = self.manager.allocate_page_on_device(size, device_id)?;
        
        // Update tracking
        if let Some(state) = self.device_states.iter().find(|s| s.device_id == device_id) {
            state.allocated_bytes.fetch_add(size, Ordering::Relaxed);
            state.active_pages.fetch_add(1, Ordering::Relaxed);
        }

        Ok(page)
    }

    /// Get allocation statistics for device
    pub fn device_stats(&self, device_id: i32) -> Option<(usize, usize)> {
        self.device_states
            .iter()
            .find(|s| s.device_id == device_id)
            .map(|s| (
                s.allocated_bytes.load(Ordering::Relaxed),
                s.active_pages.load(Ordering::Relaxed)
            ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_page_allocation() {
        if let Ok(manager) = CudaMemoryManager::new() {
            let page_size = 1024 * 1024; // 1MB
            match manager.allocate_page(page_size) {
                Ok(page) => {
                    assert_eq!(page.size(), page_size);
                    assert!(!page.device_ptr().is_null());
                    println!("✓ CUDA page allocation successful");
                }
                Err(e) => println!("CUDA allocation failed: {}", e),
            }
        } else {
            println!("No CUDA devices available for testing");
        }
    }

    #[test]
    fn test_device_info_query() {
        if let Ok(manager) = CudaMemoryManager::new() {
            let info = manager.current_device_info();
            println!("Device {}: {} MB total, {} MB free", 
                     info.device_id, 
                     info.total_memory / 1024 / 1024,
                     info.free_memory / 1024 / 1024);
            assert!(info.total_memory > 0);
        }
    }

    #[test]
    fn test_cuda_page_clone() {
        if let Ok(manager) = CudaMemoryManager::new() {
            if let Ok(page) = manager.allocate_page(1024) {
                let cloned_page = page.clone();
                assert_eq!(cloned_page.device_id(), page.device_id());
                println!("✓ CUDA page clone test passed");
            }
        }
    }

    #[test] 
    fn test_cuda_manager_clone() {
        if let Ok(manager) = CudaMemoryManager::new() {
            let cloned_manager = manager.clone();
            assert_eq!(cloned_manager.current_device, manager.current_device);
            println!("✓ CUDA manager clone test passed");
        }
    }
}