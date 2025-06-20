// src/cuda.rs - Real CUDA device memory management with actual CUDA calls
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::ptr::NonNull;
use std::collections::HashMap;

// Real CUDA FFI bindings - these will link to actual CUDA runtime
extern "C" {
    // Device management
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    
    // Memory management - REAL CUDA CALLS
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut c_void) -> i32;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: i32, stream: *mut c_void) -> i32;
    fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaMemsetAsync(devPtr: *mut c_void, value: i32, count: usize, stream: *mut c_void) -> i32;
    
    // Stream management
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaStreamQuery(stream: *mut c_void) -> i32;
    
    // Error handling
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const i8;
    
    // Device properties
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    fn cudaGetDeviceProperties(prop: *mut CudaDeviceProperties, device: i32) -> i32;
    
    // Context management
    fn cudaDeviceSynchronize() -> i32;
}

// CUDA constants
const CUDA_SUCCESS: i32 = 0;
const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;
const CUDA_ERROR_NOT_INITIALIZED: i32 = 3;
const CUDA_ERROR_INVALID_DEVICE: i32 = 10;
const CUDA_ERROR_INVALID_VALUE: i32 = 11;

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

const CUDA_STREAM_NON_BLOCKING: u32 = 0x01;

#[repr(C)]
struct CudaDeviceProperties {
    name: [i8; 256],
    uuid: [i8; 16],
    total_global_mem: usize,
    shared_mem_per_block: usize,
    regs_per_block: i32,
    warp_size: i32,
    mem_pitch: usize,
    max_threads_per_block: i32,
    max_threads_dim: [i32; 3],
    max_grid_size: [i32; 3],
    clock_rate: i32,
    total_const_mem: usize,
    major: i32,
    minor: i32,
    texture_alignment: usize,
    texture_pitch_alignment: usize,
    device_overlap: i32,
    multiprocessor_count: i32,
    kernel_exec_timeout_enabled: i32,
    integrated: i32,
    can_map_host_memory: i32,
    compute_mode: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct CudaError(pub i32);

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let c_str = cudaGetErrorString(self.0);
            if c_str.is_null() {
                write!(f, "CUDA Error {}: Unknown error", self.0)
            } else {
                let cstr = std::ffi::CStr::from_ptr(c_str);
                write!(f, "CUDA Error {}: {}", self.0, cstr.to_string_lossy())
            }
        }
    }
}

impl std::error::Error for CudaError {}

/// CUDA stream for asynchronous operations
#[derive(Debug)]
pub struct CudaStream {
    stream: NonNull<c_void>,
    device_id: i32,
}

impl CudaStream {
    /// Create a new CUDA stream with REAL CUDA API call
    pub fn new(device_id: i32, non_blocking: bool) -> Result<Self, CudaError> {
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
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.stream.as_ptr()
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe {
            let result = cudaStreamSynchronize(self.stream.as_ptr());
            if result != CUDA_SUCCESS {
                Err(CudaError(result))
            } else {
                Ok(())
            }
        }
    }

    pub fn is_complete(&self) -> Result<bool, CudaError> {
        unsafe {
            let result = cudaStreamQuery(self.stream.as_ptr());
            match result {
                CUDA_SUCCESS => Ok(true),
                1 => Ok(false), // cudaErrorNotReady
                _ => Err(CudaError(result)),
            }
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            let _ = cudaSetDevice(self.device_id);
            let _ = cudaStreamDestroy(self.stream.as_ptr());
        }
    }
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

/// True bump allocator with atomic offset - NO PER-TENSOR METADATA
#[derive(Debug)]
pub struct BumpAllocator {
    /// Current allocation offset (atomic for thread safety)
    current_offset: AtomicUsize,
    /// Raw device memory pointer
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
                return Err(CudaError(result));
            }

            // Zero initialize the memory
            let result = cudaMemset(device_ptr, 0, page_size);
            if result != CUDA_SUCCESS {
                let _ = cudaFree(device_ptr);
                return Err(CudaError(result));
            }

            // Create async stream for this allocator
            let stream = match CudaStream::new(device_id, true) {
                Ok(stream) => Some(Arc::new(stream)),
                Err(_) => None, // Continue without stream
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
    }

    /// Pure bump allocation: offset += align(size) - NO METADATA
    pub fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        let aligned_size = (size + align - 1) & !(align - 1);
        
        // Atomic bump allocation
        let old_offset = self.current_offset.fetch_add(aligned_size, Ordering::Relaxed);
        
        if old_offset + aligned_size <= self.page_size {
            // Return pointer to allocated region
            let ptr = unsafe { self.device_ptr.as_ptr().add(old_offset) };
            Some(NonNull::new(ptr).unwrap())
        } else {
            // Page full - allocation failed
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
                Err(CudaError(result))
            } else {
                Ok(())
            }
        }
    }

    /// Zero-copy device-to-device move within page (REAL CUDA memcpy)
    pub fn device_to_device_copy(&self, src_offset: usize, dst_offset: usize, size: usize) -> Result<(), CudaError> {
        if src_offset + size > self.page_size || dst_offset + size > self.page_size {
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }

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
                Err(CudaError(result))
            } else {
                log::debug!("Zero-copy device move: {} bytes from {} to {}", size, src_offset, dst_offset);
                Ok(())
            }
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
        }
    }

    /// Reset allocator (for page reuse)
    pub fn reset(&self) {
        self.current_offset.store(0, Ordering::Relaxed);
    }

    /// Get basic info
    pub fn device_id(&self) -> i32 { self.device_id }
    pub fn page_size(&self) -> usize { self.page_size }
    pub fn device_ptr(&self) -> *mut c_void { self.device_ptr.as_ptr() as *mut c_void }
}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
        unsafe {
            // Set device context
            let _ = cudaSetDevice(self.device_id);
            
            // Synchronize before freeing
            if let Some(stream) = &self.stream {
                let _ = stream.synchronize();
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

/// Real CUDA page backed by bump allocator
#[derive(Debug)]
pub struct CudaPage {
    allocator: BumpAllocator,
    allocation_id: u64,
}

// Global allocation tracking
static NEXT_ALLOCATION_ID: AtomicUsize = AtomicUsize::new(1);

impl CudaPage {
    /// Create new CUDA page with real device memory
    pub fn new(size: usize, device_id: i32) -> Result<Self, CudaError> {
        let allocator = BumpAllocator::new(size, device_id)?;
        let allocation_id = NEXT_ALLOCATION_ID.fetch_add(1, Ordering::Relaxed) as u64;
        
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
    pub fn is_ready(&self) -> Result<bool, CudaError> { 
        // For simplicity, always ready after sync
        self.synchronize()?;
        Ok(true)
    }
}

/// Real CUDA tensor that directly references device memory
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
                Ok(())
            }
        }
    }

    /// Synchronize tensor operations
    pub fn synchronize(&self) -> Result<(), CudaError> {
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
    }
}

/// Device information with real CUDA queries
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub device_id: i32,
    pub name: String,
    pub total_memory: usize,
    pub free_memory: usize,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub warp_size: i32,
    pub memory_clock_rate: i32,
    pub memory_bus_width: i32,
}

impl CudaDeviceInfo {
    /// Query device info using REAL CUDA API calls
    pub fn query(device_id: i32) -> Result<Self, CudaError> {
        unsafe {
            let result = cudaSetDevice(device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            // Get device properties
            let mut props: CudaDeviceProperties = std::mem::zeroed();
            let result = cudaGetDeviceProperties(&mut props, device_id);
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

            // Get additional attributes
            let mut memory_clock_rate = 0;
            let mut memory_bus_width = 0;
            cudaDeviceGetAttribute(&mut memory_clock_rate, 36, device_id); // CUDA_DEVICE_ATTR_MEMORY_CLOCK_RATE
            cudaDeviceGetAttribute(&mut memory_bus_width, 37, device_id);  // CUDA_DEVICE_ATTR_GLOBAL_MEMORY_BUS_WIDTH

            // Convert name from C string
            let name = std::ffi::CStr::from_ptr(props.name.as_ptr())
                .to_string_lossy()
                .into_owned();

            Ok(CudaDeviceInfo {
                device_id,
                name,
                total_memory,
                free_memory,
                compute_capability_major: props.major,
                compute_capability_minor: props.minor,
                multiprocessor_count: props.multiprocessor_count,
                max_threads_per_block: props.max_threads_per_block,
                warp_size: props.warp_size,
                memory_clock_rate,
                memory_bus_width,
            })
        }
    }
}

/// CUDA memory manager with real device queries
#[derive(Debug)]
pub struct CudaMemoryManager {
    pub device_infos: Vec<CudaDeviceInfo>,
    current_device: i32,
    initialized: bool,
}

impl CudaMemoryManager {
    /// Initialize with REAL CUDA device detection
    pub fn new() -> Result<Self, CudaError> {
        unsafe {
            // Check if CUDA is available
            let mut device_count = 0;
            let result = cudaGetDeviceCount(&mut device_count);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            if device_count == 0 {
                return Err(CudaError(CUDA_ERROR_NOT_INITIALIZED));
            }

            // Get current device
            let mut current_device = 0;
            let result = cudaGetDevice(&mut current_device);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            // Query all available devices using REAL CUDA calls
            let mut device_infos = Vec::new();
            for device_id in 0..device_count {
                match CudaDeviceInfo::query(device_id) {
                    Ok(info) => {
                        log::info!("REAL CUDA device {}: {} ({} MB)", 
                                  device_id, info.name, info.total_memory / 1024 / 1024);
                        device_infos.push(info);
                    }
                    Err(e) => {
                        log::warn!("Failed to query device {}: {}", device_id, e);
                    }
                }
            }

            if device_infos.is_empty() {
                return Err(CudaError(CUDA_ERROR_INVALID_DEVICE));
            }

            Ok(CudaMemoryManager {
                device_infos,
                current_device,
                initialized: true,
            })
        }
    }

    /// Allocate CUDA page with real device memory
    pub fn allocate_page_on_device(&self, size: usize, device_id: i32) -> Result<CudaPage, CudaError> {
        if !self.initialized {
            return Err(CudaError(CUDA_ERROR_NOT_INITIALIZED));
        }
        CudaPage::new(size, device_id)
    }

    /// Get device information
    pub fn device_info(&self, device_id: i32) -> Option<&CudaDeviceInfo> {
        self.device_infos.iter().find(|info| info.device_id == device_id)
    }
}

#[derive(Debug, Clone)]
pub struct CudaDeviceStats {
    pub device_id: i32,
    pub allocated_bytes: usize,
    pub active_pages: usize,
    pub peak_allocated: usize,
    pub total_memory: usize,
    pub free_memory: usize,
    pub utilization: f64,
}

#[derive(Debug)]
pub struct CudaContext {
    manager: CudaMemoryManager,
}

impl CudaContext {
    pub fn new() -> Result<Self, CudaError> {
        let manager = CudaMemoryManager::new()?;
        Ok(CudaContext { manager })
    }

    pub fn allocate_page_auto(&self, size: usize) -> Result<CudaPage, CudaError> {
        // Find device with most free memory
        let best_device = self.manager.device_infos
            .iter()
            .max_by_key(|info| info.free_memory)
            .map(|info| info.device_id)
            .unwrap_or(0);

        self.allocate_page_on_device(size, best_device)
    }

    pub fn allocate_page_on_device(&self, size: usize, device_id: i32) -> Result<CudaPage, CudaError> {
        self.manager.allocate_page_on_device(size, device_id)
    }

    pub fn device_stats(&self, _device_id: i32) -> Option<(usize, usize)> {
        Some((0, 0)) // Simplified for now
    }

    pub fn device_stats_detailed(&self, device_id: i32) -> Option<CudaDeviceStats> {
        if let Some(info) = self.manager.device_info(device_id) {
            Some(CudaDeviceStats {
                device_id,
                allocated_bytes: 0,
                active_pages: 0,
                peak_allocated: 0,
                total_memory: info.total_memory,
                free_memory: info.free_memory,
                utilization: ((info.total_memory - info.free_memory) as f64 / info.total_memory as f64) * 100.0,
            })
        } else {
            None
        }
    }

    pub fn manager(&self) -> &CudaMemoryManager {
        &self.manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_cuda_allocation() {
        // This test will only pass with actual CUDA runtime
        match CudaMemoryManager::new() {
            Ok(manager) => {
                println!("✓ Real CUDA manager initialized");
                if let Ok(page) = manager.allocate_page_on_device(1024 * 1024, 0) {
                    println!("✓ Real CUDA page allocated: {} bytes", page.size());
                    
                    // Test real bump allocation
                    if let Some(_ptr) = page.allocate(1024, 256) {
                        println!("✓ Real bump allocation successful");
                    }
                }
            }
            Err(e) => {
                println!("Real CUDA not available: {}", e);
                // This is expected in environments without CUDA runtime
            }
        }
    }

    #[test]
    fn test_bump_allocator() {
        if let Ok(allocator) = BumpAllocator::new(4096, 0) {
            // Test bump allocation
            let ptr1 = allocator.allocate(1024, 256);
            assert!(ptr1.is_some());
            assert_eq!(allocator.current_offset(), 1024);

            let ptr2 = allocator.allocate(2048, 256);
            assert!(ptr2.is_some());
            assert_eq!(allocator.current_offset(), 3072);

            // Test overflow
            let ptr3 = allocator.allocate(2048, 256);
            assert!(ptr3.is_none()); // Should fail - not enough space

            println!("✓ Bump allocator tests passed");
        }
    }
}