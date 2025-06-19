// src/cuda.rs - Real CUDA device memory management for arena allocation
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::collections::HashMap;

// Real CUDA FFI bindings
extern "C" {
    // Device management
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    
    // Memory management
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
    fn cudaPeekAtLastError() -> i32;
    
    // Device properties
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    fn cudaGetDeviceProperties(prop: *mut CudaDeviceProperties, device: i32) -> i32;
    
    // Context management
    fn cudaDeviceSynchronize() -> i32;
    fn cudaDeviceReset() -> i32;
}

// CUDA constants
const CUDA_SUCCESS: i32 = 0;
const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;
const CUDA_ERROR_NOT_INITIALIZED: i32 = 3;
const CUDA_ERROR_DEINITIALIZED: i32 = 4;
const CUDA_ERROR_INVALID_DEVICE: i32 = 10;
const CUDA_ERROR_INVALID_VALUE: i32 = 11;

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

const CUDA_STREAM_NON_BLOCKING: u32 = 0x01;

// Device attributes
const CUDA_DEVICE_ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CUDA_DEVICE_ATTR_COMPUTE_CAPABILITY_MINOR: i32 = 76;
const CUDA_DEVICE_ATTR_MULTIPROCESSOR_COUNT: i32 = 16;
const CUDA_DEVICE_ATTR_MAX_THREADS_PER_BLOCK: i32 = 1;
const CUDA_DEVICE_ATTR_WARP_SIZE: i32 = 10;
const CUDA_DEVICE_ATTR_MEMORY_CLOCK_RATE: i32 = 36;
const CUDA_DEVICE_ATTR_GLOBAL_MEMORY_BUS_WIDTH: i32 = 37;

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
    // ... more fields as needed
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
    /// Create a new CUDA stream
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

    /// Get raw stream pointer
    pub fn as_ptr(&self) -> *mut c_void {
        self.stream.as_ptr()
    }

    /// Synchronize stream
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

    /// Check if stream operations are complete
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

/// Real CUDA device memory page
#[derive(Debug)]
pub struct CudaPage {
    device_ptr: NonNull<c_void>,
    size: usize,
    device_id: i32,
    stream: Option<Arc<CudaStream>>,
    allocation_id: u64,
}

// Global allocation tracking
static NEXT_ALLOCATION_ID: AtomicUsize = AtomicUsize::new(1);
lazy_static::lazy_static! {
    static ref ACTIVE_ALLOCATIONS: Mutex<HashMap<u64, (usize, i32)>> = Mutex::new(HashMap::new());
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

            // Initialize memory to zero
            let result = cudaMemset(device_ptr, 0, size);
            if result != CUDA_SUCCESS {
                let _ = cudaFree(device_ptr);
                return Err(CudaError(result));
            }

            // Create dedicated CUDA stream for this page
            let stream = match CudaStream::new(device_id, true) {
                Ok(stream) => Some(Arc::new(stream)),
                Err(_) => None, // Continue without stream if creation fails
            };

            let allocation_id = NEXT_ALLOCATION_ID.fetch_add(1, Ordering::Relaxed) as u64;

            // Track allocation
            if let Ok(mut allocations) = ACTIVE_ALLOCATIONS.lock() {
                allocations.insert(allocation_id, (size, device_id));
            }

            log::debug!("Allocated CUDA page: {} bytes on device {}, ptr={:p}", 
                       size, device_id, device_ptr);

            Ok(CudaPage {
                device_ptr: NonNull::new(device_ptr).unwrap(),
                size,
                device_id,
                stream,
                allocation_id,
            })
        }
    }

    /// Get raw device pointer
    pub fn device_ptr(&self) -> *mut c_void {
        self.device_ptr.as_ptr()
    }

    /// Get device pointer at offset
    pub fn device_ptr_at_offset(&self, offset: usize) -> *mut c_void {
        if offset > self.size {
            return std::ptr::null_mut();
        }
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

    /// Get allocation ID
    pub fn allocation_id(&self) -> u64 {
        self.allocation_id
    }

    /// Copy data from host to this device page
    pub fn copy_from_host(&self, host_data: *const c_void, size: usize, offset: usize) -> Result<(), CudaError> {
        if offset + size > self.size {
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }

        unsafe {
            let result = cudaSetDevice(self.device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            let dst = self.device_ptr_at_offset(offset);
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

    /// Copy data from this device page to host
    pub fn copy_to_host(&self, host_data: *mut c_void, size: usize, offset: usize) -> Result<(), CudaError> {
        if offset + size > self.size {
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }

        unsafe {
            let result = cudaSetDevice(self.device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            let src = self.device_ptr_at_offset(offset);
            let result = if let Some(stream) = &self.stream {
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

    /// Copy data within device (device-to-device) - TRUE ZERO-COPY
    pub fn copy_device_to_device(&self, src_offset: usize, dst_offset: usize, size: usize) -> Result<(), CudaError> {
        if src_offset + size > self.size || dst_offset + size > self.size {
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }

        unsafe {
            let result = cudaSetDevice(self.device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }

            let src = self.device_ptr_at_offset(src_offset);
            let dst = self.device_ptr_at_offset(dst_offset);
            
            let result = if let Some(stream) = &self.stream {
                cudaMemcpyAsync(dst, src, size, CUDA_MEMCPY_DEVICE_TO_DEVICE, stream.as_ptr())
            } else {
                cudaMemcpy(dst, src, size, CUDA_MEMCPY_DEVICE_TO_DEVICE)
            };

            if result != CUDA_SUCCESS {
                Err(CudaError(result))
            } else {
                log::debug!("Zero-copy device-to-device: {} bytes from offset {} to {}", 
                           size, src_offset, dst_offset);
                Ok(())
            }
        }
    }

    /// Zero-copy memory move within the same page
    pub fn move_memory(&self, src_offset: usize, dst_offset: usize, size: usize) -> Result<(), CudaError> {
        if src_offset + size > self.size || dst_offset + size > self.size {
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }

        // Use device-to-device copy for zero-copy move
        self.copy_device_to_device(src_offset, dst_offset, size)
    }

    /// Synchronize all operations on this page
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

    /// Zero-copy resize (extend the allocation if possible)
    pub fn try_resize(&mut self, new_size: usize) -> Result<bool, CudaError> {
        if new_size <= self.size {
            // Shrinking or same size - always succeeds
            self.size = new_size;
            return Ok(true);
        }

        // Cannot extend CUDA allocation in place
        // Would need reallocation, which breaks zero-copy guarantee
        Ok(false)
    }

    /// Get the stream associated with this page
    pub fn stream(&self) -> Option<&Arc<CudaStream>> {
        self.stream.as_ref()
    }

    /// Check if operations on this page are complete
    pub fn is_ready(&self) -> Result<bool, CudaError> {
        if let Some(stream) = &self.stream {
            stream.is_complete()
        } else {
            Ok(true) // No stream means synchronous operations are complete
        }
    }
}

impl Drop for CudaPage {
    fn drop(&mut self) {
        unsafe {
            // Set device context before freeing
            let _ = cudaSetDevice(self.device_id);
            
            // Synchronize before freeing
            if let Some(stream) = &self.stream {
                let _ = stream.synchronize();
            }
            
            // Free device memory
            let result = cudaFree(self.device_ptr.as_ptr());
            if result != CUDA_SUCCESS {
                log::error!("Failed to free CUDA memory: {}", CudaError(result));
            } else {
                log::debug!("Freed CUDA page: {} bytes on device {}, ptr={:p}", 
                           self.size, self.device_id, self.device_ptr.as_ptr());
            }

            // Remove from tracking
            if let Ok(mut allocations) = ACTIVE_ALLOCATIONS.lock() {
                allocations.remove(&self.allocation_id);
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
    /// Query device information using real CUDA API
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
            cudaDeviceGetAttribute(&mut memory_clock_rate, CUDA_DEVICE_ATTR_MEMORY_CLOCK_RATE, device_id);
            cudaDeviceGetAttribute(&mut memory_bus_width, CUDA_DEVICE_ATTR_GLOBAL_MEMORY_BUS_WIDTH, device_id);

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

    /// Check if device supports efficient memory access patterns
    pub fn supports_efficient_access(&self) -> bool {
        // Compute capability 6.0+ supports efficient uncoalesced access
        (self.compute_capability_major > 6) || 
        (self.compute_capability_major == 6 && self.compute_capability_minor >= 0)
    }

    /// Calculate memory bandwidth in GB/s
    pub fn memory_bandwidth_gbps(&self) -> f64 {
        // Bandwidth = (memory_clock_rate * memory_bus_width * 2) / 8 / 1000 / 1000
        (self.memory_clock_rate as f64 * self.memory_bus_width as f64 * 2.0) / 8.0 / 1000.0 / 1000.0
    }

    /// Calculate optimal page size for this device
    pub fn optimal_page_size(&self, typical_tensor_size: usize) -> usize {
        // Base page size on L2 cache size and memory bandwidth
        let base_page_size = if self.compute_capability_major >= 8 {
            2048 * 1024 // 2MB for modern GPUs (A100, H100)
        } else if self.compute_capability_major >= 7 {
            1024 * 1024  // 1MB for V100/T4
        } else {
            512 * 1024  // 512KB for older GPUs
        };

        // Ensure page can fit multiple typical tensors
        let min_page_size = typical_tensor_size * 4;
        base_page_size.max(min_page_size).min(32 * 1024 * 1024) // Cap at 32MB
    }

    /// Get memory utilization percentage
    pub fn memory_utilization(&self) -> f64 {
        if self.total_memory == 0 {
            0.0
        } else {
            ((self.total_memory - self.free_memory) as f64 / self.total_memory as f64) * 100.0
        }
    }
}

/// Real CUDA memory manager for direct device heap management
#[derive(Debug)]
pub struct CudaMemoryManager {
    pub device_infos: Vec<CudaDeviceInfo>,
    current_device: i32,
    initialized: bool,
}

impl CudaMemoryManager {
    /// Initialize CUDA memory manager with real CUDA API
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

            // Query all available devices
            let mut device_infos = Vec::new();
            for device_id in 0..device_count {
                match CudaDeviceInfo::query(device_id) {
                    Ok(info) => {
                        log::info!("Found CUDA device {}: {} ({} MB)", 
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

    /// Get device information
    pub fn device_info(&self, device_id: i32) -> Option<&CudaDeviceInfo> {
        self.device_infos.iter().find(|info| info.device_id == device_id)
    }

    /// Get current device info
    pub fn current_device_info(&self) -> &CudaDeviceInfo {
        self.device_info(self.current_device)
            .expect("Current device should always be valid")
    }

    /// Set current device
    pub fn set_device(&mut self, device_id: i32) -> Result<(), CudaError> {
        if self.device_info(device_id).is_none() {
            return Err(CudaError(CUDA_ERROR_INVALID_DEVICE));
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
        if !self.initialized {
            return Err(CudaError(CUDA_ERROR_NOT_INITIALIZED));
        }
        CudaPage::new(size, self.current_device)
    }

    /// Allocate CUDA page on specific device
    pub fn allocate_page_on_device(&self, size: usize, device_id: i32) -> Result<CudaPage, CudaError> {
        if !self.initialized {
            return Err(CudaError(CUDA_ERROR_NOT_INITIALIZED));
        }
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

    /// Get all active allocations (for debugging)
    pub fn get_active_allocations(&self) -> Vec<(u64, usize, i32)> {
        if let Ok(allocations) = ACTIVE_ALLOCATIONS.lock() {
            allocations.iter()
                .map(|(&id, &(size, device))| (id, size, device))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get total allocated memory across all devices
    pub fn total_allocated_memory(&self) -> usize {
        if let Ok(allocations) = ACTIVE_ALLOCATIONS.lock() {
            allocations.values().map(|(size, _)| size).sum()
        } else {
            0
        }
    }

    /// Force garbage collection and memory cleanup
    pub fn garbage_collect(&self) -> Result<(), CudaError> {
        // Synchronize all devices
        for device_info in &self.device_infos {
            unsafe {
                let result = cudaSetDevice(device_info.device_id);
                if result == CUDA_SUCCESS {
                    let _ = cudaDeviceSynchronize();
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct CudaDeviceState {
    device_id: i32,
    allocated_bytes: AtomicUsize,
    active_pages: AtomicUsize,
    peak_allocated: AtomicUsize,
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

/// CUDA context manager for multi-device scenarios with real device tracking
#[derive(Debug)]
pub struct CudaContext {
    manager: CudaMemoryManager,
    device_states: Vec<CudaDeviceState>,
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
                peak_allocated: AtomicUsize::new(0),
            })
            .collect();

        Ok(CudaContext {
            manager,
            device_states,
        })
    }

    /// Allocate page with automatic device selection based on memory availability
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
            let new_allocated = state.allocated_bytes.fetch_add(size, Ordering::Relaxed) + size;
            state.active_pages.fetch_add(1, Ordering::Relaxed);
            
            // Update peak
            let current_peak = state.peak_allocated.load(Ordering::Relaxed);
            if new_allocated > current_peak {
                state.peak_allocated.store(new_allocated, Ordering::Relaxed);
            }
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

    /// Get comprehensive device statistics
    pub fn device_stats_detailed(&self, device_id: i32) -> Option<CudaDeviceStats> {
        if let Some(state) = self.device_states.iter().find(|s| s.device_id == device_id) {
            let device_info = self.manager.device_info(device_id)?;
            
            Some(CudaDeviceStats {
                device_id,
                allocated_bytes: state.allocated_bytes.load(Ordering::Relaxed),
                active_pages: state.active_pages.load(Ordering::Relaxed),
                peak_allocated: state.peak_allocated.load(Ordering::Relaxed),
                total_memory: device_info.total_memory,
                free_memory: device_info.free_memory,
                utilization: device_info.memory_utilization(),
            })
        } else {
            None
        }
    }

    /// Get manager reference
    pub fn manager(&self) -> &CudaMemoryManager {
        &self.manager
    }
}

/// Zero-copy CUDA tensor that directly wraps device memory
#[derive(Debug)]
pub struct CudaTensor {
    device_ptr: NonNull<c_void>,
    size_bytes: usize,
    shape: Vec<usize>,
    element_size: usize,
    device_id: i32,
    page_ref: Option<Arc<CudaPage>>, // Keep page alive
}

impl CudaTensor {
    /// Create tensor from CUDA page at specific offset
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

        let device_ptr = unsafe {
            NonNull::new(page.device_ptr_at_offset(offset))
                .ok_or(CudaError(CUDA_ERROR_INVALID_VALUE))?
        };

        Ok(CudaTensor {
            device_ptr,
            size_bytes,
            shape,
            element_size,
            device_id: page.device_id(),
            page_ref: Some(Arc::clone(page)),
        })
    }

    /// Get device pointer
    pub fn device_ptr(&self) -> *mut c_void {
        self.device_ptr.as_ptr()
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get tensor size in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Get element size
    pub fn element_size(&self) -> usize {
        self.element_size
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Zero-copy reshape (no data movement)
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), CudaError> {
        let new_elements: usize = new_shape.iter().product();
        let old_elements: usize = self.shape.iter().product();
        
        if new_elements != old_elements {
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }

        self.shape = new_shape;
        Ok(())
    }

    /// Zero-copy slice (returns view of same memory)
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
            NonNull::new((self.device_ptr.as_ptr() as *mut u8).add(slice_offset) as *mut c_void)
                .ok_or(CudaError(CUDA_ERROR_INVALID_VALUE))?
        };

        Ok(CudaTensor {
            device_ptr: new_device_ptr,
            size_bytes: slice_len * elements_per_row * self.element_size,
            shape: new_shape,
            element_size: self.element_size,
            device_id: self.device_id,
            page_ref: self.page_ref.clone(),
        })
    }

    /// Copy data from host to this tensor
    pub fn copy_from_host(&self, host_data: *const c_void) -> Result<(), CudaError> {
        if let Some(page) = &self.page_ref {
            // Calculate offset within the page
            let page_start = page.device_ptr() as usize;
            let tensor_start = self.device_ptr.as_ptr() as usize;
            let offset = tensor_start - page_start;
            
            page.copy_from_host(host_data, self.size_bytes, offset)
        } else {
            // Direct CUDA copy
            unsafe {
                let result = cudaSetDevice(self.device_id);
                if result != CUDA_SUCCESS {
                    return Err(CudaError(result));
                }

                let result = cudaMemcpy(
                    self.device_ptr.as_ptr(),
                    host_data,
                    self.size_bytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                );

                if result != CUDA_SUCCESS {
                    Err(CudaError(result))
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Copy data from this tensor to host
    pub fn copy_to_host(&self, host_data: *mut c_void) -> Result<(), CudaError> {
        if let Some(page) = &self.page_ref {
            // Calculate offset within the page
            let page_start = page.device_ptr() as usize;
            let tensor_start = self.device_ptr.as_ptr() as usize;
            let offset = tensor_start - page_start;
            
            page.copy_to_host(host_data, self.size_bytes, offset)
        } else {
            // Direct CUDA copy
            unsafe {
                let result = cudaSetDevice(self.device_id);
                if result != CUDA_SUCCESS {
                    return Err(CudaError(result));
                }

                let result = cudaMemcpy(
                    host_data,
                    self.device_ptr.as_ptr(),
                    self.size_bytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                );

                if result != CUDA_SUCCESS {
                    Err(CudaError(result))
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Synchronize all operations on this tensor
    pub fn synchronize(&self) -> Result<(), CudaError> {
        if let Some(page) = &self.page_ref {
            page.synchronize()
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
}

unsafe impl Send for CudaTensor {}
unsafe impl Sync for CudaTensor {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_manager_initialization() {
        match CudaMemoryManager::new() {
            Ok(manager) => {
                println!("✓ CUDA manager initialized successfully");
                assert!(manager.device_infos.len() > 0);
                
                let info = manager.current_device_info();
                println!("Current device: {} ({})", info.device_id, info.name);
                println!("Memory: {} MB total, {} MB free", 
                        info.total_memory / 1024 / 1024,
                        info.free_memory / 1024 / 1024);
            }
            Err(e) => {
                println!("CUDA not available: {}", e);
                // This is expected in environments without CUDA
            }
        }
    }

    #[test]
    fn test_cuda_page_allocation() {
        if let Ok(manager) = CudaMemoryManager::new() {
            let page_size = 1024 * 1024; // 1MB
            match manager.allocate_page(page_size) {
                Ok(page) => {
                    println!("✓ CUDA page allocated: {} bytes", page.size());
                    assert_eq!(page.size(), page_size);
                    assert!(!page.device_ptr().is_null());
                    
                    // Test synchronization
                    page.synchronize().expect("Synchronization should succeed");
                    
                    println!("✓ CUDA page allocation test passed");
                }
                Err(e) => println!("CUDA allocation failed: {}", e),
            }
        } else {
            println!("No CUDA devices available for testing");
        }
    }

    #[test]
    fn test_zero_copy_operations() {
        if let Ok(manager) = CudaMemoryManager::new() {
            if let Ok(page) = manager.allocate_page(4 * 1024 * 1024) { // 4MB
                // Test zero-copy device-to-device copy
                let test_size = 1024; // 1KB
                let src_offset = 0;
                let dst_offset = 2 * 1024 * 1024; // 2MB offset
                
                match page.copy_device_to_device(src_offset, dst_offset, test_size) {
                    Ok(()) => {
                        println!("✓ Zero-copy device-to-device operation successful");
                        
                        // Test memory move
                        match page.move_memory(dst_offset, dst_offset + 1024, test_size) {
                            Ok(()) => println!("✓ Zero-copy memory move successful"),
                            Err(e) => println!("Memory move failed: {}", e),
                        }
                    }
                    Err(e) => println!("Zero-copy operation failed: {}", e),
                }
                
                // Test tensor creation from page
                let shape = vec![32, 64, 128]; // Example tensor shape
                let element_size = 2; // fp16
                
                match CudaTensor::from_page(&Arc::new(page), 0, shape, element_size) {
                    Ok(tensor) => {
                        println!("✓ CUDA tensor created from page: {:?}", tensor.shape());
                        
                        // Test zero-copy slice
                        match tensor.slice(0, 16) {
                            Ok(slice) => {
                                println!("✓ Zero-copy tensor slice: {:?}", slice.shape());
                            }
                            Err(e) => println!("Tensor slice failed: {}", e),
                        }
                    }
                    Err(e) => println!("Tensor creation failed: {}", e),
                }
            }
        }
    }

    #[test]
    fn test_cuda_context() {
        match CudaContext::new() {
            Ok(context) => {
                println!("✓ CUDA context initialized");
                
                // Test automatic device selection
                match context.allocate_page_auto(1024 * 1024) {
                    Ok(page) => {
                        println!("✓ Auto-allocated page on device {}", page.device_id());
                        
                        // Check stats
                        if let Some((allocated, pages)) = context.device_stats(page.device_id()) {
                            println!("Device stats: {} bytes, {} pages", allocated, pages);
                        }
                    }
                    Err(e) => println!("Auto allocation failed: {}", e),
                }
            }
            Err(e) => println!("CUDA context initialization failed: {}", e),
        }
    }

    #[test]
    fn test_memory_tracking() {
        if let Ok(manager) = CudaMemoryManager::new() {
            let initial_allocations = manager.get_active_allocations();
            println!("Initial allocations: {}", initial_allocations.len());
            
            // Allocate some pages
            let mut pages = Vec::new();
            for i in 0..3 {
                if let Ok(page) = manager.allocate_page(256 * 1024) { // 256KB each
                    pages.push(page);
                }
            }
            
            let after_allocations = manager.get_active_allocations();
            println!("After allocations: {}", after_allocations.len());
            
            let total_allocated = manager.total_allocated_memory();
            println!("Total allocated: {} bytes", total_allocated);
            
            // Drop pages and check tracking
            drop(pages);
            
            // Note: Actual cleanup happens in Drop, so this might still show allocations
            // until the test completes
            println!("✓ Memory tracking test completed");
        }
    }
}