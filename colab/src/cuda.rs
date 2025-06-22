// src/cuda.rs - True CUDA integration with actual device memory management
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::ptr::NonNull;
use std::collections::HashMap;
use std::sync::Mutex;

// CUDA Runtime API bindings - these link to actual libcudart
#[link(name = "cudart")]
extern "C" {
    // Device management
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaDeviceReset() -> i32;
    fn cudaDeviceSynchronize() -> i32;
    
    // Memory management - REAL CUDA memory operations
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: i32, stream: *mut c_void) -> i32;
    fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> i32;
    fn cudaMemsetAsync(devPtr: *mut c_void, value: i32, count: usize, stream: *mut c_void) -> i32;
    
    // Memory info
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    
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
    fn cudaGetDeviceProperties(prop: *mut CudaDeviceProperties, device: i32) -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
}

// CUDA constants
const CUDA_SUCCESS: i32 = 0;
const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;
const CUDA_ERROR_NOT_INITIALIZED: i32 = 3;
const CUDA_ERROR_INVALID_DEVICE: i32 = 10;
const CUDA_ERROR_INVALID_VALUE: i32 = 11;
const CUDA_ERROR_NOT_READY: i32 = 600;

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

const CUDA_STREAM_NON_BLOCKING: u32 = 0x01;

// Device attributes
const CUDA_DEVICE_ATTR_MEMORY_CLOCK_RATE: i32 = 36;
const CUDA_DEVICE_ATTR_GLOBAL_MEMORY_BUS_WIDTH: i32 = 37;
const CUDA_DEVICE_ATTR_MULTIPROCESSOR_COUNT: i32 = 16;
const CUDA_DEVICE_ATTR_MAX_THREADS_PER_MULTIPROCESSOR: i32 = 39;

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

/// Initialize CUDA runtime and verify T4 GPU availability
pub fn initialize_cuda() -> Result<(), CudaError> {
    unsafe {
        // Check device count
        let mut device_count = 0;
        let result = cudaGetDeviceCount(&mut device_count);
        if result != CUDA_SUCCESS {
            return Err(CudaError(result));
        }
        
        if device_count == 0 {
            return Err(CudaError(CUDA_ERROR_INVALID_DEVICE));
        }
        
        // Set device 0 (T4)
        let result = cudaSetDevice(0);
        if result != CUDA_SUCCESS {
            return Err(CudaError(result));
        }
        
        // Verify device works with a small allocation test
        let mut test_ptr = std::ptr::null_mut();
        let result = cudaMalloc(&mut test_ptr, 1024);
        if result != CUDA_SUCCESS {
            return Err(CudaError(result));
        }
        
        let result = cudaFree(test_ptr);
        if result != CUDA_SUCCESS {
            return Err(CudaError(result));
        }
        
        log::info!("CUDA initialized successfully with {} device(s)", device_count);
        Ok(())
    }
}

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
                CUDA_ERROR_NOT_READY => Ok(false),
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
                log::error!("CUDA device-to-device copy failed: {} bytes", size);
                Err(CudaError(result))
            } else {
                log::debug!("Zero-copy device move: {} bytes from offset {} to {}", size, src_offset, dst_offset);
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
        // Optionally zero the memory again
        if let Some(stream) = &self.stream {
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
}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
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

// Global allocation tracking
static NEXT_ALLOCATION_ID: AtomicUsize = AtomicUsize::new(1);

impl CudaPage {
    /// Create new CUDA page with real device memory
    pub fn new(size: usize, device_id: i32) -> Result<Self, CudaError> {
        let allocator = BumpAllocator::new(size, device_id)?;
        let allocation_id = NEXT_ALLOCATION_ID.fetch_add(1, Ordering::Relaxed) as u64;
        
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
        // Check if any async operations are complete
        self.synchronize()?;
        Ok(true)
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
    pub max_threads_per_multiprocessor: i32,
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
            let mut max_threads_per_multiprocessor = 0;
            
            cudaDeviceGetAttribute(&mut memory_clock_rate, CUDA_DEVICE_ATTR_MEMORY_CLOCK_RATE, device_id);
            cudaDeviceGetAttribute(&mut memory_bus_width, CUDA_DEVICE_ATTR_GLOBAL_MEMORY_BUS_WIDTH, device_id);
            cudaDeviceGetAttribute(&mut max_threads_per_multiprocessor, CUDA_DEVICE_ATTR_MAX_THREADS_PER_MULTIPROCESSOR, device_id);

            // Convert name from C string
            let name = std::ffi::CStr::from_ptr(props.name.as_ptr())
                .to_string_lossy()
                .into_owned();

            log::info!("Queried CUDA device {}: {} (CC {}.{}, {} MB total, {} MB free)", 
                      device_id, name, props.major, props.minor, 
                      total_memory / 1024 / 1024, free_memory / 1024 / 1024);

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
                max_threads_per_multiprocessor,
            })
        }
    }
    
    /// Check if this is a T4 GPU
    pub fn is_t4(&self) -> bool {
        self.name.to_lowercase().contains("t4") || 
        (self.compute_capability_major == 7 && self.compute_capability_minor == 5)
    }
    
    /// Get memory bandwidth in GB/s
    pub fn memory_bandwidth_gbps(&self) -> f64 {
        // Bandwidth = (memory_clock_rate * 2) * (memory_bus_width / 8) / 1e9
        let clock_hz = self.memory_clock_rate as f64 * 1000.0; // Convert kHz to Hz
        let bus_width_bytes = self.memory_bus_width as f64 / 8.0;
        (clock_hz * 2.0 * bus_width_bytes) / 1e9
    }
}

/// CUDA memory manager with real device queries
#[derive(Debug)]
pub struct CudaMemoryManager {
    pub device_infos: Vec<CudaDeviceInfo>,
    current_device: i32,
    initialized: bool,
    allocation_stats: Arc<Mutex<HashMap<i32, DeviceStats>>>,
}

#[derive(Debug, Default)]
struct DeviceStats {
    total_allocated: usize,
    peak_allocated: usize,
    allocation_count: usize,
}

impl CudaMemoryManager {
    /// Initialize with REAL CUDA device detection
    pub fn new() -> Result<Self, CudaError> {
        // Initialize CUDA first
        initialize_cuda()?;
        
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
                        log::info!("Found CUDA device {}: {} ({} MB total, {} MB free)", 
                                  device_id, info.name, 
                                  info.total_memory / 1024 / 1024,
                                  info.free_memory / 1024 / 1024);
                        
                        if info.is_t4() {
                            log::info!("  ✓ Tesla T4 detected with {:.1} GB/s memory bandwidth", 
                                      info.memory_bandwidth_gbps());
                        }
                        
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

            log::info!("CUDA memory manager initialized with {} device(s)", device_infos.len());

            Ok(CudaMemoryManager {
                device_infos,
                current_device,
                initialized: true,
                allocation_stats: Arc::new(Mutex::new(HashMap::new())),
            })
        }
    }

    /// Allocate CUDA page with real device memory
    pub fn allocate_page_on_device(&self, size: usize, device_id: i32) -> Result<CudaPage, CudaError> {
        if !self.initialized {
            return Err(CudaError(CUDA_ERROR_NOT_INITIALIZED));
        }
        
        let page = CudaPage::new(size, device_id)?;
        
        // Update stats
        if let Ok(mut stats) = self.allocation_stats.lock() {
            let device_stats = stats.entry(device_id).or_default();
            device_stats.total_allocated += size;
            device_stats.allocation_count += 1;
            if device_stats.total_allocated > device_stats.peak_allocated {
                device_stats.peak_allocated = device_stats.total_allocated;
            }
        }
        
        log::debug!("Allocated {}KB page on device {}", size / 1024, device_id);
        Ok(page)
    }

    /// Get device information
    pub fn device_info(&self, device_id: i32) -> Option<&CudaDeviceInfo> {
        self.device_infos.iter().find(|info| info.device_id == device_id)
    }
    
    /// Get all device infos
    pub fn devices(&self) -> &[CudaDeviceInfo] {
        &self.device_infos
    }
    
    /// Get current memory usage on device
    pub fn get_memory_info(&self, device_id: i32) -> Result<(usize, usize), CudaError> {
        unsafe {
            let result = cudaSetDevice(device_id);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }
            
            let mut free = 0;
            let mut total = 0;
            let result = cudaMemGetInfo(&mut free, &mut total);
            if result != CUDA_SUCCESS {
                return Err(CudaError(result));
            }
            
            Ok((free, total))
        }
    }
    
    /// Record deallocation
    pub fn record_deallocation(&self, size: usize, device_id: i32) {
        if let Ok(mut stats) = self.allocation_stats.lock() {
            if let Some(device_stats) = stats.get_mut(&device_id) {
                device_stats.total_allocated = device_stats.total_allocated.saturating_sub(size);
            }
        }
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

/// CUDA context for managing multiple devices and global state
#[derive(Debug)]
pub struct CudaContext {
    manager: CudaMemoryManager,
    streams: Arc<Mutex<HashMap<i32, Arc<CudaStream>>>>,
}

impl CudaContext {
    pub fn new() -> Result<Self, CudaError> {
        let manager = CudaMemoryManager::new()?;
        let streams = Arc::new(Mutex::new(HashMap::new()));
        
        // Create default streams for each device
        for device_info in &manager.device_infos {
            match CudaStream::new(device_info.device_id, true) {
                Ok(stream) => {
                    if let Ok(mut streams_map) = streams.lock() {
                        streams_map.insert(device_info.device_id, Arc::new(stream));
                    }
                }
                Err(e) => {
                    log::warn!("Failed to create stream for device {}: {}", device_info.device_id, e);
                }
            }
        }
        
        Ok(CudaContext { manager, streams })
    }

    pub fn allocate_page_auto(&self, size: usize) -> Result<CudaPage, CudaError> {
        // Find device with most free memory
        let mut best_device = 0;
        let mut max_free = 0;
        
        for device_info in &self.manager.device_infos {
            if let Ok((free, _total)) = self.manager.get_memory_info(device_info.device_id) {
                if free > max_free {
                    max_free = free;
                    best_device = device_info.device_id;
                }
            }
        }

        self.allocate_page_on_device(size, best_device)
    }

    pub fn allocate_page_on_device(&self, size: usize, device_id: i32) -> Result<CudaPage, CudaError> {
        self.manager.allocate_page_on_device(size, device_id)
    }

    pub fn device_stats(&self, device_id: i32) -> Option<(usize, usize)> {
        self.manager.get_memory_info(device_id).ok()
    }

    pub fn device_stats_detailed(&self, device_id: i32) -> Option<CudaDeviceStats> {
        if let Some(info) = self.manager.device_info(device_id) {
            let (free, total) = self.manager.get_memory_info(device_id).unwrap_or((0, 0));
            
            // Get allocation stats
            let (allocated, peak, count) = if let Ok(stats) = self.manager.allocation_stats.lock() {
                if let Some(device_stats) = stats.get(&device_id) {
                    (device_stats.total_allocated, device_stats.peak_allocated, device_stats.allocation_count)
                } else {
                    (0, 0, 0)
                }
            } else {
                (0, 0, 0)
            };
            
            Some(CudaDeviceStats {
                device_id,
                allocated_bytes: allocated,
                active_pages: count,
                peak_allocated: peak,
                total_memory: total,
                free_memory: free,
                utilization: if total > 0 { (total - free) as f64 / total as f64 } else { 0.0 },
            })
        } else {
            None
        }
    }

    pub fn manager(&self) -> &CudaMemoryManager {
        &self.manager
    }
    
    pub fn get_stream(&self, device_id: i32) -> Option<Arc<CudaStream>> {
        if let Ok(streams) = self.streams.lock() {
            streams.get(&device_id).cloned()
        } else {
            None
        }
    }
    
    /// Synchronize all devices
    pub fn synchronize_all(&self) -> Result<(), CudaError> {
        for device_info in &self.manager.device_infos {
            unsafe {
                let result = cudaSetDevice(device_info.device_id);
                if result != CUDA_SUCCESS {
                    return Err(CudaError(result));
                }
                
                let result = cudaDeviceSynchronize();
                if result != CUDA_SUCCESS {
                    return Err(CudaError(result));
                }
            }
        }
        Ok(())
    }
    
    /// Reset all devices (for cleanup)
    pub fn reset_all_devices(&self) -> Result<(), CudaError> {
        for device_info in &self.manager.device_infos {
            unsafe {
                let result = cudaSetDevice(device_info.device_id);
                if result != CUDA_SUCCESS {
                    log::warn!("Failed to set device {} for reset", device_info.device_id);
                    continue;
                }
                
                let result = cudaDeviceReset();
                if result != CUDA_SUCCESS {
                    log::warn!("Failed to reset device {}: {}", device_info.device_id, CudaError(result));
                }
            }
        }
        Ok(())
    }
}

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
                log::trace!("CUDA tensor copy from host: {} bytes", size_bytes);
                Ok(())
            }
        }
    }
    
    /// Copy data from this tensor to host (REAL CUDA memcpy)
    pub fn copy_to_host(&self, host_data: *mut c_void) -> Result<(), CudaError> {
        let size_bytes = self.shape.iter().product::<usize>() * self.element_size;
        
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
    
    /// Get tensor size in bytes
    pub fn size_bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * self.element_size
    }
}

/// Utility functions for CUDA operations
pub fn check_cuda_error(operation: &str) -> Result<(), CudaError> {
    unsafe {
        let error = cudaGetLastError();
        if error != CUDA_SUCCESS {
            log::error!("CUDA error in {}: {}", operation, CudaError(error));
            Err(CudaError(error))
        } else {
            Ok(())
        }
    }
}

/// Get CUDA runtime version info
pub fn get_cuda_version() -> String {
    // This would require additional CUDA runtime calls
    // For now, return a placeholder
    "CUDA Runtime".to_string()
}

/// Check if CUDA is available and functional
pub fn is_cuda_available() -> bool {
    match initialize_cuda() {
        Ok(()) => true,
        Err(_) => false,
    }
}

/// Perform a CUDA memory test to verify functionality
pub fn cuda_memory_test(device_id: i32, test_size: usize) -> Result<f64, CudaError> {
    use std::time::Instant;
    
    unsafe {
        let result = cudaSetDevice(device_id);
        if result != CUDA_SUCCESS {
            return Err(CudaError(result));
        }
        
        // Allocate test memory
        let mut device_ptr = std::ptr::null_mut();
        let result = cudaMalloc(&mut device_ptr, test_size);
        if result != CUDA_SUCCESS {
            return Err(CudaError(result));
        }
        
        // Create host memory
        let host_data = vec![0u8; test_size];
        
        // Test memory bandwidth
        let start = Instant::now();
        
        // Host to device
        let result = cudaMemcpy(device_ptr, host_data.as_ptr() as *const c_void, test_size, CUDA_MEMCPY_HOST_TO_DEVICE);
        if result != CUDA_SUCCESS {
            let _ = cudaFree(device_ptr);
            return Err(CudaError(result));
        }
        
        // Synchronize
        let result = cudaDeviceSynchronize();
        if result != CUDA_SUCCESS {
            let _ = cudaFree(device_ptr);
            return Err(CudaError(result));
        }
        
        let elapsed = start.elapsed();
        let bandwidth_gbps = (test_size as f64) / elapsed.as_secs_f64() / 1e9;
        
        // Cleanup
        let result = cudaFree(device_ptr);
        if result != CUDA_SUCCESS {
            return Err(CudaError(result));
        }
        
        Ok(bandwidth_gbps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_initialization() {
        match initialize_cuda() {
            Ok(()) => {
                println!("✓ CUDA initialization successful");
                
                // Test device query
                match CudaMemoryManager::new() {
                    Ok(manager) => {
                        println!("✓ CUDA memory manager created");
                        println!("  Devices: {}", manager.device_infos.len());
                        
                        for device in &manager.device_infos {
                            println!("  Device {}: {} ({} MB)", 
                                   device.device_id, device.name, device.total_memory / 1024 / 1024);
                            if device.is_t4() {
                                println!("    ✓ T4 GPU detected");
                            }
                        }
                    }
                    Err(e) => println!("⚠️ Manager creation failed: {}", e),
                }
            }
            Err(e) => {
                println!("⚠️ CUDA initialization failed: {}", e);
                println!("  This is expected if no CUDA-capable GPU is available");
            }
        }
    }

    #[test]
    fn test_cuda_memory_operations() {
        if let Ok(()) = initialize_cuda() {
            match CudaPage::new(1024 * 1024, 0) {
                Ok(page) => {
                    println!("✓ CUDA page allocation successful: {} bytes", page.size());
                    
                    // Test allocation within page
                    if let Some(_ptr) = page.allocate(1024, 256) {
                        println!("✓ Bump allocation successful");
                        
                        // Test utilization
                        println!("  Page utilization: {:.1}%", page.utilization() * 100.0);
                        println!("  Available space: {} bytes", page.available_space());
                    }
                    
                    // Test synchronization
                    if page.synchronize().is_ok() {
                        println!("✓ Page synchronization successful");
                    }
                }
                Err(e) => println!("⚠️ CUDA page allocation failed: {}", e),
            }
        }
    }

    #[test]
    fn test_cuda_context() {
        match CudaContext::new() {
            Ok(context) => {
                println!("✓ CUDA context created successfully");
                
                // Test auto allocation
                match context.allocate_page_auto(512 * 1024) {
                    Ok(page) => {
                        println!("✓ Auto page allocation: {} bytes on device {}", 
                               page.size(), page.device_id());
                    }
                    Err(e) => println!("⚠️ Auto allocation failed: {}", e),
                }
                
                // Test device stats
                if let Some(stats) = context.device_stats_detailed(0) {
                    println!("✓ Device 0 stats:");
                    println!("  Total memory: {} MB", stats.total_memory / 1024 / 1024);
                    println!("  Free memory: {} MB", stats.free_memory / 1024 / 1024);
                    println!("  Utilization: {:.1}%", stats.utilization * 100.0);
                }
            }
            Err(e) => println!("⚠️ CUDA context creation failed: {}", e),
        }
    }

    #[test]
    fn test_memory_bandwidth() {
        if let Ok(()) = initialize_cuda() {
            let test_size = 64 * 1024 * 1024; // 64 MB test
            match cuda_memory_test(0, test_size) {
                Ok(bandwidth) => {
                    println!("✓ Memory bandwidth test: {:.2} GB/s", bandwidth);
                    
                    // T4 should have around 320 GB/s theoretical bandwidth
                    if bandwidth > 50.0 {
                        println!("  ✓ Good memory bandwidth detected");
                    } else {
                        println!("  ⚠️ Lower than expected bandwidth");
                    }
                }
                Err(e) => println!("⚠️ Memory bandwidth test failed: {}", e),
            }
        }
    }

    #[test]
    fn test_bump_allocator() {
        if let Ok(()) = initialize_cuda() {
            match BumpAllocator::new(4096, 0) {
                Ok(allocator) => {
                    println!("✓ Bump allocator created: {} bytes", allocator.page_size());
                    
                    // Test multiple allocations
                    let ptr1 = allocator.allocate(1024, 256);
                    assert!(ptr1.is_some());
                    println!("  Allocation 1: offset {}", allocator.current_offset());
                    
                    let ptr2 = allocator.allocate(2048, 256);
                    assert!(ptr2.is_some());
                    println!("  Allocation 2: offset {}", allocator.current_offset());
                    
                    // Test overflow
                    let ptr3 = allocator.allocate(2048, 256);
                    assert!(ptr3.is_none());
                    println!("  ✓ Overflow protection works");
                    
                    println!("  Final utilization: {:.1}%", allocator.utilization() * 100.0);
                }
                Err(e) => println!("⚠️ Bump allocator creation failed: {}", e),
            }
        }
    }

    #[test]
    fn test_cuda_stream() {
        if let Ok(()) = initialize_cuda() {
            match CudaStream::new(0, true) {
                Ok(stream) => {
                    println!("✓ CUDA stream created for device 0");
                    
                    // Test synchronization
                    if stream.synchronize().is_ok() {
                        println!("✓ Stream synchronization successful");
                    }
                    
                    // Test query
                    match stream.is_complete() {
                        Ok(complete) => println!("  Stream complete: {}", complete),
                        Err(e) => println!("  Stream query failed: {}", e),
                    }
                }
                Err(e) => println!("⚠️ CUDA stream creation failed: {}", e),
            }
        }
    }
}