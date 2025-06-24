// src/cuda/memory.rs - Memory management and pool
use super::bindings::*;
use super::error::CudaError;
use super::device::{CudaDeviceInfo, DeviceStats};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// CUDA memory manager with real device queries
#[derive(Debug)]
pub struct CudaMemoryManager {
    pub device_infos: Vec<CudaDeviceInfo>,
    current_device: i32,
    initialized: bool,
    allocation_stats: Arc<Mutex<HashMap<i32, DeviceStats>>>,
}

impl CudaMemoryManager {
    /// Initialize with REAL CUDA device detection
    pub fn new() -> Result<Self, CudaError> {
        // Initialize CUDA first
        super::diagnostics::initialize_cuda()?;
        
        #[cfg(cuda_available)]
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
        
        #[cfg(not(cuda_available))]
        {
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }

    /// Allocate CUDA page with real device memory
    pub fn allocate_page_on_device(&self, size: usize, device_id: i32) -> Result<super::allocator::CudaPage, CudaError> {
        if !self.initialized {
            return Err(CudaError(CUDA_ERROR_NOT_INITIALIZED));
        }
        
        let page = super::allocator::CudaPage::new(size, device_id)?;
        
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
        #[cfg(cuda_available)]
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
        
        #[cfg(not(cuda_available))]
        {
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
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
    
    /// Get device count
    pub fn device_count(&self) -> usize {
        self.device_infos.len()
    }
    
    /// Get current device
    pub fn current_device(&self) -> i32 {
        self.current_device
    }
    
    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
    
    /// Get allocation stats
    pub fn allocation_stats(&self) -> Arc<Mutex<HashMap<i32, DeviceStats>>> {
        Arc::clone(&self.allocation_stats)
    }
}

/// CUDA memory pool for efficient allocation
#[derive(Debug)]
pub struct CudaMemoryPool {
    pages: Vec<Arc<super::allocator::CudaPage>>,
    device_id: i32,
    page_size: usize,
    allocation_count: AtomicUsize,
}

impl CudaMemoryPool {
    /// Create new memory pool
    pub fn new(device_id: i32, page_size: usize, initial_pages: usize) -> Result<Self, CudaError> {
        let mut pages = Vec::with_capacity(initial_pages);
        
        for _ in 0..initial_pages {
            let page = super::allocator::CudaPage::new(page_size, device_id)?;
            pages.push(Arc::new(page));
        }
        
        Ok(CudaMemoryPool {
            pages,
            device_id,
            page_size,
            allocation_count: AtomicUsize::new(0),
        })
    }
    
    /// Allocate from pool
    pub fn allocate(&self, size: usize, align: usize) -> Option<(Arc<super::allocator::CudaPage>, std::ptr::NonNull<u8>)> {
        for page in &self.pages {
            if let Some(ptr) = page.allocate(size, align) {
                self.allocation_count.fetch_add(1, Ordering::Relaxed);
                return Some((Arc::clone(page), ptr));
            }
        }
        None
    }
    
    /// Add new page to pool
    pub fn expand(&mut self) -> Result<(), CudaError> {
        let page = super::allocator::CudaPage::new(self.page_size, self.device_id)?;
        self.pages.push(Arc::new(page));
        Ok(())
    }
    
    /// Reset all pages
    pub fn reset(&self) {
        for page in &self.pages {
            page.reset();
        }
        self.allocation_count.store(0, Ordering::Relaxed);
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> (usize, usize, usize) {
        let total_pages = self.pages.len();
        let total_size = total_pages * self.page_size;
        let allocations = self.allocation_count.load(Ordering::Relaxed);
        (total_pages, total_size, allocations)
    }
    
    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
    
    /// Get page size
    pub fn page_size(&self) -> usize {
        self.page_size
    }
    
    /// Get pages
    pub fn pages(&self) -> &[Arc<super::allocator::CudaPage>] {
        &self.pages
    }
}