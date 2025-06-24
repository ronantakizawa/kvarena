// src/cuda/memory.rs - Fixed memory management with safe initialization
use super::bindings::*;
use super::error::CudaError;
use super::device::{CudaDeviceInfo, DeviceStats, detect_cuda_devices};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// CUDA memory manager with safe device detection and initialization
#[derive(Debug)]
pub struct CudaMemoryManager {
    pub device_infos: Vec<CudaDeviceInfo>,
    current_device: i32,
    initialized: bool,
    allocation_stats: Arc<Mutex<HashMap<i32, DeviceStats>>>,
}

impl CudaMemoryManager {
    /// Initialize with safe CUDA device detection
    pub fn new() -> Result<Self, CudaError> {
        log::info!("🚀 Creating CUDA memory manager...");
        
        // Use safe initialization instead of direct CUDA calls
        let init_result = super::init::safe_cuda_init();
        
        if !init_result.available {
            log::error!("CUDA not available for memory manager");
            if let Some(error) = init_result.error_message {
                log::error!("CUDA error: {}", error);
            }
            return Err(CudaError(CUDA_ERROR_NOT_INITIALIZED));
        }
        
        log::info!("CUDA available, proceeding with memory manager creation...");
        
        #[cfg(cuda_available)]
        {
            // Use safe device detection instead of direct CUDA calls
            let device_infos = detect_cuda_devices();
            
            if device_infos.is_empty() {
                log::error!("No CUDA devices detected during memory manager initialization");
                return Err(CudaError(CUDA_ERROR_INVALID_DEVICE));
            }

            // Get current device safely
            let current_device = match super::diagnostics::safe_cuda_call(
                || {
                    unsafe {
                        let mut current_device = 0;
                        let result = cudaGetDevice(&mut current_device);
                        if result != CUDA_SUCCESS {
                            Err(CudaError(result))
                        } else {
                            Ok(current_device)
                        }
                    }
                },
                5, // 5 second timeout
                "cudaGetDevice".to_string()
            ) {
                Ok(device) => device,
                Err(e) => {
                    log::warn!("Failed to get current device, defaulting to 0: {}", e);
                    0 // Default to device 0
                }
            };

            // Log device information
            for info in &device_infos {
                log::info!("✓ CUDA device {}: {} ({} MB total, {} MB free)", 
                          info.device_id, info.name, 
                          info.total_memory / 1024 / 1024,
                          info.free_memory / 1024 / 1024);
                
                if info.is_t4() {
                    log::info!("  🎯 Tesla T4 detected with {:.1} GB/s memory bandwidth", 
                              info.memory_bandwidth_gbps());
                }
            }

            log::info!("✅ CUDA memory manager initialized with {} device(s)", device_infos.len());

            Ok(CudaMemoryManager {
                device_infos,
                current_device,
                initialized: true,
                allocation_stats: Arc::new(Mutex::new(HashMap::new())),
            })
        }
        
        #[cfg(not(cuda_available))]
        {
            log::error!("CUDA not available in this build");
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }

    /// Allocate CUDA page with safe error handling
    pub fn allocate_page_on_device(&self, size: usize, device_id: i32) -> Result<super::allocator::CudaPage, CudaError> {
        if !self.initialized {
            log::error!("Memory manager not initialized");
            return Err(CudaError(CUDA_ERROR_NOT_INITIALIZED));
        }
        
        // Validate device ID
        if !self.device_infos.iter().any(|info| info.device_id == device_id) {
            log::error!("Invalid device ID: {}", device_id);
            return Err(CudaError(CUDA_ERROR_INVALID_DEVICE));
        }
        
        // Validate size (prevent excessive allocations)
        if size == 0 {
            log::error!("Cannot allocate zero-sized page");
            return Err(CudaError(CUDA_ERROR_INVALID_VALUE));
        }
        
        if size > 1024 * 1024 * 1024 { // > 1GB
            log::warn!("Large allocation requested: {} MB on device {}", size / 1024 / 1024, device_id);
        }
        
        // Create page with error handling
        let page = match super::allocator::CudaPage::new(size, device_id) {
            Ok(page) => page,
            Err(e) => {
                log::error!("Failed to allocate {}KB page on device {}: {}", 
                           size / 1024, device_id, e);
                return Err(e);
            }
        };
        
        // Update stats
        if let Ok(mut stats) = self.allocation_stats.lock() {
            let device_stats = stats.entry(device_id).or_default();
            device_stats.total_allocated += size;
            device_stats.allocation_count += 1;
            if device_stats.total_allocated > device_stats.peak_allocated {
                device_stats.peak_allocated = device_stats.total_allocated;
            }
        }
        
        log::debug!("✓ Allocated {}KB page on device {} (total: {} pages)", 
                   size / 1024, device_id, 
                   self.allocation_stats.lock().map(|s| s.get(&device_id).map(|ds| ds.allocation_count).unwrap_or(0)).unwrap_or(0));
        
        Ok(page)
    }

    /// Get device information safely
    pub fn device_info(&self, device_id: i32) -> Option<&CudaDeviceInfo> {
        self.device_infos.iter().find(|info| info.device_id == device_id)
    }
    
    /// Get all device infos
    pub fn devices(&self) -> &[CudaDeviceInfo] {
        &self.device_infos
    }
    
    /// Get current memory usage on device with timeout protection
    pub fn get_memory_info(&self, device_id: i32) -> Result<(usize, usize), CudaError> {
        #[cfg(cuda_available)]
        {
            let device_info_fallback = self.device_info(device_id).cloned();
            
            super::diagnostics::safe_cuda_call(
                move || {
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
                        
                        // Sanity check values
                        if total == 0 || total > 200 * 1024 * 1024 * 1024 { // > 200GB is suspicious
                            log::warn!("Suspicious memory values for device {}: {} total", device_id, total);
                            // Use device info as fallback
                            if let Some(info) = device_info_fallback {
                                return Ok((info.free_memory, info.total_memory));
                            }
                        }
                        
                        Ok((free, total))
                    }
                },
                5, // 5 second timeout
                "cudaMemGetInfo".to_string()
            )
        }
        
        #[cfg(not(cuda_available))]
        {
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }
    
    /// Record deallocation safely
    pub fn record_deallocation(&self, size: usize, device_id: i32) {
        if let Ok(mut stats) = self.allocation_stats.lock() {
            if let Some(device_stats) = stats.get_mut(&device_id) {
                device_stats.total_allocated = device_stats.total_allocated.saturating_sub(size);
                log::trace!("Recorded deallocation: {}KB on device {}", size / 1024, device_id);
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
    
    /// Get memory statistics for all devices
    pub fn get_memory_statistics(&self) -> HashMap<i32, (usize, usize, usize)> {
        let mut stats = HashMap::new();
        
        for device_info in &self.device_infos {
            let device_id = device_info.device_id;
            
            // Get current memory info
            let (free, total) = self.get_memory_info(device_id)
                .unwrap_or((device_info.free_memory, device_info.total_memory));
            
            // Get allocation stats
            let allocated = if let Ok(allocation_stats) = self.allocation_stats.lock() {
                allocation_stats.get(&device_id)
                    .map(|ds| ds.total_allocated)
                    .unwrap_or(0)
            } else {
                0
            };
            
            stats.insert(device_id, (free, total, allocated));
        }
        
        stats
    }
    
    /// Check memory pressure across all devices
    pub fn check_memory_pressure(&self) -> f64 {
        let mut total_used = 0usize;
        let mut total_available = 0usize;
        
        for device_info in &self.device_infos {
            if let Ok((free, total)) = self.get_memory_info(device_info.device_id) {
                total_used += total - free;
                total_available += total;
            }
        }
        
        if total_available > 0 {
            total_used as f64 / total_available as f64
        } else {
            0.0
        }
    }
}

/// CUDA memory pool with safe allocation patterns
#[derive(Debug)]
pub struct CudaMemoryPool {
    pages: Vec<Arc<super::allocator::CudaPage>>,
    device_id: i32,
    page_size: usize,
    allocation_count: AtomicUsize,
    max_pages: usize,
}

impl CudaMemoryPool {
    /// Create new memory pool with limits
    pub fn new(device_id: i32, page_size: usize, initial_pages: usize, max_pages: usize) -> Result<Self, CudaError> {
        let mut pages = Vec::with_capacity(initial_pages);
        
        log::info!("Creating memory pool: {} initial pages of {}KB each on device {}", 
                  initial_pages, page_size / 1024, device_id);
        
        for i in 0..initial_pages {
            match super::allocator::CudaPage::new(page_size, device_id) {
                Ok(page) => {
                    pages.push(Arc::new(page));
                    log::trace!("Created pool page {}/{}", i + 1, initial_pages);
                }
                Err(e) => {
                    log::error!("Failed to create pool page {}/{}: {}", i + 1, initial_pages, e);
                    return Err(e);
                }
            }
        }
        
        log::info!("✓ Created memory pool with {} pages on device {}", pages.len(), device_id);
        
        Ok(CudaMemoryPool {
            pages,
            device_id,
            page_size,
            allocation_count: AtomicUsize::new(0),
            max_pages,
        })
    }
    
    /// Allocate from pool with proper alignment
    pub fn allocate(&self, size: usize, align: usize) -> Option<(Arc<super::allocator::CudaPage>, std::ptr::NonNull<u8>)> {
        for page in &self.pages {
            if let Some(ptr) = page.allocate(size, align) {
                self.allocation_count.fetch_add(1, Ordering::Relaxed);
                log::trace!("Pool allocation: {}B at {:p}", size, ptr.as_ptr());
                return Some((Arc::clone(page), ptr));
            }
        }
        
        log::debug!("Pool allocation failed: no space for {}B in {} pages", size, self.pages.len());
        None
    }
    
    /// Add new page to pool (if under limit)
    pub fn expand(&mut self) -> Result<bool, CudaError> {
        if self.pages.len() >= self.max_pages {
            log::debug!("Cannot expand pool: at maximum {} pages", self.max_pages);
            return Ok(false);
        }
        
        match super::allocator::CudaPage::new(self.page_size, self.device_id) {
            Ok(page) => {
                self.pages.push(Arc::new(page));
                log::debug!("Expanded pool to {} pages", self.pages.len());
                Ok(true)
            }
            Err(e) => {
                log::error!("Failed to expand pool: {}", e);
                Err(e)
            }
        }
    }
    
    /// Reset all pages safely
    pub fn reset(&self) {
        log::debug!("Resetting memory pool with {} pages", self.pages.len());
        for (i, page) in self.pages.iter().enumerate() {
            page.reset();
            log::trace!("Reset pool page {}", i);
        }
        self.allocation_count.store(0, Ordering::Relaxed);
        log::debug!("✓ Memory pool reset complete");
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> (usize, usize, usize, f64) {
        let total_pages = self.pages.len();
        let total_size = total_pages * self.page_size;
        let allocations = self.allocation_count.load(Ordering::Relaxed);
        
        // Calculate utilization
        let mut total_used = 0;
        for page in &self.pages {
            total_used += page.current_offset();
        }
        
        let utilization = if total_size > 0 {
            total_used as f64 / total_size as f64
        } else {
            0.0
        };
        
        (total_pages, total_size, allocations, utilization)
    }
    
    /// Get detailed page statistics
    pub fn page_stats(&self) -> Vec<(usize, usize, f64)> {
        self.pages.iter().map(|page| {
            let used = page.current_offset();
            let total = page.size();
            let util = if total > 0 { used as f64 / total as f64 } else { 0.0 };
            (used, total, util)
        }).collect()
    }
    
    /// Cleanup underutilized pages (keep at least min_pages)
    pub fn cleanup(&mut self, min_pages: usize, min_utilization: f64) -> usize {
        if self.pages.len() <= min_pages {
            return 0;
        }
        
        let mut removed = 0;
        let initial_len = self.pages.len();
        
        // Use drain_filter-like approach to avoid borrowing issues
        let mut i = 0;
        while i < self.pages.len() && self.pages.len() > min_pages {
            let should_remove = {
                let page = &self.pages[i];
                page.utilization() < min_utilization
            };
            
            if should_remove {
                let page = self.pages.remove(i);
                log::debug!("Removing underutilized page: {:.1}% utilization", 
                           page.utilization() * 100.0);
                removed += 1;
            } else {
                i += 1;
            }
        }
        
        if removed > 0 {
            log::info!("Cleaned up {} underutilized pages, {} remaining", removed, self.pages.len());
        }
        
        removed
    }
    
    // Getters
    pub fn device_id(&self) -> i32 { self.device_id }
    pub fn page_size(&self) -> usize { self.page_size }
    pub fn page_count(&self) -> usize { self.pages.len() }
    pub fn max_pages(&self) -> usize { self.max_pages }
    pub fn pages(&self) -> &[Arc<super::allocator::CudaPage>] { &self.pages }
}