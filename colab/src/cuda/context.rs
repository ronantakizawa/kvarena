// src/cuda/context.rs - CUDA context management
use super::bindings::*;
use super::error::CudaError;
use super::memory::CudaMemoryManager;
use super::allocator::CudaPage;
use super::stream::CudaStream;
use super::device::{CudaDeviceStats, CudaDeviceInfo};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

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
            let (allocated, peak, count) = if let Ok(stats) = self.manager.allocation_stats().lock() {
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
            #[cfg(cuda_available)]
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
            #[cfg(cuda_available)]
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
    
    /// Get all streams
    pub fn streams(&self) -> Arc<Mutex<HashMap<i32, Arc<CudaStream>>>> {
        Arc::clone(&self.streams)
    }
}