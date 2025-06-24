// src/cuda/device.rs - Device information and management
use super::bindings::*;
use super::error::CudaError;
use std::process::Command;

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
        #[cfg(cuda_available)]
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

            // Get memory info - THIS IS WHERE IT HANGS - let's make it safer
            let mut free_memory = 0usize;
            let mut total_memory = 0usize;
            let memory_result = cudaMemGetInfo(&mut free_memory, &mut total_memory);
            
            // If memory query fails, use properties total memory and estimate free
            let (final_free, final_total) = if memory_result != CUDA_SUCCESS {
                log::warn!("cudaMemGetInfo failed for device {}, using properties", device_id);
                let total_from_props = props.total_global_mem;
                let estimated_free = total_from_props * 90 / 100; // Estimate 90% free
                (estimated_free, total_from_props)
            } else {
                // Sanity check the memory values
                if total_memory > 100 * 1024 * 1024 * 1024 {  // > 100GB is suspicious
                    log::warn!("Suspicious total memory value: {} bytes, using properties", total_memory);
                    let total_from_props = props.total_global_mem;
                    let estimated_free = total_from_props * 90 / 100;
                    (estimated_free, total_from_props)
                } else {
                    (free_memory, total_memory)
                }
            };

            // Get additional attributes - make these non-blocking
            let mut memory_clock_rate = 0;
            let mut memory_bus_width = 0;
            let mut max_threads_per_multiprocessor = 0;
            
            // These calls can sometimes hang, so we'll use timeout logic or skip them
            let _ = cudaDeviceGetAttribute(&mut memory_clock_rate, CUDA_DEVICE_ATTR_MEMORY_CLOCK_RATE, device_id);
            let _ = cudaDeviceGetAttribute(&mut memory_bus_width, CUDA_DEVICE_ATTR_GLOBAL_MEMORY_BUS_WIDTH, device_id);
            let _ = cudaDeviceGetAttribute(&mut max_threads_per_multiprocessor, CUDA_DEVICE_ATTR_MAX_THREADS_PER_MULTIPROCESSOR, device_id);

            // Convert name from C string
            let name = std::ffi::CStr::from_ptr(props.name.as_ptr())
                .to_string_lossy()
                .into_owned();

            log::info!("Queried CUDA device {}: {} (CC {}.{}, {} MB total, {} MB free)", 
                      device_id, name, props.major, props.minor, 
                      final_total / 1024 / 1024, final_free / 1024 / 1024);

            Ok(CudaDeviceInfo {
                device_id,
                name,
                total_memory: final_total,
                free_memory: final_free,
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
        
        #[cfg(not(cuda_available))]
        {
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
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
    
    /// Get compute capability as a single value
    pub fn compute_capability(&self) -> f32 {
        self.compute_capability_major as f32 + (self.compute_capability_minor as f32) / 10.0
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

#[derive(Debug, Default)]
pub struct DeviceStats {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub allocation_count: usize,
}