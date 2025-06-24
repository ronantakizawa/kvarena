// src/cuda/device.rs - Fixed device information with timeout protection
use super::bindings::*;
use super::error::CudaError;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;

/// Device information with safe CUDA queries and timeout protection
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

/// Timeout wrapper for potentially hanging CUDA calls
fn with_timeout<F, T>(operation: F, timeout_secs: u64, operation_name: String) -> Result<T, CudaError>
where
    F: FnOnce() -> Result<T, CudaError> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        let result = operation();
        let _ = tx.send(result);
    });
    
    match rx.recv_timeout(Duration::from_secs(timeout_secs)) {
        Ok(result) => result,
        Err(_) => {
            log::warn!("CUDA operation '{}' timed out after {} seconds", operation_name, timeout_secs);
            Err(CudaError(CUDA_ERROR_NOT_READY))
        }
    }
}

impl CudaDeviceInfo {
    /// Query device info using REAL CUDA API calls with timeout protection
    pub fn query(device_id: i32) -> Result<Self, CudaError> {
        #[cfg(cuda_available)]
        {
            // First, do a quick device availability check
            let quick_check_result = with_timeout(
                move || {
                    unsafe {
                        let result = cudaSetDevice(device_id);
                        if result != CUDA_SUCCESS {
                            return Err(CudaError(result));
                        }
                        Ok(())
                    }
                },
                5, // 5 second timeout for device setting
                "cudaSetDevice".to_string()
            );
            
            if let Err(e) = quick_check_result {
                log::warn!("Failed to set device {} or operation timed out: {}", device_id, e);
                return Err(e);
            }

            // Get device properties with timeout
            let props_result = with_timeout(
                move || {
                    unsafe {
                        let mut props: CudaDeviceProperties = std::mem::zeroed();
                        let result = cudaGetDeviceProperties(&mut props, device_id);
                        if result != CUDA_SUCCESS {
                            return Err(CudaError(result));
                        }
                        Ok(props)
                    }
                },
                10, // 10 second timeout for properties
                "cudaGetDeviceProperties".to_string()
            )?;

            // Get memory info with timeout and fallback
            let (free_memory, total_memory) = with_timeout(
                move || {
                    unsafe {
                        let mut free = 0usize;
                        let mut total = 0usize;
                        let result = cudaMemGetInfo(&mut free, &mut total);
                        if result != CUDA_SUCCESS {
                            // Use properties memory as fallback
                            log::warn!("cudaMemGetInfo failed for device {}, using properties", device_id);
                            Ok((props_result.total_global_mem * 90 / 100, props_result.total_global_mem))
                        } else {
                            // Sanity check values
                            if total > 200 * 1024 * 1024 * 1024 { // > 200GB is suspicious
                                log::warn!("Suspicious memory values for device {}: {}GB total, using properties", 
                                          device_id, total / 1024 / 1024 / 1024);
                                Ok((props_result.total_global_mem * 90 / 100, props_result.total_global_mem))
                            } else {
                                Ok((free, total))
                            }
                        }
                    }
                },
                5, // 5 second timeout for memory info
                "cudaMemGetInfo".to_string()
            ).unwrap_or_else(|_| {
                // Fallback if memory query times out
                log::warn!("Memory info query timed out for device {}, using estimated values", device_id);
                (props_result.total_global_mem * 90 / 100, props_result.total_global_mem)
            });

            // Get additional attributes with timeout (non-critical)
            let (memory_clock_rate, memory_bus_width, max_threads_per_multiprocessor) = with_timeout(
                move || {
                    unsafe {
                        let mut memory_clock_rate = 0;
                        let mut memory_bus_width = 0;
                        let mut max_threads_per_multiprocessor = 0;
                        
                        // These can hang on some systems, so we use short timeouts
                        let _ = cudaDeviceGetAttribute(&mut memory_clock_rate, CUDA_DEVICE_ATTR_MEMORY_CLOCK_RATE, device_id);
                        let _ = cudaDeviceGetAttribute(&mut memory_bus_width, CUDA_DEVICE_ATTR_GLOBAL_MEMORY_BUS_WIDTH, device_id);
                        let _ = cudaDeviceGetAttribute(&mut max_threads_per_multiprocessor, CUDA_DEVICE_ATTR_MAX_THREADS_PER_MULTIPROCESSOR, device_id);
                        
                        Ok((memory_clock_rate, memory_bus_width, max_threads_per_multiprocessor))
                    }
                },
                3, // 3 second timeout for attributes
                "cudaDeviceGetAttribute".to_string()
            ).unwrap_or_else(|_| {
                log::debug!("Device attributes query timed out for device {}, using defaults", device_id);
                (0, 0, 0) // Default values if attributes can't be queried
            });

            // Convert name from C string safely
            let name = unsafe {
                std::ffi::CStr::from_ptr(props_result.name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
                    .trim_end_matches('\0')
                    .to_string()
            };

            // Validate the name
            let clean_name = if name.is_empty() || name.len() > 100 {
                format!("CUDA Device {}", device_id)
            } else {
                name
            };

            log::info!("✓ Queried CUDA device {}: {} (CC {}.{}, {} MB total, {} MB free)", 
                      device_id, clean_name, props_result.major, props_result.minor, 
                      total_memory / 1024 / 1024, free_memory / 1024 / 1024);

            Ok(CudaDeviceInfo {
                device_id,
                name: clean_name,
                total_memory,
                free_memory,
                compute_capability_major: props_result.major,
                compute_capability_minor: props_result.minor,
                multiprocessor_count: props_result.multiprocessor_count,
                max_threads_per_block: props_result.max_threads_per_block,
                warp_size: props_result.warp_size,
                memory_clock_rate,
                memory_bus_width,
                max_threads_per_multiprocessor,
            })
        }
        
        #[cfg(not(cuda_available))]
        {
            log::error!("CUDA not available - cannot query device {}", device_id);
            Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
        }
    }
    
    /// Create a minimal device info for testing (when CUDA queries fail)
    pub fn create_minimal(device_id: i32) -> Self {
        log::warn!("Creating minimal device info for device {} (CUDA queries failed)", device_id);
        CudaDeviceInfo {
            device_id,
            name: format!("CUDA Device {} (Limited Info)", device_id),
            total_memory: 8 * 1024 * 1024 * 1024, // Assume 8GB
            free_memory: 6 * 1024 * 1024 * 1024,  // Assume 6GB free
            compute_capability_major: 7,
            compute_capability_minor: 5,  // T4 capability
            multiprocessor_count: 40,     // T4 typical
            max_threads_per_block: 1024,
            warp_size: 32,
            memory_clock_rate: 5001000,   // T4 typical
            memory_bus_width: 256,        // T4 typical
            max_threads_per_multiprocessor: 1024,
        }
    }
    
    /// Check if this is a T4 GPU
    pub fn is_t4(&self) -> bool {
        self.name.to_lowercase().contains("t4") || 
        (self.compute_capability_major == 7 && self.compute_capability_minor == 5)
    }
    
    /// Get memory bandwidth in GB/s
    pub fn memory_bandwidth_gbps(&self) -> f64 {
        if self.memory_clock_rate == 0 || self.memory_bus_width == 0 {
            // Fallback for T4
            if self.is_t4() {
                return 320.0; // T4 theoretical bandwidth
            }
            return 100.0; // Conservative estimate
        }
        
        // Bandwidth = (memory_clock_rate * 2) * (memory_bus_width / 8) / 1e9
        let clock_hz = self.memory_clock_rate as f64 * 1000.0; // Convert kHz to Hz
        let bus_width_bytes = self.memory_bus_width as f64 / 8.0;
        (clock_hz * 2.0 * bus_width_bytes) / 1e9
    }
    
    /// Get compute capability as a single value
    pub fn compute_capability(&self) -> f32 {
        self.compute_capability_major as f32 + (self.compute_capability_minor as f32) / 10.0
    }
    
    /// Validate device info for consistency
    pub fn validate(&self) -> Result<(), String> {
        if self.total_memory == 0 {
            return Err("Total memory cannot be zero".to_string());
        }
        
        if self.free_memory > self.total_memory {
            return Err("Free memory cannot exceed total memory".to_string());
        }
        
        if self.compute_capability_major < 3 {
            return Err("Compute capability too low (< 3.0)".to_string());
        }
        
        if self.multiprocessor_count == 0 {
            return Err("Multiprocessor count cannot be zero".to_string());
        }
        
        Ok(())
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

/// Safe device detection with fallback options
pub fn detect_cuda_devices() -> Vec<CudaDeviceInfo> {
    let mut devices = Vec::new();
    
    #[cfg(cuda_available)]
    {
        // First try to get device count
        let device_count = match with_timeout(
            || {
                unsafe {
                    let mut count = 0;
                    let result = cudaGetDeviceCount(&mut count);
                    if result != CUDA_SUCCESS {
                        Err(CudaError(result))
                    } else {
                        Ok(count)
                    }
                }
            },
            10, // 10 second timeout
            "cudaGetDeviceCount".to_string()
        ) {
            Ok(count) => count,
            Err(e) => {
                log::error!("Failed to get CUDA device count: {}", e);
                return devices;
            }
        };

        if device_count == 0 {
            log::warn!("No CUDA devices found");
            return devices;
        }

        log::info!("Detected {} CUDA device(s), querying...", device_count);

        // Query each device with individual timeouts
        for device_id in 0..device_count {
            match CudaDeviceInfo::query(device_id) {
                Ok(info) => {
                    if let Err(validation_error) = info.validate() {
                        log::warn!("Device {} validation failed: {}, using minimal info", 
                                  device_id, validation_error);
                        devices.push(CudaDeviceInfo::create_minimal(device_id));
                    } else {
                        log::info!("✓ Device {} validated successfully", device_id);
                        devices.push(info);
                    }
                }
                Err(e) => {
                    log::warn!("Failed to query device {}: {}, using minimal info", device_id, e);
                    devices.push(CudaDeviceInfo::create_minimal(device_id));
                }
            }
        }
    }

    #[cfg(not(cuda_available))]
    {
        log::warn!("CUDA not compiled in, cannot detect devices");
    }

    devices
}

/// Quick device health check
pub fn check_device_health(device_id: i32) -> Result<(), CudaError> {
    #[cfg(cuda_available)]
    {
        with_timeout(
            move || {
                unsafe {
                    // Set device
                    let result = cudaSetDevice(device_id);
                    if result != CUDA_SUCCESS {
                        return Err(CudaError(result));
                    }
                    
                    // Try a small allocation
                    let mut ptr = std::ptr::null_mut();
                    let result = cudaMalloc(&mut ptr, 1024);
                    if result != CUDA_SUCCESS {
                        return Err(CudaError(result));
                    }
                    
                    // Free it immediately
                    let result = cudaFree(ptr);
                    if result != CUDA_SUCCESS {
                        return Err(CudaError(result));
                    }
                    
                    Ok(())
                }
            },
            5, // 5 second timeout
            "device_health_check".to_string()
        )
    }
    
    #[cfg(not(cuda_available))]
    {
        Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
    }
}