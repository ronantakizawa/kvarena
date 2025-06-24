// src/cuda/diagnostics.rs - Fixed CUDA diagnostics with hang prevention
use super::bindings::*;
use super::error::CudaError;
use super::device::{detect_cuda_devices, check_device_health};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};
use std::ffi::c_void;

/// Global flag to track CUDA initialization state using OnceLock
static CUDA_INITIALIZED: OnceLock<Arc<Mutex<bool>>> = OnceLock::new();

fn get_cuda_initialized() -> &'static Arc<Mutex<bool>> {
    CUDA_INITIALIZED.get_or_init(|| Arc::new(Mutex::new(false)))
}

/// Safe timeout wrapper for CUDA operations (public for use in other modules)
pub fn safe_cuda_call<F, T>(operation: F, timeout_secs: u64, operation_name: String) -> Result<T, CudaError>
where
    F: FnOnce() -> Result<T, CudaError> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    let operation_name_clone = operation_name.clone(); // Clone for the thread
    
    thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            operation()
        }));
        
        let final_result = match result {
            Ok(cuda_result) => cuda_result,
            Err(_) => {
                log::error!("CUDA operation '{}' panicked", operation_name_clone);
                Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
            }
        };
        
        let _ = tx.send(final_result);
    });
    
    match rx.recv_timeout(Duration::from_secs(timeout_secs)) {
        Ok(result) => result,
        Err(_) => {
            log::error!("CUDA operation '{}' timed out after {} seconds", operation_name, timeout_secs);
            Err(CudaError(CUDA_ERROR_NOT_READY))
        }
    }
}

/// CRITICAL: Verify CUDA runtime is properly linked with timeout protection
pub fn verify_cuda_runtime_linked() -> Result<(), String> {
    #[cfg(not(cuda_available))]
    {
        return Err("CUDA not compiled in - build with --features cuda".to_string());
    }
    
    #[cfg(cuda_available)]
    {
        log::info!("🔍 Verifying CUDA runtime linkage...");
        
        // Test 1: Basic device count query with timeout
        let device_count = match safe_cuda_call(
            || {
                unsafe {
                    let mut device_count = 0;
                    let result = cudaGetDeviceCount(&mut device_count);
                    
                    if result != CUDA_SUCCESS {
                        let error_str = cudaGetErrorString(result);
                        let error_cstr = std::ffi::CStr::from_ptr(error_str);
                        log::error!("cudaGetDeviceCount failed: {}", error_cstr.to_string_lossy());
                        return Err(CudaError(result));
                    }
                    
                    Ok(device_count)
                }
            },
            10, // 10 second timeout
            "cudaGetDeviceCount".to_string()
        ) {
            Ok(count) => count,
            Err(e) => {
                return Err(format!("CUDA device count query failed or timed out: {}", e));
            }
        };
        
        if device_count == 0 {
            return Err("No CUDA devices found - check nvidia-smi".to_string());
        }
        
        log::info!("✓ Found {} CUDA device(s)", device_count);
        
        // Test 2: Memory allocation test with timeout (smaller allocation)
        match safe_cuda_call(
            || {
                unsafe {
                    // Set device 0 first
                    let result = cudaSetDevice(0);
                    if result != CUDA_SUCCESS {
                        return Err(CudaError(result));
                    }
                    
                    let mut test_ptr = std::ptr::null_mut();
                    let test_size = 1024; // Small allocation for safety
                    let alloc_result = cudaMalloc(&mut test_ptr, test_size);
                    
                    if alloc_result != CUDA_SUCCESS {
                        let error_str = cudaGetErrorString(alloc_result);
                        let error_cstr = std::ffi::CStr::from_ptr(error_str);
                        log::error!("cudaMalloc failed: {}", error_cstr.to_string_lossy());
                        return Err(CudaError(alloc_result));
                    }
                    
                    // Test 3: Memory deallocation
                    let free_result = cudaFree(test_ptr);
                    if free_result != CUDA_SUCCESS {
                        let error_str = cudaGetErrorString(free_result);
                        let error_cstr = std::ffi::CStr::from_ptr(error_str);
                        log::error!("cudaFree failed: {}", error_cstr.to_string_lossy());
                        return Err(CudaError(free_result));
                    }
                    
                    Ok(())
                }
            },
            15, // 15 second timeout for allocation test
            "cuda_memory_test".to_string()
        ) {
            Ok(()) => {
                log::info!("✓ CUDA memory allocation test passed");
            }
            Err(e) => {
                return Err(format!("CUDA memory test failed or timed out: {}", e));
            }
        }
        
        log::info!("✅ CUDA runtime verification successful!");
        log::info!("   Devices found: {}", device_count);
        log::info!("   Memory allocation: OK");
        log::info!("   Memory deallocation: OK");
        
        Ok(())
    }
}

/// Safe CUDA initialization with comprehensive error handling
pub fn initialize_cuda() -> Result<(), CudaError> {
    // Check if already initialized
    if let Ok(initialized) = get_cuda_initialized().lock() {
        if *initialized {
            log::debug!("CUDA already initialized");
            return Ok(());
        }
    }
    
    log::info!("🚀 Initializing CUDA runtime...");
    
    // First verify the runtime is linked
    if let Err(e) = verify_cuda_runtime_linked() {
        log::error!("CUDA runtime verification failed: {}", e);
        return Err(CudaError(CUDA_ERROR_NOT_INITIALIZED));
    }
    
    // Check environment
    let _ = check_cuda_environment();
    
    #[cfg(cuda_available)]
    {
        // Get device count safely
        let device_count = match safe_cuda_call(
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
            10,
            "final_device_count_check".to_string()
        ) {
            Ok(count) => count,
            Err(e) => {
                log::error!("Failed to get device count during initialization: {}", e);
                return Err(e);
            }
        };
        
        if device_count == 0 {
            log::error!("No CUDA devices available");
            return Err(CudaError(CUDA_ERROR_INVALID_DEVICE));
        }
        
        // Test device 0 health
        match check_device_health(0) {
            Ok(()) => {
                log::info!("✓ Device 0 health check passed");
            }
            Err(e) => {
                log::warn!("Device 0 health check failed: {}, but continuing", e);
                // Don't fail initialization if health check fails
            }
        }
        
        // Mark as initialized
        if let Ok(mut initialized) = get_cuda_initialized().lock() {
            *initialized = true;
        }
        
        log::info!("✅ CUDA initialized successfully with {} device(s)", device_count);
        Ok(())
    }
    
    #[cfg(not(cuda_available))]
    {
        log::error!("CUDA not available in this build");
        Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
    }
}

/// Safe CUDA environment check
pub fn check_cuda_environment() -> Result<(), String> {
    log::info!("🔍 Checking CUDA environment...");
    
    // Check environment variables
    let cuda_vars = ["CUDA_PATH", "CUDA_ROOT", "CUDA_HOME"];
    let mut cuda_found = false;
    
    for var in &cuda_vars {
        if let Ok(path) = std::env::var(var) {
            log::info!("✓ Found {}: {}", var, path);
            cuda_found = true;
        }
    }
    
    if !cuda_found {
        log::warn!("⚠️ No CUDA environment variables set");
        log::info!("💡 Consider setting CUDA_PATH to your CUDA installation");
    }
    
    // Check PATH for nvcc (if available)
    match std::process::Command::new("nvcc").arg("--version").output() {
        Ok(output) => {
            if output.status.success() {
                let version_output = String::from_utf8_lossy(&output.stdout);
                if let Some(version_line) = version_output.lines().find(|line| line.contains("release")) {
                    log::info!("✓ nvcc found: {}", version_line.trim());
                }
            } else {
                log::warn!("⚠️ nvcc found but returned error");
            }
        }
        Err(_) => {
            log::warn!("⚠️ nvcc not found in PATH");
        }
    }
    
    // Check for nvidia-smi
    match std::process::Command::new("nvidia-smi").arg("--query-gpu=name").arg("--format=csv,noheader").output() {
        Ok(output) => {
            if output.status.success() {
                let gpu_list = String::from_utf8_lossy(&output.stdout);
                for (i, gpu) in gpu_list.lines().enumerate() {
                    log::info!("✓ GPU {}: {}", i, gpu.trim());
                }
            } else {
                log::warn!("⚠️ nvidia-smi found but returned error");
            }
        }
        Err(_) => {
            log::warn!("⚠️ nvidia-smi not found");
            log::info!("💡 Install NVIDIA drivers: https://developer.nvidia.com/cuda-downloads");
        }
    }
    
    // Check LD_LIBRARY_PATH (Linux/macOS)
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
            if ld_path.to_lowercase().contains("cuda") {
                log::info!("✓ LD_LIBRARY_PATH includes CUDA paths");
            } else {
                log::warn!("⚠️ LD_LIBRARY_PATH may not include CUDA lib64");
                log::info!("💡 Consider: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH");
            }
        } else {
            log::warn!("⚠️ LD_LIBRARY_PATH not set");
        }
    }
    
    Ok(())
}

/// Check if CUDA is available and functional
pub fn is_cuda_available() -> bool {
    match initialize_cuda() {
        Ok(()) => true,
        Err(e) => {
            log::debug!("CUDA not available: {}", e);
            false
        }
    }
}

/// Comprehensive CUDA memory test with safety checks
pub fn cuda_memory_test(device_id: i32, test_size: usize) -> Result<f64, CudaError> {
    let safe_test_size = test_size.min(16 * 1024 * 1024); // Limit to 16MB for safety
    
    #[cfg(cuda_available)]
    {
        safe_cuda_call(
            move || {
                unsafe {
                    let result = cudaSetDevice(device_id);
                    if result != CUDA_SUCCESS {
                        return Err(CudaError(result));
                    }
                    
                    // Allocate test memory
                    let mut device_ptr = std::ptr::null_mut();
                    let result = cudaMalloc(&mut device_ptr, safe_test_size);
                    if result != CUDA_SUCCESS {
                        return Err(CudaError(result));
                    }
                    
                    // Create host memory
                    let host_data = vec![0xAAu8; safe_test_size]; // Pattern for verification
                    
                    // Test memory bandwidth
                    let start = Instant::now();
                    
                    // Host to device
                    let result = cudaMemcpy(
                        device_ptr, 
                        host_data.as_ptr() as *const c_void, 
                        safe_test_size, 
                        CUDA_MEMCPY_HOST_TO_DEVICE
                    );
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
                    let bandwidth_gbps = (safe_test_size as f64) / elapsed.as_secs_f64() / 1e9;
                    
                    // Cleanup
                    let result = cudaFree(device_ptr);
                    if result != CUDA_SUCCESS {
                        return Err(CudaError(result));
                    }
                    
                    Ok(bandwidth_gbps)
                }
            },
            30, // 30 second timeout for memory test
            "cuda_memory_bandwidth_test".to_string()
        )
    }
    
    #[cfg(not(cuda_available))]
    {
        Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
    }
}

/// Comprehensive CUDA diagnosis with timeout protection
pub fn diagnose_cuda_issues() {
    println!("🔍 CUDA Comprehensive Diagnosis");
    println!("{}", "=".repeat(60));
    
    // 1. Check compilation flags
    #[cfg(cuda_available)]
    println!("✅ CUDA compiled in (cuda_available flag set)");
    
    #[cfg(not(cuda_available))]
    {
        println!("❌ CUDA not compiled in");
        println!("💡 Run: cargo build --features cuda");
        return;
    }
    
    // 2. Check environment
    println!("\n📋 Environment Check:");
    let _ = check_cuda_environment();
    
    // 3. Check runtime linking
    println!("\n🔗 Runtime Linkage Check:");
    match verify_cuda_runtime_linked() {
        Ok(()) => println!("✅ CUDA runtime properly linked"),
        Err(e) => {
            println!("❌ CUDA runtime linkage failed: {}", e);
            return;
        }
    }
    
    // 4. Device detection
    println!("\n🖥️ Device Detection:");
    let devices = detect_cuda_devices();
    if devices.is_empty() {
        println!("❌ No CUDA devices detected");
        return;
    }
    
    for device in &devices {
        println!("✅ Device {}: {}", device.device_id, device.name);
        println!("   Compute Capability: {}.{}", 
                device.compute_capability_major, device.compute_capability_minor);
        println!("   Memory: {} MB total, {} MB free", 
                device.total_memory / 1024 / 1024, device.free_memory / 1024 / 1024);
        
        if device.is_t4() {
            println!("   🎯 Tesla T4 detected - optimal for this project");
        }
    }
    
    // 5. Memory test
    println!("\n💾 Memory Test:");
    match cuda_memory_test(0, 1024 * 1024) {
        Ok(bandwidth) => {
            println!("✅ Memory test passed: {:.1} GB/s bandwidth", bandwidth);
        }
        Err(e) => {
            println!("⚠️ Memory test failed: {}", e);
        }
    }
    
    // 6. Final health check
    println!("\n🏥 Health Check:");
    match check_device_health(0) {
        Ok(()) => println!("✅ Device 0 health check passed"),
        Err(e) => println!("⚠️ Device 0 health check failed: {}", e),
    }
    
    println!("\n🎉 CUDA diagnosis complete!");
}

/// Safe CUDA runtime health check with timeout
pub fn cuda_runtime_health_check() {
    log::info!("🏥 Performing CUDA runtime health check...");
    
    match verify_cuda_runtime_linked() {
        Ok(()) => {
            log::info!("✅ CUDA runtime health check passed");
        }
        Err(e) => {
            log::error!("❌ CUDA runtime health check failed: {}", e);
            eprintln!("❌ CUDA Runtime Error: {}", e);
            eprintln!("This indicates CUDA is not properly linked or installed.");
            eprintln!();
            eprintln!("🔧 Troubleshooting steps:");
            eprintln!("  1. Check CUDA installation: nvcc --version");
            eprintln!("  2. Check GPU drivers: nvidia-smi");
            eprintln!("  3. Check library path: echo $LD_LIBRARY_PATH");
            eprintln!("  4. Rebuild project: cargo clean && cargo build --features cuda");
            eprintln!("  5. Try with timeout: timeout 30s your_program");
            eprintln!();
            
            // Don't exit - let the application handle the error gracefully
            log::warn!("Continuing with degraded CUDA functionality");
        }
    }
}

/// Get CUDA version info safely
pub fn get_cuda_version() -> String {
    #[cfg(cuda_available)]
    {
        match safe_cuda_call(
            || {
                // This would require cudaRuntimeGetVersion which might not be available
                // For now, return a basic string
                Ok("CUDA Runtime".to_string())
            },
            5,
            "get_cuda_version".to_string()
        ) {
            Ok(version) => version,
            Err(_) => "CUDA Runtime (version unknown)".to_string(),
        }
    }
    
    #[cfg(not(cuda_available))]
    {
        "CUDA not available".to_string()
    }
}