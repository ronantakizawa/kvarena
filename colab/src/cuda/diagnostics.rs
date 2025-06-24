// src/cuda/diagnostics.rs - CUDA diagnostics and verification
use super::bindings::*;
use super::error::CudaError;
use super::raw;
use std::process::Command;
use std::time::Instant;
use std::ffi::c_void;

/// CRITICAL: Verify CUDA runtime is properly linked
pub fn verify_cuda_runtime_linked() -> Result<(), String> {
    #[cfg(not(cuda_available))]
    {
        return Err("CUDA not compiled in - build with --features cuda".to_string());
    }
    
    #[cfg(cuda_available)]
    unsafe {
        // Test 1: Basic device count query
        let mut device_count = 0;
        let result = cudaGetDeviceCount(&mut device_count);
        
        if result != CUDA_SUCCESS {
            let error_str = cudaGetErrorString(result);
            let error_cstr = std::ffi::CStr::from_ptr(error_str);
            return Err(format!("CUDA runtime not properly linked: {}", 
                              error_cstr.to_string_lossy()));
        }
        
        if device_count == 0 {
            return Err("No CUDA devices found - check nvidia-smi".to_string());
        }
        
        // Test 2: Memory allocation test
        let mut test_ptr = std::ptr::null_mut();
        let alloc_result = cudaMalloc(&mut test_ptr, 1024);
        
        if alloc_result != CUDA_SUCCESS {
            let error_str = cudaGetErrorString(alloc_result);
            let error_cstr = std::ffi::CStr::from_ptr(error_str);
            return Err(format!("CUDA memory allocation failed: {}", 
                              error_cstr.to_string_lossy()));
        }
        
        // Test 3: Memory deallocation
        let free_result = cudaFree(test_ptr);
        if free_result != CUDA_SUCCESS {
            let error_str = cudaGetErrorString(free_result);
            let error_cstr = std::ffi::CStr::from_ptr(error_str);
            return Err(format!("CUDA memory deallocation failed: {}", 
                              error_cstr.to_string_lossy()));
        }
        
        log::info!("✅ CUDA runtime verification successful!");
        log::info!("   Devices found: {}", device_count);
        log::info!("   Memory allocation: OK");
        log::info!("   Memory deallocation: OK");
        
        Ok(())
    }
}

/// CRITICAL: Runtime health check
pub fn cuda_runtime_health_check() {
    match verify_cuda_runtime_linked() {
        Ok(()) => {
            log::info!("CUDA runtime health check passed");
        }
        Err(e) => {
            log::error!("CUDA runtime health check failed: {}", e);
            eprintln!("❌ CUDA Runtime Error: {}", e);
            eprintln!("This indicates CUDA is not properly linked or installed.");
            eprintln!("Please check:");
            eprintln!("  1. CUDA toolkit is installed: nvcc --version");
            eprintln!("  2. GPU drivers are installed: nvidia-smi");
            eprintln!("  3. Library path includes CUDA: echo $LD_LIBRARY_PATH");
            eprintln!("  4. Rebuild with: cargo clean && cargo build --features cuda");
            std::process::exit(1);
        }
    }
}

/// Check CUDA environment setup
pub fn check_cuda_environment() -> Result<(), String> {
    // Check environment variables
    let cuda_vars = ["CUDA_PATH", "CUDA_ROOT", "CUDA_HOME"];
    let mut cuda_found = false;
    
    for var in &cuda_vars {
        if let Ok(path) = std::env::var(var) {
            log::info!("Found {}: {}", var, path);
            cuda_found = true;
        }
    }
    
    if !cuda_found {
        log::warn!("No CUDA environment variables set");
        log::warn!("Consider setting CUDA_PATH to your CUDA installation");
    }
    
    // Check LD_LIBRARY_PATH (Linux/macOS)
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
            if ld_path.contains("cuda") {
                log::info!("✅ LD_LIBRARY_PATH includes CUDA paths");
            } else {
                log::warn!("LD_LIBRARY_PATH may not include CUDA lib64");
                log::warn!("Consider adding: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH");
            }
        } else {
            log::warn!("LD_LIBRARY_PATH not set");
        }
    }
    
    Ok(())
}

/// Initialize CUDA runtime and verify T4 GPU availability
pub fn initialize_cuda() -> Result<(), CudaError> {
    // CRITICAL: Verify runtime is linked first
    cuda_runtime_health_check();
    
    // Check environment
    let _ = check_cuda_environment();
    
    #[cfg(cuda_available)]
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
        
        // Verify device works with a larger allocation test
        let mut test_ptr = std::ptr::null_mut();
        let test_size = 1024 * 1024; // 1MB test
        let result = cudaMalloc(&mut test_ptr, test_size);
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
    
    #[cfg(not(cuda_available))]
    {
        Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
    }
}

/// Check if CUDA is available and functional
pub fn is_cuda_available() -> bool {
    match initialize_cuda() {
        Ok(()) => true,
        Err(_) => false,
    }
}

/// Get CUDA runtime version info
pub fn get_cuda_version() -> String {
    // This would require additional CUDA runtime calls
    // For now, return a placeholder
    "CUDA Runtime".to_string()
}

/// Perform a CUDA memory test to verify functionality
pub fn cuda_memory_test(device_id: i32, test_size: usize) -> Result<f64, CudaError> {
    #[cfg(cuda_available)]
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
    
    #[cfg(not(cuda_available))]
    {
        Err(CudaError(CUDA_ERROR_NOT_INITIALIZED))
    }
}

/// Comprehensive CUDA diagnosis
pub fn diagnose_cuda_issues() {
    println!("🔍 CUDA Diagnosis:");
    println!("{}", "=".repeat(50));
    
    // 1. Check compilation flags
    #[cfg(cuda_available)]
    println!("✅ CUDA compiled in (cuda_available flag set)");
    
    #[cfg(not(cuda_available))]
    {
        println!("❌ CUDA not compiled in");
        println!("Run: cargo build --features cuda");
        return;
    }
    
    // 2. Check environment
    let _ = check_cuda_environment();
    
    // 3. Check runtime linking
    let _ = verify_cuda_runtime_linked();
    
    // 4. Test basic operations
    #[cfg(cuda_available)]
    {
        match raw::get_device_count() {
            Ok(count) => println!("✅ Device count: {}", count),
            Err(e) => println!("❌ Failed to get device count: {}", e),
        }
        
        match raw::get_device() {
            Ok(device) => println!("✅ Current device: {}", device),
            Err(e) => println!("❌ Failed to get current device: {}", e),
        }
        
        match raw::mem_get_info() {
            Ok((free, total)) => println!("✅ Memory: {} MB free / {} MB total", 
                                          free / 1024 / 1024, total / 1024 / 1024),
            Err(e) => println!("❌ Failed to get memory info: {}", e),
        }
        
        // Test allocation
        match raw::malloc(1024) {
            Ok(ptr) => {
                println!("✅ Test allocation successful");
                let _ = raw::free(ptr);
                println!("✅ Test deallocation successful");
            }
            Err(e) => println!("❌ Test allocation failed: {}", e),
        }
    }
}