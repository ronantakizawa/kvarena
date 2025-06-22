// src/bin/minimal_cuda_test.rs - Minimal test to debug hanging issue
use arena_kv_cache::cuda::{verify_cuda_runtime_linked, is_cuda_available};

fn main() {
    println!("🔍 Minimal CUDA Test - Debugging Hanging Issue");
    println!("{}", "=".repeat(50));
    
    // Test 1: Basic runtime verification
    println!("1. Testing CUDA runtime verification...");
    match verify_cuda_runtime_linked() {
        Ok(()) => println!("   ✅ CUDA runtime verification passed"),
        Err(e) => {
            println!("   ❌ CUDA runtime verification failed: {}", e);
            return;
        }
    }
    
    // Test 2: Basic availability check
    println!("2. Testing CUDA availability...");
    if is_cuda_available() {
        println!("   ✅ CUDA is available");
    } else {
        println!("   ❌ CUDA is not available");
        return;
    }
    
    // Test 3: Direct CUDA calls to find where it hangs
    println!("3. Testing direct CUDA device count...");
    test_device_count_direct();
    
    println!("4. Testing device properties...");
    test_device_properties_direct();
    
    println!("✅ Minimal test completed successfully!");
}

fn test_device_count_direct() {
    use std::ffi::c_void;
    
    #[cfg(cuda_available)]
    extern "C" {
        fn cudaGetDeviceCount(count: *mut i32) -> i32;
        fn cudaGetErrorString(error: i32) -> *const i8;
    }
    
    #[cfg(cuda_available)]
    unsafe {
        let mut device_count = 0;
        println!("   Calling cudaGetDeviceCount...");
        
        let result = cudaGetDeviceCount(&mut device_count);
        if result == 0 {
            println!("   ✅ Device count: {}", device_count);
        } else {
            let error_str = cudaGetErrorString(result);
            let error_cstr = std::ffi::CStr::from_ptr(error_str);
            println!("   ❌ cudaGetDeviceCount failed: {}", error_cstr.to_string_lossy());
        }
    }
    
    #[cfg(not(cuda_available))]
    {
        println!("   ❌ CUDA not compiled in");
    }
}

fn test_device_properties_direct() {
    use std::ffi::c_void;
    
    #[repr(C)]
    struct CudaDeviceProperties {
        name: [i8; 256],
        total_global_mem: usize,
        // ... other fields
    }
    
    #[cfg(cuda_available)]
    extern "C" {
        fn cudaGetDeviceProperties(prop: *mut CudaDeviceProperties, device: i32) -> i32;
        fn cudaSetDevice(device: i32) -> i32;
        fn cudaGetErrorString(error: i32) -> *const i8;
    }
    
    #[cfg(cuda_available)]
    unsafe {
        println!("   Setting device 0...");
        let result = cudaSetDevice(0);
        if result != 0 {
            let error_str = cudaGetErrorString(result);
            let error_cstr = std::ffi::CStr::from_ptr(error_str);
            println!("   ❌ cudaSetDevice failed: {}", error_cstr.to_string_lossy());
            return;
        }
        
        println!("   Getting device properties...");
        let mut props: CudaDeviceProperties = std::mem::zeroed();
        let result = cudaGetDeviceProperties(&mut props, 0);
        
        if result == 0 {
            let name = std::ffi::CStr::from_ptr(props.name.as_ptr())
                .to_string_lossy();
            println!("   ✅ Device 0: {} ({} MB)", name, props.total_global_mem / 1024 / 1024);
        } else {
            let error_str = cudaGetErrorString(result);
            let error_cstr = std::ffi::CStr::from_ptr(error_str);
            println!("   ❌ cudaGetDeviceProperties failed: {}", error_cstr.to_string_lossy());
        }
    }
    
    #[cfg(not(cuda_available))]
    {
        println!("   ❌ CUDA not compiled in");
    }
}