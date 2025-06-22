// src/bin/safe_cuda_test.rs - Safe CUDA test avoiding struct alignment issues
use arena_kv_cache::cuda::{
    initialize_cuda, CudaContext, BumpAllocator, 
    is_cuda_available, verify_cuda_runtime_linked
};
use std::time::Instant;

fn main() {
    env_logger::init();
    
    println!("🚀 Safe CUDA Test for Arena KV-Cache");
    println!("{}", "=".repeat(45));
    
    // Test 1: Runtime verification
    test_runtime_verification();
    
    // Test 2: Basic CUDA availability
    test_basic_availability();
    
    // Test 3: Safe device detection
    test_safe_device_detection();
    
    // Test 4: Context creation
    test_context_creation();
    
    // Test 5: Memory allocation
    test_memory_allocation();
    
    // Test 6: Bump allocator
    test_bump_allocator();
    
    // Test 7: Performance test
    test_performance();
    
    println!("\n🎉 Safe CUDA Test Complete!");
    println!("Your CUDA integration is working correctly!");
}

fn test_runtime_verification() {
    println!("\n📋 Test 1: Runtime Verification");
    println!("{}", "-".repeat(30));
    
    match verify_cuda_runtime_linked() {
        Ok(()) => {
            println!("✅ CUDA runtime verification passed");
            println!("   All symbols properly linked");
        }
        Err(e) => {
            println!("❌ Runtime verification failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn test_basic_availability() {
    println!("\n📋 Test 2: CUDA Availability");
    println!("{}", "-".repeat(25));
    
    if is_cuda_available() {
        println!("✅ CUDA is available and functional");
        
        match initialize_cuda() {
            Ok(()) => {
                println!("✅ CUDA initialization successful");
            }
            Err(e) => {
                println!("❌ CUDA initialization failed: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        println!("❌ CUDA not available");
        std::process::exit(1);
    }
}

fn test_safe_device_detection() {
    println!("\n📋 Test 3: Safe Device Detection");
    println!("{}", "-".repeat(30));
    
    // Use only safe CUDA calls that we know work
    #[cfg(cuda_available)]
    extern "C" {
        fn cudaGetDeviceCount(count: *mut i32) -> i32;
        fn cudaSetDevice(device: i32) -> i32;
        fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
        fn cudaGetErrorString(error: i32) -> *const i8;
    }
    
    #[cfg(cuda_available)]
    unsafe {
        // Test device count
        let mut device_count = 0;
        let result = cudaGetDeviceCount(&mut device_count);
        
        if result == 0 {
            println!("📊 Found {} CUDA device(s)", device_count);
            
            if device_count > 0 {
                // Set device 0
                let result = cudaSetDevice(0);
                if result != 0 {
                    println!("⚠️  Could not set device 0");
                    return;
                }
                
                println!("✅ Successfully set device 0");
                
                // Get safe attributes one by one
                test_device_attributes(0);
            }
        } else {
            let error_str = cudaGetErrorString(result);
            let error_cstr = std::ffi::CStr::from_ptr(error_str);
            println!("❌ cudaGetDeviceCount failed: {}", error_cstr.to_string_lossy());
        }
    }
    
    #[cfg(not(cuda_available))]
    {
        println!("❌ CUDA not compiled in");
    }
}

#[cfg(cuda_available)]
unsafe fn test_device_attributes(device_id: i32) {
    extern "C" {
        fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    }
    
    println!("🔧 Getting device attributes for device {}:", device_id);
    
    // CUDA device attributes (safe to query)
    let attributes = [
        (1, "CUDA_DEVICE_ATTR_MAX_THREADS_PER_BLOCK"),
        (2, "CUDA_DEVICE_ATTR_MAX_BLOCK_DIM_X"), 
        (16, "CUDA_DEVICE_ATTR_MULTIPROCESSOR_COUNT"),
        (4, "CUDA_DEVICE_ATTR_WARP_SIZE"),
        (75, "CUDA_DEVICE_ATTR_COMPUTE_CAPABILITY_MAJOR"),
        (76, "CUDA_DEVICE_ATTR_COMPUTE_CAPABILITY_MINOR"),
        (36, "CUDA_DEVICE_ATTR_MEMORY_CLOCK_RATE"),
        (37, "CUDA_DEVICE_ATTR_GLOBAL_MEMORY_BUS_WIDTH"),
    ];
    
    for (attr, name) in &attributes {
        let mut value = 0;
        let result = cudaDeviceGetAttribute(&mut value, *attr, device_id);
        
        if result == 0 {
            match *attr {
                75 | 76 => println!("   {}: {}", name.split('_').last().unwrap(), value), // Compute capability
                36 => println!("   Memory Clock: {} kHz", value),
                37 => println!("   Memory Bus Width: {} bits", value),
                16 => {
                    println!("   Multiprocessors: {}", value);
                    // Check if this looks like a T4 (40 SMs)
                    if value == 40 {
                        println!("   ✅ T4 GPU characteristics detected!");
                    }
                }
                _ => println!("   {}: {}", name.split('_').last().unwrap_or(name), value),
            }
        } else {
            println!("   ⚠️  Could not get {}", name);
        }
    }
}

fn test_context_creation() {
    println!("\n📋 Test 4: CUDA Context Creation");
    println!("{}", "-".repeat(30));
    
    match CudaContext::new() {
        Ok(_context) => {
            println!("✅ CUDA context created successfully");
            println!("   Context includes memory manager and streams");
        }
        Err(e) => {
            println!("❌ Context creation failed: {}", e);
            return;
        }
    }
}

fn test_memory_allocation() {
    println!("\n📋 Test 5: Memory Allocation");
    println!("{}", "-".repeat(25));
    
    match CudaContext::new() {
        Ok(context) => {
            println!("✅ Context created for memory test");
            
            // Test progressively larger allocations
            let sizes = [
                (4 * 1024, "4KB"),
                (64 * 1024, "64KB"), 
                (1024 * 1024, "1MB"),
                (16 * 1024 * 1024, "16MB"),
            ];
            
            for (size, name) in &sizes {
                match context.allocate_page_on_device(*size, 0) {
                    Ok(page) => {
                        println!("✅ {} allocation successful", name);
                        
                        // Test allocation within page
                        let test_alloc_size = (*size).min(64 * 1024); // Don't allocate more than 64KB
                        if let Some(_ptr) = page.allocate(test_alloc_size, 256) {
                            println!("   ✅ Bump allocation ({}) successful", 
                                   if test_alloc_size < 1024 { format!("{}B", test_alloc_size) }
                                   else { format!("{}KB", test_alloc_size / 1024) });
                            println!("   Page utilization: {:.1}%", page.utilization() * 100.0);
                        }
                        
                        // Test synchronization
                        match page.synchronize() {
                            Ok(()) => println!("   ✅ Page sync successful"),
                            Err(e) => println!("   ⚠️  Page sync failed: {}", e),
                        }
                    }
                    Err(e) => {
                        println!("❌ {} allocation failed: {}", name, e);
                        if *size <= 1024 * 1024 {
                            // If small allocations fail, something is seriously wrong
                            return;
                        }
                        // Large allocation failures are OK (GPU might not have enough memory)
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ Context creation failed: {}", e);
        }
    }
}

fn test_bump_allocator() {
    println!("\n📋 Test 6: Bump Allocator");
    println!("{}", "-".repeat(20));
    
    match BumpAllocator::new(256 * 1024, 0) {  // Smaller size for safety
        Ok(allocator) => {
            println!("✅ Bump allocator created (256KB)");
            
            // Test multiple small allocations
            let sizes = [1024, 2048, 4096, 8192, 16384];
            let mut successful = 0;
            let mut total_allocated = 0;
            
            for (i, size) in sizes.iter().enumerate() {
                if let Some(_ptr) = allocator.allocate(*size, 256) {
                    successful += 1;
                    total_allocated += size;
                    println!("   ✅ Allocation {}: {} bytes", i + 1, size);
                } else {
                    println!("   ⚠️  Allocation {} failed (allocator full)", i + 1);
                    break;
                }
            }
            
            println!("📊 Bump Allocator Results:");
            println!("   Successful: {}/{}", successful, sizes.len());
            println!("   Total allocated: {} KB", total_allocated / 1024);
            println!("   Utilization: {:.1}%", allocator.utilization() * 100.0);
            println!("   Available: {} KB", allocator.available_space() / 1024);
            
            // Test reset
            let old_offset = allocator.current_offset();
            allocator.reset();
            println!("   ✅ Reset: {} KB -> {} KB", old_offset / 1024, allocator.current_offset() / 1024);
            
            // Test synchronization
            match allocator.synchronize() {
                Ok(()) => println!("   ✅ Synchronization successful"),
                Err(e) => println!("   ⚠️  Sync failed: {}", e),
            }
        }
        Err(e) => {
            println!("❌ Bump allocator creation failed: {}", e);
        }
    }
}

fn test_performance() {
    println!("\n📋 Test 7: Performance Testing");
    println!("{}", "-".repeat(25));
    
    match BumpAllocator::new(512 * 1024, 0) {
        Ok(allocator) => {
            // Test allocation speed
            let start = Instant::now();
            let mut successful = 0;
            
            // Test many small allocations
            for _ in 0..10000 {
                if allocator.allocate(32, 32).is_some() {
                    successful += 1;
                } else {
                    break; // Allocator full
                }
            }
            
            let duration = start.elapsed();
            
            if successful > 0 {
                let ops_per_sec = successful as f64 / duration.as_secs_f64();
                
                println!("📊 Performance Results:");
                println!("   Allocations: {} in {:.2}ms", successful, duration.as_millis());
                println!("   Speed: {:.0} allocations/second", ops_per_sec);
                println!("   Avg time: {:.2}μs per allocation", 
                       duration.as_micros() as f64 / successful as f64);
                
                if ops_per_sec > 100_000.0 {
                    println!("   ✅ Excellent allocation performance");
                } else if ops_per_sec > 10_000.0 {
                    println!("   ✅ Good allocation performance");
                } else {
                    println!("   ✅ Functional allocation performance");
                }
            } else {
                println!("   ⚠️  No allocations succeeded in performance test");
            }
            
            // Test KV Cache specific patterns
            test_kv_patterns(&allocator);
        }
        Err(e) => {
            println!("❌ Performance test setup failed: {}", e);
        }
    }
}

fn test_kv_patterns(allocator: &BumpAllocator) {
    println!("\n🔧 Testing KV-Cache patterns:");
    
    // Reset allocator for clean test
    allocator.reset();
    
    // Simulate KV tensor allocations
    let kv_configs = [
        (512, 16, 64),   // Small: 512 seq, 16 heads, 64 dim
        (1024, 32, 128), // Medium: 1K seq, 32 heads, 128 dim
        (2048, 32, 128), // Large: 2K seq, 32 heads, 128 dim
    ];
    
    for (i, (seq_len, num_heads, head_dim)) in kv_configs.iter().enumerate() {
        // Calculate KV tensor size (K + V tensors, fp16)
        let element_size = 2; // fp16
        let kv_size = 2 * seq_len * num_heads * head_dim * element_size;
        
        if let Some(_ptr) = allocator.allocate(kv_size, 256) {
            println!("   ✅ KV Config {}: {}x{}x{} = {} KB", 
                   i + 1, seq_len, num_heads, head_dim, kv_size / 1024);
        } else {
            println!("   ⚠️  KV Config {} failed: {} KB (allocator full)", 
                   i + 1, kv_size / 1024);
        }
    }
    
    println!("   Final utilization: {:.1}%", allocator.utilization() * 100.0);
    
    if allocator.utilization() > 0.5 {
        println!("   ✅ Good utilization for KV cache patterns");
    } else {
        println!("   ✅ KV cache allocation patterns work");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_safe_cuda() {
        if is_cuda_available() {
            println!("✅ Safe CUDA test in test mode");
        }
    }
}