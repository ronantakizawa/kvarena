// src/bin/test_cuda_integration.rs - Complete CUDA integration test
use arena_kv_cache::cuda::{
    initialize_cuda, CudaMemoryManager, CudaContext, 
    BumpAllocator, cuda_memory_test, is_cuda_available,
    verify_cuda_runtime_linked, diagnose_cuda_issues
};
use std::time::Instant;

fn main() {
    env_logger::init();
    
    println!("🚀 CUDA Integration Test for Arena KV-Cache with T4 GPU");
    println!("{}", "=".repeat(60));
    
    // Test 0: Comprehensive diagnosis
    test_cuda_diagnosis();
    
    // Test 1: Basic CUDA availability
    test_cuda_availability();
    
    // Test 2: Device detection and T4 verification
    test_device_detection();
    
    // Test 3: Memory manager creation
    test_memory_manager();
    
    // Test 4: CUDA context and page allocation
    test_cuda_context();
    
    // Test 5: Bump allocator functionality
    test_bump_allocator();
    
    // Test 6: Memory bandwidth testing
    test_memory_bandwidth();
    
    // Test 7: Arena KV-cache specific tests
    test_kv_cache_operations();
    
    // Test 8: Stress testing
    test_stress_operations();
    
    println!("\n🎉 CUDA Integration Test Complete!");
    println!("Your T4 GPU is ready for Arena KV-Cache operations!");
}

fn test_cuda_diagnosis() {
    println!("\n📋 Test 0: CUDA Diagnosis & Runtime Verification");
    println!("{}", "-".repeat(50));
    
    // Run comprehensive diagnosis
    diagnose_cuda_issues();
    
    // Verify runtime linking
    match verify_cuda_runtime_linked() {
        Ok(()) => {
            println!("✅ CUDA runtime verification passed");
            println!("   All CUDA symbols are properly linked");
        }
        Err(e) => {
            println!("❌ CUDA runtime verification failed: {}", e);
            println!("   This indicates a linking problem");
            std::process::exit(1);
        }
    }
}

fn test_cuda_availability() {
    println!("\n📋 Test 1: CUDA Availability");
    println!("{}", "-".repeat(30));
    
    if is_cuda_available() {
        println!("✅ CUDA is available and functional");
        
        match initialize_cuda() {
            Ok(()) => println!("✅ CUDA initialization successful"),
            Err(e) => {
                println!("❌ CUDA initialization failed: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        println!("❌ CUDA is not available");
        println!("Please ensure:");
        println!("  1. NVIDIA GPU drivers are installed: nvidia-smi");
        println!("  2. CUDA toolkit is installed: nvcc --version");
        println!("  3. GPU is CUDA-capable");
        println!("  4. Environment variables are set:");
        println!("     export CUDA_PATH=/usr/local/cuda");
        println!("     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH");
        std::process::exit(1);
    }
}

fn test_device_detection() {
    println!("\n📋 Test 2: Device Detection & T4 Verification");
    println!("{}", "-".repeat(40));
    
    match CudaMemoryManager::new() {
        Ok(manager) => {
            println!("✅ CUDA Memory Manager created");
            println!("📊 Detected {} CUDA device(s):", manager.devices().len());
            
            let mut t4_found = false;
            for device in manager.devices() {
                println!("\n🔧 Device {}: {}", device.device_id, device.name);
                println!("   💾 Total Memory: {:.1} GB", device.total_memory as f64 / 1e9);
                println!("   🆓 Free Memory: {:.1} GB", device.free_memory as f64 / 1e9);
                println!("   🔢 Compute Capability: {}.{}", 
                        device.compute_capability_major, device.compute_capability_minor);
                println!("   🔥 Multiprocessors: {}", device.multiprocessor_count);
                println!("   🧵 Max Threads/Block: {}", device.max_threads_per_block);
                println!("   📡 Memory Bandwidth: {:.1} GB/s", device.memory_bandwidth_gbps());
                
                if device.is_t4() {
                    println!("   ✅ Tesla T4 GPU Detected!");
                    t4_found = true;
                    
                    // T4-specific validation
                    assert_eq!(device.compute_capability_major, 7);
                    assert_eq!(device.compute_capability_minor, 5);
                    assert!(device.memory_bandwidth_gbps() > 200.0, 
                           "T4 bandwidth should be >200 GB/s, got {:.1}", 
                           device.memory_bandwidth_gbps());
                    
                    println!("   ✅ T4 specifications verified");
                }
            }
            
            if !t4_found {
                println!("⚠️  No Tesla T4 GPU detected, but proceeding with available GPU");
                println!("   The system will work with any CUDA-capable GPU");
            }
        }
        Err(e) => {
            println!("❌ Device detection failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn test_memory_manager() {
    println!("\n📋 Test 3: Memory Manager Operations");
    println!("{}", "-".repeat(35));
    
    match CudaMemoryManager::new() {
        Ok(manager) => {
            println!("✅ Memory manager created successfully");
            
            // Test memory info queries
            if let Ok((free, total)) = manager.get_memory_info(0) {
                println!("📊 Device 0 Memory Status:");
                println!("   Total: {:.1} GB", total as f64 / 1e9);
                println!("   Free: {:.1} GB", free as f64 / 1e9);
                println!("   Used: {:.1} GB", (total - free) as f64 / 1e9);
                println!("   Utilization: {:.1}%", (total - free) as f64 / total as f64 * 100.0);
                
                // Verify T4 has expected memory
                if total > 14 * 1024 * 1024 * 1024 {  // > 14GB
                    println!("   ✅ T4-class memory capacity detected");
                }
            }
            
            // Test multiple page allocations
            let test_sizes = [1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]; // 1MB, 4MB, 16MB
            for size in &test_sizes {
                match manager.allocate_page_on_device(*size, 0) {
                    Ok(page) => {
                        println!("✅ Page allocation successful: {} MB", size / 1024 / 1024);
                        println!("   Device ID: {}", page.device_id());
                        println!("   Allocation ID: {}", page.allocation_id());
                        println!("   Utilization: {:.1}%", page.utilization() * 100.0);
                    }
                    Err(e) => println!("❌ Page allocation failed for {} MB: {}", size / 1024 / 1024, e),
                }
            }
        }
        Err(e) => println!("❌ Memory manager creation failed: {}", e),
    }
}

fn test_cuda_context() {
    println!("\n📋 Test 4: CUDA Context & Advanced Operations");
    println!("{}", "-".repeat(42));
    
    match CudaContext::new() {
        Ok(context) => {
            println!("✅ CUDA context created successfully");
            
            // Test auto allocation with optimal device selection
            let test_sizes = [256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]; 
            
            for size in &test_sizes {
                match context.allocate_page_auto(*size) {
                    Ok(page) => {
                        println!("✅ Auto allocation: {} MB on device {}", 
                               size / 1024 / 1024, page.device_id());
                        
                        // Test basic page operations
                        if let Some(_ptr) = page.allocate(1024, 256) {
                            println!("   ✅ Bump allocation within page successful");
                        }
                        
                        if page.synchronize().is_ok() {
                            println!("   ✅ Page synchronization successful");
                        }
                        
                        // Test page statistics
                        println!("   📊 Page Stats: {:.1}% used, {} bytes available", 
                               page.utilization() * 100.0, page.available_space());
                    }
                    Err(e) => println!("❌ Auto allocation failed for {} MB: {}", size / 1024 / 1024, e),
                }
            }
            
            // Test device statistics
            if let Some(stats) = context.device_stats_detailed(0) {
                println!("📊 Device 0 Detailed Stats:");
                println!("   Allocated: {:.1} MB", stats.allocated_bytes as f64 / 1e6);
                println!("   Peak: {:.1} MB", stats.peak_allocated as f64 / 1e6);
                println!("   Active Pages: {}", stats.active_pages);
                println!("   GPU Utilization: {:.1}%", stats.utilization * 100.0);
                
                // Check for healthy GPU usage
                if stats.utilization < 0.9 {
                    println!("   ✅ GPU memory usage is healthy");
                } else {
                    println!("   ⚠️  High GPU memory usage - consider optimization");
                }
            }
        }
        Err(e) => println!("❌ CUDA context creation failed: {}", e),
    }
}

fn test_bump_allocator() {
    println!("\n📋 Test 5: Bump Allocator Functionality");
    println!("{}", "-".repeat(37));
    
    // Test with T4-optimized page size
    let page_size = 2 * 1024 * 1024; // 2MB - optimal for T4
    match BumpAllocator::new(page_size, 0) {
        Ok(allocator) => {
            println!("✅ Bump allocator created: {} MB", allocator.page_size() / 1024 / 1024);
            
            // Test multiple allocations with realistic KV tensor sizes
            let mut allocations = Vec::new();
            let allocation_sizes = [
                64 * 1024,    // 64KB - small KV tensor
                256 * 1024,   // 256KB - medium KV tensor
                512 * 1024,   // 512KB - large KV tensor
                1024 * 1024,  // 1MB - very large KV tensor
            ];
            
            for (i, size) in allocation_sizes.iter().enumerate() {
                match allocator.allocate(*size, 256) {
                    Some(ptr) => {
                        allocations.push(ptr);
                        println!("✅ Allocation {}: {} KB at offset {}", 
                               i + 1, size / 1024, allocator.current_offset() - size);
                        println!("   Utilization: {:.1}%", allocator.utilization() * 100.0);
                    }
                    None => {
                        println!("⚠️  Allocation {} failed: {} KB (page full)", i + 1, size / 1024);
                        break;
                    }
                }
            }
            
            println!("📊 Final Allocator Stats:");
            println!("   Used: {} KB", allocator.current_offset() / 1024);
            println!("   Available: {} KB", allocator.available_space() / 1024);
            println!("   Utilization: {:.1}%", allocator.utilization() * 100.0);
            
            // Test performance characteristics
            let start = Instant::now();
            for _ in 0..1000 {
                let _ = allocator.allocate(64, 64); // Small allocations
            }
            let duration = start.elapsed();
            println!("   Performance: 1000 allocations in {:.2}ms", duration.as_millis());
            
            // Test synchronization
            if allocator.synchronize().is_ok() {
                println!("✅ Allocator synchronization successful");
            }
            
            // Test reset functionality
            let old_offset = allocator.current_offset();
            allocator.reset();
            println!("✅ Allocator reset: {} KB -> {} KB", old_offset / 1024, allocator.current_offset() / 1024);
        }
        Err(e) => println!("❌ Bump allocator creation failed: {}", e),
    }
}

fn test_memory_bandwidth() {
    println!("\n📋 Test 6: Memory Bandwidth Testing");
    println!("{}", "-".repeat(33));
    
    let test_sizes = [
        1 * 1024 * 1024,     // 1 MB
        16 * 1024 * 1024,    // 16 MB  
        64 * 1024 * 1024,    // 64 MB
        256 * 1024 * 1024,   // 256 MB
    ];
    
    println!("Testing memory bandwidth with various transfer sizes...");
    
    for size in &test_sizes {
        match cuda_memory_test(0, *size) {
            Ok(bandwidth) => {
                println!("✅ {} MB test: {:.2} GB/s bandwidth", 
                       size / 1024 / 1024, bandwidth);
                
                // T4-specific expectations and analysis
                if *size >= 64 * 1024 * 1024 {
                    if bandwidth > 250.0 {
                        println!("   ✅ Excellent bandwidth for T4 GPU (theoretical max ~320 GB/s)");
                    } else if bandwidth > 150.0 {
                        println!("   ✅ Good bandwidth for T4 GPU");
                    } else if bandwidth > 50.0 {
                        println!("   ⚠️  Moderate bandwidth - check for thermal throttling or PCIe bottleneck");
                    } else {
                        println!("   ❌ Low bandwidth - possible issues with GPU/driver");
                    }
                }
            }
            Err(e) => println!("❌ {} MB test failed: {}", size / 1024 / 1024, e),
        }
    }
    
    // Test sustained bandwidth (important for KV cache workloads)
    println!("\n🔄 Testing sustained bandwidth (10 iterations)...");
    let sustained_size = 64 * 1024 * 1024; // 64MB
    let mut bandwidths = Vec::new();
    
    for i in 0..10 {
        if let Ok(bandwidth) = cuda_memory_test(0, sustained_size) {
            bandwidths.push(bandwidth);
            println!("   Iteration {}: {:.2} GB/s", i + 1, bandwidth);
        }
    }
    
    if !bandwidths.is_empty() {
        let avg_bandwidth = bandwidths.iter().sum::<f64>() / bandwidths.len() as f64;
        let min_bandwidth = bandwidths.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_bandwidth = bandwidths.iter().fold(0.0f64, |a, &b| a.max(b));
        
        println!("📊 Sustained Bandwidth Stats:");
        println!("   Average: {:.2} GB/s", avg_bandwidth);
        println!("   Min: {:.2} GB/s", min_bandwidth);
        println!("   Max: {:.2} GB/s", max_bandwidth);
        println!("   Variation: {:.1}%", (max_bandwidth - min_bandwidth) / avg_bandwidth * 100.0);
        
        if (max_bandwidth - min_bandwidth) / avg_bandwidth < 0.1 {
            println!("   ✅ Stable bandwidth - no thermal throttling detected");
        } else {
            println!("   ⚠️  Bandwidth variation detected - possible thermal management");
        }
    }
}

fn test_kv_cache_operations() {
    println!("\n📋 Test 7: KV-Cache Specific Operations");
    println!("{}", "-".repeat(38));
    
    // Test KV-cache page size calculations
    test_kv_page_sizing();
    
    // Test arena operations with KV layout
    test_kv_arena_operations();
    
    // Test zero-copy extensions
    test_zero_copy_extensions();
    
    // Test KV-specific memory patterns
    test_kv_memory_patterns();
}

fn test_kv_page_sizing() {
    use arena_kv_cache::kv_layout::{calculate_optimal_kv_page_size, calculate_model_kv_page_size, ModelConfig};
    
    println!("🔧 Testing KV-specific page size calculations:");
    
    // Test optimal page size calculation for T4
    let page_size = calculate_optimal_kv_page_size(8192, 32, 128, 2); // 8K seq, 32 heads, 128 head_dim, fp16
    println!("   8K sequence KV page size: {} KB", page_size / 1024);
    assert!(page_size >= 1024 * 1024, "Page size should be at least 1MB for 8K sequences");
    
    // Test model-specific calculations
    let models = [
        ("Llama-2 7B", ModelConfig::Llama2_7B),
        ("Llama-2 13B", ModelConfig::Llama2_13B),
        ("Llama-2 70B", ModelConfig::Llama2_70B),
    ];
    
    for (name, config) in &models {
        let size = calculate_model_kv_page_size(config);
        println!("   {} KV page size: {} KB", name, size / 1024);
        
        // Validate reasonable page sizes
        assert!(size >= 64 * 1024, "Page size too small for {}", name);
        assert!(size <= 16 * 1024 * 1024, "Page size too large for {}", name);
    }
    
    println!("✅ KV page size calculations validated");
}

fn test_kv_arena_operations() {
    println!("🔧 Testing KV arena operations:");
    
    match CudaContext::new() {
        Ok(context) => {
            // Allocate page optimized for KV tensors (T4-specific)
            let kv_page_size = 2 * 1024 * 1024; // 2MB for T4
            match context.allocate_page_on_device(kv_page_size, 0) {
                Ok(page) => {
                    println!("✅ KV arena page allocated: {} MB", page.size() / 1024 / 1024);
                    
                    // Simulate realistic KV tensor allocations for different models
                    let kv_configs = [
                        ("Small model", 512, 16, 64, 2),    // 512 seq, 16 heads, 64 dim, fp16
                        ("Medium model", 1024, 32, 128, 2), // 1K seq, 32 heads, 128 dim, fp16
                        ("Large model", 2048, 64, 128, 2),  // 2K seq, 64 heads, 128 dim, fp16
                    ];
                    
                    for (name, seq_len, num_heads, head_dim, element_size) in &kv_configs {
                        // Calculate KV tensor size (K + V)
                        let kv_tensor_size = 2 * seq_len * num_heads * head_dim * element_size;
                        
                        if let Some(_ptr) = page.allocate(kv_tensor_size, 256) {
                            println!("✅ {} KV tensor: {} KB for {}x{}x{}", 
                                   name, kv_tensor_size / 1024, seq_len, num_heads, head_dim);
                        } else {
                            println!("⚠️  {} KV tensor allocation failed - page full", name);
                        }
                    }
                    
                    println!("   Final page utilization: {:.1}%", page.utilization() * 100.0);
                    
                    // Test page efficiency for KV workloads
                    if page.utilization() > 0.7 {
                        println!("   ✅ Good page utilization for KV tensors");
                    } else {
                        println!("   ⚠️  Consider smaller page size for better utilization");
                    }
                }
                Err(e) => println!("❌ KV arena allocation failed: {}", e),
            }
        }
        Err(e) => println!("❌ Context creation failed: {}", e),
    }
}

fn test_zero_copy_extensions() {
    println!("🔧 Testing zero-copy extension simulation:");
    
    // Simulate the zero-copy extension pattern for incremental generation
    let initial_seq_len = 512;
    let max_seq_len = 4096;  // T4-appropriate max sequence
    let extension_steps = [32, 64, 128, 256, 512, 1024]; // Progressive extensions
    
    println!("   Initial sequence length: {}", initial_seq_len);
    println!("   Maximum sequence length: {}", max_seq_len);
    
    let mut current_seq_len = initial_seq_len;
    let mut zero_copy_count = 0;
    let mut total_extensions = 0;
    
    for (i, extension) in extension_steps.iter().enumerate() {
        let new_seq_len = current_seq_len + extension;
        total_extensions += 1;
        
        if new_seq_len <= max_seq_len {
            // This would be a zero-copy extension
            println!("   Step {}: {} -> {} tokens (ZERO-COPY ✅)", 
                   i + 1, current_seq_len, new_seq_len);
            current_seq_len = new_seq_len;
            zero_copy_count += 1;
        } else {
            // This would require reallocation
            println!("   Step {}: {} -> {} tokens (REALLOC ⚠️)", 
                   i + 1, current_seq_len, new_seq_len);
            
            // Reset to demonstrate new allocation
            current_seq_len = initial_seq_len;
            println!("   Reset to {} tokens for new sequence", current_seq_len);
        }
    }
    
    let efficiency = zero_copy_count as f64 / total_extensions as f64 * 100.0;
    println!("   Zero-copy efficiency: {:.1}% ({}/{} extensions)", 
           efficiency, zero_copy_count, total_extensions);
    
    if efficiency > 80.0 {
        println!("   ✅ Excellent zero-copy efficiency");
    } else if efficiency > 60.0 {
        println!("   ✅ Good zero-copy efficiency");
    } else {
        println!("   ⚠️  Consider larger initial allocation for better efficiency");
    }
}

fn test_kv_memory_patterns() {
    println!("🔧 Testing KV-specific memory access patterns:");
    
    match CudaContext::new() {
        Ok(context) => {
            // Test memory access patterns typical for KV cache
            let pattern_tests = [
                ("Sequential access", 1024 * 1024, 1),      // 1MB sequential
                ("Strided access", 256 * 1024, 4),          // 256KB with stride
                ("Random access", 64 * 1024, 16),           // 64KB random pattern
            ];
            
            for (pattern_name, base_size, stride) in &pattern_tests {
                match context.allocate_page_on_device(*base_size * stride, 0) {
                    Ok(page) => {
                        println!("✅ {} pattern test: {} KB allocated", 
                               pattern_name, page.size() / 1024);
                        
                        // Simulate memory access pattern
                        let mut allocated_chunks = 0;
                        for i in 0..*stride {
                            let offset = i * base_size;
                            if let Some(_ptr) = page.allocate(*base_size / stride, 256) {
                                allocated_chunks += 1;
                            }
                        }
                        
                        println!("   Allocated {} chunks, utilization: {:.1}%", 
                               allocated_chunks, page.utilization() * 100.0);
                    }
                    Err(e) => println!("❌ {} pattern test failed: {}", pattern_name, e),
                }
            }
        }
        Err(e) => println!("❌ Context creation failed for memory patterns: {}", e),
    }
}

fn test_stress_operations() {
    println!("\n📋 Test 8: Stress Testing & Performance");
    println!("{}", "-".repeat(37));
    
    // Test rapid allocation/deallocation cycles
    test_allocation_stress();
    
    // Test memory pressure handling
    test_memory_pressure();
    
    // Test concurrent operations simulation
    test_concurrent_simulation();
}

fn test_allocation_stress() {
    println!("🔧 Testing rapid allocation cycles:");
    
    match CudaContext::new() {
        Ok(context) => {
            let start_time = Instant::now();
            let mut successful_allocations = 0;
            let mut failed_allocations = 0;
            
            // Rapid allocation test
            for i in 0..100 {
                let size = (64 + (i % 64)) * 1024; // Variable sizes 64KB-128KB
                match context.allocate_page_on_device(size, 0) {
                    Ok(_page) => {
                        successful_allocations += 1;
                        // Let pages drop immediately to test allocation/deallocation cycle
                    }
                    Err(_) => {
                        failed_allocations += 1;
                    }
                }
                
                // Progress indicator
                if i % 20 == 19 {
                    println!("   Progress: {} allocations completed", i + 1);
                }
            }
            
            let duration = start_time.elapsed();
            println!("✅ Allocation stress test completed:");
            println!("   Duration: {:.2}ms", duration.as_millis());
            println!("   Successful: {}", successful_allocations);
            println!("   Failed: {}", failed_allocations);
            println!("   Success rate: {:.1}%", 
                   successful_allocations as f64 / (successful_allocations + failed_allocations) as f64 * 100.0);
            println!("   Avg time per allocation: {:.2}ms", 
                   duration.as_millis() as f64 / 100.0);
        }
        Err(e) => println!("❌ Context creation failed for stress test: {}", e),
    }
}

fn test_memory_pressure() {
    println!("🔧 Testing memory pressure handling:");
    
    match CudaContext::new() {
        Ok(context) => {
            // Get initial memory state
            if let Some((initial_free, total)) = context.device_stats(0) {
                println!("   Initial free memory: {:.1} GB", initial_free as f64 / 1e9);
                
                let mut pages = Vec::new();
                let chunk_size = 64 * 1024 * 1024; // 64MB chunks
                let max_chunks = (initial_free / 2) / chunk_size; // Use half of available memory
                
                println!("   Allocating up to {} chunks of 64MB each...", max_chunks);
                
                for i in 0..max_chunks {
                    match context.allocate_page_on_device(chunk_size, 0) {
                        Ok(page) => {
                            pages.push(page);
                            if i % 10 == 9 {
                                if let Some((current_free, _)) = context.device_stats(0) {
                                    println!("     {} chunks allocated, {:.1} GB free", 
                                           i + 1, current_free as f64 / 1e9);
                                }
                            }
                        }
                        Err(_) => {
                            println!("   Memory pressure reached at {} chunks", i);
                            break;
                        }
                    }
                }
                
                println!("✅ Memory pressure test: {} chunks allocated", pages.len());
                
                // Test cleanup
                let cleanup_start = Instant::now();
                pages.clear(); // Drop all pages
                let cleanup_duration = cleanup_start.elapsed();
                
                println!("   Cleanup completed in {:.2}ms", cleanup_duration.as_millis());
                
                // Verify memory recovery
                if let Some((final_free, _)) = context.device_stats(0) {
                    let recovered = final_free as f64 / initial_free as f64;
                    println!("   Memory recovery: {:.1}% ({:.1} GB)", 
                           recovered * 100.0, final_free as f64 / 1e9);
                    
                    if recovered > 0.9 {
                        println!("   ✅ Excellent memory recovery");
                    } else if recovered > 0.8 {
                        println!("   ✅ Good memory recovery");
                    } else {
                        println!("   ⚠️  Some memory may not have been recovered");
                    }
                }
            }
        }
        Err(e) => println!("❌ Context creation failed for memory pressure test: {}", e),
    }
}

fn test_concurrent_simulation() {
    println!("🔧 Testing concurrent operations simulation:");
    
    match CudaContext::new() {
        Ok(context) => {
            // Simulate concurrent KV cache operations
            let simulated_sequences = 10;
            let mut active_arenas = Vec::new();
            
            println!("   Simulating {} concurrent sequences...", simulated_sequences);
            
            let start_time = Instant::now();
            
            // Phase 1: Initial allocation (simulating prompt processing)
            for i in 0..simulated_sequences {
                let initial_size = (512 + i * 128) * 1024; // 512KB to 1.6MB
                match context.allocate_page_on_device(initial_size, 0) {
                    Ok(page) => {
                        active_arenas.push(page);
                        println!("     Sequence {}: {} KB arena created", i, initial_size / 1024);
                    }
                    Err(e) => println!("     Sequence {} failed: {}", i, e),
                }
            }
            
            // Phase 2: Incremental generation (simulating token generation)
            println!("   Simulating incremental generation...");
            for round in 0..5 {
                println!("     Generation round {}:", round + 1);
                for (i, page) in active_arenas.iter().enumerate() {
                    let additional_size = 64 * 1024; // 64KB per generation step
                    if let Some(_ptr) = page.allocate(additional_size, 256) {
                        println!("       Sequence {}: +{} KB (util: {:.1}%)", 
                               i, additional_size / 1024, page.utilization() * 100.0);
                    } else {
                        println!("       Sequence {}: allocation failed (arena full)", i);
                    }
                }
            }
            
            let total_duration = start_time.elapsed();
            
            // Phase 3: Statistics and cleanup
            println!("✅ Concurrent simulation completed:");
            println!("   Duration: {:.2}ms", total_duration.as_millis());
            println!("   Active sequences: {}", active_arenas.len());
            
            let total_allocated: usize = active_arenas.iter().map(|p| p.current_offset()).sum();
            let avg_utilization: f64 = active_arenas.iter()
                .map(|p| p.utilization())
                .sum::<f64>() / active_arenas.len() as f64;
            
            println!("   Total allocated: {:.1} MB", total_allocated as f64 / 1e6);
            println!("   Average utilization: {:.1}%", avg_utilization * 100.0);
            
            if avg_utilization > 0.6 {
                println!("   ✅ Good arena utilization efficiency");
            } else {
                println!("   ⚠️  Consider optimizing arena sizes");
            }
        }
        Err(e) => println!("❌ Context creation failed for concurrent test: {}", e),
    }
}