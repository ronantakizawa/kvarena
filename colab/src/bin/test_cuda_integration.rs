// src/bin/test_cuda_integration.rs - Test script for CUDA integration
use arena_kv_cache::cuda::{
    initialize_cuda, CudaMemoryManager, CudaContext, CudaPage, 
    BumpAllocator, cuda_memory_test, is_cuda_available
};
use std::time::Instant;

fn main() {
    env_logger::init();
    
    println!("🚀 CUDA Integration Test for Arena KV-Cache");
    
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
    
    println!("\n🎉 CUDA Integration Test Complete!");
}

fn test_cuda_availability() {
    println!("\n📋 Test 1: CUDA Availability");
    
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
        println!("  1. NVIDIA GPU drivers are installed");
        println!("  2. CUDA toolkit is installed");
        println!("  3. GPU is CUDA-capable");
        std::process::exit(1);
    }
}

fn test_device_detection() {
    println!("\n📋 Test 2: Device Detection & T4 Verification");
    
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
                    assert!(device.memory_bandwidth_gbps() > 200.0, "T4 bandwidth should be >200 GB/s");
                }
            }
            
            if !t4_found {
                println!("⚠️  No Tesla T4 GPU detected, but proceeding with available GPU");
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
            }
            
            // Test page allocation
            match manager.allocate_page_on_device(1024 * 1024, 0) {
                Ok(page) => {
                    println!("✅ Test page allocation successful: {} bytes", page.size());
                    println!("   Device ID: {}", page.device_id());
                    println!("   Allocation ID: {}", page.allocation_id());
                }
                Err(e) => println!("❌ Test page allocation failed: {}", e),
            }
        }
        Err(e) => println!("❌ Memory manager creation failed: {}", e),
    }
}

fn test_cuda_context() {
    println!("\n📋 Test 4: CUDA Context & Page Allocation");
    
    match CudaContext::new() {
        Ok(context) => {
            println!("✅ CUDA context created successfully");
            
            // Test auto allocation
            let test_sizes = [256 * 1024, 1024 * 1024, 4 * 1024 * 1024]; // 256KB, 1MB, 4MB
            
            for size in &test_sizes {
                match context.allocate_page_auto(*size) {
                    Ok(page) => {
                        println!("✅ Auto allocation: {} KB on device {}", 
                               size / 1024, page.device_id());
                        
                        // Test basic page operations
                        if let Some(_ptr) = page.allocate(1024, 256) {
                            println!("   ✅ Bump allocation within page successful");
                        }
                        
                        if page.synchronize().is_ok() {
                            println!("   ✅ Page synchronization successful");
                        }
                    }
                    Err(e) => println!("❌ Auto allocation failed for {} KB: {}", size / 1024, e),
                }
            }
            
            // Test device stats
            if let Some(stats) = context.device_stats_detailed(0) {
                println!("📊 Device 0 Detailed Stats:");
                println!("   Allocated: {:.1} MB", stats.allocated_bytes as f64 / 1e6);
                println!("   Peak: {:.1} MB", stats.peak_allocated as f64 / 1e6);
                println!("   Active Pages: {}", stats.active_pages);
                println!("   GPU Utilization: {:.1}%", stats.utilization * 100.0);
            }
        }
        Err(e) => println!("❌ CUDA context creation failed: {}", e),
    }
}

fn test_bump_allocator() {
    println!("\n📋 Test 5: Bump Allocator Functionality");
    
    match BumpAllocator::new(64 * 1024, 0) {
        Ok(allocator) => {
            println!("✅ Bump allocator created: {} KB", allocator.page_size() / 1024);
            
            // Test multiple allocations
            let mut allocations = Vec::new();
            let allocation_sizes = [1024, 2048, 4096, 8192, 16384]; // Various sizes
            
            for (i, size) in allocation_sizes.iter().enumerate() {
                match allocator.allocate(*size, 256) {
                    Some(ptr) => {
                        allocations.push(ptr);
                        println!("✅ Allocation {}: {} bytes at offset {}", 
                               i + 1, size, allocator.current_offset() - size);
                    }
                    None => {
                        println!("⚠️  Allocation {} failed: {} bytes (likely out of space)", i + 1, size);
                        break;
                    }
                }
            }
            
            println!("📊 Final Allocator Stats:");
            println!("   Used: {} KB", allocator.current_offset() / 1024);
            println!("   Available: {} KB", allocator.available_space() / 1024);
            println!("   Utilization: {:.1}%", allocator.utilization() * 100.0);
            
            // Test synchronization
            if allocator.synchronize().is_ok() {
                println!("✅ Allocator synchronization successful");
            }
            
            // Test reset
            allocator.reset();
            println!("✅ Allocator reset: offset now {}", allocator.current_offset());
        }
        Err(e) => println!("❌ Bump allocator creation failed: {}", e),
    }
}

fn test_memory_bandwidth() {
    println!("\n📋 Test 6: Memory Bandwidth Testing");
    
    let test_sizes = [
        1 * 1024 * 1024,   // 1 MB
        16 * 1024 * 1024,  // 16 MB  
        64 * 1024 * 1024,  // 64 MB
        256 * 1024 * 1024, // 256 MB
    ];
    
    for size in &test_sizes {
        match cuda_memory_test(0, *size) {
            Ok(bandwidth) => {
                println!("✅ {} MB test: {:.2} GB/s bandwidth", 
                       size / 1024 / 1024, bandwidth);
                
                // T4 specific expectations
                if *size >= 64 * 1024 * 1024 {
                    if bandwidth > 200.0 {
                        println!("   ✅ Excellent bandwidth for T4 GPU");
                    } else if bandwidth > 100.0 {
                        println!("   ⚠️  Moderate bandwidth - check for thermal throttling");
                    } else {
                        println!("   ❌ Low bandwidth - possible issues with GPU/driver");
                    }
                }
            }
            Err(e) => println!("❌ {} MB test failed: {}", size / 1024 / 1024, e),
        }
    }
}

fn test_kv_cache_operations() {
    println!("\n📋 Test 7: KV-Cache Specific Operations");
    
    // Test KV-cache page size calculations
    test_kv_page_sizing();
    
    // Test arena operations with KV layout
    test_kv_arena_operations();
    
    // Test zero-copy extensions
    test_zero_copy_extensions();
}

fn test_kv_page_sizing() {
    use arena_kv_cache::kv_layout::{calculate_optimal_kv_page_size, calculate_model_kv_page_size, ModelConfig};
    
    println!("🔧 Testing KV-specific page size calculations:");
    
    // Test optimal page size calculation
    let page_size = calculate_optimal_kv_page_size(8192, 32, 128, 2); // 8K seq, 32 heads, 128 head_dim, fp16
    println!("   8K sequence KV page size: {} KB", page_size / 1024);
    
    // Test model-specific calculations
    let models = [
        ("Llama-2 7B", ModelConfig::Llama2_7B),
        ("Llama-2 13B", ModelConfig::Llama2_13B),
        ("Llama-2 70B", ModelConfig::Llama2_70B),
    ];
    
    for (name, config) in &models {
        let size = calculate_model_kv_page_size(config);
        println!("   {} KV page size: {} KB", name, size / 1024);
    }
}

fn test_kv_arena_operations() {
    println!("🔧 Testing KV arena operations:");
    
    match CudaContext::new() {
        Ok(context) => {
            // Allocate page optimized for KV tensors
            let kv_page_size = 512 * 1024; // 512KB for KV tensors
            match context.allocate_page_on_device(kv_page_size, 0) {
                Ok(page) => {
                    println!("✅ KV arena page allocated: {} KB", page.size() / 1024);
                    
                    // Simulate KV tensor allocation
                    let seq_len = 1024;
                    let num_heads = 32;
                    let head_dim = 128;
                    let element_size = 2; // fp16
                    
                    // Calculate KV tensor size (K + V)
                    let kv_tensor_size = 2 * seq_len * num_heads * head_dim * element_size;
                    
                    if let Some(_ptr) = page.allocate(kv_tensor_size, 256) {
                        println!("✅ KV tensor allocation: {} KB for {}x{}x{}", 
                               kv_tensor_size / 1024, seq_len, num_heads, head_dim);
                        
                        println!("   Page utilization: {:.1}%", page.utilization() * 100.0);
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
    
    // Simulate the zero-copy extension pattern
    let initial_seq_len = 512;
    let max_seq_len = 2048;
    let extension_steps = [64, 128, 256, 512, 1024]; // Progressive extensions
    
    println!("   Initial sequence length: {}", initial_seq_len);
    println!("   Maximum sequence length: {}", max_seq_len);
    
    let mut current_seq_len = initial_seq_len;
    let mut zero_copy_count = 0;
    
    for (i, extension) in extension_steps.iter().enumerate() {
        let new_seq_len = current_seq_len + extension;
        
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
            break;
        }
    }
    
    let efficiency = zero_copy_count as f64 / extension_steps.len() as f64 * 100.0;
    println!("   Zero-copy efficiency: {:.1}% ({}/{} extensions)", 
           efficiency, zero_copy_count, extension_steps.len());
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn integration_test_basic() {
        if is_cuda_available() {
            // Run basic tests in test mode
            test_cuda_availability();
            test_device_detection();
        }
    }
}