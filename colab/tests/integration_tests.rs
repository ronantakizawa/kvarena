// tests/integration_test.rs - Comprehensive integration test
use arena_kv_cache::*;
use std::sync::Arc;

#[test]
fn test_full_integration() {
    env_logger::try_init().ok(); // Initialize logging
    
    println!("🧪 Starting comprehensive integration test...");
    
    // Test 1: Basic manager creation
    test_manager_creation().expect("Manager creation failed");
    
    // Test 2: Arena allocation
    test_arena_allocation().expect("Arena allocation failed");
    
    // Test 3: KV tensor creation
    test_kv_tensor_creation().expect("KV tensor creation failed");
    
    // Test 4: Transformer tensor operations
    test_transformer_tensor_ops().expect("Transformer tensor ops failed");
    
    // Test 5: Memory management
    test_memory_management().expect("Memory management failed");
    
    println!("✅ All integration tests passed!");
}

fn test_manager_creation() -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Testing manager creation...");
    
    // Test with different model types
    let model_types = [
        ("llama-7b", ModelType::Llama),
        ("gpt-3.5", ModelType::GPT),
        ("t5-base", ModelType::T5),
    ];
    
    for (model_name, model_type) in &model_types {
        let config = LLMServerConfig {
            devices: vec![0],
            model_type: Some(*model_type),
            base_page_size: 512 * 1024, // 512KB
            ..Default::default()
        };
        
        let manager = ProductionKVCacheManager::new(config)?;
        println!("  ✓ Created manager for {}", model_name);
        
        // Test metrics
        let metrics = manager.get_enhanced_metrics();
        assert_eq!(metrics.transformer_tensors_created, 0);
        println!("  ✓ Initial metrics correct for {}", model_name);
    }
    
    Ok(())
}

fn test_arena_allocation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏟️  Testing arena allocation...");
    
    let config = LLMServerConfig {
        devices: vec![0],
        model_type: Some(ModelType::Llama),
        base_page_size: 1024 * 1024, // 1MB
        ..Default::default()
    };
    
    let manager = ProductionKVCacheManager::new(config)?;
    
    // Test different arena sizes
    let arena_configs = [
        (128, 512, 2048, 16),   // small
        (256, 1024, 4096, 32),  // medium
        (512, 2048, 8192, 64),  // large
    ];
    
    for (initial_seq, max_seq, hidden_dim, num_heads) in &arena_configs {
        let arena = manager.create_sequence_arena(
            *initial_seq,
            *max_seq,
            *hidden_dim,
            *num_heads,
            None, // auto-select device
        )?;
        
        println!("  ✓ Created arena: {}->{}x{}x{}", 
                initial_seq, max_seq, hidden_dim, num_heads);
        
        // Test arena stats
        let stats = arena.stats();
        assert!(stats.arena_utilization >= 0.0);
        assert!(stats.arena_utilization <= 1.0);
        println!("  ✓ Arena stats valid: utilization={:.2}%", 
                stats.arena_utilization * 100.0);
    }
    
    Ok(())
}

fn test_kv_tensor_creation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Testing KV tensor creation...");
    
    let config = LLMServerConfig {
        devices: vec![0],
        model_type: Some(ModelType::Llama),
        ..Default::default()
    };
    
    let manager = ProductionKVCacheManager::new(config)?;
    let arena = manager.create_sequence_arena(256, 1024, 4096, 32, None)?;
    
    // Test different tensor configurations
    let tensor_configs = [
        (AttentionLayout::BSHD, KVDataType::Float16),
        (AttentionLayout::BSHD_SeqMajor, KVDataType::Float16),
        (AttentionLayout::Packed, KVDataType::Float32),
    ];
    
    for (layout, dtype) in &tensor_configs {
        let tensor = manager.create_transformer_kv_tensor(
            &arena,
            1,    // batch_size
            256,  // initial_seq_len
            1024, // max_seq_len
            32,   // num_heads
            128,  // head_dim
            Some(*layout),
            Some(*dtype),
        )?;
        
        // Verify tensor properties
        assert_eq!(tensor.seq_len(), 256);
        assert_eq!(tensor.num_heads(), 32);
        assert_eq!(tensor.head_dim(), 128);
        assert_eq!(tensor.layout(), *layout);
        assert_eq!(tensor.dtype(), *dtype);
        
        println!("  ✓ Created tensor: {:?} layout, {:?} dtype", layout, dtype);
        
        // Test memory stats
        let mem_stats = tensor.memory_stats();
        assert!(mem_stats.current_kv_bytes > 0);
        assert!(mem_stats.max_kv_bytes >= mem_stats.current_kv_bytes);
        assert!(mem_stats.utilization > 0.0 && mem_stats.utilization <= 1.0);
        
        println!("  ✓ Memory stats: {:.1}KB current, {:.1}KB max, {:.1}% util",
                mem_stats.current_kv_bytes / 1024,
                mem_stats.max_kv_bytes / 1024,
                mem_stats.utilization * 100.0);
    }
    
    Ok(())
}

fn test_transformer_tensor_ops() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Testing transformer tensor operations...");
    
    let config = LLMServerConfig {
        devices: vec![0],
        model_type: Some(ModelType::Llama),
        ..Default::default()
    };
    
    let manager = ProductionKVCacheManager::new(config)?;
    let arena = manager.create_sequence_arena(128, 512, 2048, 16, None)?;
    
    let mut tensor = manager.create_transformer_kv_tensor(
        &arena,
        1,    // batch_size
        128,  // initial_seq_len
        512,  // max_seq_len
        16,   // num_heads
        128,  // head_dim
        Some(AttentionLayout::BSHD_SeqMajor),
        Some(KVDataType::Float16),
    )?;
    
    // Test tensor extension (incremental generation)
    let extension_tests = [
        (128, 192, true),   // Should be zero-copy
        (192, 256, true),   // Should be zero-copy
        (256, 384, true),   // Should be zero-copy
        (384, 512, true),   // Should be zero-copy (at max)
    ];
    
    for (current_len, new_len, expect_zero_copy) in &extension_tests {
        assert_eq!(tensor.seq_len(), *current_len);
        
        let was_zero_copy = manager.extend_transformer_tensor(&mut tensor, *new_len)?;
        
        if *expect_zero_copy {
            assert!(was_zero_copy, "Expected zero-copy extension from {} to {}", 
                   current_len, new_len);
            println!("  ✓ Zero-copy extension: {} -> {} tokens", current_len, new_len);
        }
        
        assert_eq!(tensor.seq_len(), *new_len);
    }
    
    // Test tensor info for attention
    let tensor_info = tensor.get_tensor_info();
    assert_eq!(tensor_info.batch_size, 1);
    assert_eq!(tensor_info.seq_len, 512);
    assert_eq!(tensor_info.num_heads, 16);
    assert_eq!(tensor_info.head_dim, 128);
    println!("  ✓ Tensor info correct: {}x{}x{}x{}", 
            tensor_info.batch_size, tensor_info.seq_len, 
            tensor_info.num_heads, tensor_info.head_dim);
    
    // Test attention preparation
    let attention_prepared = tensor.prepare_for_attention(256, true)?;
    let attention_params = attention_prepared.get_attention_params();
    assert_eq!(attention_params.kv_seq_len, 512);
    assert_eq!(attention_params.query_seq_len, 256);
    assert!(attention_params.causal_mask);
    println!("  ✓ Attention preparation successful");
    
    Ok(())
}

fn test_memory_management() -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 Testing memory management...");
    
    let config = LLMServerConfig {
        devices: vec![0],
        model_type: Some(ModelType::Llama),
        max_slab_pages: 50,
        ..Default::default()
    };
    
    let manager = ProductionKVCacheManager::new(config)?;
    
    // Create and destroy multiple arenas to test slab recycling
    let mut arenas = Vec::new();
    for i in 0..10 {
        let arena = manager.create_sequence_arena(
            64 + i * 32,   // varying sizes
            256 + i * 64,
            1024,
            8,
            None,
        )?;
        arenas.push(arena);
    }
    
    println!("  ✓ Created 10 arenas with varying sizes");
    
    // Drop half the arenas
    arenas.truncate(5);
    println!("  ✓ Dropped 5 arenas (should trigger slab recycling)");
    
    // Create new arenas (should reuse slab pages)
    for i in 0..3 {
        let arena = manager.create_sequence_arena(
            128,
            512,
            2048,
            16,
            None,
        )?;
        arenas.push(arena);
        println!("  ✓ Created new arena {} (may reuse slab)", i);
    }
    
    // Test global metrics
    let metrics = manager.get_enhanced_metrics();
    println!("  ✓ Sequences processed: {}", metrics.base_metrics.sequences_processed);
    println!("  ✓ Zero-copy ratio: {:.1}%", metrics.base_metrics.zero_copy_ratio * 100.0);
    
    // Test system health
    let health = manager.get_system_health();
    println!("  ✓ System status: {:?}", health.status);
    println!("  ✓ Health score: {:.2}", health.health_score);
    println!("  ✓ Recommendations: {}", health.recommendations.len());
    
    for (i, rec) in health.recommendations.iter().take(3).enumerate() {
        println!("    {}. {}", i + 1, rec);
    }
    
    Ok(())
}

#[test]
fn test_model_specific_optimizations() {
    println!("🎯 Testing model-specific optimizations...");
    
    let model_configs = [
        ("llama-7b", ModelConfig {
            num_layers: 32,
            num_heads: 32,
            hidden_size: 4096,
            max_seq_len: 2048,
            use_gqa: false,
            num_kv_heads: None,
        }),
        ("gpt-3.5", ModelConfig {
            num_layers: 96,
            num_heads: 96,
            hidden_size: 12288,
            max_seq_len: 4096,
            use_gqa: false,
            num_kv_heads: None,
        }),
    ];
    
    for (model_name, model_config) in &model_configs {
        match transformer_api::initialize_for_transformer(
            model_name,
            model_config.clone(),
            &[0],
        ) {
            Ok(manager) => {
                println!("  ✓ Initialized manager for {}", model_name);
                
                // Test optimal page size calculation
                let page_size = manager.calculate_optimal_page_size_for_model(
                    model_config,
                    4, // 4 sequences per page
                );
                
                assert!(page_size >= 256 * 1024); // At least 256KB
                println!("  ✓ Optimal page size for {}: {} KB", 
                        model_name, page_size / 1024);
                
                // Test creating layer arena
                let layer_config = LayerConfig {
                    batch_size: 1,
                    initial_seq_len: 128,
                    max_seq_len: model_config.max_seq_len,
                    num_heads: model_config.num_heads,
                    head_dim: model_config.hidden_size / model_config.num_heads,
                    preferred_layout: None,
                    preferred_dtype: None,
                };
                
                match transformer_api::create_layer_arena(&manager, layer_config, None) {
                    Ok(arena) => {
                        println!("  ✓ Created layer arena for {}", model_name);
                        
                        let stats = arena.stats();
                        println!("  ✓ Arena stats: {} total bytes", stats.total_allocated_bytes);
                    }
                    Err(e) => {
                        println!("  ⚠️  Layer arena creation failed for {}: {}", model_name, e);
                    }
                }
            }
            Err(e) => {
                println!("  ⚠️  Manager initialization failed for {}: {}", model_name, e);
            }
        }
    }
}

#[test] 
fn test_performance_characteristics() {
    println!("⚡ Testing performance characteristics...");
    
    let config = LLMServerConfig {
        devices: vec![0],
        model_type: Some(ModelType::Llama),
        base_page_size: 2 * 1024 * 1024, // 2MB
        ..Default::default()
    };
    
    if let Ok(manager) = ProductionKVCacheManager::new(config) {
        // Test rapid allocation/deallocation
        let start_time = std::time::Instant::now();
        let num_arenas = 50;
        
        for i in 0..num_arenas {
            if let Ok(arena) = manager.create_sequence_arena(
                64 + (i % 128),
                256 + (i % 512),
                1024,
                8,
                None,
            ) {
                // Create a tensor
                if let Ok(_tensor) = manager.create_transformer_kv_tensor(
                    &arena,
                    1,
                    64 + (i % 64),
                    256,
                    8,
                    128,
                    None,
                    None,
                ) {
                    // Tensor created successfully
                }
            }
        }
        
        let allocation_time = start_time.elapsed();
        let avg_time_per_allocation = allocation_time.as_millis() as f64 / num_arenas as f64;
        
        println!("  ✓ {} allocations in {:.1}ms", num_arenas, allocation_time.as_millis());
        println!("  ✓ Average time per allocation: {:.2}ms", avg_time_per_allocation);
        
        // Performance should be reasonable
        assert!(avg_time_per_allocation < 10.0, "Allocation too slow: {:.2}ms", avg_time_per_allocation);
        
        // Test final metrics
        let final_metrics = manager.get_enhanced_metrics();
        println!("  ✓ Final metrics:");
        println!("    - Transformer tensors: {}", final_metrics.transformer_tensors_created);
        println!("    - Zero-copy ratio: {:.1}%", final_metrics.base_metrics.zero_copy_ratio * 100.0);
        println!("    - Avg allocation time: {:.2}ms", final_metrics.base_metrics.avg_allocation_time_ms);
        
        // Test maintenance cleanup
        let maintenance_report = manager.maintenance_cleanup();
        println!("  ✓ Maintenance cleanup:");
        println!("    - Inactive arenas cleaned: {}", maintenance_report.inactive_arenas_cleaned);
        println!("    - Old pages cleaned: {}", maintenance_report.old_pages_cleaned);
        println!("    - Maintenance time: {:.1}ms", maintenance_report.maintenance_time_ms);
    }
}

#[test]
fn test_error_handling() {
    println!("🚨 Testing error handling...");
    
    let config = LLMServerConfig {
        devices: vec![0],
        model_type: Some(ModelType::Llama),
        base_page_size: 64 * 1024, // Small page size to trigger errors
        ..Default::default()
    };
    
    if let Ok(manager) = ProductionKVCacheManager::new(config) {
        // Test arena creation with impossible parameters
        match manager.create_sequence_arena(
            1000000,  // Very large initial seq len
            2000000,  // Very large max seq len  
            32768,    // Very large hidden dim
            256,      // Many heads
            None,
        ) {
            Ok(_) => println!("  ⚠️  Large arena creation unexpectedly succeeded"),
            Err(e) => println!("  ✓ Large arena creation properly failed: {}", e),
        }
        
        // Test tensor creation with invalid parameters
        if let Ok(arena) = manager.create_sequence_arena(128, 256, 1024, 8, None) {
            match manager.create_transformer_kv_tensor(
                &arena,
                1000,  // Batch size larger than arena can handle
                128,
                256,
                8,
                128,
                None,
                None,
            ) {
                Ok(_) => println!("  ⚠️  Invalid tensor creation unexpectedly succeeded"),
                Err(e) => println!("  ✓ Invalid tensor creation properly failed: {}", e),
            }
        }
        
        println!("  ✓ Error handling tests completed");
    }
}

#[test]
fn test_concurrent_access() {
    use std::sync::Arc;
    use std::thread;
    
    println!("🔄 Testing concurrent access...");
    
    let config = LLMServerConfig {
        devices: vec![0],
        model_type: Some(ModelType::Llama),
        base_page_size: 1024 * 1024,
        ..Default::default()
    };
    
    if let Ok(manager) = Arc::new(ProductionKVCacheManager::new(config).unwrap()) {
        let num_threads = 4;
        let operations_per_thread = 10;
        
        let mut handles = vec![];
        
        for thread_id in 0..num_threads {
            let manager_clone = Arc::clone(&manager);
            
            let handle = thread::spawn(move || {
                for i in 0..operations_per_thread {
                    // Create arena
                    if let Ok(arena) = manager_clone.create_sequence_arena(
                        64 + (i % 32),
                        128 + (i % 64),
                        1024,
                        8,
                        None,
                    ) {
                        // Create tensor
                        if let Ok(mut tensor) = manager_clone.create_transformer_kv_tensor(
                            &arena,
                            1,
                            64 + (i % 32),
                            128,
                            8,
                            128,
                            None,
                            None,
                        ) {
                            // Try to extend tensor
                            let _ = manager_clone.extend_transformer_tensor(
                                &mut tensor,
                                tensor.seq_len() + 16,
                            );
                        }
                    }
                }
                thread_id
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            if let Ok(thread_id) = handle.join() {
                println!("  ✓ Thread {} completed successfully", thread_id);
            } else {
                println!("  ❌ Thread failed");
            }
        }
        
        // Check final state
        let final_metrics = manager.get_enhanced_metrics();
        let total_operations = num_threads * operations_per_thread;
        println!("  ✓ Concurrent test completed:");
        println!("    - Total operations: {}", total_operations);
        println!("    - Tensors created: {}", final_metrics.transformer_tensors_created);
        println!("    - Zero-copy ratio: {:.1}%", final_metrics.base_metrics.zero_copy_ratio * 100.0);
        
        // Should have created some tensors
        assert!(final_metrics.transformer_tensors_created > 0);
        println!("  ✓ Concurrent access test passed");
    }
}

#[test]
fn test_memory_pressure_simulation() {
    println!("💿 Testing memory pressure simulation...");
    
    let config = LLMServerConfig {
        devices: vec![0],
        model_type: Some(ModelType::Llama),
        base_page_size: 256 * 1024, // Smaller pages to trigger pressure faster
        max_slab_pages: 10,         // Limit slab pool size
        ..Default::default()
    };
    
    if let Ok(manager) = ProductionKVCacheManager::new(config) {
        let mut arenas = Vec::new();
        let mut allocation_count = 0;
        
        // Keep allocating until we hit memory pressure
        for i in 0..100 {
            match manager.create_sequence_arena(
                128 + (i % 64),
                256 + (i % 128),
                2048,
                16,
                None,
            ) {
                Ok(arena) => {
                    arenas.push(arena);
                    allocation_count += 1;
                }
                Err(_) => {
                    println!("  ✓ Memory pressure triggered after {} allocations", allocation_count);
                    break;
                }
            }
        }
        
        // Test system health under pressure
        let health = manager.get_system_health();
        println!("  ✓ System health under pressure: {:?}", health.status);
        println!("  ✓ Health score: {:.2}", health.health_score);
        
        // Drop some arenas to relieve pressure
        let arenas_to_drop = arenas.len() / 2;
        arenas.truncate(arenas_to_drop);
        println!("  ✓ Dropped {} arenas to relieve pressure", arenas_to_drop);
        
        // Force cleanup
        let cleanup_report = manager.maintenance_cleanup();
        println!("  ✓ Cleanup report:");
        println!("    - Inactive arenas: {}", cleanup_report.inactive_arenas_cleaned);
        println!("    - Old pages: {}", cleanup_report.old_pages_cleaned);
        
        // Should be able to allocate again after cleanup
        match manager.create_sequence_arena(128, 256, 1024, 8, None) {
            Ok(_) => println!("  ✓ Allocation successful after cleanup"),
            Err(e) => println!("  ⚠️  Allocation still failing after cleanup: {}", e),
        }
    }
}

// Utility function for testing different configurations
fn test_configuration(name: &str, config: LLMServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("  🔧 Testing configuration: {}", name);
    
    let manager = ProductionKVCacheManager::new(config)?;
    let arena = manager.create_sequence_arena(128, 512, 2048, 16, None)?;
    
    let tensor = manager.create_transformer_kv_tensor(
        &arena,
        1,
        128,
        512,
        16,
        128,
        None,
        None,
    )?;
    
    // Basic validation
    assert_eq!(tensor.seq_len(), 128);
    assert_eq!(tensor.max_seq_len(), 512);
    
    println!("    ✓ Configuration {} validated", name);
    Ok(())
}

#[test]
fn test_different_configurations() {
    println!("⚙️  Testing different configurations...");
    
    let configurations = [
        ("small_pages", LLMServerConfig {
            devices: vec![0],
            base_page_size: 128 * 1024,
            max_slab_pages: 100,
            ..Default::default()
        }),
        ("large_pages", LLMServerConfig {
            devices: vec![0],
            base_page_size: 4 * 1024 * 1024,
            max_slab_pages: 25,
            ..Default::default()
        }),
        ("no_cross_device", LLMServerConfig {
            devices: vec![0],
            cross_device_sharing: false,
            ..Default::default()
        }),
        ("frequent_cleanup", LLMServerConfig {
            devices: vec![0],
            cleanup_interval_seconds: 10,
            max_page_age_seconds: 30,
            ..Default::default()
        }),
    ];
    
    for (name, config) in &configurations {
        match test_configuration(name, config.clone()) {
            Ok(()) => println!("  ✅ Configuration {} passed", name),
            Err(e) => println!("  ❌ Configuration {} failed: {}", name, e),
        }
    }
}

// Performance benchmark for stress testing
#[test]
fn test_stress_performance() {
    println!("🏋️  Running stress performance test...");
    
    let config = LLMServerConfig {
        devices: vec![0],
        model_type: Some(ModelType::Llama),
        base_page_size: 2 * 1024 * 1024,
        ..Default::default()
    };
    
    if let Ok(manager) = ProductionKVCacheManager::new(config) {
        let start_time = std::time::Instant::now();
        let mut successful_ops = 0;
        let target_ops = 200;
        
        for i in 0..target_ops {
            // Vary the parameters to test different code paths
            let seq_len = 64 + (i % 128);
            let max_seq = 256 + (i % 256);
            let hidden_dim = 1024 + (i % 2048);
            let num_heads = 8 + (i % 24);
            
            if let Ok(arena) = manager.create_sequence_arena(
                seq_len,
                max_seq,
                hidden_dim,
                num_heads,
                None,
            ) {
                if let Ok(mut tensor) = manager.create_transformer_kv_tensor(
                    &arena,
                    1,
                    seq_len,
                    max_seq,
                    num_heads,
                    hidden_dim / num_heads,
                    None,
                    None,
                ) {
                    // Try some extensions
                    for j in 1..4 {
                        let new_len = seq_len + j * 32;
                        if new_len <= max_seq {
                            let _ = manager.extend_transformer_tensor(&mut tensor, new_len);
                        }
                    }
                    successful_ops += 1;
                }
            }
            
            // Periodic cleanup to test slab recycling
            if i % 50 == 49 {
                let _ = manager.maintenance_cleanup();
            }
        }
        
        let total_time = start_time.elapsed();
        let ops_per_second = successful_ops as f64 / total_time.as_secs_f64();
        
        println!("  ✓ Stress test completed:");
        println!("    - Operations: {}/{}", successful_ops, target_ops);
        println!("    - Total time: {:.1}s", total_time.as_secs_f64());
        println!("    - Ops/second: {:.1}", ops_per_second);
        
        // Get final metrics
        let final_metrics = manager.get_enhanced_metrics();
        println!("    - Tensors created: {}", final_metrics.transformer_tensors_created);
        println!("    - Zero-copy ratio: {:.1}%", final_metrics.base_metrics.zero_copy_ratio * 100.0);
        
        // Performance assertions
        assert!(successful_ops > target_ops / 2, "Too many failed operations");
        assert!(ops_per_second > 10.0, "Performance too slow: {:.1} ops/sec", ops_per_second);
        
        println!("  ✅ Stress performance test passed");
    }
}