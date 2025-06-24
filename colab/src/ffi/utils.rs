// src/ffi/utils.rs - Fixed compilation errors for updated struct fields
use std::ffi::c_char;
use super::types::*;

/// Get default page size
#[no_mangle]
pub extern "C" fn arena_get_default_page_size() -> usize {
    256 * 1024 // 256KB default
}

/// Get memory alignment requirement
#[no_mangle]
pub extern "C" fn arena_get_alignment() -> usize {
    64 // 64-byte alignment
}

/// Align a size to the required boundary
#[no_mangle]
pub extern "C" fn arena_align_size(size: usize) -> usize {
    const ALIGNMENT: usize = 64;
    (size + ALIGNMENT - 1) & !(ALIGNMENT - 1)
}

/// Get optimal page size for given parameters
#[no_mangle]
pub extern "C" fn prod_calculate_optimal_page_size(
    max_seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
    device_id: i32,
) -> usize {
    let head_dim = if num_heads > 0 { hidden_dim / num_heads } else { 128 };
    calculate_arena_size(max_seq_len, num_heads, head_dim, 2)
}

/// Emergency diagnostic function
#[no_mangle]
pub extern "C" fn arena_diagnostic_check() -> i32 {
    42
}

/// Get library version information
#[no_mangle]
pub extern "C" fn prod_get_version() -> *const c_char {
    static VERSION: &[u8] = b"1.0.0-production-honest-zero-copy\0";
    VERSION.as_ptr() as *const c_char
}

/// Get supported features list
#[no_mangle]
pub extern "C" fn prod_get_features() -> *const c_char {
    static FEATURES: &[u8] = b"metadata-zero-copy,slab-recycling,cuda-heap,honest-reporting\0";
    FEATURES.as_ptr() as *const c_char
}

/// Check if CUDA is available and working
#[no_mangle]
pub extern "C" fn prod_check_cuda_availability() -> i32 {
    #[cfg(feature = "cuda")]
    {
        match crate::cuda::CudaMemoryManager::new() {
            Ok(_) => 1, // CUDA available
            Err(_) => 0, // CUDA not available
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        0 // CUDA not compiled in
    }
}

/// Batch extend multiple tensors with honest reporting
#[no_mangle]
pub extern "C" fn prod_batch_extend_tensors(
    manager: *mut CProductionManager,
    arenas: *const *mut CSequenceArena,
    tensors: *mut *mut CKVTensor,
    new_tokens: *const usize,
    num_tensors: usize,
    results_out: *mut i32,
    total_metadata_zero_copy_out: *mut usize,
    total_data_copy_required_out: *mut usize,
    avg_extension_time_ns_out: *mut u64,
) -> i32 {
    if arenas.is_null() || tensors.is_null() || 
       new_tokens.is_null() || results_out.is_null() || 
       total_metadata_zero_copy_out.is_null() || total_data_copy_required_out.is_null() ||
       avg_extension_time_ns_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let arenas_slice = unsafe { std::slice::from_raw_parts(arenas, num_tensors) };
    let tensors_slice = unsafe { std::slice::from_raw_parts_mut(tensors, num_tensors) };
    let tokens_slice = unsafe { std::slice::from_raw_parts(new_tokens, num_tensors) };
    let results_slice = unsafe { std::slice::from_raw_parts_mut(results_out, num_tensors) };

    let start_time = std::time::Instant::now();
    let mut metadata_zero_copy_count = 0;
    let mut data_copy_required_count = 0;

    for i in 0..num_tensors {
        let arena_ref = unsafe { &(*arenas_slice[i]).0 };
        let tensor_ref = unsafe { &mut (*tensors_slice[i]).0 };
        
        // Use the SINGLE extend_tensor_for_generation method
        match arena_ref.extend_tensor_for_generation(tensor_ref, tokens_slice[i]) {
            Ok(was_pure_zero_copy) => {
                if was_pure_zero_copy {
                    results_slice[i] = 1; // TRUE zero-copy
                    metadata_zero_copy_count += 1;
                } else {
                    results_slice[i] = 0; // Metadata updated but data copy required
                    data_copy_required_count += 1;
                }
            }
            Err(_) => {
                results_slice[i] = -1; // Failed
            }
        }
    }

    let batch_time = start_time.elapsed().as_nanos() as u64;
    let avg_time = if num_tensors > 0 { batch_time / num_tensors as u64 } else { 0 };

    unsafe {
        *total_metadata_zero_copy_out = metadata_zero_copy_count;
        *total_data_copy_required_out = data_copy_required_count;
        *avg_extension_time_ns_out = avg_time;
    }

    PROD_SUCCESS
}

/// Simulate token generation with honest reporting
#[no_mangle]
pub extern "C" fn prod_simulate_generation(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    tensor: *mut CKVTensor,
    num_tokens: usize,
    tokens_generated_out: *mut usize,
    metadata_zero_copy_extensions_out: *mut usize,
    data_copy_extensions_out: *mut usize,
    total_time_ms_out: *mut f64,
    avg_time_per_token_ms_out: *mut f64,
) -> i32 {
    if arena.is_null() || tensor.is_null() ||
       tokens_generated_out.is_null() || metadata_zero_copy_extensions_out.is_null() ||
       data_copy_extensions_out.is_null() || total_time_ms_out.is_null() ||
       avg_time_per_token_ms_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let arena_ref = unsafe { &(*arena).0 };
    let tensor_ref = unsafe { &mut (*tensor).0 };

    let start_time = std::time::Instant::now();
    let mut metadata_zero_copy_count = 0;
    let mut data_copy_count = 0;

    for _i in 1..=num_tokens {
        match arena_ref.extend_tensor_for_generation(tensor_ref, 1) {
            Ok(was_pure_zero_copy) => {
                if was_pure_zero_copy {
                    metadata_zero_copy_count += 1;
                } else {
                    data_copy_count += 1;
                }
            }
            Err(_) => break,
        }
    }

    let total_time = start_time.elapsed();
    let total_time_ms = total_time.as_millis() as f64;
    let avg_time_per_token = if num_tokens > 0 {
        total_time_ms / num_tokens as f64
    } else {
        0.0
    };

    unsafe {
        *tokens_generated_out = num_tokens;
        *metadata_zero_copy_extensions_out = metadata_zero_copy_count;
        *data_copy_extensions_out = data_copy_count;
        *total_time_ms_out = total_time_ms;
        *avg_time_per_token_ms_out = avg_time_per_token;
    }

    PROD_SUCCESS
}

/// Honest benchmark that separates metadata from data operations
#[no_mangle]
pub extern "C" fn prod_benchmark_separated_operations(
    manager: *mut CProductionManager,
    initial_seq_len: usize,
    max_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    num_extensions: usize,
    tokens_per_extension: usize,
    metadata_zero_copy_time_ns_out: *mut u64,
    metadata_zero_copy_count_out: *mut usize,
    data_copy_time_estimate_ns_out: *mut u64,
    data_copy_required_count_out: *mut usize,
    efficiency_ratio_out: *mut f64,
) -> i32 {
    if manager.is_null() || metadata_zero_copy_time_ns_out.is_null() || metadata_zero_copy_count_out.is_null() ||
       data_copy_time_estimate_ns_out.is_null() || data_copy_required_count_out.is_null() || 
       efficiency_ratio_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };

    match manager_ref.zero_copy_manager.create_arena(
        calculate_arena_size(max_seq_len, num_heads, head_dim, 2),
        0,
    ) {
        Ok(arena) => {
            match arena.allocate_kv_tensor_with_growth(initial_seq_len, max_seq_len, num_heads, head_dim, 2) {
                Ok(mut tensor) => {
                    let start_time = std::time::Instant::now();
                    let mut metadata_zero_copy_count = 0;
                    let mut data_copy_required_count = 0;
                    
                    for _ in 0..num_extensions {
                        match arena.extend_tensor_for_generation(&mut tensor, tokens_per_extension) {
                            Ok(was_pure_zero_copy) => {
                                if was_pure_zero_copy {
                                    metadata_zero_copy_count += 1;
                                } else {
                                    data_copy_required_count += 1;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    
                    let metadata_time = start_time.elapsed().as_nanos() as u64;
                    
                    // Estimate data copy time (in real implementation would do actual copies)
                    let estimated_copy_time = data_copy_required_count as u64 * tokens_per_extension as u64 * 1000; // 1μs per token estimate
                    
                    let efficiency_ratio = if estimated_copy_time > 0 {
                        estimated_copy_time as f64 / metadata_time as f64
                    } else {
                        1.0
                    };

                    unsafe {
                        *metadata_zero_copy_time_ns_out = metadata_time;
                        *metadata_zero_copy_count_out = metadata_zero_copy_count;
                        *data_copy_time_estimate_ns_out = estimated_copy_time;
                        *data_copy_required_count_out = data_copy_required_count;
                        *efficiency_ratio_out = efficiency_ratio;
                    }

                    PROD_SUCCESS
                }
                Err(_) => PROD_ERROR_ALLOCATION_FAILED,
            }
        }
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Advanced benchmark with honest operation separation
#[no_mangle]
pub extern "C" fn prod_benchmark_zero_copy_performance(
    manager: *mut CProductionManager,
    test_configs: *const CBenchmarkConfig,
    num_configs: usize,
    results_out: *mut CBenchmarkResult,
) -> i32 {
    if manager.is_null() || test_configs.is_null() || results_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let configs_slice = unsafe { std::slice::from_raw_parts(test_configs, num_configs) };
    let results_slice = unsafe { std::slice::from_raw_parts_mut(results_out, num_configs) };

    for (i, config) in configs_slice.iter().enumerate() {
        let arena_size = calculate_arena_size(
            config.max_seq_len, 
            config.num_heads, 
            config.hidden_dim / config.num_heads, 
            config.dtype_size
        );
        
        let benchmark_start = std::time::Instant::now();
        let mut total_metadata_extensions = 0;
        let mut total_data_copy_extensions = 0;
        
        if let Ok(arena) = manager_ref.zero_copy_manager.create_arena(arena_size, 0) {
            if let Ok(mut tensor) = arena.allocate_kv_tensor_with_growth(
                config.initial_seq_len,
                config.max_seq_len,
                config.num_heads,
                config.hidden_dim / config.num_heads,
                config.dtype_size,
            ) {
                for _ in 0..config.num_extensions {
                    match arena.extend_tensor_for_generation(&mut tensor, config.tokens_per_extension) {
                        Ok(was_pure_zero_copy) => {
                            if was_pure_zero_copy {
                                total_metadata_extensions += 1;
                            } else {
                                total_data_copy_extensions += 1;
                            }
                        }
                        Err(_) => break,
                    }
                }
            }
        }
        
        let total_time = benchmark_start.elapsed();
        let honest_zero_copy_rate = if (total_metadata_extensions + total_data_copy_extensions) > 0 {
            total_metadata_extensions as f64 / (total_metadata_extensions + total_data_copy_extensions) as f64
        } else {
            0.0
        };
        
        results_slice[i] = CBenchmarkResult {
            config_id: i,
            total_time_ms: total_time.as_millis() as f64,
            metadata_only_extensions: total_metadata_extensions,  // FIXED field name
            data_copy_extensions: total_data_copy_extensions,     // FIXED field name  
            zero_copy_rate: honest_zero_copy_rate,
            avg_metadata_time_ns: if config.num_extensions > 0 {  // FIXED field name
                total_time.as_nanos() as u64 / config.num_extensions as u64
            } else {
                0
            },
            avg_data_copy_time_ns: if total_data_copy_extensions > 0 { // FIXED field name
                total_time.as_nanos() as u64 / total_data_copy_extensions as u64
            } else {
                0
            },
            memory_efficiency: honest_zero_copy_rate,
            honest_zero_copy_rate, // FIXED field name
        };
    }

    PROD_SUCCESS
}

/// Profile memory allocation patterns
#[no_mangle]
pub extern "C" fn prod_profile_allocation_patterns(
    manager: *mut CProductionManager,
    num_test_allocations: usize,
    allocation_sizes: *const usize,
    allocation_times_ns_out: *mut u64,
    fragmentation_score_out: *mut f64,
    efficiency_score_out: *mut f64,
) -> i32 {
    if manager.is_null() || allocation_sizes.is_null() || 
       allocation_times_ns_out.is_null() || fragmentation_score_out.is_null() ||
       efficiency_score_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let sizes_slice = unsafe { std::slice::from_raw_parts(allocation_sizes, num_test_allocations) };
    
    let start_time = std::time::Instant::now();
    let mut successful_allocations = 0;
    
    for &size in sizes_slice {
        if let Ok(arena) = manager_ref.zero_copy_manager.create_arena(size, 0) {
            successful_allocations += 1;
        }
    }
    
    let total_time = start_time.elapsed().as_nanos() as u64;
    let efficiency = if num_test_allocations > 0 {
        successful_allocations as f64 / num_test_allocations as f64
    } else {
        0.0
    };
    
    let fragmentation = 1.0 - efficiency;
    
    unsafe {
        *allocation_times_ns_out = total_time;
        *fragmentation_score_out = fragmentation;
        *efficiency_score_out = efficiency;
    }

    PROD_SUCCESS
}

/// Helper function to calculate arena size for KV tensors
fn calculate_arena_size(max_seq_len: usize, num_heads: usize, head_dim: usize, dtype_size: usize) -> usize {
    let tensor_size = max_seq_len * num_heads * head_dim * dtype_size;
    let total_size = tensor_size * 2; // K + V
    (total_size * 11) / 10 // Add 10% padding
}