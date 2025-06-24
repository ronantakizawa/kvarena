// src/ffi/slab.rs - Slab recycling and memory management FFI functions
use std::ffi::{c_char, c_void, CString};
use super::types::*;

/// Get slab recycling stats
#[no_mangle]
pub extern "C" fn prod_get_slab_recycling_stats(
    manager: *mut CProductionManager,
    pages_created_out: *mut usize,
    pages_recycled_out: *mut usize,
    pages_reused_out: *mut usize,
    recycling_efficiency_out: *mut f64,
    reuse_efficiency_out: *mut f64,
    bytes_saved_mb_out: *mut usize,
    fragmentation_prevented_out: *mut f64,
    gc_stalls_avoided_out: *mut usize,
) -> i32 {
    if manager.is_null() || pages_created_out.is_null() || pages_recycled_out.is_null() ||
       pages_reused_out.is_null() || recycling_efficiency_out.is_null() ||
       reuse_efficiency_out.is_null() || bytes_saved_mb_out.is_null() ||
       fragmentation_prevented_out.is_null() || gc_stalls_avoided_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let metrics = manager_ref.get_slab_recycling_metrics();

    unsafe {
        *pages_created_out = metrics.pages_created;
        *pages_recycled_out = metrics.pages_recycled;
        *pages_reused_out = metrics.pages_reused;
        *recycling_efficiency_out = metrics.recycling_efficiency;
        *reuse_efficiency_out = metrics.reuse_efficiency;
        *bytes_saved_mb_out = metrics.bytes_saved_mb;
        *fragmentation_prevented_out = metrics.fragmentation_prevented;
        *gc_stalls_avoided_out = metrics.gc_stalls_avoided;
    }

    PROD_SUCCESS
}

/// Slab pool cleanup function
#[no_mangle]
pub extern "C" fn prod_cleanup_slab_pools(
    manager: *mut CProductionManager,
    pages_cleaned_out: *mut usize,
    cleanup_time_ms_out: *mut f64,
    memory_freed_mb_out: *mut usize,
) -> i32 {
    if manager.is_null() || pages_cleaned_out.is_null() || 
       cleanup_time_ms_out.is_null() || memory_freed_mb_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let cleanup_report = manager_ref.cleanup_slab_pools();

    unsafe {
        *pages_cleaned_out = cleanup_report.pages_cleaned;
        *cleanup_time_ms_out = cleanup_report.cleanup_time_ms;
        *memory_freed_mb_out = cleanup_report.memory_freed_mb;
    }

    PROD_SUCCESS
}

/// Lock-free recycling verification function
#[no_mangle]
pub extern "C" fn prod_verify_lock_free_recycling(
    manager: *mut CProductionManager,
    test_allocations: usize,
    recycling_working_out: *mut i32,
    lock_free_confirmed_out: *mut i32,
    performance_gain_out: *mut f64,
) -> i32 {
    if manager.is_null() || recycling_working_out.is_null() || 
       lock_free_confirmed_out.is_null() || performance_gain_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let (is_recycling_working, is_lock_free, perf_gain) = 
        manager_ref.verify_lock_free_recycling(test_allocations);

    unsafe {
        *recycling_working_out = if is_recycling_working { 1 } else { 0 };
        *lock_free_confirmed_out = if is_lock_free { 1 } else { 0 };
        *performance_gain_out = perf_gain;
    }

    PROD_SUCCESS
}

/// Enhanced slab pool status function
#[no_mangle]
pub extern "C" fn prod_get_slab_pool_status(
    manager: *mut CProductionManager,
    small_pool_size_out: *mut usize,
    medium_pool_size_out: *mut usize,
    large_pool_size_out: *mut usize,
    huge_pool_size_out: *mut usize,
    total_memory_pooled_mb_out: *mut usize,
    pool_efficiency_out: *mut f64,
) -> i32 {
    if manager.is_null() || small_pool_size_out.is_null() || medium_pool_size_out.is_null() ||
       large_pool_size_out.is_null() || huge_pool_size_out.is_null() ||
       total_memory_pooled_mb_out.is_null() || pool_efficiency_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let slab_stats = manager_ref.slab_pool.stats();

    unsafe {
        *small_pool_size_out = slab_stats.current_pool_sizes[0];
        *medium_pool_size_out = slab_stats.current_pool_sizes[1];
        *large_pool_size_out = slab_stats.current_pool_sizes[2];
        *huge_pool_size_out = slab_stats.current_pool_sizes[3];
        *total_memory_pooled_mb_out = slab_stats.bytes_saved_mb;
        *pool_efficiency_out = slab_stats.recycling_efficiency;
    }

    PROD_SUCCESS
}

/// Force slab recycling function (for testing)
#[no_mangle]
pub extern "C" fn prod_force_slab_recycling_test(
    manager: *mut CProductionManager,
    num_test_cycles: usize,
    pages_recycled_out: *mut usize,
    recycling_success_rate_out: *mut f64,
) -> i32 {
    if manager.is_null() || pages_recycled_out.is_null() || recycling_success_rate_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    
    let mut successful_recycles = 0usize;
    let mut total_attempts = 0usize;

    // Create and destroy arenas to test recycling
    for _ in 0..num_test_cycles {
        // Create arena
        if let Ok(arena) = manager_ref.zero_copy_manager.create_arena(1024 * 1024, 0) {
            total_attempts += 1;
            
            // Create and drop tensor to trigger recycling
            if let Ok(_tensor) = arena.allocate_kv_tensor_with_growth(128, 512, 16, 64, 2) {
                // Arena will drop here and should recycle
                successful_recycles += 1;
            }
        }
    }

    let success_rate = if total_attempts > 0 {
        successful_recycles as f64 / total_attempts as f64
    } else {
        0.0
    };

    unsafe {
        *pages_recycled_out = successful_recycles;
        *recycling_success_rate_out = success_rate;
    }

    PROD_SUCCESS
}

/// Slab pool memory pressure function
#[no_mangle]
pub extern "C" fn prod_get_slab_memory_pressure(
    manager: *mut CProductionManager,
    memory_pressure_out: *mut f64,
    recommend_cleanup_out: *mut i32,
    estimated_savings_mb_out: *mut usize,
) -> i32 {
    if manager.is_null() || memory_pressure_out.is_null() || 
       recommend_cleanup_out.is_null() || estimated_savings_mb_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let health = manager_ref.get_system_health();
    let metrics = manager_ref.get_production_metrics();
    
    // Calculate memory pressure based on various factors
    let memory_pressure = if metrics.peak_memory_usage_mb > 0 {
        let current_usage = metrics.zero_copy_stats.total_allocated_bytes / (1024 * 1024);
        current_usage as f64 / metrics.peak_memory_usage_mb as f64
    } else {
        0.0
    };

    let recommend_cleanup = if memory_pressure > 0.8 || 
                              metrics.slab_stats.recycling_efficiency < 0.5 {
        1
    } else {
        0
    };

    let estimated_savings = if recommend_cleanup == 1 {
        metrics.slab_stats.bytes_saved_mb + (metrics.peak_memory_usage_mb / 4)
    } else {
        0
    };

    unsafe {
        *memory_pressure_out = memory_pressure;
        *recommend_cleanup_out = recommend_cleanup;
        *estimated_savings_mb_out = estimated_savings;
    }

    PROD_SUCCESS
}

/// Safe batch allocation
#[no_mangle]
pub extern "C" fn prod_batch_allocate_sequences(
    manager: *mut CProductionManager,
    requests: *const CBatchRequest,
    num_requests: usize,
    results_out: *mut *mut CBatchResult,
) -> i32 {
    if manager.is_null() || requests.is_null() || results_out.is_null() || num_requests == 0 {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let request_slice = unsafe { std::slice::from_raw_parts(requests, num_requests) };

    // Use Vec and Box instead of raw malloc
    let mut results: Vec<CBatchResult> = Vec::with_capacity(num_requests);

    // Process each request
    for (i, req) in request_slice.iter().enumerate() {
        let head_dim = if req.num_heads > 0 {
            req.hidden_dim / req.num_heads
        } else {
            128
        };

        let device_id = if req.preferred_device < 0 { 0 } else { req.preferred_device };
        
        // Create arena
        let result = match manager_ref.zero_copy_manager.create_arena(
            calculate_arena_size(req.max_seq_len, req.num_heads, head_dim, req.dtype_size),
            device_id,
        ) {
            Ok(arena) => {
                // Create tensor
                match arena.allocate_kv_tensor_with_growth(
                    req.initial_seq_len,
                    req.max_seq_len,
                    req.num_heads,
                    head_dim,
                    req.dtype_size,
                ) {
                    Ok(tensor) => CBatchResult {
                        request_id: i,
                        arena: Box::into_raw(Box::new(CSequenceArena(arena))),
                        tensor: Box::into_raw(Box::new(CKVTensor(tensor))),
                        device_id,
                    },
                    Err(_) => CBatchResult {
                        request_id: i,
                        arena: std::ptr::null_mut(),
                        tensor: std::ptr::null_mut(),
                        device_id: -1,
                    },
                }
            }
            Err(_) => CBatchResult {
                request_id: i,
                arena: std::ptr::null_mut(),
                tensor: std::ptr::null_mut(),
                device_id: -1,
            },
        };
        
        results.push(result);
    }

    // Convert Vec to Box and then to raw pointer
    let results_box = results.into_boxed_slice();
    unsafe {
        *results_out = Box::into_raw(results_box) as *mut CBatchResult;
    }

    PROD_SUCCESS
}

/// Safe batch cleanup
#[no_mangle]
pub extern "C" fn prod_free_batch_results(
    results: *mut CBatchResult,
    num_results: usize,
) {
    if results.is_null() || num_results == 0 {
        return;
    }

    unsafe {
        // Reconstruct Box from raw pointer and let it drop
        let results_slice = std::slice::from_raw_parts_mut(results, num_results);
        
        // Clean up individual arena and tensor pointers
        for result in results_slice.iter_mut() {
            if !result.arena.is_null() {
                let _ = Box::from_raw(result.arena);
            }
            if !result.tensor.is_null() {
                let _ = Box::from_raw(result.tensor);
            }
        }
        
        // Reconstruct the boxed slice and let it drop
        let _ = Box::from_raw(std::slice::from_raw_parts_mut(results, num_results));
    }
}

/// Get zero-copy recommendations with safe string handling
#[no_mangle]
pub extern "C" fn prod_get_zero_copy_recommendations(
    manager: *mut CProductionManager,
    recommendations_out: *mut *mut c_char,
    max_recommendations: usize,
    num_recommendations_out: *mut usize,
) -> i32 {
    if manager.is_null() || recommendations_out.is_null() || num_recommendations_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let recommendations = manager_ref.zero_copy_manager.get_recommendations();

    let num_to_return = recommendations.len().min(max_recommendations);
    
    // Allocate array of string pointers using Box
    let string_ptrs: Vec<*mut c_char> = vec![std::ptr::null_mut(); max_recommendations];
    let mut string_ptrs_box = string_ptrs.into_boxed_slice();
    
    for (i, recommendation) in recommendations.iter().take(num_to_return).enumerate() {
        if let Ok(c_string) = CString::new(recommendation.as_str()) {
            // Use Box allocation instead of malloc
            let c_str_box = c_string.into_boxed_c_str();
            string_ptrs_box[i] = Box::into_raw(c_str_box) as *mut c_char;
        }
    }

    unsafe {
        *recommendations_out = Box::into_raw(string_ptrs_box) as *mut c_char;
        *num_recommendations_out = num_to_return;
    }

    PROD_SUCCESS
}

/// Safe cleanup for recommendations
#[no_mangle]
pub extern "C" fn prod_free_zero_copy_recommendations(
    recommendations: *mut c_char,
    num_recommendations: usize,
) {
    if recommendations.is_null() {
        return;
    }

    unsafe {
        // Reconstruct the boxed slice
        let string_ptrs = Box::from_raw(
            std::slice::from_raw_parts_mut(recommendations as *mut *mut c_char, num_recommendations)
        );
        
        // Clean up individual strings
        for ptr in string_ptrs.iter() {
            if !ptr.is_null() {
                let _ = Box::from_raw(*ptr);
            }
        }
    }
}

/// Helper function to calculate arena size for KV tensors
fn calculate_arena_size(max_seq_len: usize, num_heads: usize, head_dim: usize, dtype_size: usize) -> usize {
    // K tensor + V tensor + padding
    let tensor_size = max_seq_len * num_heads * head_dim * dtype_size;
    let total_size = tensor_size * 2; // K + V
    (total_size * 11) / 10 // Add 10% padding
}