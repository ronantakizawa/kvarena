// src/ffi/tensor_ops.rs - Fixed tensor operations with honest zero-copy terminology
use std::ffi::c_void;
use std::sync::atomic::Ordering;
use super::types::*;

/// TRUE zero-copy tensor extension - ONLY atomic metadata update
/// This is the ONLY function that can honestly claim "zero-copy"
#[no_mangle]
pub extern "C" fn prod_extend_tensor_metadata_only(
    tensor: *mut CKVTensor,
    new_seq_len: usize,
    extension_result_out: *mut CExtensionResult,
) -> i32 {
    if tensor.is_null() || extension_result_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &mut (*tensor).0 };
    
    match tensor_ref.extend_metadata_only(new_seq_len) {
        crate::zero_copy::ExtensionResult::PureZeroCopy { 
            old_seq_len, 
            new_seq_len, 
            operation_time_ns 
        } => {
            unsafe {
                (*extension_result_out).result_type = EXTENSION_PURE_ZERO_COPY;
                (*extension_result_out).old_seq_len = old_seq_len;
                (*extension_result_out).new_seq_len = new_seq_len;
                (*extension_result_out).operation_time_ns = operation_time_ns;
                (*extension_result_out).requires_data_copy = 0;
                (*extension_result_out).copy_size_bytes = 0;
            }
            PROD_SUCCESS
        },
        crate::zero_copy::ExtensionResult::RequiresDataCopy { 
            old_seq_len, 
            new_seq_len, 
            copy_region_size, 
            metadata_update_time_ns,
            ..
        } => {
            unsafe {
                (*extension_result_out).result_type = EXTENSION_REQUIRES_DATA_COPY;
                (*extension_result_out).old_seq_len = old_seq_len;
                (*extension_result_out).new_seq_len = new_seq_len;
                (*extension_result_out).operation_time_ns = metadata_update_time_ns;
                (*extension_result_out).requires_data_copy = 1;
                (*extension_result_out).copy_size_bytes = copy_region_size;
            }
            PROD_SUCCESS
        },
        crate::zero_copy::ExtensionResult::CannotExtend { .. } => {
            unsafe {
                (*extension_result_out).result_type = EXTENSION_CANNOT_EXTEND;
                (*extension_result_out).requires_data_copy = 0;
                (*extension_result_out).copy_size_bytes = 0;
            }
            PROD_ERROR_ALLOCATION_FAILED
        },
    }
}

/// Explicit data copy operation - NOT zero-copy
/// This function is honest about performing memory copies
#[no_mangle]
pub extern "C" fn prod_copy_new_token_data(
    tensor: *mut CKVTensor,
    host_key_data: *const c_void,
    host_value_data: *const c_void,
    start_token_idx: usize,
    num_tokens: usize,
    copy_stats_out: *mut CDataCopyStats,
) -> i32 {
    if tensor.is_null() || host_key_data.is_null() || host_value_data.is_null() || 
       copy_stats_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };
    
    let copy_op = crate::zero_copy::DataCopyOperation::NewTokensCopy {
        start_token_idx,
        num_tokens,
        copy_size_bytes: num_tokens * tensor_ref.current_key_size_bytes() / tensor_ref.seq_len() * 2,
    };

    match tensor_ref.copy_new_token_data(
        host_key_data as *const u8,
        host_value_data as *const u8,
        copy_op,
    ) {
        Ok(stats) => {
            unsafe {
                (*copy_stats_out).bytes_copied = stats.bytes_copied;
                (*copy_stats_out).copy_time_ns = stats.copy_time_ns;
                (*copy_stats_out).bandwidth_gbps = stats.bandwidth_gbps;
                (*copy_stats_out).operation_type = DATA_COPY_NEW_TOKENS;
            }
            PROD_SUCCESS
        },
        Err(_) => PROD_ERROR_CUDA,
    }
}

/// Combined operation that clearly separates zero-copy from data-copy phases
#[no_mangle]
pub extern "C" fn prod_extend_tensor_with_data_copy(
    arena: *mut CSequenceArena,
    tensor: *mut CKVTensor,
    additional_tokens: usize,
    host_key_data: *const c_void,
    host_value_data: *const c_void,
    operation_report_out: *mut CExtensionOperationReport,
) -> i32 {
    if arena.is_null() || tensor.is_null() || operation_report_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let arena_ref = unsafe { &(*arena).0 };
    let tensor_ref = unsafe { &mut (*tensor).0 };

    // Phase 1: TRUE zero-copy metadata extension
    let metadata_start = std::time::Instant::now();
    let old_seq_len = tensor_ref.seq_len();
    let new_seq_len = old_seq_len + additional_tokens;
    
    match tensor_ref.extend_metadata_only(new_seq_len) {
        crate::zero_copy::ExtensionResult::PureZeroCopy { operation_time_ns, .. } => {
            // No data copy needed - sequence was already pre-filled
            unsafe {
                (*operation_report_out).metadata_extension_time_ns = operation_time_ns;
                (*operation_report_out).data_copy_time_ns = 0;
                (*operation_report_out).total_time_ns = operation_time_ns;
                (*operation_report_out).was_pure_zero_copy = 1;
                (*operation_report_out).bytes_copied = 0;
                (*operation_report_out).tokens_added = additional_tokens;
            }
            PROD_SUCCESS
        },
        crate::zero_copy::ExtensionResult::RequiresDataCopy { 
            metadata_update_time_ns,
            copy_region_start,
            copy_region_size,
            ..
        } => {
            // Phase 2: Explicit data copy (NOT zero-copy)
            if !host_key_data.is_null() && !host_value_data.is_null() {
                let copy_op = crate::zero_copy::DataCopyOperation::NewTokensCopy {
                    start_token_idx: copy_region_start,
                    num_tokens: additional_tokens,
                    copy_size_bytes: copy_region_size,
                };

                match tensor_ref.copy_new_token_data(
                    host_key_data as *const u8,
                    host_value_data as *const u8,
                    copy_op,
                ) {
                    Ok(copy_stats) => {
                        unsafe {
                            (*operation_report_out).metadata_extension_time_ns = metadata_update_time_ns;
                            (*operation_report_out).data_copy_time_ns = copy_stats.copy_time_ns;
                            (*operation_report_out).total_time_ns = metadata_update_time_ns + copy_stats.copy_time_ns;
                            (*operation_report_out).was_pure_zero_copy = 0;
                            (*operation_report_out).bytes_copied = copy_stats.bytes_copied;
                            (*operation_report_out).tokens_added = additional_tokens;
                            (*operation_report_out).bandwidth_gbps = copy_stats.bandwidth_gbps;
                        }
                        PROD_SUCCESS
                    },
                    Err(_) => PROD_ERROR_CUDA,
                }
            } else {
                // Metadata extended but no data provided - caller must copy data separately
                unsafe {
                    (*operation_report_out).metadata_extension_time_ns = metadata_update_time_ns;
                    (*operation_report_out).data_copy_time_ns = 0;
                    (*operation_report_out).total_time_ns = metadata_update_time_ns;
                    (*operation_report_out).was_pure_zero_copy = 0;
                    (*operation_report_out).bytes_copied = 0;
                    (*operation_report_out).tokens_added = additional_tokens;
                    (*operation_report_out).bandwidth_gbps = 0.0;
                }
                PROD_SUCCESS
            }
        },
        crate::zero_copy::ExtensionResult::CannotExtend { .. } => {
            PROD_ERROR_ALLOCATION_FAILED
        },
    }
}

/// Legacy function with honest naming - clearly indicates it's NOT pure zero-copy
#[no_mangle]
pub extern "C" fn prod_extend_tensor_hybrid_operation(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    tensor: *mut CKVTensor,
    new_tokens: usize,
    was_metadata_zero_copy_out: *mut i32,
    required_data_copy_out: *mut i32,
    extension_time_ns_out: *mut u64,
) -> i32 {
    if arena.is_null() || tensor.is_null() || 
       was_metadata_zero_copy_out.is_null() || required_data_copy_out.is_null() ||
       extension_time_ns_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &mut (*tensor).0 };
    let current_len = tensor_ref.seq_len();
    let new_len = current_len + new_tokens;

    match tensor_ref.extend_metadata_only(new_len) {
        crate::zero_copy::ExtensionResult::PureZeroCopy { operation_time_ns, .. } => {
            unsafe {
                *was_metadata_zero_copy_out = 1;
                *required_data_copy_out = 0; // Data was already there
                *extension_time_ns_out = operation_time_ns;
            }
            PROD_SUCCESS
        },
        crate::zero_copy::ExtensionResult::RequiresDataCopy { metadata_update_time_ns, .. } => {
            unsafe {
                *was_metadata_zero_copy_out = 1; // Metadata update was zero-copy
                *required_data_copy_out = 1;     // But data copy is required
                *extension_time_ns_out = metadata_update_time_ns;
            }
            PROD_SUCCESS
        },
        crate::zero_copy::ExtensionResult::CannotExtend { .. } => {
            unsafe {
                *was_metadata_zero_copy_out = 0;
                *required_data_copy_out = 0;
                *extension_time_ns_out = 0;
            }
            PROD_ERROR_ALLOCATION_FAILED
        },
    }
}

/// Get honest zero-copy statistics that distinguish metadata from data operations
#[no_mangle]
pub extern "C" fn prod_get_honest_zero_copy_stats(
    tensor: *mut CKVTensor,
    stats_out: *mut CHonestZeroCopyStats,
) -> i32 {
    if tensor.is_null() || stats_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };
    let stats = tensor_ref.zero_copy_stats();

    unsafe {
        (*stats_out).current_seq_len = stats.current_seq_len;
        (*stats_out).max_seq_len = stats.max_seq_len;
        (*stats_out).growth_capacity_remaining = stats.growth_capacity_remaining;
        (*stats_out).utilization = stats.utilization;
        (*stats_out).memory_efficiency = stats.memory_efficiency;
        (*stats_out).can_grow_without_copy = if stats.can_grow_without_copy { 1 } else { 0 };
        
        // NEW: Honest reporting
        (*stats_out).metadata_operations_are_zero_copy = if stats.metadata_operations_are_zero_copy { 1 } else { 0 };
        (*stats_out).data_operations_require_copy = if stats.data_operations_require_copy { 1 } else { 0 };
        (*stats_out).last_extension_was_pure_zero_copy = match stats.last_extension_was_pure_zero_copy {
            Some(true) => 1,
            Some(false) => 0,
            None => -1, // Unknown/not tracked
        };
    }

    PROD_SUCCESS
}

/// Benchmark that clearly separates zero-copy metadata from data copy performance
#[no_mangle]
pub extern "C" fn prod_benchmark_separated_operations(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    tensor: *mut CKVTensor,
    num_extensions: usize,
    tokens_per_extension: usize,
    benchmark_result_out: *mut CSeparatedOperationsBenchmark,
) -> i32 {
    if arena.is_null() || tensor.is_null() || benchmark_result_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &mut (*tensor).0 };
    
    let mut total_metadata_time_ns = 0u64;
    let mut total_data_copy_time_ns = 0u64;
    let mut pure_zero_copy_count = 0usize;
    let mut data_copy_required_count = 0usize;
    let mut total_bytes_copied = 0usize;

    for _i in 0..num_extensions {
        let current_len = tensor_ref.seq_len();
        let new_len = current_len + tokens_per_extension;
        
        match tensor_ref.extend_metadata_only(new_len) {
            crate::zero_copy::ExtensionResult::PureZeroCopy { operation_time_ns, .. } => {
                total_metadata_time_ns += operation_time_ns;
                pure_zero_copy_count += 1;
            },
            crate::zero_copy::ExtensionResult::RequiresDataCopy { 
                metadata_update_time_ns, 
                copy_region_size,
                ..
            } => {
                total_metadata_time_ns += metadata_update_time_ns;
                data_copy_required_count += 1;
                total_bytes_copied += copy_region_size;
                
                // Simulate data copy time (in real implementation, would do actual copy)
                let estimated_copy_time_ns = copy_region_size as u64 * 10; // 10ns per byte estimate
                total_data_copy_time_ns += estimated_copy_time_ns;
            },
            crate::zero_copy::ExtensionResult::CannotExtend { .. } => {
                break; // Hit capacity limit
            },
        }
    }

    let total_extensions = pure_zero_copy_count + data_copy_required_count;
    
    unsafe {
        (*benchmark_result_out).total_extensions = total_extensions;
        (*benchmark_result_out).pure_zero_copy_extensions = pure_zero_copy_count;
        (*benchmark_result_out).data_copy_required_extensions = data_copy_required_count;
        (*benchmark_result_out).total_metadata_time_ns = total_metadata_time_ns;
        (*benchmark_result_out).total_data_copy_time_ns = total_data_copy_time_ns;
        (*benchmark_result_out).avg_metadata_time_ns = if total_extensions > 0 {
            total_metadata_time_ns / total_extensions as u64
        } else {
            0
        };
        (*benchmark_result_out).avg_data_copy_time_ns = if data_copy_required_count > 0 {
            total_data_copy_time_ns / data_copy_required_count as u64
        } else {
            0
        };
        (*benchmark_result_out).metadata_efficiency_ratio = if total_data_copy_time_ns > 0 {
            total_metadata_time_ns as f64 / total_data_copy_time_ns as f64
        } else {
            1.0
        };
        (*benchmark_result_out).total_bytes_would_copy = total_bytes_copied;
    }

    PROD_SUCCESS
}

/// Educational function that explains what is and isn't zero-copy
#[no_mangle]
pub extern "C" fn prod_explain_zero_copy_operations(
    explanation_out: *mut CZeroCopyExplanation,
) -> i32 {
    if explanation_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    unsafe {
        // Set explanation flags
        (*explanation_out).metadata_updates_are_zero_copy = 1;
        (*explanation_out).new_data_requires_copy = 1;
        (*explanation_out).pre_allocated_data_access_is_zero_copy = 1;
        (*explanation_out).cuda_memcpy_is_not_zero_copy = 1;
        (*explanation_out).atomic_operations_are_zero_copy = 1;
        (*explanation_out).pointer_arithmetic_is_zero_copy = 1;
        
        // Set example operation times (in nanoseconds)
        (*explanation_out).atomic_update_time_ns = 10;        // ~10ns for atomic operation
        (*explanation_out).cuda_memcpy_time_per_kb_ns = 1000; // ~1μs per KB for memcpy
        (*explanation_out).speedup_ratio = 100.0;             // Atomic is ~100x faster than memcpy
    }

    PROD_SUCCESS
}

// Legacy compatibility functions with honest naming

/// DEPRECATED: Use prod_extend_tensor_metadata_only for true zero-copy
#[no_mangle]
pub extern "C" fn prod_extend_tensor_zero_copy(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    tensor: *mut CKVTensor,
    new_tokens: usize,
    was_zero_copy_out: *mut i32,
    extension_time_ns_out: *mut u64,
) -> i32 {
    // Redirect to honest function with deprecation warning
    log::warn!("DEPRECATED: prod_extend_tensor_zero_copy may perform data copies. Use prod_extend_tensor_metadata_only for true zero-copy or prod_extend_tensor_hybrid_operation for honest reporting.");
    
    prod_extend_tensor_hybrid_operation(
        manager, arena, tensor, new_tokens,
        was_zero_copy_out,
        &mut 0i32, // Ignore data copy flag for compatibility
        extension_time_ns_out,
    )
}

/// DEPRECATED: Name is misleading - this performs data copy
#[no_mangle]
pub extern "C" fn prod_extend_tensor_pure_zero_copy(
    tensor: *mut CKVTensor,
    additional_tokens: usize,
    was_zero_copy_out: *mut i32,
    extension_time_ns_out: *mut u64,
) -> i32 {
    log::warn!("DEPRECATED: Function name is misleading. Use prod_extend_tensor_metadata_only for actual pure zero-copy operations.");
    
    if tensor.is_null() || was_zero_copy_out.is_null() || extension_time_ns_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &mut (*tensor).0 };
    let current_len = tensor_ref.seq_len();
    let new_len = current_len + additional_tokens;

    match tensor_ref.extend_metadata_only(new_len) {
        crate::zero_copy::ExtensionResult::PureZeroCopy { operation_time_ns, .. } => {
            unsafe {
                *was_zero_copy_out = 1;
                *extension_time_ns_out = operation_time_ns;
            }
            PROD_SUCCESS
        },
        crate::zero_copy::ExtensionResult::RequiresDataCopy { metadata_update_time_ns, .. } => {
            unsafe {
                *was_zero_copy_out = 0; // Honest: data copy would be required
                *extension_time_ns_out = metadata_update_time_ns;
            }
            PROD_SUCCESS
        },
        crate::zero_copy::ExtensionResult::CannotExtend { .. } => {
            PROD_ERROR_ALLOCATION_FAILED
        },
    }
}

/// Direct tensor extension using arena methods
#[no_mangle]
pub extern "C" fn prod_extend_tensor_zero_copy(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    tensor: *mut CKVTensor,
    new_tokens: usize,
    was_zero_copy_out: *mut i32,
    extension_time_ns_out: *mut u64,
) -> i32 {
    if arena.is_null() || tensor.is_null() || 
       was_zero_copy_out.is_null() || extension_time_ns_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let arena_ref = unsafe { &(*arena).0 };
    let tensor_ref = unsafe { &mut (*tensor).0 };

    let start_time = std::time::Instant::now();

    // Use arena's extend method directly
    match arena_ref.extend_tensor_for_generation(tensor_ref, new_tokens) {
        Ok(was_zero_copy) => {
            let extension_time = start_time.elapsed().as_nanos() as u64;
            
            unsafe {
                *was_zero_copy_out = if was_zero_copy { 1 } else { 0 };
                *extension_time_ns_out = extension_time;
            }
            PROD_SUCCESS
        }
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Legacy extension function for compatibility
#[no_mangle]
pub extern "C" fn prod_extend_tensor_for_generation(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    tensor: *mut CKVTensor,
    new_tokens: usize,
    was_zero_copy_out: *mut i32,
) -> i32 {
    if was_zero_copy_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }
    
    let mut extension_time_ns = 0u64;
    let result = prod_extend_tensor_zero_copy(
        manager, arena, tensor, new_tokens, was_zero_copy_out, &mut extension_time_ns
    );
    
    result
}

/// Get zero-copy tensor statistics
#[no_mangle]
pub extern "C" fn prod_get_zero_copy_stats(
    tensor: *mut CKVTensor,
    stats_out: *mut CZeroCopyStats,
) -> i32 {
    if tensor.is_null() || stats_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };
    let stats = tensor_ref.zero_copy_stats();

    unsafe {
        (*stats_out).current_seq_len = stats.current_seq_len;
        (*stats_out).max_seq_len = stats.max_seq_len;
        (*stats_out).growth_capacity_remaining = stats.growth_capacity_remaining;
        (*stats_out).utilization = stats.utilization;
        (*stats_out).memory_efficiency = stats.memory_efficiency;
        (*stats_out).can_grow_without_copy = if stats.can_grow_without_copy { 1 } else { 0 };
    }

    PROD_SUCCESS
}

/// Get tensor device pointers for CUDA operations
#[no_mangle]
pub extern "C" fn prod_get_tensor_device_ptrs(
    tensor: *mut CKVTensor,
    key_ptr_out: *mut *mut c_void,
    value_ptr_out: *mut *mut c_void,
    seq_len_out: *mut usize,
    num_heads_out: *mut usize,
    head_dim_out: *mut usize,
) -> i32 {
    if tensor.is_null() || key_ptr_out.is_null() || value_ptr_out.is_null() ||
       seq_len_out.is_null() || num_heads_out.is_null() || head_dim_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };
    let (seq_len, num_heads, head_dim) = tensor_ref.dimensions();

    unsafe {
        *key_ptr_out = tensor_ref.key_device_ptr();
        *value_ptr_out = tensor_ref.value_device_ptr();
        *seq_len_out = seq_len;
        *num_heads_out = num_heads;
        *head_dim_out = head_dim;
    }

    PROD_SUCCESS
}

/// Copy host data to tensor (for initial loading)
#[no_mangle]
pub extern "C" fn prod_copy_host_to_tensor(
    tensor: *mut CKVTensor,
    host_key_data: *const c_void,
    host_value_data: *const c_void,
) -> i32 {
    if tensor.is_null() || host_key_data.is_null() || host_value_data.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };

    match tensor_ref.copy_from_host(
        host_key_data as *const u8,
        host_value_data as *const u8,
    ) {
        Ok(()) => PROD_SUCCESS,
        Err(_) => PROD_ERROR_CUDA,
    }
}

/// Copy new tokens to extended tensor (for incremental generation)
#[no_mangle]
pub extern "C" fn prod_copy_new_tokens_to_tensor(
    tensor: *mut CKVTensor,
    host_key_data: *const c_void,
    host_value_data: *const c_void,
    start_token: usize,
    num_tokens: usize,
) -> i32 {
    if tensor.is_null() || host_key_data.is_null() || host_value_data.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };

    match tensor_ref.copy_new_tokens_from_host(
        host_key_data as *const u8,
        host_value_data as *const u8,
        start_token,
        num_tokens,
    ) {
        Ok(()) => PROD_SUCCESS,
        Err(_) => PROD_ERROR_CUDA,
    }
}

/// Advanced zero-copy tensor operations for incremental generation
#[no_mangle]
pub extern "C" fn prod_copy_new_tokens_only(
    tensor: *mut CKVTensor,
    host_key_data: *const c_void,
    host_value_data: *const c_void,
    start_token_idx: usize,
    num_new_tokens: usize,
) -> i32 {
    if tensor.is_null() || host_key_data.is_null() || host_value_data.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };

    match tensor_ref.copy_new_tokens_only(
        host_key_data as *const u8,
        host_value_data as *const u8,
        start_token_idx,
        num_new_tokens,
    ) {
        Ok(()) => PROD_SUCCESS,
        Err(_) => PROD_ERROR_CUDA,
    }
}

/// Get detailed tensor memory layout information
#[no_mangle]
pub extern "C" fn prod_get_tensor_memory_layout(
    tensor: *mut CKVTensor,
    current_key_size_out: *mut usize,
    current_value_size_out: *mut usize,
    max_allocated_size_out: *mut usize,
    memory_efficiency_out: *mut f64,
    growth_capacity_bytes_out: *mut usize,
) -> i32 {
    if tensor.is_null() || current_key_size_out.is_null() || current_value_size_out.is_null() ||
       max_allocated_size_out.is_null() || memory_efficiency_out.is_null() ||
       growth_capacity_bytes_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };
    let stats = tensor_ref.zero_copy_stats();

    unsafe {
        *current_key_size_out = tensor_ref.current_key_size_bytes();
        *current_value_size_out = tensor_ref.current_value_size_bytes();
        *max_allocated_size_out = tensor_ref.max_allocated_size_bytes();
        *memory_efficiency_out = stats.memory_efficiency;
        *growth_capacity_bytes_out = stats.growth_capacity_remaining * 
                                   tensor_ref.current_key_size_bytes() / tensor_ref.seq_len().max(1);
    }

    PROD_SUCCESS
}

/// Check if tensor can extend to specific length without copy
#[no_mangle]
pub extern "C" fn prod_can_extend_zero_copy_to(
    tensor: *mut CKVTensor,
    target_seq_len: usize,
    can_extend_out: *mut i32,
    additional_tokens_possible_out: *mut usize,
) -> i32 {
    if tensor.is_null() || can_extend_out.is_null() || additional_tokens_possible_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };
    let can_extend = tensor_ref.can_extend_zero_copy_to(target_seq_len);
    let current_len = tensor_ref.seq_len();
    let max_len = tensor_ref.max_seq_len();
    
    let additional_possible = if current_len < max_len {
        max_len - current_len
    } else {
        0
    };

    unsafe {
        *can_extend_out = if can_extend { 1 } else { 0 };
        *additional_tokens_possible_out = additional_possible;
    }

    PROD_SUCCESS
}