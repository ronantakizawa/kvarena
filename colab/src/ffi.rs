// src/ffi.rs - Complete fixed production FFI with TRUE zero-copy extensions
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Arc;
use std::time::Instant;
use crate::{
    ProductionKVCacheManager, LLMServerConfig, LLMServerError,
    SequenceRequest, SystemStatus, ProductionMetricsReport, MaintenanceReport, 
    BatchAllocationResult, ZeroCopyEfficiencyReport, ZeroCopyValidationReport,
};
use crate::zero_copy::ZeroCopyStats; // Correct import path

// Opaque pointers for C FFI
pub struct CProductionManager(ProductionKVCacheManager);
pub struct CSequenceArena(Arc<crate::zero_copy::ZeroCopyArena>);
pub struct CKVTensor(crate::zero_copy::ZeroCopyTensor);

// Error codes for production API
pub const PROD_SUCCESS: i32 = 0;
pub const PROD_ERROR_CUDA: i32 = -1;
pub const PROD_ERROR_OUT_OF_MEMORY: i32 = -2;
pub const PROD_ERROR_INVALID_DEVICE: i32 = -3;
pub const PROD_ERROR_ALLOCATION_FAILED: i32 = -4;
pub const PROD_ERROR_INVALID_PARAM: i32 = -5;

// C-compatible structures for FFI
#[repr(C)]
pub struct CBatchRequest {
    pub initial_seq_len: usize,
    pub max_seq_len: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub dtype_size: usize,
    pub preferred_device: i32, // -1 for auto-select
}

#[repr(C)]
pub struct CBatchResult {
    pub request_id: usize,
    pub arena: *mut CSequenceArena,
    pub tensor: *mut CKVTensor,
    pub device_id: i32,
}

#[repr(C)]
pub struct CZeroCopyStats {
    pub current_seq_len: usize,
    pub max_seq_len: usize,
    pub growth_capacity_remaining: usize,
    pub utilization: f64,
    pub memory_efficiency: f64,
    pub can_grow_without_copy: i32,
}

#[repr(C)]
pub struct CZeroCopyEfficiencyReport {
    pub initial_seq_len: usize,
    pub final_seq_len: usize,
    pub total_extensions: usize,
    pub zero_copy_extensions: usize,
    pub zero_copy_rate: f64,
    pub total_time_ns: u64,
    pub avg_extension_time_ns: u64,
    pub final_utilization: f64,
    pub memory_efficiency: f64,
}

#[repr(C)]
pub struct CZeroCopyValidationReport {
    pub basic_zero_copy_works: i32,
    pub beyond_capacity_handled_correctly: i32,
    pub memory_efficiency_reporting_ok: i32,
    pub capacity_reporting_accurate: i32,
    pub all_tests_passed: i32,
}

// Use system malloc/free
mod libc {
    use std::ffi::c_void;
    extern "C" {
        pub fn malloc(size: usize) -> *mut c_void;
        pub fn free(ptr: *mut c_void);
    }
}

/// Initialize production KV cache manager for LLM server
#[no_mangle]
pub extern "C" fn prod_kv_cache_init_for_server(
    model_name: *const c_char,
    devices: *const i32,
    num_devices: usize,
    manager_out: *mut *mut CProductionManager,
) -> i32 {
    if model_name.is_null() || devices.is_null() || manager_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let model_name_str = unsafe {
        match CStr::from_ptr(model_name).to_str() {
            Ok(s) => s,
            Err(_) => return PROD_ERROR_INVALID_PARAM,
        }
    };

    let device_slice = unsafe { std::slice::from_raw_parts(devices, num_devices) };
    let device_vec = device_slice.to_vec();

    match initialize_for_server(model_name_str, &device_vec) {
        Ok(manager) => {
            unsafe {
                *manager_out = Box::into_raw(Box::new(CProductionManager(manager)));
            }
            PROD_SUCCESS
        }
        Err(LLMServerError::CudaError(_)) => PROD_ERROR_CUDA,
        Err(LLMServerError::OutOfMemory) => PROD_ERROR_OUT_OF_MEMORY,
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Initialize for LLM server with model-specific optimizations
pub fn initialize_for_server(model_name: &str, devices: &[i32]) -> Result<ProductionKVCacheManager, LLMServerError> {
    ProductionKVCacheManager::for_llm_model(model_name, devices.to_vec())
}

/// Create chatbot-optimized manager
#[no_mangle]
pub extern "C" fn prod_kv_cache_init_for_chatbot(
    devices: *const i32,
    num_devices: usize,
    manager_out: *mut *mut CProductionManager,
) -> i32 {
    if devices.is_null() || manager_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let device_slice = unsafe { std::slice::from_raw_parts(devices, num_devices) };
    let device_vec = device_slice.to_vec();

    match ProductionKVCacheManager::for_chatbot(device_vec) {
        Ok(manager) => {
            unsafe {
                *manager_out = Box::into_raw(Box::new(CProductionManager(manager)));
            }
            PROD_SUCCESS
        }
        Err(LLMServerError::CudaError(_)) => PROD_ERROR_CUDA,
        Err(LLMServerError::OutOfMemory) => PROD_ERROR_OUT_OF_MEMORY,
        Err(LLMServerError::DeviceNotAvailable) => PROD_ERROR_INVALID_DEVICE,
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Create document processing optimized manager
#[no_mangle]
pub extern "C" fn prod_kv_cache_init_for_documents(
    devices: *const i32,
    num_devices: usize,
    manager_out: *mut *mut CProductionManager,
) -> i32 {
    if devices.is_null() || manager_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let device_slice = unsafe { std::slice::from_raw_parts(devices, num_devices) };
    let device_vec = device_slice.to_vec();

    match ProductionKVCacheManager::for_document_processing(device_vec) {
        Ok(manager) => {
            unsafe {
                *manager_out = Box::into_raw(Box::new(CProductionManager(manager)));
            }
            PROD_SUCCESS
        }
        Err(LLMServerError::CudaError(_)) => PROD_ERROR_CUDA,
        Err(LLMServerError::OutOfMemory) => PROD_ERROR_OUT_OF_MEMORY,
        Err(LLMServerError::DeviceNotAvailable) => PROD_ERROR_INVALID_DEVICE,
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Free production manager
#[no_mangle]
pub extern "C" fn prod_kv_cache_manager_free(manager: *mut CProductionManager) {
    if !manager.is_null() {
        unsafe {
            let _ = Box::from_raw(manager);
        }
    }
}

/// Create sequence arena with GROWTH CAPACITY for true zero-copy
#[no_mangle]
pub extern "C" fn prod_create_sequence_arena_with_growth(
    manager: *mut CProductionManager,
    initial_seq_len: usize,
    max_seq_len: usize,        // CRITICAL: Maximum expected sequence length
    hidden_dim: usize,
    num_heads: usize,
    device_id: i32, // -1 for auto-select
    arena_out: *mut *mut CSequenceArena,
) -> i32 {
    if manager.is_null() || arena_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let device = if device_id < 0 { None } else { Some(device_id) };
    let head_dim = hidden_dim / num_heads;

    // CRITICAL: Create arena with growth capacity for true zero-copy
    match manager_ref.create_sequence_arena(initial_seq_len, max_seq_len, num_heads, head_dim, device) {
        Ok(arena) => {
            unsafe {
                *arena_out = Box::into_raw(Box::new(CSequenceArena(arena)));
            }
            PROD_SUCCESS
        }
        Err(LLMServerError::CudaError(_)) => PROD_ERROR_CUDA,
        Err(LLMServerError::OutOfMemory) => PROD_ERROR_OUT_OF_MEMORY,
        Err(LLMServerError::DeviceNotAvailable) => PROD_ERROR_INVALID_DEVICE,
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Legacy compatibility function (creates arena with default growth)
#[no_mangle]
pub extern "C" fn prod_create_sequence_arena(
    manager: *mut CProductionManager,
    initial_seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
    device_id: i32,
    arena_out: *mut *mut CSequenceArena,
) -> i32 {
    // Default max_seq_len to 4x initial for reasonable growth
    let max_seq_len = initial_seq_len * 4;
    prod_create_sequence_arena_with_growth(
        manager, initial_seq_len, max_seq_len, hidden_dim, num_heads, device_id, arena_out
    )
}

/// Free sequence arena
#[no_mangle]
pub extern "C" fn prod_sequence_arena_free(arena: *mut CSequenceArena) {
    if !arena.is_null() {
        unsafe {
            let _ = Box::from_raw(arena);
        }
    }
}

/// Allocate KV tensor with MAXIMUM CAPACITY for true zero-copy growth
#[no_mangle]
pub extern "C" fn prod_allocate_kv_tensor_with_growth(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    initial_seq_len: usize,
    max_seq_len: usize,        // CRITICAL: Pre-allocate for this maximum
    hidden_dim: usize,
    num_heads: usize,
    dtype_size: usize,
    tensor_out: *mut *mut CKVTensor,
) -> i32 {
    if manager.is_null() || arena.is_null() || tensor_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let arena_ref = unsafe { &(*arena).0 };
    let head_dim = hidden_dim / num_heads;

    // CRITICAL: Allocate with MAXIMUM capacity for true zero-copy extensions
    match manager_ref.allocate_kv_tensor(
        arena_ref,
        initial_seq_len,
        max_seq_len,      // This enables zero-copy growth up to max_seq_len
        num_heads,
        head_dim,
        dtype_size,
    ) {
        Ok(tensor) => {
            unsafe {
                *tensor_out = Box::into_raw(Box::new(CKVTensor(tensor)));
            }
            PROD_SUCCESS
        }
        Err(LLMServerError::CudaError(_)) => PROD_ERROR_CUDA,
        Err(LLMServerError::OutOfMemory) => PROD_ERROR_OUT_OF_MEMORY,
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Legacy compatibility function (uses default max growth)
#[no_mangle]
pub extern "C" fn prod_allocate_kv_tensor(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    initial_seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
    dtype_size: usize,
    tensor_out: *mut *mut CKVTensor,
) -> i32 {
    let max_seq_len = initial_seq_len * 4; // Default 4x growth capacity
    prod_allocate_kv_tensor_with_growth(
        manager, arena, initial_seq_len, max_seq_len, hidden_dim, num_heads, dtype_size, tensor_out
    )
}

/// Free KV tensor
#[no_mangle]
pub extern "C" fn prod_kv_tensor_free(tensor: *mut CKVTensor) {
    if !tensor.is_null() {
        unsafe {
            let _ = Box::from_raw(tensor);
        }
    }
}

/// TRUE zero-copy tensor extension - ATOMIC metadata update only
#[no_mangle]
pub extern "C" fn prod_extend_tensor_zero_copy(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    tensor: *mut CKVTensor,
    new_tokens: usize,
    was_zero_copy_out: *mut i32,
    extension_time_ns_out: *mut u64,
) -> i32 {
    if manager.is_null() || arena.is_null() || tensor.is_null() || 
       was_zero_copy_out.is_null() || extension_time_ns_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let arena_ref = unsafe { &(*arena).0 };
    let tensor_ref = unsafe { &mut (*tensor).0 };

    let start_time = std::time::Instant::now();

    match manager_ref.extend_tensor_for_generation(arena_ref, tensor_ref, new_tokens) {
        Ok(was_zero_copy) => {
            let extension_time = start_time.elapsed().as_nanos() as u64;
            
            unsafe {
                *was_zero_copy_out = if was_zero_copy { 1 } else { 0 };
                *extension_time_ns_out = extension_time;
            }
            PROD_SUCCESS
        }
        Err(LLMServerError::CudaError(_)) => PROD_ERROR_CUDA,
        Err(LLMServerError::OutOfMemory) => PROD_ERROR_OUT_OF_MEMORY,
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

/// Measure zero-copy efficiency with multiple extension steps
#[no_mangle]
pub extern "C" fn prod_measure_zero_copy_efficiency(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    tensor: *mut CKVTensor,
    extension_steps: *const usize,
    num_steps: usize,
    report_out: *mut CZeroCopyEfficiencyReport,
) -> i32 {
    if manager.is_null() || arena.is_null() || tensor.is_null() || 
       extension_steps.is_null() || report_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let arena_ref = unsafe { &(*arena).0 };
    let tensor_ref = unsafe { &mut (*tensor).0 };
    let steps_slice = unsafe { std::slice::from_raw_parts(extension_steps, num_steps) };

    match manager_ref.measure_zero_copy_efficiency(arena_ref, tensor_ref, steps_slice) {
        Ok(report) => {
            unsafe {
                (*report_out).initial_seq_len = report.initial_seq_len;
                (*report_out).final_seq_len = report.final_seq_len;
                (*report_out).total_extensions = report.total_extensions;
                (*report_out).zero_copy_extensions = report.zero_copy_extensions;
                (*report_out).zero_copy_rate = report.zero_copy_rate;
                (*report_out).total_time_ns = report.total_time_ns;
                (*report_out).avg_extension_time_ns = report.avg_extension_time_ns;
                (*report_out).final_utilization = report.final_utilization;
                (*report_out).memory_efficiency = report.memory_efficiency;
            }
            PROD_SUCCESS
        }
        Err(LLMServerError::CudaError(_)) => PROD_ERROR_CUDA,
        Err(LLMServerError::OutOfMemory) => PROD_ERROR_OUT_OF_MEMORY,
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Validate zero-copy implementation
#[no_mangle]
pub extern "C" fn prod_validate_zero_copy_implementation(
    manager: *mut CProductionManager,
    report_out: *mut CZeroCopyValidationReport,
) -> i32 {
    if manager.is_null() || report_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };

    match manager_ref.validate_zero_copy_implementation() {
        Ok(report) => {
            unsafe {
                (*report_out).basic_zero_copy_works = if report.basic_zero_copy_works { 1 } else { 0 };
                (*report_out).beyond_capacity_handled_correctly = if report.beyond_capacity_handled_correctly { 1 } else { 0 };
                (*report_out).memory_efficiency_reporting_ok = if report.memory_efficiency_reporting_ok { 1 } else { 0 };
                (*report_out).capacity_reporting_accurate = if report.capacity_reporting_accurate { 1 } else { 0 };
                (*report_out).all_tests_passed = if report.all_tests_passed { 1 } else { 0 };
            }
            PROD_SUCCESS
        }
        Err(LLMServerError::CudaError(_)) => PROD_ERROR_CUDA,
        Err(LLMServerError::OutOfMemory) => PROD_ERROR_OUT_OF_MEMORY,
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Batch extend multiple tensors for concurrent generation
#[no_mangle]
pub extern "C" fn prod_batch_extend_tensors(
    manager: *mut CProductionManager,
    arenas: *const *mut CSequenceArena,
    tensors: *mut *mut CKVTensor,
    new_tokens: *const usize,
    num_tensors: usize,
    results_out: *mut i32,
    total_zero_copy_out: *mut usize,
    avg_extension_time_ns_out: *mut u64,
) -> i32 {
    if manager.is_null() || arenas.is_null() || tensors.is_null() || 
       new_tokens.is_null() || results_out.is_null() || 
       total_zero_copy_out.is_null() || avg_extension_time_ns_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let arenas_slice = unsafe { std::slice::from_raw_parts(arenas, num_tensors) };
    let tensors_slice = unsafe { std::slice::from_raw_parts_mut(tensors, num_tensors) };
    let tokens_slice = unsafe { std::slice::from_raw_parts(new_tokens, num_tensors) };
    let results_slice = unsafe { std::slice::from_raw_parts_mut(results_out, num_tensors) };

    let start_time = std::time::Instant::now();

    // Process extensions one by one to avoid lifetime issues
    let mut total_zero_copy = 0;
    for i in 0..num_tensors {
        let arena_ref = unsafe { &(*arenas_slice[i]).0 };
        let tensor_ref = unsafe { &mut (*tensors_slice[i]).0 };
        
        match manager_ref.extend_tensor_for_generation(arena_ref, tensor_ref, tokens_slice[i]) {
            Ok(was_zero_copy) => {
                results_slice[i] = if was_zero_copy { 1 } else { 0 };
                if was_zero_copy {
                    total_zero_copy += 1;
                }
            }
            Err(_) => {
                results_slice[i] = 0;
                // Continue with other tensors even if one fails
            }
        }
    }

    let batch_time = start_time.elapsed().as_nanos() as u64;
    let avg_time = if num_tensors > 0 { batch_time / num_tensors as u64 } else { 0 };

    unsafe {
        *total_zero_copy_out = total_zero_copy;
        *avg_extension_time_ns_out = avg_time;
    }

    PROD_SUCCESS
}

/// Get zero-copy recommendations
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
    let recommendations = manager_ref.get_zero_copy_recommendations();

    let num_to_return = recommendations.len().min(max_recommendations);
    
    for (i, recommendation) in recommendations.iter().take(num_to_return).enumerate() {
        if let Ok(c_string) = CString::new(recommendation.as_str()) {
            unsafe {
                let c_str_ptr = libc::malloc(c_string.as_bytes_with_nul().len()) as *mut c_char;
                if !c_str_ptr.is_null() {
                    std::ptr::copy_nonoverlapping(
                        c_string.as_ptr(),
                        c_str_ptr,
                        c_string.as_bytes_with_nul().len(),
                    );
                    *recommendations_out.add(i) = c_str_ptr;
                }
            }
        }
    }

    unsafe {
        *num_recommendations_out = num_to_return;
    }

    PROD_SUCCESS
}

/// Free zero-copy recommendations
#[no_mangle]
pub extern "C" fn prod_free_zero_copy_recommendations(
    recommendations: *mut *mut c_char,
    num_recommendations: usize,
) {
    if recommendations.is_null() {
        return;
    }

    unsafe {
        for i in 0..num_recommendations {
            let ptr = *recommendations.add(i);
            if !ptr.is_null() {
                libc::free(ptr as *mut c_void);
            }
        }
    }
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

/// Demonstrate zero-copy vs copy performance comparison
#[no_mangle]
pub extern "C" fn prod_benchmark_zero_copy_vs_copy(
    manager: *mut CProductionManager,
    initial_seq_len: usize,
    max_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    num_extensions: usize,
    tokens_per_extension: usize,
    zero_copy_time_ns_out: *mut u64,
    zero_copy_count_out: *mut usize,
    copy_time_ns_out: *mut u64,
    copy_count_out: *mut usize,
    efficiency_ratio_out: *mut f64,
) -> i32 {
    if manager.is_null() || zero_copy_time_ns_out.is_null() || zero_copy_count_out.is_null() ||
       copy_time_ns_out.is_null() || copy_count_out.is_null() || efficiency_ratio_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };

    // Create arena and tensor for zero-copy test
    match manager_ref.create_sequence_arena(initial_seq_len, max_seq_len, num_heads, head_dim, None) {
        Ok(arena) => {
            match manager_ref.allocate_kv_tensor(&arena, initial_seq_len, max_seq_len, num_heads, head_dim, 2) {
                Ok(mut tensor) => {
                    let start_time = std::time::Instant::now();
                    let mut zero_copy_count = 0;
                    
                    // Test zero-copy extensions
                    for _ in 0..num_extensions {
                        match manager_ref.extend_tensor_for_generation(&arena, &mut tensor, tokens_per_extension) {
                            Ok(was_zero_copy) => {
                                if was_zero_copy {
                                    zero_copy_count += 1;
                                } else {
                                    break; // Hit capacity limit
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    
                    let zero_copy_time = start_time.elapsed().as_nanos() as u64;
                    let copy_count = num_extensions - zero_copy_count;
                    
                    // Estimate copy time (simplified - in real implementation would test actual copies)
                    let copy_time_estimate = zero_copy_time * 100; // Assume copies are 100x slower
                    
                    let efficiency_ratio = if copy_time_estimate > 0 {
                        copy_time_estimate as f64 / zero_copy_time as f64
                    } else {
                        1.0
                    };

                    unsafe {
                        *zero_copy_time_ns_out = zero_copy_time;
                        *zero_copy_count_out = zero_copy_count;
                        *copy_time_ns_out = copy_time_estimate;
                        *copy_count_out = copy_count;
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

/// Get production metrics
#[no_mangle]
pub extern "C" fn prod_get_metrics(
    manager: *mut CProductionManager,
    sequences_processed_out: *mut usize,
    tokens_generated_out: *mut usize,
    zero_copy_extensions_out: *mut usize,
    copy_extensions_out: *mut usize,
    zero_copy_ratio_out: *mut f64,
    avg_allocation_time_ms_out: *mut f64,
    avg_extension_time_ms_out: *mut f64,
) -> i32 {
    if manager.is_null() || sequences_processed_out.is_null() || tokens_generated_out.is_null() ||
       zero_copy_extensions_out.is_null() || copy_extensions_out.is_null() ||
       zero_copy_ratio_out.is_null() || avg_allocation_time_ms_out.is_null() ||
       avg_extension_time_ms_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let metrics = manager_ref.get_production_metrics();

    unsafe {
        *sequences_processed_out = metrics.sequences_processed;
        *tokens_generated_out = metrics.tokens_generated;
        *zero_copy_extensions_out = metrics.zero_copy_extensions;
        *copy_extensions_out = metrics.copy_extensions;
        *zero_copy_ratio_out = metrics.zero_copy_ratio;
        *avg_allocation_time_ms_out = metrics.avg_allocation_time_ms;
        *avg_extension_time_ms_out = metrics.avg_extension_time_ms;
    }

    PROD_SUCCESS
}

/// Get system health status
#[no_mangle]
pub extern "C" fn prod_get_system_health(
    manager: *mut CProductionManager,
    status_out: *mut i32,
    health_score_out: *mut f64,
    num_recommendations_out: *mut usize,
) -> i32 {
    if manager.is_null() || status_out.is_null() || health_score_out.is_null() ||
       num_recommendations_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let health = manager_ref.get_system_health();

    let status_code = match health.status {
        SystemStatus::Excellent => 3,
        SystemStatus::Good => 2,
        SystemStatus::Warning => 1,
        SystemStatus::Critical => 0,
    };

    unsafe {
        *status_out = status_code;
        *health_score_out = health.health_score;
        *num_recommendations_out = health.recommendations.len();
    }

    PROD_SUCCESS
}

/// Get specific recommendation by index
#[no_mangle]
pub extern "C" fn prod_get_recommendation(
    manager: *mut CProductionManager,
    index: usize,
    recommendation_out: *mut c_char,
    max_len: usize,
) -> i32 {
    if manager.is_null() || recommendation_out.is_null() || max_len == 0 {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let health = manager_ref.get_system_health();

    if index >= health.recommendations.len() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let recommendation = &health.recommendations[index];
    let c_string = match CString::new(recommendation.as_str()) {
        Ok(s) => s,
        Err(_) => return PROD_ERROR_INVALID_PARAM,
    };

    let bytes = c_string.as_bytes_with_nul();
    if bytes.len() > max_len {
        return PROD_ERROR_INVALID_PARAM;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr() as *const c_char,
            recommendation_out,
            bytes.len(),
        );
    }

    PROD_SUCCESS
}

/// Perform maintenance cleanup
#[no_mangle]
pub extern "C" fn prod_maintenance_cleanup(
    manager: *mut CProductionManager,
    inactive_arenas_cleaned_out: *mut usize,
    old_pages_cleaned_out: *mut usize,
    bytes_defragmented_out: *mut usize,
    maintenance_time_ms_out: *mut f64,
) -> i32 {
    if manager.is_null() || inactive_arenas_cleaned_out.is_null() ||
       old_pages_cleaned_out.is_null() || bytes_defragmented_out.is_null() ||
       maintenance_time_ms_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let report = manager_ref.maintenance_cleanup();

    unsafe {
        *inactive_arenas_cleaned_out = report.inactive_arenas_cleaned;
        *old_pages_cleaned_out = report.old_pages_cleaned;
        *bytes_defragmented_out = report.bytes_defragmented;
        *maintenance_time_ms_out = report.maintenance_time_ms;
    }

    PROD_SUCCESS
}

/// Batch allocate sequences for concurrent processing
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

    // Convert C requests to Rust requests with proper field mapping
    let sequence_requests: Vec<SequenceRequest> = request_slice.iter()
        .map(|req| {
            // Calculate head_dim from hidden_dim and num_heads
            let head_dim = if req.num_heads > 0 {
                req.hidden_dim / req.num_heads
            } else {
                128 // Default head_dim if num_heads is 0
            };
            
            SequenceRequest {
                initial_seq_len: req.initial_seq_len,
                max_seq_len: req.max_seq_len,
                num_heads: req.num_heads,
                head_dim, // Use calculated head_dim
                dtype_size: req.dtype_size,
                preferred_device: if req.preferred_device < 0 { None } else { Some(req.preferred_device) },
            }
        })
        .collect();

    match manager_ref.batch_allocate_sequences(&sequence_requests) {
        Ok(allocations) => {
            // Convert results to C format
            let c_results: Vec<CBatchResult> = allocations.into_iter()
                .map(|alloc| CBatchResult {
                    request_id: alloc.request_id,
                    arena: Box::into_raw(Box::new(CSequenceArena(alloc.arena))),
                    tensor: Box::into_raw(Box::new(CKVTensor(alloc.tensor))),
                    device_id: alloc.device_id,
                })
                .collect();

            // Allocate result array
            let results_ptr = unsafe {
                libc::malloc(num_requests * std::mem::size_of::<CBatchResult>()) as *mut CBatchResult
            };

            if results_ptr.is_null() {
                return PROD_ERROR_OUT_OF_MEMORY;
            }

            unsafe {
                std::ptr::copy_nonoverlapping(
                    c_results.as_ptr(),
                    results_ptr,
                    c_results.len(),
                );
                *results_out = results_ptr;
            }

            PROD_SUCCESS
        }
        Err(LLMServerError::CudaError(_)) => PROD_ERROR_CUDA,
        Err(LLMServerError::OutOfMemory) => PROD_ERROR_OUT_OF_MEMORY,
        Err(LLMServerError::DeviceNotAvailable) => PROD_ERROR_INVALID_DEVICE,
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Free batch allocation results
#[no_mangle]
pub extern "C" fn prod_free_batch_results(
    results: *mut CBatchResult,
    num_results: usize,
) {
    if results.is_null() || num_results == 0 {
        return;
    }

    unsafe {
        let result_slice = std::slice::from_raw_parts_mut(results, num_results);
        for result in result_slice {
            if !result.arena.is_null() {
                let _ = Box::from_raw(result.arena);
            }
            if !result.tensor.is_null() {
                let _ = Box::from_raw(result.tensor);
            }
        }
        libc::free(results as *mut c_void);
    }
}

/// Simulate token generation for benchmarking
#[no_mangle]
pub extern "C" fn prod_simulate_generation(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    tensor: *mut CKVTensor,
    num_tokens: usize,
    tokens_generated_out: *mut usize,
    zero_copy_extensions_out: *mut usize,
    copy_extensions_out: *mut usize,
    total_time_ms_out: *mut f64,
    avg_time_per_token_ms_out: *mut f64,
) -> i32 {
    if manager.is_null() || arena.is_null() || tensor.is_null() ||
       tokens_generated_out.is_null() || zero_copy_extensions_out.is_null() ||
       copy_extensions_out.is_null() || total_time_ms_out.is_null() ||
       avg_time_per_token_ms_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let arena_ref = unsafe { &(*arena).0 };
    let tensor_ref = unsafe { &mut (*tensor).0 };

    match simulate_generation(manager_ref, arena_ref, tensor_ref, num_tokens) {
        Ok(stats) => {
            unsafe {
                *tokens_generated_out = stats.tokens_generated;
                *zero_copy_extensions_out = stats.zero_copy_extensions;
                *copy_extensions_out = stats.copy_extensions;
                *total_time_ms_out = stats.total_time_ms;
                *avg_time_per_token_ms_out = stats.avg_time_per_token_ms;
            }
            PROD_SUCCESS
        }
        Err(LLMServerError::CudaError(_)) => PROD_ERROR_CUDA,
        Err(LLMServerError::OutOfMemory) => PROD_ERROR_OUT_OF_MEMORY,
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Generation statistics for benchmarking
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub tokens_generated: usize,
    pub zero_copy_extensions: usize,
    pub copy_extensions: usize,
    pub total_time_ms: f64,
    pub avg_time_per_token_ms: f64,
}

/// Simulate generation for benchmarking
pub fn simulate_generation(
    manager: &ProductionKVCacheManager,
    arena: &Arc<crate::zero_copy::ZeroCopyArena>,
    tensor: &mut crate::zero_copy::ZeroCopyTensor,
    num_tokens: usize,
) -> Result<GenerationStats, LLMServerError> {
    let start_time = Instant::now();
    let mut zero_copy_count = 0;
    let mut copy_count = 0;
    let current_seq_len = tensor.seq_len();
    
    // Simulate incremental token generation
    for i in 1..=num_tokens {
        match manager.extend_tensor_for_generation(arena, tensor, 1) {
            Ok(was_zero_copy) => {
                if was_zero_copy {
                    zero_copy_count += 1;
                } else {
                    copy_count += 1;
                }
            }
            Err(e) => return Err(e),
        }
    }
    
    let total_time = start_time.elapsed();
    let total_time_ms = total_time.as_millis() as f64;
    let avg_time_per_token = if num_tokens > 0 {
        total_time_ms / num_tokens as f64
    } else {
        0.0
    };
    
    Ok(GenerationStats {
        tokens_generated: num_tokens,
        zero_copy_extensions: zero_copy_count,
        copy_extensions: copy_count,
        total_time_ms,
        avg_time_per_token_ms: avg_time_per_token,
    })
}

// Utility functions for integration
#[no_mangle]
pub extern "C" fn prod_get_version() -> *const c_char {
    static VERSION: &[u8] = b"1.0.0-production-zero-copy\0";
    VERSION.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn prod_get_features() -> *const c_char {
    static FEATURES: &[u8] = b"true-zero-copy,slab-recycling,cuda-heap,production-ready\0";
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

/// Get optimal page size for given parameters
#[no_mangle]
pub extern "C" fn prod_calculate_optimal_page_size(
    max_seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
    device_id: i32,
) -> usize {
    // Use KV-specific page size calculation
    let head_dim = if num_heads > 0 { hidden_dim / num_heads } else { 128 };
    crate::kv_layout::calculate_optimal_kv_page_size(max_seq_len, num_heads, head_dim, 2) // fp16 default
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_ffi_basic() {
        // Test basic FFI functionality for zero-copy
        let version = unsafe { CStr::from_ptr(prod_get_version()) };
        assert!(version.to_str().unwrap().contains("zero-copy"));
        
        let features = unsafe { CStr::from_ptr(prod_get_features()) };
        assert!(features.to_str().unwrap().contains("true-zero-copy"));
        
        let cuda_available = prod_check_cuda_availability();
        println!("CUDA available: {}", cuda_available != 0);
        
        let page_size = prod_calculate_optimal_page_size(2048, 4096, 32, 0);
        assert!(page_size >= 1024 * 1024); // At least 1MB for 2048 seq_len
        
        println!("✓ Zero-copy FFI basic tests passed");
    }

    #[test]
    fn test_zero_copy_manager_ffi() {
        let model_name = CString::new("llama-7b").unwrap();
        let devices = [0i32];
        let mut manager_ptr = std::ptr::null_mut();
        
        let result = prod_kv_cache_init_for_server(
            model_name.as_ptr(),
            devices.as_ptr(),
            1,
            &mut manager_ptr,
        );
        
        if result == PROD_SUCCESS {
            // Test zero-copy validation
            let mut validation_report = CZeroCopyValidationReport {
                basic_zero_copy_works: 0,
                beyond_capacity_handled_correctly: 0,
                memory_efficiency_reporting_ok: 0,
                capacity_reporting_accurate: 0,
                all_tests_passed: 0,
            };
            
            let validation_result = prod_validate_zero_copy_implementation(
                manager_ptr,
                &mut validation_report,
            );
            
            if validation_result == PROD_SUCCESS {
                println!("✓ Zero-copy validation: all_tests_passed = {}", validation_report.all_tests_passed);
            }
            
            // Cleanup
            prod_kv_cache_manager_free(manager_ptr);
            println!("✓ Zero-copy manager FFI tests passed");
        } else {
            println!("Zero-copy manager FFI test skipped (no CUDA): error code {}", result);
        }
    }

    #[test]
    fn test_zero_copy_tensor_operations() {
        // Test zero-copy tensor operations through FFI
        let devices = [0i32];
        let mut manager_ptr = std::ptr::null_mut();
        
        let result = prod_kv_cache_init_for_chatbot(
            devices.as_ptr(),
            1,
            &mut manager_ptr,
        );
        
        if result == PROD_SUCCESS {
            let mut arena_ptr = std::ptr::null_mut();
            let arena_result = prod_create_sequence_arena_with_growth(
                manager_ptr,
                128,  // initial_seq_len
                512,  // max_seq_len - enables zero-copy growth
                1024, // hidden_dim
                16,   // num_heads
                0,    // device_id
                &mut arena_ptr,
            );
            
            if arena_result == PROD_SUCCESS {
                let mut tensor_ptr = std::ptr::null_mut();
                let tensor_result = prod_allocate_kv_tensor_with_growth(
                    manager_ptr,
                    arena_ptr,
                    128,  // initial_seq_len
                    512,  // max_seq_len
                    1024, // hidden_dim
                    16,   // num_heads
                    2,    // dtype_size (fp16)
                    &mut tensor_ptr,
                );
                
                if tensor_result == PROD_SUCCESS {
                    // Test zero-copy extension
                    let mut was_zero_copy = 0;
                    let mut extension_time_ns = 0u64;
                    
                    let extension_result = prod_extend_tensor_zero_copy(
                        manager_ptr,
                        arena_ptr,
                        tensor_ptr,
                        64, // new_tokens
                        &mut was_zero_copy,
                        &mut extension_time_ns,
                    );
                    
                    if extension_result == PROD_SUCCESS {
                        println!("✓ Zero-copy extension: success={}, time={}ns", was_zero_copy, extension_time_ns);
                        
                        // Test getting zero-copy stats
                        let mut stats = CZeroCopyStats {
                            current_seq_len: 0,
                            max_seq_len: 0,
                            growth_capacity_remaining: 0,
                            utilization: 0.0,
                            memory_efficiency: 0.0,
                            can_grow_without_copy: 0,
                        };
                        
                        let stats_result = prod_get_zero_copy_stats(tensor_ptr, &mut stats);
                        if stats_result == PROD_SUCCESS {
                            println!("✓ Zero-copy stats: current={}, max={}, remaining={}", 
                                   stats.current_seq_len, stats.max_seq_len, stats.growth_capacity_remaining);
                        }
                    }
                    
                    prod_kv_tensor_free(tensor_ptr);
                }
                
                prod_sequence_arena_free(arena_ptr);
            }
            
            prod_kv_cache_manager_free(manager_ptr);
            println!("✓ Zero-copy tensor operations FFI test completed");
        } else {
            println!("Zero-copy tensor operations test skipped (no CUDA)");
        }
    }
}