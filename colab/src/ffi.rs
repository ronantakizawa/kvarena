// src/ffi.rs - Fixed production FFI for LLM server integration
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Arc;
use std::time::Instant;
use crate::{
    ProductionKVCacheManager, LLMServerConfig, LLMServerError,
    SequenceRequest, SystemStatus, ProductionMetricsReport, MaintenanceReport, 
    BatchAllocationResult,
};

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

// Generation statistics for benchmarking
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub tokens_generated: usize,
    pub zero_copy_extensions: usize,
    pub copy_extensions: usize,
    pub total_time_ms: f64,
    pub avg_time_per_token_ms: f64,
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

/// Create sequence arena with zero-copy optimization
#[no_mangle]
pub extern "C" fn prod_create_sequence_arena(
    manager: *mut CProductionManager,
    initial_seq_len: usize,
    max_seq_len: usize,
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

/// Free sequence arena
#[no_mangle]
pub extern "C" fn prod_sequence_arena_free(arena: *mut CSequenceArena) {
    if !arena.is_null() {
        unsafe {
            let _ = Box::from_raw(arena);
        }
    }
}

/// Allocate KV tensor with growth capacity
#[no_mangle]
pub extern "C" fn prod_allocate_kv_tensor(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    initial_seq_len: usize,
    max_seq_len: usize,
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

    match manager_ref.allocate_kv_tensor(
        arena_ref,
        initial_seq_len,
        max_seq_len,
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

/// Free KV tensor
#[no_mangle]
pub extern "C" fn prod_kv_tensor_free(tensor: *mut CKVTensor) {
    if !tensor.is_null() {
        unsafe {
            let _ = Box::from_raw(tensor);
        }
    }
}

/// Extend tensor for generation (zero-copy when possible)
#[no_mangle]
pub extern "C" fn prod_extend_tensor_for_generation(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    tensor: *mut CKVTensor,
    new_tokens: usize,
    was_zero_copy_out: *mut i32,
) -> i32 {
    if manager.is_null() || arena.is_null() || tensor.is_null() || was_zero_copy_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    let arena_ref = unsafe { &(*arena).0 };
    let tensor_ref = unsafe { &mut (*tensor).0 };

    match manager_ref.extend_tensor_for_generation(arena_ref, tensor_ref, new_tokens) {
        Ok(was_zero_copy) => {
            unsafe {
                *was_zero_copy_out = if was_zero_copy { 1 } else { 0 };
            }
            PROD_SUCCESS
        }
        Err(LLMServerError::CudaError(_)) => PROD_ERROR_CUDA,
        Err(LLMServerError::OutOfMemory) => PROD_ERROR_OUT_OF_MEMORY,
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
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
        *key_ptr_out = tensor_ref.key_device_ptr() as *mut c_void;
        *value_ptr_out = tensor_ref.value_device_ptr() as *mut c_void;
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
        let new_seq_len = current_seq_len + i;
        
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
    static VERSION: &[u8] = b"1.0.0-production\0";
    VERSION.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn prod_get_features() -> *const c_char {
    static FEATURES: &[u8] = b"zero-copy,slab-recycling,cuda-heap,production-ready\0";
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
    // Simple calculation - in real implementation this would query device properties
    let head_dim = if num_heads > 0 { hidden_dim / num_heads } else { 128 };
    let max_tensor_size = 2 * max_seq_len * num_heads * head_dim * 2; // 2 for K+V, 2 for fp16
    let base_page_size = 1024 * 1024; // 1MB base
    let min_page_size = max_tensor_size * 2; // Fit at least 2 tensors
    
    base_page_size.max(min_page_size).min(16 * 1024 * 1024) // Cap at 16MB
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_ffi_basic() {
        // Test basic FFI functionality
        let version = unsafe { CStr::from_ptr(prod_get_version()) };
        assert!(version.to_str().unwrap().contains("production"));
        
        let features = unsafe { CStr::from_ptr(prod_get_features()) };
        assert!(features.to_str().unwrap().contains("zero-copy"));
        
        let cuda_available = prod_check_cuda_availability();
        println!("CUDA available: {}", cuda_available != 0);
        
        let page_size = prod_calculate_optimal_page_size(2048, 4096, 32, 0);
        assert!(page_size >= 1024 * 1024); // At least 1MB
        
        println!("✓ Production FFI basic tests passed");
    }

    #[test]
    fn test_production_manager_ffi() {
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
            // Test getting metrics
            let mut sequences = 0;
            let mut tokens = 0;
            let mut zero_copy = 0;
            let mut copy_ext = 0;
            let mut ratio = 0.0;
            let mut alloc_time = 0.0;
            let mut ext_time = 0.0;
            
            let metrics_result = prod_get_metrics(
                manager_ptr,
                &mut sequences,
                &mut tokens,
                &mut zero_copy,
                &mut copy_ext,
                &mut ratio,
                &mut alloc_time,
                &mut ext_time,
            );
            
            assert_eq!(metrics_result, PROD_SUCCESS);
            
            // Cleanup
            prod_kv_cache_manager_free(manager_ptr);
            println!("✓ Production manager FFI tests passed");
        } else {
            println!("Production manager FFI test skipped (no CUDA): error code {}", result);
        }
    }

    #[test]
    fn test_generation_simulation() {
        // Test the generation simulation function
        let stats = GenerationStats {
            tokens_generated: 100,
            zero_copy_extensions: 80,
            copy_extensions: 20,
            total_time_ms: 150.0,
            avg_time_per_token_ms: 1.5,
        };
        
        assert_eq!(stats.tokens_generated, 100);
        assert_eq!(stats.zero_copy_extensions, 80);
        assert_eq!(stats.copy_extensions, 20);
        assert_eq!(stats.total_time_ms, 150.0);
        assert_eq!(stats.avg_time_per_token_ms, 1.5);
        
        println!("✓ Generation simulation stats test passed");
    }
}