// src/ffi/tensor_ops.rs - Fixed tensor extension and operation FFI functions
use std::ffi::c_void;
use std::sync::atomic::Ordering;
use super::types::*;

/// Safe sequence arena tensor extension with CUDA crash prevention
#[no_mangle]
pub extern "C" fn sequence_arena_extend_tensor_safe(
    arena_ptr: *mut c_void,
    offset: usize,
    size: usize,
    seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
    new_seq_len: usize,
    dtype_size: usize,
    extended_in_place_out: *mut i32,
    new_offset_out: *mut usize,
    new_size_out: *mut usize,
) -> i32 {
    // Comprehensive parameter validation
    if arena_ptr.is_null() || extended_in_place_out.is_null() || 
       new_offset_out.is_null() || new_size_out.is_null() {
        return -1;
    }
    
    // Validate all numeric parameters
    if seq_len == 0 || hidden_dim == 0 || num_heads == 0 || 
       new_seq_len == 0 || dtype_size == 0 {
        return -1;
    }
    
    // Prevent integer overflow
    if new_seq_len > 1000000 || hidden_dim > 100000 || num_heads > 1000 {
        return -1;
    }
    
    let arena = unsafe { &*(arena_ptr as *const CSequenceArena) };
    let arena_ref = &arena.0;
    let tensor_id = offset; // offset is actually tensor ID
    
    // Check if we can extend in place (simple heuristic for KV tensors)
    let can_extend_in_place = new_seq_len <= seq_len.saturating_mul(4); // Allow 4x growth
    let head_dim = hidden_dim / num_heads;
    
    // Calculate new size with overflow protection
    let new_size = new_seq_len
        .saturating_mul(num_heads)
        .saturating_mul(head_dim)
        .saturating_mul(dtype_size)
        .saturating_mul(2); // K + V
    
    // Reject unreasonably large sizes
    if new_size > 1024 * 1024 * 1024 { // > 1GB
        return -1;
    }
    
    if can_extend_in_place {
        // Try to extend existing KV tensor
        if let Ok(mut storage) = TENSOR_STORAGE.lock() {
            if let Some(metadata) = storage.get_mut(&tensor_id) {
                // Update metadata for KV tensor
                metadata.seq_len = new_seq_len;
                
                // Resize host buffer using Vec::resize (not realloc)
                if metadata.host_buffer.len() < new_size {
                    metadata.host_buffer.resize(new_size, 0);
                }
                
                unsafe {
                    *extended_in_place_out = 1;
                    *new_offset_out = tensor_id;
                    *new_size_out = new_size;
                }
                return 0;
            }
        }
    }
    
    // Create new KV tensor if can't extend in place
    match arena_ref.allocate_kv_tensor_with_growth(new_seq_len, new_seq_len * 2, num_heads, head_dim, dtype_size) {
        Ok(_tensor) => {
            let new_tensor_id = NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed);
            let new_host_buffer = vec![0u8; new_size];
            
            let new_metadata = TensorMetadata {
                seq_len: new_seq_len,
                num_heads,
                head_dim,
                dtype_size,
                host_buffer: new_host_buffer,
            };
            
            if let Ok(mut storage) = TENSOR_STORAGE.lock() {
                // Copy data from old KV tensor if it exists
                if let Some(_old_metadata) = storage.get(&tensor_id) {
                    // Note: In a real implementation, we'd copy the actual KV tensor data here
                }
                storage.insert(new_tensor_id, new_metadata);
            }
            
            unsafe {
                *extended_in_place_out = 0;
                *new_offset_out = new_tensor_id;
                *new_size_out = new_size;
            }
            0
        }
        Err(_) => -1,
    }
}

/// Extend tensor (legacy interface)
#[no_mangle]
pub extern "C" fn sequence_arena_extend_tensor(
    arena_ptr: *mut c_void,
    offset: usize,
    size: usize,
    seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
    new_seq_len: usize,
    dtype_size: usize,
    extended_in_place_out: *mut i32,
    new_offset_out: *mut usize,
    new_size_out: *mut usize,
) -> i32 {
    if arena_ptr.is_null() || extended_in_place_out.is_null() || 
       new_offset_out.is_null() || new_size_out.is_null() {
        return -1;
    }
    
    let arena = unsafe { &*(arena_ptr as *const Arc<crate::SequenceArena>) };
    let tensor_id = offset; // offset is actually tensor ID
    
    // Check if we can extend in place (simple heuristic for KV tensors)
    let can_extend_in_place = new_seq_len <= seq_len * 2;
    let head_dim = hidden_dim / num_heads;
    let new_size = new_seq_len * num_heads * head_dim * dtype_size * 2; // K + V
    
    if can_extend_in_place {
        // Try to extend existing KV tensor
        if let Ok(mut storage) = TENSOR_STORAGE.lock() {
            if let Some(metadata) = storage.get_mut(&tensor_id) {
                // Update metadata for KV tensor
                metadata.seq_len = new_seq_len;
                // Resize host buffer if needed
                if metadata.host_buffer.len() < new_size {
                    metadata.host_buffer.resize(new_size, 0);
                }
                
                unsafe {
                    *extended_in_place_out = 1;
                    *new_offset_out = tensor_id;
                    *new_size_out = new_size;
                }
                return 0;
            }
        }
    }
    
    // Create new KV tensor if can't extend in place
    let head_dim = hidden_dim / num_heads;
    match arena.allocate_tensor_with_growth(new_seq_len, new_seq_len * 2, num_heads, head_dim, dtype_size) {
        Ok(_tensor) => {
            let new_tensor_id = NEXT_TENSOR_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let new_host_buffer = vec![0u8; new_size];
            
            let new_metadata = TensorMetadata {
                seq_len: new_seq_len,
                num_heads,
                head_dim,
                dtype_size,
                host_buffer: new_host_buffer,
            };
            
            if let Ok(mut storage) = TENSOR_STORAGE.lock() {
                // Copy data from old KV tensor if it exists
                if let Some(_old_metadata) = storage.get(&tensor_id) {
                    // Note: In a real implementation, we'd copy the actual KV tensor data here
                }
                storage.insert(new_tensor_id, new_metadata);
            }
            
            unsafe {
                *extended_in_place_out = 0;
                *new_offset_out = new_tensor_id;
                *new_size_out = new_size;
            }
            0
        }
        Err(_) => -1,
    }
}

/// TRUE zero-copy tensor extension - ONLY atomic metadata update
#[no_mangle]
pub extern "C" fn prod_extend_tensor_pure_zero_copy(
    tensor: *mut CKVTensor,
    additional_tokens: usize,
    was_zero_copy_out: *mut i32,
    extension_time_ns_out: *mut u64,
) -> i32 {
    if tensor.is_null() || was_zero_copy_out.is_null() || extension_time_ns_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &mut (*tensor).0 };
    let start_time = std::time::Instant::now();

    // Get current length
    let current_len = tensor_ref.seq_len();
    let new_len = current_len + additional_tokens;

    // TRUE zero-copy: just atomic metadata update
    match tensor_ref.extend_zero_copy(new_len) {
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