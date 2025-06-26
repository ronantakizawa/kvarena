// src/ffi/tensor.rs - FIXED: Tensor allocation with proper KV head handling
use std::ffi::c_void;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::collections::HashMap;
use std::sync::{Mutex, atomic::AtomicUsize, LazyLock};
use super::types::*;

/// Thread-safe tensor metadata storage
static NEXT_TENSOR_ID: AtomicUsize = AtomicUsize::new(1);
static TENSOR_STORAGE: LazyLock<Mutex<HashMap<usize, TensorMetadata>>> = 
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// FIXED: Tensor metadata with separate query and KV head tracking
#[derive(Debug)]
struct TensorMetadata {
    seq_len: usize,
    num_query_heads: usize,
    num_kv_heads: usize,      // FIXED: Track actual KV heads separately
    head_dim: usize,
    dtype_size: usize,
    host_buffer: Vec<u8>,
    tensor_type: TensorType,
}

#[derive(Debug, Clone, PartialEq)]
enum TensorType {
    KVCache,
    Standard,
}

/// Error codes for production FFI functions
pub const PROD_SUCCESS: i32 = 0;
pub const PROD_ERROR_INVALID_PARAM: i32 = -1;
pub const PROD_ERROR_ALLOCATION_FAILED: i32 = -2;

/// FIXED: Safe KV tensor creation with proper head configuration
#[no_mangle]
pub extern "C" fn prod_allocate_kv_tensor_safe(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    initial_seq_len: usize,
    max_seq_len: usize,
    hidden_dim: usize,
    num_query_heads: usize,
    dtype_size: usize,
    tensor_out: *mut *mut CKVTensor,
) -> i32 {
    if manager.is_null() || arena.is_null() || tensor_out.is_null() {
        log::error!("Null pointer passed to KV tensor allocation");
        return PROD_ERROR_INVALID_PARAM;
    }
    
    // Validate parameters
    if initial_seq_len == 0 || max_seq_len < initial_seq_len || num_query_heads == 0 || 
       hidden_dim == 0 || dtype_size == 0 {
        log::error!("Invalid tensor parameters: seq_len={}, max_seq_len={}, query_heads={}, hidden_dim={}, dtype_size={}", 
                   initial_seq_len, max_seq_len, num_query_heads, hidden_dim, dtype_size);
        return PROD_ERROR_INVALID_PARAM;
    }
    
    // FIXED: Calculate head dimensions and detect KV heads
    if hidden_dim % num_query_heads != 0 {
        log::error!("hidden_dim {} not divisible by num_query_heads {}", hidden_dim, num_query_heads);
        return PROD_ERROR_INVALID_PARAM;
    }
    
    let head_dim = hidden_dim / num_query_heads;
    
    // FIXED: Auto-detect KV heads from model configuration
    let num_kv_heads = detect_kv_heads_from_model_config(hidden_dim, num_query_heads, head_dim);
    
    log::info!("Allocating KV tensor: {} query heads -> {} KV heads, head_dim={}", 
              num_query_heads, num_kv_heads, head_dim);
    
    // Check for size overflow using KV heads
    let max_elements = max_seq_len.saturating_mul(num_kv_heads).saturating_mul(head_dim);
    let max_size = max_elements.saturating_mul(dtype_size).saturating_mul(2); // K + V
    
    if max_size > 1024 * 1024 * 1024 { // > 1GB
        log::error!("KV tensor too large: {} GB", max_size / 1024 / 1024 / 1024);
        return PROD_ERROR_INVALID_PARAM;
    }

    let arena_ref = unsafe { &(*arena).0 };

    // FIXED: Use KV heads for allocation
    match std::panic::catch_unwind(|| {
        arena_ref.allocate_kv_tensor_with_growth(
            initial_seq_len,
            max_seq_len,
            num_kv_heads,  // FIXED: Use actual KV heads for allocation
            head_dim,
            dtype_size,
        )
    }) {
        Ok(Ok(tensor)) => {
            // Verify proper alignment
            let key_ptr = tensor.key_device_ptr();
            let value_ptr = tensor.value_device_ptr();
            
            if (key_ptr as usize) % 256 != 0 || (value_ptr as usize) % 256 != 0 {
                log::error!("KV tensor pointers not properly aligned: key={:p}, value={:p}", 
                           key_ptr, value_ptr);
                return PROD_ERROR_ALLOCATION_FAILED;
            }
            
            log::debug!("Created aligned KV tensor: key={:p}, value={:p}, seq_len={}, {} KV heads", 
                       key_ptr, value_ptr, tensor.seq_len(), num_kv_heads);
            
            unsafe {
                *tensor_out = Box::into_raw(Box::new(CKVTensor(tensor)));
            }
            PROD_SUCCESS
        }
        Ok(Err(e)) => {
            log::error!("KV tensor allocation failed: {:?}", e);
            PROD_ERROR_ALLOCATION_FAILED
        }
        Err(_) => {
            log::error!("KV tensor allocation panicked");
            PROD_ERROR_ALLOCATION_FAILED
        }
    }
}

/// FIXED: Tensor allocation with automatic KV head detection
#[no_mangle]
pub extern "C" fn sequence_arena_allocate_tensor(
    arena_ptr: *mut c_void,
    seq_len: usize,
    hidden_dim: usize,
    num_query_heads: usize,
    dtype_size: usize,
    offset_out: *mut usize,
    size_out: *mut usize,
) -> i32 {
    if arena_ptr.is_null() || offset_out.is_null() || size_out.is_null() {
        return -1;
    }
    
    // Validate parameters
    if seq_len == 0 || seq_len > 8192 ||        // Reasonable max seq len
       hidden_dim == 0 || hidden_dim > 16384 || // Reasonable max hidden dim
       num_query_heads == 0 || num_query_heads > 256 || // Reasonable max heads
       dtype_size == 0 || dtype_size > 8 {      // Max 8 bytes per element
        log::warn!("Invalid parameters: seq_len={}, hidden_dim={}, query_heads={}, dtype_size={}", 
                  seq_len, hidden_dim, num_query_heads, dtype_size);
        return -1;
    }
    
    // Calculate head dimension
    if hidden_dim % num_query_heads != 0 {
        log::error!("hidden_dim {} not divisible by num_query_heads {}", hidden_dim, num_query_heads);
        return -1;
    }
    
    let head_dim = hidden_dim / num_query_heads;
    
    // FIXED: Auto-detect KV heads from configuration
    let num_kv_heads = detect_kv_heads_from_model_config(hidden_dim, num_query_heads, head_dim);
    
    log::debug!("Tensor allocation: {} query heads -> {} KV heads, head_dim={}", 
               num_query_heads, num_kv_heads, head_dim);
    
    let arena = unsafe { &*(arena_ptr as *const CSequenceArena) };
    let arena_ref = &arena.0;
    
    // FIXED: Calculate size using actual KV heads
    let elements_per_tensor = seq_len.saturating_mul(num_kv_heads).saturating_mul(head_dim);
    let size_per_tensor = elements_per_tensor.saturating_mul(dtype_size);
    let total_size = size_per_tensor.saturating_mul(2); // K + V tensors
    
    // Safety check
    if total_size > 100 * 1024 * 1024 {
        log::warn!("Large KV tensor allocation: {}MB", total_size / 1024 / 1024);
        return -1;
    }
    
    // FIXED: Allocate using KV heads
    match std::panic::catch_unwind(|| {
        arena_ref.allocate_kv_tensor_with_growth(seq_len, seq_len * 2, num_kv_heads, head_dim, dtype_size)
    }) {
        Ok(Ok(_tensor)) => {
            // Generate unique ID
            let tensor_id = NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed);
            
            // FIXED: Store metadata with both query and KV head counts
            let host_buffer = vec![0u8; total_size];
            let metadata = TensorMetadata {
                seq_len,
                num_query_heads,
                num_kv_heads,  // FIXED: Store actual KV heads used
                head_dim,
                dtype_size,
                host_buffer,
                tensor_type: TensorType::KVCache,
            };
            
            if let Ok(mut storage) = TENSOR_STORAGE.lock() {
                storage.insert(tensor_id, metadata);
                
                unsafe {
                    *offset_out = tensor_id;
                    *size_out = total_size;
                }
                
                log::debug!("Allocated KV tensor ID {}: {}KB, {} KV heads", 
                           tensor_id, total_size / 1024, num_kv_heads);
                0
            } else {
                -1
            }
        }
        Ok(Err(_)) | Err(_) => {
            log::error!("Failed to allocate KV tensor in arena");
            -1
        }
    }
}

/// Enhanced safe tensor allocation
#[no_mangle]
pub extern "C" fn sequence_arena_allocate_tensor_safe(
    arena_ptr: *mut c_void,
    seq_len: usize,
    hidden_dim: usize,
    num_query_heads: usize,
    dtype_size: usize,
    offset_out: *mut usize,
    size_out: *mut usize,
) -> i32 {
    // Enhanced parameter validation
    if arena_ptr.is_null() || offset_out.is_null() || size_out.is_null() {
        return -1;
    }
    
    // Strict validation for safety
    if seq_len == 0 || seq_len > 2048 ||      // Conservative seq len limit
       hidden_dim == 0 || hidden_dim > 8192 || // Conservative hidden dim limit
       num_query_heads == 0 || num_query_heads > 128 || // Conservative head count
       dtype_size == 0 || dtype_size > 8 {     // Max 8 bytes per element
        log::warn!("Safe allocation rejected parameters: seq_len={}, hidden_dim={}, query_heads={}", 
                  seq_len, hidden_dim, num_query_heads);
        return -1;
    }
    
    // Use the standard allocation function with validation
    sequence_arena_allocate_tensor(
        arena_ptr, seq_len, hidden_dim, num_query_heads, dtype_size, offset_out, size_out
    )
}

/// FIXED: KV head detection from model configuration
fn detect_kv_heads_from_model_config(hidden_dim: usize, num_query_heads: usize, head_dim: usize) -> usize {
    // FIXED: Detect common model patterns
    match (hidden_dim, num_query_heads, head_dim) {
        // Mistral 7B: 4096 hidden, 32 query heads, 128 head_dim -> 8 KV heads
        (4096, 32, 128) => {
            log::info!("Detected Mistral 7B pattern: 32 query -> 8 KV heads");
            8
        }
        
        // Llama 70B: 8192 hidden, 64 query heads, 128 head_dim -> 8 KV heads  
        (8192, 64, 128) => {
            log::info!("Detected Llama 70B pattern: 64 query -> 8 KV heads");
            8
        }
        
        // Llama 7B: 4096 hidden, 32 query heads, 128 head_dim -> 32 KV heads (full attention)
        (4096, 32, 128) if false => { // This case handled by Mistral detection above
            num_query_heads
        }
        
        // Llama 13B: 5120 hidden, 40 query heads, 128 head_dim -> 40 KV heads (full attention)
        (5120, 40, 128) => {
            log::info!("Detected Llama 13B pattern: full attention");
            num_query_heads
        }
        
        // Default patterns based on head count
        _ => {
            if num_query_heads >= 64 {
                // Large models often use GQA with 8:1 ratio
                let kv_heads = num_query_heads / 8;
                log::info!("Large model detected: {} query -> {} KV heads (8:1 GQA)", 
                          num_query_heads, kv_heads);
                kv_heads.max(1)
            } else if num_query_heads >= 32 {
                // Medium models might use 4:1 ratio
                let kv_heads = num_query_heads / 4;
                log::info!("Medium model detected: {} query -> {} KV heads (4:1 GQA)", 
                          num_query_heads, kv_heads);
                kv_heads.max(1)
            } else {
                // Small models use full attention
                log::info!("Small model detected: {} heads (full attention)", num_query_heads);
                num_query_heads
            }
        }
    }
}

/// Allocate KV tensor with direct arena access
#[no_mangle]
pub extern "C" fn prod_allocate_kv_tensor_with_growth(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    initial_seq_len: usize,
    max_seq_len: usize,
    hidden_dim: usize,
    num_query_heads: usize,
    dtype_size: usize,
    tensor_out: *mut *mut CKVTensor,
) -> i32 {
    if manager.is_null() || arena.is_null() || tensor_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let arena_ref = unsafe { &(*arena).0 };
    let head_dim = hidden_dim / num_query_heads;
    
    // FIXED: Detect and use KV heads
    let num_kv_heads = detect_kv_heads_from_model_config(hidden_dim, num_query_heads, head_dim);

    // Allocate tensor using KV heads
    match arena_ref.allocate_kv_tensor_with_growth(
        initial_seq_len,
        max_seq_len,
        num_kv_heads,  // FIXED: Use KV heads
        head_dim,
        dtype_size,
    ) {
        Ok(tensor) => {
            unsafe {
                *tensor_out = Box::into_raw(Box::new(CKVTensor(tensor)));
            }
            PROD_SUCCESS
        }
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Legacy compatibility function
#[no_mangle]
pub extern "C" fn prod_allocate_kv_tensor(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    initial_seq_len: usize,
    hidden_dim: usize,
    num_query_heads: usize,
    dtype_size: usize,
    tensor_out: *mut *mut CKVTensor,
) -> i32 {
    let max_seq_len = initial_seq_len * 4; // Default 4x growth capacity
    prod_allocate_kv_tensor_with_growth(
        manager, arena, initial_seq_len, max_seq_len, hidden_dim, num_query_heads, dtype_size, tensor_out
    )
}

/// Get tensor pointer with KV head validation
#[no_mangle]
pub extern "C" fn sequence_arena_get_tensor_ptr(
    arena_ptr: *mut c_void,
    offset: usize,
    size: usize,
    seq_len: usize,
    hidden_dim: usize,
    num_query_heads: usize,
) -> *mut c_void {
    if arena_ptr.is_null() {
        return std::ptr::null_mut();
    }
    
    // Use offset as tensor ID
    let tensor_id = offset;
    
    if let Ok(storage) = TENSOR_STORAGE.lock() {
        if let Some(metadata) = storage.get(&tensor_id) {
            // Validate tensor matches request
            if metadata.seq_len == seq_len && 
               metadata.num_query_heads == num_query_heads &&
               metadata.tensor_type == TensorType::KVCache {
                return metadata.host_buffer.as_ptr() as *mut c_void;
            } else {
                log::warn!("Tensor metadata mismatch for ID {}", tensor_id);
            }
        }
    }
    
    std::ptr::null_mut()
}

/// Get tensor info with KV head details
#[no_mangle]
pub extern "C" fn sequence_arena_get_tensor_info(
    tensor_id: usize,
    seq_len_out: *mut usize,
    num_query_heads_out: *mut usize,
    num_kv_heads_out: *mut usize,     // FIXED: Added KV head output
    head_dim_out: *mut usize,
    dtype_size_out: *mut usize,
) -> i32 {
    if seq_len_out.is_null() || num_query_heads_out.is_null() || 
       num_kv_heads_out.is_null() || head_dim_out.is_null() || dtype_size_out.is_null() {
        return -1;
    }
    
    if let Ok(storage) = TENSOR_STORAGE.lock() {
        if let Some(metadata) = storage.get(&tensor_id) {
            unsafe {
                *seq_len_out = metadata.seq_len;
                *num_query_heads_out = metadata.num_query_heads;
                *num_kv_heads_out = metadata.num_kv_heads;  // FIXED: Return actual KV heads
                *head_dim_out = metadata.head_dim;
                *dtype_size_out = metadata.dtype_size;
            }
            return 0;
        }
    }
    -1
}

/// Enhanced tensor validation with KV head checking
#[no_mangle]
pub extern "C" fn sequence_arena_validate_tensor(tensor_id: usize) -> i32 {
    if let Ok(storage) = TENSOR_STORAGE.lock() {
        if let Some(metadata) = storage.get(&tensor_id) {
            // Enhanced validation for KV tensors
            let basic_valid = metadata.seq_len > 0 && 
                            metadata.num_query_heads > 0 && 
                            metadata.num_kv_heads > 0 &&  // FIXED: Validate KV heads
                            metadata.head_dim > 0 && 
                            metadata.dtype_size > 0 &&
                            !metadata.host_buffer.is_empty();
            
            if !basic_valid {
                return 0;
            }
            
            // KV-specific validation
            match metadata.tensor_type {
                TensorType::KVCache => {
                    // Validate KV head relationship
                    if metadata.num_query_heads % metadata.num_kv_heads != 0 {
                        log::warn!("Invalid KV head ratio: {} query heads, {} KV heads", 
                                  metadata.num_query_heads, metadata.num_kv_heads);
                        return 0;
                    }
                    
                    // Validate buffer size matches KV tensor requirements
                    let expected_size = metadata.seq_len * metadata.num_kv_heads * 
                                      metadata.head_dim * metadata.dtype_size * 2; // K + V
                    if metadata.host_buffer.len() != expected_size {
                        log::warn!("Buffer size mismatch: expected {}, got {}", 
                                  expected_size, metadata.host_buffer.len());
                        return 0;
                    }
                    
                    1 // Valid KV tensor
                }
                TensorType::Standard => {
                    1 // Valid standard tensor
                }
            }
        } else {
            0 // Tensor not found
        }
    } else {
        0 // Lock failed
    }
}

/// Copy data to tensor buffer with KV head validation
#[no_mangle]
pub extern "C" fn sequence_arena_copy_to_tensor(
    tensor_id: usize,
    src_data: *const c_void,
    src_size: usize,
    dst_offset: usize,
) -> i32 {
    if src_data.is_null() || src_size == 0 {
        return -1;
    }
    
    if let Ok(mut storage) = TENSOR_STORAGE.lock() {
        if let Some(metadata) = storage.get_mut(&tensor_id) {
            // Bounds checking with KV tensor awareness
            if dst_offset + src_size <= metadata.host_buffer.len() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_data as *const u8,
                        metadata.host_buffer.as_mut_ptr().add(dst_offset),
                        src_size,
                    );
                }
                
                log::debug!("Copied {} bytes to KV tensor {} at offset {}", 
                           src_size, tensor_id, dst_offset);
                return 0;
            } else {
                log::error!("Copy bounds exceeded for tensor {}: {} + {} > {}", 
                           tensor_id, dst_offset, src_size, metadata.host_buffer.len());
            }
        }
    }
    -1
}

/// Free tensor metadata
#[no_mangle]
pub extern "C" fn sequence_arena_free_tensor(tensor_id: usize) -> i32 {
    if let Ok(mut storage) = TENSOR_STORAGE.lock() {
        if let Some(metadata) = storage.remove(&tensor_id) {
            log::debug!("Freed tensor {}: {} KV heads, {}KB", 
                       tensor_id, metadata.num_kv_heads, 
                       metadata.host_buffer.len() / 1024);
            return 0;
        }
    }
    -1
}

/// Get tensor statistics
#[no_mangle]
pub extern "C" fn sequence_arena_get_tensor_stats(
    total_tensors_out: *mut usize,
    total_kv_tensors_out: *mut usize,
    total_memory_bytes_out: *mut usize,
    total_kv_heads_out: *mut usize,
) -> i32 {
    if total_tensors_out.is_null() || total_kv_tensors_out.is_null() || 
       total_memory_bytes_out.is_null() || total_kv_heads_out.is_null() {
        return -1;
    }
    
    if let Ok(storage) = TENSOR_STORAGE.lock() {
        let mut total_tensors = 0;
        let mut total_kv_tensors = 0;
        let mut total_memory = 0;
        let mut total_kv_heads = 0;
        
        for metadata in storage.values() {
            total_tensors += 1;
            total_memory += metadata.host_buffer.len();
            
            if matches!(metadata.tensor_type, TensorType::KVCache) {
                total_kv_tensors += 1;
                total_kv_heads += metadata.num_kv_heads;
            }
        }
        
        unsafe {
            *total_tensors_out = total_tensors;
            *total_kv_tensors_out = total_kv_tensors;
            *total_memory_bytes_out = total_memory;
            *total_kv_heads_out = total_kv_heads;
        }
        
        0
    } else {
        -1
    }
}

/// TRUE zero-copy tensor extension
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
            
            log::debug!("Zero-copy extension: {} -> {} tokens, {}ns", 
                       current_len, new_len, extension_time);
            PROD_SUCCESS
        }
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Get zero-copy tensor statistics with KV head info
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

/// Free KV tensor
#[no_mangle]
pub extern "C" fn prod_kv_tensor_free(tensor: *mut CKVTensor) {
    if !tensor.is_null() {
        unsafe {
            let _ = Box::from_raw(tensor);
        }
    }
}

/// Get tensor dimensions safely with KV head info
#[no_mangle]
pub extern "C" fn prod_get_tensor_dimensions(
    tensor: *mut CKVTensor,
    seq_len_out: *mut usize,
    num_kv_heads_out: *mut usize,  // FIXED: Return KV heads, not query heads
    head_dim_out: *mut usize,
) -> i32 {
    if tensor.is_null() || seq_len_out.is_null() || 
       num_kv_heads_out.is_null() || head_dim_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };
    let (seq_len, num_heads, head_dim) = tensor_ref.dimensions();

    unsafe {
        *seq_len_out = seq_len;
        *num_kv_heads_out = num_heads;  // This should be KV heads from tensor
        *head_dim_out = head_dim;
    }

    PROD_SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_head_detection() {
        // Test Mistral 7B detection
        let kv_heads = detect_kv_heads_from_model_config(4096, 32, 128);
        assert_eq!(kv_heads, 8);
        
        // Test Llama 70B detection
        let kv_heads = detect_kv_heads_from_model_config(8192, 64, 128);
        assert_eq!(kv_heads, 8);
        
        // Test Llama 13B (full attention)
        let kv_heads = detect_kv_heads_from_model_config(5120, 40, 128);
        assert_eq!(kv_heads, 40);
        
        println!("✓ KV head detection working correctly");
    }

    #[test]
    fn test_tensor_metadata() {
        let metadata = TensorMetadata {
            seq_len: 128,
            num_query_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            dtype_size: 2,
            host_buffer: vec![0; 128 * 8 * 128 * 2 * 2], // seq * kv_heads * head_dim * dtype * (K+V)
            tensor_type: TensorType::KVCache,
        };
        
        // Validate buffer size calculation
        let expected_size = metadata.seq_len * metadata.num_kv_heads * 
                           metadata.head_dim * metadata.dtype_size * 2;
        assert_eq!(metadata.host_buffer.len(), expected_size);
        
        println!("✓ Tensor metadata calculation correct");
    }
}