// src/ffi/tensor.rs - Tensor allocation and manipulation FFI functions with safety fixes
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

/// Tensor metadata for host-side tracking
#[derive(Debug)]
struct TensorMetadata {
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    dtype_size: usize,
    host_buffer: Vec<u8>,
}

/// Comprehensive tensor diagnostics structure
#[repr(C)]
#[derive(Debug)]
pub struct CTensorDiagnostics {
    pub seq_len: usize,
    pub max_seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub current_bytes: usize,
    pub max_bytes: usize,
    pub utilization: f64,
    pub memory_efficiency: f64,
    pub can_grow: i32,
    pub device_id: i32,
    pub arena_id: u64,
    pub key_ptr_aligned: i32,
    pub value_ptr_aligned: i32,
}

/// Error codes for production FFI functions
pub const PROD_SUCCESS: i32 = 0;
pub const PROD_ERROR_INVALID_PARAM: i32 = -1;
pub const PROD_ERROR_ALLOCATION_FAILED: i32 = -2;

/// Safe tensor creation with CUDA alignment verification
#[no_mangle]
pub extern "C" fn prod_allocate_kv_tensor_safe(
    manager: *mut CProductionManager,
    arena: *mut CSequenceArena,
    initial_seq_len: usize,
    max_seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
    dtype_size: usize,
    tensor_out: *mut *mut CKVTensor,
) -> i32 {
    // Comprehensive parameter validation to prevent bad allocations
    if manager.is_null() || arena.is_null() || tensor_out.is_null() {
        log::error!("Null pointer passed to tensor allocation");
        return PROD_ERROR_INVALID_PARAM;
    }
    
    // Validate parameters that could cause alignment issues
    if initial_seq_len == 0 || max_seq_len < initial_seq_len || num_heads == 0 || 
       hidden_dim == 0 || dtype_size == 0 {
        log::error!("Invalid tensor parameters: seq_len={}, max_seq_len={}, heads={}, hidden_dim={}, dtype_size={}", 
                   initial_seq_len, max_seq_len, num_heads, hidden_dim, dtype_size);
        return PROD_ERROR_INVALID_PARAM;
    }
    
    // Check for dimension overflow that could cause alignment issues
    if hidden_dim % num_heads != 0 {
        log::error!("hidden_dim {} not divisible by num_heads {}", hidden_dim, num_heads);
        return PROD_ERROR_INVALID_PARAM;
    }
    
    let head_dim = hidden_dim / num_heads;
    
    // Check for size overflow before allocation
    let max_elements = max_seq_len.saturating_mul(num_heads).saturating_mul(head_dim);
    let max_size = max_elements.saturating_mul(dtype_size).saturating_mul(2); // K + V
    
    if max_size > 1024 * 1024 * 1024 { // > 1GB
        log::error!("Tensor too large: {} GB", max_size / 1024 / 1024 / 1024);
        return PROD_ERROR_INVALID_PARAM;
    }
    
    // Ensure alignment-friendly sizes
    if dtype_size > 8 || (dtype_size & (dtype_size - 1)) != 0 {
        log::error!("dtype_size {} not power of 2 or too large", dtype_size);
        return PROD_ERROR_INVALID_PARAM;
    }

    let arena_ref = unsafe { &(*arena).0 };

    // Use a safer wrapper around the allocation
    match std::panic::catch_unwind(|| {
        arena_ref.allocate_kv_tensor_with_growth(
            initial_seq_len,
            max_seq_len,
            num_heads,
            head_dim,
            dtype_size,
        )
    }) {
        Ok(Ok(tensor)) => {
            // Verify the tensor was created with proper alignment
            let key_ptr = tensor.key_device_ptr();
            let value_ptr = tensor.value_device_ptr();
            
            // Check alignment of the pointers
            if (key_ptr as usize) % 256 != 0 || (value_ptr as usize) % 256 != 0 {
                log::error!("Tensor pointers not properly aligned: key={:p}, value={:p}", 
                           key_ptr, value_ptr);
                return PROD_ERROR_ALLOCATION_FAILED;
            }
            
            log::debug!("Created aligned tensor: key={:p}, value={:p}, seq_len={}", 
                       key_ptr, value_ptr, tensor.seq_len());
            
            unsafe {
                *tensor_out = Box::into_raw(Box::new(CKVTensor(tensor)));
            }
            PROD_SUCCESS
        }
        Ok(Err(e)) => {
            log::error!("Tensor allocation failed: {:?}", e);
            PROD_ERROR_ALLOCATION_FAILED
        }
        Err(_) => {
            log::error!("Tensor allocation panicked");
            PROD_ERROR_ALLOCATION_FAILED
        }
    }
}

/// Create arena with proper page alignment - removed duplicate, use the one in existing codebase

/// Safe tensor creation with conservative parameters only
#[no_mangle]
pub extern "C" fn sequence_arena_allocate_tensor_safe(
    arena_ptr: *mut c_void,
    seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
    dtype_size: usize,
    offset_out: *mut usize,
    size_out: *mut usize,
) -> i32 {
    // Extremely conservative parameter validation
    if arena_ptr.is_null() || offset_out.is_null() || size_out.is_null() {
        return -1;
    }
    
    // Reject any parameters that could cause overflow
    if seq_len == 0 || seq_len > 1024 ||      // Max 1024 tokens
       hidden_dim == 0 || hidden_dim > 4096 || // Max 4096 hidden dim
       num_heads == 0 || num_heads > 128 ||    // Max 128 heads
       dtype_size == 0 || dtype_size > 8 {     // Max 8 bytes per element
        return -1;
    }
    
    // Check that hidden_dim is divisible by num_heads
    if hidden_dim % num_heads != 0 {
        return -1;
    }
    
    let arena = unsafe { &*(arena_ptr as *const CSequenceArena) };
    let arena_ref = &arena.0;
    let head_dim = hidden_dim / num_heads;
    
    // Pre-calculate size and check for overflow
    let elements_per_tensor = seq_len.saturating_mul(num_heads).saturating_mul(head_dim);
    let size_per_tensor = elements_per_tensor.saturating_mul(dtype_size);
    let total_size = size_per_tensor.saturating_mul(2); // K + V tensors
    
    // Reject if total size is too large (>100MB)
    if total_size > 100 * 1024 * 1024 {
        return -1;
    }
    
    // Use safe allocation wrapper
    match std::panic::catch_unwind(|| {
        arena_ref.allocate_kv_tensor_with_growth(seq_len, seq_len * 2, num_heads, head_dim, dtype_size)
    }) {
        Ok(Ok(_tensor)) => {
            // Generate unique ID
            let tensor_id = NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed);
            
            // Store metadata using safe Vec allocation
            let host_buffer = vec![0u8; total_size];
            let metadata = TensorMetadata {
                seq_len,
                num_heads,
                head_dim,
                dtype_size,
                host_buffer,
            };
            
            if let Ok(mut storage) = TENSOR_STORAGE.lock() {
                storage.insert(tensor_id, metadata);
                
                unsafe {
                    *offset_out = tensor_id;
                    *size_out = total_size;
                }
                0
            } else {
                -1
            }
        }
        Ok(Err(_)) | Err(_) => -1,
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
    num_heads: usize,
    dtype_size: usize,
    tensor_out: *mut *mut CKVTensor,
) -> i32 {
    if manager.is_null() || arena.is_null() || tensor_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let arena_ref = unsafe { &(*arena).0 };
    let head_dim = hidden_dim / num_heads;

    // Allocate tensor directly through arena
    match arena_ref.allocate_kv_tensor_with_growth(
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

/// Create tensor with pure bump allocation
#[no_mangle]
pub extern "C" fn prod_allocate_tensor_pure_bump(
    arena: *mut CSequenceArena,
    initial_seq_len: usize,
    max_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    dtype_size: usize,
    tensor_out: *mut *mut CKVTensor,
) -> i32 {
    if arena.is_null() || tensor_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let arena_ref = unsafe { &(*arena).0 };

    // Pure bump allocation
    match arena_ref.allocate_kv_tensor_with_growth(
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

/// Allocate tensor (legacy interface)
#[no_mangle]
pub extern "C" fn sequence_arena_allocate_tensor(
    arena_ptr: *mut c_void,
    seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
    dtype_size: usize,
    offset_out: *mut usize,
    size_out: *mut usize,
) -> i32 {
    if arena_ptr.is_null() || offset_out.is_null() || size_out.is_null() {
        return -1;
    }
    
    let arena = unsafe { &*(arena_ptr as *const Arc<crate::SequenceArena>) };
    
    // Calculate head_dim from hidden_dim and num_heads
    let head_dim = hidden_dim / num_heads;
    
    // Try to allocate KV tensor in the arena
    match arena.allocate_tensor_with_growth(seq_len, seq_len * 2, num_heads, head_dim, dtype_size) {
        Ok(_tensor) => {
            // Calculate KV tensor size (K + V tensors)
            let tensor_size = seq_len * hidden_dim * dtype_size * 2;
            
            // Create host memory buffer for the KV tensor data
            let host_buffer = vec![0u8; tensor_size];
            
            // Generate a unique tensor ID
            let tensor_id = NEXT_TENSOR_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            
            // Store the KV tensor metadata and host buffer
            let metadata = TensorMetadata {
                seq_len,
                num_heads,
                head_dim,
                dtype_size,
                host_buffer,
            };
            
            if let Ok(mut storage) = TENSOR_STORAGE.lock() {
                storage.insert(tensor_id, metadata);
            }
            
            unsafe {
                *offset_out = tensor_id; // Use tensor ID as "offset"
                *size_out = tensor_size;
            }
            0 // Success
        }
        Err(_) => -1,
    }
}

/// Get tensor pointer (legacy interface)
#[no_mangle]
pub extern "C" fn sequence_arena_get_tensor_ptr(
    arena_ptr: *mut c_void,
    offset: usize,
    size: usize,
    seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
) -> *mut c_void {
    if arena_ptr.is_null() {
        return std::ptr::null_mut();
    }
    
    // Use offset as tensor ID
    let tensor_id = offset;
    
    if let Ok(storage) = TENSOR_STORAGE.lock() {
        if let Some(metadata) = storage.get(&tensor_id) {
            return metadata.host_buffer.as_ptr() as *mut c_void;
        }
    }
    
    std::ptr::null_mut()
}

/// Safe version of get tensor pointer
#[no_mangle]
pub extern "C" fn sequence_arena_get_tensor_ptr_safe(
    arena_ptr: *mut c_void,
    offset: usize,
    size: usize,
    seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
) -> *mut c_void {
    if arena_ptr.is_null() {
        return std::ptr::null_mut();
    }
    
    // Parameter validation
    if seq_len == 0 || hidden_dim == 0 || num_heads == 0 {
        return std::ptr::null_mut();
    }
    
    // Use offset as tensor ID
    let tensor_id = offset;
    
    if let Ok(storage) = TENSOR_STORAGE.lock() {
        if let Some(metadata) = storage.get(&tensor_id) {
            return metadata.host_buffer.as_ptr() as *mut c_void;
        }
    }
    
    std::ptr::null_mut()
}

/// Free tensor metadata
#[no_mangle]
pub extern "C" fn sequence_arena_free_tensor(tensor_id: usize) -> i32 {
    if let Ok(mut storage) = TENSOR_STORAGE.lock() {
        if storage.remove(&tensor_id).is_some() {
            return 0; // Success
        }
    }
    -1 // Failed to remove or lock failed
}

/// Get tensor info without accessing raw memory
#[no_mangle]
pub extern "C" fn sequence_arena_get_tensor_info(
    tensor_id: usize,
    seq_len_out: *mut usize,
    num_heads_out: *mut usize,
    head_dim_out: *mut usize,
    dtype_size_out: *mut usize,
) -> i32 {
    if seq_len_out.is_null() || num_heads_out.is_null() || 
       head_dim_out.is_null() || dtype_size_out.is_null() {
        return -1;
    }
    
    if let Ok(storage) = TENSOR_STORAGE.lock() {
        if let Some(metadata) = storage.get(&tensor_id) {
            unsafe {
                *seq_len_out = metadata.seq_len;
                *num_heads_out = metadata.num_heads;
                *head_dim_out = metadata.head_dim;
                *dtype_size_out = metadata.dtype_size;
            }
            return 0;
        }
    }
    -1
}

/// Copy data to tensor buffer with bounds checking
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
            // Bounds checking before copy
            if dst_offset + src_size <= metadata.host_buffer.len() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_data as *const u8,
                        metadata.host_buffer.as_mut_ptr().add(dst_offset),
                        src_size,
                    );
                }
                return 0;
            }
        }
    }
    -1
}

/// Validate tensor state for debugging
#[no_mangle]
pub extern "C" fn sequence_arena_validate_tensor(tensor_id: usize) -> i32 {
    if let Ok(storage) = TENSOR_STORAGE.lock() {
        if let Some(metadata) = storage.get(&tensor_id) {
            // Basic validation checks
            if metadata.seq_len > 0 && 
               metadata.num_heads > 0 && 
               metadata.head_dim > 0 && 
               metadata.dtype_size > 0 &&
               !metadata.host_buffer.is_empty() {
                return 1; // Valid
            }
        }
    }
    0 // Invalid or not found
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

/// Copy tensor data to device with alignment verification
#[no_mangle]
pub extern "C" fn prod_copy_tensor_to_device(
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

    // Verify pointers are properly aligned
    if (host_key_data as usize) % 8 != 0 || (host_value_data as usize) % 8 != 0 {
        log::error!("Host data pointers not properly aligned");
        return PROD_ERROR_INVALID_PARAM;
    }

    match tensor_ref.copy_new_tokens_from_host(
        host_key_data as *const u8,
        host_value_data as *const u8,
        start_token,
        num_tokens,
    ) {
        Ok(()) => PROD_SUCCESS,
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Get tensor device pointers with alignment verification
#[no_mangle]
pub extern "C" fn prod_get_tensor_device_ptrs(
    tensor: *mut CKVTensor,
    key_ptr_out: *mut *mut c_void,
    value_ptr_out: *mut *mut c_void,
) -> i32 {
    if tensor.is_null() || key_ptr_out.is_null() || value_ptr_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };
    
    let key_ptr = tensor_ref.key_device_ptr();
    let value_ptr = tensor_ref.value_device_ptr();

    // Verify alignment before returning pointers
    if (key_ptr as usize) % 256 != 0 || (value_ptr as usize) % 256 != 0 {
        log::error!("Device pointers not properly aligned: key={:p}, value={:p}", key_ptr, value_ptr);
        return PROD_ERROR_ALLOCATION_FAILED;
    }

    unsafe {
        *key_ptr_out = key_ptr;
        *value_ptr_out = value_ptr;
    }

    PROD_SUCCESS
}

/// Get tensor dimensions safely
#[no_mangle]
pub extern "C" fn prod_get_tensor_dimensions(
    tensor: *mut CKVTensor,
    seq_len_out: *mut usize,
    num_heads_out: *mut usize,
    head_dim_out: *mut usize,
) -> i32 {
    if tensor.is_null() || seq_len_out.is_null() || 
       num_heads_out.is_null() || head_dim_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };
    let (seq_len, num_heads, head_dim) = tensor_ref.dimensions();

    unsafe {
        *seq_len_out = seq_len;
        *num_heads_out = num_heads;
        *head_dim_out = head_dim;
    }

    PROD_SUCCESS
}

/// Synchronize tensor operations
#[no_mangle]
pub extern "C" fn prod_tensor_synchronize(tensor: *mut CKVTensor) -> i32 {
    if tensor.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };
    
    match tensor_ref.synchronize() {
        Ok(()) => PROD_SUCCESS,
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Check if tensor can extend to given length without copy
#[no_mangle]
pub extern "C" fn prod_tensor_can_extend_zero_copy(
    tensor: *mut CKVTensor,
    target_seq_len: usize,
) -> i32 {
    if tensor.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };
    
    if tensor_ref.can_extend_zero_copy_to(target_seq_len) {
        1 // Can extend
    } else {
        0 // Cannot extend
    }
}

/// Get tensor memory usage
#[no_mangle]
pub extern "C" fn prod_get_tensor_memory_usage(
    tensor: *mut CKVTensor,
    current_bytes_out: *mut usize,
    max_bytes_out: *mut usize,
    utilization_out: *mut f64,
) -> i32 {
    if tensor.is_null() || current_bytes_out.is_null() || 
       max_bytes_out.is_null() || utilization_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };

    unsafe {
        *current_bytes_out = tensor_ref.size_bytes();
        *max_bytes_out = tensor_ref.max_allocated_size_bytes();
        *utilization_out = tensor_ref.utilization();
    }

    PROD_SUCCESS
}

/// Validate tensor integrity
#[no_mangle]
pub extern "C" fn prod_validate_tensor_integrity(tensor: *mut CKVTensor) -> i32 {
    if tensor.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };
    
    // Check basic tensor state
    if tensor_ref.seq_len() == 0 || tensor_ref.max_seq_len() == 0 {
        return 0; // Invalid
    }
    
    if tensor_ref.seq_len() > tensor_ref.max_seq_len() {
        return 0; // Invalid
    }
    
    // Check pointer alignment
    let key_ptr = tensor_ref.key_device_ptr();
    let value_ptr = tensor_ref.value_device_ptr();
    
    if (key_ptr as usize) % 256 != 0 || (value_ptr as usize) % 256 != 0 {
        return 0; // Invalid alignment
    }
    
    1 // Valid
}

/// Reset tensor to initial state (for reuse)
#[no_mangle]
pub extern "C" fn prod_reset_tensor(
    tensor: *mut CKVTensor,
    new_seq_len: usize,
) -> i32 {
    if tensor.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };
    
    // Check if new length is within bounds
    if new_seq_len > tensor_ref.max_seq_len() {
        return PROD_ERROR_INVALID_PARAM;
    }
    
    // Reset to new length (this is essentially a zero-copy "shrink")
    match tensor_ref.extend_zero_copy(new_seq_len) {
        Ok(_) => PROD_SUCCESS,
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Get tensor device ID
#[no_mangle]
pub extern "C" fn prod_get_tensor_device_id(tensor: *mut CKVTensor) -> i32 {
    if tensor.is_null() {
        return -1;
    }

    let tensor_ref = unsafe { &(*tensor).0 };
    tensor_ref.device_id()
}

/// Copy partial tensor data (for incremental updates)
#[no_mangle]
pub extern "C" fn prod_copy_partial_tensor_data(
    tensor: *mut CKVTensor,
    host_key_data: *const c_void,
    host_value_data: *const c_void,
    start_token: usize,
    num_tokens: usize,
    verify_bounds: i32,
) -> i32 {
    if tensor.is_null() || host_key_data.is_null() || host_value_data.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };

    // Optional bounds verification
    if verify_bounds != 0 {
        if start_token + num_tokens > tensor_ref.seq_len() {
            log::error!("Copy bounds exceed tensor length: {} + {} > {}", 
                       start_token, num_tokens, tensor_ref.seq_len());
            return PROD_ERROR_INVALID_PARAM;
        }
    }

    match tensor_ref.copy_new_tokens_only(
        host_key_data as *const u8,
        host_value_data as *const u8,
        start_token,
        num_tokens,
    ) {
        Ok(()) => PROD_SUCCESS,
        Err(_) => PROD_ERROR_ALLOCATION_FAILED,
    }
}

/// Batch tensor operations for efficiency - removed duplicate, use the one in utils.rs

/// Get comprehensive tensor diagnostics
#[no_mangle]
pub extern "C" fn prod_get_tensor_diagnostics(
    tensor: *mut CKVTensor,
    diagnostics_out: *mut CTensorDiagnostics,
) -> i32 {
    if tensor.is_null() || diagnostics_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let tensor_ref = unsafe { &(*tensor).0 };
    let stats = tensor_ref.zero_copy_stats();
    let (seq_len, num_heads, head_dim) = tensor_ref.dimensions();

    unsafe {
        (*diagnostics_out).seq_len = seq_len;
        (*diagnostics_out).max_seq_len = stats.max_seq_len;
        (*diagnostics_out).num_heads = num_heads;
        (*diagnostics_out).head_dim = head_dim;
        (*diagnostics_out).current_bytes = tensor_ref.size_bytes();
        (*diagnostics_out).max_bytes = tensor_ref.max_allocated_size_bytes();
        (*diagnostics_out).utilization = stats.utilization;
        (*diagnostics_out).memory_efficiency = stats.memory_efficiency;
        (*diagnostics_out).can_grow = if stats.can_grow_without_copy { 1 } else { 0 };
        (*diagnostics_out).device_id = tensor_ref.device_id();
        (*diagnostics_out).arena_id = tensor_ref.arena_id();
        (*diagnostics_out).key_ptr_aligned = if (tensor_ref.key_device_ptr() as usize) % 256 == 0 { 1 } else { 0 };
        (*diagnostics_out).value_ptr_aligned = if (tensor_ref.value_device_ptr() as usize) % 256 == 0 { 1 } else { 0 };
    }

    PROD_SUCCESS
}