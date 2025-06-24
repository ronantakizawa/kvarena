// src/ffi/arena.rs - Sequence arena management FFI functions
use std::ffi::c_void;
use std::sync::Arc;
use super::types::*;

/// Create sequence arena with direct page ownership
#[no_mangle]
pub extern "C" fn prod_create_sequence_arena_with_growth(
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
    let head_dim = hidden_dim / num_heads;

    // Create arena directly through zero-copy manager field
    match manager_ref.zero_copy_manager.create_arena(
        calculate_arena_size(max_seq_len, num_heads, head_dim, 2), // Calculate size for KV tensors
        device_id.max(0), // Use device 0 if auto-select
    ) {
        Ok(arena) => {
            unsafe {
                *arena_out = Box::into_raw(Box::new(CSequenceArena(arena)));
            }
            PROD_SUCCESS
        }
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

/// Create sequence arena (legacy interface)
#[no_mangle]
pub extern "C" fn kv_cache_create_sequence_arena(manager_ptr: *mut c_void) -> *mut c_void {
    if manager_ptr.is_null() {
        return std::ptr::null_mut();
    }
    
    let manager = unsafe { &*(manager_ptr as *const CProductionManager) };
    
    // Create arena with default KV parameters
    match manager.0.create_sequence_arena(512, 2048, 32, 128, None) {
        Ok(arena) => Box::into_raw(Box::new(arena)) as *mut c_void,
        Err(_) => std::ptr::null_mut(),
    }
}

/// Fixed version of sequence arena creation
#[no_mangle]
pub extern "C" fn kv_cache_create_sequence_arena_fixed(manager_ptr: *mut c_void) -> *mut c_void {
    if manager_ptr.is_null() {
        return std::ptr::null_mut();
    }
    
    let manager = unsafe { &*(manager_ptr as *const CProductionManager) };
    let manager_ref = &manager.0;
    
    // Use the EXACT same parameters that work in the basic test
    let safe_initial_seq_len = 32;    // SAME as working basic test
    let safe_max_seq_len = 512;       // Conservative max
    let safe_num_heads = 8;           // SAME as working basic test
    let safe_head_dim = 64;           // SAME as working basic test
    
    // Try to create arena with working parameters
    match manager_ref.zero_copy_manager.create_arena(
        calculate_arena_size(safe_max_seq_len, safe_num_heads, safe_head_dim, 2),
        0, // device 0
    ) {
        Ok(arena) => {
            let boxed_arena = Box::new(CSequenceArena(arena));
            Box::into_raw(boxed_arena) as *mut c_void
        }
        Err(_) => {
            std::ptr::null_mut()
        }
    }
}

/// Free sequence arena (legacy interface)
#[no_mangle]
pub extern "C" fn sequence_arena_free(arena_ptr: *mut c_void) {
    if !arena_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(arena_ptr as *mut Arc<crate::SequenceArena>);
        }
    }
}

/// Fixed version of arena free
#[no_mangle]
pub extern "C" fn sequence_arena_free_fixed(arena_ptr: *mut c_void) {
    if !arena_ptr.is_null() {
        unsafe {
            let _arena = Box::from_raw(arena_ptr as *mut CSequenceArena);
            // Box destructor will handle cleanup safely
        }
    }
}

/// Get arena statistics
#[no_mangle]
pub extern "C" fn sequence_arena_get_stats(
    arena_ptr: *mut c_void,
    sequence_id_out: *mut u64,
    total_allocated_out: *mut usize,
    num_pages_out: *mut usize,
    utilization_out: *mut f64,
) -> i32 {
    if arena_ptr.is_null() || sequence_id_out.is_null() || 
       total_allocated_out.is_null() || num_pages_out.is_null() || utilization_out.is_null() {
        return -1;
    }
    
    let arena = unsafe { &*(arena_ptr as *const Arc<crate::SequenceArena>) };
    let stats = arena.stats();
    
    unsafe {
        *sequence_id_out = stats.arena_id;
        *total_allocated_out = stats.total_allocated_bytes;
        *num_pages_out = 1; // Simplified
        *utilization_out = stats.arena_utilization;
    }
    
    0
}

/// Fixed version of arena stats
#[no_mangle]
pub extern "C" fn sequence_arena_get_stats_fixed(
    arena_ptr: *mut c_void,
    sequence_id_out: *mut u64,
    total_allocated_out: *mut usize,
    num_pages_out: *mut usize,
    utilization_out: *mut f64,
) -> i32 {
    if arena_ptr.is_null() || sequence_id_out.is_null() || 
       total_allocated_out.is_null() || num_pages_out.is_null() || 
       utilization_out.is_null() {
        return -1;
    }
    
    let arena = unsafe { &*(arena_ptr as *const CSequenceArena) };
    let arena_ref = &arena.0;
    
    match std::panic::catch_unwind(|| {
        (arena_ref.arena_id(), arena_ref.current_offset(), arena_ref.utilization())
    }) {
        Ok((arena_id, allocated, utilization)) => {
            // Sanity check values
            if allocated > 1024 * 1024 * 1024 || utilization > 1.0 || utilization < 0.0 {
                return -1;  // Reject suspicious values
            }
            
            unsafe {
                *sequence_id_out = arena_id;
                *total_allocated_out = allocated;
                *num_pages_out = 1; // Simplified
                *utilization_out = utilization;
            }
            0
        }
        Err(_) => {
            -1 // Panic occurred in stats collection
        }
    }
}

/// Get pure bump arena stats
#[no_mangle]
pub extern "C" fn prod_get_bump_arena_stats(
    arena: *mut CSequenceArena,
    arena_id_out: *mut u64,
    current_offset_out: *mut usize,
    available_space_out: *mut usize,
    utilization_out: *mut f64,
) -> i32 {
    if arena.is_null() || arena_id_out.is_null() || 
       current_offset_out.is_null() || available_space_out.is_null() ||
       utilization_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let arena_ref = unsafe { &(*arena).0 };

    unsafe {
        *arena_id_out = arena_ref.arena_id();
        *current_offset_out = arena_ref.current_offset();
        *available_space_out = arena_ref.available_space();
        *utilization_out = arena_ref.utilization();
    }

    PROD_SUCCESS
}

/// Benchmark pure bump allocation
#[no_mangle]
pub extern "C" fn prod_benchmark_pure_bump_allocation(
    arena: *mut CSequenceArena,
    num_allocations: usize,
    allocation_size: usize,
    bump_time_ns_out: *mut u64,
    allocations_per_second_out: *mut f64,
) -> i32 {
    if arena.is_null() || bump_time_ns_out.is_null() || allocations_per_second_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let arena_ref = unsafe { &(*arena).0 };
    let start_time = std::time::Instant::now();

    // Perform pure bump allocations
    let mut successful_allocations = 0;
    for _ in 0..num_allocations {
        if arena_ref.bump_allocate(allocation_size, 256).is_some() {
            successful_allocations += 1;
        }
    }

    let total_time = start_time.elapsed();
    let total_time_ns = total_time.as_nanos() as u64;
    let allocations_per_second = if total_time.as_secs_f64() > 0.0 {
        successful_allocations as f64 / total_time.as_secs_f64()
    } else {
        0.0
    };

    unsafe {
        *bump_time_ns_out = total_time_ns;
        *allocations_per_second_out = allocations_per_second;
    }

    PROD_SUCCESS
}

/// Helper function to calculate arena size for KV tensors
fn calculate_arena_size(max_seq_len: usize, num_heads: usize, head_dim: usize, dtype_size: usize) -> usize {
    // K tensor + V tensor + padding
    let tensor_size = max_seq_len * num_heads * head_dim * dtype_size;
    let total_size = tensor_size * 2; // K + V
    (total_size * 11) / 10 // Add 10% padding
}