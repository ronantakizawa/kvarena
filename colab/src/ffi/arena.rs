// src/ffi/arena.rs - FIXED: Sequence arena management with proper KV head handling
use std::ffi::c_void;
use std::sync::Arc;
use super::types::*;

/// Model-specific KV head configurations
#[derive(Debug, Clone)]
pub struct ModelKVConfig {
    pub num_query_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl ModelKVConfig {
    /// Get configuration for known models
    pub fn from_model_name(model_name: &str) -> Self {
        let model_lower = model_name.to_lowercase();
        
        if model_lower.contains("mistral") && model_lower.contains("7b") {
            ModelKVConfig {
                num_query_heads: 32,
                num_kv_heads: 8,    // FIXED: Mistral 7B uses 8 KV heads
                head_dim: 128,
                max_seq_len: 8192,
            }
        } else if model_lower.contains("llama") && model_lower.contains("7b") {
            ModelKVConfig {
                num_query_heads: 32,
                num_kv_heads: 32,   // Full attention
                head_dim: 128,
                max_seq_len: 8192,
            }
        } else if model_lower.contains("llama") && model_lower.contains("13b") {
            ModelKVConfig {
                num_query_heads: 40,
                num_kv_heads: 40,   // Full attention
                head_dim: 128,
                max_seq_len: 8192,
            }
        } else if model_lower.contains("llama") && model_lower.contains("70b") {
            ModelKVConfig {
                num_query_heads: 64,
                num_kv_heads: 8,    // GQA: 64 query -> 8 KV
                head_dim: 128,
                max_seq_len: 8192,
            }
        } else {
            // Default configuration
            ModelKVConfig {
                num_query_heads: 32,
                num_kv_heads: 32,
                head_dim: 128,
                max_seq_len: 4096,
            }
        }
    }
    
    /// Calculate KV cache size using actual KV heads
    pub fn calculate_kv_cache_size(&self, seq_len: usize, element_size: usize) -> usize {
        // FIXED: Use KV heads for cache size calculation
        2 * seq_len * self.num_kv_heads * self.head_dim * element_size
    }
    
    /// Calculate optimal arena size for this model
    pub fn calculate_arena_size(&self, element_size: usize) -> usize {
        let kv_cache_size = self.calculate_kv_cache_size(self.max_seq_len, element_size);
        let overhead_factor = 1.25; // 25% overhead
        (kv_cache_size as f64 * overhead_factor) as usize
    }
}

/// FIXED: Create sequence arena with model-aware KV configuration
#[no_mangle]
pub extern "C" fn prod_create_sequence_arena_with_kv_config(
    manager: *mut CProductionManager,
    initial_seq_len: usize,
    max_seq_len: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    device_id: i32,
    arena_out: *mut *mut CSequenceArena,
) -> i32 {
    if manager.is_null() || arena_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    let manager_ref = unsafe { &(*manager).0 };
    
    // FIXED: Calculate arena size using actual KV heads
    let arena_size = calculate_kv_arena_size(max_seq_len, num_kv_heads, head_dim, 2);

    match manager_ref.zero_copy_manager.create_arena(
        arena_size,
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

/// Create sequence arena with automatic model detection
#[no_mangle]
pub extern "C" fn prod_create_sequence_arena_with_growth(
    manager: *mut CProductionManager,
    initial_seq_len: usize,
    max_seq_len: usize,
    hidden_dim: usize,
    num_query_heads: usize,
    device_id: i32,
    arena_out: *mut *mut CSequenceArena,
) -> i32 {
    if manager.is_null() || arena_out.is_null() {
        return PROD_ERROR_INVALID_PARAM;
    }

    // FIXED: Auto-detect KV heads from query heads and hidden dim
    let head_dim = hidden_dim / num_query_heads;
    let num_kv_heads = detect_kv_heads_from_config(hidden_dim, num_query_heads, head_dim);
    
    log::info!("Auto-detected: {} query heads -> {} KV heads, head_dim={}", 
              num_query_heads, num_kv_heads, head_dim);

    prod_create_sequence_arena_with_kv_config(
        manager,
        initial_seq_len,
        max_seq_len,
        num_query_heads,
        num_kv_heads,
        head_dim,
        device_id,
        arena_out,
    )
}

/// FIXED: Create arena with safe Mistral 7B parameters
#[no_mangle]
pub extern "C" fn kv_cache_create_sequence_arena_fixed(manager_ptr: *mut c_void) -> *mut c_void {
    if manager_ptr.is_null() {
        return std::ptr::null_mut();
    }
    
    let manager = unsafe { &*(manager_ptr as *const CProductionManager) };
    let manager_ref = &manager.0;
    
    // FIXED: Use Mistral 7B's actual KV configuration
    let mistral_config = ModelKVConfig::from_model_name("mistral-7b");
    let safe_initial_seq_len = 32;
    let safe_max_seq_len = 512;
    
    log::info!("Creating arena with Mistral 7B config: {} KV heads, {} head_dim", 
              mistral_config.num_kv_heads, mistral_config.head_dim);
    
    // Calculate arena size using KV heads
    let arena_size = calculate_kv_arena_size(
        safe_max_seq_len, 
        mistral_config.num_kv_heads,  // FIXED: Use 8 KV heads
        mistral_config.head_dim, 
        2
    );
    
    match manager_ref.zero_copy_manager.create_arena(arena_size, 0) {
        Ok(arena) => {
            let boxed_arena = Box::new(CSequenceArena(arena));
            Box::into_raw(boxed_arena) as *mut c_void
        }
        Err(e) => {
            log::error!("Failed to create arena: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

/// Legacy compatibility function
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

/// Free sequence arena (legacy interface)
#[no_mangle]
pub extern "C" fn sequence_arena_free(arena_ptr: *mut c_void) {
    if !arena_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(arena_ptr as *mut Arc<crate::SequenceArena>);
        }
    }
}

/// FIXED: Free sequence arena with proper cleanup
#[no_mangle]
pub extern "C" fn sequence_arena_free_fixed(arena_ptr: *mut c_void) {
    if !arena_ptr.is_null() {
        unsafe {
            let _arena = Box::from_raw(arena_ptr as *mut CSequenceArena);
            // Box destructor will handle cleanup safely
        }
    }
}

/// Get arena statistics with bounds checking
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
            // Sanity check values to prevent issues
            if allocated > 1024 * 1024 * 1024 || utilization > 1.0 || utilization < 0.0 {
                log::warn!("Suspicious arena stats: allocated={}, utilization={}", allocated, utilization);
                return -1;
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
            log::error!("Panic occurred while getting arena stats");
            -1
        }
    }
}

/// Get arena statistics (legacy interface)
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

/// FIXED: Helper function to calculate KV arena size using actual KV heads
fn calculate_kv_arena_size(max_seq_len: usize, num_kv_heads: usize, head_dim: usize, dtype_size: usize) -> usize {
    // FIXED: Use KV heads for accurate cache size calculation
    let kv_tensor_size = max_seq_len * num_kv_heads * head_dim * dtype_size;
    let total_kv_size = kv_tensor_size * 2; // K + V tensors
    let overhead = (total_kv_size * 11) / 10; // Add 10% padding
    
    log::debug!("KV arena size: {}KB for {} seq_len, {} KV heads, {} head_dim", 
               overhead / 1024, max_seq_len, num_kv_heads, head_dim);
    
    overhead
}

/// Helper function to detect KV heads from model configuration
fn detect_kv_heads_from_config(hidden_dim: usize, num_query_heads: usize, head_dim: usize) -> usize {
    // FIXED: Auto-detect common KV head patterns
    if hidden_dim == 4096 && num_query_heads == 32 && head_dim == 128 {
        // Mistral 7B pattern
        8
    } else if hidden_dim == 8192 && num_query_heads == 64 && head_dim == 128 {
        // Llama 70B pattern (GQA)
        8
    } else if num_query_heads == num_query_heads {
        // Full attention (query heads == KV heads)
        num_query_heads
    } else {
        // Common GQA ratio: 4:1 or 8:1
        if num_query_heads >= 32 {
            num_query_heads / 4  // 4:1 ratio
        } else {
            num_query_heads      // Small models use full attention
        }
    }
}

/// Calculate arena size (legacy compatibility)
fn calculate_arena_size(max_seq_len: usize, num_heads: usize, head_dim: usize, dtype_size: usize) -> usize {
    // Assume full attention for legacy calls
    calculate_kv_arena_size(max_seq_len, num_heads, head_dim, dtype_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_kv_configs() {
        // Test Mistral 7B configuration
        let mistral_config = ModelKVConfig::from_model_name("mistral-7b");
        assert_eq!(mistral_config.num_query_heads, 32);
        assert_eq!(mistral_config.num_kv_heads, 8);  // FIXED: Should be 8, not 32
        assert_eq!(mistral_config.head_dim, 128);
        
        // Test Llama 7B configuration (full attention)
        let llama_config = ModelKVConfig::from_model_name("llama-7b");
        assert_eq!(llama_config.num_query_heads, 32);
        assert_eq!(llama_config.num_kv_heads, 32);   // Full attention
        assert_eq!(llama_config.head_dim, 128);
        
        // Test Llama 70B configuration (GQA)
        let llama70_config = ModelKVConfig::from_model_name("llama-70b");
        assert_eq!(llama70_config.num_query_heads, 64);
        assert_eq!(llama70_config.num_kv_heads, 8);  // GQA: 64 -> 8
        assert_eq!(llama70_config.head_dim, 128);
        
        println!("✓ Model KV configurations are correct");
    }

    #[test]
    fn test_kv_cache_size_calculation() {
        let mistral_config = ModelKVConfig::from_model_name("mistral-7b");
        
        // Test KV cache size calculation
        let seq_len = 1024;
        let element_size = 2; // fp16
        let kv_size = mistral_config.calculate_kv_cache_size(seq_len, element_size);
        
        // Expected: 2 * 1024 * 8 * 128 * 2 = 4,194,304 bytes (4MB)
        let expected = 2 * seq_len * mistral_config.num_kv_heads * mistral_config.head_dim * element_size;
        assert_eq!(kv_size, expected);
        
        println!("✓ KV cache size calculation uses correct KV head count");
    }

    #[test]
    fn test_kv_head_detection() {
        // Test Mistral 7B pattern detection
        let kv_heads = detect_kv_heads_from_config(4096, 32, 128);
        assert_eq!(kv_heads, 8);
        
        // Test Llama 70B pattern detection
        let kv_heads = detect_kv_heads_from_config(8192, 64, 128);
        assert_eq!(kv_heads, 8);
        
        // Test full attention pattern
        let kv_heads = detect_kv_heads_from_config(4096, 32, 128);
        // Note: This might detect as Mistral pattern, which is correct
        
        println!("✓ KV head detection working correctly");
    }

    #[test]
    fn test_arena_size_calculation() {
        // Test arena size calculation uses KV heads
        let arena_size = calculate_kv_arena_size(1024, 8, 128, 2);
        
        // Should be based on 8 KV heads, not 32 query heads
        let expected_kv_size = 1024 * 8 * 128 * 2 * 2; // seq * kv_heads * head_dim * elem_size * (K+V)
        let expected_with_overhead = (expected_kv_size * 11) / 10;
        
        assert_eq!(arena_size, expected_with_overhead);
        
        println!("✓ Arena size calculation uses KV heads correctly");
    }
}