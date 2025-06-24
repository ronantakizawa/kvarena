// src/ffi/types.rs - C-compatible types and constants for FFI
use std::ffi::c_void;
use std::collections::HashMap;
use std::sync::Mutex;

/// Opaque pointers for C FFI
pub struct CProductionManager(pub(crate) crate::ProductionKVCacheManager);
pub struct CSequenceArena(pub(crate) crate::zero_copy::ZeroCopyArena);
pub struct CKVTensor(pub(crate) crate::zero_copy::ZeroCopyTensor);

/// Error codes for production API
pub const PROD_SUCCESS: i32 = 0;
pub const PROD_ERROR_CUDA: i32 = -1;
pub const PROD_ERROR_OUT_OF_MEMORY: i32 = -2;
pub const PROD_ERROR_INVALID_DEVICE: i32 = -3;
pub const PROD_ERROR_ALLOCATION_FAILED: i32 = -4;
pub const PROD_ERROR_INVALID_PARAM: i32 = -5;

/// C-compatible batch request structure
#[repr(C)]
pub struct CBatchRequest {
    pub initial_seq_len: usize,
    pub max_seq_len: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub dtype_size: usize,
    pub preferred_device: i32, // -1 for auto-select
}

/// C-compatible batch result structure
#[repr(C)]
pub struct CBatchResult {
    pub request_id: usize,
    pub arena: *mut CSequenceArena,
    pub tensor: *mut CKVTensor,
    pub device_id: i32,
}

/// C-compatible zero-copy statistics
#[repr(C)]
pub struct CZeroCopyStats {
    pub current_seq_len: usize,
    pub max_seq_len: usize,
    pub growth_capacity_remaining: usize,
    pub utilization: f64,
    pub memory_efficiency: f64,
    pub can_grow_without_copy: i32,
}

/// C-compatible zero-copy efficiency report
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

/// C-compatible zero-copy validation report
#[repr(C)]
pub struct CZeroCopyValidationReport {
    pub basic_zero_copy_works: i32,
    pub beyond_capacity_handled_correctly: i32,
    pub memory_efficiency_reporting_ok: i32,
    pub capacity_reporting_accurate: i32,
    pub all_tests_passed: i32,
}

/// C-compatible benchmark configuration
#[repr(C)]
pub struct CBenchmarkConfig {
    pub initial_seq_len: usize,
    pub max_seq_len: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub dtype_size: usize,
    pub num_extensions: usize,
    pub tokens_per_extension: usize,
}

/// C-compatible benchmark result
#[repr(C)]
pub struct CBenchmarkResult {
    pub config_id: usize,
    pub total_time_ms: f64,
    pub zero_copy_extensions: usize,
    pub copy_extensions: usize,
    pub zero_copy_rate: f64,
    pub avg_extension_time_ns: u64,
    pub memory_efficiency: f64,
}

/// Simplified tensor metadata for safe FFI operations
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub dtype_size: usize,
    pub host_buffer: Vec<u8>,
}

/// Global storage for tensor metadata (thread-safe)
lazy_static::lazy_static! {
    pub static ref TENSOR_STORAGE: Mutex<HashMap<usize, TensorMetadata>> = 
        Mutex::new(HashMap::new());
    pub static ref NEXT_TENSOR_ID: std::sync::atomic::AtomicUsize = 
        std::sync::atomic::AtomicUsize::new(1);
}