// src/ffi/types.rs - Updated types with honest zero-copy reporting
use std::ffi::c_void;
use std::collections::HashMap;
use std::sync::Mutex;

/// Opaque pointers for C FFI (unchanged)
pub struct CProductionManager(pub(crate) crate::ProductionKVCacheManager);
pub struct CSequenceArena(pub(crate) crate::zero_copy::ZeroCopyArena);
pub struct CKVTensor(pub(crate) crate::zero_copy::ZeroCopyTensor);

/// Error codes for production API (unchanged)
pub const PROD_SUCCESS: i32 = 0;
pub const PROD_ERROR_CUDA: i32 = -1;
pub const PROD_ERROR_OUT_OF_MEMORY: i32 = -2;
pub const PROD_ERROR_INVALID_DEVICE: i32 = -3;
pub const PROD_ERROR_ALLOCATION_FAILED: i32 = -4;
pub const PROD_ERROR_INVALID_PARAM: i32 = -5;

/// Extension result type constants
pub const EXTENSION_PURE_ZERO_COPY: i32 = 1;
pub const EXTENSION_REQUIRES_DATA_COPY: i32 = 2;
pub const EXTENSION_CANNOT_EXTEND: i32 = 3;

/// Data copy operation type constants
pub const DATA_COPY_NEW_TOKENS: i32 = 1;
pub const DATA_COPY_FULL_TENSOR: i32 = 2;
pub const DATA_COPY_INCREMENTAL: i32 = 3;

/// C-compatible extension result that clearly distinguishes zero-copy from data-copy
#[repr(C)]
pub struct CExtensionResult {
    pub result_type: i32,              // EXTENSION_* constants
    pub old_seq_len: usize,
    pub new_seq_len: usize,
    pub operation_time_ns: u64,
    pub requires_data_copy: i32,       // 0 = no, 1 = yes
    pub copy_size_bytes: usize,        // 0 if no copy required
}

/// C-compatible data copy statistics (explicitly NOT zero-copy)
#[repr(C)]
pub struct CDataCopyStats {
    pub operation_type: i32,           // DATA_COPY_* constants
    pub bytes_copied: usize,
    pub copy_time_ns: u64,
    pub bandwidth_gbps: f64,
}

/// C-compatible operation report that separates zero-copy from data-copy phases
#[repr(C)]
pub struct CExtensionOperationReport {
    pub metadata_extension_time_ns: u64,  // TRUE zero-copy phase
    pub data_copy_time_ns: u64,            // NOT zero-copy phase (may be 0)
    pub total_time_ns: u64,
    pub was_pure_zero_copy: i32,           // 1 = no data copy needed, 0 = data copy required
    pub bytes_copied: usize,               // 0 if pure zero-copy
    pub tokens_added: usize,
    pub bandwidth_gbps: f64,               // 0.0 if no copy performed
}

/// Honest zero-copy statistics that clearly separate metadata from data operations
#[repr(C)]
pub struct CHonestZeroCopyStats {
    // Basic tensor info
    pub current_seq_len: usize,
    pub max_seq_len: usize,
    pub growth_capacity_remaining: usize,
    pub utilization: f64,
    pub memory_efficiency: f64,
    pub can_grow_without_copy: i32,
    
    // HONEST reporting of what is and isn't zero-copy
    pub metadata_operations_are_zero_copy: i32,     // Always 1 (true)
    pub data_operations_require_copy: i32,          // Always 1 (true) 
    pub last_extension_was_pure_zero_copy: i32,     // 1=yes, 0=no, -1=unknown
}

/// Benchmark result that separates metadata operations from data copy operations
#[repr(C)]
pub struct CSeparatedOperationsBenchmark {
    pub total_extensions: usize,
    pub pure_zero_copy_extensions: usize,      // Only metadata updated
    pub data_copy_required_extensions: usize,  // Metadata + data copy needed
    pub total_metadata_time_ns: u64,           // TRUE zero-copy time
    pub total_data_copy_time_ns: u64,          // NOT zero-copy time
    pub avg_metadata_time_ns: u64,             // Average zero-copy operation time
    pub avg_data_copy_time_ns: u64,            // Average data copy time
    pub metadata_efficiency_ratio: f64,        // How much faster metadata ops are
    pub total_bytes_would_copy: usize,         // Total data that would need copying
}

/// Educational structure to explain what operations are actually zero-copy
#[repr(C)]
pub struct CZeroCopyExplanation {
    // What IS zero-copy
    pub metadata_updates_are_zero_copy: i32,           // Atomic seq_len updates
    pub pre_allocated_data_access_is_zero_copy: i32,   // Accessing existing data
    pub atomic_operations_are_zero_copy: i32,          // Atomic loads/stores
    pub pointer_arithmetic_is_zero_copy: i32,          // Computing offsets
    
    // What is NOT zero-copy
    pub new_data_requires_copy: i32,                   // cudaMemcpy for new tokens
    pub cuda_memcpy_is_not_zero_copy: i32,            // Explicit data movement
    
    // Performance characteristics
    pub atomic_update_time_ns: u64,                    // ~10ns for atomic operation
    pub cuda_memcpy_time_per_kb_ns: u64,              // ~1000ns per KB
    pub speedup_ratio: f64,                            // How much faster zero-copy is
}

/// Updated zero-copy stats (legacy compatibility with honest naming)
#[repr(C)]
pub struct CZeroCopyStats {
    pub current_seq_len: usize,
    pub max_seq_len: usize,
    pub growth_capacity_remaining: usize,
    pub utilization: f64,
    pub memory_efficiency: f64,
    pub can_grow_without_copy: i32,
    // Note: This structure name is kept for compatibility but should be considered
    // to represent "metadata zero-copy" stats, not "data zero-copy" stats
}

/// Batch request structure (unchanged)
#[repr(C)]
pub struct CBatchRequest {
    pub initial_seq_len: usize,
    pub max_seq_len: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub dtype_size: usize,
    pub preferred_device: i32,
}

/// Batch result structure (unchanged)
#[repr(C)]
pub struct CBatchResult {
    pub request_id: usize,
    pub arena: *mut CSequenceArena,
    pub tensor: *mut CKVTensor,
    pub device_id: i32,
}

/// Updated benchmark configuration with honest expectations
#[repr(C)]
pub struct CBenchmarkConfig {
    pub initial_seq_len: usize,
    pub max_seq_len: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub dtype_size: usize,
    pub num_extensions: usize,
    pub tokens_per_extension: usize,
    pub expect_data_copies: i32,       // NEW: 1 if expecting data copies, 0 if pre-filled
}

/// Updated benchmark result with separated timings
#[repr(C)]
pub struct CBenchmarkResult {
    pub config_id: usize,
    pub total_time_ms: f64,
    pub metadata_only_extensions: usize,      // TRUE zero-copy operations
    pub data_copy_extensions: usize,          // Operations requiring data copy
    pub zero_copy_rate: f64,                  // Fraction that were metadata-only
    pub avg_metadata_time_ns: u64,            // Average time for metadata updates
    pub avg_data_copy_time_ns: u64,           // Average time for data copies
    pub memory_efficiency: f64,
    pub honest_zero_copy_rate: f64,           // Same as zero_copy_rate but honest naming
}

/// Performance comparison result
#[repr(C)]
pub struct CPerformanceComparison {
    pub metadata_operations_per_second: f64,
    pub data_copy_operations_per_second: f64,
    pub metadata_speedup_factor: f64,          // How much faster metadata ops are
    pub effective_zero_copy_benefit: f64,      // Actual performance benefit
}

/// Efficiency report that separates different operation types
#[repr(C)]
pub struct CZeroCopyEfficiencyReport {
    pub initial_seq_len: usize,
    pub final_seq_len: usize,
    
    // Separated operation counts
    pub total_extensions: usize,
    pub pure_metadata_extensions: usize,       // TRUE zero-copy
    pub data_copy_extensions: usize,           // Required data movement
    
    // Separated timing
    pub pure_zero_copy_time_ns: u64,           // Time for metadata-only operations
    pub data_copy_time_ns: u64,                // Time for actual data movement
    pub total_time_ns: u64,
    
    // Honest efficiency metrics
    pub metadata_efficiency: f64,              // How efficiently we update metadata
    pub data_copy_efficiency: f64,             // How efficiently we copy data
    pub overall_efficiency: f64,               // Combined efficiency
    
    // What would happen with naive copying
    pub naive_copy_time_estimate_ns: u64,      // If we copied everything every time
    pub actual_speedup_achieved: f64,          // Real performance improvement
}

/// Validation report for zero-copy implementation correctness
#[repr(C)]
pub struct CZeroCopyValidationReport {
    // Test results
    pub metadata_updates_work: i32,                        // Basic functionality
    pub beyond_capacity_handled_correctly: i32,           // Error handling
    pub atomic_operations_are_thread_safe: i32,           // Concurrency safety
    pub no_data_corruption_in_extensions: i32,            // Data integrity
    
    // Performance validation
    pub metadata_ops_faster_than_copies: i32,             // Performance benefit exists
    pub timing_measurements_consistent: i32,               // Benchmark reliability
    
    // Honest reporting validation
    pub distinguishes_metadata_from_data_ops: i32,        // Clear separation
    pub reports_data_copy_requirements_accurately: i32,   // Honest about limitations
    
    pub all_tests_passed: i32,
}

/// Memory layout information for debugging
#[repr(C)]
pub struct CMemoryLayoutInfo {
    pub key_tensor_offset: usize,
    pub value_tensor_offset: usize,
    pub current_data_size: usize,
    pub allocated_data_size: usize,
    pub wasted_space: usize,
    pub alignment_padding: usize,
    pub is_properly_aligned: i32,
}

/// Simplified tensor metadata for safe FFI operations (unchanged)
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub dtype_size: usize,
    pub host_buffer: Vec<u8>,
}

/// Global storage for tensor metadata (unchanged)
lazy_static::lazy_static! {
    pub static ref TENSOR_STORAGE: Mutex<HashMap<usize, TensorMetadata>> = 
        Mutex::new(HashMap::new());
    pub static ref NEXT_TENSOR_ID: std::sync::atomic::AtomicUsize = 
        std::sync::atomic::AtomicUsize::new(1);
}