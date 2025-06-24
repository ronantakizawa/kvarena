// src/cuda/mod.rs - Main CUDA module with re-exports

pub mod bindings;
pub mod error;
pub mod device;
pub mod memory;
pub mod stream;
pub mod context;
pub mod allocator;
pub mod tensor;
pub mod raw;
pub mod diagnostics;

// Re-export commonly used types and functions
pub use bindings::*;
pub use error::{CudaError, check_cuda_error};
pub use device::{CudaDeviceInfo, CudaDeviceStats};
pub use memory::{CudaMemoryManager, CudaMemoryPool};
pub use stream::CudaStream;
pub use context::CudaContext;
pub use allocator::{BumpAllocator, CudaPage};
pub use tensor::CudaTensor;
pub use diagnostics::{
    verify_cuda_runtime_linked,
    cuda_runtime_health_check,
    check_cuda_environment,
    initialize_cuda,
    is_cuda_available,
    diagnose_cuda_issues,
    cuda_memory_test,
    get_cuda_version,
};

// Global allocation tracking
pub static NEXT_ALLOCATION_ID: std::sync::atomic::AtomicUsize = 
    std::sync::atomic::AtomicUsize::new(1);