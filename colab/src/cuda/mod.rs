// src/cuda/mod.rs - Updated CUDA module with safe initialization

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
pub mod init;  // Add the new safe initialization module

// Re-export commonly used types and functions
pub use bindings::*;
pub use error::{CudaError, check_cuda_error};
pub use device::{CudaDeviceInfo, CudaDeviceStats, detect_cuda_devices, check_device_health};
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
    safe_cuda_call,
};
pub use init::{
    safe_cuda_init,
    create_safe_memory_manager,
    create_safe_cuda_context,
    get_cuda_init_info,
    CudaInitResult,
};

// Global allocation tracking
pub static NEXT_ALLOCATION_ID: std::sync::atomic::AtomicUsize = 
    std::sync::atomic::AtomicUsize::new(1);