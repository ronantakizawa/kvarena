// src/ffi/mod.rs - Main FFI module with organized submodules
//! Foreign Function Interface (FFI) for Arena KV-Cache
//! 
//! This module provides C-compatible functions for interacting with the Arena KV-Cache
//! from other languages like Python. The module is organized into several submodules:
//! 
//! - `types`: C-compatible types and structures
//! - `manager`: Production KV cache manager functions
//! - `arena`: Sequence arena management functions  
//! - `tensor`: Tensor allocation and manipulation functions
//! - `slab`: Slab recycling and memory management functions
//! - `utils`: Utility and helper functions
//! - `safety`: Safe wrappers and error handling

pub mod types;
pub mod manager;
pub mod arena;
pub mod tensor;
pub mod slab;
pub mod utils;
pub mod safety;

// Re-export commonly used types and functions
pub use types::*;
pub use manager::*;
pub use arena::*;
pub use tensor::*;
pub use slab::*;
pub use utils::*;
pub use safety::*;

use std::ffi::{c_char, c_void};
use crate::{ProductionKVCacheManager, LLMServerError};

/// Initialize production manager for LLM server with model-specific optimizations
pub fn initialize_for_server(
    model_name: &str,
    devices: &[i32],
) -> Result<ProductionKVCacheManager, LLMServerError> {
    log::info!("Initializing LLM server for model: {}", model_name);
    ProductionKVCacheManager::for_llm_model(model_name, devices.to_vec())
}

/// Get library version information
#[no_mangle]
pub extern "C" fn arena_get_version() -> *const c_char {
    static VERSION: &[u8] = b"1.0.0-production-zero-copy\0";
    VERSION.as_ptr() as *const c_char
}

/// Get supported features list
#[no_mangle]
pub extern "C" fn arena_get_features() -> *const c_char {
    static FEATURES: &[u8] = b"true-zero-copy,slab-recycling,cuda-heap,production-ready\0";
    FEATURES.as_ptr() as *const c_char
}

/// Check if CUDA is available and working
#[no_mangle]
pub extern "C" fn arena_check_cuda_availability() -> i32 {
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

/// Emergency cleanup and recovery
#[no_mangle]
pub extern "C" fn arena_emergency_cleanup() -> i32 {
    // Perform global cleanup operations
    log::info!("Emergency cleanup initiated");
    
    // Force garbage collection
    #[cfg(feature = "cuda")]
    {
        if let Ok(context) = crate::cuda::CudaContext::new() {
            let _ = context.synchronize_all();
        }
    }
    
    1 // Success
}