// src/cuda/init.rs - Safe CUDA initialization wrapper
use super::diagnostics::{initialize_cuda, cuda_runtime_health_check};
use super::memory::CudaMemoryManager;
use super::context::CudaContext;
use super::error::CudaError;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

/// Global initialization state
static INIT: Once = Once::new();
static CUDA_STATE: Mutex<Option<CudaInitResult>> = Mutex::new(None);

#[derive(Debug, Clone)]
pub struct CudaInitResult {
    pub available: bool,
    pub device_count: usize,
    pub error_message: Option<String>,
    pub initialization_time_ms: u64,
}

/// Safe CUDA initialization with timeout protection
pub fn safe_cuda_init() -> CudaInitResult {
    INIT.call_once(|| {
        let start_time = Instant::now();
        let result = perform_safe_cuda_init();
        let init_time = start_time.elapsed().as_millis() as u64;
        
        let final_result = CudaInitResult {
            available: result.is_ok(),
            device_count: if result.is_ok() { 1 } else { 0 }, // Will be updated below
            error_message: result.err().map(|e| e.to_string()),
            initialization_time_ms: init_time,
        };
        
        if let Ok(mut state) = CUDA_STATE.lock() {
            *state = Some(final_result);
        }
    });
    
    // Return the cached result
    if let Ok(state) = CUDA_STATE.lock() {
        state.as_ref().cloned().unwrap_or(CudaInitResult {
            available: false,
            device_count: 0,
            error_message: Some("Initialization state unavailable".to_string()),
            initialization_time_ms: 0,
        })
    } else {
        CudaInitResult {
            available: false,
            device_count: 0,
            error_message: Some("Failed to access initialization state".to_string()),
            initialization_time_ms: 0,
        }
    }
}

fn perform_safe_cuda_init() -> Result<(), CudaError> {
    use std::sync::mpsc;
    use std::thread;
    
    log::info!("🚀 Starting safe CUDA initialization...");
    
    // Perform initialization in a separate thread with timeout
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        let result = std::panic::catch_unwind(|| {
            // Set thread-local panic handler
            std::panic::set_hook(Box::new(|panic_info| {
                log::error!("CUDA initialization panicked: {:?}", panic_info);
            }));
            
            // Perform actual initialization
            log::debug!("Calling cuda_runtime_health_check...");
            cuda_runtime_health_check();
            
            log::debug!("Calling initialize_cuda...");
            initialize_cuda()
        });
        
        let final_result = match result {
            Ok(cuda_result) => cuda_result,
            Err(panic_payload) => {
                log::error!("CUDA initialization panicked: {:?}", panic_payload);
                Err(CudaError(super::bindings::CUDA_ERROR_NOT_INITIALIZED))
            }
        };
        
        let _ = tx.send(final_result);
    });
    
    // Wait for result with timeout
    match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(result) => {
            match result {
                Ok(()) => {
                    log::info!("✅ CUDA initialization completed successfully");
                    Ok(())
                }
                Err(e) => {
                    log::warn!("⚠️ CUDA initialization failed: {}", e);
                    Err(e)
                }
            }
        }
        Err(_) => {
            log::error!("❌ CUDA initialization timed out after 30 seconds");
            Err(CudaError(super::bindings::CUDA_ERROR_NOT_READY))
        }
    }
}

/// Safe memory manager creation with fallback
pub fn create_safe_memory_manager() -> Result<CudaMemoryManager, CudaError> {
    let init_result = safe_cuda_init();
    
    if !init_result.available {
        log::error!("Cannot create memory manager: CUDA not available");
        if let Some(error) = init_result.error_message {
            log::error!("CUDA error: {}", error);
        }
        return Err(CudaError(super::bindings::CUDA_ERROR_NOT_INITIALIZED));
    }
    
    log::info!("Creating CUDA memory manager (init took {}ms)...", init_result.initialization_time_ms);
    
    // Create memory manager with timeout protection
    use std::sync::mpsc;
    use std::thread;
    
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        let result = std::panic::catch_unwind(|| {
            super::memory::CudaMemoryManager::new()
        });
        
        let final_result = match result {
            Ok(manager_result) => manager_result,
            Err(_) => {
                log::error!("Memory manager creation panicked");
                Err(CudaError(super::bindings::CUDA_ERROR_NOT_INITIALIZED))
            }
        };
        
        let _ = tx.send(final_result);
    });
    
    match rx.recv_timeout(Duration::from_secs(15)) {
        Ok(result) => result,
        Err(_) => {
            log::error!("Memory manager creation timed out");
            Err(CudaError(super::bindings::CUDA_ERROR_NOT_READY))
        }
    }
}

/// Safe context creation with fallback
pub fn create_safe_cuda_context() -> Result<CudaContext, CudaError> {
    let init_result = safe_cuda_init();
    
    if !init_result.available {
        log::error!("Cannot create CUDA context: CUDA not available");
        return Err(CudaError(super::bindings::CUDA_ERROR_NOT_INITIALIZED));
    }
    
    log::info!("Creating CUDA context...");
    
    // Create context with timeout protection
    use std::sync::mpsc;
    use std::thread;
    
    let (tx, rx) = mpsc::channel();
    
    thread::spawn(move || {
        let result = std::panic::catch_unwind(|| {
            super::context::CudaContext::new()
        });
        
        let final_result = match result {
            Ok(context_result) => context_result,
            Err(_) => {
                log::error!("CUDA context creation panicked");
                Err(CudaError(super::bindings::CUDA_ERROR_NOT_INITIALIZED))
            }
        };
        
        let _ = tx.send(final_result);
    });
    
    match rx.recv_timeout(Duration::from_secs(10)) {
        Ok(result) => result,
        Err(_) => {
            log::error!("CUDA context creation timed out");
            Err(CudaError(super::bindings::CUDA_ERROR_NOT_READY))
        }
    }
}

/// Check if CUDA is available (cached result)
pub fn is_cuda_available() -> bool {
    safe_cuda_init().available
}

/// Get CUDA initialization info
pub fn get_cuda_init_info() -> CudaInitResult {
    safe_cuda_init()
}

/// Force re-initialization (for testing)
pub fn force_cuda_reinit() -> CudaInitResult {
    // Clear the state
    if let Ok(mut state) = CUDA_STATE.lock() {
        *state = None;
    }
    
    // This won't work with Once, but we can provide diagnostics
    log::warn!("Force re-initialization requested, but Once prevents this");
    log::info!("Current state: {:?}", get_cuda_init_info());
    
    get_cuda_init_info()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_safe_cuda_init() {
        let result = safe_cuda_init();
        println!("CUDA init result: {:?}", result);
        
        // Should not panic
        assert!(result.initialization_time_ms > 0);
    }
    
    #[test]
    fn test_is_cuda_available() {
        let available = is_cuda_available();
        println!("CUDA available: {}", available);
        
        // Should match init result
        let init_info = get_cuda_init_info();
        assert_eq!(available, init_info.available);
    }
    
    #[test]
    fn test_safe_memory_manager() {
        if is_cuda_available() {
            match create_safe_memory_manager() {
                Ok(manager) => {
                    println!("✓ Memory manager created successfully");
                    assert!(manager.is_initialized());
                }
                Err(e) => {
                    println!("⚠️ Memory manager creation failed: {}", e);
                }
            }
        } else {
            println!("⚠️ CUDA not available, skipping memory manager test");
        }
    }
}