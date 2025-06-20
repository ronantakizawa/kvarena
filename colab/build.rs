// build.rs - Build script for arena_kv_cache
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    
    // Handle CUDA linking if the cuda feature is enabled
    if cfg!(feature = "cuda") {
        setup_cuda_linking();
    } else {
        println!("cargo:warning=CUDA feature not enabled, building CPU-only version");
    }
    
    // Set up library exports for Python bindings
    setup_library_exports();
}

fn setup_cuda_linking() {
    println!("cargo:warning=Setting up CUDA linking...");
    
    // Common CUDA library paths
    let cuda_paths = vec![
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/opt/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-11/lib64",
    ];
    
    // Try to find CUDA installation
    let mut cuda_found = false;
    
    // Check environment variables first
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let lib_path = format!("{}/lib64", cuda_path);
        if std::path::Path::new(&lib_path).exists() {
            println!("cargo:rustc-link-search=native={}", lib_path);
            cuda_found = true;
        }
    }
    
    if let Ok(cuda_root) = env::var("CUDA_ROOT") {
        let lib_path = format!("{}/lib64", cuda_root);
        if std::path::Path::new(&lib_path).exists() {
            println!("cargo:rustc-link-search=native={}", lib_path);
            cuda_found = true;
        }
    }
    
    // If not found in env vars, try common paths
    if !cuda_found {
        for path in &cuda_paths {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search=native={}", path);
                cuda_found = true;
                break;
            }
        }
    }
    
    if cuda_found {
        // Link CUDA runtime
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:warning=CUDA runtime library linked successfully");
        
        // Set feature flag for conditional compilation
        println!("cargo:rustc-cfg=cuda_available");
    } else {
        println!("cargo:warning=CUDA not found, falling back to CPU-only implementation");
        println!("cargo:warning=To enable CUDA, set CUDA_PATH or install CUDA in standard locations");
    }
}

fn setup_library_exports() {
    // Ensure symbols are exported for Python bindings
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,--export-dynamic");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-Wl,-export_dynamic");
    }
    
    // Set up for shared library creation
    println!("cargo:rustc-link-arg=-Wl,-soname,libarena_kv_cache.so");
}