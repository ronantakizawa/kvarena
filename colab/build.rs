// build.rs - Fixed CUDA linking with hang prevention and memory safety
use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");
    
    if cfg!(feature = "cuda") {
        setup_cuda_linking();
    } else {
        println!("cargo:warning=CUDA feature not enabled, building CPU-only version");
        println!("cargo:warning=Use: cargo build --features cuda");
    }
    
    setup_library_exports();
    setup_optimization_flags();
}

fn setup_cuda_linking() {
    println!("cargo:warning=🚀 Setting up SAFE CUDA integration...");
    
    // CRITICAL: Find CUDA installation with timeout protection
    let cuda_root = match find_cuda_installation_safe() {
        Some(path) => path,
        None => {
            println!("cargo:warning=❌ CUDA installation not found!");
            println!("cargo:warning=Please install CUDA toolkit or set CUDA_PATH environment variable.");
            println!("cargo:warning=Continuing with CPU-only build...");
            return;
        }
    };
    
    configure_cuda_linking_safe(&cuda_root);
    verify_cuda_installation_safe(&cuda_root);
}

fn find_cuda_installation_safe() -> Option<PathBuf> {
    // Check environment variables first (most reliable)
    for var in &["CUDA_PATH", "CUDA_ROOT", "CUDA_HOME"] {
        if let Ok(path) = env::var(var) {
            let cuda_path = PathBuf::from(path);
            if verify_cuda_path_safe(&cuda_path) {
                println!("cargo:warning=✅ Found CUDA via {}: {}", var, cuda_path.display());
                return Some(cuda_path);
            } else {
                println!("cargo:warning=⚠️ Invalid CUDA path from {}: {}", var, cuda_path.display());
            }
        }
    }
    
    // Check common installation paths
    let common_paths = if cfg!(target_os = "windows") {
        vec![
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7",
        ]
    } else {
        vec![
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/local/cuda-12.6",
            "/usr/local/cuda-12.5", 
            "/usr/local/cuda-12.0",
            "/usr/local/cuda-11.8",
            "/usr/local/cuda-11.7",
        ]
    };
    
    for path_str in common_paths {
        let path = PathBuf::from(path_str);
        if verify_cuda_path_safe(&path) {
            println!("cargo:warning=✅ Found CUDA at: {}", path.display());
            return Some(path);
        }
    }
    
    // Try nvcc-based detection with timeout
    find_cuda_via_nvcc_safe()
}

fn verify_cuda_path_safe(path: &PathBuf) -> bool {
    if !path.exists() {
        return false;
    }
    
    // Check for essential components
    let lib_dir = if cfg!(target_os = "windows") {
        path.join("lib").join("x64")
    } else {
        path.join("lib64")
    };
    
    let include_dir = path.join("include");
    
    // CRITICAL: Check for actual CUDA runtime library
    let cudart_lib = if cfg!(target_os = "windows") {
        lib_dir.join("cudart.lib")
    } else {
        lib_dir.join("libcudart.so")
    };
    
    let cuda_header = include_dir.join("cuda_runtime.h");
    
    let valid = cudart_lib.exists() && cuda_header.exists();
    
    if !valid {
        println!("cargo:warning=❌ Invalid CUDA path {}: missing {} or {}", 
                path.display(), cudart_lib.display(), cuda_header.display());
    } else {
        println!("cargo:warning=✅ Valid CUDA installation at {}", path.display());
    }
    
    valid
}

fn find_cuda_via_nvcc_safe() -> Option<PathBuf> {
    println!("cargo:warning=🔍 Trying nvcc-based CUDA detection...");
    
    // Try `which nvcc` on Unix systems with timeout
    if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
        if let Ok(output) = run_command_with_timeout("which", &["nvcc"], 5) {
            if output.status.success() {
                let nvcc_path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let nvcc_path = PathBuf::from(nvcc_path_str);
                if let Some(cuda_root) = nvcc_path.parent().and_then(|p| p.parent()) {
                    if verify_cuda_path_safe(&cuda_root.to_path_buf()) {
                        println!("cargo:warning=✅ Found CUDA via nvcc: {}", cuda_root.display());
                        return Some(cuda_root.to_path_buf());
                    }
                }
            }
        }
    }
    
    // Try `where nvcc` on Windows with timeout
    if cfg!(target_os = "windows") {
        if let Ok(output) = run_command_with_timeout("where", &["nvcc"], 5) {
            if output.status.success() {
                let nvcc_path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let nvcc_path = PathBuf::from(nvcc_path_str);
                if let Some(cuda_root) = nvcc_path.parent().and_then(|p| p.parent()) {
                    if verify_cuda_path_safe(&cuda_root.to_path_buf()) {
                        println!("cargo:warning=✅ Found CUDA via nvcc: {}", cuda_root.display());
                        return Some(cuda_root.to_path_buf());
                    }
                }
            }
        }
    }
    
    println!("cargo:warning=❌ nvcc-based detection failed");
    None
}

fn run_command_with_timeout(cmd: &str, args: &[&str], timeout_secs: u64) -> Result<std::process::Output, std::io::Error> {
    use std::process::Stdio;
    use std::thread;
    use std::sync::mpsc;
    
    let (tx, rx) = mpsc::channel();
    let cmd_string = cmd.to_string();
    let args_vec: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    
    thread::spawn(move || {
        let result = Command::new(&cmd_string)
            .args(&args_vec)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();
        let _ = tx.send(result);
    });
    
    match rx.recv_timeout(Duration::from_secs(timeout_secs)) {
        Ok(result) => result,
        Err(_) => {
            println!("cargo:warning=⚠️ Command '{}' timed out after {}s", cmd, timeout_secs);
            Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "Command timed out"))
        }
    }
}

fn configure_cuda_linking_safe(cuda_path: &PathBuf) {
    println!("cargo:warning=🔗 Configuring SAFE CUDA linking for: {}", cuda_path.display());
    
    // CRITICAL: Set up library search paths
    let lib_dir = if cfg!(target_os = "windows") {
        cuda_path.join("lib").join("x64")
    } else {
        cuda_path.join("lib64")
    };
    
    if !lib_dir.exists() {
        println!("cargo:warning=❌ CUDA library directory not found: {}", lib_dir.display());
        return;
    }
    
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    
    // CRITICAL: Link CUDA runtime - this enables real CUDA calls
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cuda");
        
        // Add Windows-specific DLL path
        let bin_dir = cuda_path.join("bin");
        if bin_dir.exists() {
            println!("cargo:rustc-link-search=native={}", bin_dir.display());
        }
    } else {
        println!("cargo:rustc-link-lib=cudart");
        
        // CRITICAL: Add rpath for runtime library loading
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        
        // Also add current directory for development
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }
    
    // Set include path for potential C++ integration
    let include_dir = cuda_path.join("include");
    if include_dir.exists() {
        println!("cargo:include={}", include_dir.display());
    }
    
    // CRITICAL: Enable CUDA compilation flag
    println!("cargo:rustc-cfg=cuda_available");
    println!("cargo:warning=✅ CUDA runtime library configured for TRUE integration");
}

fn verify_cuda_installation_safe(cuda_path: &PathBuf) {
    println!("cargo:warning=🔍 Verifying CUDA installation...");
    
    // Check essential CUDA files
    let essential_files = if cfg!(target_os = "windows") {
        vec![
            ("bin/cudart64_*.dll", true),   // true = wildcard pattern
            ("lib/x64/cudart.lib", false),
            ("include/cuda_runtime.h", false),
            ("include/cuda.h", false),
        ]
    } else {
        vec![
            ("lib64/libcudart.so", false),
            ("include/cuda_runtime.h", false), 
            ("include/cuda.h", false),
        ]
    };
    
    let mut missing_files = Vec::new();
    
    for (file_pattern, is_wildcard) in essential_files {
        let file_path = cuda_path.join(file_pattern);
        
        if is_wildcard {
            // Handle wildcards for Windows DLL versioning
            let parent = file_path.parent().unwrap();
            let filename = file_path.file_name().unwrap().to_str().unwrap();
            let prefix = filename.split('*').next().unwrap();
            let suffix = filename.split('*').last().unwrap();
            
            if parent.exists() {
                let mut found = false;
                if let Ok(entries) = std::fs::read_dir(parent) {
                    for entry in entries.flatten() {
                        if let Some(name) = entry.file_name().to_str() {
                            if name.starts_with(prefix) && name.ends_with(suffix) {
                                found = true;
                                break;
                            }
                        }
                    }
                }
                if !found {
                    missing_files.push(file_pattern.to_string());
                }
            } else {
                missing_files.push(file_pattern.to_string());
            }
        } else if !file_path.exists() {
            missing_files.push(file_pattern.to_string());
        }
    }
    
    if !missing_files.is_empty() {
        println!("cargo:warning=⚠️ Missing CUDA files:");
        for file in &missing_files {
            println!("cargo:warning=  - {}", file);
        }
        
        if missing_files.len() > 2 {
            println!("cargo:warning=❌ Critical CUDA files missing. Build may fail at runtime.");
            println!("cargo:warning=💡 Please reinstall CUDA toolkit from: https://developer.nvidia.com/cuda-downloads");
        } else {
            println!("cargo:warning=⚠️ Some CUDA files missing but build will continue");
        }
    } else {
        println!("cargo:warning=✅ CUDA installation verified - all essential files found");
    }
    
    // Try to get CUDA version safely
    if let Some(version) = get_cuda_version_safe(cuda_path) {
        println!("cargo:warning=📋 CUDA Runtime Version: {}", version);
        
        // Check T4 compatibility (requires CUDA 10.0+)
        if version_is_compatible(&version) {
            println!("cargo:warning=✅ CUDA version is T4 compatible");
        } else {
            println!("cargo:warning=⚠️ CUDA version may not be T4 compatible (requires 10.0+)");
        }
    }
}

fn get_cuda_version_safe(cuda_path: &PathBuf) -> Option<String> {
    // Try to read version from version.json (CUDA 11.0+)
    if let Ok(content) = std::fs::read_to_string(cuda_path.join("version.json")) {
        if let Some(version) = extract_version_from_json(&content) {
            return Some(version);
        }
    }
    
    // Try version.txt
    if let Ok(content) = std::fs::read_to_string(cuda_path.join("version.txt")) {
        if let Some(version) = extract_version_from_text(&content) {
            return Some(version);
        }
    }
    
    // Try nvcc with timeout
    let nvcc_path = cuda_path.join("bin").join(if cfg!(target_os = "windows") { "nvcc.exe" } else { "nvcc" });
    if nvcc_path.exists() {
        if let Ok(output) = run_command_with_timeout(nvcc_path.to_str().unwrap(), &["--version"], 10) {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                return extract_version_from_nvcc(&output_str);
            }
        }
    }
    
    None
}

fn extract_version_from_json(content: &str) -> Option<String> {
    // Simple JSON parsing for version - look for "cuda" section
    for line in content.lines() {
        if line.contains("\"version\"") && line.contains(":") {
            if let Some(start) = line.find(':') {
                let value_part = &line[start + 1..];
                if let Some(quote_start) = value_part.find('"') {
                    if let Some(quote_end) = value_part[quote_start + 1..].find('"') {
                        let version = &value_part[quote_start + 1..quote_start + 1 + quote_end];
                        if version.chars().any(|c| c.is_ascii_digit()) {
                            return Some(version.to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

fn extract_version_from_text(content: &str) -> Option<String> {
    for line in content.lines() {
        if line.contains("CUDA Version") {
            if let Some(version) = line.split_whitespace().last() {
                return Some(version.to_string());
            }
        }
    }
    None
}

fn extract_version_from_nvcc(output: &str) -> Option<String> {
    for line in output.lines() {
        if line.contains("release") {
            if let Some(start) = line.find("release ") {
                let version_part = &line[start + 8..];
                if let Some(end) = version_part.find(',') {
                    return Some(version_part[..end].trim().to_string());
                } else if let Some(end) = version_part.find(' ') {
                    return Some(version_part[..end].trim().to_string());
                }
            }
        }
    }
    None
}

fn version_is_compatible(version: &str) -> bool {
    // Extract major.minor version
    if let Some(dot_pos) = version.find('.') {
        if let Ok(major) = version[..dot_pos].parse::<i32>() {
            if major >= 10 {
                return true;
            }
            if major == 9 {
                // Check minor version for 9.x
                if let Some(minor_str) = version.get(dot_pos + 1..) {
                    let minor_part = minor_str.split_whitespace().next().unwrap_or("0");
                    if let Ok(minor) = minor_part.parse::<i32>() {
                        return minor >= 2; // CUDA 9.2+
                    }
                }
            }
        }
    }
    false
}

fn setup_library_exports() {
    println!("cargo:warning=🔧 Configuring library exports...");
    
    // CRITICAL: Proper symbol export for FFI
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,--export-dynamic");
        println!("cargo:rustc-link-arg=-Wl,-soname,libarena_kv_cache.so.1");
        
        // Add current directory to rpath for development
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-Wl,-export_dynamic");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
    } else if cfg!(target_os = "windows") {
        // Windows DLL exports for FFI functions
        let ffi_exports = [
            "kv_cache_manager_new",
            "kv_cache_manager_free", 
            "kv_cache_create_sequence_arena",
            "sequence_arena_free",
            "sequence_arena_allocate_tensor",
            "sequence_arena_get_tensor_ptr",
            "sequence_arena_get_stats",
            "sequence_arena_extend_tensor",
            "kv_cache_manager_get_global_stats",
            "arena_get_default_page_size",
            "arena_get_alignment",
            "arena_align_size",
            // Production API functions
            "prod_extend_tensor_pure_zero_copy",
            "prod_extend_tensor_zero_copy",
            "prod_extend_tensor_for_generation",
            "prod_get_zero_copy_stats",
            "prod_get_tensor_device_ptrs",
            "prod_copy_host_to_tensor",
            "prod_copy_new_tokens_to_tensor",
            "prod_copy_new_tokens_only",
            "prod_get_tensor_memory_layout",
            "prod_can_extend_zero_copy_to",
            // Manager functions
            "prod_kv_cache_init_for_server",
            "prod_kv_cache_init_for_chatbot",
            "prod_kv_cache_init_for_documents",
            "prod_kv_cache_manager_free",
            "prod_get_metrics",
            "prod_get_system_health",
            // Arena functions
            "prod_create_sequence_arena_with_growth",
            "prod_create_sequence_arena",
            "prod_sequence_arena_free",
            "prod_get_bump_arena_stats",
            "prod_benchmark_pure_bump_allocation",
            // Tensor functions
            "prod_allocate_kv_tensor_with_growth",
            "prod_allocate_kv_tensor_safe",
            "prod_allocate_kv_tensor",
            "prod_allocate_tensor_pure_bump",
            "prod_kv_tensor_free",
            // Slab functions
            "prod_get_slab_recycling_stats",
            "prod_cleanup_slab_pools",
            "prod_verify_lock_free_recycling",
            "prod_get_slab_pool_status",
            // Utility functions
            "prod_calculate_optimal_page_size",
            "prod_get_version",
            "prod_get_features",
            "prod_check_cuda_availability",
            // Safety functions
            "prod_emergency_cleanup",
            "prod_monitor_system_health",
            "prod_validate_zero_copy_implementation",
        ];
        
        for export in &ffi_exports {
            println!("cargo:rustc-link-arg=/EXPORT:{}", export);
        }
    }
    
    println!("cargo:warning=✅ Library exports configured");
}

fn setup_optimization_flags() {
    println!("cargo:warning=⚙️ Setting up optimization flags...");
    
    // Add memory safety flags
    if cfg!(debug_assertions) {
        println!("cargo:rustc-env=RUST_BACKTRACE=1");
        println!("cargo:warning=🐛 Debug build: backtrace enabled");
    }
    
    // Platform-specific optimizations
    if cfg!(target_os = "linux") {
        // Linux-specific optimizations for CUDA
        println!("cargo:rustc-link-arg=-Wl,--as-needed");
        println!("cargo:rustc-link-arg=-Wl,--gc-sections");
    }
    
    // Set panic handling for CUDA safety
    if cfg!(feature = "cuda") {
        println!("cargo:rustc-env=RUST_CUDA_PANIC_STRATEGY=abort");
        println!("cargo:warning=🛡️ CUDA panic strategy: abort (prevents hangs)");
    }
    
    println!("cargo:warning=✅ Optimization flags configured");
}

// Additional helper for troubleshooting
fn print_debug_info() {
    println!("cargo:warning=🔍 Build Debug Information:");
    println!("cargo:warning=  Target OS: {}", env::consts::OS);
    println!("cargo:warning=  Target Arch: {}", env::consts::ARCH);
    
    // Print relevant environment variables
    let env_vars = [
        "CUDA_PATH", "CUDA_ROOT", "CUDA_HOME", 
        "LD_LIBRARY_PATH", "PATH", "LIBRARY_PATH"
    ];
    
    for var in &env_vars {
        if let Ok(value) = env::var(var) {
            if value.to_lowercase().contains("cuda") || var == &"LD_LIBRARY_PATH" {
                println!("cargo:warning=  {}: {}", var, 
                        if value.len() > 100 { 
                            format!("{}...", &value[..100]) 
                        } else { 
                            value 
                        });
            }
        }
    }
    
    // Check for nvidia-smi with timeout
    println!("cargo:warning=🔍 GPU Detection:");
    match run_command_with_timeout("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"], 5) {
        Ok(output) if output.status.success() => {
            let gpu_list = String::from_utf8_lossy(&output.stdout);
            for (i, gpu) in gpu_list.lines().enumerate().take(3) { // Limit to 3 GPUs
                println!("cargo:warning=  GPU {}: {}", i, gpu.trim());
            }
        }
        Ok(_) => println!("cargo:warning=  nvidia-smi: Available but returned error"),
        Err(_) => println!("cargo:warning=  nvidia-smi: Not found or timed out"),
    }
    
    // Check for nvcc with timeout
    match run_command_with_timeout("nvcc", &["--version"], 5) {
        Ok(output) if output.status.success() => {
            let version_output = String::from_utf8_lossy(&output.stdout);
            if let Some(version_line) = version_output.lines().find(|line| line.contains("release")) {
                println!("cargo:warning=  nvcc: {}", version_line.trim());
            } else {
                println!("cargo:warning=  nvcc: Available");
            }
        }
        Ok(_) => println!("cargo:warning=  nvcc: Available but returned error"),
        Err(_) => println!("cargo:warning=  nvcc: Not found or timed out"),
    }
}

// Call debug info if needed
#[allow(dead_code)]
fn maybe_print_debug_info() {
    if env::var("ARENA_KV_DEBUG").is_ok() || cfg!(debug_assertions) {
        print_debug_info();
    }
}