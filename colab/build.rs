// build.rs - Complete fixed CUDA linking for true integration
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    
    if cfg!(feature = "cuda") {
        setup_cuda_linking();
    } else {
        println!("cargo:warning=CUDA feature not enabled, building CPU-only version");
    }
    
    setup_library_exports();
}

fn setup_cuda_linking() {
    println!("cargo:warning=Setting up TRUE CUDA integration...");
    
    // CRITICAL: Find CUDA installation
    let cuda_root = find_cuda_installation()
        .expect("CUDA installation not found! Please install CUDA toolkit or set CUDA_PATH environment variable.");
    
    configure_cuda_linking(&cuda_root);
    verify_cuda_installation(&cuda_root);
}

fn find_cuda_installation() -> Option<PathBuf> {
    // Check environment variables first
    for var in &["CUDA_PATH", "CUDA_ROOT", "CUDA_HOME"] {
        if let Ok(path) = env::var(var) {
            let cuda_path = PathBuf::from(path);
            if verify_cuda_path(&cuda_path) {
                println!("cargo:warning=Found CUDA via {}: {}", var, cuda_path.display());
                return Some(cuda_path);
            }
        }
    }
    
    // Check common installation paths
    let common_paths = if cfg!(target_os = "windows") {
        vec![
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5",
        ]
    } else {
        vec![
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/local/cuda-12.0",
            "/usr/local/cuda-11.8",
            "/usr/local/cuda-11.7",
            "/usr/local/cuda-11.6",
            "/usr/local/cuda-11.5",
        ]
    };
    
    for path_str in common_paths {
        let path = PathBuf::from(path_str);
        if verify_cuda_path(&path) {
            println!("cargo:warning=Found CUDA at: {}", path.display());
            return Some(path);
        }
    }
    
    // Try nvcc-based detection
    find_cuda_via_nvcc()
}

fn verify_cuda_path(path: &PathBuf) -> bool {
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
        println!("cargo:warning=Invalid CUDA path {}: missing {} or {}", 
                path.display(), cudart_lib.display(), cuda_header.display());
    }
    
    valid
}

fn find_cuda_via_nvcc() -> Option<PathBuf> {
    // Try `which nvcc` on Unix systems
    if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
        if let Ok(output) = Command::new("which").arg("nvcc").output() {
            if output.status.success() {
                let nvcc_path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let nvcc_path = PathBuf::from(nvcc_path_str);
                if let Some(cuda_root) = nvcc_path.parent()?.parent() {
                    if verify_cuda_path(&cuda_root.to_path_buf()) {
                        println!("cargo:warning=Found CUDA via nvcc: {}", cuda_root.display());
                        return Some(cuda_root.to_path_buf());
                    }
                }
            }
        }
    }
    
    // Try `where nvcc` on Windows
    if cfg!(target_os = "windows") {
        if let Ok(output) = Command::new("where").arg("nvcc").output() {
            if output.status.success() {
                let nvcc_path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let nvcc_path = PathBuf::from(nvcc_path_str);
                if let Some(cuda_root) = nvcc_path.parent()?.parent() {
                    if verify_cuda_path(&cuda_root.to_path_buf()) {
                        println!("cargo:warning=Found CUDA via nvcc: {}", cuda_root.display());
                        return Some(cuda_root.to_path_buf());
                    }
                }
            }
        }
    }
    
    None
}

fn configure_cuda_linking(cuda_path: &PathBuf) {
    println!("cargo:warning=Configuring CUDA linking for: {}", cuda_path.display());
    
    // CRITICAL: Set up library search paths
    let lib_dir = if cfg!(target_os = "windows") {
        cuda_path.join("lib").join("x64")
    } else {
        cuda_path.join("lib64")
    };
    
    if !lib_dir.exists() {
        panic!("CUDA library directory not found: {}", lib_dir.display());
    }
    
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    
    // CRITICAL: Link CUDA runtime - this is what enables real CUDA calls
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=cudart");
        // Windows may need additional libraries
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
    println!("cargo:warning=✓ CUDA runtime library configured for true integration");
}

fn verify_cuda_installation(cuda_path: &PathBuf) {
    println!("cargo:warning=Verifying CUDA installation...");
    
    // Check essential CUDA files
    let essential_files = if cfg!(target_os = "windows") {
        vec![
            "bin/cudart64_*.dll",
            "lib/x64/cudart.lib",
            "include/cuda_runtime.h",
            "include/cuda.h",
        ]
    } else {
        vec![
            "lib64/libcudart.so",
            "include/cuda_runtime.h", 
            "include/cuda.h",
        ]
    };
    
    let mut missing_files = Vec::new();
    
    for file_pattern in essential_files {
        let file_path = cuda_path.join(file_pattern);
        
        // Handle wildcards for Windows DLL versioning
        if file_pattern.contains('*') {
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
        println!("cargo:warning=CUDA installation may be incomplete");
        
        // Don't fail the build, but warn
        if missing_files.len() > 2 {
            panic!("Critical CUDA files missing. Please reinstall CUDA toolkit.");
        }
    } else {
        println!("cargo:warning=✓ CUDA installation verified");
    }
    
    // Try to get CUDA version
    if let Some(version) = get_cuda_version(cuda_path) {
        println!("cargo:warning=CUDA Runtime Version: {}", version);
        
        // Check T4 compatibility (requires CUDA 10.0+)
        if version_is_compatible(&version) {
            println!("cargo:warning=✓ CUDA version is T4 compatible");
        } else {
            println!("cargo:warning=⚠️ CUDA version may not be T4 compatible (requires 10.0+)");
        }
    }
}

fn get_cuda_version(cuda_path: &PathBuf) -> Option<String> {
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
    
    // Try nvcc
    let nvcc_path = cuda_path.join("bin").join(if cfg!(target_os = "windows") { "nvcc.exe" } else { "nvcc" });
    if nvcc_path.exists() {
        if let Ok(output) = Command::new(nvcc_path).arg("--version").output() {
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
        if line.contains("\"cuda\"") {
            // Look for version in the next few lines
            continue;
        }
        if line.contains("\"version\"") && line.contains(":") {
            if let Some(start) = line.find('"') {
                if let Some(version_start) = line[start + 1..].find('"') {
                    let actual_start = start + 1 + version_start + 1;
                    if let Some(end) = line[actual_start..].find('"') {
                        let version = &line[actual_start..actual_start + end];
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
                    if let Some(space_pos) = minor_str.find(' ') {
                        if let Ok(minor) = minor_str[..space_pos].parse::<i32>() {
                            return minor >= 2; // CUDA 9.2+
                        }
                    } else if let Ok(minor) = minor_str.parse::<i32>() {
                        return minor >= 2;
                    }
                }
            }
        }
    }
    false
}

fn setup_library_exports() {
    println!("cargo:warning=Configuring library exports...");
    
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
        ];
        
        for export in &ffi_exports {
            println!("cargo:rustc-link-arg=/EXPORT:{}", export);
        }
    }
    
    println!("cargo:warning=✓ Library exports configured");
}

// Additional helper for troubleshooting
fn print_debug_info() {
    println!("cargo:warning=Build Debug Information:");
    println!("cargo:warning=  Target OS: {}", env::consts::OS);
    println!("cargo:warning=  Target Arch: {}", env::consts::ARCH);
    
    // Print environment variables
    for (key, value) in env::vars() {
        if key.contains("CUDA") || key.contains("NVIDIA") {
            println!("cargo:warning=  {}: {}", key, value);
        }
    }
    
    // Check for nvidia-smi
    if Command::new("nvidia-smi").output().is_ok() {
        println!("cargo:warning=  nvidia-smi: Available");
    } else {
        println!("cargo:warning=  nvidia-smi: Not found");
    }
    
    // Check for nvcc
    if Command::new("nvcc").arg("--version").output().is_ok() {
        println!("cargo:warning=  nvcc: Available");
    } else {
        println!("cargo:warning=  nvcc: Not found");
    }
}