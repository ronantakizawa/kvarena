// build.rs - Enhanced build script for true CUDA integration
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    
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
    println!("cargo:warning=Setting up TRUE CUDA integration...");
    
    // Try to find CUDA installation
    let cuda_root = find_cuda_installation();
    
    match cuda_root {
        Some(cuda_path) => {
            configure_cuda_linking(&cuda_path);
            verify_cuda_setup(&cuda_path);
        }
        None => {
            println!("cargo:warning=CUDA not found! Please install CUDA toolkit or set CUDA_PATH");
            println!("cargo:warning=Common CUDA installation locations:");
            println!("cargo:warning=  Linux: /usr/local/cuda, /opt/cuda");
            println!("cargo:warning=  Windows: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*");
            std::process::exit(1);
        }
    }
}

fn find_cuda_installation() -> Option<PathBuf> {
    // Check environment variables first
    let env_vars = ["CUDA_PATH", "CUDA_ROOT", "CUDA_HOME"];
    for var in &env_vars {
        if let Ok(path) = env::var(var) {
            let cuda_path = PathBuf::from(path);
            if cuda_path.exists() {
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
        ]
    } else {
        vec![
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/local/cuda-12.0",
            "/usr/local/cuda-11.8",
            "/usr/local/cuda-11.7",
            "/usr/local/cuda-11.6",
        ]
    };
    
    for path_str in common_paths {
        let path = PathBuf::from(path_str);
        if path.exists() {
            println!("cargo:warning=Found CUDA at: {}", path.display());
            return Some(path);
        }
    }
    
    // Try to find via nvidia-smi or nvcc
    if let Some(path) = find_cuda_via_tools() {
        return Some(path);
    }
    
    None
}

fn find_cuda_via_tools() -> Option<PathBuf> {
    // Try nvcc first
    if let Ok(output) = Command::new("nvcc").arg("--version").output() {
        if output.status.success() {
            println!("cargo:warning=Found nvcc, attempting to locate CUDA root");
            
            // Try to find CUDA root from nvcc location
            if let Ok(which_output) = Command::new("which").arg("nvcc").output() {
                if which_output.status.success() {
                    let nvcc_path = String::from_utf8_lossy(&which_output.stdout);
                    let nvcc_path = nvcc_path.trim();
                    if let Some(cuda_root) = PathBuf::from(nvcc_path).parent()?.parent() {
                        if cuda_root.join("lib64").exists() || cuda_root.join("lib").exists() {
                            println!("cargo:warning=Derived CUDA root from nvcc: {}", cuda_root.display());
                            return Some(cuda_root.to_path_buf());
                        }
                    }
                }
            }
        }
    }
    
    // Try nvidia-smi to confirm CUDA is available
    if let Ok(output) = Command::new("nvidia-smi").output() {
        if output.status.success() {
            println!("cargo:warning=nvidia-smi found, CUDA driver is available");
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Some(cuda_version) = extract_cuda_version(&output_str) {
                println!("cargo:warning=CUDA Driver Version: {}", cuda_version);
            }
        }
    }
    
    None
}

fn extract_cuda_version(nvidia_smi_output: &str) -> Option<String> {
    for line in nvidia_smi_output.lines() {
        if line.contains("CUDA Version:") {
            if let Some(version_part) = line.split("CUDA Version:").nth(1) {
                return Some(version_part.trim().to_string());
            }
        }
    }
    None
}

fn configure_cuda_linking(cuda_path: &PathBuf) {
    println!("cargo:warning=Configuring CUDA linking for: {}", cuda_path.display());
    
    // Determine library directory
    let lib_dirs = if cfg!(target_os = "windows") {
        vec!["lib/x64", "lib"]
    } else {
        vec!["lib64", "lib"]
    };
    
    let mut lib_path = None;
    for lib_dir in lib_dirs {
        let potential_path = cuda_path.join(lib_dir);
        if potential_path.exists() {
            lib_path = Some(potential_path);
            break;
        }
    }
    
    let lib_path = match lib_path {
        Some(path) => path,
        None => {
            println!("cargo:warning=Could not find CUDA library directory in: {}", cuda_path.display());
            std::process::exit(1);
        }
    };
    
    println!("cargo:warning=Using CUDA lib path: {}", lib_path.display());
    println!("cargo:rustc-link-search=native={}", lib_path.display());
    
    // Link CUDA runtime library
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=cudart");
        // For Windows, might need additional libraries
        println!("cargo:rustc-link-lib=cuda");
    } else {
        println!("cargo:rustc-link-lib=cudart");
    }
    
    // Add include path for potential C++ integration
    let include_path = cuda_path.join("include");
    if include_path.exists() {
        println!("cargo:include={}", include_path.display());
    }
    
    // Set feature flag for conditional compilation
    println!("cargo:rustc-cfg=cuda_available");
    println!("cargo:warning=✓ CUDA runtime library linked successfully");
}

fn verify_cuda_setup(cuda_path: &PathBuf) {
    println!("cargo:warning=Verifying CUDA setup...");
    
    // Check for essential CUDA files
    let essential_files = if cfg!(target_os = "windows") {
        vec![
            "bin/cudart64_*.dll",
            "lib/x64/cudart.lib",
            "include/cuda_runtime.h",
        ]
    } else {
        vec![
            "lib64/libcudart.so",
            "include/cuda_runtime.h",
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
            
            if parent.exists() {
                let mut found = false;
                if let Ok(entries) = std::fs::read_dir(parent) {
                    for entry in entries.flatten() {
                        if let Some(name) = entry.file_name().to_str() {
                            if name.starts_with(prefix) && name.ends_with(".dll") {
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
        for file in missing_files {
            println!("cargo:warning=  - {}", file);
        }
        println!("cargo:warning=CUDA installation may be incomplete");
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
    // Try to read version from version.txt or version.json
    let version_files = ["version.txt", "version.json"];
    
    for version_file in version_files {
        let version_path = cuda_path.join(version_file);
        if let Ok(content) = std::fs::read_to_string(version_path) {
            if let Some(version) = extract_version_from_content(&content) {
                return Some(version);
            }
        }
    }
    
    // Try nvcc if available
    if let Ok(output) = Command::new(cuda_path.join("bin").join("nvcc"))
        .arg("--version")
        .output() 
    {
        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Some(version) = extract_version_from_nvcc(&output_str) {
                return Some(version);
            }
        }
    }
    
    None
}

fn extract_version_from_content(content: &str) -> Option<String> {
    // Simple version extraction - look for patterns like "12.0" or "11.8"
    for line in content.lines() {
        if let Some(captures) = regex::Regex::new(r"(\d+\.\d+)")
            .ok()?
            .captures(line) 
        {
            return Some(captures[1].to_string());
        }
    }
    None
}

fn extract_version_from_nvcc(output: &str) -> Option<String> {
    for line in output.lines() {
        if line.contains("release") {
            if let Some(captures) = regex::Regex::new(r"release (\d+\.\d+)")
                .ok()?
                .captures(line)
            {
                return Some(captures[1].to_string());
            }
        }
    }
    None
}

fn version_is_compatible(version: &str) -> bool {
    if let Ok(version_num) = version.parse::<f32>() {
        version_num >= 10.0
    } else {
        false
    }
}

fn setup_library_exports() {
    println!("cargo:warning=Configuring library exports...");
    
    // Ensure symbols are exported for Python bindings and FFI
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,--export-dynamic");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-Wl,-export_dynamic");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
    } else if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-arg=/EXPORT:kv_cache_manager_new");
        println!("cargo:rustc-link-arg=/EXPORT:kv_cache_manager_free");
        println!("cargo:rustc-link-arg=/EXPORT:kv_cache_create_sequence_arena");
    }
    
    // Set up for shared library creation
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,-soname,libarena_kv_cache.so.1");
    }
    
    println!("cargo:warning=✓ Library exports configured");
}

// Simplified regex implementation to avoid dependency
mod regex {
    pub struct Regex {
        pattern: String,
    }
    
    impl Regex {
        pub fn new(pattern: &str) -> Result<Self, ()> {
            Ok(Regex {
                pattern: pattern.to_string(),
            })
        }
        
        pub fn captures<'t>(&self, text: &'t str) -> Option<Captures<'t>> {
            // Very basic regex implementation for version extraction
            if self.pattern == r"(\d+\.\d+)" {
                for (i, c) in text.chars().enumerate() {
                    if c.is_ascii_digit() {
                        let start = i;
                        let mut end = i + 1;
                        let chars: Vec<char> = text.chars().collect();
                        
                        // Find end of first number
                        while end < chars.len() && chars[end].is_ascii_digit() {
                            end += 1;
                        }
                        
                        // Check for dot
                        if end < chars.len() && chars[end] == '.' {
                            end += 1;
                            // Find end of second number
                            while end < chars.len() && chars[end].is_ascii_digit() {
                                end += 1;
                            }
                            
                            let version: String = chars[start..end].iter().collect();
                            return Some(Captures {
                                text,
                                matches: vec![version.clone()],
                            });
                        }
                    }
                }
            } else if self.pattern == r"release (\d+\.\d+)" {
                if let Some(release_pos) = text.find("release ") {
                    let after_release = &text[release_pos + 8..];
                    let simple_regex = Regex::new(r"(\d+\.\d+)").ok()?;
                    return simple_regex.captures(after_release);
                }
            }
            None
        }
    }
    
    pub struct Captures<'t> {
        text: &'t str,
        matches: Vec<String>,
    }
    
    impl<'t> std::ops::Index<usize> for Captures<'t> {
        type Output = str;
        
        fn index(&self, index: usize) -> &Self::Output {
            if index == 0 {
                self.text
            } else if index <= self.matches.len() {
                &self.matches[index - 1]
            } else {
                ""
            }
        }
    }
}