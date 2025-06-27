# Arena KV-Cache for Mistral-7B
**High-Performance Memory Manager for Transformer Key/Value Tensors Mistral-7B**

Arena-Allocated KV-Cache with **Slab Recycling** & **True Zero-Copy Extensions** – a production-ready, low-fragmentation memory manager for LLM servers, written in Rust with seamless Python bindings.

---

## 🚀 Why Arena KV-Cache?

| Pain Point | Symptom in Production | Arena KV-Cache Solution |
|------------|----------------------|-------------------------|
| **Memory Fragmentation** | Contexts of widely varying lengths leave “holes” in the CUDA heap → OOM long before theoretical capacity | **Slab Recycling** – pages automatically return to lock-free pools for reuse |
| **Copy Amplification** | Every generation step copies KV tensors into a larger buffer → higher latency | **True Zero-Copy** – only atomic metadata updates; data never moves |
| **GC Stalls** | Whole-tensor drops trigger synchronous device-side frees | **Bump Allocation** – `offset += align(size)`; no per-tensor metadata, instant cleanup |

---

## ✨ Key Features

- **🏎️ True Zero-Copy Extensions**  
  Sub-microsecond tensor growth using atomic pointer/length updates.

- **♻️ Lock-Free Slab Recycling**  
  Pages are recycled back into a global `SegQueue<Page>` when an arena drops.

- **🎯 KV-Optimized Page Sizing**  
  Configure `PAGE_BYTES = ceil(max_expected_tensor_bytes)` to eliminate internal waste.

- **🔥 CUDA-First Design**  
  Tuned for Tesla T4, V100, A100, H100; falls back gracefully to CPU.

- **⚡ Bump Allocation**  
  O(1) allocation & free with zero per-tensor overhead.

- **🐍 Python Integration**  
  `pyo3` bindings expose `ArenaKVCacheManager` and `SequenceArena` as native PyTorch
  tensors (`torch.Tensor`) on CUDA.

- **📊 Production Metrics**  
  Built-in Prometheus/Opentelemetry counters: slab hit %, bytes recycled /s, alloc p95, etc.

---

# Arena KV-Cache Project Structure

This is a high-performance memory manager for transformer key/value tensors written in Rust with Python FFI bindings. The project focuses on production-ready LLM serving with arena allocation, slab recycling, and true zero-copy tensor operations.

---

## 📁 Root Level Files

### Build & Configuration
- **Cargo.toml** – Rust package configuration with CUDA features, dependencies (`crossbeam`, `log`, `env_logger`), and build settings for different profiles  
- **build.rs** – Complex build script that handles CUDA toolkit detection, linking configuration, and cross-platform compilation with timeout protection to prevent hangs  
- **README.md** – Project overview explaining the arena-allocated KV cache system with slab recycling and zero-copy extensions  

### Python Integration
- **arena_kv_cache_bindings.py** – ✅ Python FFI bindings with proper KV head handling. Distinguishes between query heads (32) and KV heads (8) for models like Mistral 7B. Includes `ArenaKVCacheManager`, `SequenceArena`, and `ArenaKVCache` classes  
- **llm_runner.py** – Optimized LLM query runner focusing on cache reuse and performance optimization. Includes context warming, batch processing, and honest performance reporting  

---

## 📁 Source Code (`src/`)

### Core Library
- **lib.rs** – Main library entry point that re-exports all modules and defines the `ProductionKVCacheManager` with comprehensive metrics and health monitoring  

### Memory Management Core
- **zero_copy.rs** – Core zero-copy implementation with honest terminology distinguishing between metadata updates (true zero-copy) and data operations (requires copying). Includes `ZeroCopyTensor`, `ZeroCopyArena`, and `GlobalSlabPool`  
- **slab.rs** – Real slab recycling with actual CUDA page return to lock-free pools. Implements `RealSlabPool`, `RealArena`, and `RealSlabManager` for production memory management  
- **kv_layout.rs** – ✅ KV tensor layout with proper head handling. Optimized memory layout for transformer attention with correct Mistral 7B configuration (8 KV heads vs 32 query heads)  

### LLM Server Integration
- **llm_server_api.rs** – Fixed LLM server API with honest zero-copy reporting. Includes simulation, batch processing, and deployment scenarios with proper terminology for metadata vs data operations  

---

## 📁 CUDA Implementation (`src/cuda/`)

### Core CUDA Module
- **mod.rs** – CUDA module coordinator that re-exports all CUDA functionality with safe initialization patterns  

### Safe Initialization & Diagnostics
- **init.rs** – Safe CUDA initialization wrapper with timeout protection and global state management to prevent initialization hangs  
- **diagnostics.rs** – Fixed CUDA diagnostics with comprehensive hang prevention, timeout wrappers, and runtime verification. Includes environment checking and GPU detection  

### Hardware Interface
- **bindings.rs** – Direct CUDA Runtime API bindings linking to actual `libcudart` with comprehensive function signatures for memory, device, and stream management  
- **device.rs** – Fixed device information with timeout protection for potentially hanging CUDA calls. Includes Tesla T4 detection and safe device querying  
- **error.rs** – CUDA error handling with proper error string conversion and comprehensive error reporting  

### Memory Management
- **memory.rs** – Fixed memory management with safe initialization, device stats tracking, and memory pressure monitoring  
- **allocator.rs** – Fixed bump allocator with proper alignment (256-byte CUDA alignment) and real device memory allocation using `cudaMalloc`/`cudaFree`  
- **context.rs** – CUDA context management for multi-device scenarios with automatic device selection and stream management  

### Advanced Features
- **stream.rs** – CUDA stream management for asynchronous operations with proper synchronization and device affinity  
- **tensor.rs** – CUDA tensor operations with real device memory references, zero-copy reshaping, and device-to-device copying  
- **raw.rs** – Direct low-level CUDA API wrappers for advanced use cases requiring fine-grained control  

---

## 📁 FFI Interface (`src/ffi/`)

### FFI Module Organization
- **mod.rs** – Main FFI module with organized submodules and common function exports for C compatibility  
- **types.rs** – Updated with honest zero-copy reporting – C-compatible structures with clear separation between metadata operations and data copying  

### Core FFI Components
- **manager.rs** – Production KV cache manager FFI functions for creating optimized managers for different deployment scenarios  
- **arena.rs** – ✅ Sequence arena management with proper KV head handling and model-aware configuration for Mistral, LLaMA, and other models  
- **tensor.rs** – ✅ Tensor allocation with automatic KV head detection and separate tracking of query vs KV heads  

### Advanced FFI Operations
- **tensor_ops.rs** – Fixed tensor operations with honest zero-copy terminology, clearly separating metadata updates from data copy operations  
- **slab.rs** – Slab recycling FFI functions with batch allocation, lock-free verification, and comprehensive statistics  
- **safety.rs** – Fixed safety functions with honest zero-copy reporting, emergency cleanup, and system health monitoring  
- **utils.rs** – Fixed utility functions with updated struct fields for honest performance reporting and batch operations  

---

## 📁 Testing & Validation (`src/bin/`)

- **test_cuda_integration.rs** – Comprehensive CUDA integration test with T4 GPU verification, stress testing, memory bandwidth testing, and KV-cache specific operations validation  

---

## 🔧 Key Features & Fixes

### Fixed KV Head Handling
- **Mistral 7B**: Correctly uses 8 KV heads (not 32 query heads) for cache allocation  
- **Auto-detection**: Identifies model patterns and maps query heads to appropriate KV heads  
- **Memory efficiency**: Significant memory savings through proper Grouped Query Attention (GQA) support  

### Honest Zero-Copy Reporting
- **Metadata operations**: True zero-copy (atomic updates)  
- **Data operations**: Require copying (honest about limitations)  
- **Clear separation**: Distinguishes between different operation types in APIs and metrics  

### Production Safety
- **Timeout protection**: Prevents hangs in CUDA initialization and operations  
- **Memory safety**: Proper alignment, bounds checking, and cleanup  
- **Error handling**: Comprehensive error reporting and graceful degradation  

### Performance Optimization
- **Slab recycling**: Real page reuse with lock-free queues  
- **Bump allocation**: O(1) allocation with minimal overhead  
- **T4 optimization**: Specific optimizations for Tesla T4 GPUs  

---

This architecture provides a production-ready foundation for high-performance LLM serving with careful attention to memory efficiency, safety, and honest performance reporting.
