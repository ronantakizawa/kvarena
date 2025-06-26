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