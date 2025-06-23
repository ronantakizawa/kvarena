# KV Arena

Arena-Allocated KV-Cache with Slab Recycling & Bump Allocation A high-throughput, low-fragmentation memory manager for transformer key/value tensors written in Rust.

1. Why replace the “flat” KV-cache?
Pain point	Symptom in production LLM servers
Fragmentation	Contexts of wildly different lengths leave “holes” in the CUDA heap; allocator falls back to page-sized mallocs → OOM before capacity.
Copy amplification	Every generation step copies KV tensors into a larger buffer when the next token arrives; hurts latency.
GC stalls	Whole-tensor drops trigger device-side frees that are synchronous or trigger CUDA IPC ref-count churn.
Page size = round-up of largest KV tensor you expect (e.g., 256 KiB for 4-bit 8K-seq Llama-2).
Bump allocation = offset += align(size); no per-tensor metadata.
Slab recycling = when SequenceArena drops, its pages go back to GlobalSlabPool (a lock-free SegQueue<Page>).

