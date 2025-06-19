pub mod ffi;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use crossbeam::queue::SegQueue;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

// Configuration constants
const DEFAULT_PAGE_SIZE: usize = 256 * 1024; // 256 KiB
const ALIGNMENT: usize = 64; // CUDA memory alignment requirement
const MAX_PAGES_PER_ARENA: usize = 1024;

// Custom error type to replace std::alloc::AllocError
#[derive(Debug, Clone, Copy)]
pub struct AllocError;

impl std::fmt::Display for AllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Memory allocation failed")
    }
}

impl std::error::Error for AllocError {}

/// Represents a memory page for KV cache storage
#[derive(Debug)]
pub struct Page {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
}

impl Page {
    /// Allocate a new page with the given size
    fn new(size: usize) -> Result<Self, AllocError> {
        let layout = Layout::from_size_align(size, ALIGNMENT)
            .map_err(|_| AllocError)?;
        
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(AllocError);
        }

        Ok(Page {
            ptr: NonNull::new(ptr).unwrap(),
            size,
            layout,
        })
    }

    /// Get a raw pointer to the page memory
    fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get the size of the page
    fn size(&self) -> usize {
        self.size
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

unsafe impl Send for Page {}
unsafe impl Sync for Page {}

/// Global slab pool for page recycling
#[derive(Debug)]
pub struct GlobalSlabPool {
    pages: SegQueue<Page>,
    total_allocated: AtomicUsize,
    total_recycled: AtomicUsize,
    page_size: usize,
}

impl GlobalSlabPool {
    /// Create a new global slab pool
    pub fn new(page_size: usize) -> Self {
        Self {
            pages: SegQueue::new(),
            total_allocated: AtomicUsize::new(0),
            total_recycled: AtomicUsize::new(0),
            page_size,
        }
    }

    /// Get a page from the pool or allocate a new one
    pub fn get_page(&self) -> Result<Page, AllocError> {
        if let Some(page) = self.pages.pop() {
            self.total_recycled.fetch_add(1, Ordering::Relaxed);
            Ok(page)
        } else {
            let page = Page::new(self.page_size)?;
            self.total_allocated.fetch_add(1, Ordering::Relaxed);
            Ok(page)
        }
    }

    /// Return a page to the pool for recycling
    pub fn return_page(&self, page: Page) {
        self.pages.push(page);
    }

    /// Get allocation statistics
    pub fn stats(&self) -> (usize, usize) {
        (
            self.total_allocated.load(Ordering::Relaxed),
            self.total_recycled.load(Ordering::Relaxed),
        )
    }
}

/// Represents an allocated KV tensor within an arena
#[derive(Debug)]
pub struct KVTensor {
    offset: usize,
    size: usize,
    seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
}

impl KVTensor {
    pub fn new(offset: usize, size: usize, seq_len: usize, hidden_dim: usize, num_heads: usize) -> Self {
        Self {
            offset,
            size,
            seq_len,
            hidden_dim,
            num_heads,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
}

/// Arena allocator for KV cache tensors
#[derive(Debug)]
pub struct SequenceArena {
    pages: Vec<Page>,
    current_page_idx: usize,
    current_offset: usize,
    total_allocated: usize,
    slab_pool: Arc<GlobalSlabPool>,
    sequence_id: u64,
}

impl SequenceArena {
    /// Create a new sequence arena
    pub fn new(slab_pool: Arc<GlobalSlabPool>, sequence_id: u64) -> Result<Self, AllocError> {
        let initial_page = slab_pool.get_page()?;
        let mut pages = Vec::with_capacity(MAX_PAGES_PER_ARENA);
        pages.push(initial_page);

        Ok(Self {
            pages,
            current_page_idx: 0,
            current_offset: 0,
            total_allocated: 0,
            slab_pool,
            sequence_id,
        })
    }

    /// Allocate a KV tensor with bump allocation
    pub fn allocate_kv_tensor(
        &mut self,
        seq_len: usize,
        hidden_dim: usize,
        num_heads: usize,
        dtype_size: usize,
    ) -> Result<KVTensor, AllocError> {
        // Calculate tensor size (key + value)
        let tensor_size = 2 * seq_len * hidden_dim * num_heads * dtype_size;
        let aligned_size = align_up(tensor_size, ALIGNMENT);

        // Check if current page has enough space
        if self.current_offset + aligned_size > self.pages[self.current_page_idx].size() {
            // Need a new page
            if self.current_page_idx + 1 >= self.pages.len() {
                // Allocate new page
                let new_page = self.slab_pool.get_page()?;
                self.pages.push(new_page);
            }
            self.current_page_idx += 1;
            self.current_offset = 0;
        }

        let offset = self.current_offset;
        self.current_offset += aligned_size;
        self.total_allocated += aligned_size;

        Ok(KVTensor::new(
            offset,
            aligned_size,
            seq_len,
            hidden_dim,
            num_heads,
        ))
    }

    /// Get a pointer to tensor data
    pub fn get_tensor_ptr(&self, tensor: &KVTensor) -> *mut u8 {
        // Find the page containing this tensor
        let mut current_offset = 0;
        for (_page_idx, page) in self.pages.iter().enumerate() {
            if current_offset + page.size() > tensor.offset() {
                let page_offset = tensor.offset() - current_offset;
                return unsafe { page.as_ptr().add(page_offset) };
            }
            current_offset += page.size();
        }
        panic!("Tensor offset out of bounds");
    }

    /// Extend a KV tensor for new tokens (zero-copy when possible)
    pub fn extend_kv_tensor(
        &mut self,
        tensor: &mut KVTensor,
        new_seq_len: usize,
        dtype_size: usize,
    ) -> Result<bool, AllocError> {
        let new_size = 2 * new_seq_len * tensor.hidden_dim * tensor.num_heads * dtype_size;
        let aligned_new_size = align_up(new_size, ALIGNMENT);
        
        // Check if we can extend in place
        let available_space = self.pages[self.current_page_idx].size() - self.current_offset;
        let size_increase = aligned_new_size - tensor.size;
        
        if size_increase <= available_space {
            // Can extend in place
            tensor.size = aligned_new_size;
            tensor.seq_len = new_seq_len;
            self.current_offset += size_increase;
            self.total_allocated += size_increase;
            Ok(true) // Extended in place
        } else {
            // Need to allocate a new tensor
            let new_tensor = self.allocate_kv_tensor(
                new_seq_len,
                tensor.hidden_dim,
                tensor.num_heads,
                dtype_size,
            )?;
            
            // Copy old data to new location (would be CUDA memcpy in real implementation)
            // This is where you'd implement the actual tensor copying logic
            
            *tensor = new_tensor;
            Ok(false) // Required copy
        }
    }

    /// Get arena statistics
    pub fn stats(&self) -> ArenaStats {
        ArenaStats {
            sequence_id: self.sequence_id,
            total_allocated: self.total_allocated,
            num_pages: self.pages.len(),
            current_page_utilization: if self.pages.is_empty() {
                0.0
            } else {
                self.current_offset as f64 / self.pages[self.current_page_idx].size() as f64
            },
        }
    }
}

impl Drop for SequenceArena {
    fn drop(&mut self) {
        // Return all pages to the global slab pool
        for page in self.pages.drain(..) {
            self.slab_pool.return_page(page);
        }
    }
}

/// Statistics for arena performance monitoring
#[derive(Debug, Clone)]
pub struct ArenaStats {
    pub sequence_id: u64,
    pub total_allocated: usize,
    pub num_pages: usize,
    pub current_page_utilization: f64,
}

/// KV Cache manager that orchestrates multiple sequence arenas
#[derive(Debug)]
pub struct KVCacheManager {
    slab_pool: Arc<GlobalSlabPool>,
    next_sequence_id: AtomicUsize,
}

impl KVCacheManager {
    /// Create a new KV cache manager
    pub fn new(page_size: usize) -> Self {
        Self {
            slab_pool: Arc::new(GlobalSlabPool::new(page_size)),
            next_sequence_id: AtomicUsize::new(0),
        }
    }

    /// Create a new sequence arena
    pub fn create_sequence_arena(&self) -> Result<SequenceArena, AllocError> {
        let sequence_id = self.next_sequence_id.fetch_add(1, Ordering::Relaxed) as u64;
        SequenceArena::new(Arc::clone(&self.slab_pool), sequence_id)
    }

    /// Get global pool statistics
    pub fn global_stats(&self) -> (usize, usize) {
        self.slab_pool.stats()
    }
}

/// Helper function to align size up to the nearest alignment boundary
fn align_up(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

/// Example usage and benchmarking
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_basic_allocation() {
        let manager = KVCacheManager::new(DEFAULT_PAGE_SIZE);
        let mut arena = manager.create_sequence_arena().unwrap();

        // Allocate a KV tensor for a sequence
        let tensor = arena.allocate_kv_tensor(512, 4096, 32, 2).unwrap(); // 16-bit dtype
        assert_eq!(tensor.seq_len(), 512);
        
        let stats = arena.stats();
        println!("Arena stats: {:?}", stats);
    }

    #[test]
    fn test_tensor_extension() {
        let manager = KVCacheManager::new(DEFAULT_PAGE_SIZE);
        let mut arena = manager.create_sequence_arena().unwrap();

        let mut tensor = arena.allocate_kv_tensor(512, 4096, 32, 2).unwrap();
        
        // Extend the tensor
        let extended_in_place = arena.extend_kv_tensor(&mut tensor, 1024, 2).unwrap();
        assert_eq!(tensor.seq_len(), 1024);
        
        println!("Extended in place: {}", extended_in_place);
    }

    #[test]
    fn test_page_recycling() {
        let manager = KVCacheManager::new(DEFAULT_PAGE_SIZE);
        
        // Create and drop multiple arenas to test recycling
        for _ in 0..10 {
            let mut arena = manager.create_sequence_arena().unwrap();
            let _tensor = arena.allocate_kv_tensor(1024, 4096, 32, 2).unwrap();
        } // Arena drops here, pages should be recycled
        
        let (allocated, recycled) = manager.global_stats();
        println!("Pages allocated: {}, recycled: {}", allocated, recycled);
        assert!(recycled > 0);
    }

    #[test]
    fn benchmark_allocation() {
        let manager = KVCacheManager::new(DEFAULT_PAGE_SIZE);
        let mut arena = manager.create_sequence_arena().unwrap();

        let start = Instant::now();
        let mut tensors = Vec::new();
        
        // Allocate 1000 small tensors
        for i in 0..1000 {
            let tensor = arena.allocate_kv_tensor(64 + i % 100, 2048, 16, 2).unwrap();
            tensors.push(tensor);
        }
        
        let duration = start.elapsed();
        println!("Allocated 1000 tensors in {:?}", duration);
        
        let stats = arena.stats();
        println!("Final arena stats: {:?}", stats);
    }
}

/// Python FFI bindings (would require PyO3 in a real implementation)
pub mod python_bindings {
    use super::*;
    
    /// Simplified interface for Python integration
    pub struct PyKVCacheManager {
        inner: KVCacheManager,
    }
    
    impl PyKVCacheManager {
        pub fn new() -> Self {
            Self {
                inner: KVCacheManager::new(DEFAULT_PAGE_SIZE),
            }
        }
        
        pub fn create_arena(&self) -> PySequenceArena {
            PySequenceArena {
                inner: self.inner.create_sequence_arena().unwrap(),
            }
        }
    }
    
    pub struct PySequenceArena {
        inner: SequenceArena,
    }
    
    impl PySequenceArena {
        pub fn allocate_tensor(&mut self, seq_len: usize, hidden_dim: usize, num_heads: usize) -> usize {
            let tensor = self.inner.allocate_kv_tensor(seq_len, hidden_dim, num_heads, 2).unwrap();
            tensor.offset() // Return offset as handle
        }
        
        pub fn get_stats(&self) -> (usize, usize, f64) {
            let stats = self.inner.stats();
            (stats.total_allocated, stats.num_pages, stats.current_page_utilization)
        }
    }
}

// Example integration with your existing Python code
pub fn example_integration() {
    println!("Arena-Allocated KV-Cache Example");
    println!("================================");
    
    let manager = KVCacheManager::new(DEFAULT_PAGE_SIZE);
    let mut arena = manager.create_sequence_arena().unwrap();
    
    // Simulate allocation for different sequence lengths (like your varying contexts)
    let sequence_lengths = vec![128, 512, 1024, 2048, 4096];
    let mut tensors = Vec::new();
    
    for seq_len in sequence_lengths {
        let tensor = arena.allocate_kv_tensor(seq_len, 4096, 32, 2).unwrap();
        println!("Allocated tensor for seq_len {}: offset={}, size={}", 
                 seq_len, tensor.offset(), tensor.size());
        tensors.push(tensor);
    }
    
    let stats = arena.stats();
    println!("\nFinal arena stats:");
    println!("  Total allocated: {} bytes", stats.total_allocated);
    println!("  Pages used: {}", stats.num_pages);
    println!("  Current page utilization: {:.2}%", stats.current_page_utilization * 100.0);
    
    let (allocated, recycled) = manager.global_stats();
    println!("\nGlobal pool stats:");
    println!("  Total pages allocated: {}", allocated);
    println!("  Pages recycled: {}", recycled);
}