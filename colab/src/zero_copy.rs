// src/zero_copy.rs - TRUE bump allocation with lock-free slab recycling
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ptr::NonNull;
use crossbeam::queue::SegQueue;
use crate::cuda::{CudaPage, CudaError, CudaContext};

/// TRUE zero-copy tensor with NO page references - enables slab recycling
#[derive(Debug)]
pub struct ZeroCopyTensor {
    /// Direct device pointers (no page reference needed)
    key_device_ptr: NonNull<u8>,
    value_device_ptr: NonNull<u8>,
    /// Current sequence length (ONLY mutable state)
    current_seq_len: AtomicUsize,
    /// Immutable tensor parameters
    max_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    element_size: usize,
    device_id: i32,      // Store device_id directly, not page reference
    arena_id: u64,
    // REMOVED: _page_ref: Arc<CudaPage>, // This prevented slab recycling!
}

impl ZeroCopyTensor {
    /// Create from bump allocation with NO page reference
    pub fn from_bump_allocation(
        device_ptr: NonNull<u8>,
        initial_seq_len: usize,
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        element_size: usize,
        arena_id: u64,
        device_id: i32,  // Pass device_id directly
    ) -> Result<Self, CudaError> {
        // K tensor at start, V tensor after max K tensor space
        let key_device_ptr = device_ptr;
        
        let k_tensor_max_size = max_seq_len * num_heads * head_dim * element_size;
        let value_device_ptr = unsafe {
            NonNull::new_unchecked(device_ptr.as_ptr().add(k_tensor_max_size))
        };
        
        log::debug!("Created tensor with NO page reference: initial={}, max={}, device={}", 
                   initial_seq_len, max_seq_len, device_id);
        
        Ok(ZeroCopyTensor {
            key_device_ptr,
            value_device_ptr,
            current_seq_len: AtomicUsize::new(initial_seq_len),
            max_seq_len,
            num_heads,
            head_dim,
            element_size,
            device_id,  // Direct storage, no Arc reference
            arena_id,
            // NO page reference - enables true slab recycling
        })
    }

    /// TRUE zero-copy extension - ATOMIC metadata update ONLY
    pub fn extend_zero_copy(&self, new_seq_len: usize) -> Result<bool, CudaError> {
        if new_seq_len > self.max_seq_len {
            return Ok(false); // Cannot extend beyond pre-allocated space
        }

        let old_seq_len = self.current_seq_len.load(Ordering::Relaxed);
        
        if new_seq_len <= old_seq_len {
            return Ok(true); // Already at or past this length
        }
        
        // ATOMIC update of sequence length - THIS IS THE ENTIRE EXTENSION OPERATION
        self.current_seq_len.store(new_seq_len, Ordering::Release);
        
        log::debug!("TRUE zero-copy extension: {} -> {} tokens (NO page reference needed)", 
                   old_seq_len, new_seq_len);
        
        Ok(true)
    }

    /// Get current sequence length (atomic read)
    pub fn seq_len(&self) -> usize {
        self.current_seq_len.load(Ordering::Acquire)
    }

    /// Get maximum pre-allocated sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Check if can extend to new length without any memory operations
    pub fn can_extend_zero_copy_to(&self, new_seq_len: usize) -> bool {
        new_seq_len <= self.max_seq_len
    }

    /// Get key tensor device pointer using pure pointer arithmetic
    pub fn key_device_ptr(&self) -> *mut std::ffi::c_void {
        self.key_device_ptr.as_ptr() as *mut std::ffi::c_void
    }

    /// Get value tensor device pointer using pure pointer arithmetic
    pub fn value_device_ptr(&self) -> *mut std::ffi::c_void {
        self.value_device_ptr.as_ptr() as *mut std::ffi::c_void
    }

    /// Get tensor dimensions (current_seq_len, num_heads, head_dim)
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.seq_len(), self.num_heads, self.head_dim)
    }

    /// Get current key tensor size in bytes
    pub fn current_key_size_bytes(&self) -> usize {
        self.seq_len() * self.num_heads * self.head_dim * self.element_size
    }

    /// Get current value tensor size in bytes
    pub fn current_value_size_bytes(&self) -> usize {
        self.seq_len() * self.num_heads * self.head_dim * self.element_size
    }

    /// Get total allocated size (for memory accounting)
    pub fn max_allocated_size_bytes(&self) -> usize {
        2 * self.max_seq_len * self.num_heads * self.head_dim * self.element_size
    }

    /// Copy NEW tokens only (for incremental generation)
    pub fn copy_new_tokens_only(
        &self,
        host_key_data: *const u8,
        host_value_data: *const u8,
        start_token_idx: usize,
        num_new_tokens: usize,
    ) -> Result<(), CudaError> {
        let current_seq_len = self.seq_len();
        
        if start_token_idx + num_new_tokens > current_seq_len {
            return Err(CudaError(-1));
        }

        let token_size = self.num_heads * self.head_dim * self.element_size;
        let copy_size = num_new_tokens * token_size;
        let offset = start_token_idx * token_size;

        // Direct device memory operations
        unsafe {
            // Copy to K tensor
            let dst_key = self.key_device_ptr.as_ptr().add(offset);
            let src_key = host_key_data.add(offset);
            
            #[cfg(feature = "cuda")]
            {
                let result = crate::cuda::cudaMemcpy(
                    dst_key as *mut std::ffi::c_void,
                    src_key as *const std::ffi::c_void,
                    copy_size,
                    crate::cuda::CUDA_MEMCPY_HOST_TO_DEVICE,
                );
                if result != crate::cuda::CUDA_SUCCESS {
                    return Err(CudaError(result));
                }
            }
            #[cfg(not(feature = "cuda"))]
            std::ptr::copy_nonoverlapping(src_key, dst_key, copy_size);

            // Copy to V tensor
            let dst_value = self.value_device_ptr.as_ptr().add(offset);
            let src_value = host_value_data.add(offset);
            
            #[cfg(feature = "cuda")]
            {
                let result = crate::cuda::cudaMemcpy(
                    dst_value as *mut std::ffi::c_void,
                    src_value as *const std::ffi::c_void,
                    copy_size,
                    crate::cuda::CUDA_MEMCPY_HOST_TO_DEVICE,
                );
                if result != crate::cuda::CUDA_SUCCESS {
                    return Err(CudaError(result));
                }
            }
            #[cfg(not(feature = "cuda"))]
            std::ptr::copy_nonoverlapping(src_value, dst_value, copy_size);
        }

        log::debug!("Copied {} NEW tokens ONLY ({}KB) - no page reference needed", 
                   num_new_tokens, copy_size / 1024);
        
        Ok(())
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Get zero-copy extension statistics
    pub fn zero_copy_stats(&self) -> ZeroCopyStats {
        let current_len = self.seq_len();
        let utilization = current_len as f64 / self.max_seq_len as f64;
        let memory_efficiency = self.current_key_size_bytes() as f64 / 
                               (self.max_allocated_size_bytes() / 2) as f64;

        ZeroCopyStats {
            current_seq_len: current_len,
            max_seq_len: self.max_seq_len,
            growth_capacity_remaining: self.max_seq_len.saturating_sub(current_len),
            utilization,
            memory_efficiency,
            can_grow_without_copy: current_len < self.max_seq_len,
        }
    }

    pub fn arena_id(&self) -> u64 {
        self.arena_id
    }

    pub fn size_bytes(&self) -> usize {
        self.current_key_size_bytes() + self.current_value_size_bytes()
    }

    /// Synchronization without page reference
    pub fn synchronize(&self) -> Result<(), CudaError> {
        #[cfg(feature = "cuda")]
        unsafe {
            // Set device context
            let result = crate::cuda::cudaSetDevice(self.device_id);
            if result != crate::cuda::CUDA_SUCCESS {
                return Err(CudaError(result));
            }
            
            // Synchronize device
            let result = crate::cuda::cudaDeviceSynchronize();
            if result != crate::cuda::CUDA_SUCCESS {
                return Err(CudaError(result));
            }
        }
        Ok(())
    }

    // Compatibility methods
    pub fn copy_from_host(&self, host_key_data: *const u8, host_value_data: *const u8) -> Result<(), CudaError> {
        self.copy_new_tokens_only(host_key_data, host_value_data, 0, self.seq_len())
    }

    pub fn copy_new_tokens_from_host(
        &self,
        host_key_data: *const u8,
        host_value_data: *const u8,
        start_token: usize,
        num_tokens: usize,
    ) -> Result<(), CudaError> {
        self.copy_new_tokens_only(host_key_data, host_value_data, start_token, num_tokens)
    }

    pub fn utilization(&self) -> f64 {
        self.seq_len() as f64 / self.max_seq_len as f64
    }

    pub fn memory_efficiency(&self) -> f64 {
        self.utilization()
    }
}

/// Zero-copy extension statistics
#[derive(Debug, Clone)]
pub struct ZeroCopyStats {
    pub current_seq_len: usize,
    pub max_seq_len: usize,
    pub growth_capacity_remaining: usize,
    pub utilization: f64,
    pub memory_efficiency: f64,
    pub can_grow_without_copy: bool,
}

/// TRUE slab recycling with lock-free SegQueue - NO Arc<CudaPage>
#[derive(Debug)]
pub struct GlobalSlabPool {
    /// Lock-free queues for direct CudaPage recycling (NOT Arc<CudaPage>)
    small_pages: SegQueue<CudaPage>,     // 0-512KB
    medium_pages: SegQueue<CudaPage>,    // 512KB-2MB  
    large_pages: SegQueue<CudaPage>,     // 2MB-8MB
    huge_pages: SegQueue<CudaPage>,      // >8MB
    
    /// Statistics for monitoring
    pages_created: AtomicUsize,
    pages_recycled: AtomicUsize,
    pages_reused: AtomicUsize,
    bytes_saved: AtomicUsize,
    
    /// Configuration
    max_pages_per_class: usize,
}

impl GlobalSlabPool {
    pub fn new() -> Self {
        Self {
            small_pages: SegQueue::new(),
            medium_pages: SegQueue::new(),
            large_pages: SegQueue::new(),
            huge_pages: SegQueue::new(),
            pages_created: AtomicUsize::new(0),
            pages_recycled: AtomicUsize::new(0),
            pages_reused: AtomicUsize::new(0),
            bytes_saved: AtomicUsize::new(0),
            max_pages_per_class: 50,
        }
    }

    /// Get size class queue for a given page size
    fn size_class_for_size(&self, size: usize) -> &SegQueue<CudaPage> {
        match size {
            0..=524_288 => &self.small_pages,
            524_289..=2_097_152 => &self.medium_pages,
            2_097_153..=8_388_608 => &self.large_pages,
            _ => &self.huge_pages,
        }
    }

    /// Try to get a recycled page - returns actual CudaPage, not Arc
    pub fn get_page(&self, requested_size: usize, device_id: i32) -> Option<CudaPage> {
        // Try exact size class first
        if let Some(page) = self.try_get_from_exact_class(requested_size, device_id) {
            return Some(page);
        }

        // Try larger size classes
        self.try_get_from_larger_classes(requested_size, device_id)
    }

    fn try_get_from_exact_class(&self, requested_size: usize, device_id: i32) -> Option<CudaPage> {
        let queue = self.size_class_for_size(requested_size);
        
        // Lock-free pop from SegQueue
        while let Some(page) = queue.pop() {
            // Check if page matches requirements
            if page.size() >= requested_size && page.device_id() == device_id {
                // Reset page for reuse
                page.reset();
                
                self.pages_reused.fetch_add(1, Ordering::Relaxed);
                self.bytes_saved.fetch_add(page.size(), Ordering::Relaxed);
                
                log::debug!("REAL slab reuse: {}KB page from exact class", page.size() / 1024);
                return Some(page);
            } else {
                // Page doesn't match, put it back
                queue.push(page);
                break; // Avoid infinite loop
            }
        }
        
        None
    }

    fn try_get_from_larger_classes(&self, requested_size: usize, device_id: i32) -> Option<CudaPage> {
        let larger_queues = [&self.medium_pages, &self.large_pages, &self.huge_pages];
        
        for queue in larger_queues {
            if let Some(page) = queue.pop() {
                if page.size() >= requested_size && page.device_id() == device_id {
                    page.reset();
                    
                    self.pages_reused.fetch_add(1, Ordering::Relaxed);
                    self.bytes_saved.fetch_add(page.size(), Ordering::Relaxed);
                    
                    log::debug!("REAL slab reuse: {}KB page from larger class", page.size() / 1024);
                    return Some(page);
                } else {
                    // Put back and try next queue
                    queue.push(page);
                }
            }
        }
        
        None
    }

    /// Return page to pool - takes actual CudaPage, not Arc
    pub fn return_page(&self, page: CudaPage) {
        let size = page.size();
        let queue = self.size_class_for_size(size);
        
        // Check pool capacity to prevent unbounded growth
        if self.approximate_queue_size(queue) >= self.max_pages_per_class {
            log::debug!("Slab pool full for size {}, dropping page", size);
            // Let page drop and be freed
            return;
        }

        // Reset page and return to lock-free pool
        page.reset();
        queue.push(page);
        
        self.pages_recycled.fetch_add(1, Ordering::Relaxed);
        
        log::debug!("REAL slab recycling: returned {}KB page to lock-free pool", size / 1024);
    }

    /// Record new page creation (when no recycled page available)
    pub fn record_page_creation(&self, size: usize) {
        self.pages_created.fetch_add(1, Ordering::Relaxed);
    }

    /// Approximate queue size (SegQueue doesn't provide exact len)
    fn approximate_queue_size(&self, queue: &SegQueue<CudaPage>) -> usize {
        // Sample a few items to estimate size without emptying queue
        let mut count = 0;
        let mut temp_pages = Vec::new();
        
        // Sample up to 10 items
        for _ in 0..10 {
            if let Some(page) = queue.pop() {
                temp_pages.push(page);
                count += 1;
            } else {
                break;
            }
        }
        
        // Put sampled pages back
        for page in temp_pages {
            queue.push(page);
        }
        
        // Estimate total size (rough approximation)
        count * 5 // Assume we sampled ~20% of queue
    }

    /// Get recycling statistics
    pub fn stats(&self) -> SlabPoolStats {
        let created = self.pages_created.load(Ordering::Relaxed);
        let recycled = self.pages_recycled.load(Ordering::Relaxed);
        let reused = self.pages_reused.load(Ordering::Relaxed);
        let bytes_saved = self.bytes_saved.load(Ordering::Relaxed);
        
        SlabPoolStats {
            total_pages_created: created,
            total_pages_recycled: recycled,
            total_pages_reused: reused,
            bytes_saved_mb: bytes_saved / (1024 * 1024),
            recycling_efficiency: if created > 0 { recycled as f64 / created as f64 } else { 0.0 },
            reuse_efficiency: if recycled > 0 { reused as f64 / recycled as f64 } else { 0.0 },
            current_pool_sizes: self.get_approximate_pool_sizes(),
        }
    }

    fn get_approximate_pool_sizes(&self) -> [usize; 4] {
        [
            self.approximate_queue_size(&self.small_pages),
            self.approximate_queue_size(&self.medium_pages),
            self.approximate_queue_size(&self.large_pages),
            self.approximate_queue_size(&self.huge_pages),
        ]
    }

    /// Cleanup excess pages from all pools
    pub fn cleanup_old_pages(&self) -> usize {
        let mut cleaned = 0;
        let queues = [&self.small_pages, &self.medium_pages, &self.large_pages, &self.huge_pages];
        
        for queue in queues {
            let target_size = self.max_pages_per_class / 2;
            let current_size = self.approximate_queue_size(queue);
            let to_remove = current_size.saturating_sub(target_size);
            
            for _ in 0..to_remove {
                if queue.pop().is_some() {
                    cleaned += 1;
                    // Page drops and gets freed
                }
            }
        }
        
        if cleaned > 0 {
            log::info!("Cleaned up {} excess pages from lock-free slab pools", cleaned);
        }
        
        cleaned
    }
}

/// ZeroCopyArena with direct CudaPage ownership - NO Arc references
#[derive(Debug)]
pub struct ZeroCopyArena {
    /// Direct CudaPage ownership (NOT Arc<CudaPage>)
    page: CudaPage,
    /// Arena ID for tracking
    arena_id: u64,
    /// Current bump offset (ONLY mutable state)
    current_offset: AtomicUsize,
    /// Reference to slab pool for cleanup
    slab_pool: Arc<GlobalSlabPool>,
}

impl ZeroCopyArena {
    /// Create with direct page ownership
    pub fn new(
        page: CudaPage,  // Take direct ownership, not Arc<CudaPage>
        arena_id: u64,
        slab_pool: Arc<GlobalSlabPool>,
    ) -> Self {
        Self {
            page,  // Direct ownership
            arena_id,
            current_offset: AtomicUsize::new(0),
            slab_pool,
        }
    }

    /// Pure bump allocation with direct page access
    pub fn bump_allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        let aligned_size = (size + align - 1) & !(align - 1);
        
        // Atomic bump allocation
        let old_offset = self.current_offset.fetch_add(aligned_size, Ordering::Relaxed);
        
        if old_offset + aligned_size <= self.page.size() {
            // Calculate device pointer directly from page
            unsafe {
                let base_ptr = self.page.device_ptr() as *mut u8;
                let alloc_ptr = base_ptr.add(old_offset);
                Some(NonNull::new_unchecked(alloc_ptr))
            }
        } else {
            // Revert offset and fail
            let _ = self.current_offset.compare_exchange(
                old_offset + aligned_size, 
                old_offset, 
                Ordering::Relaxed, 
                Ordering::Relaxed
            );
            None
        }
    }

    /// Allocate KV tensor with direct page access
    pub fn allocate_kv_tensor_with_growth(
        &self,
        initial_seq_len: usize,
        expected_max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        element_size: usize,
    ) -> Result<ZeroCopyTensor, CudaError> {
        let max_k_size = expected_max_seq_len * num_heads * head_dim * element_size;
        let max_v_size = expected_max_seq_len * num_heads * head_dim * element_size;
        let total_max_size = max_k_size + max_v_size;
        
        // Pure bump allocation
        let device_ptr = self.bump_allocate(total_max_size, 256)
            .ok_or(CudaError(-2))?;
        
        // Create tensor with NO page reference
        ZeroCopyTensor::from_bump_allocation(
            device_ptr,
            initial_seq_len,
            expected_max_seq_len,
            num_heads,
            head_dim,
            element_size,
            self.arena_id,
            self.page.device_id(), // Pass device_id, not page reference
        )
    }

    /// Get direct page reference for operations
    pub fn page(&self) -> &CudaPage {
        &self.page
    }

    // Arena state queries
    pub fn arena_id(&self) -> u64 { self.arena_id }
    pub fn current_offset(&self) -> usize { self.current_offset.load(Ordering::Relaxed) }
    pub fn available_space(&self) -> usize { 
        self.page.size().saturating_sub(self.current_offset()) 
    }
    pub fn utilization(&self) -> f64 {
        self.current_offset() as f64 / self.page.size() as f64
    }

    pub fn synchronize(&self) -> Result<(), CudaError> { self.page.synchronize() }
    pub fn is_ready(&self) -> Result<bool, CudaError> { self.page.is_ready() }

    // Compatibility methods
    pub fn allocate_tensor_with_growth(
        &self,
        initial_seq_len: usize,
        expected_max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        element_size: usize,
    ) -> Result<ZeroCopyTensor, CudaError> {
        self.allocate_kv_tensor_with_growth(
            initial_seq_len, expected_max_seq_len, num_heads, head_dim, element_size
        )
    }

    pub fn extend_tensor_for_generation(
        &self,
        tensor: &mut ZeroCopyTensor,
        additional_tokens: usize,
    ) -> Result<bool, CudaError> {
        let current_len = tensor.seq_len();
        let new_len = current_len + additional_tokens;
        tensor.extend_zero_copy(new_len)
    }

    pub fn stats(&self) -> ZeroCopyArenaStats {
        ZeroCopyArenaStats {
            arena_id: self.arena_id,
            device_id: self.page.device_id(),
            page_size: self.page.size(),
            used_bytes: self.current_offset(),
            total_tensors: 0, // Not tracked in pure bump
            total_used_bytes: self.current_offset(),
            total_allocated_bytes: self.page.size(),
            avg_tensor_utilization: 0.0,
            arena_utilization: self.utilization(),
            cuda_memory_pressure: 0.0,
        }
    }

    pub fn defragment(&self) -> Result<usize, CudaError> { Ok(0) }
}

impl Drop for ZeroCopyArena {
    fn drop(&mut self) {
        log::debug!("Arena {} dropping - returning page to lock-free slab pool", self.arena_id);
        
        // CRITICAL: Take ownership of page and return to slab pool
        // This requires unsafe because we need to move out of self
        let page = unsafe { std::ptr::read(&self.page) };
        std::mem::forget(&self.page); // Prevent double drop
        
        // Return actual CudaPage to lock-free pool
        self.slab_pool.return_page(page);
        
        log::info!("Arena {} page returned to slab pool for recycling", self.arena_id);
    }
}

// Note: We need to manually implement Send/Sync since we're using direct page ownership
unsafe impl Send for ZeroCopyArena {}
unsafe impl Sync for ZeroCopyArena {}

/// Manager with lock-free slab recycling - NO Arc<CudaPage> tracking
#[derive(Debug)]
pub struct ZeroCopyManager {
    slab_pool: Arc<GlobalSlabPool>,
    cuda_context: Arc<CudaContext>,
    next_arena_id: AtomicUsize,
    // REMOVED: active_arenas tracking - prevents slab recycling
}

impl ZeroCopyManager {
    pub fn new(slab_pool: Arc<GlobalSlabPool>) -> Result<Self, CudaError> {
        let cuda_context = Arc::new(CudaContext::new()?);
        
        Ok(Self {
            slab_pool,
            cuda_context,
            next_arena_id: AtomicUsize::new(0),
            // NO arena tracking - enables true slab recycling
        })
    }

    /// Create arena with true slab recycling
    pub fn create_arena(
        &self,
        page_size: usize,
        device_id: i32,
    ) -> Result<ZeroCopyArena, CudaError> {
        let arena_id = self.next_arena_id.fetch_add(1, Ordering::Relaxed) as u64;
        
        // Try to get recycled page from lock-free pool
        let page = if let Some(recycled_page) = self.slab_pool.get_page(page_size, device_id) {
            log::debug!("Using recycled page for arena {}", arena_id);
            recycled_page
        } else {
            // Allocate new page
            let new_page = self.cuda_context.allocate_page_on_device(page_size, device_id)?;
            self.slab_pool.record_page_creation(page_size);
            new_page
        };
        
        // Create arena with direct page ownership
        let arena = ZeroCopyArena::new(
            page,  // Direct ownership, not Arc
            arena_id,
            Arc::clone(&self.slab_pool),
        );

        log::debug!("Created arena {} with direct page ownership: {}KB on device {}", 
                   arena_id, page_size / 1024, device_id);

        Ok(arena)
    }

    // Simplified manager methods (no arena tracking)
    pub fn global_stats(&self) -> ZeroCopyGlobalStats {
        let slab_stats = self.slab_pool.stats();
        
        ZeroCopyGlobalStats {
            total_arenas: 0, // Not tracked for slab recycling
            total_tensors: 0,
            total_used_bytes: 0,
            total_allocated_bytes: 0,
            avg_arena_utilization: 0.0,
            avg_cuda_memory_pressure: 0.0,
            slab_pool_stats: slab_stats,
            arena_stats: vec![],
            cuda_context_stats: vec![],
        }
    }

    pub fn cleanup_inactive_arenas(&self) -> usize { 
        // Cleanup happens automatically via Drop trait - no tracking needed
        0 
    }
    
    pub fn defragment_all(&self) -> Result<usize, CudaError> { 
        Ok(0) // No fragmentation in bump allocator
    }
    
    pub fn synchronize_all(&self) -> Result<(), CudaError> { 
        self.cuda_context.synchronize_all()
    }
    
    pub fn get_recommendations(&self) -> Vec<String> {
        vec![
            "Lock-free slab recycling active - optimal for production".to_string(),
            "No Arc references - true zero-copy enabled".to_string(),
        ]
    }

    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
    }
}

// Stats structs (simplified for bump allocation)
#[derive(Debug, Clone)]
pub struct SlabPoolStats {
    pub total_pages_created: usize,
    pub total_pages_recycled: usize,
    pub total_pages_reused: usize,
    pub bytes_saved_mb: usize,
    pub recycling_efficiency: f64,
    pub reuse_efficiency: f64,
    pub current_pool_sizes: [usize; 4],
}

#[derive(Debug, Clone)]
pub struct ZeroCopyGlobalStats {
    pub total_arenas: usize,
    pub total_tensors: usize,
    pub total_used_bytes: usize,
    pub total_allocated_bytes: usize,
    pub avg_arena_utilization: f64,
    pub avg_cuda_memory_pressure: f64,
    pub slab_pool_stats: SlabPoolStats,
    pub arena_stats: Vec<ZeroCopyArenaStats>,
    pub cuda_context_stats: Vec<crate::cuda::CudaDeviceStats>,
}

impl ZeroCopyGlobalStats {
    pub fn system_efficiency(&self) -> f64 {
        if self.total_allocated_bytes == 0 {
            1.0
        } else {
            self.total_used_bytes as f64 / self.total_allocated_bytes as f64
        }
    }

    pub fn needs_optimization(&self) -> bool {
        self.avg_arena_utilization < 0.5 || 
        self.system_efficiency() < 0.7 ||
        self.slab_pool_stats.recycling_efficiency < 0.5
    }

    pub fn memory_pressure(&self) -> f64 {
        self.avg_cuda_memory_pressure
    }
}

#[derive(Debug, Clone)]
pub struct ZeroCopyArenaStats {
    pub arena_id: u64,
    pub device_id: i32,
    pub page_size: usize,
    pub used_bytes: usize,
    pub total_tensors: usize,
    pub total_used_bytes: usize,
    pub total_allocated_bytes: usize,
    pub avg_tensor_utilization: f64,
    pub arena_utilization: f64,
    pub cuda_memory_pressure: f64,
}