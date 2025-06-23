// src/zero_copy.rs - TRUE bump allocation with pure pointer arithmetic
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ptr::NonNull;
use std::collections::HashMap;
use std::sync::Mutex;
use crate::cuda::{CudaPage, CudaError, CudaContext};

/// TRUE zero-copy tensor - just device pointers and atomic seq_len
#[derive(Debug)]
pub struct ZeroCopyTensor {
    /// Direct pointer to K tensor start in device memory
    key_device_ptr: NonNull<u8>,
    /// Direct pointer to V tensor start in device memory  
    value_device_ptr: NonNull<u8>,
    /// Current sequence length (ONLY mutable state)
    current_seq_len: AtomicUsize,
    /// Immutable tensor parameters
    max_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    element_size: usize,
    arena_id: u64,
    /// Keep page alive for memory safety
    _page_ref: Arc<CudaPage>,
}

impl ZeroCopyTensor {
    /// Create from pure bump allocation - NO complex metadata needed
    pub fn from_bump_allocation(
        device_ptr: NonNull<u8>,
        initial_seq_len: usize,
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        element_size: usize,
        arena_id: u64,
        page_ref: Arc<CudaPage>,
    ) -> Result<Self, CudaError> {
        // K tensor at start, V tensor after max K tensor space
        let key_device_ptr = device_ptr;
        
        let k_tensor_max_size = max_seq_len * num_heads * head_dim * element_size;
        let value_device_ptr = unsafe {
            NonNull::new_unchecked(device_ptr.as_ptr().add(k_tensor_max_size))
        };
        
        log::debug!("TRUE bump allocated tensor: initial={}, max={}, ptr={:p}", 
                   initial_seq_len, max_seq_len, device_ptr.as_ptr());
        
        Ok(ZeroCopyTensor {
            key_device_ptr,
            value_device_ptr,
            current_seq_len: AtomicUsize::new(initial_seq_len),
            max_seq_len,
            num_heads,
            head_dim,
            element_size,
            arena_id,
            _page_ref: page_ref,
        })
    }

    /// TRUE ZERO-COPY extension - ONLY atomic metadata update
    pub fn extend_zero_copy(&self, new_seq_len: usize) -> Result<bool, CudaError> {
        if new_seq_len > self.max_seq_len {
            return Ok(false); // Cannot extend beyond pre-allocated space
        }

        let old_seq_len = self.current_seq_len.load(Ordering::Relaxed);
        
        if new_seq_len <= old_seq_len {
            return Ok(true); // Already at or past this length
        }
        
        // PURE atomic update - this IS the entire extension operation
        self.current_seq_len.store(new_seq_len, Ordering::Release);
        
        log::debug!("TRUE zero-copy extension: {} -> {} tokens (ZERO operations)", 
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

    /// Get total current size in bytes
    pub fn size_bytes(&self) -> usize {
        self.current_key_size_bytes() + self.current_value_size_bytes()
    }

    /// Get utilization ratio
    pub fn utilization(&self) -> f64 {
        self.seq_len() as f64 / self.max_seq_len as f64
    }

    /// Get memory efficiency
    pub fn memory_efficiency(&self) -> f64 {
        self.utilization()
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

        // Direct device memory copy - no page reference needed
        unsafe {
            let dst_key = self.key_device_ptr.as_ptr().add(offset);
            let src_key = host_key_data.add(offset);
            std::ptr::copy_nonoverlapping(src_key, dst_key, copy_size);

            let dst_value = self.value_device_ptr.as_ptr().add(offset);
            let src_value = host_value_data.add(offset);
            std::ptr::copy_nonoverlapping(src_value, dst_value, copy_size);
        }

        log::debug!("Copied {} NEW tokens ONLY ({}KB)", 
                   num_new_tokens, copy_size / 1024);
        
        Ok(())
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self._page_ref.device_id()
    }

    /// Get zero-copy extension statistics
    pub fn zero_copy_stats(&self) -> ZeroCopyStats {
        let current_len = self.seq_len();
        ZeroCopyStats {
            current_seq_len: current_len,
            max_seq_len: self.max_seq_len,
            growth_capacity_remaining: self.max_seq_len.saturating_sub(current_len),
            utilization: self.utilization(),
            memory_efficiency: self.memory_efficiency(),
            can_grow_without_copy: current_len < self.max_seq_len,
        }
    }

    pub fn arena_id(&self) -> u64 {
        self.arena_id
    }

    /// Minimal synchronization - use page reference
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self._page_ref.synchronize()
    }

    // Compatibility methods for existing FFI
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

/// TRUE bump allocator arena - just offset tracking, no metadata
#[derive(Debug)]
pub struct ZeroCopyArena {
    /// REAL CUDA page with device memory
    page: Arc<CudaPage>,
    /// Arena ID for tracking
    arena_id: u64,
    /// Current bump offset (ONLY state that changes)
    current_offset: AtomicUsize,
    /// Reference to slab pool for cleanup
    slab_pool: Arc<GlobalSlabPool>,
}

impl ZeroCopyArena {
    pub fn new(
        page: CudaPage, 
        arena_id: u64,
        slab_pool: Arc<GlobalSlabPool>,
    ) -> Self {
        Self {
            page: Arc::new(page),
            arena_id,
            current_offset: AtomicUsize::new(0),  // Start at beginning
            slab_pool,
        }
    }

    /// PURE bump allocation - just increment offset, return pointer
    pub fn bump_allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        // Align size to boundary
        let aligned_size = (size + align - 1) & !(align - 1);
        
        // Atomic bump: load current, try to increment
        let old_offset = self.current_offset.fetch_add(aligned_size, Ordering::Relaxed);
        
        // Check if allocation fits in page
        if old_offset + aligned_size <= self.page.size() {
            // Calculate device pointer at this offset
            unsafe {
                let base_ptr = self.page.device_ptr() as *mut u8;
                let alloc_ptr = base_ptr.add(old_offset);
                Some(NonNull::new_unchecked(alloc_ptr))
            }
        } else {
            // Allocation failed - page full
            // Try to revert the offset (best effort)
            let _ = self.current_offset.compare_exchange(
                old_offset + aligned_size, 
                old_offset, 
                Ordering::Relaxed, 
                Ordering::Relaxed
            );
            None
        }
    }

    /// Allocate KV tensor with TRUE bump allocation - no metadata tracking
    pub fn allocate_kv_tensor_with_growth(
        &self,
        initial_seq_len: usize,
        expected_max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        element_size: usize,
    ) -> Result<ZeroCopyTensor, CudaError> {
        // Calculate size for MAXIMUM sequence length (for zero-copy growth)
        let max_k_size = expected_max_seq_len * num_heads * head_dim * element_size;
        let max_v_size = expected_max_seq_len * num_heads * head_dim * element_size;
        let total_max_size = max_k_size + max_v_size;
        
        // PURE bump allocation - just get a pointer
        let device_ptr = self.bump_allocate(total_max_size, 256)
            .ok_or(CudaError(-2))?; // Arena full
        
        // Create tensor with direct device pointer - NO arena state tracking
        ZeroCopyTensor::from_bump_allocation(
            device_ptr,
            initial_seq_len,
            expected_max_seq_len,
            num_heads,
            head_dim,
            element_size,
            self.arena_id,
            Arc::clone(&self.page),
        )
    }

    /// Compatibility alias
    pub fn allocate_tensor_with_growth(
        &self,
        initial_seq_len: usize,
        expected_max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        element_size: usize,
    ) -> Result<ZeroCopyTensor, CudaError> {
        self.allocate_kv_tensor_with_growth(
            initial_seq_len, 
            expected_max_seq_len, 
            num_heads, 
            head_dim, 
            element_size
        )
    }

    /// Extension is handled entirely by the tensor itself - no arena involvement
    pub fn extend_tensor_for_generation(
        &self,
        tensor: &mut ZeroCopyTensor,
        additional_tokens: usize,
    ) -> Result<bool, CudaError> {
        let current_len = tensor.seq_len();
        let new_len = current_len + additional_tokens;
        
        // Tensor handles its own extension - arena doesn't track anything
        tensor.extend_zero_copy(new_len)
    }

    // Arena state queries (minimal)
    pub fn arena_id(&self) -> u64 { self.arena_id }
    pub fn current_offset(&self) -> usize { self.current_offset.load(Ordering::Relaxed) }
    pub fn available_space(&self) -> usize { 
        self.page.size().saturating_sub(self.current_offset()) 
    }
    pub fn utilization(&self) -> f64 {
        self.current_offset() as f64 / self.page.size() as f64
    }

    pub fn stats(&self) -> ZeroCopyArenaStats {
        ZeroCopyArenaStats {
            arena_id: self.arena_id,
            device_id: self.page.device_id(),
            page_size: self.page.size(),
            used_bytes: self.current_offset(),
            total_tensors: 0, // Not tracked in bump allocator
            total_used_bytes: self.current_offset(),
            total_allocated_bytes: self.page.size(),
            avg_tensor_utilization: 0.0, // Not tracked
            arena_utilization: self.utilization(),
            cuda_memory_pressure: 0.0,
        }
    }

    // Minimal operations
    pub fn defragment(&self) -> Result<usize, CudaError> { Ok(0) } // No fragmentation in bump allocator
    pub fn synchronize(&self) -> Result<(), CudaError> { self.page.synchronize() }
    pub fn cuda_page(&self) -> &Arc<CudaPage> { &self.page }
    pub fn is_ready(&self) -> Result<bool, CudaError> { self.page.is_ready() }
}

impl Drop for ZeroCopyArena {
    fn drop(&mut self) {
        log::debug!("Bump arena {} dropping - returning page to slab pool", self.arena_id);
        self.slab_pool.return_page(Arc::clone(&self.page));
    }
}

// Simplified global slab pool for pure bump allocation
#[derive(Debug)]
pub struct GlobalSlabPool {
    small_pages: Mutex<Vec<Arc<CudaPage>>>,
    medium_pages: Mutex<Vec<Arc<CudaPage>>>,
    large_pages: Mutex<Vec<Arc<CudaPage>>>,
    huge_pages: Mutex<Vec<Arc<CudaPage>>>,
    pages_created: AtomicUsize,
    pages_recycled: AtomicUsize,
    pages_reused: AtomicUsize,
    bytes_saved: AtomicUsize,
    max_pages_per_class: usize,
}

impl GlobalSlabPool {
    pub fn new() -> Self {
        Self {
            small_pages: Mutex::new(Vec::new()),
            medium_pages: Mutex::new(Vec::new()),
            large_pages: Mutex::new(Vec::new()),
            huge_pages: Mutex::new(Vec::new()),
            pages_created: AtomicUsize::new(0),
            pages_recycled: AtomicUsize::new(0),
            pages_reused: AtomicUsize::new(0),
            bytes_saved: AtomicUsize::new(0),
            max_pages_per_class: 50,
        }
    }

    fn size_class_for_size(&self, size: usize) -> &Mutex<Vec<Arc<CudaPage>>> {
        match size {
            0..=524_288 => &self.small_pages,      // 0-512KB
            524_289..=2_097_152 => &self.medium_pages,   // 512KB-2MB
            2_097_153..=8_388_608 => &self.large_pages,   // 2MB-8MB
            _ => &self.huge_pages,                         // >8MB
        }
    }

    pub fn get_page(&self, requested_size: usize, device_id: i32) -> Option<Arc<CudaPage>> {
        // Try exact size class first
        if let Some(page) = self.try_get_from_class(requested_size, device_id) {
            return Some(page);
        }

        // Try larger size classes
        let size_classes = [&self.medium_pages, &self.large_pages, &self.huge_pages];
        for class in &size_classes {
            if let Ok(mut pages) = class.lock() {
                if let Some(pos) = pages.iter().position(|p| {
                    p.size() >= requested_size && p.device_id() == device_id
                }) {
                    let page = pages.remove(pos);
                    self.pages_reused.fetch_add(1, Ordering::Relaxed);
                    self.bytes_saved.fetch_add(page.size(), Ordering::Relaxed);
                    
                    // Reset page for reuse (clear bump offset)
                    page.reset();
                    
                    log::debug!("Recycled page: {}KB from larger class", page.size() / 1024);
                    return Some(page);
                }
            }
        }

        None
    }

    fn try_get_from_class(&self, requested_size: usize, device_id: i32) -> Option<Arc<CudaPage>> {
        let class = self.size_class_for_size(requested_size);
        
        if let Ok(mut pages) = class.lock() {
            if let Some(pos) = pages.iter().position(|p| {
                p.size() >= requested_size && p.device_id() == device_id
            }) {
                let page = pages.remove(pos);
                self.pages_reused.fetch_add(1, Ordering::Relaxed);
                self.bytes_saved.fetch_add(page.size(), Ordering::Relaxed);
                
                // Reset page for reuse
                page.reset();
                
                log::debug!("Recycled page: {}KB from exact class", page.size() / 1024);
                return Some(page);
            }
        }
        
        None
    }

    pub fn return_page(&self, page: Arc<CudaPage>) {
        let size = page.size();
        let class = self.size_class_for_size(size);
        
        if let Ok(mut pages) = class.lock() {
            if pages.len() >= self.max_pages_per_class {
                log::debug!("Slab pool class full, dropping page ({}KB)", size / 1024);
                return; // Page will be dropped and freed
            }

            // Reset page for next use
            page.reset();
            pages.push(page);
            self.pages_recycled.fetch_add(1, Ordering::Relaxed);
            
            log::debug!("Returned page to slab pool: {}KB", size / 1024);
        }
    }

    pub fn record_page_creation(&self, size: usize) {
        self.pages_created.fetch_add(1, Ordering::Relaxed);
    }

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
            current_pool_sizes: self.get_pool_sizes(),
        }
    }

    fn get_pool_sizes(&self) -> [usize; 4] {
        [
            self.small_pages.lock().map(|p| p.len()).unwrap_or(0),
            self.medium_pages.lock().map(|p| p.len()).unwrap_or(0),
            self.large_pages.lock().map(|p| p.len()).unwrap_or(0),
            self.huge_pages.lock().map(|p| p.len()).unwrap_or(0),
        ]
    }

    pub fn cleanup_old_pages(&self) -> usize {
        let mut cleaned = 0;
        let classes = [&self.small_pages, &self.medium_pages, &self.large_pages, &self.huge_pages];
        
        for class in &classes {
            if let Ok(mut pages) = class.lock() {
                let target_size = self.max_pages_per_class / 2;
                let to_remove = pages.len().saturating_sub(target_size);
                
                for _ in 0..to_remove {
                    if pages.pop().is_some() {
                        cleaned += 1;
                    }
                }
            }
        }
        
        if cleaned > 0 {
            log::info!("Cleaned up {} old pages from slab pools", cleaned);
        }
        
        cleaned
    }
}

/// Manager with pure bump allocation - minimal state tracking
#[derive(Debug)]
pub struct ZeroCopyManager {
    slab_pool: Arc<GlobalSlabPool>,
    cuda_context: Arc<CudaContext>,
    next_arena_id: AtomicUsize,
}

impl ZeroCopyManager {
    pub fn new(slab_pool: Arc<GlobalSlabPool>) -> Result<Self, CudaError> {
        let cuda_context = Arc::new(CudaContext::new()?);
        
        Ok(Self {
            slab_pool,
            cuda_context,
            next_arena_id: AtomicUsize::new(0),
        })
    }

    /// Create arena with pure bump allocation
    pub fn create_arena(
        &self,
        page_size: usize,
        device_id: i32,
    ) -> Result<ZeroCopyArena, CudaError> {
        let arena_id = self.next_arena_id.fetch_add(1, Ordering::Relaxed) as u64;
        
        // Try to get recycled page first
        let page = if let Some(recycled_page) = self.slab_pool.get_page(page_size, device_id) {
            log::debug!("Using recycled page for arena {}", arena_id);
            match Arc::try_unwrap(recycled_page) {
                Ok(page) => page,
                Err(_) => {
                    // Can't unwrap, allocate new
                    let new_page = self.cuda_context.allocate_page_on_device(page_size, device_id)?;
                    self.slab_pool.record_page_creation(page_size);
                    new_page
                }
            }
        } else {
            // Allocate new page
            let new_page = self.cuda_context.allocate_page_on_device(page_size, device_id)?;
            self.slab_pool.record_page_creation(page_size);
            new_page
        };
        
        // Create pure bump allocator arena
        let arena = ZeroCopyArena::new(
            page,
            arena_id,
            Arc::clone(&self.slab_pool),
        );

        log::debug!("Created pure bump arena {}: {}KB on device {}", 
                   arena_id, page_size / 1024, device_id);

        Ok(arena)
    }

    // Minimal manager methods
    pub fn global_stats(&self) -> ZeroCopyGlobalStats {
        let slab_stats = self.slab_pool.stats();
        
        ZeroCopyGlobalStats {
            total_arenas: 0, // Not tracked in pure bump allocation
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

    pub fn cleanup_inactive_arenas(&self) -> usize { 0 } // No tracking in pure bump
    pub fn defragment_all(&self) -> Result<usize, CudaError> { Ok(0) } // No fragmentation
    pub fn synchronize_all(&self) -> Result<(), CudaError> { Ok(()) }
    
    pub fn get_recommendations(&self) -> Vec<String> {
        vec!["Pure bump allocation active - maximum performance mode".to_string()]
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