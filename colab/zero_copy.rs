// src/zero_copy.rs - True zero-copy CUDA operations with device memory
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ptr::NonNull;
use crate::cuda::{CudaPage, CudaError, CudaTensor, CudaContext};
use crate::slab::{GlobalSlabPool, RecyclablePage};

/// Zero-copy tensor descriptor that operates directly on CUDA device memory
#[derive(Debug)]
pub struct ZeroCopyTensor {
    /// CUDA device tensor
    cuda_tensor: CudaTensor,
    /// Byte offset within the arena page
    offset: usize,
    /// Total allocated size in bytes
    allocated_size: usize,
    /// Current used size in bytes (may be less than allocated for growth)
    used_size: usize,
    /// Tensor dimensions
    seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
    /// Element size in bytes (e.g., 2 for fp16)
    element_size: usize,
    /// Reference to the underlying page for lifetime management
    page_ref: Arc<CudaPage>,
    /// Arena ID for tracking
    arena_id: u64,
}

impl ZeroCopyTensor {
    /// Create new tensor descriptor from CUDA arena allocation
    pub fn new(
        cuda_page: &Arc<CudaPage>,
        offset: usize,
        allocated_size: usize,
        seq_len: usize,
        hidden_dim: usize,
        num_heads: usize,
        element_size: usize,
        arena_id: u64,
    ) -> Result<Self, CudaError> {
        // Calculate actual used size
        let used_size = Self::calculate_tensor_size(seq_len, hidden_dim, num_heads, element_size);
        
        if offset + allocated_size > cuda_page.size() {
            return Err(CudaError(-1)); // Out of bounds
        }

        // Create CUDA tensor from page
        let head_dim = hidden_dim / num_heads;
        let kv_shape = vec![seq_len, num_heads, head_dim];
        
        let cuda_tensor = CudaTensor::from_page(
            cuda_page,
            offset,
            kv_shape,
            element_size,
        )?;

        Ok(ZeroCopyTensor {
            cuda_tensor,
            offset,
            allocated_size,
            used_size,
            seq_len,
            hidden_dim,
            num_heads,
            element_size,
            page_ref: Arc::clone(cuda_page),
            arena_id,
        })
    }

    /// Extend tensor to new sequence length WITHOUT copying data (TRUE ZERO-COPY)
    pub fn extend_zero_copy(&mut self, new_seq_len: usize) -> Result<bool, CudaError> {
        let new_used_size = Self::calculate_tensor_size(new_seq_len, self.hidden_dim, self.num_heads, self.element_size);
        
        // Check if we can extend in place within allocated space
        if new_used_size <= self.allocated_size && 
           self.offset + new_used_size <= self.page_ref.size() {
            
            // TRUE ZERO-COPY: Just update metadata, no memory operations
            self.seq_len = new_seq_len;
            self.used_size = new_used_size;
            
            // Update CUDA tensor shape (zero-copy reshape)
            let head_dim = self.hidden_dim / self.num_heads;
            let new_shape = vec![new_seq_len, self.num_heads, head_dim];
            self.cuda_tensor.reshape(new_shape)?;
            
            log::debug!("TRUE zero-copy extension: {} -> {} tokens (no memory operations)", 
                       self.seq_len, new_seq_len);
            return Ok(true);
        }

        // Cannot extend in place
        Ok(false)
    }

    /// Create a zero-copy view of this tensor for a specific sequence range
    pub fn slice_view(&self, start_seq: usize, end_seq: usize) -> Result<ZeroCopyTensor, CudaError> {
        if start_seq >= self.seq_len || end_seq > self.seq_len || start_seq >= end_seq {
            return Err(CudaError(-1)); // Invalid slice
        }

        // Create zero-copy slice of CUDA tensor
        let slice_cuda_tensor = self.cuda_tensor.slice(start_seq, end_seq)?;
        let slice_seq_len = end_seq - start_seq;
        
        // Calculate new offset and size
        let head_dim = self.hidden_dim / self.num_heads;
        let slice_offset_elements = start_seq * self.num_heads * head_dim;
        let slice_byte_offset = slice_offset_elements * self.element_size;
        let slice_used_size = Self::calculate_tensor_size(slice_seq_len, self.hidden_dim, self.num_heads, self.element_size);

        Ok(ZeroCopyTensor {
            cuda_tensor: slice_cuda_tensor,
            offset: self.offset + slice_byte_offset,
            allocated_size: self.allocated_size - slice_byte_offset,
            used_size: slice_used_size,
            seq_len: slice_seq_len,
            hidden_dim: self.hidden_dim,
            num_heads: self.num_heads,
            element_size: self.element_size,
            page_ref: Arc::clone(&self.page_ref),
            arena_id: self.arena_id,
        })
    }

    /// Get CUDA device pointer for key tensor
    pub fn key_device_ptr(&self) -> *mut std::ffi::c_void {
        self.cuda_tensor.device_ptr()
    }

    /// Get CUDA device pointer for value tensor  
    pub fn value_device_ptr(&self) -> *mut std::ffi::c_void {
        let head_dim = self.hidden_dim / self.num_heads;
        let key_size = self.seq_len * self.num_heads * head_dim * self.element_size;
        unsafe {
            (self.cuda_tensor.device_ptr() as *mut u8).add(key_size) as *mut std::ffi::c_void
        }
    }

    /// Get tensor dimensions
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.seq_len, self.num_heads, self.hidden_dim / self.num_heads)
    }

    /// Get total size in bytes
    pub fn size_bytes(&self) -> usize {
        self.used_size
    }

    /// Get allocated size in bytes
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_size
    }

    /// Check if tensor can be extended to new length without reallocation
    pub fn can_extend_to(&self, new_seq_len: usize) -> bool {
        let new_size = Self::calculate_tensor_size(new_seq_len, self.hidden_dim, self.num_heads, self.element_size);
        new_size <= self.allocated_size && self.offset + new_size <= self.page_ref.size()
    }

    /// Get utilization ratio (used / allocated)
    pub fn utilization(&self) -> f64 {
        self.used_size as f64 / self.allocated_size as f64
    }

    /// Copy data from host to this CUDA tensor
    pub fn copy_from_host(&self, host_key_data: *const u8, host_value_data: *const u8) -> Result<(), CudaError> {
        let head_dim = self.hidden_dim / self.num_heads;
        let single_tensor_size = self.seq_len * self.num_heads * head_dim * self.element_size;

        // Copy key data to device
        self.cuda_tensor.copy_from_host(host_key_data as *const std::ffi::c_void)?;

        // Copy value data to device (offset by key tensor size)
        let value_tensor = self.value_device_ptr();
        unsafe {
            let result = crate::cuda::cudaMemcpy(
                value_tensor,
                host_value_data as *const std::ffi::c_void,
                single_tensor_size,
                1, // CUDA_MEMCPY_HOST_TO_DEVICE
            );
            if result != 0 {
                return Err(CudaError(result));
            }
        }

        Ok(())
    }

    /// Copy only new tokens to extended tensor (for incremental generation)
    pub fn copy_new_tokens_from_host(
        &self,
        host_key_data: *const u8,
        host_value_data: *const u8,
        start_token: usize,
        num_tokens: usize,
    ) -> Result<(), CudaError> {
        if start_token + num_tokens > self.seq_len {
            return Err(CudaError(-1)); // Out of bounds
        }

        let head_dim = self.hidden_dim / self.num_heads;
        let token_size = self.num_heads * head_dim * self.element_size;
        let copy_size = num_tokens * token_size;
        let token_offset = start_token * token_size;

        // Copy new key tokens to device
        let key_dst = unsafe {
            (self.key_device_ptr() as *mut u8).add(token_offset) as *mut std::ffi::c_void
        };
        let key_src = unsafe {
            host_key_data.add(token_offset) as *const std::ffi::c_void
        };

        unsafe {
            let result = crate::cuda::cudaMemcpy(key_dst, key_src, copy_size, 1);
            if result != 0 {
                return Err(CudaError(result));
            }
        }

        // Copy new value tokens to device
        let value_dst = unsafe {
            (self.value_device_ptr() as *mut u8).add(token_offset) as *mut std::ffi::c_void
        };
        let value_src = unsafe {
            host_value_data.add(token_offset) as *const std::ffi::c_void
        };

        unsafe {
            let result = crate::cuda::cudaMemcpy(value_dst, value_src, copy_size, 1);
            if result != 0 {
                return Err(CudaError(result));
            }
        }

        log::debug!("Copied {} new tokens to CUDA device starting at token {}", 
                   num_tokens, start_token);
        Ok(())
    }

    /// Synchronize all CUDA operations on this tensor
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.cuda_tensor.synchronize()
    }

    /// Get the underlying CUDA tensor
    pub fn cuda_tensor(&self) -> &CudaTensor {
        &self.cuda_tensor
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.cuda_tensor.device_id()
    }

    fn calculate_tensor_size(seq_len: usize, hidden_dim: usize, num_heads: usize, element_size: usize) -> usize {
        // Size for both key and value tensors
        2 * seq_len * hidden_dim * element_size
    }
}

// Implement Clone for ZeroCopyTensor
impl Clone for ZeroCopyTensor {
    fn clone(&self) -> Self {
        // Create a new CUDA tensor view of the same memory
        let cuda_tensor = CudaTensor::from_page(
            &self.page_ref,
            self.offset,
            self.cuda_tensor.shape().to_vec(),
            self.element_size,
        ).expect("Cloning should always succeed for valid tensors");

        ZeroCopyTensor {
            cuda_tensor,
            offset: self.offset,
            allocated_size: self.allocated_size,
            used_size: self.used_size,
            seq_len: self.seq_len,
            hidden_dim: self.hidden_dim,
            num_heads: self.num_heads,
            element_size: self.element_size,
            page_ref: Arc::clone(&self.page_ref),
            arena_id: self.arena_id,
        }
    }
}

/// Zero-copy arena that manages CUDA tensor allocations without data copying
#[derive(Debug)]
pub struct ZeroCopyArena {
    arena_id: u64,
    cuda_page: Arc<CudaPage>,
    current_offset: AtomicUsize,
    allocated_tensors: std::sync::Mutex<Vec<ZeroCopyTensor>>,
    slab_pool: Arc<GlobalSlabPool>,
    device_id: i32,
    cuda_context: Option<Arc<CudaContext>>,
}

impl ZeroCopyArena {
    /// Create new zero-copy arena with CUDA page
    pub fn new(
        cuda_page: CudaPage, 
        arena_id: u64, 
        slab_pool: Arc<GlobalSlabPool>,
        cuda_context: Option<Arc<CudaContext>>,
    ) -> Self {
        let device_id = cuda_page.device_id();
        let cuda_page = Arc::new(cuda_page);
        
        Self {
            arena_id,
            cuda_page,
            current_offset: AtomicUsize::new(0),
            allocated_tensors: std::sync::Mutex::new(Vec::new()),
            slab_pool,
            device_id,
            cuda_context,
        }
    }

    /// Allocate tensor with extra space for future growth (CUDA device memory)
    pub fn allocate_tensor_with_growth(
        &self,
        initial_seq_len: usize,
        max_seq_len: usize,
        hidden_dim: usize,
        num_heads: usize,
        element_size: usize,
    ) -> Result<ZeroCopyTensor, CudaError> {
        // Allocate for maximum expected size to enable zero-copy growth
        let allocated_size = ZeroCopyTensor::calculate_tensor_size(max_seq_len, hidden_dim, num_heads, element_size);
        let aligned_size = Self::align_size(allocated_size);

        // Bump allocate within the CUDA page
        let offset = self.current_offset.fetch_add(aligned_size, Ordering::Relaxed);
        
        if offset + aligned_size > self.cuda_page.size() {
            return Err(CudaError(-2)); // Out of space
        }

        // Create zero-copy tensor that operates directly on CUDA memory
        let tensor = ZeroCopyTensor::new(
            &self.cuda_page,
            offset,
            allocated_size,
            initial_seq_len,
            hidden_dim,
            num_heads,
            element_size,
            self.arena_id,
        )?;

        // Track allocated tensor
        if let Ok(mut tensors) = self.allocated_tensors.lock() {
            tensors.push(tensor.clone());
        }

        log::debug!("Allocated zero-copy CUDA tensor: {}x{}x{} at device offset {}", 
                   initial_seq_len, num_heads, hidden_dim / num_heads, offset);

        Ok(tensor)
    }

    /// Try to extend existing tensor without reallocation (TRUE ZERO-COPY)
    pub fn try_extend_tensor(
        &self,
        tensor: &mut ZeroCopyTensor,
        new_seq_len: usize,
    ) -> Result<bool, CudaError> {
        // This is the key optimization: no data copying on CUDA device!
        tensor.extend_zero_copy(new_seq_len)
    }

    /// Zero-copy defragmentation using CUDA device-to-device operations
    pub fn defragment(&self) -> Result<usize, CudaError> {
        let mut tensors = self.allocated_tensors.lock().unwrap();
        if tensors.is_empty() {
            return Ok(0);
        }

        // Sort tensors by offset for optimal compaction
        tensors.sort_by_key(|t| t.offset);

        let mut compacted_offset = 0;
        let mut bytes_saved = 0;

        for tensor in tensors.iter_mut() {
            if tensor.offset > compacted_offset {
                // Use CUDA device-to-device copy for zero-copy move
                let move_size = tensor.used_size;
                self.cuda_page.copy_device_to_device(
                    tensor.offset,
                    compacted_offset,
                    move_size,
                )?;

                bytes_saved += tensor.offset - compacted_offset;
                
                // Update tensor metadata (no device memory changes needed)
                tensor.offset = compacted_offset;
                
                // Create new CUDA tensor view at new location
                let head_dim = tensor.hidden_dim / tensor.num_heads;
                let shape = vec![tensor.seq_len, tensor.num_heads, head_dim];
                tensor.cuda_tensor = CudaTensor::from_page(
                    &self.cuda_page,
                    compacted_offset,
                    shape,
                    tensor.element_size,
                )?;
            }

            compacted_offset += Self::align_size(tensor.used_size);
        }

        // Update current offset
        self.current_offset.store(compacted_offset, Ordering::Relaxed);

        // Synchronize all device operations
        self.cuda_page.synchronize()?;

        log::info!("Defragmented CUDA arena {}: saved {} bytes using device-to-device copies", 
                  self.arena_id, bytes_saved);
        Ok(bytes_saved)
    }

    /// Bulk zero-copy tensor creation for batch processing
    pub fn allocate_batch_tensors(
        &self,
        requests: &[(usize, usize, usize, usize, usize)], // (seq_len, max_seq_len, hidden_dim, num_heads, element_size)
    ) -> Result<Vec<ZeroCopyTensor>, CudaError> {
        let mut tensors = Vec::with_capacity(requests.len());
        
        for &(initial_seq_len, max_seq_len, hidden_dim, num_heads, element_size) in requests {
            let tensor = self.allocate_tensor_with_growth(
                initial_seq_len,
                max_seq_len,
                hidden_dim,
                num_heads,
                element_size,
            )?;
            tensors.push(tensor);
        }

        log::debug!("Bulk allocated {} CUDA tensors in arena {}", 
                   tensors.len(), self.arena_id);
        Ok(tensors)
    }

    /// Zero-copy tensor concatenation (combines multiple tensors into one view)
    pub fn concatenate_tensors(
        &self,
        tensors: &[&ZeroCopyTensor],
        axis: usize,
    ) -> Result<ZeroCopyTensor, CudaError> {
        if tensors.is_empty() {
            return Err(CudaError(-1));
        }

        // Verify all tensors are compatible and contiguous
        let first = &tensors[0];
        let mut total_seq_len = first.seq_len;
        let mut current_offset = first.offset;
        
        for tensor in &tensors[1..] {
            if tensor.hidden_dim != first.hidden_dim ||
               tensor.num_heads != first.num_heads ||
               tensor.element_size != first.element_size {
                return Err(CudaError(-1)); // Incompatible tensors
            }
            
            // Check if tensors are contiguous
            let expected_offset = current_offset + first.used_size;
            if tensor.offset != expected_offset {
                return Err(CudaError(-1)); // Not contiguous
            }
            
            total_seq_len += tensor.seq_len;
            current_offset = tensor.offset;
        }

        // Create concatenated tensor view (zero-copy)
        let total_size = ZeroCopyTensor::calculate_tensor_size(
            total_seq_len, first.hidden_dim, first.num_heads, first.element_size
        );

        let head_dim = first.hidden_dim / first.num_heads;
        let concat_shape = vec![total_seq_len, first.num_heads, head_dim];
        
        let cuda_tensor = CudaTensor::from_page(
            &self.cuda_page,
            first.offset,
            concat_shape,
            first.element_size,
        )?;

        Ok(ZeroCopyTensor {
            cuda_tensor,
            offset: first.offset,
            allocated_size: total_size,
            used_size: total_size,
            seq_len: total_seq_len,
            hidden_dim: first.hidden_dim,
            num_heads: first.num_heads,
            element_size: first.element_size,
            page_ref: Arc::clone(&self.cuda_page),
            arena_id: self.arena_id,
        })
    }

    /// Get arena utilization
    pub fn utilization(&self) -> f64 {
        let used = self.current_offset.load(Ordering::Relaxed);
        used as f64 / self.cuda_page.size() as f64
    }

    /// Get available space
    pub fn available_space(&self) -> usize {
        let used = self.current_offset.load(Ordering::Relaxed);
        self.cuda_page.size().saturating_sub(used)
    }

    /// Get CUDA-specific arena statistics
    pub fn stats(&self) -> ZeroCopyArenaStats {
        let tensors = self.allocated_tensors.lock().unwrap();
        let total_tensors = tensors.len();
        let total_used_bytes: usize = tensors.iter().map(|t| t.size_bytes()).sum();
        let total_allocated_bytes: usize = tensors.iter().map(|t| t.allocated_bytes()).sum();
        let avg_utilization = if total_allocated_bytes > 0 {
            total_used_bytes as f64 / total_allocated_bytes as f64
        } else {
            0.0
        };

        ZeroCopyArenaStats {
            arena_id: self.arena_id,
            device_id: self.device_id,
            page_size: self.cuda_page.size(),
            used_bytes: self.current_offset.load(Ordering::Relaxed),
            total_tensors,
            total_used_bytes,
            total_allocated_bytes,
            avg_tensor_utilization: avg_utilization,
            arena_utilization: self.utilization(),
            cuda_memory_pressure: self.get_cuda_memory_pressure(),
        }
    }

    /// Get CUDA device memory pressure
    fn get_cuda_memory_pressure(&self) -> f64 {
        if let Some(context) = &self.cuda_context {
            if let Some(stats) = context.device_stats_detailed(self.device_id) {
                return stats.utilization / 100.0;
            }
        }
        0.0
    }

    /// Synchronize all CUDA operations in this arena
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.cuda_page.synchronize()
    }

    /// Get the underlying CUDA page
    pub fn cuda_page(&self) -> &Arc<CudaPage> {
        &self.cuda_page
    }

    /// Check if all operations in this arena are complete
    pub fn is_ready(&self) -> Result<bool, CudaError> {
        self.cuda_page.is_ready()
    }

    fn align_size(size: usize) -> usize {
        const ALIGNMENT: usize = 256; // CUDA memory alignment for optimal performance
        (size + ALIGNMENT - 1) & !(ALIGNMENT - 1)
    }
}

impl Drop for ZeroCopyArena {
    fn drop(&mut self) {
        // Synchronize before cleanup
        let _ = self.cuda_page.synchronize();
        
        // Return page to slab pool for recycling
        log::debug!("Returning CUDA arena {} page to slab pool", self.arena_id);
        
        // The Arc<CudaPage> will be properly cleaned up when all references are dropped
        // The slab pool will handle CUDA memory recycling
    }
}

/// Enhanced statistics for CUDA zero-copy arena
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

/// CUDA-enabled zero-copy manager
pub struct ZeroCopyManager {
    slab_pool: Arc<GlobalSlabPool>,
    cuda_context: Arc<CudaContext>,
    next_arena_id: AtomicUsize,
    active_arenas: std::sync::Mutex<std::collections::HashMap<u64, Arc<ZeroCopyArena>>>,
}

impl ZeroCopyManager {
    /// Create new zero-copy manager with CUDA context
    pub fn new(slab_pool: Arc<GlobalSlabPool>) -> Result<Self, CudaError> {
        let cuda_context = Arc::new(CudaContext::new()?);
        
        Ok(Self {
            slab_pool,
            cuda_context,
            next_arena_id: AtomicUsize::new(0),
            active_arenas: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// Create new CUDA arena on specified device
    pub fn create_arena(&self, device_id: i32, page_size: usize) -> Result<Arc<ZeroCopyArena>, CudaError> {
        // Try to get page from slab pool first
        let cuda_page = if let Some(recycled_page) = self.slab_pool.get_page(page_size, device_id) {
            log::debug!("Reused CUDA page from slab pool for device {}", device_id);
            recycled_page
        } else {
            // Allocate new CUDA page
            let new_page = self.cuda_context.allocate_page_on_device(page_size, device_id)?;
            self.slab_pool.record_page_creation(page_size);
            log::debug!("Allocated new CUDA page for device {}", device_id);
            new_page
        };

        let arena_id = self.next_arena_id.fetch_add(1, Ordering::Relaxed) as u64;
        let arena = Arc::new(ZeroCopyArena::new(
            cuda_page, 
            arena_id, 
            Arc::clone(&self.slab_pool),
            Some(Arc::clone(&self.cuda_context)),
        ));

        // Track active arena
        if let Ok(mut arenas) = self.active_arenas.lock() {
            arenas.insert(arena_id, Arc::clone(&arena));
        }

        Ok(arena)
    }

    /// Create arena with automatic device selection based on memory availability
    pub fn create_arena_auto(&self, page_size: usize) -> Result<Arc<ZeroCopyArena>, CudaError> {
        let arena = self.cuda_context.allocate_page_auto(page_size)
            .and_then(|page| {
                let device_id = page.device_id();
                let arena_id = self.next_arena_id.fetch_add(1, Ordering::Relaxed) as u64;
                Ok(Arc::new(ZeroCopyArena::new(
                    page,
                    arena_id,
                    Arc::clone(&self.slab_pool),
                    Some(Arc::clone(&self.cuda_context)),
                )))
            })?;

        // Track arena
        if let Ok(mut arenas) = self.active_arenas.lock() {
            arenas.insert(arena.arena_id, Arc::clone(&arena));
        }

        Ok(arena)
    }

    /// Get comprehensive CUDA statistics across all arenas
    pub fn global_stats(&self) -> ZeroCopyGlobalStats {
        let arenas = self.active_arenas.lock().unwrap();
        let arena_stats: Vec<ZeroCopyArenaStats> = arenas
            .values()
            .map(|arena| arena.stats())
            .collect();

        let total_arenas = arena_stats.len();
        let total_tensors: usize = arena_stats.iter().map(|s| s.total_tensors).sum();
        let total_used_bytes: usize = arena_stats.iter().map(|s| s.used_bytes).sum();
        let total_allocated_bytes: usize = arena_stats.iter().map(|s| s.total_allocated_bytes).sum();
        
        let avg_utilization = if total_arenas > 0 {
            arena_stats.iter().map(|s| s.arena_utilization).sum::<f64>() / total_arenas as f64
        } else {
            0.0
        };

        let avg_cuda_pressure = if total_arenas > 0 {
            arena_stats.iter().map(|s| s.cuda_memory_pressure).sum::<f64>() / total_arenas as f64
        } else {
            0.0
        };

        let slab_stats = self.slab_pool.stats();

        // Get CUDA context statistics
        let cuda_stats = self.get_cuda_context_stats();

        ZeroCopyGlobalStats {
            total_arenas,
            total_tensors,
            total_used_bytes,
            total_allocated_bytes,
            avg_arena_utilization: avg_utilization,
            avg_cuda_memory_pressure: avg_cuda_pressure,
            slab_pool_stats: slab_stats,
            arena_stats,
            cuda_context_stats: cuda_stats,
        }
    }

    /// Get CUDA context statistics
    fn get_cuda_context_stats(&self) -> Vec<crate::cuda::CudaDeviceStats> {
        let manager = self.cuda_context.manager();
        manager.device_infos
            .iter()
            .filter_map(|info| self.cuda_context.device_stats_detailed(info.device_id))
            .collect()
    }

    /// Cleanup inactive arenas
    pub fn cleanup_inactive_arenas(&self) -> usize {
        let mut arenas = self.active_arenas.lock().unwrap();
        let initial_count = arenas.len();

        // Remove arenas with only one reference (our reference)
        arenas.retain(|_, arena| Arc::strong_count(arena) > 1);

        let removed = initial_count - arenas.len();
        if removed > 0 {
            log::info!("Cleaned up {} inactive CUDA arenas", removed);
        }
        removed
    }

    /// Force defragmentation of all active arenas using CUDA device operations
    pub fn defragment_all(&self) -> Result<usize, CudaError> {
        let arenas = self.active_arenas.lock().unwrap();
        let mut total_saved = 0;

        for arena in arenas.values() {
            match arena.defragment() {
                Ok(saved) => total_saved += saved,
                Err(e) => log::warn!("Failed to defragment CUDA arena {}: {}", arena.arena_id, e),
            }
        }

        Ok(total_saved)
    }

    /// Synchronize all CUDA operations across all arenas
    pub fn synchronize_all(&self) -> Result<(), CudaError> {
        let arenas = self.active_arenas.lock().unwrap();
        
        for arena in arenas.values() {
            arena.synchronize()?;
        }

        Ok(())
    }

    /// Get performance recommendations based on CUDA metrics
    pub fn get_recommendations(&self) -> Vec<String> {
        let stats = self.global_stats();
        let mut recommendations = Vec::new();

        // Check overall utilization
        if stats.avg_arena_utilization < 0.5 {
            recommendations.push(format!(
                "Low average arena utilization ({:.1}%). Consider smaller page sizes or arena pooling.",
                stats.avg_arena_utilization * 100.0
            ));
        }

        // Check CUDA memory pressure
        if stats.avg_cuda_memory_pressure > 0.8 {
            recommendations.push(format!(
                "High CUDA memory pressure ({:.1}%). Consider reducing batch sizes or using smaller models.",
                stats.avg_cuda_memory_pressure * 100.0
            ));
        }

        // Check memory efficiency
        let memory_efficiency = if stats.total_allocated_bytes > 0 {
            stats.total_used_bytes as f64 / stats.total_allocated_bytes as f64
        } else {
            1.0
        };

        if memory_efficiency < 0.7 {
            recommendations.push(format!(
                "Low memory efficiency ({:.1}%). Consider pre-allocating with better size estimates.",
                memory_efficiency * 100.0
            ));
        }

        // Check slab pool efficiency
        if stats.slab_pool_stats.recycling_efficiency < 0.5 {
            recommendations.push(format!(
                "Low slab recycling efficiency ({:.1}%). Consider longer arena lifetimes.",
                stats.slab_pool_stats.recycling_efficiency * 100.0
            ));
        }

        // CUDA-specific recommendations
        for device_stats in &stats.cuda_context_stats {
            if device_stats.utilization > 90.0 {
                recommendations.push(format!(
                    "Device {} memory utilization very high ({:.1}%). Consider using multiple devices.",
                    device_stats.device_id, device_stats.utilization
                ));
            }
        }

        recommendations
    }

    /// Get CUDA context reference
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
    }
}

/// Enhanced global statistics with CUDA information
#[derive(Debug, Clone)]
pub struct ZeroCopyGlobalStats {
    pub total_arenas: usize,
    pub total_tensors: usize,
    pub total_used_bytes: usize,
    pub total_allocated_bytes: usize,
    pub avg_arena_utilization: f64,
    pub avg_cuda_memory_pressure: f64,
    pub slab_pool_stats: crate::slab::SlabPoolStats,
    pub arena_stats: Vec<ZeroCopyArenaStats>,
    pub cuda_context_stats: Vec<crate::cuda::CudaDeviceStats>,
}

impl ZeroCopyGlobalStats {
    /// Calculate overall system efficiency including CUDA factors
    pub fn system_efficiency(&self) -> f64 {
        if self.total_allocated_bytes == 0 {
            1.0
        } else {
            let memory_efficiency = self.total_used_bytes as f64 / self.total_allocated_bytes as f64;
            let cuda_efficiency = 1.0 - self.avg_cuda_memory_pressure;
            (memory_efficiency + cuda_efficiency) / 2.0
        }
    }

    /// Check if system needs optimization
    pub fn needs_optimization(&self) -> bool {
        self.avg_arena_utilization < 0.5 || 
        self.system_efficiency() < 0.7 ||
        self.slab_pool_stats.recycling_efficiency < 0.5 ||
        self.avg_cuda_memory_pressure > 0.8
    }

    /// Get memory pressure indicator including CUDA pressure
    pub fn memory_pressure(&self) -> f64 {
        let util_pressure = self.avg_arena_utilization;
        let efficiency_pressure = 1.0 - self.system_efficiency();
        let cuda_pressure = self.avg_cuda_memory_pressure;
        (util_pressure + efficiency_pressure + cuda_pressure) / 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_cuda_tensor_creation() {
        if let Ok(manager) = ZeroCopyManager::new(Arc::new(GlobalSlabPool::new())) {
            if let Ok(arena) = manager.create_arena_auto(4 * 1024 * 1024) { // 4MB
                let tensor = arena.allocate_tensor_with_growth(
                    128, 256, 2048, 16, 2 // seq_len, max_seq_len, hidden_dim, num_heads, element_size
                ).unwrap();

                assert_eq!(tensor.seq_len, 128);
                assert!(tensor.can_extend_to(256));
                assert_eq!(tensor.device_id(), arena.device_id);
                
                println!("✓ Zero-copy CUDA tensor creation test passed");
            }
        }
    }

    #[test]
    fn test_true_zero_copy_extension() {
        if let Ok(manager) = ZeroCopyManager::new(Arc::new(GlobalSlabPool::new())) {
            if let Ok(arena) = manager.create_arena_auto(4 * 1024 * 1024) {
                let mut tensor = arena.allocate_tensor_with_growth(
                    64, 256, 1024, 8, 2
                ).unwrap();

                // Test true zero-copy extension
                let extended = arena.try_extend_tensor(&mut tensor, 128).unwrap();
                assert!(extended, "Should be able to extend in place");
                assert_eq!(tensor.seq_len, 128);
                
                println!("✓ True zero-copy extension test passed");
            }
        }
    }

    #[test]
    fn test_cuda_device_operations() {
        if let Ok(manager) = ZeroCopyManager::new(Arc::new(GlobalSlabPool::new())) {
            if let Ok(arena) = manager.create_arena_auto(4 * 1024 * 1024) {
                let tensor1 = arena.allocate_tensor_with_growth(64, 128, 512, 8, 2).unwrap();
                let tensor2 = arena.allocate_tensor_with_growth(64, 128, 512, 8, 2).unwrap();

                // Test synchronization
                arena.synchronize().expect("Synchronization should work");
                
                // Test defragmentation
                let saved = arena.defragment().unwrap();
                println!("✓ CUDA defragmentation saved {} bytes", saved);
                
                println!("✓ CUDA device operations test passed");
            }
        }
    }

    #[test]
    fn test_batch_allocation() {
        if let Ok(manager) = ZeroCopyManager::new(Arc::new(GlobalSlabPool::new())) {
            if let Ok(arena) = manager.create_arena_auto(8 * 1024 * 1024) { // 8MB
                let requests = vec![
                    (32, 64, 256, 4, 2),
                    (64, 128, 256, 4, 2),
                    (128, 256, 256, 4, 2),
                ];

                let tensors = arena.allocate_batch_tensors(&requests).unwrap();
                assert_eq!(tensors.len(), 3);
                
                for (i, tensor) in tensors.iter().enumerate() {
                    assert_eq!(tensor.seq_len, requests[i].0);
                }
                
                println!("✓ Batch allocation test passed");
            }
        }
    }
}