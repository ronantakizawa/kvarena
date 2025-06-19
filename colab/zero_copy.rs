// src/zero_copy.rs - Eliminate copy amplification with true zero-copy extensions
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ptr::NonNull;
use crate::cuda::{CudaPage, CudaError};
use crate::slab::{GlobalSlabPool, RecyclablePage};

/// Zero-copy tensor descriptor that tracks memory layout without copying data
#[derive(Debug, Clone)]
pub struct ZeroCopyTensor {
    /// Device pointer to the start of tensor data
    device_ptr: NonNull<u8>,
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
    page_ref: Arc<PageReference>,
}

/// Reference to underlying page to prevent premature deallocation
#[derive(Debug)]
struct PageReference {
    cuda_page: CudaPage,
    arena_id: u64,
    ref_count: AtomicUsize,
}

impl PageReference {
    fn new(cuda_page: CudaPage, arena_id: u64) -> Self {
        Self {
            cuda_page,
            arena_id,
            ref_count: AtomicUsize::new(1),
        }
    }

    fn clone_ref(&self) -> Arc<PageReference> {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
        // This is safe because we're creating a new Arc to the same data
        unsafe {
            Arc::from_raw(self as *const PageReference)
        }
    }
}

impl Drop for PageReference {
    fn drop(&mut self) {
        let prev_count = self.ref_count.fetch_sub(1, Ordering::Relaxed);
        if prev_count == 1 {
            // Last reference, page can be recycled
            log::debug!("Releasing page for arena {} to slab pool", self.arena_id);
        }
    }
}

impl ZeroCopyTensor {
    /// Create new tensor descriptor from arena allocation
    pub fn new(
        cuda_page: &CudaPage,
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
        
        if offset + used_size > cuda_page.size() {
            return Err(CudaError(-1)); // Out of bounds
        }

        // Get device pointer at offset
        let device_ptr = unsafe {
            NonNull::new_unchecked(cuda_page.device_ptr_at_offset(offset) as *mut u8)
        };

        // Create page reference (this clones the CudaPage, but the underlying device memory is shared)
        let page_ref = Arc::new(PageReference::new(
            // We need to clone the CudaPage here, but this is just metadata
            // The actual device memory is not copied
            unsafe { std::ptr::read(cuda_page) }, // This is a hack - in real impl, we'd have a proper clone
            arena_id,
        ));

        Ok(ZeroCopyTensor {
            device_ptr,
            offset,
            allocated_size,
            used_size,
            seq_len,
            hidden_dim,
            num_heads,
            element_size,
            page_ref,
        })
    }

    /// Extend tensor to new sequence length WITHOUT copying data
    pub fn extend_zero_copy(&mut self, new_seq_len: usize) -> Result<bool, CudaError> {
        let new_used_size = Self::calculate_tensor_size(new_seq_len, self.hidden_dim, self.num_heads, self.element_size);
        
        // Check if we can extend in place
        if self.offset + new_used_size <= self.page_ref.cuda_page.size() && 
           new_used_size <= self.allocated_size {
            // Zero-copy extension: just update metadata
            self.seq_len = new_seq_len;
            self.used_size = new_used_size;
            
            log::debug!("Zero-copy extension: {} -> {} tokens", 
                       self.seq_len, new_seq_len);
            return Ok(true);
        }

        // Cannot extend in place
        Ok(false)
    }

    /// Create a view of this tensor for a specific sequence range
    pub fn slice_view(&self, start_seq: usize, end_seq: usize) -> Result<ZeroCopyTensor, CudaError> {
        if start_seq >= self.seq_len || end_seq > self.seq_len || start_seq >= end_seq {
            return Err(CudaError(-1)); // Invalid slice
        }

        let slice_seq_len = end_seq - start_seq;
        let head_dim = self.hidden_dim / self.num_heads;
        
        // Calculate offset for the slice (both key and value)
        let tokens_per_tensor = self.seq_len * self.num_heads * head_dim;
        let slice_offset_tokens = start_seq * self.num_heads * head_dim;
        let slice_byte_offset = slice_offset_tokens * self.element_size;
        
        // Create new tensor pointing to the same memory but different offset
        let slice_device_ptr = unsafe {
            NonNull::new_unchecked(
                (self.device_ptr.as_ptr() as *mut u8).add(slice_byte_offset)
            )
        };

        let slice_used_size = Self::calculate_tensor_size(slice_seq_len, self.hidden_dim, self.num_heads, self.element_size);

        Ok(ZeroCopyTensor {
            device_ptr: slice_device_ptr,
            offset: self.offset + slice_byte_offset,
            allocated_size: self.allocated_size - slice_byte_offset,
            used_size: slice_used_size,
            seq_len: slice_seq_len,
            hidden_dim: self.hidden_dim,
            num_heads: self.num_heads,
            element_size: self.element_size,
            page_ref: Arc::clone(&self.page_ref),
        })
    }

    /// Get device pointer for key tensor
    pub fn key_ptr(&self) -> *mut u8 {
        self.device_ptr.as_ptr()
    }

    /// Get device pointer for value tensor  
    pub fn value_ptr(&self) -> *mut u8 {
        let head_dim = self.hidden_dim / self.num_heads;
        let key_size = self.seq_len * self.num_heads * head_dim * self.element_size;
        unsafe {
            self.device_ptr.as_ptr().add(key_size)
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
        new_size <= self.allocated_size && self.offset + new_size <= self.page_ref.cuda_page.size()
    }

    /// Get utilization ratio (used / allocated)
    pub fn utilization(&self) -> f64 {
        self.used_size as f64 / self.allocated_size as f64
    }

    /// Copy data from host to this tensor
    pub fn copy_from_host(&self, host_key_data: *const u8, host_value_data: *const u8) -> Result<(), CudaError> {
        let head_dim = self.hidden_dim / self.num_heads;
        let single_tensor_size = self.seq_len * self.num_heads * head_dim * self.element_size;

        // Copy key data
        self.page_ref.cuda_page.copy_from_host(
            host_key_data as *const std::ffi::c_void,
            single_tensor_size,
            self.offset,
        )?;

        // Copy value data
        self.page_ref.cuda_page.copy_from_host(
            host_value_data as *const std::ffi::c_void,
            single_tensor_size,
            self.offset + single_tensor_size,
        )?;

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

        // Copy new key tokens
        self.page_ref.cuda_page.copy_from_host(
            unsafe { host_key_data.add(token_offset) } as *const std::ffi::c_void,
            copy_size,
            self.offset + token_offset,
        )?;

        // Copy new value tokens
        let single_tensor_size = self.seq_len * token_size / num_tokens; // Recalculate for safety
        self.page_ref.cuda_page.copy_from_host(
            unsafe { host_value_data.add(token_offset) } as *const std::ffi::c_void,
            copy_size,
            self.offset + single_tensor_size + token_offset,
        )?;

        Ok(())
    }

    fn calculate_tensor_size(seq_len: usize, hidden_dim: usize, num_heads: usize, element_size: usize) -> usize {
        // Size for both key and value tensors
        2 * seq_len * hidden_dim * element_size
    }
}

/// Zero-copy arena that manages tensor allocations without data copying
#[derive(Debug)]
pub struct ZeroCopyArena {
    arena_id: u64,
    cuda_page: CudaPage,
    current_offset: AtomicUsize,
    allocated_tensors: std::sync::Mutex<Vec<ZeroCopyTensor>>,
    slab_pool: Arc<GlobalSlabPool>,
    device_id: i32,
}

impl ZeroCopyArena {
    /// Create new zero-copy arena with CUDA page
    pub fn new(cuda_page: CudaPage, arena_id: u64, slab_pool: Arc<GlobalSlabPool>) -> Self {
        let device_id = cuda_page.device_id();
        
        Self {
            arena_id,
            cuda_page,
            current_offset: AtomicUsize::new(0),
            allocated_tensors: std::sync::Mutex::new(Vec::new()),
            slab_pool,
            device_id,
        }
    }

    /// Allocate tensor with extra space for future growth
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

        // Bump allocate within the page
        let offset = self.current_offset.fetch_add(aligned_size, Ordering::Relaxed);
        
        if offset + aligned_size > self.cuda_page.size() {
            return Err(CudaError(-2)); // Out of space
        }

        // Create zero-copy tensor
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

        log::debug!("Allocated zero-copy tensor: {}x{}x{} at offset {}", 
                   initial_seq_len, num_heads, hidden_dim / num_heads, offset);

        Ok(tensor)
    }

    /// Try to extend existing tensor without reallocation
    pub fn try_extend_tensor(
        &self,
        tensor: &mut ZeroCopyTensor,
        new_seq_len: usize,
    ) -> Result<bool, CudaError> {
        // This is the key optimization: no data copying!
        tensor.extend_zero_copy(new_seq_len)
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

    /// Get arena statistics
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
        }
    }

    /// Defragment arena by compacting tensors (advanced operation)
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
                // Move tensor data to compacted position
                let move_size = tensor.used_size;
                self.cuda_page.copy_device_to_device(
                    tensor.offset,
                    compacted_offset,
                    move_size,
                )?;

                bytes_saved += tensor.offset - compacted_offset;
                
                // Update tensor metadata
                tensor.offset = compacted_offset;
                tensor.device_ptr = unsafe {
                    std::ptr::NonNull::new_unchecked(
                        self.cuda_page.device_ptr_at_offset(compacted_offset) as *mut u8
                    )
                };
            }

            compacted_offset += Self::align_size(tensor.used_size);
        }

        // Update current offset
        self.current_offset.store(compacted_offset, Ordering::Relaxed);

        log::info!("Defragmented arena {}: saved {} bytes", self.arena_id, bytes_saved);
        Ok(bytes_saved)
    }

    fn align_size(size: usize) -> usize {
        const ALIGNMENT: usize = 64; // CUDA memory alignment
        (size + ALIGNMENT - 1) & !(ALIGNMENT - 1)
    }
}

impl Drop for ZeroCopyArena {
    fn drop(&mut self) {
        // Return page to slab pool for recycling
        log::debug!("Returning arena {} page to slab pool", self.arena_id);
        
        // Note: We need to take ownership of cuda_page here
        // This is a simplified version - in practice we'd need more careful lifetime management
        let cuda_page = unsafe { std::ptr::read(&self.cuda_page) };
        self.slab_pool.return_page(cuda_page);
    }
}

/// Statistics for zero-copy arena performance monitoring
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
}

/// Zero-copy manager that coordinates multiple arenas
pub struct ZeroCopyManager {
    slab_pool: Arc<GlobalSlabPool>,
    next_arena_id: AtomicUsize,
    active_arenas: std::sync::Mutex<std::collections::HashMap<u64, Arc<ZeroCopyArena>>>,
    device_managers: std::sync::RwLock<std::collections::HashMap<i32, crate::cuda::CudaMemoryManager>>,
}

impl ZeroCopyManager {
    /// Create new zero-copy manager
    pub fn new(slab_pool: Arc<GlobalSlabPool>) -> Self {
        Self {
            slab_pool,
            next_arena_id: AtomicUsize::new(0),
            active_arenas: std::sync::Mutex::new(std::collections::HashMap::new()),
            device_managers: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Create new arena on specified device
    pub fn create_arena(&self, device_id: i32, page_size: usize) -> Result<Arc<ZeroCopyArena>, CudaError> {
        // Try to get page from slab pool first
        let cuda_page = if let Some(recycled_page) = self.slab_pool.get_page(page_size, device_id) {
            log::debug!("Reused page from slab pool for device {}", device_id);
            recycled_page
        } else {
            // Allocate new page
            let manager = self.get_or_create_device_manager(device_id)?;
            let new_page = manager.allocate_page_on_device(page_size, device_id)?;
            self.slab_pool.record_page_creation(page_size);
            log::debug!("Allocated new page for device {}", device_id);
            new_page
        };

        let arena_id = self.next_arena_id.fetch_add(1, Ordering::Relaxed) as u64;
        let arena = Arc::new(ZeroCopyArena::new(cuda_page, arena_id, Arc::clone(&self.slab_pool)));

        // Track active arena
        if let Ok(mut arenas) = self.active_arenas.lock() {
            arenas.insert(arena_id, Arc::clone(&arena));
        }

        Ok(arena)
    }

    /// Create arena with automatic device selection
    pub fn create_arena_auto(&self, page_size: usize) -> Result<Arc<ZeroCopyArena>, CudaError> {
        // Find device with most available memory
        let device_managers = self.device_managers.read().unwrap();
        let best_device = device_managers
            .values()
            .min_by_key(|manager| (manager.memory_pressure() * 1000.0) as u32)
            .map(|manager| manager.current_device_info().device_id)
            .unwrap_or(0);

        drop(device_managers);
        self.create_arena(best_device, page_size)
    }

    /// Get comprehensive statistics across all arenas
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

        let slab_stats = self.slab_pool.stats();

        ZeroCopyGlobalStats {
            total_arenas,
            total_tensors,
            total_used_bytes,
            total_allocated_bytes,
            avg_arena_utilization: avg_utilization,
            slab_pool_stats: slab_stats,
            arena_stats,
        }
    }

    /// Cleanup inactive arenas
    pub fn cleanup_inactive_arenas(&self) -> usize {
        let mut arenas = self.active_arenas.lock().unwrap();
        let initial_count = arenas.len();

        // Remove arenas with only one reference (our reference)
        arenas.retain(|_, arena| Arc::strong_count(arena) > 1);

        let removed = initial_count - arenas.len();
        if removed > 0 {
            log::info!("Cleaned up {} inactive arenas", removed);
        }
        removed
    }

    /// Force defragmentation of all active arenas
    pub fn defragment_all(&self) -> Result<usize, CudaError> {
        let arenas = self.active_arenas.lock().unwrap();
        let mut total_saved = 0;

        for arena in arenas.values() {
            match arena.defragment() {
                Ok(saved) => total_saved += saved,
                Err(e) => log::warn!("Failed to defragment arena {}: {}", arena.arena_id, e),
            }
        }

        Ok(total_saved)
    }

    /// Get performance recommendations
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

        // Check for fragmentation
        let fragmented_arenas = stats.arena_stats.iter()
            .filter(|s| s.arena_utilization > 0.9 && s.avg_tensor_utilization < 0.6)
            .count();

        if fragmented_arenas > 0 {
            recommendations.push(format!(
                "{} arenas show signs of fragmentation. Consider running defragmentation.",
                fragmented_arenas
            ));
        }

        recommendations
    }

    fn get_or_create_device_manager(&self, device_id: i32) -> Result<crate::cuda::CudaMemoryManager, CudaError> {
        // Try to get existing manager
        {
            let managers = self.device_managers.read().unwrap();
            if let Some(manager) = managers.get(&device_id) {
                // We need to clone the manager here, but CudaMemoryManager doesn't implement Clone
                // In a real implementation, we'd store Arc<CudaMemoryManager> or redesign this
                return crate::cuda::CudaMemoryManager::new();
            }
        }

        // Create new manager
        let mut managers = self.device_managers.write().unwrap();
        let manager = crate::cuda::CudaMemoryManager::new()?;
        // managers.insert(device_id, manager.clone()); // Would need Clone implementation
        Ok(manager)
    }
}

/// Global statistics for zero-copy system
#[derive(Debug, Clone)]
pub struct ZeroCopyGlobalStats {
    pub total_arenas: usize,
    pub total_tensors: usize,
    pub total_used_bytes: usize,
    pub total_allocated_bytes: usize,
    pub avg_arena_utilization: f64,
    pub slab_pool_stats: crate::slab::SlabPoolStats,
    pub arena_stats: Vec<ZeroCopyArenaStats>,
}

impl ZeroCopyGlobalStats {
    /// Calculate overall system efficiency
    pub fn system_efficiency(&self) -> f64 {
        if self.total_allocated_bytes == 0 {
            1.0
        } else {
            self.total_used_bytes as f64 / self.total_allocated_bytes as f64
        }
    }

    /// Check if system needs optimization
    pub fn needs_optimization(&self) -> bool {
        self.avg_arena_utilization < 0.5 || 
        self.system_efficiency() < 0.7 ||
        self.slab_pool_stats.recycling_efficiency < 0.5
    }

    /// Get memory pressure indicator (0.0 = low, 1.0 = high)
    pub fn memory_pressure(&self) -> f64 {
        // Simple heuristic based on utilization and efficiency
        let util_pressure = self.avg_arena_utilization;
        let efficiency_pressure = 1.0 - self.system_efficiency();
        (util_pressure + efficiency_pressure) / 2.0
    }
}

/// Builder for zero-copy tensors with automatic optimization
pub struct ZeroCopyTensorBuilder {
    arena: Arc<ZeroCopyArena>,
    seq_len: usize,
    hidden_dim: usize,
    num_heads: usize,
    element_size: usize,
    growth_factor: f64,
    max_seq_len: Option<usize>,
}

impl ZeroCopyTensorBuilder {
    pub fn new(arena: Arc<ZeroCopyArena>) -> Self {
        Self {
            arena,
            seq_len: 512,
            hidden_dim: 4096,
            num_heads: 32,
            element_size: 2,
            growth_factor: 1.5,
            max_seq_len: None,
        }
    }

    pub fn seq_len(mut self, seq_len: usize) -> Self {
        self.seq_len = seq_len;
        self
    }

    pub fn hidden_dim(mut self, hidden_dim: usize) -> Self {
        self.hidden_dim = hidden_dim;
        self
    }

    pub fn num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    pub fn element_size(mut self, element_size: usize) -> Self {
        self.element_size = element_size;
        self
    }

    pub fn growth_factor(mut self, factor: f64) -> Self {
        self.growth_factor = factor;
        self
    }

    pub fn max_seq_len(mut self, max_len: usize) -> Self {
        self.max_seq_len = Some(max_len);
        self
    }

    pub fn build(self) -> Result<ZeroCopyTensor, CudaError> {
        let max_seq_len = self.max_seq_len.unwrap_or_else(|| {
            (self.seq_len as f64 * self.growth_factor) as usize
        });

        self.arena.allocate_tensor_with_growth(
            self.seq_len,
            max_seq_len,
            self.hidden_dim,
            self.num_heads,
            self.element_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaMemoryManager;
    use crate::slab::GlobalSlabPool;

    #[test]
    fn test_zero_copy_tensor_extension() {
        // This test requires CUDA, mock if not available
        if let Ok(manager) = CudaMemoryManager::new() {
            if let Ok(cuda_page) = manager.allocate_page(1024 * 1024) {
                let mut tensor = ZeroCopyTensor::new(
                    &cuda_page,
                    0,
                    1024 * 1024,
                    512,
                    4096,
                    32,
                    2,
                    1,
                ).unwrap();

                // Test zero-copy extension
                let extended = tensor.extend_zero_copy(1024).unwrap();
                assert!(extended, "Should be able to extend in place");
                assert_eq!(tensor.seq_len, 1024);
                
                println!("✓ Zero-copy extension test passed");
            }
        }
    }

    #[test]
    fn test_zero_copy_arena() {
        if let Ok(manager) = CudaMemoryManager::new() {
            if let Ok(cuda_page) = manager.allocate_page(4 * 1024 * 1024) {
                let slab_pool = Arc::new(GlobalSlabPool::new());
                let arena = ZeroCopyArena::new(cuda_page, 1, slab_pool);

                // Test tensor allocation
                let tensor = arena.allocate_tensor_with_growth(
                    256, 512, 2048, 16, 2
                ).unwrap();

                assert_eq!(tensor.seq_len, 256);
                assert!(tensor.can_extend_to(512));
                
                println!("✓ Zero-copy arena test passed");
            }
        }
    }

    #[test]
    fn test_tensor_builder() {
        if let Ok(manager) = CudaMemoryManager::new() {
            if let Ok(cuda_page) = manager.allocate_page(4 * 1024 * 1024) {
                let slab_pool = Arc::new(GlobalSlabPool::new());
                let arena = Arc::new(ZeroCopyArena::new(cuda_page, 1, slab_pool));

                let tensor = ZeroCopyTensorBuilder::new(arena)
                    .seq_len(128)
                    .hidden_dim(2048)
                    .num_heads(16)
                    .growth_factor(2.0)
                    .build()
                    .unwrap();

                assert_eq!(tensor.seq_len, 128);
                assert!(tensor.can_extend_to(256)); // 2x growth factor
                
                println!("✓ Tensor builder test passed");
            }
        }
    }
}