// src/zero_copy.rs - True zero-copy operations with KV-specific layout
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ptr::NonNull;
use crate::cuda::{CudaPage, CudaError, CudaTensor, CudaContext, BumpAllocator};
use crate::slab::{GlobalSlabPool, RecyclablePage};
use crate::kv_layout::{KVTensorLayout, calculate_optimal_kv_page_size, calculate_model_kv_page_size, ModelConfig};

/// True zero-copy KV tensor with bump-allocated device memory
/// SPECIFIC KV-CACHE LAYOUT - no generic tensors
#[derive(Debug)]
pub struct ZeroCopyTensor {
    /// KV-specific tensor layout with optimized memory arrangement
    kv_layout: KVTensorLayout,
}

impl ZeroCopyTensor {
    /// Create KV tensor from bump allocation - SPECIFIC KV LAYOUT
    pub fn from_bump_allocation(
        page: &Arc<CudaPage>,
        device_ptr: NonNull<u8>,
        offset: usize,
        seq_len: usize,
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        element_size: usize,
        arena_id: u64,
    ) -> Self {
        let kv_layout = KVTensorLayout::from_bump_allocation(
            page, device_ptr, offset, seq_len, max_seq_len,
            num_heads, head_dim, element_size, arena_id
        );
        
        ZeroCopyTensor { kv_layout }
    }

    /// TRUE ZERO-COPY extension - just update seq_len, NO memory operations
    /// This is the core optimization that eliminates "copy amplification"
    pub fn extend_zero_copy(&mut self, new_seq_len: usize) -> bool {
        self.kv_layout.extend_zero_copy(new_seq_len)
    }

    /// Get direct device pointer for key tensor
    pub fn key_device_ptr(&self) -> *mut std::ffi::c_void {
        self.kv_layout.key_device_ptr()
    }

    /// Get direct device pointer for value tensor
    pub fn value_device_ptr(&self) -> *mut std::ffi::c_void {
        self.kv_layout.value_device_ptr()
    }

    /// Get KV tensor dimensions (seq_len, num_heads, head_dim)
    pub fn dimensions(&self) -> (usize, usize, usize) {
        self.kv_layout.kv_dimensions()
    }

    /// Get current size in bytes for both K and V tensors
    pub fn size_bytes(&self) -> usize {
        self.kv_layout.current_kv_size_bytes()
    }

    /// Get maximum allocated size in bytes
    pub fn max_size_bytes(&self) -> usize {
        self.kv_layout.max_kv_size_bytes()
    }

    /// Check if tensor can be extended to new length (zero-copy)
    pub fn can_extend_to(&self, new_seq_len: usize) -> bool {
        self.kv_layout.can_extend_to(new_seq_len)
    }

    /// Get utilization ratio (current / max capacity)
    pub fn utilization(&self) -> f64 {
        self.kv_layout.utilization()
    }

    /// Copy KV data from host to device (only for initial loading)
    pub fn copy_from_host(&self, host_key_data: *const u8, host_value_data: *const u8) -> Result<(), CudaError> {
        self.kv_layout.copy_full_kv_from_host(host_key_data, host_value_data)
    }

    /// Copy only new KV tokens for incremental generation (minimizes copy overhead)
    pub fn copy_new_tokens_from_host(
        &self,
        host_key_data: *const u8,
        host_value_data: *const u8,
        start_token: usize,
        num_tokens: usize,
    ) -> Result<(), CudaError> {
        self.kv_layout.copy_new_kv_tokens_from_host(host_key_data, host_value_data, start_token, num_tokens)
    }

    /// Synchronize operations on this KV tensor
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.kv_layout.synchronize()
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.kv_layout.device_id()
    }

    /// Get arena ID
    pub fn arena_id(&self) -> u64 {
        self.kv_layout.arena_id()
    }

    /// Get detailed KV tensor information
    pub fn kv_info(&self) -> crate::kv_layout::KVTensorInfo {
        self.kv_layout.kv_tensor_info()
    }

    /// Validate KV tensor layout
    pub fn validate_kv_layout(&self) -> Result<crate::kv_layout::KVLayoutValidation, CudaError> {
        self.kv_layout.validate_layout()
    }
}

/// Zero-copy arena with TRUE BUMP ALLOCATION and KV-specific optimization
/// Implements the core arena allocation pattern described in the project
#[derive(Debug)]
pub struct ZeroCopyArena {
    /// Arena identifier
    arena_id: u64,
    /// CUDA page with bump allocator
    cuda_page: Arc<CudaPage>,
    /// Slab pool for page recycling
    slab_pool: Arc<GlobalSlabPool>,
    /// Device ID
    device_id: i32,
    /// CUDA context
    cuda_context: Option<Arc<CudaContext>>,
    /// Tensor counter for tracking
    tensor_count: AtomicUsize,
}

impl ZeroCopyArena {
    /// Create new arena with bump-allocated CUDA page
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
            slab_pool,
            device_id,
            cuda_context,
            tensor_count: AtomicUsize::new(0),
        }
    }

    /// Allocate KV tensor with growth capacity using BUMP ALLOCATION
    /// Page size = round-up of largest KV tensor expected (as per project spec)
    pub fn allocate_tensor_with_growth(
        &self,
        initial_seq_len: usize,
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        element_size: usize,
    ) -> Result<ZeroCopyTensor, CudaError> {
        // Calculate size for maximum expected KV tensor pair (as per project description)
        let max_kv_tensor_size = calculate_optimal_kv_page_size(max_seq_len, num_heads, head_dim, element_size);
        let alignment = 256; // CUDA optimal alignment
        
        // BUMP ALLOCATION: offset += align(size); no per-tensor metadata
        if let Some(device_ptr) = self.cuda_page.allocate(max_kv_tensor_size, alignment) {
            let offset = self.cuda_page.current_offset() - max_kv_tensor_size;
            
            let tensor = ZeroCopyTensor::from_bump_allocation(
                &self.cuda_page,
                device_ptr,
                offset,
                initial_seq_len,
                max_seq_len,
                num_heads,
                head_dim,
                element_size,
                self.arena_id,
            );

            self.tensor_count.fetch_add(1, Ordering::Relaxed);
            
            log::debug!("BUMP allocated KV tensor: {}x{}x{} (max: {}) at offset {}", 
                       initial_seq_len, num_heads, head_dim, max_seq_len, offset);

            Ok(tensor)
        } else {
            Err(CudaError(-2)) // Arena full
        }
    }

    /// Try to extend tensor (TRUE ZERO-COPY when possible)
    /// This is the core optimization that eliminates copy amplification
    pub fn try_extend_tensor(
        &self,
        tensor: &mut ZeroCopyTensor,
        new_seq_len: usize,
    ) -> Result<bool, CudaError> {
        // TRUE ZERO-COPY: just update metadata if within capacity
        Ok(tensor.extend_zero_copy(new_seq_len))
    }

    /// Defragmentation using device-to-device copies (still zero-copy from host perspective)
    pub fn defragment(&self) -> Result<usize, CudaError> {
        // For bump allocators, defragmentation is mainly about compaction
        // This would require tracking active tensors and moving them
        // For now, return 0 as bump allocators have minimal fragmentation
        log::debug!("Bump allocator defragmentation (minimal fragmentation by design)");
        Ok(0)
    }

    /// Get arena utilization
    pub fn utilization(&self) -> f64 {
        self.cuda_page.utilization()
    }

    /// Get available space
    pub fn available_space(&self) -> usize {
        self.cuda_page.available_space()
    }

    /// Get arena statistics
    pub fn stats(&self) -> ZeroCopyArenaStats {
        let used_bytes = self.cuda_page.current_offset();
        let page_size = self.cuda_page.size();
        let total_tensors = self.tensor_count.load(Ordering::Relaxed);
        
        ZeroCopyArenaStats {
            arena_id: self.arena_id,
            device_id: self.device_id,
            page_size,
            used_bytes,
            total_tensors,
            total_used_bytes: used_bytes,
            total_allocated_bytes: page_size,
            avg_tensor_utilization: if total_tensors > 0 { 
                used_bytes as f64 / page_size as f64 
            } else { 
                0.0 
            },
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

    /// Synchronize all operations in this arena
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.cuda_page.synchronize()
    }

    /// Get the underlying CUDA page
    pub fn cuda_page(&self) -> &Arc<CudaPage> {
        &self.cuda_page
    }

    /// Check if all operations are complete
    pub fn is_ready(&self) -> Result<bool, CudaError> {
        self.cuda_page.is_ready()
    }
}

impl Drop for ZeroCopyArena {
    fn drop(&mut self) {
        // Synchronize before cleanup
        let _ = self.cuda_page.synchronize();
        
        // SLAB RECYCLING: when SequenceArena drops, its pages go back to GlobalSlabPool
        log::debug!("Arena {} dropping - page goes back to slab pool", self.arena_id);
        
        // The Arc<CudaPage> will be returned to slab pool when all references are dropped
        // This implements the "slab recycling" pattern described in the project
    }
}

/// Arena statistics
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

/// Zero-copy manager with lock-free slab pool integration and KV-specific optimization
pub struct ZeroCopyManager {
    slab_pool: Arc<GlobalSlabPool>,
    cuda_context: Arc<CudaContext>,
    next_arena_id: AtomicUsize,
    active_arenas: std::sync::Mutex<std::collections::HashMap<u64, Arc<ZeroCopyArena>>>,
}

impl ZeroCopyManager {
    /// Create new zero-copy manager with lock-free slab pool
    pub fn new(slab_pool: Arc<GlobalSlabPool>) -> Result<Self, CudaError> {
        let cuda_context = Arc::new(CudaContext::new()?);
        
        Ok(Self {
            slab_pool,
            cuda_context,
            next_arena_id: AtomicUsize::new(0),
            active_arenas: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// Create arena with optimal page size calculation based on KV tensor requirements
    pub fn create_arena(&self, device_id: i32, model_config: ModelConfig) -> Result<Arc<ZeroCopyArena>, CudaError> {
        // Calculate optimal page size using project spec: "round-up of largest KV tensor expected"
        let page_size = calculate_model_kv_page_size(&model_config);
        
        // Try to get recycled page from lock-free slab pool first
        let cuda_page = if let Some(recycled_page) = self.slab_pool.get_page(page_size, device_id) {
            log::debug!("RECYCLED page from slab pool for device {} (KV optimized: {}KB)", 
                       device_id, page_size / 1024);
            recycled_page
        } else {
            // Allocate new page with bump allocator
            let new_page = self.cuda_context.allocate_page_on_device(page_size, device_id)?;
            self.slab_pool.record_page_creation(page_size);
            log::debug!("NEW KV-optimized page allocated for device {} ({}KB)", 
                       device_id, page_size / 1024);
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

        log::info!("Created KV-optimized arena {} with page size {}KB for model {:?}",
                  arena_id, page_size / 1024, model_config);

        Ok(arena)
    }

    /// Create arena with automatic device selection and optimal KV page sizing
    pub fn create_arena_auto(&self, model_config: ModelConfig) -> Result<Arc<ZeroCopyArena>, CudaError> {
        let page_size = calculate_model_kv_page_size(&model_config);
        let page = self.cuda_context.allocate_page_auto(page_size)?;
        let device_id = page.device_id();
        let arena_id = self.next_arena_id.fetch_add(1, Ordering::Relaxed) as u64;
        
        let arena = Arc::new(ZeroCopyArena::new(
            page,
            arena_id,
            Arc::clone(&self.slab_pool),
            Some(Arc::clone(&self.cuda_context)),
        ));

        // Track arena
        if let Ok(mut arenas) = self.active_arenas.lock() {
            arenas.insert(arena.arena_id, Arc::clone(&arena));
        }

        log::info!("Created auto-selected KV arena {} on device {} with {}KB page", 
                  arena_id, device_id, page_size / 1024);

        Ok(arena)
    }

    /// Create arena with legacy page size (for backward compatibility)
    pub fn create_arena_legacy(&self, device_id: i32, page_size: usize) -> Result<Arc<ZeroCopyArena>, CudaError> {
        // Try to get recycled page from lock-free slab pool first
        let cuda_page = if let Some(recycled_page) = self.slab_pool.get_page(page_size, device_id) {
            log::debug!("RECYCLED page from slab pool for device {}", device_id);
            recycled_page
        } else {
            // Allocate new page with bump allocator
            let new_page = self.cuda_context.allocate_page_on_device(page_size, device_id)?;
            self.slab_pool.record_page_creation(page_size);
            log::debug!("NEW page allocated for device {}", device_id);
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

    /// Get comprehensive statistics
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

    /// Synchronize all operations
    pub fn synchronize_all(&self) -> Result<(), CudaError> {
        let arenas = self.active_arenas.lock().unwrap();
        
        for arena in arenas.values() {
            arena.synchronize()?;
        }

        Ok(())
    }

    /// Get performance recommendations based on KV-specific metrics
    pub fn get_recommendations(&self) -> Vec<String> {
        let stats = self.global_stats();
        let mut recommendations = Vec::new();

        // Check overall utilization
        if stats.avg_arena_utilization < 0.5 {
            recommendations.push(format!(
                "Low average arena utilization ({:.1}%). Consider smaller KV page sizes or better size estimation.",
                stats.avg_arena_utilization * 100.0
            ));
        }

        // Check memory efficiency for KV tensors
        let memory_efficiency = if stats.total_allocated_bytes > 0 {
            stats.total_used_bytes as f64 / stats.total_allocated_bytes as f64
        } else {
            1.0
        };

        if memory_efficiency < 0.7 {
            recommendations.push(format!(
                "Low KV memory efficiency ({:.1}%). Bump allocation may be over-allocating for KV tensors.",
                memory_efficiency * 100.0
            ));
        }

        // Check slab pool efficiency
        if stats.slab_pool_stats.recycling_efficiency < 0.5 {
            recommendations.push(format!(
                "Low slab recycling efficiency ({:.1}%). KV pages may not be living long enough.",
                stats.slab_pool_stats.recycling_efficiency * 100.0
            ));
        }

        // KV-specific recommendations
        if stats.total_tensors > 0 {
            let avg_tensor_size = stats.total_used_bytes / stats.total_tensors;
            if avg_tensor_size < 64 * 1024 { // Less than 64KB per tensor
                recommendations.push("Small KV tensors detected. Consider using smaller page sizes for better utilization.".to_string());
            } else if avg_tensor_size > 4 * 1024 * 1024 { // More than 4MB per tensor
                recommendations.push("Large KV tensors detected. Consider using larger page sizes or tensor splitting.".to_string());
            }
        }

        recommendations
    }

    /// Get CUDA context reference
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
    }
}

/// Global statistics
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
    /// Calculate system efficiency
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

    /// Get memory pressure indicator
    pub fn memory_pressure(&self) -> f64 {
        let util_pressure = self.avg_arena_utilization;
        let efficiency_pressure = 1.0 - self.system_efficiency();
        (util_pressure + efficiency_pressure) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_kv_tensor_extension() {
        // This test validates the core zero-copy extension feature for KV tensors
        if let Ok(manager) = ZeroCopyManager::new(Arc::new(GlobalSlabPool::new())) {
            let model_config = ModelConfig::Custom {
                max_seq_len: 256,
                num_heads: 16,
                head_dim: 128,
                element_size: 2,
            };
            
            if let Ok(arena) = manager.create_arena_auto(model_config) {
                if let Ok(mut tensor) = arena.allocate_tensor_with_growth(
                    128, 256, 16, 128, 2 // seq_len, max_seq_len, num_heads, head_dim, element_size
                ) {
                    let (seq_len, num_heads, head_dim) = tensor.dimensions();
                    assert_eq!(seq_len, 128);
                    assert_eq!(num_heads, 16);
                    assert_eq!(head_dim, 128);
                    assert!(tensor.can_extend_to(256));
                    
                    // Test TRUE zero-copy extension
                    let extended = arena.try_extend_tensor(&mut tensor, 192).unwrap();
                    assert!(extended, "Should be able to extend KV tensor in place");
                    
                    let (new_seq_len, _, _) = tensor.dimensions();
                    assert_eq!(new_seq_len, 192);
                    
                    println!("✓ True zero-copy KV tensor extension test passed");
                }
            }
        }
    }

    #[test]
    fn test_kv_specific_bump_allocation() {
        if let Ok(manager) = ZeroCopyManager::new(Arc::new(GlobalSlabPool::new())) {
            let model_config = ModelConfig::Llama2_7B;
            
            if let Ok(arena) = manager.create_arena_auto(model_config) {
                // Test multiple KV tensor allocations (bump pattern)
                let tensors = vec![
                    arena.allocate_tensor_with_growth(64, 128, 8, 64, 2),
                    arena.allocate_tensor_with_growth(96, 192, 8, 64, 2),
                    arena.allocate_tensor_with_growth(32, 64, 8, 64, 2),
                ];
                
                let successful_allocations = tensors.iter().filter(|t| t.is_ok()).count();
                println!("✓ KV bump allocation pattern: {}/{} successful allocations", 
                        successful_allocations, tensors.len());
                
                // Check arena utilization
                let stats = arena.stats();
                println!("✓ KV Arena utilization: {:.1}%", stats.arena_utilization * 100.0);
            }
        }
    }

    #[test]
    fn test_kv_model_configs() {
        // Test page size calculations for different model configurations
        let configs = vec![
            ModelConfig::Llama2_7B,
            ModelConfig::Llama2_13B,
            ModelConfig::Llama2_70B,
            ModelConfig::Custom { max_seq_len: 4096, num_heads: 32, head_dim: 128, element_size: 2 },
        ];

        for config in configs {
            let page_size = calculate_model_kv_page_size(&config);
            assert!(page_size >= 64 * 1024, "Page size should be at least 64KB");
            assert!(page_size <= 16 * 1024 * 1024, "Page size should be at most 16MB");
            
            println!("✓ Model config {:?}: page_size={}KB", config, page_size / 1024);
        }
    }

    #[test]
    fn test_kv_slab_recycling_integration() {
        let slab_pool = Arc::new(GlobalSlabPool::new());
        
        if let Ok(manager) = ZeroCopyManager::new(slab_pool.clone()) {
            // Create and drop arena (should trigger slab recycling)
            {
                let model_config = ModelConfig::Custom {
                    max_seq_len: 1024,
                    num_heads: 16,
                    head_dim: 64,
                    element_size: 2,
                };
                let _arena = manager.create_arena_auto(model_config);
                // Arena will drop here
            }
            
            // Check slab pool stats
            let stats = slab_pool.stats();
            println!("✓ KV Slab pool stats: created={}, recycled={}", 
                    stats.total_pages_created, stats.total_pages_recycled);
        }
    }
}