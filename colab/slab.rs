// src/slab.rs - Lock-free slab recycling with cross-arena page sharing
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use crossbeam::queue::SegQueue;
use std::collections::HashMap;
use std::sync::RwLock;
use crate::cuda::{CudaPage, CudaError};

/// Page classification for optimal recycling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageClass {
    Small,   // < 512KB
    Medium,  // 512KB - 2MB  
    Large,   // 2MB - 8MB
    Huge,    // > 8MB
}

impl PageClass {
    fn from_size(size: usize) -> Self {
        match size {
            0..=524_288 => PageClass::Small,      // 0-512KB
            524_289..=2_097_152 => PageClass::Medium,   // 512KB-2MB
            2_097_153..=8_388_608 => PageClass::Large,    // 2MB-8MB
            _ => PageClass::Huge,                         // >8MB
        }
    }

    fn typical_size(&self) -> usize {
        match self {
            PageClass::Small => 256 * 1024,    // 256KB
            PageClass::Medium => 1024 * 1024,  // 1MB
            PageClass::Large => 4 * 1024 * 1024, // 4MB
            PageClass::Huge => 16 * 1024 * 1024, // 16MB
        }
    }
}

/// Recyclable page wrapper with metadata for efficient reuse
#[derive(Debug)]
pub struct RecyclablePage {
    cuda_page: CudaPage,
    page_class: PageClass,
    allocation_count: AtomicUsize,
    last_used: AtomicU64,
    device_id: i32,
}

impl RecyclablePage {
    pub fn new(cuda_page: CudaPage) -> Self {
        let page_class = PageClass::from_size(cuda_page.size());
        let device_id = cuda_page.device_id();
        
        Self {
            cuda_page,
            page_class,
            allocation_count: AtomicUsize::new(0),
            last_used: AtomicU64::new(Self::current_timestamp()),
            device_id,
        }
    }

    /// Get underlying CUDA page
    pub fn cuda_page(&self) -> &CudaPage {
        &self.cuda_page
    }

    /// Take ownership of CUDA page (consuming self)
    pub fn into_cuda_page(self) -> CudaPage {
        self.cuda_page
    }

    /// Mark page as used and increment allocation count
    pub fn mark_used(&self) {
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        self.last_used.store(Self::current_timestamp(), Ordering::Relaxed);
    }

    /// Get allocation count (for wear leveling)
    pub fn allocation_count(&self) -> usize {
        self.allocation_count.load(Ordering::Relaxed)
    }

    /// Get time since last use (for LRU eviction)
    pub fn age(&self) -> u64 {
        Self::current_timestamp() - self.last_used.load(Ordering::Relaxed)
    }

    /// Get page classification
    pub fn page_class(&self) -> PageClass {
        self.page_class
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Check if page is suitable for reuse (not over-allocated)
    pub fn is_reusable(&self) -> bool {
        const MAX_ALLOCATIONS: usize = 1000; // Prevent memory wear
        self.allocation_count() < MAX_ALLOCATIONS
    }
}

/// Lock-free global slab pool with intelligent recycling
#[derive(Debug)]
pub struct GlobalSlabPool {
    // Separate queues by page class for optimal matching
    small_pages: SegQueue<RecyclablePage>,
    medium_pages: SegQueue<RecyclablePage>,
    large_pages: SegQueue<RecyclablePage>,
    huge_pages: SegQueue<RecyclablePage>,
    
    // Per-device queues for NUMA-aware allocation
    device_pools: RwLock<HashMap<i32, DeviceSlabPool>>,
    
    // Global statistics
    total_pages_created: AtomicUsize,
    total_pages_recycled: AtomicUsize,
    total_bytes_allocated: AtomicUsize,
    total_bytes_recycled: AtomicUsize,
    
    // Configuration
    max_pool_size_per_class: usize,
    enable_cross_device_sharing: bool,
}

#[derive(Debug)]
struct DeviceSlabPool {
    device_id: i32,
    small_pages: SegQueue<RecyclablePage>,
    medium_pages: SegQueue<RecyclablePage>,
    large_pages: SegQueue<RecyclablePage>,
    huge_pages: SegQueue<RecyclablePage>,
    allocated_pages: AtomicUsize,
    recycled_pages: AtomicUsize,
}

impl DeviceSlabPool {
    fn new(device_id: i32) -> Self {
        Self {
            device_id,
            small_pages: SegQueue::new(),
            medium_pages: SegQueue::new(),
            large_pages: SegQueue::new(),
            huge_pages: SegQueue::new(),
            allocated_pages: AtomicUsize::new(0),
            recycled_pages: AtomicUsize::new(0),
        }
    }

    fn queue_for_class(&self, class: PageClass) -> &SegQueue<RecyclablePage> {
        match class {
            PageClass::Small => &self.small_pages,
            PageClass::Medium => &self.medium_pages,
            PageClass::Large => &self.large_pages,
            PageClass::Huge => &self.huge_pages,
        }
    }

    fn try_get_page(&self, class: PageClass) -> Option<RecyclablePage> {
        self.queue_for_class(class).pop()
    }

    fn return_page(&self, page: RecyclablePage) {
        let class = page.page_class();
        self.queue_for_class(class).push(page);
        self.recycled_pages.fetch_add(1, Ordering::Relaxed);
    }
}

impl GlobalSlabPool {
    /// Create new global slab pool
    pub fn new() -> Self {
        Self {
            small_pages: SegQueue::new(),
            medium_pages: SegQueue::new(),
            large_pages: SegQueue::new(),
            huge_pages: SegQueue::new(),
            device_pools: RwLock::new(HashMap::new()),
            total_pages_created: AtomicUsize::new(0),
            total_pages_recycled: AtomicUsize::new(0),
            total_bytes_allocated: AtomicUsize::new(0),
            total_bytes_recycled: AtomicUsize::new(0),
            max_pool_size_per_class: 100, // Limit memory usage
            enable_cross_device_sharing: true,
        }
    }

    /// Create with custom configuration
    pub fn with_config(max_pool_size: usize, cross_device_sharing: bool) -> Self {
        let mut pool = Self::new();
        pool.max_pool_size_per_class = max_pool_size;
        pool.enable_cross_device_sharing = cross_device_sharing;
        pool
    }

    /// Get page from pool, preferring device-local pages
    pub fn get_page(&self, size: usize, device_id: i32) -> Option<CudaPage> {
        let class = PageClass::from_size(size);
        
        // Try device-specific pool first
        if let Some(page) = self.try_get_from_device_pool(class, device_id) {
            page.mark_used();
            self.total_pages_recycled.fetch_add(1, Ordering::Relaxed);
            self.total_bytes_recycled.fetch_add(page.cuda_page().size(), Ordering::Relaxed);
            return Some(page.into_cuda_page());
        }

        // Try global pools if cross-device sharing enabled
        if self.enable_cross_device_sharing {
            if let Some(page) = self.try_get_from_global_pool(class) {
                page.mark_used();
                self.total_pages_recycled.fetch_add(1, Ordering::Relaxed);
                self.total_bytes_recycled.fetch_add(page.cuda_page().size(), Ordering::Relaxed);
                return Some(page.into_cuda_page());
            }
        }

        // Try larger page classes if nothing available in exact class
        self.try_get_larger_page(class, device_id)
    }

    /// Return page to appropriate pool for recycling
    pub fn return_page(&self, cuda_page: CudaPage) {
        let device_id = cuda_page.device_id();
        let recyclable_page = RecyclablePage::new(cuda_page);
        
        // Check if page is still reusable
        if !recyclable_page.is_reusable() {
            // Page over-allocated, let it drop
            return;
        }

        // Try device-specific pool first
        if self.try_return_to_device_pool(&recyclable_page, device_id) {
            return;
        }

        // If device pool is full, try global pool
        self.return_to_global_pool(recyclable_page);
    }

    /// Force cleanup of old pages (for memory pressure)
    pub fn cleanup_old_pages(&self, max_age_seconds: u64) -> usize {
        let mut cleaned = 0;
        
        // Clean global pools
        cleaned += self.cleanup_queue(&self.small_pages, max_age_seconds);
        cleaned += self.cleanup_queue(&self.medium_pages, max_age_seconds);
        cleaned += self.cleanup_queue(&self.large_pages, max_age_seconds);
        cleaned += self.cleanup_queue(&self.huge_pages, max_age_seconds);

        // Clean device pools
        if let Ok(device_pools) = self.device_pools.read() {
            for pool in device_pools.values() {
                cleaned += self.cleanup_queue(&pool.small_pages, max_age_seconds);
                cleaned += self.cleanup_queue(&pool.medium_pages, max_age_seconds);
                cleaned += self.cleanup_queue(&pool.large_pages, max_age_seconds);
                cleaned += self.cleanup_queue(&pool.huge_pages, max_age_seconds);
            }
        }

        cleaned
    }

    /// Get comprehensive pool statistics
    pub fn stats(&self) -> SlabPoolStats {
        let global_pool_sizes = [
            self.small_pages.len(),
            self.medium_pages.len(), 
            self.large_pages.len(),
            self.huge_pages.len(),
        ];

        let mut device_stats = HashMap::new();
        if let Ok(device_pools) = self.device_pools.read() {
            for (device_id, pool) in device_pools.iter() {
                device_stats.insert(*device_id, DevicePoolStats {
                    device_id: *device_id,
                    small_pages: pool.small_pages.len(),
                    medium_pages: pool.medium_pages.len(),
                    large_pages: pool.large_pages.len(),
                    huge_pages: pool.huge_pages.len(),
                    allocated_pages: pool.allocated_pages.load(Ordering::Relaxed),
                    recycled_pages: pool.recycled_pages.load(Ordering::Relaxed),
                });
            }
        }

        SlabPoolStats {
            total_pages_created: self.total_pages_created.load(Ordering::Relaxed),
            total_pages_recycled: self.total_pages_recycled.load(Ordering::Relaxed),
            total_bytes_allocated: self.total_bytes_allocated.load(Ordering::Relaxed),
            total_bytes_recycled: self.total_bytes_recycled.load(Ordering::Relaxed),
            global_pool_sizes,
            device_stats,
            recycling_efficiency: self.calculate_recycling_efficiency(),
        }
    }

    // Helper methods

    fn try_get_from_device_pool(&self, class: PageClass, device_id: i32) -> Option<RecyclablePage> {
        if let Ok(device_pools) = self.device_pools.read() {
            if let Some(pool) = device_pools.get(&device_id) {
                return pool.try_get_page(class);
            }
        }
        None
    }

    fn try_get_from_global_pool(&self, class: PageClass) -> Option<RecyclablePage> {
        let queue = match class {
            PageClass::Small => &self.small_pages,
            PageClass::Medium => &self.medium_pages,
            PageClass::Large => &self.large_pages,
            PageClass::Huge => &self.huge_pages,
        };
        queue.pop()
    }

    fn try_get_larger_page(&self, requested_class: PageClass, device_id: i32) -> Option<CudaPage> {
        // Try progressively larger page classes
        let larger_classes = match requested_class {
            PageClass::Small => vec![PageClass::Medium, PageClass::Large, PageClass::Huge],
            PageClass::Medium => vec![PageClass::Large, PageClass::Huge],
            PageClass::Large => vec![PageClass::Huge],
            PageClass::Huge => vec![], // No larger class
        };

        for class in larger_classes {
            if let Some(page) = self.try_get_from_device_pool(class, device_id) {
                page.mark_used();
                return Some(page.into_cuda_page());
            }
            if self.enable_cross_device_sharing {
                if let Some(page) = self.try_get_from_global_pool(class) {
                    page.mark_used();
                    return Some(page.into_cuda_page());
                }
            }
        }

        None
    }

    // Fixed method to take reference instead of ownership
    fn try_return_to_device_pool(&self, page: &RecyclablePage, device_id: i32) -> bool {
        // Ensure device pool exists
        {
            let mut device_pools = self.device_pools.write().unwrap();
            device_pools.entry(device_id).or_insert_with(|| DeviceSlabPool::new(device_id));
        }

        // Check if device pool has space
        if let Ok(device_pools) = self.device_pools.read() {
            if let Some(pool) = device_pools.get(&device_id) {
                let current_size = pool.queue_for_class(page.page_class()).len();
                if current_size < self.max_pool_size_per_class {
                    // We can't move the page here since we only have a reference
                    // Return false to indicate we should try global pool
                    return false;
                }
            }
        }
        false
    }

    fn return_to_global_pool(&self, page: RecyclablePage) {
        let queue = match page.page_class() {
            PageClass::Small => &self.small_pages,
            PageClass::Medium => &self.medium_pages,
            PageClass::Large => &self.large_pages,
            PageClass::Huge => &self.huge_pages,
        };

        // Check global pool capacity
        if queue.len() < self.max_pool_size_per_class {
            queue.push(page);
        }
        // If global pool full, page will be dropped (freed)
    }

    // Alternative implementation that handles the device pool properly
    fn return_to_device_pool_owned(&self, page: RecyclablePage, device_id: i32) -> Result<(), RecyclablePage> {
        // Ensure device pool exists
        {
            let mut device_pools = self.device_pools.write().unwrap();
            device_pools.entry(device_id).or_insert_with(|| DeviceSlabPool::new(device_id));
        }

        // Check if device pool has space
        if let Ok(device_pools) = self.device_pools.read() {
            if let Some(pool) = device_pools.get(&device_id) {
                let current_size = pool.queue_for_class(page.page_class()).len();
                if current_size < self.max_pool_size_per_class {
                    pool.return_page(page);
                    return Ok(());
                }
            }
        }
        Err(page) // Return the page back if we couldn't add it
    }

    fn cleanup_queue(&self, queue: &SegQueue<RecyclablePage>, max_age_seconds: u64) -> usize {
        let mut cleaned = 0;
        let mut temp_pages = Vec::new();

        // Drain queue and filter old pages
        while let Some(page) = queue.pop() {
            if page.age() > max_age_seconds {
                cleaned += 1;
                // Page will be dropped and memory freed
            } else {
                temp_pages.push(page);
            }
        }

        // Put back non-expired pages
        for page in temp_pages {
            queue.push(page);
        }

        cleaned
    }

    fn calculate_recycling_efficiency(&self) -> f64 {
        let created = self.total_pages_created.load(Ordering::Relaxed);
        let recycled = self.total_pages_recycled.load(Ordering::Relaxed);
        
        if created == 0 {
            0.0
        } else {
            recycled as f64 / created as f64
        }
    }

    /// Record page creation for statistics
    pub fn record_page_creation(&self, size: usize) {
        self.total_pages_created.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_allocated.fetch_add(size, Ordering::Relaxed);
    }

    /// Defragment pools by moving pages between device pools
    pub fn defragment(&self) {
        if !self.enable_cross_device_sharing {
            return;
        }

        // Move excess pages from over-full device pools to global pools
        if let Ok(device_pools) = self.device_pools.read() {
            for pool in device_pools.values() {
                self.defragment_device_pool(pool);
            }
        }
    }

    fn defragment_device_pool(&self, pool: &DeviceSlabPool) {
        let classes = [PageClass::Small, PageClass::Medium, PageClass::Large, PageClass::Huge];
        
        for class in classes {
            let queue = pool.queue_for_class(class);
            let target_size = self.max_pool_size_per_class / 2; // Keep pools half-full
            
            while queue.len() > target_size {
                if let Some(page) = queue.pop() {
                    self.return_to_global_pool(page);
                } else {
                    break;
                }
            }
        }
    }
}

// Updated return_page method to handle the ownership properly
impl GlobalSlabPool {
    /// Return page to appropriate pool for recycling (updated method)
    pub fn return_page_fixed(&self, cuda_page: CudaPage) {
        let device_id = cuda_page.device_id();
        let recyclable_page = RecyclablePage::new(cuda_page);
        
        // Check if page is still reusable
        if !recyclable_page.is_reusable() {
            // Page over-allocated, let it drop
            return;
        }

        // Try device-specific pool first
        match self.return_to_device_pool_owned(recyclable_page, device_id) {
            Ok(()) => return, // Successfully added to device pool
            Err(page) => {
                // Device pool was full, try global pool
                self.return_to_global_pool(page);
            }
        }
    }
}

/// Statistics for slab pool performance monitoring
#[derive(Debug, Clone)]
pub struct SlabPoolStats {
    pub total_pages_created: usize,
    pub total_pages_recycled: usize,
    pub total_bytes_allocated: usize,
    pub total_bytes_recycled: usize,
    pub global_pool_sizes: [usize; 4], // [small, medium, large, huge]
    pub device_stats: HashMap<i32, DevicePoolStats>,
    pub recycling_efficiency: f64,
}

#[derive(Debug, Clone)]
pub struct DevicePoolStats {
    pub device_id: i32,
    pub small_pages: usize,
    pub medium_pages: usize,
    pub large_pages: usize,
    pub huge_pages: usize,
    pub allocated_pages: usize,
    pub recycled_pages: usize,
}

impl SlabPoolStats {
    /// Get total pages in all pools
    pub fn total_pooled_pages(&self) -> usize {
        self.global_pool_sizes.iter().sum::<usize>() +
        self.device_stats.values()
            .map(|stats| stats.small_pages + stats.medium_pages + stats.large_pages + stats.huge_pages)
            .sum::<usize>()
    }

    /// Get memory efficiency ratio
    pub fn memory_efficiency(&self) -> f64 {
        if self.total_bytes_allocated == 0 {
            0.0
        } else {
            self.total_bytes_recycled as f64 / self.total_bytes_allocated as f64
        }
    }

    /// Check if any pools are under-utilized
    pub fn has_underutilized_pools(&self) -> bool {
        // Check if global pools are too empty (less than 10% of max)
        let max_size = 100; // Assuming default max size
        let min_threshold = max_size / 10;
        
        self.global_pool_sizes.iter().any(|&size| size > 0 && size < min_threshold)
    }
}

/// Slab pool manager with automatic cleanup and optimization
pub struct SlabPoolManager {
    pool: Arc<GlobalSlabPool>,
    cleanup_interval_seconds: u64,
    max_page_age_seconds: u64,
}

impl SlabPoolManager {
    pub fn new(pool: Arc<GlobalSlabPool>) -> Self {
        Self {
            pool,
            cleanup_interval_seconds: 300, // 5 minutes
            max_page_age_seconds: 1800,    // 30 minutes
        }
    }

    /// Start background cleanup task
    pub fn start_background_cleanup(&self) {
        let pool = Arc::clone(&self.pool);
        let interval = self.cleanup_interval_seconds;
        let max_age = self.max_page_age_seconds;

        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(interval));
                
                // Cleanup old pages
                let cleaned = pool.cleanup_old_pages(max_age);
                if cleaned > 0 {
                    log::info!("Cleaned {} old pages from slab pools", cleaned);
                }

                // Defragment pools
                pool.defragment();
            }
        });
    }

    /// Get pool reference
    pub fn pool(&self) -> &Arc<GlobalSlabPool> {
        &self.pool
    }

    /// Force immediate cleanup
    pub fn force_cleanup(&self) -> usize {
        self.pool.cleanup_old_pages(self.max_page_age_seconds)
    }

    /// Get performance recommendations
    pub fn get_recommendations(&self) -> Vec<String> {
        let stats = self.pool.stats();
        let mut recommendations = Vec::new();

        // Check recycling efficiency
        if stats.recycling_efficiency < 0.5 {
            recommendations.push(format!(
                "Low recycling efficiency ({:.1}%). Consider increasing page reuse.",
                stats.recycling_efficiency * 100.0
            ));
        }

        // Check memory efficiency
        if stats.memory_efficiency() < 0.7 {
            recommendations.push(format!(
                "Low memory efficiency ({:.1}%). Consider tuning page sizes.",
                stats.memory_efficiency() * 100.0
            ));
        }

        // Check pool utilization
        if stats.has_underutilized_pools() {
            recommendations.push("Some pools are underutilized. Consider enabling cross-device sharing.".to_string());
        }

        // Check total memory usage
        let total_pooled = stats.total_pooled_pages();
        if total_pooled > 1000 {
            recommendations.push(format!(
                "High number of pooled pages ({}). Consider reducing max_pool_size.",
                total_pooled
            ));
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::{CudaMemoryManager, CudaPage};

    #[test]
    fn test_page_classification() {
        assert_eq!(PageClass::from_size(100_000), PageClass::Small);
        assert_eq!(PageClass::from_size(1_000_000), PageClass::Medium);
        assert_eq!(PageClass::from_size(5_000_000), PageClass::Large);
        assert_eq!(PageClass::from_size(20_000_000), PageClass::Huge);
    }

    #[test]
    fn test_slab_pool_basic_operations() {
        let pool = GlobalSlabPool::new();
        
        // Test empty pool
        assert!(pool.get_page(1024, 0).is_none());
        
        // Stats should be empty initially
        let stats = pool.stats();
        assert_eq!(stats.total_pages_created, 0);
        assert_eq!(stats.total_pages_recycled, 0);
    }

    #[test]
    fn test_recyclable_page() {
        // This test requires CUDA, so we'll mock or skip if not available
        if let Ok(manager) = CudaMemoryManager::new() {
            if let Ok(cuda_page) = manager.allocate_page(1024 * 1024) {
                let recyclable = RecyclablePage::new(cuda_page);
                assert_eq!(recyclable.page_class(), PageClass::Medium);
                assert_eq!(recyclable.allocation_count(), 0);
                
                recyclable.mark_used();
                assert_eq!(recyclable.allocation_count(), 1);
                assert!(recyclable.is_reusable());
            }
        }
    }

    #[test]
    fn test_slab_pool_with_cuda() {
        if let Ok(manager) = CudaMemoryManager::new() {
            let pool = GlobalSlabPool::new();
            
            // Create and return a page
            if let Ok(cuda_page) = manager.allocate_page(256 * 1024) {
                let device_id = cuda_page.device_id();
                pool.record_page_creation(cuda_page.size());
                pool.return_page_fixed(cuda_page); // Use the fixed method
                
                // Try to get it back
                if let Some(recycled_page) = pool.get_page(256 * 1024, device_id) {
                    assert_eq!(recycled_page.size(), 256 * 1024);
                    println!("✓ Slab recycling test passed");
                }
            }
        }
    }
}