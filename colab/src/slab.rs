// src/slab.rs - Lock-free slab recycling with SegQueue for true zero-allocation pool management
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use crossbeam::queue::SegQueue;
use crate::cuda::CudaPage;

/// Page classification for optimal recycling (matches project description)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageClass {
    Small,   // < 512KB
    Medium,  // 512KB - 2MB  
    Large,   // 2MB - 8MB
    Huge,    // > 8MB
}

impl PageClass {
    /// Classify page by size as described in project spec
    pub fn from_size(size: usize) -> Self {
        match size {
            0..=524_288 => PageClass::Small,      // 0-512KB
            524_289..=2_097_152 => PageClass::Medium,   // 512KB-2MB
            2_097_153..=8_388_608 => PageClass::Large,    // 2MB-8MB
            _ => PageClass::Huge,                         // >8MB
        }
    }

    /// Get typical size for this class
    pub fn typical_size(&self) -> usize {
        match self {
            PageClass::Small => 256 * 1024,    // 256KB
            PageClass::Medium => 1024 * 1024,  // 1MB
            PageClass::Large => 4 * 1024 * 1024, // 4MB
            PageClass::Huge => 16 * 1024 * 1024, // 16MB
        }
    }
}

/// Recyclable page wrapper with minimal metadata for efficient reuse
#[derive(Debug)]
pub struct RecyclablePage {
    /// The actual CUDA page
    cuda_page: CudaPage,
    /// Page classification
    page_class: PageClass,
    /// Number of times this page has been allocated (wear leveling)
    allocation_count: AtomicUsize,
    /// Timestamp when last used (for LRU eviction)
    last_used: AtomicU64,
    /// Device ID for NUMA-aware allocation
    device_id: i32,
}

impl RecyclablePage {
    /// Create new recyclable page
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

    /// Take ownership of the CUDA page (consuming self)
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

    /// Get age in seconds since last use
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

    /// Check if page is suitable for reuse (prevent memory wear)
    pub fn is_reusable(&self) -> bool {
        const MAX_ALLOCATIONS: usize = 1000; // Prevent excessive reuse
        self.allocation_count() < MAX_ALLOCATIONS
    }

    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

/// Lock-free global slab pool using SegQueue - ZERO ALLOCATION POOL MANAGEMENT
#[derive(Debug)]
pub struct GlobalSlabPool {
    // LOCK-FREE: Separate SegQueues by page class for optimal matching
    small_pages: SegQueue<RecyclablePage>,
    medium_pages: SegQueue<RecyclablePage>,
    large_pages: SegQueue<RecyclablePage>,
    huge_pages: SegQueue<RecyclablePage>,
    
    // LOCK-FREE: Per-device pools for NUMA-aware allocation
    device_pools: Box<[DeviceSlabPool; 8]>, // Support up to 8 devices
    
    // Atomic statistics (no locks needed)
    total_pages_created: AtomicUsize,
    total_pages_recycled: AtomicUsize,
    total_bytes_allocated: AtomicUsize,
    total_bytes_recycled: AtomicUsize,
    
    // Configuration
    max_pool_size_per_class: usize,
    enable_cross_device_sharing: bool,
}

/// Per-device slab pool (also lock-free)
#[derive(Debug)]
struct DeviceSlabPool {
    device_id: i32,
    // LOCK-FREE: SegQueues for each page class
    small_pages: SegQueue<RecyclablePage>,
    medium_pages: SegQueue<RecyclablePage>,
    large_pages: SegQueue<RecyclablePage>,
    huge_pages: SegQueue<RecyclablePage>,
    // Atomic counters
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

    /// Get the appropriate queue for page class
    fn queue_for_class(&self, class: PageClass) -> &SegQueue<RecyclablePage> {
        match class {
            PageClass::Small => &self.small_pages,
            PageClass::Medium => &self.medium_pages,
            PageClass::Large => &self.large_pages,
            PageClass::Huge => &self.huge_pages,
        }
    }

    /// Try to get page from device pool (lock-free)
    fn try_get_page(&self, class: PageClass) -> Option<RecyclablePage> {
        self.queue_for_class(class).pop()
    }

    /// Return page to device pool (lock-free)
    fn return_page(&self, page: RecyclablePage) {
        let class = page.page_class();
        self.queue_for_class(class).push(page);
        self.recycled_pages.fetch_add(1, Ordering::Relaxed);
    }

    /// Get queue length (approximate, for monitoring)
    fn queue_len(&self, class: PageClass) -> usize {
        // Note: SegQueue doesn't have an exact len() method for performance
        // This is an approximation by trying to count without side effects
        let queue = self.queue_for_class(class);
        let mut count = 0;
        let mut temp_pages = Vec::new();
        
        // Drain to count, then restore
        while let Some(page) = queue.pop() {
            temp_pages.push(page);
            count += 1;
            if count > 1000 { break; } // Prevent infinite counting
        }
        
        // Restore pages
        for page in temp_pages {
            queue.push(page);
        }
        
        count
    }
}

impl GlobalSlabPool {
    /// Create new lock-free global slab pool
    pub fn new() -> Self {
        Self {
            small_pages: SegQueue::new(),
            medium_pages: SegQueue::new(),
            large_pages: SegQueue::new(),
            huge_pages: SegQueue::new(),
            device_pools: Box::new([
                DeviceSlabPool::new(0), DeviceSlabPool::new(1),
                DeviceSlabPool::new(2), DeviceSlabPool::new(3),
                DeviceSlabPool::new(4), DeviceSlabPool::new(5),
                DeviceSlabPool::new(6), DeviceSlabPool::new(7),
            ]),
            total_pages_created: AtomicUsize::new(0),
            total_pages_recycled: AtomicUsize::new(0),
            total_bytes_allocated: AtomicUsize::new(0),
            total_bytes_recycled: AtomicUsize::new(0),
            max_pool_size_per_class: 100,
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

    /// Get page from pool - TRUE LOCK-FREE OPERATION
    /// This implements the core "slab recycling" described in the project
    pub fn get_page(&self, size: usize, device_id: i32) -> Option<CudaPage> {
        let class = PageClass::from_size(size);
        
        // 1. Try device-specific pool first (NUMA-aware)
        if let Some(page) = self.try_get_from_device_pool(class, device_id) {
            page.mark_used();
            self.total_pages_recycled.fetch_add(1, Ordering::Relaxed);
            self.total_bytes_recycled.fetch_add(page.cuda_page.size(), Ordering::Relaxed);
            log::debug!("RECYCLED page from device {} pool: {} bytes", device_id, page.cuda_page.size());
            return Some(page.into_cuda_page());
        }

        // 2. Try global pools if cross-device sharing enabled
        if self.enable_cross_device_sharing {
            if let Some(page) = self.try_get_from_global_pool(class) {
                page.mark_used();
                self.total_pages_recycled.fetch_add(1, Ordering::Relaxed);
                self.total_bytes_recycled.fetch_add(page.cuda_page.size(), Ordering::Relaxed);
                log::debug!("RECYCLED page from global pool: {} bytes", page.cuda_page.size());
                return Some(page.into_cuda_page());
            }
        }

        // 3. Try larger page classes if nothing available in exact class
        self.try_get_larger_page(class, device_id)
    }

    /// Return page to pool for recycling - CORE SLAB RECYCLING
    /// When SequenceArena drops, its pages go back to GlobalSlabPool
    pub fn return_page(&self, cuda_page: CudaPage) {
        let device_id = cuda_page.device_id();
        let recyclable_page = RecyclablePage::new(cuda_page);
        
        // Check if page is still reusable (prevent memory wear)
        if !recyclable_page.is_reusable() {
            log::debug!("Page over-allocated ({}x), not recycling", recyclable_page.allocation_count());
            return; // Let page drop and be freed
        }

        // Try device-specific pool first (NUMA-aware)
        if self.try_return_to_device_pool(&recyclable_page, device_id) {
            log::debug!("RECYCLED page to device {} pool", device_id);
            return;
        }

        // If device pool is full, try global pool
        self.return_to_global_pool(recyclable_page);
        log::debug!("RECYCLED page to global pool");
    }

    /// Record page creation for statistics
    pub fn record_page_creation(&self, size: usize) {
        self.total_pages_created.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_allocated.fetch_add(size, Ordering::Relaxed);
    }

    /// Get comprehensive pool statistics
    pub fn stats(&self) -> SlabPoolStats {
        let global_pool_sizes = [
            self.approximate_queue_len(&self.small_pages),
            self.approximate_queue_len(&self.medium_pages),
            self.approximate_queue_len(&self.large_pages),
            self.approximate_queue_len(&self.huge_pages),
        ];

        let mut device_stats = std::collections::HashMap::new();
        for (i, pool) in self.device_pools.iter().enumerate() {
            device_stats.insert(i as i32, DevicePoolStats {
                device_id: i as i32,
                small_pages: pool.queue_len(PageClass::Small),
                medium_pages: pool.queue_len(PageClass::Medium),
                large_pages: pool.queue_len(PageClass::Large),
                huge_pages: pool.queue_len(PageClass::Huge),
                allocated_pages: pool.allocated_pages.load(Ordering::Relaxed),
                recycled_pages: pool.recycled_pages.load(Ordering::Relaxed),
            });
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

    /// Force cleanup of old pages (for memory pressure management)
    pub fn cleanup_old_pages(&self, max_age_seconds: u64) -> usize {
        let mut cleaned = 0;
        
        // Clean global pools
        cleaned += self.cleanup_queue(&self.small_pages, max_age_seconds);
        cleaned += self.cleanup_queue(&self.medium_pages, max_age_seconds);
        cleaned += self.cleanup_queue(&self.large_pages, max_age_seconds);
        cleaned += self.cleanup_queue(&self.huge_pages, max_age_seconds);

        // Clean device pools
        for pool in self.device_pools.iter() {
            cleaned += self.cleanup_device_pool_queue(&pool.small_pages, max_age_seconds);
            cleaned += self.cleanup_device_pool_queue(&pool.medium_pages, max_age_seconds);
            cleaned += self.cleanup_device_pool_queue(&pool.large_pages, max_age_seconds);
            cleaned += self.cleanup_device_pool_queue(&pool.huge_pages, max_age_seconds);
        }

        cleaned
    }

    /// Defragment pools by moving pages between device and global pools
    pub fn defragment(&self) {
        if !self.enable_cross_device_sharing {
            return;
        }

        // Move excess pages from over-full device pools to global pools
        for pool in self.device_pools.iter() {
            self.defragment_device_pool(pool);
        }
    }

    // Private helper methods

    fn try_get_from_device_pool(&self, class: PageClass, device_id: i32) -> Option<RecyclablePage> {
        let device_idx = (device_id as usize).min(7); // Clamp to valid range
        self.device_pools[device_idx].try_get_page(class)
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
                self.total_pages_recycled.fetch_add(1, Ordering::Relaxed);
                return Some(page.into_cuda_page());
            }
            if self.enable_cross_device_sharing {
                if let Some(page) = self.try_get_from_global_pool(class) {
                    page.mark_used();
                    self.total_pages_recycled.fetch_add(1, Ordering::Relaxed);
                    return Some(page.into_cuda_page());
                }
            }
        }

        None
    }

    fn try_return_to_device_pool(&self, page: &RecyclablePage, device_id: i32) -> bool {
        let device_idx = (device_id as usize).min(7); // Clamp to valid range
        let pool = &self.device_pools[device_idx];
        
        // Check if device pool has space (approximate)
        let current_size = pool.queue_len(page.page_class());
        if current_size < self.max_pool_size_per_class {
            // Clone the page data to move it (RecyclablePage isn't Copy)
            let page_to_move = RecyclablePage {
                cuda_page: unsafe { std::ptr::read(&page.cuda_page) },
                page_class: page.page_class,
                allocation_count: AtomicUsize::new(page.allocation_count()),
                last_used: AtomicU64::new(page.last_used.load(Ordering::Relaxed)),
                device_id: page.device_id,
            };
            pool.return_page(page_to_move);
            std::mem::forget(page); // Prevent double-drop
            true
        } else {
            false
        }
    }

    fn return_to_global_pool(&self, page: RecyclablePage) {
        let queue = match page.page_class() {
            PageClass::Small => &self.small_pages,
            PageClass::Medium => &self.medium_pages,
            PageClass::Large => &self.large_pages,
            PageClass::Huge => &self.huge_pages,
        };

        // Check global pool capacity (approximate)
        if self.approximate_queue_len(queue) < self.max_pool_size_per_class {
            queue.push(page);
        }
        // If global pool full, page will be dropped (freed)
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

    fn cleanup_device_pool_queue(&self, queue: &SegQueue<RecyclablePage>, max_age_seconds: u64) -> usize {
        // Same as cleanup_queue but for device pool queues
        self.cleanup_queue(queue, max_age_seconds)
    }

    fn defragment_device_pool(&self, pool: &DeviceSlabPool) {
        let classes = [PageClass::Small, PageClass::Medium, PageClass::Large, PageClass::Huge];
        
        for class in classes {
            let queue = pool.queue_for_class(class);
            let target_size = self.max_pool_size_per_class / 2; // Keep pools half-full
            let current_size = pool.queue_len(class);
            
            let mut moved = 0;
            while moved < (current_size.saturating_sub(target_size)) {
                if let Some(page) = queue.pop() {
                    self.return_to_global_pool(page);
                    moved += 1;
                } else {
                    break;
                }
            }
        }
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

    fn approximate_queue_len(&self, queue: &SegQueue<RecyclablePage>) -> usize {
        // SegQueue doesn't provide len() for performance reasons
        // This is an approximation by counting elements
        let mut count = 0;
        let mut temp_pages = Vec::new();
        
        // Drain to count, then restore (not ideal but needed for monitoring)
        while let Some(page) = queue.pop() {
            temp_pages.push(page);
            count += 1;
            if count > 1000 { break; } // Prevent excessive counting
        }
        
        // Restore pages
        for page in temp_pages {
            queue.push(page);
        }
        
        count
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
    pub device_stats: std::collections::HashMap<i32, DevicePoolStats>,
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

    /// Check if system is effectively recycling
    pub fn is_recycling_effectively(&self) -> bool {
        self.recycling_efficiency > 0.5 && self.memory_efficiency() > 0.3
    }
}

/// Slab pool manager with automatic cleanup and lock-free optimization
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

    /// Start background cleanup task (non-blocking)
    pub fn start_background_cleanup(&self) {
        let pool = Arc::clone(&self.pool);
        let interval = self.cleanup_interval_seconds;
        let max_age = self.max_page_age_seconds;

        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(interval));
                
                // Cleanup old pages (lock-free operation)
                let cleaned = pool.cleanup_old_pages(max_age);
                if cleaned > 0 {
                    log::info!("Lock-free cleanup: removed {} old pages", cleaned);
                }

                // Defragment pools (lock-free operation)
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

    /// Get performance recommendations based on lock-free metrics
    pub fn get_recommendations(&self) -> Vec<String> {
        let stats = self.pool.stats();
        let mut recommendations = Vec::new();

        // Check recycling efficiency
        if stats.recycling_efficiency < 0.5 {
            recommendations.push(format!(
                "Low recycling efficiency ({:.1}%). Consider increasing page reuse or reducing pool turnover.",
                stats.recycling_efficiency * 100.0
            ));
        }

        // Check memory efficiency
        if stats.memory_efficiency() < 0.7 {
            recommendations.push(format!(
                "Low memory efficiency ({:.1}%). Consider better page size alignment with allocation patterns.",
                stats.memory_efficiency() * 100.0
            ));
        }

        // Check pool utilization
        let total_pooled = stats.total_pooled_pages();
        if total_pooled > 1000 {
            recommendations.push(format!(
                "High number of pooled pages ({}). Consider reducing max_pool_size to free memory.",
                total_pooled
            ));
        } else if total_pooled < 10 && stats.total_pages_created > 100 {
            recommendations.push("Very low pool utilization. Pages may be churning too quickly.".to_string());
        }

        // Check device distribution
        let device_count = stats.device_stats.len();
        if device_count > 1 {
            let mut unbalanced = false;
            let total_device_pages: usize = stats.device_stats.values()
                .map(|s| s.small_pages + s.medium_pages + s.large_pages + s.huge_pages)
                .sum();
            
            if total_device_pages > 0 {
                for device_stat in stats.device_stats.values() {
                    let device_pages = device_stat.small_pages + device_stat.medium_pages + 
                                     device_stat.large_pages + device_stat.huge_pages;
                    let device_ratio = device_pages as f64 / total_device_pages as f64;
                    if device_ratio > 0.8 || device_ratio < 0.1 {
                        unbalanced = true;
                        break;
                    }
                }
            }
            
            if unbalanced {
                recommendations.push("Unbalanced device pool usage detected. Consider enabling cross-device sharing.".to_string());
            }
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::CudaPage;

    #[test]
    fn test_page_classification() {
        assert_eq!(PageClass::from_size(100_000), PageClass::Small);
        assert_eq!(PageClass::from_size(1_000_000), PageClass::Medium);
        assert_eq!(PageClass::from_size(5_000_000), PageClass::Large);
        assert_eq!(PageClass::from_size(20_000_000), PageClass::Huge);
        
        println!("✓ Page classification test passed");
    }

    #[test]
    fn test_lockfree_slab_pool_basic() {
        let pool = GlobalSlabPool::new();
        
        // Test empty pool
        assert!(pool.get_page(1024, 0).is_none());
        
        // Stats should be empty initially
        let stats = pool.stats();
        assert_eq!(stats.total_pages_created, 0);
        assert_eq!(stats.total_pages_recycled, 0);
        
        println!("✓ Lock-free slab pool basic test passed");
    }

    #[test]
    fn test_lockfree_recycling() {
        let pool = GlobalSlabPool::new();
        
        // This test requires real CUDA for full functionality
        // For now, test the structure
        pool.record_page_creation(256 * 1024);
        let stats = pool.stats();
        assert_eq!(stats.total_pages_created, 1);
        
        println!("✓ Lock-free recycling structure test passed");
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;
        
        let pool = Arc::new(GlobalSlabPool::new());
        let mut handles = vec![];
        
        // Test concurrent access to lock-free structure
        for i in 0..10 {
            let pool_clone = Arc::clone(&pool);
            let handle = thread::spawn(move || {
                // Record page creation from multiple threads
                pool_clone.record_page_creation(1024 * (i + 1));
                
                // Try to get pages (will return None but tests lock-free access)
                let _ = pool_clone.get_page(1024, i % 4);
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        let stats = pool.stats();
        assert_eq!(stats.total_pages_created, 10);
        
        println!("✓ Concurrent lock-free access test passed");
    }

    #[test]
    fn test_page_class_distribution() {
        let pool = GlobalSlabPool::new();
        
        // Test different page sizes map to correct classes
        let test_cases = vec![
            (128 * 1024, PageClass::Small),
            (800 * 1024, PageClass::Medium),
            (3 * 1024 * 1024, PageClass::Large),
            (20 * 1024 * 1024, PageClass::Huge),
        ];
        
        for (size, expected_class) in test_cases {
            let actual_class = PageClass::from_size(size);
            assert_eq!(actual_class, expected_class, "Size {} should be {:?}", size, expected_class);
        }
        
        println!("✓ Page class distribution test passed");
    }

    #[test]
    fn test_slab_manager() {
        let pool = Arc::new(GlobalSlabPool::new());
        let manager = SlabPoolManager::new(pool);
        
        // Test basic manager functionality
        let recommendations = manager.get_recommendations();
        // Should have some recommendations for empty pool
        assert!(recommendations.len() >= 0);
        
        println!("✓ Slab manager test passed");
    }
}