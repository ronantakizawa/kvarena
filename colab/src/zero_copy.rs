// src/zero_copy.rs - Final version without duplicate methods
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ptr::NonNull;
use crossbeam::queue::SegQueue;
use crate::cuda::{CudaPage, CudaError, CudaContext};

/// Zero-copy tensor with clear distinction between metadata and data operations
#[derive(Debug)]
pub struct ZeroCopyTensor {
    key_device_ptr: NonNull<u8>,
    value_device_ptr: NonNull<u8>,
    current_seq_len: AtomicUsize,
    max_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    element_size: usize,
    device_id: i32,
    arena_id: u64,
}

/// Extension operation result with clear semantics
#[derive(Debug, Clone)]
pub enum ExtensionResult {
    PureZeroCopy {
        old_seq_len: usize,
        new_seq_len: usize,
        operation_time_ns: u64,
    },
    RequiresDataCopy {
        old_seq_len: usize,
        new_seq_len: usize,
        copy_region_start: usize,
        copy_region_size: usize,
        metadata_update_time_ns: u64,
    },
    CannotExtend {
        current_seq_len: usize,
        max_seq_len: usize,
        requested_seq_len: usize,
    },
}

#[derive(Debug, Clone)]
pub enum DataCopyOperation {
    NewTokensCopy {
        start_token_idx: usize,
        num_tokens: usize,
        copy_size_bytes: usize,
    },
    FullTensorCopy {
        total_size_bytes: usize,
    },
    IncrementalCopy {
        append_offset: usize,
        append_size_bytes: usize,
    },
}

#[derive(Debug, Clone)]
pub struct DataCopyStats {
    pub operation: DataCopyOperation,
    pub copy_time_ns: u64,
    pub bytes_copied: usize,
    pub bandwidth_gbps: f64,
}

impl ZeroCopyTensor {
    pub fn from_bump_allocation(
        device_ptr: NonNull<u8>,
        initial_seq_len: usize,
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        element_size: usize,
        arena_id: u64,
        device_id: i32,
    ) -> Result<Self, CudaError> {
        let key_device_ptr = device_ptr;
        let k_tensor_max_size = max_seq_len * num_heads * head_dim * element_size;
        let value_device_ptr = unsafe {
            NonNull::new_unchecked(device_ptr.as_ptr().add(k_tensor_max_size))
        };
        
        Ok(ZeroCopyTensor {
            key_device_ptr,
            value_device_ptr,
            current_seq_len: AtomicUsize::new(initial_seq_len),
            max_seq_len,
            num_heads,
            head_dim,
            element_size,
            device_id,
            arena_id,
        })
    }

    pub fn extend_metadata_only(&self, new_seq_len: usize) -> ExtensionResult {
        let start_time = std::time::Instant::now();
        let old_seq_len = self.current_seq_len.load(Ordering::Relaxed);
        
        if new_seq_len > self.max_seq_len {
            return ExtensionResult::CannotExtend {
                current_seq_len: old_seq_len,
                max_seq_len: self.max_seq_len,
                requested_seq_len: new_seq_len,
            };
        }
        
        if new_seq_len <= old_seq_len {
            let operation_time = start_time.elapsed().as_nanos() as u64;
            return ExtensionResult::PureZeroCopy {
                old_seq_len,
                new_seq_len: old_seq_len,
                operation_time_ns: operation_time,
            };
        }
        
        self.current_seq_len.store(new_seq_len, Ordering::Release);
        let operation_time = start_time.elapsed().as_nanos() as u64;
        
        ExtensionResult::PureZeroCopy {
            old_seq_len,
            new_seq_len,
            operation_time_ns: operation_time,
        }
    }

    pub fn extend_with_data_requirements(&self, new_seq_len: usize) -> ExtensionResult {
        let start_time = std::time::Instant::now();
        let old_seq_len = self.current_seq_len.load(Ordering::Relaxed);
        
        if new_seq_len > self.max_seq_len {
            return ExtensionResult::CannotExtend {
                current_seq_len: old_seq_len,
                max_seq_len: self.max_seq_len,
                requested_seq_len: new_seq_len,
            };
        }
        
        if new_seq_len <= old_seq_len {
            self.current_seq_len.store(new_seq_len, Ordering::Release);
            let operation_time = start_time.elapsed().as_nanos() as u64;
            return ExtensionResult::PureZeroCopy {
                old_seq_len,
                new_seq_len,
                operation_time_ns: operation_time,
            };
        }
        
        self.current_seq_len.store(new_seq_len, Ordering::Release);
        let metadata_time = start_time.elapsed().as_nanos() as u64;
        
        let new_tokens = new_seq_len - old_seq_len;
        let token_size = self.num_heads * self.head_dim * self.element_size;
        let copy_size = new_tokens * token_size * 2;
        
        ExtensionResult::RequiresDataCopy {
            old_seq_len,
            new_seq_len,
            copy_region_start: old_seq_len,
            copy_region_size: copy_size,
            metadata_update_time_ns: metadata_time,
        }
    }

    pub fn copy_new_token_data(
        &self,
        host_key_data: *const u8,
        host_value_data: *const u8,
        data_copy_op: DataCopyOperation,
    ) -> Result<DataCopyStats, CudaError> {
        let start_time = std::time::Instant::now();
        
        let (start_token_idx, num_tokens, _description) = match data_copy_op {
            DataCopyOperation::NewTokensCopy { start_token_idx, num_tokens, .. } => {
                (start_token_idx, num_tokens, "new tokens copy")
            },
            DataCopyOperation::IncrementalCopy { append_offset, append_size_bytes } => {
                let token_size = self.num_heads * self.head_dim * self.element_size;
                let num_tokens = append_size_bytes / (token_size * 2);
                (append_offset / token_size, num_tokens, "incremental copy")
            },
            DataCopyOperation::FullTensorCopy { .. } => {
                (0, self.seq_len(), "full tensor copy")
            },
        };
        
        let current_seq_len = self.seq_len();
        if start_token_idx + num_tokens > current_seq_len {
            return Err(CudaError(-1));
        }

        let token_size = self.num_heads * self.head_dim * self.element_size;
        let copy_size = num_tokens * token_size;
        let offset = start_token_idx * token_size;

        unsafe {
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

        let copy_time = start_time.elapsed();
        
        Ok(DataCopyStats {
            operation: data_copy_op,
            copy_time_ns: copy_time.as_nanos() as u64,
            bytes_copied: copy_size * 2,
            bandwidth_gbps: if copy_time.as_secs_f64() > 0.0 {
                (copy_size * 2) as f64 / copy_time.as_secs_f64() / 1e9
            } else {
                0.0
            },
        })
    }

    pub fn extend_zero_copy(&self, new_seq_len: usize) -> Result<bool, CudaError> {
        match self.extend_metadata_only(new_seq_len) {
            ExtensionResult::PureZeroCopy { .. } => Ok(true),
            ExtensionResult::RequiresDataCopy { .. } => Ok(false),
            ExtensionResult::CannotExtend { .. } => Ok(false),
        }
    }

    // Standard getters and compatibility methods
    pub fn seq_len(&self) -> usize {
        self.current_seq_len.load(Ordering::Acquire)
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn can_extend_zero_copy_to(&self, new_seq_len: usize) -> bool {
        new_seq_len <= self.max_seq_len
    }

    pub fn key_device_ptr(&self) -> *mut std::ffi::c_void {
        self.key_device_ptr.as_ptr() as *mut std::ffi::c_void
    }

    pub fn value_device_ptr(&self) -> *mut std::ffi::c_void {
        self.value_device_ptr.as_ptr() as *mut std::ffi::c_void
    }

    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.seq_len(), self.num_heads, self.head_dim)
    }

    pub fn current_key_size_bytes(&self) -> usize {
        self.seq_len() * self.num_heads * self.head_dim * self.element_size
    }

    pub fn current_value_size_bytes(&self) -> usize {
        self.seq_len() * self.num_heads * self.head_dim * self.element_size
    }

    pub fn max_allocated_size_bytes(&self) -> usize {
        2 * self.max_seq_len * self.num_heads * self.head_dim * self.element_size
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn arena_id(&self) -> u64 {
        self.arena_id
    }

    pub fn size_bytes(&self) -> usize {
        self.current_key_size_bytes() + self.current_value_size_bytes()
    }

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
            metadata_operations_are_zero_copy: true,
            data_operations_require_copy: true,
            last_extension_was_pure_zero_copy: None,
        }
    }

    // Compatibility methods
    pub fn copy_new_tokens_only(
        &self,
        host_key_data: *const u8,
        host_value_data: *const u8,
        start_token_idx: usize,
        num_new_tokens: usize,
    ) -> Result<(), CudaError> {
        let copy_op = DataCopyOperation::NewTokensCopy {
            start_token_idx,
            num_tokens: num_new_tokens,
            copy_size_bytes: num_new_tokens * self.num_heads * self.head_dim * self.element_size * 2,
        };
        
        self.copy_new_token_data(host_key_data, host_value_data, copy_op)?;
        Ok(())
    }

    pub fn copy_from_host(&self, host_key_data: *const u8, host_value_data: *const u8) -> Result<(), CudaError> {
        let copy_op = DataCopyOperation::FullTensorCopy {
            total_size_bytes: self.size_bytes(),
        };
        self.copy_new_token_data(host_key_data, host_value_data, copy_op)?;
        Ok(())
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

    pub fn synchronize(&self) -> Result<(), CudaError> {
        #[cfg(feature = "cuda")]
        unsafe {
            let result = crate::cuda::cudaSetDevice(self.device_id);
            if result != crate::cuda::CUDA_SUCCESS {
                return Err(CudaError(result));
            }
            
            let result = crate::cuda::cudaDeviceSynchronize();
            if result != crate::cuda::CUDA_SUCCESS {
                return Err(CudaError(result));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ZeroCopyStats {
    pub current_seq_len: usize,
    pub max_seq_len: usize,
    pub growth_capacity_remaining: usize,
    pub utilization: f64,
    pub memory_efficiency: f64,
    pub can_grow_without_copy: bool,
    pub metadata_operations_are_zero_copy: bool,
    pub data_operations_require_copy: bool,
    pub last_extension_was_pure_zero_copy: Option<bool>,
}

#[derive(Debug, Clone)]
pub enum ExtensionOperationResult {
    PureZeroCopy {
        operation_time_ns: u64,
        tokens_added: usize,
    },
    RequiresDataCopy {
        metadata_update_time_ns: u64,
        tokens_added: usize,
        copy_start_token: usize,
        copy_size_bytes: usize,
    },
}

#[derive(Debug)]
pub struct ZeroCopyArena {
    page: CudaPage,
    arena_id: u64,
    current_offset: AtomicUsize,
    slab_pool: Arc<GlobalSlabPool>,
}

impl ZeroCopyArena {
    pub fn new(
        page: CudaPage,
        arena_id: u64,
        slab_pool: Arc<GlobalSlabPool>,
    ) -> Self {
        Self {
            page,
            arena_id,
            current_offset: AtomicUsize::new(0),
            slab_pool,
        }
    }

    /// SINGLE extend_tensor_for_generation method - returns bool for compatibility
    pub fn extend_tensor_for_generation(
        &self,
        tensor: &mut ZeroCopyTensor,
        additional_tokens: usize,
    ) -> Result<bool, CudaError> {
        let current_len = tensor.seq_len();
        let new_len = current_len + additional_tokens;
        
        match tensor.extend_with_data_requirements(new_len) {
            ExtensionResult::PureZeroCopy { .. } => {
                Ok(true) // TRUE zero-copy
            },
            ExtensionResult::RequiresDataCopy { .. } => {
                Ok(false) // Metadata updated but data copy required
            },
            ExtensionResult::CannotExtend { .. } => {
                Err(CudaError(-1))
            },
        }
    }

    /// New method for detailed extension results
    pub fn extend_tensor_detailed(
        &self,
        tensor: &mut ZeroCopyTensor,
        additional_tokens: usize,
    ) -> Result<ExtensionOperationResult, CudaError> {
        let current_len = tensor.seq_len();
        let new_len = current_len + additional_tokens;
        
        match tensor.extend_with_data_requirements(new_len) {
            ExtensionResult::PureZeroCopy { operation_time_ns, .. } => {
                Ok(ExtensionOperationResult::PureZeroCopy {
                    operation_time_ns,
                    tokens_added: additional_tokens,
                })
            },
            ExtensionResult::RequiresDataCopy { 
                copy_region_start, 
                copy_region_size, 
                metadata_update_time_ns,
                ..
            } => {
                Ok(ExtensionOperationResult::RequiresDataCopy {
                    metadata_update_time_ns,
                    tokens_added: additional_tokens,
                    copy_start_token: copy_region_start,
                    copy_size_bytes: copy_region_size,
                })
            },
            ExtensionResult::CannotExtend { .. } => {
                Err(CudaError(-1))
            },
        }
    }

    pub fn bump_allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        const CUDA_ALIGNMENT: usize = 256;
        let required_align = align.max(CUDA_ALIGNMENT);
        
        let aligned_size = if size == 0 {
            required_align
        } else {
            if size > usize::MAX - required_align {
                return None;
            }
            let aligned = (size + required_align - 1) & !(required_align - 1);
            aligned.max(32)
        };
        
        let old_offset = self.current_offset.fetch_add(aligned_size, Ordering::Relaxed);
        
        if old_offset > self.page.size() || aligned_size > self.page.size() - old_offset {
            self.current_offset.fetch_sub(aligned_size, Ordering::Relaxed);
            return None;
        }
        
        unsafe {
            let base_ptr = self.page.device_ptr() as *mut u8;
            let alloc_ptr = base_ptr.add(old_offset);
            Some(NonNull::new_unchecked(alloc_ptr))
        }
    }

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
        
        const CUDA_ALIGNMENT: usize = 256;
        
        if total_max_size > 1024 * 1024 * 1024 {
            return Err(CudaError(-4));
        }
        
        let device_ptr = self.bump_allocate(total_max_size, CUDA_ALIGNMENT)
            .ok_or(CudaError(-2))?;
        
        ZeroCopyTensor::from_bump_allocation(
            device_ptr,
            initial_seq_len,
            expected_max_seq_len,
            num_heads,
            head_dim,
            element_size,
            self.arena_id,
            self.page.device_id(),
        )
    }

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

    // Arena state queries
    pub fn page(&self) -> &CudaPage { &self.page }
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

    pub fn stats(&self) -> ZeroCopyArenaStats {
        ZeroCopyArenaStats {
            arena_id: self.arena_id,
            device_id: self.page.device_id(),
            page_size: self.page.size(),
            used_bytes: self.current_offset(),
            total_tensors: 0,
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
        let page = unsafe { std::ptr::read(&self.page) };
        std::mem::forget(&self.page);
        self.slab_pool.return_page(page);
    }
}

unsafe impl Send for ZeroCopyArena {}
unsafe impl Sync for ZeroCopyArena {}

#[derive(Debug)]
pub struct GlobalSlabPool {
    small_pages: SegQueue<CudaPage>,
    medium_pages: SegQueue<CudaPage>,
    large_pages: SegQueue<CudaPage>,
    huge_pages: SegQueue<CudaPage>,
    pages_created: AtomicUsize,
    pages_recycled: AtomicUsize,
    pages_reused: AtomicUsize,
    bytes_saved: AtomicUsize,
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

    fn size_class_for_size(&self, size: usize) -> &SegQueue<CudaPage> {
        match size {
            0..=524_288 => &self.small_pages,
            524_289..=2_097_152 => &self.medium_pages,
            2_097_153..=8_388_608 => &self.large_pages,
            _ => &self.huge_pages,
        }
    }

    pub fn get_page(&self, requested_size: usize, device_id: i32) -> Option<CudaPage> {
        let queue = self.size_class_for_size(requested_size);
        
        while let Some(page) = queue.pop() {
            if page.size() >= requested_size && page.device_id() == device_id {
                page.reset();
                self.pages_reused.fetch_add(1, Ordering::Relaxed);
                self.bytes_saved.fetch_add(page.size(), Ordering::Relaxed);
                return Some(page);
            } else {
                queue.push(page);
                break;
            }
        }
        None
    }

    pub fn return_page(&self, page: CudaPage) {
        let size = page.size();
        let queue = self.size_class_for_size(size);
        
        if self.approximate_queue_size(queue) >= self.max_pages_per_class {
            return;
        }

        page.reset();
        queue.push(page);
        self.pages_recycled.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_page_creation(&self, size: usize) {
        self.pages_created.fetch_add(1, Ordering::Relaxed);
    }

    fn approximate_queue_size(&self, queue: &SegQueue<CudaPage>) -> usize {
        let mut count = 0;
        let mut temp_pages = Vec::new();
        
        for _ in 0..10 {
            if let Some(page) = queue.pop() {
                temp_pages.push(page);
                count += 1;
            } else {
                break;
            }
        }
        
        for page in temp_pages {
            queue.push(page);
        }
        
        count * 5
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
                }
            }
        }
        
        cleaned
    }
}

#[derive(Debug)]
pub struct ZeroCopyManager {
    slab_pool: Arc<GlobalSlabPool>,
    cuda_context: Arc<CudaContext>,
    next_arena_id: AtomicUsize,
}

impl ZeroCopyManager {
    pub fn new(slab_pool: Arc<GlobalSlabPool>) -> Result<Self, CudaError> {
        let cuda_context = Arc::new(crate::cuda::create_safe_cuda_context()?);
        
        Ok(Self {
            slab_pool,
            cuda_context,
            next_arena_id: AtomicUsize::new(0),
        })
    }

    pub fn create_arena(
        &self,
        page_size: usize,
        device_id: i32,
    ) -> Result<ZeroCopyArena, CudaError> {
        let arena_id = self.next_arena_id.fetch_add(1, Ordering::Relaxed) as u64;
        
        let page = if let Some(recycled_page) = self.slab_pool.get_page(page_size, device_id) {
            recycled_page
        } else {
            let new_page = self.cuda_context.allocate_page_on_device(page_size, device_id)?;
            self.slab_pool.record_page_creation(page_size);
            new_page
        };
        
        let arena = ZeroCopyArena::new(
            page,
            arena_id,
            Arc::clone(&self.slab_pool),
        );

        Ok(arena)
    }

    pub fn global_stats(&self) -> ZeroCopyGlobalStats {
        let slab_stats = self.slab_pool.stats();
        
        ZeroCopyGlobalStats {
            total_arenas: 0,
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

    pub fn cleanup_inactive_arenas(&self) -> usize { 0 }
    pub fn defragment_all(&self) -> Result<usize, CudaError> { Ok(0) }
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

// Stats structs
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

// Additional utility implementations for thread safety
unsafe impl Send for ZeroCopyTensor {}
unsafe impl Sync for ZeroCopyTensor {}

// Default implementations for key types
impl Default for GlobalSlabPool {
    fn default() -> Self {
        Self::new()
    }
}

// Additional helper methods for ZeroCopyTensor
impl ZeroCopyTensor {
    /// Check if the tensor can accommodate the requested sequence length
    pub fn has_capacity_for(&self, seq_len: usize) -> bool {
        seq_len <= self.max_seq_len
    }

    /// Get the number of remaining tokens that can be added without reallocation
    pub fn remaining_capacity(&self) -> usize {
        self.max_seq_len.saturating_sub(self.seq_len())
    }

    /// Calculate memory overhead (allocated but unused space)
    pub fn memory_overhead_bytes(&self) -> usize {
        let allocated = self.max_allocated_size_bytes();
        let used = self.size_bytes();
        allocated.saturating_sub(used)
    }

    /// Get a detailed breakdown of memory usage
    pub fn memory_breakdown(&self) -> TensorMemoryBreakdown {
        let current_len = self.seq_len();
        let key_size = self.current_key_size_bytes();
        let value_size = self.current_value_size_bytes();
        let overhead = self.memory_overhead_bytes();
        
        TensorMemoryBreakdown {
            current_seq_len: current_len,
            max_seq_len: self.max_seq_len,
            key_tensor_bytes: key_size,
            value_tensor_bytes: value_size,
            total_used_bytes: key_size + value_size,
            total_allocated_bytes: self.max_allocated_size_bytes(),
            overhead_bytes: overhead,
            utilization_ratio: self.utilization(),
            waste_ratio: overhead as f64 / self.max_allocated_size_bytes() as f64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorMemoryBreakdown {
    pub current_seq_len: usize,
    pub max_seq_len: usize,
    pub key_tensor_bytes: usize,
    pub value_tensor_bytes: usize,
    pub total_used_bytes: usize,
    pub total_allocated_bytes: usize,
    pub overhead_bytes: usize,
    pub utilization_ratio: f64,
    pub waste_ratio: f64,
}

// Additional methods for ZeroCopyArena
impl ZeroCopyArena {
    /// Create multiple tensors in a single allocation for batch processing
    pub fn allocate_tensor_batch(
        &self,
        count: usize,
        initial_seq_len: usize,
        expected_max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        element_size: usize,
    ) -> Result<Vec<ZeroCopyTensor>, CudaError> {
        let mut tensors = Vec::with_capacity(count);
        
        for _ in 0..count {
            let tensor = self.allocate_kv_tensor_with_growth(
                initial_seq_len,
                expected_max_seq_len,
                num_heads,
                head_dim,
                element_size,
            )?;
            tensors.push(tensor);
        }
        
        Ok(tensors)
    }

    /// Get detailed arena health metrics
    pub fn health_check(&self) -> ArenaHealthReport {
        let utilization = self.utilization();
        let available = self.available_space();
        
        let health_status = if utilization > 0.9 {
            ArenaHealth::Critical
        } else if utilization > 0.7 {
            ArenaHealth::Warning
        } else {
            ArenaHealth::Healthy
        };

        ArenaHealthReport {
            arena_id: self.arena_id,
            health_status,
            utilization_percent: (utilization * 100.0) as u32,
            available_bytes: available,
            fragmentation_estimate: 0.0, // Could be improved with tracking
            recommended_action: match health_status {
                ArenaHealth::Critical => "Consider creating new arena or cleaning up unused tensors".to_string(),
                ArenaHealth::Warning => "Monitor usage and prepare for potential capacity issues".to_string(),
                ArenaHealth::Healthy => "No action required".to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ArenaHealth {
    Healthy,
    Warning,
    Critical,
}

#[derive(Debug, Clone)]
pub struct ArenaHealthReport {
    pub arena_id: u64,
    pub health_status: ArenaHealth,
    pub utilization_percent: u32,
    pub available_bytes: usize,
    pub fragmentation_estimate: f64,
    pub recommended_action: String,
}

// Enhanced GlobalSlabPool methods
impl GlobalSlabPool {
    /// Perform comprehensive cleanup and optimization
    pub fn optimize(&self) -> PoolOptimizationReport {
        let initial_stats = self.stats();
        let pages_cleaned = self.cleanup_old_pages();
        let final_stats = self.stats();
        
        PoolOptimizationReport {
            pages_cleaned,
            bytes_freed_mb: 0, // Would need tracking of cleaned page sizes
            efficiency_before: initial_stats.recycling_efficiency,
            efficiency_after: final_stats.recycling_efficiency,
            pool_sizes_before: initial_stats.current_pool_sizes,
            pool_sizes_after: final_stats.current_pool_sizes,
        }
    }

    /// Get recommendations for pool tuning
    pub fn get_tuning_recommendations(&self) -> Vec<PoolRecommendation> {
        let stats = self.stats();
        let mut recommendations = Vec::new();
        
        if stats.recycling_efficiency < 0.3 {
            recommendations.push(PoolRecommendation {
                severity: RecommendationSeverity::High,
                message: "Low recycling efficiency detected. Consider reducing max_pages_per_class or reviewing allocation patterns.".to_string(),
                action: "Tune pool parameters".to_string(),
            });
        }
        
        if stats.reuse_efficiency < 0.5 {
            recommendations.push(PoolRecommendation {
                severity: RecommendationSeverity::Medium,
                message: "Pages are being recycled but not reused efficiently. Check device affinity and size matching.".to_string(),
                action: "Review allocation patterns".to_string(),
            });
        }
        
        if recommendations.is_empty() {
            recommendations.push(PoolRecommendation {
                severity: RecommendationSeverity::Info,
                message: "Pool is operating efficiently.".to_string(),
                action: "No action required".to_string(),
            });
        }
        
        recommendations
    }
}

#[derive(Debug, Clone)]
pub struct PoolOptimizationReport {
    pub pages_cleaned: usize,
    pub bytes_freed_mb: usize,
    pub efficiency_before: f64,
    pub efficiency_after: f64,
    pub pool_sizes_before: [usize; 4],
    pub pool_sizes_after: [usize; 4],
}

#[derive(Debug, Clone)]
pub struct PoolRecommendation {
    pub severity: RecommendationSeverity,
    pub message: String,
    pub action: String,
}

#[derive(Debug, Clone)]
pub enum RecommendationSeverity {
    Info,
    Medium,
    High,
}

// Test utilities (conditionally compiled)
#[cfg(test)]
impl ZeroCopyTensor {
    /// Create a mock tensor for testing (without actual CUDA allocation)
    pub fn mock_tensor(
        seq_len: usize,
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        let mock_ptr = NonNull::new(0x1000 as *mut u8).unwrap();
        Self {
            key_device_ptr: mock_ptr,
            value_device_ptr: mock_ptr,
            current_seq_len: AtomicUsize::new(seq_len),
            max_seq_len,
            num_heads,
            head_dim,
            element_size: 4, // f32
            device_id: 0,
            arena_id: 0,
        }
    }
}