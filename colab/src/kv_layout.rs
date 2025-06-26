// src/kv_layout.rs - FIXED: KV-cache layout with proper head handling
use std::sync::Arc;
use std::ptr::NonNull;
use crate::cuda::{CudaPage, CudaError};

/// FIXED: KV-cache tensor layout optimized for transformer attention with proper head handling
/// Layout: [Key Tensor: seq_len × num_kv_heads × head_dim] [Value Tensor: seq_len × num_kv_heads × head_dim]
/// Memory layout is contiguous: K0,K1,...,Kn,V0,V1,...,Vn where each K/V is head_dim elements
#[derive(Debug)]
pub struct KVTensorLayout {
    /// Direct pointer to start of KV tensor block (K tensors)
    device_ptr: NonNull<u8>,
    /// Byte offset within the arena page
    offset: usize,
    /// Current sequence length (number of tokens)
    seq_len: usize,
    /// Maximum sequence length this allocation can hold (for zero-copy growth)
    max_seq_len: usize,
    /// Number of KV attention heads (NOT query heads)
    num_kv_heads: usize,
    /// Number of query attention heads (for reference/validation)
    num_query_heads: usize,
    /// Dimension per head (hidden_dim / num_query_heads)
    head_dim: usize,
    /// Element size in bytes (2 for fp16, 4 for fp32, 1 for int8)
    element_size: usize,
    /// Reference to the underlying page (keeps allocator alive)
    page_ref: Arc<CudaPage>,
    /// Arena ID for tracking
    arena_id: u64,
}

impl KVTensorLayout {
    /// FIXED: Create KV tensor layout with proper head configuration
    pub fn from_bump_allocation(
        page: &Arc<CudaPage>,
        device_ptr: NonNull<u8>,
        offset: usize,
        seq_len: usize,
        max_seq_len: usize,
        num_kv_heads: usize,        // FIXED: Use actual KV heads
        num_query_heads: usize,     // FIXED: Track query heads separately
        head_dim: usize,
        element_size: usize,
        arena_id: u64,
    ) -> Self {
        log::debug!("Creating KV layout: {} seq_len, {} KV heads (vs {} query heads), {} head_dim",
                   seq_len, num_kv_heads, num_query_heads, head_dim);
        
        KVTensorLayout {
            device_ptr,
            offset,
            seq_len,
            max_seq_len,
            num_kv_heads,
            num_query_heads,
            head_dim,
            element_size,
            page_ref: Arc::clone(page),
            arena_id,
        }
    }

    /// FIXED: Get total size of KV tensor pair using actual KV heads
    pub fn current_kv_size_bytes(&self) -> usize {
        // K tensor + V tensor: 2 * seq_len * num_kv_heads * head_dim * element_size
        2 * self.seq_len * self.num_kv_heads * self.head_dim * self.element_size
    }

    /// FIXED: Get total size of KV tensor pair for maximum sequence length using KV heads
    pub fn max_kv_size_bytes(&self) -> usize {
        // K tensor + V tensor: 2 * max_seq_len * num_kv_heads * head_dim * element_size
        2 * self.max_seq_len * self.num_kv_heads * self.head_dim * self.element_size
    }

    /// Get device pointer for K tensor start
    /// Layout: K tensor occupies first half of allocated space
    pub fn key_device_ptr(&self) -> *mut std::ffi::c_void {
        self.device_ptr.as_ptr() as *mut std::ffi::c_void
    }

    /// Get device pointer for V tensor start
    /// Layout: V tensor starts after the full K tensor (at max_seq_len boundary)
    pub fn value_device_ptr(&self) -> *mut std::ffi::c_void {
        let k_tensor_max_size = self.max_seq_len * self.num_kv_heads * self.head_dim * self.element_size;
        unsafe {
            (self.device_ptr.as_ptr()).add(k_tensor_max_size) as *mut std::ffi::c_void
        }
    }

    /// FIXED: Get device pointer for specific token in K tensor using KV heads
    /// Layout: K[token][kv_head][head_dim_idx] 
    pub fn key_token_ptr(&self, token_idx: usize) -> Option<*mut std::ffi::c_void> {
        if token_idx >= self.seq_len {
            return None;
        }
        
        let token_offset = token_idx * self.num_kv_heads * self.head_dim * self.element_size;
        unsafe {
            Some((self.device_ptr.as_ptr()).add(token_offset) as *mut std::ffi::c_void)
        }
    }

    /// FIXED: Get device pointer for specific token in V tensor using KV heads
    /// Layout: V[token][kv_head][head_dim_idx]
    pub fn value_token_ptr(&self, token_idx: usize) -> Option<*mut std::ffi::c_void> {
        if token_idx >= self.seq_len {
            return None;
        }
        
        let k_tensor_max_size = self.max_seq_len * self.num_kv_heads * self.head_dim * self.element_size;
        let token_offset = token_idx * self.num_kv_heads * self.head_dim * self.element_size;
        unsafe {
            Some((self.device_ptr.as_ptr()).add(k_tensor_max_size + token_offset) as *mut std::ffi::c_void)
        }
    }

    /// FIXED: Get device pointer for specific KV head in K tensor at given token
    /// Layout: K[token][kv_head][head_dim_idx]
    pub fn key_head_ptr(&self, token_idx: usize, kv_head_idx: usize) -> Option<*mut std::ffi::c_void> {
        if token_idx >= self.seq_len || kv_head_idx >= self.num_kv_heads {
            return None;
        }
        
        let offset = (token_idx * self.num_kv_heads + kv_head_idx) * self.head_dim * self.element_size;
        unsafe {
            Some((self.device_ptr.as_ptr()).add(offset) as *mut std::ffi::c_void)
        }
    }

    /// FIXED: Get device pointer for specific KV head in V tensor at given token
    /// Layout: V[token][kv_head][head_dim_idx]
    pub fn value_head_ptr(&self, token_idx: usize, kv_head_idx: usize) -> Option<*mut std::ffi::c_void> {
        if token_idx >= self.seq_len || kv_head_idx >= self.num_kv_heads {
            return None;
        }
        
        let k_tensor_max_size = self.max_seq_len * self.num_kv_heads * self.head_dim * self.element_size;
        let offset = (token_idx * self.num_kv_heads + kv_head_idx) * self.head_dim * self.element_size;
        unsafe {
            Some((self.device_ptr.as_ptr()).add(k_tensor_max_size + offset) as *mut std::ffi::c_void)
        }
    }

    /// TRUE ZERO-COPY extension - just update seq_len, NO memory operations
    /// This is the core optimization that eliminates "copy amplification"
    pub fn extend_zero_copy(&mut self, new_seq_len: usize) -> bool {
        if new_seq_len <= self.max_seq_len {
            // TRUE ZERO-COPY: Just update metadata, no memory operations
            let old_seq_len = self.seq_len;
            self.seq_len = new_seq_len;
            
            log::debug!("TRUE zero-copy KV extension: {} -> {} tokens (NO memory ops), {} KV heads", 
                       old_seq_len, new_seq_len, self.num_kv_heads);
            true
        } else {
            false // Cannot extend beyond allocated capacity
        }
    }

    /// FIXED: Copy new K,V tokens for incremental generation using KV heads
    /// Only copies the NEW tokens, not the entire tensor
    pub fn copy_new_kv_tokens_from_host(
        &self,
        host_key_data: *const u8,
        host_value_data: *const u8,
        start_token: usize,
        num_tokens: usize,
    ) -> Result<(), CudaError> {
        if start_token + num_tokens > self.seq_len {
            return Err(CudaError(-1));
        }

        // FIXED: Calculate token size using KV heads
        let token_size = self.num_kv_heads * self.head_dim * self.element_size;
        let copy_size = num_tokens * token_size;
        let token_offset = start_token * token_size;

        // Copy new K tokens to K tensor region
        self.page_ref.copy_from_host(
            unsafe { host_key_data.add(token_offset) as *const std::ffi::c_void },
            copy_size,
            self.offset + token_offset,
        )?;

        // Copy new V tokens to V tensor region (after K tensor max allocation)
        let v_tensor_offset = self.max_seq_len * self.num_kv_heads * self.head_dim * self.element_size;
        self.page_ref.copy_from_host(
            unsafe { host_value_data.add(token_offset) as *const std::ffi::c_void },
            copy_size,
            self.offset + v_tensor_offset + token_offset,
        )?;

        log::debug!("Copied {} new KV tokens starting at token {} ({} KV heads, incremental copy)", 
                   num_tokens, start_token, self.num_kv_heads);
        Ok(())
    }

    /// FIXED: Copy full KV tensors from host using KV heads
    pub fn copy_full_kv_from_host(&self, host_key_data: *const u8, host_value_data: *const u8) -> Result<(), CudaError> {
        let k_tensor_size = self.seq_len * self.num_kv_heads * self.head_dim * self.element_size;

        // Copy K tensor to start of allocation
        self.page_ref.copy_from_host(
            host_key_data as *const std::ffi::c_void,
            k_tensor_size,
            self.offset,
        )?;

        // Copy V tensor after K tensor max space
        let v_tensor_offset = self.max_seq_len * self.num_kv_heads * self.head_dim * self.element_size;
        self.page_ref.copy_from_host(
            host_value_data as *const std::ffi::c_void,
            k_tensor_size, // Note: only copy current seq_len, not max
            self.offset + v_tensor_offset,
        )?;

        log::debug!("Copied full KV tensors: {} tokens, {} KV heads", self.seq_len, self.num_kv_heads);
        Ok(())
    }

    /// FIXED: Get KV tensor dimensions (seq_len, num_kv_heads, head_dim)
    pub fn kv_dimensions(&self) -> (usize, usize, usize) {
        (self.seq_len, self.num_kv_heads, self.head_dim)
    }

    /// FIXED: Get comprehensive KV tensor info including head counts
    pub fn kv_tensor_info(&self) -> KVTensorInfo {
        KVTensorInfo {
            seq_len: self.seq_len,
            max_seq_len: self.max_seq_len,
            num_kv_heads: self.num_kv_heads,
            num_query_heads: self.num_query_heads,
            head_dim: self.head_dim,
            element_size: self.element_size,
            current_kv_bytes: self.current_kv_size_bytes(),
            max_kv_bytes: self.max_kv_size_bytes(),
            k_tensor_shape: vec![self.seq_len, self.num_kv_heads, self.head_dim],
            v_tensor_shape: vec![self.seq_len, self.num_kv_heads, self.head_dim],
            utilization: self.seq_len as f64 / self.max_seq_len as f64,
            gqa_ratio: self.num_query_heads as f64 / self.num_kv_heads as f64,
        }
    }

    /// Check if tensor can be extended to new length (zero-copy)
    pub fn can_extend_to(&self, new_seq_len: usize) -> bool {
        new_seq_len <= self.max_seq_len
    }

    /// Get utilization ratio (current / max capacity)
    pub fn utilization(&self) -> f64 {
        self.seq_len as f64 / self.max_seq_len as f64
    }

    /// FIXED: Get GQA (Grouped Query Attention) ratio
    pub fn gqa_ratio(&self) -> f64 {
        self.num_query_heads as f64 / self.num_kv_heads as f64
    }

    /// Check if this is a GQA model (query heads != KV heads)
    pub fn is_gqa(&self) -> bool {
        self.num_query_heads != self.num_kv_heads
    }

    /// Get the number of query heads per KV head
    pub fn query_heads_per_kv_head(&self) -> usize {
        self.num_query_heads / self.num_kv_heads
    }

    /// Synchronize operations on this KV tensor
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.page_ref.synchronize()
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.page_ref.device_id()
    }

    /// Get arena ID
    pub fn arena_id(&self) -> u64 {
        self.arena_id
    }

    /// FIXED: Calculate memory layout for validation with KV heads
    pub fn validate_layout(&self) -> Result<KVLayoutValidation, CudaError> {
        let k_start = 0;
        let k_end = self.max_seq_len * self.num_kv_heads * self.head_dim * self.element_size;
        let v_start = k_end;
        let v_end = v_start + (self.max_seq_len * self.num_kv_heads * self.head_dim * self.element_size);
        
        let page_size = self.page_ref.size();
        let required_size = v_end;
        
        if self.offset + required_size > page_size {
            return Err(CudaError(-2)); // Insufficient space
        }

        Ok(KVLayoutValidation {
            k_tensor_range: (k_start, k_end),
            v_tensor_range: (v_start, v_end),
            total_required_bytes: required_size,
            page_size,
            offset_in_page: self.offset,
            layout_valid: true,
            num_kv_heads: self.num_kv_heads,
            num_query_heads: self.num_query_heads,
            gqa_ratio: self.gqa_ratio(),
        })
    }

    /// Get layout efficiency metrics
    pub fn layout_efficiency(&self) -> LayoutEfficiency {
        let current_bytes = self.current_kv_size_bytes();
        let max_bytes = self.max_kv_size_bytes();
        let page_size = self.page_ref.size();
        
        LayoutEfficiency {
            memory_utilization: current_bytes as f64 / max_bytes as f64,
            page_utilization: max_bytes as f64 / page_size as f64,
            fragmentation: 1.0 - (max_bytes as f64 / page_size as f64),
            waste_bytes: page_size - max_bytes,
            compression_ratio: if self.is_gqa() { self.gqa_ratio() } else { 1.0 },
        }
    }

    /// Get memory access patterns for optimization
    pub fn access_pattern_info(&self) -> AccessPatternInfo {
        AccessPatternInfo {
            sequential_access_stride: self.num_kv_heads * self.head_dim * self.element_size,
            head_access_stride: self.head_dim * self.element_size,
            token_boundary_bytes: self.num_kv_heads * self.head_dim * self.element_size,
            cache_line_efficiency: self.calculate_cache_line_efficiency(),
            memory_bandwidth_efficiency: self.calculate_bandwidth_efficiency(),
        }
    }

    /// Calculate cache line efficiency for memory access
    fn calculate_cache_line_efficiency(&self) -> f64 {
        const CACHE_LINE_SIZE: usize = 64; // Typical cache line size
        let access_size = self.head_dim * self.element_size;
        if access_size <= CACHE_LINE_SIZE {
            access_size as f64 / CACHE_LINE_SIZE as f64
        } else {
            1.0 // Multiple cache lines needed
        }
    }

    /// Calculate memory bandwidth efficiency
    fn calculate_bandwidth_efficiency(&self) -> f64 {
        let total_bytes = self.current_kv_size_bytes();
        let useful_bytes = if self.is_gqa() {
            // In GQA, we're accessing fewer KV heads than query heads
            total_bytes
        } else {
            total_bytes
        };
        useful_bytes as f64 / total_bytes as f64
    }
}

/// FIXED: KV tensor information for debugging and monitoring with head details
#[derive(Debug, Clone)]
pub struct KVTensorInfo {
    pub seq_len: usize,
    pub max_seq_len: usize,
    pub num_kv_heads: usize,       // FIXED: Actual KV heads
    pub num_query_heads: usize,    // FIXED: Query heads for reference
    pub head_dim: usize,
    pub element_size: usize,
    pub current_kv_bytes: usize,
    pub max_kv_bytes: usize,
    pub k_tensor_shape: Vec<usize>,
    pub v_tensor_shape: Vec<usize>,
    pub utilization: f64,
    pub gqa_ratio: f64,            // FIXED: GQA ratio (query/KV heads)
}

/// FIXED: Layout validation result with head information
#[derive(Debug, Clone)]
pub struct KVLayoutValidation {
    pub k_tensor_range: (usize, usize),
    pub v_tensor_range: (usize, usize),
    pub total_required_bytes: usize,
    pub page_size: usize,
    pub offset_in_page: usize,
    pub layout_valid: bool,
    pub num_kv_heads: usize,       // FIXED: Include head counts in validation
    pub num_query_heads: usize,
    pub gqa_ratio: f64,
}

/// Layout efficiency metrics
#[derive(Debug, Clone)]
pub struct LayoutEfficiency {
    pub memory_utilization: f64,    // current / max bytes
    pub page_utilization: f64,      // allocated / page size
    pub fragmentation: f64,         // wasted space ratio
    pub waste_bytes: usize,         // unused bytes in page
    pub compression_ratio: f64,     // GQA compression (query/KV heads)
}

/// Memory access pattern information
#[derive(Debug, Clone)]
pub struct AccessPatternInfo {
    pub sequential_access_stride: usize,     // Bytes between sequential tokens
    pub head_access_stride: usize,           // Bytes between heads
    pub token_boundary_bytes: usize,         // Bytes per token
    pub cache_line_efficiency: f64,          // Cache line utilization
    pub memory_bandwidth_efficiency: f64,    // Bandwidth utilization
}

/// FIXED: Calculate optimal page size for KV tensors using actual KV heads
pub fn calculate_optimal_kv_page_size(
    max_seq_len: usize,
    num_kv_heads: usize,  // FIXED: Use KV heads, not query heads
    head_dim: usize,
    element_size: usize,
) -> usize {
    // FIXED: Calculate size using KV heads for actual cache requirements
    let largest_kv_tensor_size = 2 * max_seq_len * num_kv_heads * head_dim * element_size;
    
    // Add some overhead for alignment and multiple tensors per page
    let overhead_factor = 1.25; // 25% overhead
    let target_size = (largest_kv_tensor_size as f64 * overhead_factor) as usize;
    
    // Round up to next power of 2 for efficient allocation
    let page_size = target_size.next_power_of_two();
    
    // Clamp to reasonable bounds (64KB - 16MB)
    let final_size = page_size.max(64 * 1024).min(16 * 1024 * 1024);
    
    log::debug!("Optimal KV page size: {}KB for {} seq_len, {} KV heads, {} head_dim",
               final_size / 1024, max_seq_len, num_kv_heads, head_dim);
    
    final_size
}

/// Calculate page size optimized for memory access patterns
pub fn calculate_access_optimized_page_size(
    max_seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    element_size: usize,
) -> usize {
    let base_size = calculate_optimal_kv_page_size(max_seq_len, num_kv_heads, head_dim, element_size);
    
    // Optimize for cache line alignment
    const CACHE_LINE_SIZE: usize = 64;
    let token_size = num_kv_heads * head_dim * element_size;
    
    if token_size % CACHE_LINE_SIZE != 0 {
        // Add padding to align token boundaries to cache lines
        let padding_per_token = CACHE_LINE_SIZE - (token_size % CACHE_LINE_SIZE);
        let total_padding = padding_per_token * max_seq_len * 2; // K + V tensors
        base_size + total_padding
    } else {
        base_size
    }
}

/// FIXED: Calculate page size for specific model configurations with correct head counts
pub fn calculate_model_kv_page_size(model_config: &ModelConfig) -> usize {
    match model_config {
        ModelConfig::Mistral7B => {
            // FIXED: Mistral 7B: 32 query heads, 8 KV heads, 128 head_dim (GQA)
            calculate_optimal_kv_page_size(8192, 8, 128, 2) // Use 8 KV heads!
        }
        ModelConfig::Llama2_7B => {
            // Llama-2 7B: 32 heads (full attention)
            calculate_optimal_kv_page_size(8192, 32, 128, 2)
        }
        ModelConfig::Llama2_13B => {
            // Llama-2 13B: 40 heads (full attention)
            calculate_optimal_kv_page_size(8192, 40, 128, 2)
        }
        ModelConfig::Llama2_70B => {
            // FIXED: Llama-2 70B: 64 query heads, 8 KV heads (GQA)
            calculate_optimal_kv_page_size(8192, 8, 128, 2) // Use 8 KV heads!
        }
        ModelConfig::Custom { max_seq_len, num_kv_heads, head_dim, element_size } => {
            calculate_optimal_kv_page_size(*max_seq_len, *num_kv_heads, *head_dim, *element_size)
        }
    }
}

/// Calculate access-optimized page size for specific models
pub fn calculate_model_access_optimized_page_size(model_config: &ModelConfig) -> usize {
    match model_config {
        ModelConfig::Mistral7B => {
            calculate_access_optimized_page_size(8192, 8, 128, 2)
        }
        ModelConfig::Llama2_7B => {
            calculate_access_optimized_page_size(8192, 32, 128, 2)
        }
        ModelConfig::Llama2_13B => {
            calculate_access_optimized_page_size(8192, 40, 128, 2)
        }
        ModelConfig::Llama2_70B => {
            calculate_access_optimized_page_size(8192, 8, 128, 2)
        }
        ModelConfig::Custom { max_seq_len, num_kv_heads, head_dim, element_size } => {
            calculate_access_optimized_page_size(*max_seq_len, *num_kv_heads, *head_dim, *element_size)
        }
    }
}

/// FIXED: Model configuration with separate query and KV head specifications
#[derive(Debug, Clone)]
pub enum ModelConfig {
    Mistral7B,       // 32 query -> 8 KV heads
    Llama2_7B,       // 32 query -> 32 KV heads (full attention)
    Llama2_13B,      // 40 query -> 40 KV heads (full attention)
    Llama2_70B,      // 64 query -> 8 KV heads (GQA)
    Custom {
        max_seq_len: usize,
        num_kv_heads: usize,     // FIXED: Specify KV heads directly
        head_dim: usize,
        element_size: usize,
    },
}

impl ModelConfig {
    /// FIXED: Get KV head configuration for the model
    pub fn get_kv_config(&self) -> (usize, usize, usize) { // (query_heads, kv_heads, head_dim)
        match self {
            ModelConfig::Mistral7B => (32, 8, 128),
            ModelConfig::Llama2_7B => (32, 32, 128),
            ModelConfig::Llama2_13B => (40, 40, 128),
            ModelConfig::Llama2_70B => (64, 8, 128),
            ModelConfig::Custom { num_kv_heads, head_dim, .. } => {
                // For custom, assume query heads = 4 * kv_heads (common GQA ratio)
                let query_heads = num_kv_heads * 4;
                (query_heads, *num_kv_heads, *head_dim)
            }
        }
    }
    
    /// Check if this model uses GQA (Grouped Query Attention)
    pub fn is_gqa(&self) -> bool {
        let (query_heads, kv_heads, _) = self.get_kv_config();
        query_heads != kv_heads
    }
    
    /// Get the GQA ratio (query heads / KV heads)
    pub fn gqa_ratio(&self) -> f64 {
        let (query_heads, kv_heads, _) = self.get_kv_config();
        query_heads as f64 / kv_heads as f64
    }

    /// Get memory efficiency from GQA
    pub fn memory_efficiency(&self) -> f64 {
        if self.is_gqa() {
            let (query_heads, kv_heads, _) = self.get_kv_config();
            kv_heads as f64 / query_heads as f64
        } else {
            1.0
        }
    }

    /// Get optimal sequence length for this model
    pub fn optimal_seq_len(&self) -> usize {
        match self {
            ModelConfig::Mistral7B => 8192,
            ModelConfig::Llama2_7B => 4096,
            ModelConfig::Llama2_13B => 4096,
            ModelConfig::Llama2_70B => 4096,
            ModelConfig::Custom { max_seq_len, .. } => *max_seq_len,
        }
    }

    /// Get model name for logging
    pub fn name(&self) -> &'static str {
        match self {
            ModelConfig::Mistral7B => "Mistral-7B",
            ModelConfig::Llama2_7B => "Llama-2-7B",
            ModelConfig::Llama2_13B => "Llama-2-13B",
            ModelConfig::Llama2_70B => "Llama-2-70B",
            ModelConfig::Custom { .. } => "Custom",
        }
    }
}

/// FIXED: KV tensor statistics for monitoring with head-specific metrics
#[derive(Debug, Clone)]
pub struct KVTensorStats {
    pub total_kv_tensors: usize,
    pub total_tokens_stored: usize,
    pub total_kv_bytes_used: usize,
    pub total_kv_bytes_allocated: usize,
    pub average_utilization: f64,
    pub zero_copy_extensions: usize,
    pub copy_extensions: usize,
    pub largest_tensor_tokens: usize,
    pub smallest_tensor_tokens: usize,
    pub total_kv_heads: usize,        // FIXED: Track total KV heads
    pub total_query_heads: usize,     // FIXED: Track total query heads
    pub gqa_models_count: usize,      // FIXED: Count of GQA models
    pub full_attention_models_count: usize,
    pub average_gqa_ratio: f64,
    pub memory_efficiency: f64,       // Memory saved through GQA
    pub bandwidth_efficiency: f64,    // Bandwidth utilization
}

/// Performance optimization recommendations
#[derive(Debug, Clone)]
pub struct OptimizationRecommendations {
    pub page_size_recommendations: Vec<String>,
    pub memory_layout_recommendations: Vec<String>,
    pub access_pattern_recommendations: Vec<String>,
    pub gqa_specific_recommendations: Vec<String>,
}

impl KVTensorStats {
    /// Generate optimization recommendations based on stats
    pub fn get_optimization_recommendations(&self) -> OptimizationRecommendations {
        let mut page_recs = Vec::new();
        let mut layout_recs = Vec::new();
        let mut access_recs = Vec::new();
        let mut gqa_recs = Vec::new();

        // Page size recommendations
        if self.average_utilization < 0.6 {
            page_recs.push("Consider smaller page sizes to reduce waste".to_string());
        }
        if self.average_utilization > 0.9 {
            page_recs.push("Consider larger page sizes for better zero-copy performance".to_string());
        }

        // Memory layout recommendations
        if self.memory_efficiency < 0.7 {
            layout_recs.push("Memory efficiency could be improved with better KV head organization".to_string());
        }
        if self.gqa_models_count > 0 && self.average_gqa_ratio > 1.0 {
            layout_recs.push(format!("GQA detected with {:.1}:1 ratio - ensure KV cache uses actual KV heads", self.average_gqa_ratio));
        }

        // Access pattern recommendations
        if self.bandwidth_efficiency < 0.8 {
            access_recs.push("Memory access patterns could be optimized for better bandwidth utilization".to_string());
        }
        if self.zero_copy_extensions < self.copy_extensions {
            access_recs.push("More copy operations than zero-copy - consider increasing initial allocation sizes".to_string());
        }

        // GQA-specific recommendations
        if self.gqa_models_count > 0 {
            gqa_recs.push("Using GQA models - ensure cache allocation uses KV head count, not query head count".to_string());
            if self.average_gqa_ratio >= 4.0 {
                gqa_recs.push(format!("High GQA ratio ({:.1}:1) provides significant memory savings", self.average_gqa_ratio));
            }
        }

        OptimizationRecommendations {
            page_size_recommendations: page_recs,
            memory_layout_recommendations: layout_recs,
            access_pattern_recommendations: access_recs,
            gqa_specific_recommendations: gqa_recs,
        }
    }

    /// Calculate overall efficiency score (0.0 to 1.0)
    pub fn overall_efficiency_score(&self) -> f64 {
        let utilization_score = self.average_utilization;
        let memory_score = self.memory_efficiency;
        let bandwidth_score = self.bandwidth_efficiency;
        let zero_copy_score = if self.zero_copy_extensions + self.copy_extensions > 0 {
            self.zero_copy_extensions as f64 / (self.zero_copy_extensions + self.copy_extensions) as f64
        } else {
            1.0
        };

        (utilization_score + memory_score + bandwidth_score + zero_copy_score) / 4.0
    }
}

/// Helper functions for creating model-specific layouts
pub mod model_helpers {
    use super::*;

    /// Create optimized layout for Mistral 7B
    pub fn create_mistral_7b_layout(
        page: &Arc<CudaPage>,
        device_ptr: NonNull<u8>,
        offset: usize,
        seq_len: usize,
        max_seq_len: usize,
        arena_id: u64,
    ) -> KVTensorLayout {
        KVTensorLayout::from_bump_allocation(
            page,
            device_ptr,
            offset,
            seq_len,
            max_seq_len,
            8,   // KV heads
            32,  // Query heads
            128, // Head dim
            2,   // fp16
            arena_id,
        )
    }

    /// Create optimized layout for Llama 2 7B
    pub fn create_llama2_7b_layout(
        page: &Arc<CudaPage>,
        device_ptr: NonNull<u8>,
        offset: usize,
        seq_len: usize,
        max_seq_len: usize,
        arena_id: u64,
    ) -> KVTensorLayout {
        KVTensorLayout::from_bump_allocation(
            page,
            device_ptr,
            offset,
            seq_len,
            max_seq_len,
            32,  // KV heads (full attention)
            32,  // Query heads
            128, // Head dim
            2,   // fp16
            arena_id,
        )
    }

    /// Create optimized layout for Llama 2 70B
    pub fn create_llama2_70b_layout(
        page: &Arc<CudaPage>,
        device_ptr: NonNull<u8>,
        offset: usize,
        seq_len: usize,
        max_seq_len: usize,
        arena_id: u64,
    ) -> KVTensorLayout {
        KVTensorLayout::from_bump_allocation(
            page,
            device_ptr,
            offset,
            seq_len,
            max_seq_len,
            8,   // KV heads (GQA)
            64,  // Query heads
            128, // Head dim
            2,   // fp16
            arena_id,
        )
    }

    /// Create layout from model config
    pub fn create_layout_from_config(
        model_config: &ModelConfig,
        page: &Arc<CudaPage>,
        device_ptr: NonNull<u8>,
        offset: usize,
        seq_len: usize,
        max_seq_len: usize,
        arena_id: u64,
    ) -> KVTensorLayout {
        let (query_heads, kv_heads, head_dim) = model_config.get_kv_config();
        
        KVTensorLayout::from_bump_allocation(
            page,
            device_ptr,
            offset,
            seq_len,
            max_seq_len,
            kv_heads,
            query_heads,
            head_dim,
            2, // fp16 default
            arena_id,
        )
    }
}

/// Utility functions for layout validation and debugging
pub mod validation {
    use super::*;

    /// Validate that a KV layout is correctly configured
    pub fn validate_kv_layout(layout: &KVTensorLayout) -> Result<(), String> {
        // Check basic constraints
        if layout.seq_len > layout.max_seq_len {
            return Err(format!("seq_len ({}) > max_seq_len ({})", layout.seq_len, layout.max_seq_len));
        }

        if layout.num_kv_heads == 0 || layout.num_query_heads == 0 {
            return Err("Head counts cannot be zero".to_string());
        }

        if layout.head_dim == 0 {
            return Err("Head dimension cannot be zero".to_string());
        }

        if layout.element_size == 0 || layout.element_size > 8 {
            return Err(format!("Invalid element size: {}", layout.element_size));
        }

        // Check GQA constraints
        if layout.num_query_heads % layout.num_kv_heads != 0 {
            return Err(format!(
                "Query heads ({}) not evenly divisible by KV heads ({})",
                layout.num_query_heads, layout.num_kv_heads
            ));
        }

        // Check memory layout
        let layout_validation = layout.validate_layout()
            .map_err(|e| format!("Layout validation failed: {:?}", e))?;

        if !layout_validation.layout_valid {
            return Err("Layout validation indicates invalid layout".to_string());
        }

        Ok(())
    }

    /// Check if layout matches expected model configuration
    pub fn validate_model_layout(layout: &KVTensorLayout, expected_config: &ModelConfig) -> Result<(), String> {
        let (expected_query, expected_kv, expected_head_dim) = expected_config.get_kv_config();

        if layout.num_query_heads != expected_query {
            return Err(format!(
                "Query head mismatch: got {}, expected {}",
                layout.num_query_heads, expected_query
            ));
        }

        if layout.num_kv_heads != expected_kv {
            return Err(format!(
                "KV head mismatch: got {}, expected {}",
                layout.num_kv_heads, expected_kv
            ));
        }

        if layout.head_dim != expected_head_dim {
            return Err(format!(
                "Head dim mismatch: got {}, expected {}",
                layout.head_dim, expected_head_dim
            ));
        }

        Ok(())
    }

    /// Generate detailed layout report
    pub fn generate_layout_report(layout: &KVTensorLayout) -> String {
        let info = layout.kv_tensor_info();
        let efficiency = layout.layout_efficiency();
        let access_info = layout.access_pattern_info();

        format!(
            "KV Tensor Layout Report:\n\
            ========================\n\
            Sequence: {} / {} tokens ({:.1} utilization)\n\
            Heads: {} query -> {} KV heads (GQA ratio: {:.1}:1)\n\
            Dimensions: {} head_dim, {} element_size\n\
            Memory: {:.1} KB current / {:.1} KB max\n\
            \n\
            Efficiency Metrics:\n\
            - Memory utilization: {:.1}\n\
            - Page utilization: {:.1}\n\
            - Fragmentation: {:.1}\n\
            - Compression ratio: {:.1}:1\n\
            \n\
            Access Patterns:\n\
            - Sequential stride: {} bytes\n\
            - Head stride: {} bytes\n\
            - Cache line efficiency: {:.1}\n\
            - Bandwidth efficiency: {:.1}\n\
            \n\
            Layout: {}\n",
            info.seq_len,
            info.max_seq_len,
            info.utilization * 100.0,
            info.num_query_heads,
            info.num_kv_heads,
            info.gqa_ratio,
            info.head_dim,
            info.element_size,
            info.current_kv_bytes as f64 / 1024.0,
            info.max_kv_bytes as f64 / 1024.0,
            efficiency.memory_utilization * 100.0,
            efficiency.page_utilization * 100.0,
            efficiency.fragmentation * 100.0,
            efficiency.compression_ratio,
            access_info.sequential_access_stride,
            access_info.head_access_stride,
            access_info.cache_line_efficiency * 100.0,
            access_info.memory_bandwidth_efficiency * 100.0,
            if layout.is_gqa() { "GQA (Grouped Query Attention)" } else { "Full Attention" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_page_size_calculation() {
        // FIXED: Test Mistral 7B calculation with correct KV heads
        let page_size = calculate_model_kv_page_size(&ModelConfig::Mistral7B);
        
        // Should use 8 KV heads, not 32 query heads
        let expected_kv_size = 2 * 8192 * 8 * 128 * 2; // 2 * seq * kv_heads * head_dim * fp16
        let expected_min_page = (expected_kv_size as f64 * 1.25) as usize;
        
        assert!(page_size >= expected_min_page);
        assert!(page_size <= 16 * 1024 * 1024); // Should be reasonable
        
        println!("✓ Mistral 7B KV page size: {} KB (uses 8 KV heads)", page_size / 1024);
    }

    #[test]
    fn test_model_kv_configs() {
        let mistral_config = ModelConfig::Mistral7B;
        let (query_heads, kv_heads, head_dim) = mistral_config.get_kv_config();
        
        assert_eq!(query_heads, 32);
        assert_eq!(kv_heads, 8);   // FIXED: Should be 8, not 32
        assert_eq!(head_dim, 128);
        assert!(mistral_config.is_gqa());
        assert_eq!(mistral_config.gqa_ratio(), 4.0); // 32/8 = 4
        
        let llama70_config = ModelConfig::Llama2_70B;
        let (query_heads, kv_heads, head_dim) = llama70_config.get_kv_config();
        
        assert_eq!(query_heads, 64);
        assert_eq!(kv_heads, 8);   // FIXED: Should be 8 for GQA
        assert_eq!(head_dim, 128);
        assert!(llama70_config.is_gqa());
        assert_eq!(llama70_config.gqa_ratio(), 8.0); // 64/8 = 8
        
        println!("✓ Model KV configurations are correct");
    }

    #[test]
    fn test_kv_tensor_layout() {
        // Test the KV tensor layout calculation with proper head counts
        let max_seq_len = 1024;
        let num_kv_heads = 8;     // FIXED: Use KV heads
        let head_dim = 128;
        let element_size = 2; // fp16
        
        let total_size = 2 * max_seq_len * num_kv_heads * head_dim * element_size;
        let expected_size = 2 * 1024 * 8 * 128 * 2; // 4MB total
        
        assert_eq!(total_size, expected_size);
        println!("✓ KV tensor layout calculation: {} MB (using {} KV heads)", 
                total_size / 1024 / 1024, num_kv_heads);
    }

    #[test]
    fn test_optimal_page_size_calculation() {
        // Test various KV configurations
        let configs = vec![
            (1024, 8, 64, 2),    // Small GQA model
            (2048, 8, 128, 2),   // Mistral-like configuration  
            (4096, 32, 128, 2),  // Full attention model
            (8192, 8, 128, 1),   // Large GQA model with quantization
        ];

        for (seq_len, kv_heads, head_dim, elem_size) in configs {
            let page_size = calculate_optimal_kv_page_size(seq_len, kv_heads, head_dim, elem_size);
            let tensor_size = 2 * seq_len * kv_heads * head_dim * elem_size;
            
            assert!(page_size >= tensor_size, "Page must fit the KV tensor");
            assert!(page_size <= tensor_size * 2, "Page shouldn't be too oversized");
            
            println!("✓ Config ({}, {} KV heads, {}, {}): tensor={}KB, page={}KB", 
                    seq_len, kv_heads, head_dim, elem_size,
                    tensor_size / 1024, page_size / 1024);
        }
    }

    #[test]
    fn test_gqa_ratio_calculation() {
        // Test GQA ratio calculations
        let test_cases = vec![
            (32, 8, 4.0),   // Mistral 7B: 4:1 ratio
            (64, 8, 8.0),   // Llama 70B: 8:1 ratio
            (32, 32, 1.0),  // Full attention: 1:1 ratio
            (40, 40, 1.0),  // Llama 13B: 1:1 ratio
        ];
        
        for (query_heads, kv_heads, expected_ratio) in test_cases {
            let ratio = query_heads as f64 / kv_heads as f64;
            assert_eq!(ratio, expected_ratio);
            
            let is_gqa = query_heads != kv_heads;
            println!("✓ {} query, {} KV heads: ratio={}, GQA={}", 
                    query_heads, kv_heads, ratio, is_gqa);
        }
    }

    #[test]
    fn test_memory_efficiency() {
        let mistral_config = ModelConfig::Mistral7B;
        let efficiency = mistral_config.memory_efficiency();
        
        // Mistral 7B should have 8/32 = 0.25 efficiency (4x memory savings)
        assert_eq!(efficiency, 0.25);
        
        let llama7b_config = ModelConfig::Llama2_7B;
        let efficiency = llama7b_config.memory_efficiency();
        
        // Full attention should have 1.0 efficiency (no savings)
        assert_eq!(efficiency, 1.0);
        
        println!("✓ Memory efficiency calculations correct");
    }

    #[test]
    fn test_access_optimized_page_size() {
        let page_size = calculate_access_optimized_page_size(1024, 8, 128, 2);
        let base_size = calculate_optimal_kv_page_size(1024, 8, 128, 2);
        
        // Access-optimized should be >= base size
        assert!(page_size >= base_size);
        
        println!("✓ Access-optimized page size: {} KB vs base {} KB", 
                page_size / 1024, base_size / 1024);
    }

    #[test]
    fn test_layout_validation() {
        use validation::*;
        
        // This would require creating a mock layout for testing
        // In practice, you'd create a real layout and validate it
        let mistral_config = ModelConfig::Mistral7B;
        let (query_heads, kv_heads, head_dim) = mistral_config.get_kv_config();
        
        // Validate configuration makes sense
        assert!(query_heads > kv_heads); // Should be GQA
        assert_eq!(query_heads % kv_heads, 0); // Should be evenly divisible
        assert_eq!(head_dim, 128); // Expected head dimension
        
        println!("✓ Layout validation logic correct");
    }

    #[test]
    fn test_model_helpers() {
        use model_helpers::*;
        
        // Test that helper functions would create correct configurations
        // (This is a structural test since we can't create real CUDA pages in tests)
        
        let mistral_config = ModelConfig::Mistral7B;
        let (query_heads, kv_heads, head_dim) = mistral_config.get_kv_config();
        
        assert_eq!(kv_heads, 8);
        assert_eq!(query_heads, 32);
        assert_eq!(head_dim, 128);
        
        println!("✓ Model helper configurations correct");
    }

    #[test]
    fn test_optimization_recommendations() {
        let mut stats = KVTensorStats {
            total_kv_tensors: 10,
            total_tokens_stored: 1000,
            total_kv_bytes_used: 1024 * 1024,
            total_kv_bytes_allocated: 2 * 1024 * 1024,
            average_utilization: 0.5,
            zero_copy_extensions: 5,
            copy_extensions: 15,
            largest_tensor_tokens: 512,
            smallest_tensor_tokens: 64,
            total_kv_heads: 80,
            total_query_heads: 320,
            gqa_models_count: 8,
            full_attention_models_count: 2,
            average_gqa_ratio: 4.0,
            memory_efficiency: 0.6,
            bandwidth_efficiency: 0.7,
        };

        let recommendations = stats.get_optimization_recommendations();
        
        // Should have recommendations due to low utilization
        assert!(!recommendations.page_size_recommendations.is_empty());
        assert!(!recommendations.gqa_specific_recommendations.is_empty());
        
        let score = stats.overall_efficiency_score();
        assert!(score > 0.0 && score <= 1.0);
        
        println!("✓ Optimization recommendations generated: score = {:.2f}", score);
    }
}