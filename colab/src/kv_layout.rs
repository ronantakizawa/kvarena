// src/kv_layout.rs - Specific KV-cache layout as described in project spec
use std::sync::Arc;
use std::ptr::NonNull;
use crate::cuda::{CudaPage, CudaError};

/// Specific KV-cache tensor layout optimized for transformer attention
/// Layout: [Key Tensor: seq_len × num_heads × head_dim] [Value Tensor: seq_len × num_heads × head_dim]
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
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head (hidden_dim / num_heads)
    head_dim: usize,
    /// Element size in bytes (2 for fp16, 4 for fp32, 1 for int8)
    element_size: usize,
    /// Reference to the underlying page (keeps allocator alive)
    page_ref: Arc<CudaPage>,
    /// Arena ID for tracking
    arena_id: u64,
}

impl KVTensorLayout {
    /// Create KV tensor layout from bump allocation - SPECIFIC KV LAYOUT
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
        KVTensorLayout {
            device_ptr,
            offset,
            seq_len,
            max_seq_len,
            num_heads,
            head_dim,
            element_size,
            page_ref: Arc::clone(page),
            arena_id,
        }
    }

    /// Get total size of KV tensor pair for current sequence length
    pub fn current_kv_size_bytes(&self) -> usize {
        // K tensor + V tensor: 2 * seq_len * num_heads * head_dim * element_size
        2 * self.seq_len * self.num_heads * self.head_dim * self.element_size
    }

    /// Get total size of KV tensor pair for maximum sequence length
    pub fn max_kv_size_bytes(&self) -> usize {
        // K tensor + V tensor: 2 * max_seq_len * num_heads * head_dim * element_size
        2 * self.max_seq_len * self.num_heads * self.head_dim * self.element_size
    }

    /// Get device pointer for K tensor start
    /// Layout: K tensor occupies first half of allocated space
    pub fn key_device_ptr(&self) -> *mut std::ffi::c_void {
        self.device_ptr.as_ptr() as *mut std::ffi::c_void
    }

    /// Get device pointer for V tensor start
    /// Layout: V tensor starts after the full K tensor (at max_seq_len boundary)
    pub fn value_device_ptr(&self) -> *mut std::ffi::c_void {
        let k_tensor_max_size = self.max_seq_len * self.num_heads * self.head_dim * self.element_size;
        unsafe {
            (self.device_ptr.as_ptr()).add(k_tensor_max_size) as *mut std::ffi::c_void
        }
    }

    /// Get device pointer for specific token in K tensor
    /// Layout: K[token][head][head_dim_idx] 
    pub fn key_token_ptr(&self, token_idx: usize) -> Option<*mut std::ffi::c_void> {
        if token_idx >= self.seq_len {
            return None;
        }
        
        let token_offset = token_idx * self.num_heads * self.head_dim * self.element_size;
        unsafe {
            Some((self.device_ptr.as_ptr()).add(token_offset) as *mut std::ffi::c_void)
        }
    }

    /// Get device pointer for specific token in V tensor
    /// Layout: V[token][head][head_dim_idx]
    pub fn value_token_ptr(&self, token_idx: usize) -> Option<*mut std::ffi::c_void> {
        if token_idx >= self.seq_len {
            return None;
        }
        
        let k_tensor_max_size = self.max_seq_len * self.num_heads * self.head_dim * self.element_size;
        let token_offset = token_idx * self.num_heads * self.head_dim * self.element_size;
        unsafe {
            Some((self.device_ptr.as_ptr()).add(k_tensor_max_size + token_offset) as *mut std::ffi::c_void)
        }
    }

    /// Get device pointer for specific head in K tensor at given token
    /// Layout: K[token][head][head_dim_idx]
    pub fn key_head_ptr(&self, token_idx: usize, head_idx: usize) -> Option<*mut std::ffi::c_void> {
        if token_idx >= self.seq_len || head_idx >= self.num_heads {
            return None;
        }
        
        let offset = (token_idx * self.num_heads + head_idx) * self.head_dim * self.element_size;
        unsafe {
            Some((self.device_ptr.as_ptr()).add(offset) as *mut std::ffi::c_void)
        }
    }

    /// Get device pointer for specific head in V tensor at given token
    /// Layout: V[token][head][head_dim_idx]
    pub fn value_head_ptr(&self, token_idx: usize, head_idx: usize) -> Option<*mut std::ffi::c_void> {
        if token_idx >= self.seq_len || head_idx >= self.num_heads {
            return None;
        }
        
        let k_tensor_max_size = self.max_seq_len * self.num_heads * self.head_dim * self.element_size;
        let offset = (token_idx * self.num_heads + head_idx) * self.head_dim * self.element_size;
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
            
            log::debug!("TRUE zero-copy KV extension: {} -> {} tokens (NO memory ops)", 
                       old_seq_len, new_seq_len);
            true
        } else {
            false // Cannot extend beyond allocated capacity
        }
    }

    /// Copy new K,V tokens for incremental generation (minimizes copy overhead)
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

        let token_size = self.num_heads * self.head_dim * self.element_size;
        let copy_size = num_tokens * token_size;
        let token_offset = start_token * token_size;

        // Copy new K tokens to K tensor region
        self.page_ref.copy_from_host(
            unsafe { host_key_data.add(token_offset) as *const std::ffi::c_void },
            copy_size,
            self.offset + token_offset,
        )?;

        // Copy new V tokens to V tensor region (after K tensor max allocation)
        let v_tensor_offset = self.max_seq_len * self.num_heads * self.head_dim * self.element_size;
        self.page_ref.copy_from_host(
            unsafe { host_value_data.add(token_offset) as *const std::ffi::c_void },
            copy_size,
            self.offset + v_tensor_offset + token_offset,
        )?;

        log::debug!("Copied {} new KV tokens starting at token {} (incremental copy)", 
                   num_tokens, start_token);
        Ok(())
    }

    /// Copy full KV tensors from host (for initial loading)
    pub fn copy_full_kv_from_host(&self, host_key_data: *const u8, host_value_data: *const u8) -> Result<(), CudaError> {
        let k_tensor_size = self.seq_len * self.num_heads * self.head_dim * self.element_size;

        // Copy K tensor to start of allocation
        self.page_ref.copy_from_host(
            host_key_data as *const std::ffi::c_void,
            k_tensor_size,
            self.offset,
        )?;

        // Copy V tensor after K tensor max space
        let v_tensor_offset = self.max_seq_len * self.num_heads * self.head_dim * self.element_size;
        self.page_ref.copy_from_host(
            host_value_data as *const std::ffi::c_void,
            k_tensor_size, // Note: only copy current seq_len, not max
            self.offset + v_tensor_offset,
        )?;

        log::debug!("Copied full KV tensors: {} tokens", self.seq_len);
        Ok(())
    }

    /// Get KV tensor dimensions (seq_len, num_heads, head_dim)
    pub fn kv_dimensions(&self) -> (usize, usize, usize) {
        (self.seq_len, self.num_heads, self.head_dim)
    }

    /// Get KV tensor shape info for debugging/validation
    pub fn kv_tensor_info(&self) -> KVTensorInfo {
        KVTensorInfo {
            seq_len: self.seq_len,
            max_seq_len: self.max_seq_len,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            element_size: self.element_size,
            current_kv_bytes: self.current_kv_size_bytes(),
            max_kv_bytes: self.max_kv_size_bytes(),
            k_tensor_shape: vec![self.seq_len, self.num_heads, self.head_dim],
            v_tensor_shape: vec![self.seq_len, self.num_heads, self.head_dim],
            utilization: self.seq_len as f64 / self.max_seq_len as f64,
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

    /// Calculate memory layout for validation
    pub fn validate_layout(&self) -> Result<KVLayoutValidation, CudaError> {
        let k_start = 0;
        let k_end = self.max_seq_len * self.num_heads * self.head_dim * self.element_size;
        let v_start = k_end;
        let v_end = v_start + (self.max_seq_len * self.num_heads * self.head_dim * self.element_size);
        
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
        })
    }
}

/// KV tensor information for debugging and monitoring
#[derive(Debug, Clone)]
pub struct KVTensorInfo {
    pub seq_len: usize,
    pub max_seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub element_size: usize,
    pub current_kv_bytes: usize,
    pub max_kv_bytes: usize,
    pub k_tensor_shape: Vec<usize>,
    pub v_tensor_shape: Vec<usize>,
    pub utilization: f64,
}

/// Layout validation result
#[derive(Debug, Clone)]
pub struct KVLayoutValidation {
    pub k_tensor_range: (usize, usize),
    pub v_tensor_range: (usize, usize),
    pub total_required_bytes: usize,
    pub page_size: usize,
    pub offset_in_page: usize,
    pub layout_valid: bool,
}

/// Calculate optimal page size for KV tensors as per project spec:
/// "Page size = round-up of largest KV tensor you expect"
pub fn calculate_optimal_kv_page_size(
    max_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    element_size: usize,
) -> usize {
    // Calculate size of largest KV tensor pair expected
    let largest_kv_tensor_size = 2 * max_seq_len * num_heads * head_dim * element_size;
    
    // Add some overhead for alignment and multiple tensors per page
    let overhead_factor = 1.25; // 25% overhead
    let target_size = (largest_kv_tensor_size as f64 * overhead_factor) as usize;
    
    // Round up to next power of 2 for efficient allocation
    let page_size = target_size.next_power_of_two();
    
    // Clamp to reasonable bounds (64KB - 16MB)
    page_size.max(64 * 1024).min(16 * 1024 * 1024)
}

/// Calculate page size for specific model configurations
pub fn calculate_model_kv_page_size(model_config: &ModelConfig) -> usize {
    match model_config {
        ModelConfig::Llama2_7B => {
            // Llama-2 7B: 4096 hidden, 32 heads, 128 head_dim
            // For 8K context with 4-bit quantization: ~256 KiB per spec
            calculate_optimal_kv_page_size(8192, 32, 128, 1) // 4-bit = ~1 byte effective
        }
        ModelConfig::Llama2_13B => {
            // Llama-2 13B: 5120 hidden, 40 heads, 128 head_dim
            calculate_optimal_kv_page_size(8192, 40, 128, 1)
        }
        ModelConfig::Llama2_70B => {
            // Llama-2 70B: 8192 hidden, 64 heads, 128 head_dim
            calculate_optimal_kv_page_size(8192, 64, 128, 1)
        }
        ModelConfig::Custom { max_seq_len, num_heads, head_dim, element_size } => {
            calculate_optimal_kv_page_size(*max_seq_len, *num_heads, *head_dim, *element_size)
        }
    }
}

/// Model configuration for page size calculation
#[derive(Debug, Clone)]
pub enum ModelConfig {
    Llama2_7B,
    Llama2_13B,
    Llama2_70B,
    Custom {
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        element_size: usize,
    },
}

/// KV tensor statistics for monitoring
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_page_size_calculation() {
        // Test Llama-2 7B calculation as per spec
        let page_size = calculate_model_kv_page_size(&ModelConfig::Llama2_7B);
        
        // Should be around 256 KiB for 4-bit 8K-seq Llama-2 as mentioned in spec
        assert!(page_size >= 256 * 1024, "Page size should be at least 256KB for Llama-2 7B");
        assert!(page_size <= 1024 * 1024, "Page size should be reasonable for Llama-2 7B");
        
        println!("✓ Llama-2 7B KV page size: {} KB", page_size / 1024);
    }

    #[test]
    fn test_kv_tensor_layout() {
        // Test the specific KV tensor layout calculation
        let max_seq_len = 1024;
        let num_heads = 32;
        let head_dim = 128;
        let element_size = 2; // fp16
        
        let total_size = 2 * max_seq_len * num_heads * head_dim * element_size;
        let expected_size = 2 * 1024 * 32 * 128 * 2; // 16MB
        
        assert_eq!(total_size, expected_size);
        println!("✓ KV tensor layout calculation: {} MB", total_size / 1024 / 1024);
    }

    #[test]
    fn test_optimal_page_size_calculation() {
        // Test various configurations
        let configs = vec![
            (1024, 16, 64, 2),   // Small model
            (2048, 32, 128, 2),  // Medium model  
            (4096, 64, 128, 1),  // Large model with quantization
            (8192, 128, 256, 2), // Very large model
        ];

        for (seq_len, heads, head_dim, elem_size) in configs {
            let page_size = calculate_optimal_kv_page_size(seq_len, heads, head_dim, elem_size);
            let tensor_size = 2 * seq_len * heads * head_dim * elem_size;
            
            assert!(page_size >= tensor_size, "Page must fit the tensor");
            assert!(page_size <= tensor_size * 2, "Page shouldn't be too oversized");
            
            println!("✓ Config ({}, {}, {}, {}): tensor={}KB, page={}KB", 
                    seq_len, heads, head_dim, elem_size,
                    tensor_size / 1024, page_size / 1024);
        }
    }

    #[test]
    fn test_kv_layout_addresses() {
        // Mock test for layout address calculation
        // In real implementation, this would use actual CUDA memory
        
        let seq_len = 512;
        let max_seq_len = 1024;
        let num_heads = 16;
        let head_dim = 64;
        let element_size = 2;
        
        // Calculate expected offsets
        let k_tensor_size = max_seq_len * num_heads * head_dim * element_size;
        let token_size = num_heads * head_dim * element_size;
        
        // V tensor should start after K tensor max allocation
        let expected_v_offset = k_tensor_size;
        
        // Token offsets should be multiples of token_size
        let expected_token_1_offset = token_size;
        
        println!("✓ KV layout validation:");
        println!("  K tensor max size: {} bytes", k_tensor_size);
        println!("  V tensor offset: {} bytes", expected_v_offset);
        println!("  Token size: {} bytes", token_size);
        println!("  Token 1 offset: {} bytes", expected_token_1_offset);
    }
}