// src/cuda/bindings.rs - CUDA Runtime API bindings
use std::ffi::c_void;

// CUDA Runtime API bindings - these link to actual libcudart
#[cfg(cuda_available)]
#[link(name = "cudart")]
extern "C" {
    // Device management
    pub fn cudaSetDevice(device: i32) -> i32;
    pub fn cudaGetDevice(device: *mut i32) -> i32;
    pub fn cudaGetDeviceCount(count: *mut i32) -> i32;
    pub fn cudaDeviceReset() -> i32;
    pub fn cudaDeviceSynchronize() -> i32;
    
    // Memory management - REAL CUDA memory operations
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(devPtr: *mut c_void) -> i32;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: i32, stream: *mut c_void) -> i32;
    pub fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> i32;
    pub fn cudaMemsetAsync(devPtr: *mut c_void, value: i32, count: usize, stream: *mut c_void) -> i32;
    
    // Memory info
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    
    // Stream management
    pub fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    pub fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    pub fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    pub fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    pub fn cudaStreamQuery(stream: *mut c_void) -> i32;
    
    // Error handling
    pub fn cudaGetLastError() -> i32;
    pub fn cudaGetErrorString(error: i32) -> *const i8;
    
    // Device properties
    pub fn cudaGetDeviceProperties(prop: *mut CudaDeviceProperties, device: i32) -> i32;
    pub fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
}

// CUDA constants
pub const CUDA_SUCCESS: i32 = 0;
pub const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;
pub const CUDA_ERROR_NOT_INITIALIZED: i32 = 3;
pub const CUDA_ERROR_INVALID_DEVICE: i32 = 10;
pub const CUDA_ERROR_INVALID_VALUE: i32 = 11;
pub const CUDA_ERROR_NOT_READY: i32 = 600;

pub const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

pub const CUDA_STREAM_NON_BLOCKING: u32 = 0x01;

// Device attributes
pub const CUDA_DEVICE_ATTR_MEMORY_CLOCK_RATE: i32 = 36;
pub const CUDA_DEVICE_ATTR_GLOBAL_MEMORY_BUS_WIDTH: i32 = 37;
pub const CUDA_DEVICE_ATTR_MULTIPROCESSOR_COUNT: i32 = 16;
pub const CUDA_DEVICE_ATTR_MAX_THREADS_PER_MULTIPROCESSOR: i32 = 39;

#[repr(C)]
pub struct CudaDeviceProperties {
    pub name: [i8; 256],
    pub uuid: [u8; 16],
    pub luid: [i8; 8],
    pub luid_device_node_mask: u32,
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: i32,
    pub warp_size: i32,
    pub mem_pitch: usize,
    pub max_threads_per_block: i32,
    pub max_threads_dim: [i32; 3],
    pub max_grid_size: [i32; 3],
    pub clock_rate: i32,
    pub total_const_mem: usize,
    pub major: i32,
    pub minor: i32,
    pub texture_alignment: usize,
    pub texture_pitch_alignment: usize,
    pub device_overlap: i32,
    pub multiprocessor_count: i32,
    pub kernel_exec_timeout_enabled: i32,
    pub integrated: i32,
    pub can_map_host_memory: i32,
    pub compute_mode: i32,
    pub max_texture_1d: i32,
    pub max_texture_1d_mipmap: i32,
    pub max_texture_1d_linear: i32,
    pub max_texture_2d: [i32; 2],
    pub max_texture_2d_mipmap: [i32; 2],
    pub max_texture_2d_linear: [i32; 3],
    pub max_texture_2d_gather: [i32; 2],
    pub max_texture_3d: [i32; 3],
    pub max_texture_3d_alt: [i32; 3],
    pub max_texture_cubemap: i32,
    pub max_texture_1d_layered: [i32; 2],
    pub max_texture_2d_layered: [i32; 3],
    pub max_texture_cubemap_layered: [i32; 2],
    pub max_surface_1d: i32,
    pub max_surface_2d: [i32; 2],
    pub max_surface_3d: [i32; 3],
    pub max_surface_1d_layered: [i32; 2],
    pub max_surface_2d_layered: [i32; 3],
    pub max_surface_cubemap: i32,
    pub max_surface_cubemap_layered: [i32; 2],
    pub surface_alignment: usize,
    pub concurrent_kernels: i32,
    pub ecc_enabled: i32,
    pub pci_bus_id: i32,
    pub pci_device_id: i32,
    pub pci_domain_id: i32,
    pub tcc_driver: i32,
    pub async_engine_count: i32,
    pub unified_addressing: i32,
    pub memory_clock_rate: i32,
    pub memory_bus_width: i32,
    pub l2_cache_size: i32,
    pub persist_ing_l2_cache_max_size: i32,
    pub max_threads_per_multiprocessor: i32,
    pub stream_priorities_supported: i32,
    pub global_l1_cache_supported: i32,
    pub local_l1_cache_supported: i32,
    pub shared_mem_per_multiprocessor: usize,
    pub regs_per_multiprocessor: i32,
    pub managed_memory: i32,
    pub is_multi_gpu_board: i32,
    pub multi_gpu_board_group_id: i32,
    pub host_native_atomic_supported: i32,
    pub single_to_double_precision_perf_ratio: i32,
    pub pageable_memory_access: i32,
    pub concurrent_managed_access: i32,
    pub compute_preemption_supported: i32,
    pub can_use_host_pointer_for_registered_mem: i32,
    pub cooperative_launch: i32,
    pub cooperative_multi_device_launch: i32,
    pub shared_mem_per_block_optin: usize,
    pub pageable_memory_access_uses_host_page_tables: i32,
    pub direct_managed_mem_access_from_host: i32,
    pub max_blocks_per_multiprocessor: i32,
    pub access_policy_max_window_size: i32,
    pub reserved_shared_mem_per_block: usize,
}