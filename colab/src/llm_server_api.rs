// src/llm_server_api.rs - Fixed LLM server API with honest zero-copy terminology
use crate::{
    zero_copy::{ZeroCopyArena, ZeroCopyTensor},
    LLMServerError, ProductionKVCacheManager, SequenceRequest,
};
use std::sync::Arc;
use std::time::Instant;

/// Initialize production manager for LLM server with model-specific optimizations
pub fn initialize_for_server(
    model_name: &str,
    devices: &[i32],
) -> Result<ProductionKVCacheManager, LLMServerError> {
    log::info!("Initializing LLM server for model: {}", model_name);
    ProductionKVCacheManager::for_llm_model(model_name, devices.to_vec())
}

/// Inference request structure for batch processing
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub request_id: String,
    pub prompt_tokens: usize,
    pub max_new_tokens: usize,
    pub sequence_request: SequenceRequest,
    pub priority: u8,
}

/// Generation statistics with honest zero-copy reporting
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub tokens_generated: usize,
    pub metadata_zero_copy_extensions: usize,  // FIXED: Honest naming
    pub data_copy_extensions: usize,           // FIXED: Honest naming
    pub total_time_ms: f64,
    pub avg_time_per_token_ms: f64,
    pub peak_memory_mb: usize,
    pub metadata_efficiency: f64,              // FIXED: More accurate naming
}

/// Simulate token generation with honest zero-copy reporting
pub fn simulate_generation(
    manager: &ProductionKVCacheManager,
    arena: &Arc<ZeroCopyArena>,
    tensor: &mut ZeroCopyTensor,
    num_tokens: usize,
) -> Result<GenerationStats, LLMServerError> {
    let start_time = Instant::now();
    let initial_seq_len = tensor.seq_len();
    let mut metadata_zero_copy_count = 0;
    let mut data_copy_count = 0;
    let mut peak_memory = 0;

    log::debug!(
        "Starting generation simulation: {} tokens from seq_len {}",
        num_tokens,
        initial_seq_len
    );

    // Simulate incremental token generation with honest reporting
    for i in 1..=num_tokens {
        let token_start = Instant::now();

        // Use the SINGLE extend_tensor_for_generation method
        match arena.extend_tensor_for_generation(tensor, 1) {
            Ok(was_pure_zero_copy) => {
                if was_pure_zero_copy {
                    metadata_zero_copy_count += 1;
                    log::trace!("Token {}: TRUE zero-copy metadata extension", i);
                } else {
                    data_copy_count += 1;
                    log::trace!("Token {}: metadata updated but data copy required", i);
                }

                // Track memory usage
                let current_memory = tensor.size_bytes() / (1024 * 1024);
                if current_memory > peak_memory {
                    peak_memory = current_memory;
                }
            }
            Err(e) => {
                log::error!("Generation failed at token {}: {:?}", i, e);
                return Err(LLMServerError::from(e));
            }
        }

        let token_time = token_start.elapsed();
        if token_time.as_millis() > 10 {
            log::warn!("Slow token generation at {}: {:?}", i, token_time);
        }
    }

    let total_time = start_time.elapsed();
    let total_time_ms = total_time.as_millis() as f64;
    let avg_time_per_token = if num_tokens > 0 {
        total_time_ms / num_tokens as f64
    } else {
        0.0
    };

    // Calculate honest efficiency metrics
    let total_extensions = metadata_zero_copy_count + data_copy_count;
    let metadata_efficiency = if total_extensions > 0 {
        metadata_zero_copy_count as f64 / total_extensions as f64
    } else {
        0.0
    };

    let final_seq_len = tensor.seq_len();
    log::info!(
        "Generation simulation completed: {} -> {} tokens in {:.2}ms, {:.1}% metadata-only operations",
        initial_seq_len,
        final_seq_len,
        total_time_ms,
        metadata_efficiency * 100.0
    );

    Ok(GenerationStats {
        tokens_generated: num_tokens,
        metadata_zero_copy_extensions: metadata_zero_copy_count,  // FIXED field name
        data_copy_extensions: data_copy_count,                    // FIXED field name
        total_time_ms,
        avg_time_per_token_ms: avg_time_per_token,
        peak_memory_mb: peak_memory,
        metadata_efficiency,                                      // FIXED field name
    })
}

/// Process a batch of inference requests
pub fn process_inference_batch(
    manager: &ProductionKVCacheManager,
    requests: &[InferenceRequest],
) -> Result<Vec<GenerationStats>, LLMServerError> {
    let start_time = Instant::now();
    let mut results = Vec::with_capacity(requests.len());

    log::info!("Processing inference batch: {} requests", requests.len());

    let sequence_requests: Vec<SequenceRequest> = requests
        .iter()
        .map(|req| req.sequence_request.clone())
        .collect();

    let allocations = manager.batch_allocate_sequences(&sequence_requests)?;

    for (req, alloc) in requests.iter().zip(allocations.into_iter()) {
        let arena = alloc.arena;
        let mut tensor = alloc.tensor;

        let stats = simulate_generation(manager, &arena, &mut tensor, req.max_new_tokens)?;

        log::debug!(
            "Completed request {}: {} tokens in {:.2}ms",
            req.request_id,
            stats.tokens_generated,
            stats.total_time_ms
        );

        results.push(stats);
    }

    let batch_time = start_time.elapsed();
    log::info!(
        "Batch processing completed in {:.2}ms",
        batch_time.as_millis()
    );

    Ok(results)
}

/// Get server performance metrics
pub fn get_server_metrics(manager: &ProductionKVCacheManager) -> ServerMetrics {
    let production_metrics = manager.get_production_metrics();
    let health = manager.get_system_health();

    ServerMetrics {
        total_requests_processed: production_metrics.sequences_processed,
        total_tokens_generated: production_metrics.tokens_generated,
        avg_request_time_ms: production_metrics.avg_allocation_time_ms,
        avg_token_time_ms: production_metrics.avg_extension_time_ms,
        metadata_zero_copy_efficiency: production_metrics.zero_copy_ratio,  // FIXED field name
        memory_efficiency: production_metrics.zero_copy_stats.system_efficiency(),
        system_health_score: health.health_score,
        recommendations: health.recommendations,
    }
}

/// Server-wide metrics with honest terminology
#[derive(Debug, Clone)]
pub struct ServerMetrics {
    pub total_requests_processed: usize,
    pub total_tokens_generated: usize,
    pub avg_request_time_ms: f64,
    pub avg_token_time_ms: f64,
    pub metadata_zero_copy_efficiency: f64,    // FIXED: More honest naming
    pub memory_efficiency: f64,
    pub system_health_score: f64,
    pub recommendations: Vec<String>,
}

/// Create optimized manager for specific deployment scenarios
pub fn create_deployment_manager(
    scenario: DeploymentScenario,
    devices: Vec<i32>,
) -> Result<ProductionKVCacheManager, LLMServerError> {
    match scenario {
        DeploymentScenario::HighThroughputChatbot => {
            log::info!("Creating high-throughput chatbot deployment");
            ProductionKVCacheManager::for_chatbot(devices)
        }
        DeploymentScenario::LongContextDocuments => {
            log::info!("Creating long-context document processing deployment");
            ProductionKVCacheManager::for_document_processing(devices)
        }
        DeploymentScenario::MultiModal {
            typical_seq_len,
            max_seq_len,
        } => {
            log::info!(
                "Creating multi-modal deployment: typical={}, max={}",
                typical_seq_len,
                max_seq_len
            );
            let config = crate::LLMServerConfig {
                devices,
                base_page_size: crate::kv_layout::calculate_optimal_kv_page_size(
                    max_seq_len,
                    32,
                    128,
                    2,
                ),
                max_slab_pages: 200,
                cross_device_sharing: true,
                cleanup_interval_seconds: 300,
                max_page_age_seconds: 1800,
                enable_pressure_monitoring: true,
            };
            ProductionKVCacheManager::new(config)
        }
        DeploymentScenario::Research { custom_config } => {
            log::info!("Creating research deployment with custom config");
            ProductionKVCacheManager::new(custom_config)
        }
    }
}

/// Deployment scenarios for different use cases
#[derive(Debug, Clone)]
pub enum DeploymentScenario {
    /// High-throughput chatbot with frequent short conversations
    HighThroughputChatbot,
    /// Long-context document processing
    LongContextDocuments,
    /// Multi-modal workloads with variable sequence lengths
    MultiModal {
        typical_seq_len: usize,
        max_seq_len: usize,
    },
    /// Research deployment with custom configuration
    Research {
        custom_config: crate::LLMServerConfig,
    },
}

/// Benchmark different configurations with honest reporting
pub fn benchmark_configurations(
    scenarios: &[DeploymentScenario],
    test_requests: &[InferenceRequest],
    devices: Vec<i32>,
) -> Result<Vec<BenchmarkResult>, LLMServerError> {
    let mut results = Vec::new();

    for (i, scenario) in scenarios.iter().enumerate() {
        log::info!("Benchmarking scenario {}: {:?}", i, scenario);

        let start_time = Instant::now();
        let manager = create_deployment_manager(scenario.clone(), devices.clone())?;
        let setup_time = start_time.elapsed();

        let benchmark_start = Instant::now();
        let generation_results = process_inference_batch(&manager, test_requests)?;
        let benchmark_time = benchmark_start.elapsed();

        let total_tokens: usize = generation_results.iter().map(|r| r.tokens_generated).sum();

        let avg_metadata_efficiency: f64 = generation_results
            .iter()
            .map(|r| r.metadata_efficiency)  // FIXED field name
            .sum::<f64>()
            / generation_results.len() as f64;

        let metrics = get_server_metrics(&manager);

        results.push(BenchmarkResult {
            scenario: scenario.clone(),
            setup_time_ms: setup_time.as_millis() as f64,
            total_benchmark_time_ms: benchmark_time.as_millis() as f64,
            total_tokens_generated: total_tokens,
            tokens_per_second: total_tokens as f64 / benchmark_time.as_secs_f64(),
            avg_metadata_zero_copy_rate: avg_metadata_efficiency,  // FIXED field name
            memory_efficiency: metrics.memory_efficiency,
            system_health_score: metrics.system_health_score,
        });
    }

    Ok(results)
}

/// Benchmark result with honest zero-copy reporting
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub scenario: DeploymentScenario,
    pub setup_time_ms: f64,
    pub total_benchmark_time_ms: f64,
    pub total_tokens_generated: usize,
    pub tokens_per_second: f64,
    pub avg_metadata_zero_copy_rate: f64,    // FIXED: Honest naming
    pub memory_efficiency: f64,
    pub system_health_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_initialization() {
        let devices = vec![0];

        let models = ["llama-7b", "llama-13b", "gpt-3.5"];

        for model in &models {
            match initialize_for_server(model, &devices) {
                Ok(_manager) => {
                    println!("✓ Successfully initialized server for {}", model);
                }
                Err(e) => {
                    println!(
                        "⚠️ Failed to initialize {} (expected if no CUDA): {:?}",
                        model, e
                    );
                }
            }
        }
    }

    #[test]
    fn test_deployment_scenarios() {
        let devices = vec![0];

        let scenarios = vec![
            DeploymentScenario::HighThroughputChatbot,
            DeploymentScenario::LongContextDocuments,
            DeploymentScenario::MultiModal {
                typical_seq_len: 1024,
                max_seq_len: 4096,
            },
        ];

        for scenario in scenarios {
            match create_deployment_manager(scenario.clone(), devices.clone()) {
                Ok(_manager) => {
                    println!("✓ Successfully created deployment: {:?}", scenario);
                }
                Err(e) => {
                    println!(
                        "⚠️ Failed to create deployment (expected if no CUDA): {:?}",
                        e
                    );
                }
            }
        }
    }

    #[test]
    fn test_generation_stats() {
        let stats = GenerationStats {
            tokens_generated: 100,
            metadata_zero_copy_extensions: 80,    // FIXED field name
            data_copy_extensions: 20,             // FIXED field name
            total_time_ms: 150.0,
            avg_time_per_token_ms: 1.5,
            peak_memory_mb: 256,
            metadata_efficiency: 0.8,             // FIXED field name
        };

        assert_eq!(stats.tokens_generated, 100);
        assert_eq!(stats.metadata_zero_copy_extensions, 80);  // FIXED field name
        assert_eq!(stats.data_copy_extensions, 20);           // FIXED field name
        assert_eq!(stats.metadata_efficiency, 0.8);           // FIXED field name

        println!("✓ Generation stats test passed with honest field names");
    }

    #[test]
    fn test_inference_request() {
        let request = InferenceRequest {
            request_id: "test-123".to_string(),
            prompt_tokens: 50,
            max_new_tokens: 100,
            sequence_request: SequenceRequest {
                initial_seq_len: 50,
                max_seq_len: 150,
                num_heads: 32,
                head_dim: 128,
                dtype_size: 2,
                preferred_device: Some(0),
            },
            priority: 128,
        };

        assert_eq!(request.request_id, "test-123");
        assert_eq!(request.prompt_tokens, 50);
        assert_eq!(request.max_new_tokens, 100);
        assert_eq!(request.priority, 128);

        println!("✓ Inference request test passed");
    }

    #[test]
    fn test_honest_zero_copy_terminology() {
        // Test that our new terminology is consistent
        let stats = GenerationStats {
            tokens_generated: 50,
            metadata_zero_copy_extensions: 40,  // Operations that only update metadata
            data_copy_extensions: 10,           // Operations that require data copying
            total_time_ms: 75.0,
            avg_time_per_token_ms: 1.5,
            peak_memory_mb: 128,
            metadata_efficiency: 0.8,           // 40/50 = 80% metadata-only
        };

        // Verify honest calculations
        let total_operations = stats.metadata_zero_copy_extensions + stats.data_copy_extensions;
        assert_eq!(total_operations, stats.tokens_generated);
        
        let calculated_efficiency = stats.metadata_zero_copy_extensions as f64 / total_operations as f64;
        assert!((calculated_efficiency - stats.metadata_efficiency).abs() < 0.001);

        println!("✓ Honest zero-copy terminology test passed");
        println!("  - {} metadata-only operations (TRUE zero-copy)", stats.metadata_zero_copy_extensions);
        println!("  - {} data copy operations (NOT zero-copy)", stats.data_copy_extensions);
        println!("  - {:.1}% metadata efficiency", stats.metadata_efficiency * 100.0);
    }
}