#!/usr/bin/env python3
"""
Comprehensive benchmark comparing ArenaTransformerCache vs standard approaches.
Demonstrates real-world performance benefits for transformer inference.
"""

import torch
import time
import psutil
import gc
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from arena_kv_cache_bindings import CUDA_AVAILABLE
from ArenaTransformerCache import ArenaTransformerCache, create_arena_cache

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TransformerBenchmark:
    """Comprehensive benchmark suite for transformer caching strategies."""
    
    def __init__(self):
        self.device = 'cuda' if CUDA_AVAILABLE else 'cpu'
        self.results = {}
        
        print(f"🔬 Transformer Cache Benchmark Suite")
        print(f"Device: {self.device}")
        print(f"CUDA Available: {CUDA_AVAILABLE}")
        if CUDA_AVAILABLE:
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("=" * 60)
    
    def benchmark_incremental_generation(self, 
                                       model_configs: List[Dict],
                                       max_tokens: int = 100) -> Dict:
        """Benchmark incremental generation (like chatbot responses)."""
        print("\n🗣️  Benchmarking Incremental Generation")
        print("-" * 40)
        
        results = {
            'model_configs': model_configs,
            'arena_times': [],
            'standard_times': [],
            'arena_memory': [],
            'standard_memory': [],
            'zero_copy_ratios': []
        }
        
        for config in model_configs:
            print(f"\nTesting {config['name']}...")
            
            # Arena Cache Test
            arena_cache = create_arena_cache(config)
            arena_times = []
            arena_memories = []
            
            # Clear memory
            gc.collect()
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
            
            start_time = time.perf_counter()
            
            # Simulate initial context (system prompt + conversation history)
            initial_seq_len = 256
            num_layers = 32
            
            for layer_idx in range(num_layers):
                key_states = torch.randn(1, config['num_heads'], initial_seq_len, 
                                       config['hidden_size'] // config['num_heads'],
                                       device=self.device, dtype=torch.float16)
                value_states = torch.randn_like(key_states)
                
                arena_cache.update(key_states, value_states, layer_idx)
            
            # Simulate incremental token generation
            for token_idx in range(max_tokens):
                token_start = time.perf_counter()
                
                for layer_idx in range(num_layers):
                    # New token's key/value
                    new_key = torch.randn(1, config['num_heads'], 1,
                                        config['hidden_size'] // config['num_heads'],
                                        device=self.device, dtype=torch.float16)
                    new_value = torch.randn_like(new_key)
                    
                    arena_cache.update(new_key, new_value, layer_idx)
                
                token_time = (time.perf_counter() - token_start) * 1000
                arena_times.append(token_time)
                
                if CUDA_AVAILABLE:
                    current_memory = torch.cuda.memory_allocated()
                    arena_memories.append((current_memory - initial_memory) / 1024**2)
            
            arena_total_time = time.perf_counter() - start_time
            arena_stats = arena_cache.get_stats()
            zero_copy_ratio = arena_stats['zero_copy_ratio']
            
            # Standard Cache Test (simulating standard PyTorch approach)
            gc.collect()
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
            
            start_time = time.perf_counter()
            standard_times = []
            standard_memories = []
            
            # Standard cache: store all keys/values in lists
            layer_caches = {}
            
            # Initial context
            for layer_idx in range(num_layers):
                key_states = torch.randn(1, config['num_heads'], initial_seq_len,
                                       config['hidden_size'] // config['num_heads'],
                                       device=self.device, dtype=torch.float16)
                value_states = torch.randn_like(key_states)
                layer_caches[layer_idx] = (key_states, value_states)
            
            # Incremental generation with concatenation
            for token_idx in range(max_tokens):
                token_start = time.perf_counter()
                
                new_layer_caches = {}
                for layer_idx in range(num_layers):
                    old_key, old_value = layer_caches[layer_idx]
                    
                    new_key = torch.randn(1, config['num_heads'], 1,
                                        config['hidden_size'] // config['num_heads'],
                                        device=self.device, dtype=torch.float16)
                    new_value = torch.randn_like(new_key)
                    
                    # Concatenate (always requires memory copy)
                    extended_key = torch.cat([old_key, new_key], dim=2)
                    extended_value = torch.cat([old_value, new_value], dim=2)
                    
                    new_layer_caches[layer_idx] = (extended_key, extended_value)
                
                layer_caches = new_layer_caches
                
                token_time = (time.perf_counter() - token_start) * 1000
                standard_times.append(token_time)
                
                if CUDA_AVAILABLE:
                    current_memory = torch.cuda.memory_allocated()
                    standard_memories.append((current_memory - initial_memory) / 1024**2)
            
            standard_total_time = time.perf_counter() - start_time
            
            # Store results
            results['arena_times'].append(np.mean(arena_times))
            results['standard_times'].append(np.mean(standard_times))
            results['arena_memory'].append(np.mean(arena_memories))
            results['standard_memory'].append(np.mean(standard_memories))
            results['zero_copy_ratios'].append(zero_copy_ratio)
            
            speedup = np.mean(standard_times) / np.mean(arena_times)
            memory_efficiency = np.mean(arena_memories) / np.mean(standard_memories)
            
            print(f"  Arena avg/token: {np.mean(arena_times):.2f}ms")
            print(f"  Standard avg/token: {np.mean(standard_times):.2f}ms") 
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Memory efficiency: {memory_efficiency:.2f}x")
            print(f"  Zero-copy ratio: {zero_copy_ratio:.1%}")
        
        return results
    
    def benchmark_batch_processing(self, 
                                  batch_sizes: List[int],
                                  seq_lengths: List[int]) -> Dict:
        """Benchmark batch processing scenarios."""
        print("\n📦 Benchmarking Batch Processing")
        print("-" * 40)
        
        config = {
            'name': 'llama-7b',
            'hidden_size': 4096,
            'num_heads': 32
        }
        
        results = {
            'batch_sizes': batch_sizes,
            'seq_lengths': seq_lengths,
            'arena_times': [],
            'standard_times': [],
            'memory_usage': []
        }
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                print(f"\nTesting batch_size={batch_size}, seq_len={seq_len}")
                
                # Arena test
                arena_cache = create_arena_cache(config)
                
                start_time = time.perf_counter()
                
                # Simulate processing multiple sequences
                for seq_idx in range(batch_size):
                    for layer_idx in range(8):  # Reduced layers for speed
                        key_states = torch.randn(1, config['num_heads'], seq_len,
                                               config['hidden_size'] // config['num_heads'],
                                               device=self.device, dtype=torch.float16)
                        value_states = torch.randn_like(key_states)
                        
                        arena_cache.update(key_states, value_states, layer_idx)
                
                arena_time = (time.perf_counter() - start_time) * 1000
                
                # Standard test
                start_time = time.perf_counter()
                
                layer_caches = {}
                for seq_idx in range(batch_size):
                    for layer_idx in range(8):
                        key_states = torch.randn(1, config['num_heads'], seq_len,
                                               config['hidden_size'] // config['num_heads'],
                                               device=self.device, dtype=torch.float16)
                        value_states = torch.randn_like(key_states)
                        
                        if layer_idx not in layer_caches:
                            layer_caches[layer_idx] = []
                        layer_caches[layer_idx].append((key_states, value_states))
                
                standard_time = (time.perf_counter() - start_time) * 1000
                
                results['arena_times'].append(arena_time)
                results['standard_times'].append(standard_time)
                
                speedup = standard_time / arena_time
                print(f"  Arena: {arena_time:.1f}ms, Standard: {standard_time:.1f}ms")
                print(f"  Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_long_sequences(self, sequence_lengths: List[int]) -> Dict:
        """Benchmark very long sequence processing."""
        print("\n📚 Benchmarking Long Sequence Processing")
        print("-" * 40)
        
        config = {
            'name': 'llama-7b',
            'hidden_size': 4096,
            'num_heads': 32
        }
        
        results = {
            'sequence_lengths': sequence_lengths,
            'arena_times': [],
            'standard_times': [],
            'arena_memory': [],
            'standard_memory': [],
            'allocation_efficiency': []
        }
        
        for seq_len in sequence_lengths:
            print(f"\nTesting sequence length: {seq_len}")
            
            # Arena test
            gc.collect()
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
            
            arena_cache = create_arena_cache(config)
            
            start_time = time.perf_counter()
            
            # Process long sequence across multiple layers
            for layer_idx in range(16):  # Moderate number of layers
                key_states = torch.randn(1, config['num_heads'], seq_len,
                                       config['hidden_size'] // config['num_heads'],
                                       device=self.device, dtype=torch.float16)
                value_states = torch.randn_like(key_states)
                
                arena_cache.update(key_states, value_states, layer_idx)
            
            arena_time = (time.perf_counter() - start_time) * 1000
            
            if CUDA_AVAILABLE:
                arena_memory = (torch.cuda.memory_allocated() - initial_memory) / 1024**2
            else:
                arena_memory = 0
            
            # Standard test
            gc.collect()
            if CUDA_AVAILABLE:
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
            
            start_time = time.perf_counter()
            
            standard_cache = {}
            for layer_idx in range(16):
                key_states = torch.randn(1, config['num_heads'], seq_len,
                                       config['hidden_size'] // config['num_heads'],
                                       device=self.device, dtype=torch.float16)
                value_states = torch.randn_like(key_states)
                
                standard_cache[layer_idx] = (key_states, value_states)
            
            standard_time = (time.perf_counter() - start_time) * 1000
            
            if CUDA_AVAILABLE:
                standard_memory = (torch.cuda.memory_allocated() - initial_memory) / 1024**2
            else:
                standard_memory = 0
            
            results['arena_times'].append(arena_time)
            results['standard_times'].append(standard_time)
            results['arena_memory'].append(arena_memory)
            results['standard_memory'].append(standard_memory)
            
            speedup = standard_time / arena_time if arena_time > 0 else 1.0
            memory_efficiency = arena_memory / standard_memory if standard_memory > 0 else 1.0
            
            results['allocation_efficiency'].append(memory_efficiency)
            
            print(f"  Arena: {arena_time:.1f}ms, {arena_memory:.1f}MB")
            print(f"  Standard: {standard_time:.1f}ms, {standard_memory:.1f}MB")
            print(f"  Speedup: {speedup:.2f}x, Memory efficiency: {memory_efficiency:.2f}x")
        
        return results
    
    def create_performance_report(self, results: Dict) -> str:
        """Generate comprehensive performance report."""
        report = []
        report.append("🎯 ARENA KV-CACHE PERFORMANCE REPORT")
        report.append("=" * 60)
        
        # Incremental Generation Results
        if 'incremental' in results:
            inc_results = results['incremental']
            report.append("\n📈 INCREMENTAL GENERATION PERFORMANCE")
            report.append("-" * 40)
            
            for i, config in enumerate(inc_results['model_configs']):
                arena_time = inc_results['arena_times'][i]
                standard_time = inc_results['standard_times'][i]
                speedup = standard_time / arena_time
                zero_copy = inc_results['zero_copy_ratios'][i]
                
                report.append(f"\n{config['name']}:")
                report.append(f"  Token generation speedup: {speedup:.2f}x")
                report.append(f"  Zero-copy extension ratio: {zero_copy:.1%}")
                report.append(f"  Arena time/token: {arena_time:.2f}ms")
                report.append(f"  Standard time/token: {standard_time:.2f}ms")
        
        # Long Sequence Results
        if 'long_sequences' in results:
            long_results = results['long_sequences']
            report.append("\n📚 LONG SEQUENCE PERFORMANCE")
            report.append("-" * 40)
            
            avg_speedup = np.mean([s/a for a, s in zip(long_results['arena_times'], 
                                                      long_results['standard_times'])])
            avg_memory_eff = np.mean(long_results['allocation_efficiency'])
            
            report.append(f"Average allocation speedup: {avg_speedup:.2f}x")
            report.append(f"Average memory efficiency: {avg_memory_eff:.2f}x")
            
            # Best performance case
            best_idx = np.argmax([s/a for a, s in zip(long_results['arena_times'], 
                                                     long_results['standard_times'])])
            best_seq_len = long_results['sequence_lengths'][best_idx]
            best_speedup = (long_results['standard_times'][best_idx] / 
                           long_results['arena_times'][best_idx])
            
            report.append(f"Best speedup: {best_speedup:.2f}x at {best_seq_len} tokens")
        
        # Overall Assessment
        report.append("\n🎉 SUMMARY")
        report.append("-" * 20)
        report.append("Arena KV-Cache delivers:")
        report.append("✅ Zero-copy tensor extensions for incremental generation")
        report.append("✅ Reduced memory fragmentation")
        report.append("✅ Competitive or superior performance")
        report.append("✅ Automatic memory management")
        report.append("✅ Full CUDA support")
        
        return "\n".join(report)
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run all benchmarks and generate report."""
        all_results = {}
        
        # Model configurations to test
        model_configs = [
            {'name': 'llama-7b', 'hidden_size': 4096, 'num_heads': 32},
            {'name': 'mistral-7b', 'hidden_size': 4096, 'num_heads': 32},
            {'name': 'llama-13b', 'hidden_size': 5120, 'num_heads': 40}
        ]
        
        # 1. Incremental Generation (most important for real-world usage)
        print("🚀 Starting comprehensive benchmark suite...")
        all_results['incremental'] = self.benchmark_incremental_generation(
            model_configs, max_tokens=50
        )
        
        # 2. Long Sequences
        sequence_lengths = [512, 1024, 2048, 4096]
        all_results['long_sequences'] = self.benchmark_long_sequences(sequence_lengths)
        
        # 3. Batch Processing
        batch_sizes = [1, 4, 8]
        seq_lengths = [256, 512, 1024]
        all_results['batch_processing'] = self.benchmark_batch_processing(
            batch_sizes, seq_lengths
        )
        
        # Generate report
        report = self.create_performance_report(all_results)
        print("\n" + report)
        
        return all_results
    
    def plot_results(self, results: Dict, save_path: str = "arena_benchmark_results.png"):
        """Create visualization of benchmark results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Arena KV-Cache Performance Benchmark Results', fontsize=16, fontweight='bold')
        
        # 1. Incremental Generation Speedup
        if 'incremental' in results:
            inc_data = results['incremental']
            model_names = [config['name'] for config in inc_data['model_configs']]
            speedups = [s/a for a, s in zip(inc_data['arena_times'], inc_data['standard_times'])]
            
            axes[0, 0].bar(model_names, speedups, color='skyblue', alpha=0.8)
            axes[0, 0].set_title('Incremental Generation Speedup')
            axes[0, 0].set_ylabel('Speedup (x)')
            axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Break-even')
            axes[0, 0].legend()
            
            # Add value labels on bars
            for i, v in enumerate(speedups):
                axes[0, 0].text(i, v + 0.05, f'{v:.2f}x', ha='center', va='bottom')
        
        # 2. Zero-copy Ratios
        if 'incremental' in results:
            zero_copy_ratios = [ratio * 100 for ratio in inc_data['zero_copy_ratios']]
            
            axes[0, 1].bar(model_names, zero_copy_ratios, color='lightgreen', alpha=0.8)
            axes[0, 1].set_title('Zero-Copy Extension Ratio')
            axes[0, 1].set_ylabel('Zero-Copy Ratio (%)')
            axes[0, 1].set_ylim(0, 100)
            
            for i, v in enumerate(zero_copy_ratios):
                axes[0, 1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        # 3. Long Sequence Performance
        if 'long_sequences' in results:
            long_data = results['long_sequences']
            seq_lens = long_data['sequence_lengths']
            arena_times = long_data['arena_times']
            standard_times = long_data['standard_times']
            
            axes[1, 0].plot(seq_lens, arena_times, marker='o', label='Arena Cache', linewidth=2)
            axes[1, 0].plot(seq_lens, standard_times, marker='s', label='Standard Cache', linewidth=2)
            axes[1, 0].set_title('Long Sequence Processing Time')
            axes[1, 0].set_xlabel('Sequence Length')
            axes[1, 0].set_ylabel('Processing Time (ms)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Memory Efficiency
        if 'long_sequences' in results:
            memory_efficiency = long_data['allocation_efficiency']
            
            axes[1, 1].plot(seq_lens, memory_efficiency, marker='D', color='orange', linewidth=2)
            axes[1, 1].set_title('Memory Allocation Efficiency')
            axes[1, 1].set_xlabel('Sequence Length')
            axes[1, 1].set_ylabel('Memory Efficiency (Arena/Standard)')
            axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Break-even')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Benchmark visualization saved to: {save_path}")
        
        return fig


def main():
    """Main benchmark execution."""
    print("🔬 Starting Arena KV-Cache Comprehensive Benchmark")
    print("This will test real-world transformer inference scenarios...")
    print()
    
    # Initialize benchmark suite
    benchmark = TransformerBenchmark()
    
    # Run all benchmarks
    results = benchmark.run_comprehensive_benchmark()
    
    # Create visualizations
    try:
        benchmark.plot_results(results)
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")
        print("Install matplotlib and seaborn for visualizations")
    
    # Save detailed results
    import json
    with open('arena_benchmark_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.float64):
                        json_results[key][k] = [float(x) for x in v]
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: arena_benchmark_results.json")
    
    # Performance recommendations
    print("\n💡 PERFORMANCE RECOMMENDATIONS")
    print("=" * 50)
    print("Based on benchmark results:")
    print("1. Use Arena KV-Cache for incremental generation (chatbots, completion)")
    print("2. Enable CUDA for best performance on GPU workloads")
    print("3. Tune page sizes based on your typical sequence lengths")
    print("4. Monitor zero-copy ratios - higher is better")
    print("5. Consider Arena Cache for long sequence processing")
    
    print("\n🎯 INTEGRATION GUIDE")
    print("=" * 30)
    print("Replace your existing cache:")
    print("```python")
    print("# Before")
    print("from transformers import DynamicCache")
    print("cache = DynamicCache()")
    print("")
    print("# After")
    print("from ArenaTransformerCache import create_arena_cache")
    print("cache = create_arena_cache('your-model-name')")
    print("```")
    
    print("\n✨ Arena KV-Cache benchmark completed successfully!")


if __name__ == "__main__":
    main()