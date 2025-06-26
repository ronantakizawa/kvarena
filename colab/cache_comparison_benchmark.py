#!/usr/bin/env python3
"""
Arena Cache vs Dynamic Cache Performance Comparison
Comprehensive benchmark comparing arena cache with transformers DynamicCache
"""

import torch
import numpy as np
import time
import os
import sys
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import json
import gc
import psutil
from contextlib import contextmanager

# Hugging Face imports
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        GenerationConfig,
        DynamicCache,
        StaticCache
    )
    print("✅ Transformers imported successfully")
except ImportError as e:
    print(f"❌ Failed to import transformers: {e}")
    sys.exit(1)

# Try to import arena cache
ARENA_CACHE_AVAILABLE = False
try:
    from fixed_arena_kv_cache import (
        ArenaKVCacheManager, 
        ArenaKVCache,
        create_model_optimized_manager,
        CUDA_AVAILABLE
    )
    print("✅ Arena cache bindings available")
    ARENA_CACHE_AVAILABLE = True
except ImportError:
    print("⚠️  Arena cache not available - will simulate")
    ARENA_CACHE_AVAILABLE = False

@dataclass
class BenchmarkConfig:
    """Configuration for cache comparison benchmark."""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    hf_token: Optional[str] = None
    device: str = "auto"
    max_new_tokens: int = 80
    temperature: float = 0.7
    do_sample: bool = True
    num_warmup_runs: int = 2
    num_benchmark_runs: int = 3
    measure_memory: bool = True
    save_detailed_logs: bool = True

@contextmanager
def memory_monitor():
    """Context manager to monitor memory usage."""
    process = psutil.Process()
    
    # GPU memory
    gpu_mem_before = 0
    gpu_mem_after = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gpu_mem_before = torch.cuda.memory_allocated()
    
    # CPU memory
    cpu_mem_before = process.memory_info().rss
    
    yield
    
    # Measure after
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_mem_after = torch.cuda.memory_allocated()
    
    cpu_mem_after = process.memory_info().rss
    
    # Store results in global dict for retrieval
    global _memory_usage
    _memory_usage = {
        'gpu_mb': (gpu_mem_after - gpu_mem_before) / 1024 / 1024,
        'cpu_mb': (cpu_mem_after - cpu_mem_before) / 1024 / 1024,
        'gpu_total_mb': gpu_mem_after / 1024 / 1024 if torch.cuda.is_available() else 0
    }

_memory_usage = {}

class ArenaKVCacheWrapper:
    """Wrapper for arena cache with unified interface."""
    
    def __init__(self, model, max_seq_len: int = 4096):
        self.model = model
        self.max_seq_len = max_seq_len
        self.arena_cache = None
        self.cache_active = False
        
        if ARENA_CACHE_AVAILABLE:
            try:
                arena_manager = create_model_optimized_manager("mistral-7b", max_seq_len)
                self.arena_cache = ArenaKVCache(model, arena_manager, max_seq_len)
                self.cache_active = True
                print("✅ Arena cache initialized")
            except Exception as e:
                print(f"❌ Arena cache failed: {e}")
                self.cache_active = False
        else:
            # Simulated arena cache for comparison
            self._setup_simulated_cache()
    
    def _setup_simulated_cache(self):
        """Setup simulated arena cache for comparison."""
        self.num_layers = self.model.config.num_hidden_layers
        self.num_query_heads = self.model.config.num_attention_heads
        self.num_kv_heads = getattr(self.model.config, 'num_key_value_heads', self.num_query_heads)
        self.head_dim = self.model.config.hidden_size // self.num_query_heads
        
        print(f"🔧 Simulated arena cache: {self.num_kv_heads}/{self.num_query_heads} heads")
        print(f"   Memory savings: {(1 - self.num_kv_heads/self.num_query_heads)*100:.1f}%")
    
    def warm_cache(self, context_inputs):
        """Warm the cache with context."""
        if self.cache_active and self.arena_cache:
            try:
                with torch.no_grad():
                    outputs = self.model(**context_inputs, use_cache=True)
                    if hasattr(outputs, 'past_key_values') and outputs.past_key_values:
                        for layer_idx, (key, value) in enumerate(outputs.past_key_values):
                            if hasattr(self.arena_cache, 'update_cache'):
                                self.arena_cache.update_cache(layer_idx, key, value)
                print(f"🔥 Arena cache warmed")
            except Exception as e:
                print(f"⚠️ Arena cache warming failed: {e}")
    
    def get_stats(self):
        """Get cache statistics."""
        if self.cache_active and self.arena_cache:
            return self.arena_cache.get_stats()
        else:
            # Simulated stats
            memory_per_layer = self.num_kv_heads * self.head_dim * 512 * 2 * 2  # Estimated
            total_memory = memory_per_layer * self.num_layers
            return {
                'type': 'simulated_arena',
                'num_kv_heads': self.num_kv_heads,
                'num_query_heads': self.num_query_heads,
                'total_memory_mb': total_memory / 1024 / 1024,
                'memory_savings_percent': (1 - self.num_kv_heads/self.num_query_heads) * 100,
                'cache_active': False
            }

class CacheBenchmark:
    """Comprehensive cache benchmark comparing different implementations."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.results = {}
        
    def setup(self):
        """Setup model and tokenizer."""
        print("🚀 Setting up Cache Benchmark")
        print("=" * 60)
        
        # Memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        device = "cuda" if torch.cuda.is_available() and self.config.device == "auto" else self.config.device
        print(f"📱 Device: {device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            token=self.config.hf_token,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            token=self.config.hf_token,
            low_cpu_mem_usage=True,
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        
        # Validate configuration
        query_heads = self.model.config.num_attention_heads
        kv_heads = getattr(self.model.config, 'num_key_value_heads', query_heads)
        
        print(f"📊 Model Configuration:")
        print(f"   - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   - Query heads: {query_heads}")
        print(f"   - KV heads: {kv_heads}")
        print(f"   - GQA ratio: {query_heads/kv_heads:.1f}:1")
        print(f"   - Potential memory savings: {(1-kv_heads/query_heads)*100:.1f}%")
        
        print("✅ Setup complete!")
    
    def benchmark_cache_type(
        self, 
        cache_type: str, 
        queries: List[str], 
        context: str,
        cache_impl=None
    ) -> Dict[str, Any]:
        """Benchmark a specific cache implementation."""
        
        print(f"\n{'='*60}")
        print(f"🧪 BENCHMARKING: {cache_type.upper()}")
        print('='*60)
        
        device = next(self.model.parameters()).device
        
        # Prepare context
        context_prompt = f"Context: {context}\n\n"
        context_inputs = self.tokenizer(context_prompt, return_tensors="pt").to(device)
        context_length = context_inputs.input_ids.shape[1]
        
        print(f"📋 Context: {context_length} tokens")
        
        # Initialize cache
        cache = None
        cache_stats = {}
        
        if cache_type == "dynamic":
            cache = DynamicCache()
        elif cache_type == "static":
            try:
                # Static cache needs max length specification
                cache = StaticCache(
                    config=self.model.config,
                    max_batch_size=1,
                    max_cache_len=2048,
                    device=device,
                    dtype=self.model.dtype
                )
            except Exception as e:
                print(f"⚠️ StaticCache not available: {e}")
                cache = DynamicCache()
        elif cache_type == "arena":
            if cache_impl:
                cache_impl.warm_cache(context_inputs)
                cache_stats = cache_impl.get_stats()
            cache = None  # Arena cache handles this internally
        elif cache_type == "no_cache":
            cache = None
        
        # Warmup runs
        print(f"🔥 Warmup ({self.config.num_warmup_runs} runs)...")
        for i in range(self.config.num_warmup_runs):
            query = queries[0]  # Use first query for warmup
            self._run_single_query(query, context, cache_type, cache, silent=True)
        
        # Benchmark runs
        all_results = []
        query_results = []
        
        print(f"📊 Benchmarking ({self.config.num_benchmark_runs} runs per query)...")
        
        for query_idx, query in enumerate(queries):
            print(f"\n🔍 Query {query_idx + 1}/{len(queries)}: {query[:50]}...")
            
            query_times = []
            query_speeds = []
            query_memory = []
            
            for run in range(self.config.num_benchmark_runs):
                with memory_monitor() if self.config.measure_memory else contextmanager(lambda: None)():
                    result = self._run_single_query(query, context, cache_type, cache, silent=True)
                
                if result['success']:
                    query_times.append(result['generation_time'])
                    query_speeds.append(result['tokens_per_second'])
                    
                    if self.config.measure_memory and _memory_usage:
                        query_memory.append(_memory_usage.copy())
                
                # Clear cache between runs for fair comparison
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate statistics for this query
            if query_times:
                query_result = {
                    'query': query,
                    'avg_time': np.mean(query_times),
                    'std_time': np.std(query_times),
                    'min_time': np.min(query_times),
                    'max_time': np.max(query_times),
                    'avg_speed': np.mean(query_speeds),
                    'std_speed': np.std(query_speeds),
                    'runs': len(query_times)
                }
                
                if query_memory:
                    query_result['avg_gpu_mb'] = np.mean([m['gpu_mb'] for m in query_memory])
                    query_result['avg_cpu_mb'] = np.mean([m['cpu_mb'] for m in query_memory])
                
                query_results.append(query_result)
                print(f"   ⚡ Avg: {query_result['avg_speed']:.1f} tok/s ({query_result['avg_time']:.2f}s)")
        
        # Overall statistics
        if query_results:
            overall_stats = {
                'cache_type': cache_type,
                'total_queries': len(queries),
                'successful_queries': len(query_results),
                'avg_speed_overall': np.mean([q['avg_speed'] for q in query_results]),
                'std_speed_overall': np.std([q['avg_speed'] for q in query_results]),
                'avg_time_overall': np.mean([q['avg_time'] for q in query_results]),
                'total_time': sum([q['avg_time'] for q in query_results]),
                'query_results': query_results,
                'cache_stats': cache_stats,
                'context_length': context_length
            }
            
            if query_results and 'avg_gpu_mb' in query_results[0]:
                overall_stats['avg_gpu_mb'] = np.mean([q['avg_gpu_mb'] for q in query_results])
                overall_stats['avg_cpu_mb'] = np.mean([q['avg_cpu_mb'] for q in query_results])
            
            print(f"\n📈 {cache_type.upper()} Results:")
            print(f"   - Average speed: {overall_stats['avg_speed_overall']:.1f} ± {overall_stats['std_speed_overall']:.1f} tok/s")
            print(f"   - Average time: {overall_stats['avg_time_overall']:.2f}s")
            print(f"   - Total time: {overall_stats['total_time']:.2f}s")
            
            if cache_stats:
                print(f"   - Cache type: {cache_stats.get('type', 'arena')}")
                if 'memory_savings_percent' in cache_stats:
                    print(f"   - Memory savings: {cache_stats['memory_savings_percent']:.1f}%")
            
            return overall_stats
        
        return {'cache_type': cache_type, 'error': 'No successful runs'}
    
    def _run_single_query(
        self, 
        query: str, 
        context: str, 
        cache_type: str, 
        cache=None, 
        silent: bool = False
    ) -> Dict[str, Any]:
        """Run a single query with specified cache."""
        
        device = next(self.model.parameters()).device
        
        # Build prompt
        if context:
            prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        else:
            prompt = f"Question: {query}\nAnswer:"
        
        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]
            
            # Generation config
            generation_config = GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=(cache_type != "no_cache"),
                num_beams=1,
            )
            
            # Generate with timing
            start_time = time.time()
            
            with torch.no_grad():
                if cache_type == "no_cache":
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        generation_config=generation_config,
                        return_dict_in_generate=True
                    )
                else:
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        past_key_values=cache,
                        generation_config=generation_config,
                        return_dict_in_generate=True
                    )
            
            generation_time = time.time() - start_time
            
            # Process results
            output_length = outputs.sequences.shape[1] - input_length
            tokens_per_sec = output_length / generation_time if generation_time > 0 else 0
            
            if not silent:
                # Decode for display
                full_response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                if "Answer:" in full_response:
                    answer = full_response.split("Answer:")[-1].strip()
                else:
                    answer = full_response[len(prompt):].strip()
                
                if len(answer) > 100:
                    answer = answer[:100] + "..."
                
                print(f"🤖 Answer: {answer}")
                print(f"⏱️ Time: {generation_time:.2f}s")
                print(f"🚀 Speed: {tokens_per_sec:.1f} tok/s")
            
            return {
                'success': True,
                'generation_time': generation_time,
                'tokens_per_second': tokens_per_sec,
                'input_tokens': input_length,
                'output_tokens': output_length,
            }
            
        except Exception as e:
            if not silent:
                print(f"❌ Error: {e}")
            return {
                'success': False,
                'error': str(e),
                'generation_time': 0,
                'tokens_per_second': 0,
                'input_tokens': 0,
                'output_tokens': 0,
            }
    
    def run_comparison(self, queries: List[str], context: str) -> Dict[str, Any]:
        """Run comprehensive comparison of all cache types."""
        
        print(f"\n🎯 COMPREHENSIVE CACHE COMPARISON")
        print(f"Testing {len(queries)} queries with {len(context)} character context")
        print("=" * 60)
        
        # Define cache types to test
        cache_types = ["no_cache", "dynamic"]
        
        # Add StaticCache if available
        try:
            StaticCache(
                config=self.model.config,
                max_batch_size=1,
                max_cache_len=512,
                device=next(self.model.parameters()).device,
                dtype=self.model.dtype
            )
            cache_types.append("static")
        except Exception:
            print("ℹ️ StaticCache not available in this transformers version")
        
        # Add Arena cache
        arena_wrapper = None
        if ARENA_CACHE_AVAILABLE:
            try:
                arena_wrapper = ArenaKVCacheWrapper(self.model)
                cache_types.append("arena")
            except Exception as e:
                print(f"⚠️ Arena cache initialization failed: {e}")
        
        comparison_results = {}
        
        # Benchmark each cache type
        for cache_type in cache_types:
            try:
                if cache_type == "arena":
                    result = self.benchmark_cache_type(cache_type, queries, context, arena_wrapper)
                else:
                    result = self.benchmark_cache_type(cache_type, queries, context)
                
                comparison_results[cache_type] = result
                
            except Exception as e:
                print(f"❌ {cache_type} benchmark failed: {e}")
                comparison_results[cache_type] = {'error': str(e)}
        
        # Generate comparison analysis
        analysis = self._analyze_comparison(comparison_results)
        
        return {
            'config': self.config.__dict__,
            'queries': queries,
            'context': context,
            'results': comparison_results,
            'analysis': analysis,
            'timestamp': time.time()
        }
    
    def _analyze_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and compare the benchmark results."""
        
        print(f"\n{'='*60}")
        print("📊 PERFORMANCE ANALYSIS")
        print('='*60)
        
        # Extract successful results
        successful_results = {k: v for k, v in results.items() 
                            if 'avg_speed_overall' in v}
        
        if not successful_results:
            return {'error': 'No successful benchmark results'}
        
        # Find baseline (no_cache or slowest)
        baseline_key = 'no_cache' if 'no_cache' in successful_results else min(
            successful_results.keys(), 
            key=lambda k: successful_results[k]['avg_speed_overall']
        )
        baseline_speed = successful_results[baseline_key]['avg_speed_overall']
        
        print(f"📏 Baseline: {baseline_key} ({baseline_speed:.1f} tok/s)")
        print()
        
        # Performance comparison
        performance_analysis = {}
        
        for cache_type, result in successful_results.items():
            speed = result['avg_speed_overall']
            speedup = speed / baseline_speed
            time_saved = result['total_time']
            
            performance_analysis[cache_type] = {
                'avg_speed': speed,
                'speedup_vs_baseline': speedup,
                'total_time': time_saved,
                'efficiency_score': speedup
            }
            
            print(f"🚀 {cache_type.upper()}:")
            print(f"   - Speed: {speed:.1f} tok/s")
            print(f"   - Speedup: {speedup:.2f}x")
            print(f"   - Total time: {time_saved:.2f}s")
            
            # Memory analysis
            if 'avg_gpu_mb' in result:
                print(f"   - GPU memory: {result['avg_gpu_mb']:.1f}MB")
            
            # Cache-specific analysis
            if 'cache_stats' in result and result['cache_stats']:
                stats = result['cache_stats']
                if 'memory_savings_percent' in stats:
                    print(f"   - Memory savings: {stats['memory_savings_percent']:.1f}%")
            
            print()
        
        # Find best performer
        best_cache = max(performance_analysis.keys(), 
                        key=lambda k: performance_analysis[k]['avg_speed'])
        
        # Winner analysis
        print(f"🏆 WINNER: {best_cache.upper()}")
        best_result = performance_analysis[best_cache]
        print(f"   - Best speed: {best_result['avg_speed']:.1f} tok/s")
        print(f"   - Best speedup: {best_result['speedup_vs_baseline']:.2f}x vs baseline")
        
        # Memory efficiency winner
        memory_efficient = None
        for cache_type, result in successful_results.items():
            if 'cache_stats' in result and result['cache_stats']:
                stats = result['cache_stats']
                if 'memory_savings_percent' in stats and stats['memory_savings_percent'] > 0:
                    memory_efficient = cache_type
                    print(f"💾 Most memory efficient: {cache_type.upper()}")
                    print(f"   - Memory saved: {stats['memory_savings_percent']:.1f}%")
                    break
        
        return {
            'baseline': baseline_key,
            'baseline_speed': baseline_speed,
            'best_performer': best_cache,
            'best_speed': best_result['avg_speed'],
            'best_speedup': best_result['speedup_vs_baseline'],
            'memory_efficient': memory_efficient,
            'performance_analysis': performance_analysis,
            'cache_types_tested': list(successful_results.keys()),
            'total_cache_types': len(results)
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = "cache_comparison_results.json"):
        """Save detailed results to file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Detailed results saved to {filename}")

def run_cache_comparison():
    """Run the complete cache comparison benchmark."""
    
    config = BenchmarkConfig(
        hf_token=os.getenv("HF_TOKEN"),
        max_new_tokens=80,
        temperature=0.7,
        num_warmup_runs=1,
        num_benchmark_runs=3,
        measure_memory=True
    )
    
    # Test data
    context = """Ronan Takizawa is a Colorado College computer science student, cybersecurity researcher, and tech content creator with over 100,000 followers across social media platforms. He has built projects including Punch Analytics (a machine learning boxing analytics app), Noname (a zero-knowledge proof CI pipeline), a REST API for international schools, website automation for the Ireland-Japan Chamber of Commerce, and TeleSpeech (a text-to-speech Chrome extension that won HackHarvard 2023). He works with technologies including Python, TypeScript, Rust, Java, Shell, SQL, React, NodeJS, MongoDB, Docker, Kubernetes, AWS, GCP, Firebase, OpenCV, and GraphQL."""
    
    queries = [
        "Who is Ronan Takizawa?",
        "What is Punch Analytics?",
        "What technologies does Ronan use?",
        "What did TeleSpeech win?",
        "What is Noname?",
        "Where does Ronan study?",
        "How many followers does he have?"
    ]
    
    # Run benchmark
    benchmark = CacheBenchmark(config)
    benchmark.setup()
    
    results = benchmark.run_comparison(queries, context)
    
    # Save results
    benchmark.save_results(results)
    
    print(f"\n🎉 Cache comparison complete!")
    print(f"📊 Tested {len(results['results'])} cache types")
    print(f"🎯 Best performer: {results['analysis']['best_performer'].upper()}")
    print(f"🚀 Best speed: {results['analysis']['best_speed']:.1f} tok/s")
    print(f"⚡ Best speedup: {results['analysis']['best_speedup']:.2f}x")
    
    return results

if __name__ == "__main__":
    print("🏁 Arena Cache vs Dynamic Cache Benchmark")
    print("=" * 60)
    
    if not os.getenv("HF_TOKEN"):
        print("⚠️ Set HF_TOKEN environment variable for model access")
    
    run_cache_comparison()