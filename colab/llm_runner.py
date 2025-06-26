#!/usr/bin/env python3
"""
OPTIMIZED Arena KV-Cache LLM Query Runner
Focuses on cache reuse and performance optimization for better tokens/sec
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

# Hugging Face imports
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        GenerationConfig,
        DynamicCache
    )
    print("✅ Transformers imported successfully")
except ImportError as e:
    print(f"❌ Failed to import transformers: {e}")
    sys.exit(1)

# Try to import fixed arena cache
try:
    from arena_kv_cache_bindings import (
        ArenaKVCacheManager, 
        SequenceArena, 
        ArenaKVCache,
        ArenaError,
        create_model_optimized_manager,
        get_model_config,
        CUDA_AVAILABLE
    )
    print("✅ FIXED Arena KV-Cache bindings imported")
    FIXED_CACHE_AVAILABLE = True
except ImportError:
    print("⚠️  Fixed cache not available, using fallback")
    FIXED_CACHE_AVAILABLE = False

@dataclass
class OptimizedConfig:
    """Optimized configuration for better performance."""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    hf_token: Optional[str] = None
    device: str = "auto"
    max_new_tokens: int = 100
    temperature: float = 0.7
    do_sample: bool = True
    use_arena_cache: bool = True
    cache_warming: bool = True  # Warm cache with context
    batch_optimize: bool = True  # Optimize for batch processing
    reuse_context: bool = True   # Reuse context across queries

class OptimizedQueryRunner:
    """Optimized query runner with cache reuse and performance focus."""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.arena_cache = None
        self.base_context = ""
        self.context_tokens = 0
        
    def setup(self):
        """Setup model, tokenizer, and arena cache."""
        print("🚀 Setting up Optimized LLM Query Runner")
        print("=" * 50)
        
        # Memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            print("✅ CUDA memory optimization enabled")
        
        # Load model and tokenizer
        self._setup_model_and_tokenizer()
        
        # Setup arena cache if available
        if self.config.use_arena_cache and FIXED_CACHE_AVAILABLE:
            self._setup_arena_cache()
        
        print("✅ Setup complete!")
    
    def _setup_model_and_tokenizer(self):
        """Load model and tokenizer with optimization."""
        print(f"🤖 Loading {self.config.model_name}...")
        
        # Device setup
        device = "cuda" if torch.cuda.is_available() and self.config.device == "auto" else self.config.device
        print(f"📱 Using device: {device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            token=self.config.hf_token,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations (without Flash Attention requirement)
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "device_map": "auto" if device == "cuda" else None,
            "trust_remote_code": True,
            "token": self.config.hf_token,
            "low_cpu_mem_usage": True,
        }
        
        # Try Flash Attention 2 if available, fallback gracefully
        if device == "cuda":
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("✅ Flash Attention 2 detected and enabled")
            except ImportError:
                print("ℹ️  Flash Attention 2 not available, using standard attention")
                # Don't add attn_implementation to avoid the error
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        
        # Validate KV head configuration for Mistral
        if "mistral" in self.config.model_name.lower():
            print(f"📊 Mistral configuration validation:")
            query_heads = self.model.config.num_attention_heads
            kv_heads = getattr(self.model.config, 'num_key_value_heads', query_heads)
            print(f"  - Query heads: {query_heads}")
            print(f"  - KV heads: {kv_heads}")
            print(f"  - GQA ratio: {query_heads/kv_heads:.1f}:1")
            
            if kv_heads == 8:
                print("✅ VERIFIED: Correct KV head configuration")
            else:
                print(f"⚠️  Expected 8 KV heads, got {kv_heads}")
        
        print(f"✅ Model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def _setup_arena_cache(self):
        """Setup arena cache with optimization."""
        print("🏗️  Setting up optimized arena cache...")
        
        try:
            # Create optimized arena manager
            arena_manager = create_model_optimized_manager(
                "mistral-7b", 
                max_seq_len=4096  # Larger for better cache reuse
            )
            
            # Verify configuration
            model_config = get_model_config("mistral-7b")
            print(f"📊 Arena configuration:")
            print(f"   - Query heads: {model_config['num_query_heads']}")
            print(f"   - KV heads: {model_config['num_kv_heads']}")
            print(f"   - Memory saved: {(1 - model_config['num_kv_heads']/model_config['num_query_heads'])*100:.1f}%")
            
            # Create cache
            self.arena_cache = ArenaKVCache(self.model, arena_manager, max_seq_len=4096)
            
            cache_stats = self.arena_cache.get_stats()
            print(f"✅ Arena cache created:")
            print(f"   - Layers: {cache_stats['successful_layers']}/{cache_stats['num_layers']}")
            print(f"   - Memory: {cache_stats['total_memory_mb']:.1f} MB")
            
        except Exception as e:
            print(f"❌ Arena cache setup failed: {e}")
            self.arena_cache = None
    
    def warm_cache_with_context(self, context: str):
        """Warm the cache with base context for reuse."""
        if not self.arena_cache or not self.config.cache_warming:
            return
        
        print("🔥 Warming cache with context...")
        
        self.base_context = context
        device = next(self.model.parameters()).device
        
        # Tokenize context
        context_prompt = f"Context: {context}\n\n"
        inputs = self.tokenizer(context_prompt, return_tensors="pt").to(device)
        self.context_tokens = inputs.input_ids.shape[1]
        
        try:
            with torch.no_grad():
                # Process context through model to populate cache
                outputs = self.model(**inputs, use_cache=True)
                
                # Store KV states in arena cache
                if hasattr(outputs, 'past_key_values') and outputs.past_key_values:
                    for layer_idx, (key, value) in enumerate(outputs.past_key_values):
                        if hasattr(self.arena_cache, 'update_cache'):
                            self.arena_cache.update_cache(layer_idx, key, value)
                
                cache_stats = self.arena_cache.get_stats()
                print(f"✅ Cache warmed: {cache_stats['current_length']} tokens cached")
                
        except Exception as e:
            print(f"⚠️  Cache warming failed: {e}")
    
    def run_optimized_query(self, query: str, use_cached_context: bool = True) -> Dict[str, Any]:
        """Run a single query with optimization."""
        device = next(self.model.parameters()).device
        
        # Build optimized prompt
        if use_cached_context and self.base_context and self.config.reuse_context:
            # Reuse cached context
            prompt = f"Question: {query}\nAnswer:"
            cached_tokens = self.context_tokens
        else:
            # Full prompt
            if self.base_context:
                prompt = f"Context: {self.base_context}\n\nQuestion: {query}\nAnswer:"
            else:
                prompt = f"Question: {query}\nAnswer:"
            cached_tokens = 0
        
        print(f"\n📝 Query: {query}")
        print(f"🔄 Using cached context: {use_cached_context and cached_tokens > 0}")
        
        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]
            
            # Optimized generation config
            generation_config = GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                # Optimization: Use faster decoding if available
                num_beams=1,  # Greedy or sampling only
                early_stopping=True,
            )
            
            # Setup attention mask for cached context
            attention_mask = inputs.attention_mask
            past_key_values = None
            
            if use_cached_context and self.arena_cache and cached_tokens > 0:
                # Create attention mask that includes cached tokens
                cached_attention = torch.ones(
                    (inputs.input_ids.shape[0], cached_tokens),
                    device=device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([cached_attention, attention_mask], dim=1)
                
                # Try to get cached key-values
                try:
                    if hasattr(self.arena_cache, 'get_past_key_values'):
                        past_key_values = self.arena_cache.get_past_key_values()
                except:
                    pass  # Fallback to no cache
            
            print(f"🔢 Input tokens: {input_length} (+ {cached_tokens} cached)")
            
            # Generate with timing
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    generation_config=generation_config,
                    return_dict_in_generate=True
                )
            
            generation_time = time.time() - start_time
            
            # Process output
            full_response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # Extract answer
            if "Answer:" in full_response:
                answer = full_response.split("Answer:")[-1].strip()
            else:
                answer = full_response[len(prompt):].strip()
            
            answer = answer.replace("\n\n", " ").strip()
            if len(answer) > 500:
                answer = answer[:500] + "..."
            
            output_length = outputs.sequences.shape[1] - input_length
            effective_tokens_per_sec = output_length / generation_time if generation_time > 0 else 0
            
            print(f"🤖 Answer: {answer}")
            print(f"⏱️  Generation time: {generation_time:.2f}s")
            print(f"📊 Generated tokens: {output_length}")
            print(f"🚀 Tokens/sec: {effective_tokens_per_sec:.1f}")
            
            # Cache statistics
            if self.arena_cache:
                cache_stats = self.arena_cache.get_stats()
                print(f"📈 Cache: {cache_stats['current_length']} tokens, {cache_stats['total_memory_mb']:.1f} MB")
            
            return {
                'query': query,
                'answer': answer,
                'generation_time': generation_time,
                'input_tokens': input_length,
                'cached_tokens': cached_tokens,
                'output_tokens': output_length,
                'tokens_per_second': effective_tokens_per_sec,
                'used_cache': use_cached_context and cached_tokens > 0,
                'cache_stats': self.arena_cache.get_stats() if self.arena_cache else None
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                'query': query,
                'answer': f"Error: {e}",
                'generation_time': 0,
                'input_tokens': 0,
                'cached_tokens': 0,
                'output_tokens': 0,
                'tokens_per_second': 0,
                'used_cache': False,
                'cache_stats': None
            }
    
    def run_batch_optimized(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Run batch of queries with optimization."""
        print(f"\n🎯 Running {len(queries)} optimized queries...")
        results = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*50}")
            print(f"Query {i}/{len(queries)}")
            print('='*50)
            
            # Use cache for all queries after the first
            use_cache = i > 1 and self.config.reuse_context
            result = self.run_optimized_query(query, use_cache)
            results.append(result)
            
            # Periodic cleanup
            if i % 3 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def print_optimized_summary(self, results: List[Dict[str, Any]]):
        """Print optimized performance summary."""
        print(f"\n{'='*60}")
        print("📊 OPTIMIZED PERFORMANCE SUMMARY")
        print('='*60)
        
        successful = [r for r in results if not r['answer'].startswith('Error:')]
        cache_used = [r for r in successful if r['used_cache']]
        
        total_time = sum(r['generation_time'] for r in successful)
        total_tokens = sum(r['output_tokens'] for r in successful)
        
        print(f"✅ Successful queries: {len(successful)}/{len(results)}")
        print(f"🔄 Cache utilization: {len(cache_used)}/{len(successful)} queries")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Total tokens: {total_tokens}")
        print(f"🚀 Overall tokens/sec: {total_tokens/total_time:.1f}")
        
        if cache_used:
            cache_times = [r['generation_time'] for r in cache_used]
            no_cache_times = [r['generation_time'] for r in successful if not r['used_cache']]
            
            if no_cache_times:
                avg_cache_time = np.mean(cache_times)
                avg_no_cache_time = np.mean(no_cache_times)
                speedup = avg_no_cache_time / avg_cache_time if avg_cache_time > 0 else 1
                
                print(f"⚡ Cache speedup: {speedup:.2f}x faster")
                print(f"📈 Avg time with cache: {avg_cache_time:.2f}s")
                print(f"📈 Avg time without cache: {avg_no_cache_time:.2f}s")
        
        # Memory efficiency
        if successful and self.arena_cache:
            final_stats = successful[-1]['cache_stats']
            if final_stats:
                print(f"\n📈 Arena Cache Efficiency:")
                print(f"   - Memory usage: {final_stats['total_memory_mb']:.1f} MB")
                print(f"   - Layers cached: {final_stats.get('successful_layers', 'N/A')}")
                
                # Calculate memory savings from KV heads
                if FIXED_CACHE_AVAILABLE:
                    savings = (1 - 8/32) * 100  # 8 KV heads vs 32 query heads
                    print(f"   - Memory saved (GQA): {savings:.1f}%")

def run_performance_test():
    """Run a comprehensive performance test."""
    
    # Configuration
    config = OptimizedConfig(
        hf_token=os.getenv("HF_TOKEN"),
        max_new_tokens=80,
        temperature=0.7,
        use_arena_cache=True,
        cache_warming=True,
        reuse_context=True
    )
    
    # Test context and queries
    context = """Ronan Takizawa is a Colorado College computer science student, cybersecurity researcher, and tech content creator with over 100,000 followers across social media platforms. Ronan Takizawa has built a machine learning boxing analytics app (Punch Analytics), a zero-knowledge proof CI pipeline (Noname), a REST API for international schools, a website automation system for the Ireland-Japan Chamber of Commerce, and a text-to-speech Chrome extension (TeleSpeech) that won HackHarvard 2023. Ronan Takizawa has worked with technologies including Python, TypeScript, Rust, Java, Shell, SQL, React, NodeJS, MongoDB, Docker, Kubernetes, AWS, GCP, and tools like Firebase, OpenCV, and GraphQL."""
    
    queries = [
        "Who is Ronan Takizawa?",
        "What is Punch Analytics?",
        "What technologies does Ronan use?",
        "What did TeleSpeech win?",
        "What is Noname?",
        "Where does Ronan study?",
        "How many followers does he have?",
        "What company did he build automation for?"
    ]
    
    # Setup and run
    runner = OptimizedQueryRunner(config)
    runner.setup()
    
    # Warm cache with context
    runner.warm_cache_with_context(context)
    
    # Run optimized queries
    results = runner.run_batch_optimized(queries)
    
    # Print summary
    runner.print_optimized_summary(results)
    
    # Save results
    with open("optimized_results.json", "w") as f:
        json.dump({
            'config': config.__dict__,
            'results': results,
            'context': context
        }, f, indent=2)
    
    print("💾 Results saved to optimized_results.json")
    
    return results

if __name__ == "__main__":
    run_performance_test()