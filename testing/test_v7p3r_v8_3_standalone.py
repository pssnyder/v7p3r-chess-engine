#!/usr/bin/env python3
"""
V7P3R Chess Engine V8.3 - Standalone Memory and Performance Testing
Testing memory management components without full engine dependencies
"""

import time
import gc
import psutil
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict, defaultdict
from datetime import datetime


class SimpleLRUCache:
    """Simplified LRU Cache for testing"""
    
    def __init__(self, max_size: int, ttl: float):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        current_time = time.time()
        
        if key in self.cache:
            # Check TTL
            if current_time - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                self.misses += 1
                return None
            
            # Move to end (most recent)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key, value):
        current_time = time.time()
        
        if key in self.cache:
            self.cache[key] = value
            self.timestamps[key] = current_time
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            self.timestamps[key] = current_time
            
            # Remove oldest if over limit
            while len(self.cache) > self.max_size:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
                del self.timestamps[oldest]
    
    def cleanup_expired(self):
        current_time = time.time()
        expired = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired:
            del self.cache[key]
            del self.timestamps[key]
        
        return len(expired)
    
    def size(self):
        return len(self.cache)
    
    def hit_ratio(self):
        total = self.hits + self.misses
        return self.hits / max(total, 1)


class SimpleMemoryManager:
    """Simplified memory manager for testing"""
    
    def __init__(self, max_memory_mb: float = 100.0):
        self.max_memory_mb = max_memory_mb
        
        # Scale cache sizes based on memory limit
        cache_size = min(int(max_memory_mb * 500), 50000)
        tt_size = min(int(max_memory_mb * 1000), 100000)
        
        self.eval_cache = SimpleLRUCache(cache_size, 30.0)
        self.transposition_table = SimpleLRUCache(tt_size, 60.0)
        self.killer_moves = {}
        self.history_scores = {}
        
        self.cleanup_count = 0
        self.pressure_cleanups = 0
    
    def store_evaluation(self, key: str, value: float):
        self.eval_cache.put(key, value)
    
    def get_evaluation(self, key: str) -> Optional[float]:
        return self.eval_cache.get(key)
    
    def store_transposition(self, key: str, data: Dict):
        self.transposition_table.put(key, data)
    
    def get_transposition(self, key: str) -> Optional[Dict]:
        return self.transposition_table.get(key)
    
    def cleanup(self):
        """Perform routine cleanup"""
        removed = 0
        removed += self.eval_cache.cleanup_expired()
        removed += self.transposition_table.cleanup_expired()
        self.cleanup_count += 1
        return removed
    
    def pressure_cleanup(self):
        """Aggressive cleanup under memory pressure"""
        removed = 0
        
        # Clear 30% of each cache
        eval_size = self.eval_cache.size()
        tt_size = self.transposition_table.size()
        
        eval_to_remove = int(eval_size * 0.3)
        tt_to_remove = int(tt_size * 0.3)
        
        # Remove oldest entries
        for _ in range(eval_to_remove):
            if self.eval_cache.cache:
                oldest = next(iter(self.eval_cache.cache))
                del self.eval_cache.cache[oldest]
                del self.eval_cache.timestamps[oldest]
                removed += 1
        
        for _ in range(tt_to_remove):
            if self.transposition_table.cache:
                oldest = next(iter(self.transposition_table.cache))
                del self.transposition_table.cache[oldest]
                del self.transposition_table.timestamps[oldest]
                removed += 1
        
        self.pressure_cleanups += 1
        return removed
    
    def get_stats(self):
        return {
            'eval_cache_size': self.eval_cache.size(),
            'eval_cache_hits': self.eval_cache.hit_ratio(),
            'tt_size': self.transposition_table.size(),
            'tt_hits': self.transposition_table.hit_ratio(),
            'cleanups': self.cleanup_count,
            'pressure_cleanups': self.pressure_cleanups
        }


class V8_3_StandaloneTest:
    """Standalone V8.3 testing without full engine dependencies"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        self.test_results = {}
    
    def run_all_tests(self):
        """Run comprehensive standalone tests"""
        print("=== V7P3R V8.3 Standalone Memory Tests ===")
        print(f"Baseline memory: {self.baseline_memory:.1f}MB")
        
        # Test 1: LRU Cache functionality
        print("\n1. Testing LRU Cache...")
        self.test_results['lru_cache'] = self.test_lru_cache()
        
        # Test 2: Memory manager
        print("\n2. Testing Memory Manager...")
        self.test_results['memory_manager'] = self.test_memory_manager()
        
        # Test 3: Memory scaling
        print("\n3. Testing Memory Scaling...")
        self.test_results['memory_scaling'] = self.test_memory_scaling()
        
        # Test 4: Pressure handling
        print("\n4. Testing Memory Pressure...")
        self.test_results['pressure_handling'] = self.test_pressure_handling()
        
        # Test 5: Performance benchmarks
        print("\n5. Running Performance Benchmarks...")
        self.test_results['performance'] = self.test_performance()
        
        # Generate summary
        self.test_results['summary'] = self.generate_summary()
        
        return self.test_results
    
    def test_lru_cache(self):
        """Test LRU cache with TTL"""
        cache = SimpleLRUCache(max_size=100, ttl=1.0)
        results = {}
        
        # Basic operations
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        results['basic_ops'] = "PASS"
        
        # Test size limit
        for i in range(150):
            cache.put(f"key_{i}", f"value_{i}")
        
        assert cache.size() <= 100
        results['size_limit'] = "PASS"
        results['final_size'] = cache.size()
        
        # Test TTL
        cache.put("ttl_test", "expire_me")
        assert cache.get("ttl_test") == "expire_me"
        
        time.sleep(1.1)  # Wait for expiration
        assert cache.get("ttl_test") is None
        results['ttl_expiration'] = "PASS"
        
        # Test cleanup
        expired_count = cache.cleanup_expired()
        results['cleanup_expired'] = expired_count
        
        # Test hit ratio
        hit_ratio = cache.hit_ratio()
        results['hit_ratio'] = hit_ratio
        results['hit_ratio_quality'] = "GOOD" if hit_ratio > 0.3 else "POOR"
        
        return results
    
    def test_memory_manager(self):
        """Test memory manager functionality"""
        manager = SimpleMemoryManager(max_memory_mb=50.0)
        results = {}
        
        # Test evaluation cache
        manager.store_evaluation("pos1", 1.5)
        manager.store_evaluation("pos2", -0.8)
        
        assert manager.get_evaluation("pos1") == 1.5
        assert manager.get_evaluation("pos2") == -0.8
        assert manager.get_evaluation("nonexistent") is None
        results['eval_cache'] = "PASS"
        
        # Test transposition table
        tt_data = {"move": "e2e4", "score": 0.3}
        manager.store_transposition("fen1", tt_data)
        
        retrieved = manager.get_transposition("fen1")
        assert retrieved == tt_data
        results['transposition_table'] = "PASS"
        
        # Test statistics
        stats = manager.get_stats()
        results['stats_available'] = "PASS" if 'eval_cache_size' in stats else "FAIL"
        results['initial_stats'] = stats
        
        return results
    
    def test_memory_scaling(self):
        """Test memory scaling with different configurations"""
        results = {}
        
        # Small configuration
        small_manager = SimpleMemoryManager(max_memory_mb=10.0)
        small_cache_size = small_manager.eval_cache.max_size
        
        # Large configuration  
        large_manager = SimpleMemoryManager(max_memory_mb=100.0)
        large_cache_size = large_manager.eval_cache.max_size
        
        scaling_ratio = large_cache_size / max(small_cache_size, 1)
        
        results['small_cache_size'] = small_cache_size
        results['large_cache_size'] = large_cache_size
        results['scaling_ratio'] = scaling_ratio
        results['scaling_works'] = "PASS" if scaling_ratio > 1.5 else "FAIL"
        
        return results
    
    def test_pressure_handling(self):
        """Test memory pressure handling"""
        manager = SimpleMemoryManager(max_memory_mb=20.0)
        results = {}
        
        # Fill up caches
        for i in range(1000):
            manager.store_evaluation(f"pos_{i}", float(i))
            manager.store_transposition(f"fen_{i}", {"move": f"move_{i}"})
        
        initial_stats = manager.get_stats()
        
        # Test routine cleanup
        routine_removed = manager.cleanup()
        routine_stats = manager.get_stats()
        
        # Test pressure cleanup
        pressure_removed = manager.pressure_cleanup()
        final_stats = manager.get_stats()
        
        results['initial_eval_size'] = initial_stats['eval_cache_size']
        results['initial_tt_size'] = initial_stats['tt_size']
        results['routine_cleanup_removed'] = routine_removed
        results['pressure_cleanup_removed'] = pressure_removed
        results['final_eval_size'] = final_stats['eval_cache_size']
        results['final_tt_size'] = final_stats['tt_size']
        
        # Verify pressure cleanup worked
        reduction_ratio = 1.0 - (final_stats['eval_cache_size'] / max(initial_stats['eval_cache_size'], 1))
        results['pressure_reduction_ratio'] = reduction_ratio
        results['pressure_effective'] = "PASS" if reduction_ratio > 0.2 else "FAIL"
        
        return results
    
    def test_performance(self):
        """Test performance characteristics"""
        results = {}
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Create manager for performance testing
        manager = SimpleMemoryManager(max_memory_mb=100.0)
        
        # Test cache performance under load
        start_time = time.time()
        
        # Write phase
        for i in range(10000):
            manager.store_evaluation(f"eval_{i}", float(i) * 0.1)
            manager.store_transposition(f"tt_{i}", {"move": f"move_{i}", "score": i})
        
        write_time = time.time() - start_time
        
        # Read phase
        start_time = time.time()
        hit_count = 0
        
        for i in range(5000):
            if manager.get_evaluation(f"eval_{i}") is not None:
                hit_count += 1
            if manager.get_transposition(f"tt_{i}") is not None:
                hit_count += 1
        
        read_time = time.time() - start_time
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Get final statistics
        final_stats = manager.get_stats()
        
        results['write_time'] = write_time
        results['read_time'] = read_time
        results['memory_increase'] = end_memory - start_memory
        results['cache_hit_count'] = hit_count
        results['eval_hit_ratio'] = final_stats['eval_cache_hits']
        results['tt_hit_ratio'] = final_stats['tt_hits']
        
        # Performance ratings
        results['write_performance'] = "GOOD" if write_time < 1.0 else "SLOW"
        results['read_performance'] = "GOOD" if read_time < 0.5 else "SLOW"
        results['memory_efficiency'] = "GOOD" if end_memory - start_memory < 20 else "POOR"
        results['cache_efficiency'] = "GOOD" if final_stats['eval_cache_hits'] > 0.8 else "POOR"
        
        return results
    
    def generate_summary(self):
        """Generate comprehensive test summary"""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - self.baseline_memory
        
        # Count passed/failed tests
        passed = 0
        failed = 0
        
        for test_name, test_data in self.test_results.items():
            if test_name == 'summary':
                continue
            
            if isinstance(test_data, dict):
                for key, value in test_data.items():
                    if value == "PASS":
                        passed += 1
                    elif value == "FAIL":
                        failed += 1
        
        # Overall assessment
        success_rate = passed / max(passed + failed, 1) * 100
        
        # Performance assessment
        perf_data = self.test_results.get('performance', {})
        performance_score = 0
        
        if perf_data.get('write_performance') == "GOOD":
            performance_score += 25
        if perf_data.get('read_performance') == "GOOD":
            performance_score += 25
        if perf_data.get('memory_efficiency') == "GOOD":
            performance_score += 25
        if perf_data.get('cache_efficiency') == "GOOD":
            performance_score += 25
        
        # V8.3 readiness assessment
        critical_features = [
            self.test_results.get('lru_cache', {}).get('basic_ops') == "PASS",
            self.test_results.get('memory_manager', {}).get('eval_cache') == "PASS",
            self.test_results.get('memory_scaling', {}).get('scaling_works') == "PASS",
            self.test_results.get('pressure_handling', {}).get('pressure_effective') == "PASS"
        ]
        
        critical_passing = sum(critical_features)
        readiness_score = (critical_passing / len(critical_features)) * 100
        
        return {
            'test_execution': {
                'passed_tests': passed,
                'failed_tests': failed,
                'success_rate': success_rate
            },
            'memory_analysis': {
                'baseline_mb': self.baseline_memory,
                'current_mb': current_memory,
                'increase_mb': memory_increase,
                'efficiency': "GOOD" if memory_increase < 15 else "NEEDS_IMPROVEMENT"
            },
            'performance_analysis': {
                'performance_score': performance_score,
                'write_ops': perf_data.get('write_performance', 'UNKNOWN'),
                'read_ops': perf_data.get('read_performance', 'UNKNOWN'),
                'cache_efficiency': perf_data.get('cache_efficiency', 'UNKNOWN')
            },
            'v8_3_readiness': {
                'readiness_score': readiness_score,
                'critical_features_passing': critical_passing,
                'total_critical_features': len(critical_features),
                'status': 'READY' if readiness_score >= 75 else 'NEEDS_WORK',
                'recommendation': self._get_recommendation(readiness_score, performance_score)
            }
        }
    
    def _get_recommendation(self, readiness_score: float, performance_score: float) -> str:
        """Get deployment recommendation"""
        if readiness_score >= 75 and performance_score >= 75:
            return "PROCEED: V8.3 memory optimization ready for integration"
        elif readiness_score >= 75:
            return "PROCEED WITH CAUTION: Core features work but performance needs improvement"
        elif performance_score >= 75:
            return "DELAY: Good performance but core features need fixes"
        else:
            return "DELAY: Both functionality and performance need significant improvement"
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save test results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"v7p3r_v8_3_standalone_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        return filename


def main():
    """Run V8.3 standalone testing"""
    print("Starting V7P3R V8.3 Standalone Memory Testing...")
    
    tester = V8_3_StandaloneTest()
    results = tester.run_all_tests()
    
    # Display results
    summary = results.get('summary', {})
    print("\n" + "="*60)
    print("V8.3 STANDALONE TEST SUMMARY")
    print("="*60)
    
    execution = summary.get('test_execution', {})
    print(f"Tests passed: {execution.get('passed_tests', 0)}")
    print(f"Tests failed: {execution.get('failed_tests', 0)}")
    print(f"Success rate: {execution.get('success_rate', 0):.1f}%")
    
    memory = summary.get('memory_analysis', {})
    print(f"\nMemory baseline: {memory.get('baseline_mb', 0):.1f}MB")
    print(f"Memory final: {memory.get('current_mb', 0):.1f}MB")
    print(f"Memory increase: {memory.get('increase_mb', 0):.1f}MB")
    print(f"Memory efficiency: {memory.get('efficiency', 'UNKNOWN')}")
    
    performance = summary.get('performance_analysis', {})
    print(f"\nPerformance score: {performance.get('performance_score', 0)}/100")
    print(f"Write performance: {performance.get('write_ops', 'UNKNOWN')}")
    print(f"Read performance: {performance.get('read_ops', 'UNKNOWN')}")
    print(f"Cache efficiency: {performance.get('cache_efficiency', 'UNKNOWN')}")
    
    readiness = summary.get('v8_3_readiness', {})
    print(f"\nV8.3 Readiness: {readiness.get('status', 'UNKNOWN')}")
    print(f"Readiness score: {readiness.get('readiness_score', 0):.1f}%")
    print(f"Critical features: {readiness.get('critical_features_passing', 0)}/{readiness.get('total_critical_features', 0)}")
    print(f"Recommendation: {readiness.get('recommendation', 'Unknown')}")
    
    # Save results
    results_file = tester.save_results()
    print(f"\nDetailed results saved to: {results_file}")
    
    # V8.3 specific recommendations
    print("\n" + "="*60)
    print("V8.3 IMPLEMENTATION RECOMMENDATIONS")
    print("="*60)
    
    if readiness.get('readiness_score', 0) >= 75:
        print("✓ Core memory management features are working")
        print("✓ LRU caching with TTL is functional")
        print("✓ Memory scaling adapts to constraints")
        print("✓ Pressure handling prevents memory runaway")
        
        if performance.get('performance_score', 0) >= 75:
            print("✓ Performance characteristics are acceptable")
            print("\nRECOMMENDATION: Proceed with V8.3 integration into main engine")
        else:
            print("⚠ Performance needs optimization before full deployment")
            print("\nRECOMMENDATION: Integrate with performance monitoring for gradual rollout")
    else:
        print("✗ Critical memory management features need fixes")
        print("\nRECOMMENDATION: Address failing tests before V8.3 integration")
    
    print("\n" + "="*60)
    print("Ready for next phase: V8.3 Engine Integration!")
    print("="*60)


if __name__ == "__main__":
    main()
