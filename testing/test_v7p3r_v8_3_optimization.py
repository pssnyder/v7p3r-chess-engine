#!/usr/bin/env python3
"""
V7P3R Chess Engine V8.3 - Memory Optimization and Performance Testing
Comprehensive testing of memory management and performance monitoring features
"""

import sys
import os
import time
import psutil
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r_memory_manager import V7P3RMemoryManager, MemoryPolicy, create_memory_manager, LRUCacheWithTTL
from v7p3r_performance_monitor import PerformanceProfiler, profile, profiled_section, get_profiler


class V8_3_MemoryOptimizationTester:
    """Comprehensive testing suite for V8.3 memory optimization features"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive V8.3 testing suite"""
        print("=== V7P3R V8.3 Memory Optimization Testing ===")
        print(f"Baseline memory: {self.baseline_memory:.1f}MB")
        
        # Test 1: LRU Cache with TTL
        print("\n1. Testing LRU Cache with TTL...")
        self.test_results['lru_cache'] = self.test_lru_cache_with_ttl()
        
        # Test 2: Memory Manager Basic Operations
        print("\n2. Testing Memory Manager Basic Operations...")
        self.test_results['memory_manager_basic'] = self.test_memory_manager_basic()
        
        # Test 3: Dynamic Memory Scaling
        print("\n3. Testing Dynamic Memory Scaling...")
        self.test_results['dynamic_scaling'] = self.test_dynamic_memory_scaling()
        
        # Test 4: Memory Pressure Handling
        print("\n4. Testing Memory Pressure Handling...")
        self.test_results['pressure_handling'] = self.test_memory_pressure_handling()
        
        # Test 5: Performance Profiler
        print("\n5. Testing Performance Profiler...")
        self.test_results['performance_profiler'] = self.test_performance_profiler()
        
        # Test 6: Game Phase Optimization
        print("\n6. Testing Game Phase Optimization...")
        self.test_results['game_phase_optimization'] = self.test_game_phase_optimization()
        
        # Test 7: Integrated Memory + Performance
        print("\n7. Testing Integrated Memory + Performance...")
        self.test_results['integrated_testing'] = self.test_integrated_features()
        
        # Generate summary
        self.test_results['summary'] = self.generate_test_summary()
        
        return self.test_results
    
    def test_lru_cache_with_ttl(self) -> Dict[str, Any]:
        """Test LRU cache with time-to-live functionality"""
        cache = LRUCacheWithTTL(max_size=100, ttl=1.0)
        results = {}
        
        # Basic operations
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Test retrieval
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        
        results['basic_operations'] = "PASS"
        
        # Test LRU eviction
        for i in range(150):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Should have only 100 items (max_size)
        stats = cache.get_stats()
        assert stats['size'] <= 100
        results['lru_eviction'] = "PASS"
        results['lru_size_after_overflow'] = stats['size']
        
        # Test TTL expiration
        cache.put("ttl_test", "expire_me")
        assert cache.get("ttl_test") == "expire_me"
        
        time.sleep(1.1)  # Wait for TTL expiration
        assert cache.get("ttl_test") is None
        results['ttl_expiration'] = "PASS"
        
        # Test cleanup
        expired_count = cache.cleanup_expired()
        results['cleanup_expired_count'] = expired_count
        
        # Test pressure cleanup
        initial_size = cache.get_stats()['size']
        removed_count = cache.cleanup_pressure(0.5)  # Remove 50%
        final_size = cache.get_stats()['size']
        
        results['pressure_cleanup'] = {
            'initial_size': initial_size,
            'removed_count': removed_count,
            'final_size': final_size,
            'reduction_percentage': ((initial_size - final_size) / initial_size) * 100
        }
        
        return results
    
    def test_memory_manager_basic(self) -> Dict[str, Any]:
        """Test basic memory manager functionality"""
        manager = create_memory_manager(max_memory_mb=50.0)
        results = {}
        
        # Test evaluation cache
        manager.store_evaluation("pos1", 1.5)
        manager.store_evaluation("pos2", -0.8)
        
        assert manager.get_evaluation("pos1") == 1.5
        assert manager.get_evaluation("pos2") == -0.8
        assert manager.get_evaluation("nonexistent") is None
        
        results['evaluation_cache'] = "PASS"
        
        # Test transposition table
        tt_data = {"best_move": "e2e4", "evaluation": 0.3, "depth": 5}
        manager.store_transposition("fen1", tt_data)
        
        retrieved = manager.get_transposition("fen1")
        assert retrieved == tt_data
        
        results['transposition_table'] = "PASS"
        
        # Test killer moves
        manager.store_killer_move(1, "e2e4")
        manager.store_killer_move(1, "d2d4")
        
        killers = manager.get_killer_moves(1)
        assert len(killers) <= 2
        assert "e2e4" in [str(move) for move in killers]
        
        results['killer_moves'] = "PASS"
        
        # Test history scores
        manager.update_history_score("e2e4", 100)
        manager.update_history_score("e2e4", 50)
        
        score = manager.get_history_score("e2e4")
        assert score > 0
        
        results['history_scores'] = "PASS"
        
        # Test memory statistics
        stats = manager.get_memory_stats()
        assert 'evaluation_cache' in stats
        assert 'transposition_table' in stats
        
        results['memory_stats'] = stats
        
        return results
    
    def test_dynamic_memory_scaling(self) -> Dict[str, Any]:
        """Test dynamic memory scaling based on usage patterns"""
        results = {}
        
        # Test small memory configuration
        small_manager = create_memory_manager(max_memory_mb=10.0)
        small_stats = small_manager.get_memory_stats()
        
        # Test large memory configuration
        large_manager = create_memory_manager(max_memory_mb=200.0)
        large_stats = large_manager.get_memory_stats()
        
        # Verify scaling
        small_cache_size = small_stats['memory_policy']['max_cache_size']
        large_cache_size = large_stats['memory_policy']['max_cache_size']
        
        assert large_cache_size > small_cache_size
        
        results['scaling_verification'] = {
            'small_cache_size': small_cache_size,
            'large_cache_size': large_cache_size,
            'scaling_ratio': large_cache_size / small_cache_size
        }
        
        # Test game phase optimization
        manager = create_memory_manager(max_memory_mb=100.0)
        
        # Opening optimization
        manager.optimize_for_game_phase("opening")
        opening_policy = manager.policy
        
        # Endgame optimization
        manager.optimize_for_game_phase("endgame")
        endgame_policy = manager.policy
        
        results['phase_optimization'] = {
            'opening_tt_size': opening_policy.max_tt_size,
            'endgame_tt_size': endgame_policy.max_tt_size,
            'opening_cache_size': opening_policy.max_cache_size,
            'endgame_cache_size': endgame_policy.max_cache_size
        }
        
        return results
    
    def test_memory_pressure_handling(self) -> Dict[str, Any]:
        """Test memory pressure detection and cleanup"""
        # Create manager with low memory limits to trigger pressure
        policy = MemoryPolicy(
            max_cache_size=100,
            max_tt_size=100,
            memory_pressure_mb=20.0,  # Low threshold
            critical_memory_mb=25.0
        )
        
        manager = V7P3RMemoryManager(policy)
        results = {}
        
        # Fill up caches
        for i in range(150):
            manager.store_evaluation(f"pos_{i}", float(i))
            manager.store_transposition(f"fen_{i}", {"move": f"move_{i}", "score": i})
        
        initial_stats = manager.get_memory_stats()
        
        # Trigger pressure cleanup
        removed_count = manager.pressure_cleanup()
        
        final_stats = manager.get_memory_stats()
        
        results['pressure_cleanup_effectiveness'] = {
            'initial_eval_cache': initial_stats['evaluation_cache']['size'],
            'final_eval_cache': final_stats['evaluation_cache']['size'],
            'initial_tt_size': initial_stats['transposition_table']['size'],
            'final_tt_size': final_stats['transposition_table']['size'],
            'total_removed': removed_count,
            'cleanup_stats': final_stats['cleanup_stats']
        }
        
        # Test routine cleanup
        manager.routine_cleanup()
        routine_stats = manager.get_memory_stats()
        
        results['routine_cleanup'] = {
            'cleanup_count': routine_stats['cleanup_stats']['total_cleanups'],
            'entries_removed': routine_stats['cleanup_stats']['entries_removed']
        }
        
        return results
    
    def test_performance_profiler(self) -> Dict[str, Any]:
        """Test performance profiling functionality"""
        profiler = PerformanceProfiler()
        results = {}
        
        # Test function profiling
        @profiler.profile_function
        def slow_function(n):
            time.sleep(0.01)  # Simulate slow operation
            return sum(range(n))
        
        @profiler.profile_function
        def fast_function(x):
            return x * 2
        
        @profiler.profile_function
        def memory_intensive_function():
            # Create temporary memory usage
            data = [i for i in range(10000)]
            return len(data)
        
        # Execute functions multiple times
        for i in range(5):
            slow_function(100)
            fast_function(i)
        
        memory_intensive_function()
        
        # Get performance report
        report = profiler.get_performance_report()
        
        results['function_profiles'] = len(profiler.function_profiles)
        results['total_runtime'] = report['runtime_summary']['total_runtime']
        results['slow_function_detected'] = any(
            issue['category'] == 'PERFORMANCE' 
            for issue in report['performance_issues']
        )
        
        # Test profiled sections
        with profiled_section("test_section"):
            time.sleep(0.005)
            result = sum(range(1000))
        
        section_report = profiler.get_performance_report()
        
        results['profiled_sections'] = "test_section" in profiler.function_profiles
        
        # Test optimization recommendations
        recommendations = profiler.get_optimization_recommendations()
        results['recommendations_generated'] = len(recommendations)
        
        return results
    
    def test_game_phase_optimization(self) -> Dict[str, Any]:
        """Test game phase specific optimizations"""
        results = {}
        
        phases = ["opening", "middlegame", "endgame"]
        phase_configs = {}
        
        for phase in phases:
            manager = create_memory_manager(max_memory_mb=100.0, game_phase=phase)
            manager.optimize_for_game_phase(phase)
            
            stats = manager.get_memory_stats()
            phase_configs[phase] = {
                'max_tt_size': manager.policy.max_tt_size,
                'max_cache_size': manager.policy.max_cache_size,
                'cache_ttl': manager.policy.cache_ttl,
                'tt_ttl': manager.policy.tt_ttl,
                'cleanup_interval': manager.policy.cleanup_interval
            }
        
        results['phase_configurations'] = phase_configs
        
        # Verify phase-specific optimizations
        results['opening_prioritizes_tt'] = (
            phase_configs['opening']['max_tt_size'] > 
            phase_configs['middlegame']['max_tt_size']
        )
        
        results['endgame_prioritizes_cache'] = (
            phase_configs['endgame']['max_cache_size'] > 
            phase_configs['middlegame']['max_cache_size']
        )
        
        return results
    
    def test_integrated_features(self) -> Dict[str, Any]:
        """Test integration of memory management and performance monitoring"""
        results = {}
        
        # Create integrated system
        manager = create_memory_manager(max_memory_mb=50.0)
        profiler = PerformanceProfiler()
        
        @profiler.profile_function
        def integrated_test_function():
            # Simulate engine-like operations
            for i in range(100):
                # Store evaluations
                manager.store_evaluation(f"pos_{i}", float(i) * 0.1)
                
                # Store transpositions
                manager.store_transposition(f"fen_{i}", {
                    "move": f"move_{i}",
                    "score": i,
                    "depth": 3
                })
                
                # Update killer moves
                manager.store_killer_move(i % 10, f"killer_{i}")
                
                # Update history
                manager.update_history_score(f"move_{i}", i * 10)
            
            # Retrieve some cached data
            for i in range(50):
                eval_result = manager.get_evaluation(f"pos_{i}")
                tt_result = manager.get_transposition(f"fen_{i}")
            
            return True
        
        # Execute integrated test
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        with profiled_section("integrated_operations"):
            result = integrated_test_function()
        
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Get comprehensive statistics
        memory_stats = manager.get_memory_stats()
        performance_report = profiler.get_performance_report()
        
        results['integration_success'] = result
        results['memory_usage'] = {
            'start_mb': start_memory,
            'end_mb': end_memory,
            'increase_mb': end_memory - start_memory
        }
        
        results['cache_performance'] = {
            'eval_cache_hits': memory_stats['evaluation_cache']['hit_ratio'],
            'tt_cache_hits': memory_stats['transposition_table']['hit_ratio']
        }
        
        results['performance_analysis'] = {
            'total_functions_profiled': len(profiler.function_profiles),
            'issues_detected': len(performance_report['performance_issues']),
            'top_bottleneck': performance_report['top_bottlenecks'][0]['name'] if performance_report['top_bottlenecks'] else None
        }
        
        # Test cleanup under load
        cleanup_count = manager.pressure_cleanup()
        results['cleanup_under_load'] = cleanup_count
        
        return results
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        end_memory = self.process.memory_info().rss / 1024 / 1024
        total_time = time.time() - self.start_time
        
        # Count test results
        passed_tests = 0
        failed_tests = 0
        
        for test_name, test_result in self.test_results.items():
            if test_name == 'summary':
                continue
                
            if isinstance(test_result, dict):
                for key, value in test_result.items():
                    if value == "PASS":
                        passed_tests += 1
                    elif value == "FAIL":
                        failed_tests += 1
        
        # Generate performance baselines
        baselines = {
            'memory_efficiency': {
                'baseline_mb': self.baseline_memory,
                'peak_mb': end_memory,
                'efficiency_ratio': self.baseline_memory / end_memory
            },
            'cache_performance': self._analyze_cache_performance(),
            'optimization_opportunities': self._identify_optimization_opportunities()
        }
        
        return {
            'test_execution': {
                'total_time_seconds': total_time,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / (passed_tests + failed_tests + 1) * 100
            },
            'memory_analysis': {
                'baseline_memory_mb': self.baseline_memory,
                'final_memory_mb': end_memory,
                'memory_increase_mb': end_memory - self.baseline_memory,
                'memory_efficiency': 'GOOD' if end_memory - self.baseline_memory < 20 else 'NEEDS_IMPROVEMENT'
            },
            'performance_baselines': baselines,
            'v8_3_readiness': self._assess_v8_3_readiness()
        }
    
    def _analyze_cache_performance(self) -> Dict[str, Any]:
        """Analyze cache performance across all tests"""
        # Extract cache performance from test results
        cache_analysis = {
            'lru_functionality': 'WORKING' if self.test_results.get('lru_cache', {}).get('basic_operations') == 'PASS' else 'ISSUES',
            'ttl_functionality': 'WORKING' if self.test_results.get('lru_cache', {}).get('ttl_expiration') == 'PASS' else 'ISSUES',
            'pressure_handling': 'EFFECTIVE' if self.test_results.get('pressure_handling', {}).get('pressure_cleanup_effectiveness', {}).get('total_removed', 0) > 0 else 'INEFFECTIVE'
        }
        
        return cache_analysis
    
    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify optimization opportunities based on test results"""
        opportunities = []
        
        # Check memory usage
        if self.test_results.get('summary', {}).get('memory_analysis', {}).get('memory_increase_mb', 0) > 15:
            opportunities.append("HIGH: Implement more aggressive memory cleanup")
        
        # Check cache effectiveness
        pressure_test = self.test_results.get('pressure_handling', {})
        if pressure_test.get('pressure_cleanup_effectiveness', {}).get('total_removed', 0) < 10:
            opportunities.append("MEDIUM: Improve cache eviction strategies")
        
        # Check integration performance
        integrated_test = self.test_results.get('integrated_testing', {})
        if integrated_test.get('performance_analysis', {}).get('issues_detected', 0) > 5:
            opportunities.append("MEDIUM: Address performance bottlenecks")
        
        return opportunities
    
    def _assess_v8_3_readiness(self) -> Dict[str, Any]:
        """Assess readiness for V8.3 deployment"""
        critical_tests = [
            'lru_cache',
            'memory_manager_basic',
            'dynamic_scaling',
            'pressure_handling',
            'performance_profiler'
        ]
        
        passing_critical = 0
        for test in critical_tests:
            test_result = self.test_results.get(test, {})
            if any(value == "PASS" for value in test_result.values() if isinstance(value, str)):
                passing_critical += 1
        
        readiness_score = (passing_critical / len(critical_tests)) * 100
        
        return {
            'readiness_score': readiness_score,
            'status': 'READY' if readiness_score >= 80 else 'NEEDS_WORK',
            'critical_tests_passing': passing_critical,
            'total_critical_tests': len(critical_tests),
            'recommendation': 'Proceed with V8.3 integration' if readiness_score >= 80 else 'Address failing tests before integration'
        }
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save comprehensive test results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"v7p3r_v8_3_optimization_test_{timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        return filepath


def main():
    """Run comprehensive V8.3 memory optimization testing"""
    print("Starting V7P3R V8.3 Memory Optimization Testing...")
    
    tester = V8_3_MemoryOptimizationTester()
    results = tester.run_all_tests()
    
    # Display summary
    summary = results.get('summary', {})
    print("\n" + "="*60)
    print("V8.3 MEMORY OPTIMIZATION TEST SUMMARY")
    print("="*60)
    
    execution = summary.get('test_execution', {})
    print(f"Tests passed: {execution.get('passed_tests', 0)}")
    print(f"Tests failed: {execution.get('failed_tests', 0)}")
    print(f"Success rate: {execution.get('success_rate', 0):.1f}%")
    
    memory = summary.get('memory_analysis', {})
    print(f"\nMemory baseline: {memory.get('baseline_memory_mb', 0):.1f}MB")
    print(f"Memory final: {memory.get('final_memory_mb', 0):.1f}MB")
    print(f"Memory increase: {memory.get('memory_increase_mb', 0):.1f}MB")
    print(f"Memory efficiency: {memory.get('memory_efficiency', 'UNKNOWN')}")
    
    readiness = summary.get('v8_3_readiness', {})
    print(f"\nV8.3 Readiness: {readiness.get('status', 'UNKNOWN')}")
    print(f"Readiness score: {readiness.get('readiness_score', 0):.1f}%")
    print(f"Recommendation: {readiness.get('recommendation', 'Unknown')}")
    
    # Show optimization opportunities
    opportunities = summary.get('performance_baselines', {}).get('optimization_opportunities', [])
    if opportunities:
        print(f"\nOptimization Opportunities:")
        for opp in opportunities:
            print(f"  - {opp}")
    
    # Save detailed results
    results_file = tester.save_results()
    print(f"\nDetailed results saved to: {results_file}")
    
    print("\n" + "="*60)
    print("V8.3 Memory Optimization Testing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
