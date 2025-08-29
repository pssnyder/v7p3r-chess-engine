#!/usr/bin/env python3
"""
V7P3R Chess Engine V8.3 - Memory Profiling and Analysis Tool
Comprehensive memor        # Engine-specific memory usage
        engine_memory: Dict[str, Any] = {
            'transposition_table_entries': len(self.engine.transposition_table),
            'killer_moves_entries': len(self.engine.killer_moves),
            'history_scores_entries': len(self.engine.history_scores),
            'evaluation_cache_entries': len(self.engine.evaluation_cache),
            'estimated_tt_size_mb': self._estimate_dict_size(self.engine.transposition_table),
            'estimated_killer_size_mb': self._estimate_dict_size(self.engine.killer_moves),
            'estimated_history_size_mb': self._estimate_dict_size(self.engine.history_scores),
            'estimated_cache_size_mb': self._estimate_dict_size(self.engine.evaluation_cache)
        }sis and optimization identification
"""

import chess
import chess.engine
import psutil
import gc
import sys
import time
import tracemalloc
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3RCleanEngine


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point"""
    timestamp: float
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    heap_objects: int
    heap_size_mb: float
    gc_stats: Dict[str, int]
    engine_specific: Dict[str, Any]


@dataclass
class PerformanceMetrics:
    """Performance metrics for engine operations"""
    nodes_per_second: float
    average_move_time: float
    cache_hit_ratio: float
    search_efficiency: float
    memory_efficiency: float


class V7P3RMemoryProfiler:
    """Comprehensive memory profiling and analysis for V7P3R engine"""
    
    def __init__(self):
        self.engine = V7P3RCleanEngine()
        self.snapshots: List[MemorySnapshot] = []
        self.process = psutil.Process()
        self.baseline_memory = None
        self.start_time = None
        
        # Enable detailed memory tracking
        tracemalloc.start()
        
        # Standard test positions for consistent benchmarking
        self.test_positions = [
            # Starting position
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            # Middle game position
            "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            # Complex tactical position
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            # Endgame position
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            # Queen endgame
            "8/8/8/8/8/3k4/4q3/4K3 w - - 0 1"
        ]
    
    def take_memory_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a comprehensive memory snapshot"""
        # Force garbage collection for accurate measurement
        gc.collect()
        
        # System memory info
        memory_info = self.process.memory_info()
        
        # Python heap info
        current, peak = tracemalloc.get_traced_memory()
        
        # GC statistics
        gc_stats = {f"gen_{i}": gc.get_count()[i] for i in range(3)}
        
        # Engine-specific memory usage
        engine_memory = self._analyze_engine_memory()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            heap_objects=len(gc.get_objects()),
            heap_size_mb=current / 1024 / 1024,
            gc_stats=gc_stats,
            engine_specific=engine_memory
        )
        
        self.snapshots.append(snapshot)
        
        if label:
            print(f"Memory snapshot ({label}): RSS={snapshot.rss_mb:.1f}MB, "
                  f"Heap={snapshot.heap_size_mb:.1f}MB, Objects={snapshot.heap_objects}")
        
        return snapshot
    
    def _analyze_engine_memory(self) -> Dict[str, Any]:
        """Analyze engine-specific memory usage"""
        engine_memory = {
            'transposition_table_entries': len(self.engine.transposition_table),
            'killer_moves_entries': len(self.engine.killer_moves),
            'history_scores_entries': len(self.engine.history_scores),
            'evaluation_cache_entries': len(self.engine.evaluation_cache),
        }
        
        # Estimate memory usage of each component
        engine_memory['estimated_tt_size_mb'] = self._estimate_dict_size(self.engine.transposition_table)
        engine_memory['estimated_killer_size_mb'] = self._estimate_dict_size(self.engine.killer_moves)
        engine_memory['estimated_history_size_mb'] = self._estimate_dict_size(self.engine.history_scores)
        engine_memory['estimated_cache_size_mb'] = self._estimate_dict_size(self.engine.evaluation_cache)
        
        return engine_memory
    
    def _estimate_dict_size(self, dictionary: Dict) -> float:
        """Estimate memory usage of a dictionary in MB"""
        try:
            # Rough estimation: each dict entry is approximately 100-200 bytes
            # depending on key/value types
            estimated_bytes = len(dictionary) * 150  # Conservative estimate
            return estimated_bytes / 1024 / 1024
        except:
            return 0.0
    
    def run_memory_stress_test(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Run comprehensive memory stress test"""
        print(f"Starting memory stress test for {duration_seconds} seconds...")
        
        self.start_time = time.time()
        self.take_memory_snapshot("baseline")
        self.baseline_memory = self.snapshots[-1]
        
        test_results = {
            'start_time': datetime.now().isoformat(),
            'duration_seconds': duration_seconds,
            'positions_tested': [],
            'performance_metrics': [],
            'memory_growth': [],
            'cache_statistics': {},
            'issues_detected': []
        }
        
        end_time = time.time() + duration_seconds
        position_count = 0
        
        while time.time() < end_time:
            # Test each position multiple times
            for fen in self.test_positions:
                if time.time() >= end_time:
                    break
                
                position_count += 1
                board = chess.Board(fen)
                
                # Take snapshot before search
                pre_search = self.take_memory_snapshot()
                
                # Perform search with timing
                start_search = time.time()
                try:
                    move = self.engine.search(board, time_limit=1.0)
                    search_time = time.time() - start_search
                    
                    # Record performance metrics
                    nps = self.engine.nodes_searched / max(search_time, 0.001)
                    
                    test_results['performance_metrics'].append({
                        'position': fen,
                        'nodes_searched': self.engine.nodes_searched,
                        'search_time': search_time,
                        'nodes_per_second': nps,
                        'move': str(move) if move else None
                    })
                    
                except Exception as e:
                    test_results['issues_detected'].append({
                        'position': fen,
                        'error': str(e),
                        'timestamp': time.time()
                    })
                
                # Take snapshot after search
                post_search = self.take_memory_snapshot()
                
                # Analyze memory growth
                memory_growth = post_search.rss_mb - pre_search.rss_mb
                if memory_growth > 0.1:  # More than 100KB growth
                    test_results['memory_growth'].append({
                        'position': fen,
                        'growth_mb': memory_growth,
                        'timestamp': time.time()
                    })
                
                test_results['positions_tested'].append({
                    'fen': fen,
                    'position_count': position_count,
                    'memory_before_mb': pre_search.rss_mb,
                    'memory_after_mb': post_search.rss_mb
                })
        
        # Final analysis
        final_snapshot = self.take_memory_snapshot("final")
        
        test_results['cache_statistics'] = self._analyze_cache_performance()
        test_results['memory_analysis'] = self._analyze_memory_patterns()
        test_results['total_positions_tested'] = position_count
        test_results['final_memory_mb'] = final_snapshot.rss_mb
        test_results['memory_increase_mb'] = final_snapshot.rss_mb - self.baseline_memory.rss_mb
        
        return test_results
    
    def _analyze_cache_performance(self) -> Dict[str, Any]:
        """Analyze cache hit ratios and effectiveness"""
        stats = self.engine.search_stats
        
        total_lookups = stats.get('cache_hits', 0) + stats.get('cache_misses', 0)
        hit_ratio = stats.get('cache_hits', 0) / max(total_lookups, 1)
        
        return {
            'cache_hits': stats.get('cache_hits', 0),
            'cache_misses': stats.get('cache_misses', 0),
            'hit_ratio': hit_ratio,
            'total_lookups': total_lookups,
            'transposition_entries': len(self.engine.transposition_table),
            'evaluation_cache_entries': len(self.engine.evaluation_cache),
            'killer_moves_entries': len(self.engine.killer_moves),
            'history_scores_entries': len(self.engine.history_scores)
        }
    
    def _analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory usage patterns and identify issues"""
        if len(self.snapshots) < 2:
            return {}
        
        memory_deltas = []
        for i in range(1, len(self.snapshots)):
            delta = self.snapshots[i].rss_mb - self.snapshots[i-1].rss_mb
            memory_deltas.append(delta)
        
        avg_growth = sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0
        max_growth = max(memory_deltas) if memory_deltas else 0
        
        # Detect potential memory leaks
        consistent_growth = sum(1 for delta in memory_deltas if delta > 0.05)  # >50KB growth
        leak_risk = consistent_growth / len(memory_deltas) if memory_deltas else 0
        
        return {
            'average_growth_per_search_mb': avg_growth,
            'maximum_growth_mb': max_growth,
            'leak_risk_percentage': leak_risk * 100,
            'total_snapshots': len(self.snapshots),
            'memory_volatility': max(memory_deltas) - min(memory_deltas) if memory_deltas else 0
        }
    
    def generate_optimization_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate specific optimization recommendations based on test results"""
        recommendations = []
        
        # Memory growth analysis
        memory_increase = test_results.get('memory_increase_mb', 0)
        if memory_increase > 10:  # More than 10MB growth
            recommendations.append(
                f"HIGH PRIORITY: Significant memory growth detected ({memory_increase:.1f}MB). "
                "Implement aggressive cache cleanup and memory management."
            )
        
        # Cache performance analysis
        cache_stats = test_results.get('cache_statistics', {})
        hit_ratio = cache_stats.get('hit_ratio', 0)
        if hit_ratio < 0.3:  # Less than 30% hit ratio
            recommendations.append(
                f"MEDIUM PRIORITY: Low cache hit ratio ({hit_ratio:.1%}). "
                "Consider optimizing cache size and eviction strategy."
            )
        
        # Performance analysis
        avg_nps = 0
        perf_metrics = test_results.get('performance_metrics', [])
        if perf_metrics:
            avg_nps = sum(m.get('nodes_per_second', 0) for m in perf_metrics) / len(perf_metrics)
            if avg_nps < 10000:  # Less than 10K nodes per second
                recommendations.append(
                    f"HIGH PRIORITY: Low search performance ({avg_nps:.0f} NPS). "
                    "Profile move generation and evaluation functions."
                )
        
        # Memory leak analysis
        memory_analysis = test_results.get('memory_analysis', {})
        leak_risk = memory_analysis.get('leak_risk_percentage', 0)
        if leak_risk > 50:  # More than 50% of searches show growth
            recommendations.append(
                f"CRITICAL: High memory leak risk ({leak_risk:.1f}%). "
                "Implement immediate memory cleanup after each search."
            )
        
        # Cache size recommendations
        tt_entries = cache_stats.get('transposition_entries', 0)
        if tt_entries > 100000:  # More than 100K entries
            recommendations.append(
                "MEDIUM PRIORITY: Large transposition table detected. "
                "Implement LRU eviction and size limits."
            )
        
        if not recommendations:
            recommendations.append(
                f"GOOD: Engine shows healthy memory usage. "
                f"Current performance: {avg_nps:.0f} NPS, "
                f"Cache hit ratio: {hit_ratio:.1%}, "
                f"Memory growth: {memory_increase:.1f}MB"
            )
        
        return recommendations
    
    def save_results(self, test_results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"v7p3r_v8_3_memory_profile_{timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        # Make results JSON serializable
        serializable_results = test_results.copy()
        serializable_results['snapshots'] = [
            {
                'timestamp': s.timestamp,
                'rss_mb': s.rss_mb,
                'vms_mb': s.vms_mb,
                'heap_objects': s.heap_objects,
                'heap_size_mb': s.heap_size_mb,
                'gc_stats': s.gc_stats,
                'engine_specific': s.engine_specific
            }
            for s in self.snapshots
        ]
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return filepath


def main():
    """Run comprehensive memory profiling test"""
    profiler = V7P3RMemoryProfiler()
    
    print("=== V7P3R V8.3 Memory Profiling Test ===")
    print("Testing memory usage patterns and optimization opportunities...")
    
    # Run stress test
    test_results = profiler.run_memory_stress_test(duration_seconds=120)  # 2-minute test
    
    # Generate recommendations
    recommendations = profiler.generate_optimization_recommendations(test_results)
    
    # Display results
    print("\n=== MEMORY PROFILING RESULTS ===")
    print(f"Total positions tested: {test_results['total_positions_tested']}")
    print(f"Final memory usage: {test_results['final_memory_mb']:.1f}MB")
    print(f"Memory increase: {test_results['memory_increase_mb']:.1f}MB")
    
    cache_stats = test_results.get('cache_statistics', {})
    print(f"Cache hit ratio: {cache_stats.get('hit_ratio', 0):.1%}")
    print(f"Average NPS: {sum(m.get('nodes_per_second', 0) for m in test_results.get('performance_metrics', [])) / max(len(test_results.get('performance_metrics', [])), 1):.0f}")
    
    print("\n=== OPTIMIZATION RECOMMENDATIONS ===")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # Save results
    results_file = profiler.save_results(test_results)
    print(f"\nDetailed results saved to: {results_file}")
    
    print("\n=== NEXT STEPS ===")
    print("1. Review recommendations and implement high-priority optimizations")
    print("2. Set up automated memory monitoring for continuous testing")
    print("3. Create dynamic memory management for V8.3 implementation")
    print("4. Establish performance baselines for regression testing")


if __name__ == "__main__":
    main()
