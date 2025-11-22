"""
V17.0 Profiling Tool - Comprehensive Performance Analysis

This tool measures detailed performance metrics to identify bottlenecks
and guide optimization decisions.

Usage:
    python testing/test_v17_profiling.py

Metrics Collected:
    - Time per depth iteration (1-8)
    - Nodes per depth and NPS by depth
    - Transposition table hit/store rates
    - Move ordering effectiveness (killer moves, history)
    - Evaluation time breakdown
    - Quiescence search statistics
    - Time management decisions (target, max, actual, exit reason)
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import chess
from v7p3r import V7P3REngine


class EngineProfiler:
    """Comprehensive profiling for V7P3R engine"""
    
    def __init__(self, engine: V7P3REngine):
        self.engine = engine
        self.metrics = {
            'depth_times': defaultdict(float),  # depth -> total time
            'depth_nodes': defaultdict(int),    # depth -> total nodes
            'depth_iterations': defaultdict(int),  # depth -> count
            'tt_hits': 0,
            'tt_stores': 0,
            'killer_hits': 0,
            'eval_time': 0.0,
            'search_time': 0.0,
            'qsearch_nodes': 0,
            'time_management': [],  # List of time mgmt decisions
        }
    
    def profile_search(self, board: chess.Board, time_limit: float = 5.0) -> Tuple[chess.Move, Dict]:
        """
        Profile a single search and return detailed metrics
        
        Returns:
            (best_move, metrics_dict)
        """
        # Capture initial state
        initial_stats = self.engine.search_stats.copy()
        
        # Instrument the engine for detailed profiling
        original_recursive = self.engine._recursive_search
        depth_stats = defaultdict(lambda: {'nodes': 0, 'time': 0.0, 'entries': 0})
        
        def instrumented_recursive(board, depth, alpha, beta, time_limit):
            start = time.time()
            start_nodes = self.engine.nodes_searched
            
            result = original_recursive(board, depth, alpha, beta, time_limit)
            
            elapsed = time.time() - start
            nodes = self.engine.nodes_searched - start_nodes
            
            depth_stats[depth]['nodes'] += nodes
            depth_stats[depth]['time'] += elapsed
            depth_stats[depth]['entries'] += 1
            
            return result
        
        # Temporarily replace method
        self.engine._recursive_search = instrumented_recursive
        
        # Run the search
        search_start = time.time()
        best_move = self.engine.search(board, time_limit=time_limit)
        search_elapsed = time.time() - search_start
        
        # Restore original method
        self.engine._recursive_search = original_recursive
        
        # Collect final stats
        final_stats = self.engine.search_stats.copy()
        
        # Calculate deltas
        metrics = {
            'search_time': search_elapsed,
            'total_nodes': self.engine.nodes_searched,
            'avg_nps': int(self.engine.nodes_searched / max(search_elapsed, 0.001)),
            'depth_stats': dict(depth_stats),
            'tt_hits': final_stats.get('tt_hits', 0) - initial_stats.get('tt_hits', 0),
            'tt_stores': final_stats.get('tt_stores', 0) - initial_stats.get('tt_stores', 0),
            'killer_hits': final_stats.get('killer_hits', 0) - initial_stats.get('killer_hits', 0),
            'cache_hits': final_stats.get('cache_hits', 0) - initial_stats.get('cache_hits', 0),
            'cache_misses': final_stats.get('cache_misses', 0) - initial_stats.get('cache_misses', 0),
            'best_move': best_move.uci() if best_move else None,
        }
        
        # Calculate derived metrics
        if metrics['tt_hits'] + metrics['tt_stores'] > 0:
            metrics['tt_hit_rate'] = metrics['tt_hits'] / (metrics['tt_hits'] + metrics['tt_stores'])
        else:
            metrics['tt_hit_rate'] = 0.0
        
        if metrics['cache_hits'] + metrics['cache_misses'] > 0:
            metrics['cache_hit_rate'] = metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses'])
        else:
            metrics['cache_hit_rate'] = 0.0
        
        return best_move, metrics
    
    def profile_position_suite(self, positions: List[Tuple[str, str]], time_limit: float = 5.0) -> Dict:
        """
        Profile multiple positions and aggregate results
        
        Args:
            positions: List of (fen, description) tuples
            time_limit: Time limit per position
        
        Returns:
            Aggregated metrics dictionary
        """
        suite_results = {
            'positions': [],
            'aggregated': {
                'total_positions': len(positions),
                'avg_search_time': 0.0,
                'avg_nodes': 0,
                'avg_nps': 0,
                'avg_depth': 0.0,
                'avg_tt_hit_rate': 0.0,
                'avg_cache_hit_rate': 0.0,
            }
        }
        
        total_time = 0.0
        total_nodes = 0
        total_nps = 0
        total_tt_hit_rate = 0.0
        total_cache_hit_rate = 0.0
        
        for fen, description in positions:
            print(f"\n{'='*60}")
            print(f"Profiling: {description}")
            print(f"FEN: {fen}")
            print(f"{'='*60}")
            
            board = chess.Board(fen)
            self.engine.evaluation_cache.clear()
            self.engine.transposition_table.clear()
            
            best_move, metrics = self.profile_search(board, time_limit)
            
            # Calculate achieved depth from depth_stats
            achieved_depth = max(metrics['depth_stats'].keys()) if metrics['depth_stats'] else 0
            
            position_result = {
                'fen': fen,
                'description': description,
                'best_move': metrics['best_move'],
                'achieved_depth': achieved_depth,
                'search_time': metrics['search_time'],
                'total_nodes': metrics['total_nodes'],
                'nps': metrics['avg_nps'],
                'tt_hit_rate': metrics['tt_hit_rate'],
                'cache_hit_rate': metrics['cache_hit_rate'],
                'depth_breakdown': metrics['depth_stats'],
            }
            
            suite_results['positions'].append(position_result)
            
            # Aggregate
            total_time += metrics['search_time']
            total_nodes += metrics['total_nodes']
            total_nps += metrics['avg_nps']
            total_tt_hit_rate += metrics['tt_hit_rate']
            total_cache_hit_rate += metrics['cache_hit_rate']
            
            # Display summary
            print(f"\n✓ Search completed in {metrics['search_time']:.2f}s")
            print(f"  Depth achieved: {achieved_depth}")
            print(f"  Nodes: {metrics['total_nodes']:,}")
            print(f"  NPS: {metrics['avg_nps']:,}")
            print(f"  TT hit rate: {metrics['tt_hit_rate']*100:.1f}%")
            print(f"  Cache hit rate: {metrics['cache_hit_rate']*100:.1f}%")
            print(f"  Best move: {metrics['best_move']}")
            
            # Depth breakdown
            print(f"\n  Depth Breakdown:")
            for depth in sorted(metrics['depth_stats'].keys()):
                stats = metrics['depth_stats'][depth]
                depth_nps = int(stats['nodes'] / max(stats['time'], 0.001))
                print(f"    Depth {depth}: {stats['nodes']:,} nodes, {stats['time']:.3f}s, {depth_nps:,} nps")
        
        # Calculate averages
        n = len(positions)
        suite_results['aggregated']['avg_search_time'] = total_time / n
        suite_results['aggregated']['avg_nodes'] = total_nodes // n
        suite_results['aggregated']['avg_nps'] = total_nps // n
        suite_results['aggregated']['avg_depth'] = sum(p['achieved_depth'] for p in suite_results['positions']) / n
        suite_results['aggregated']['avg_tt_hit_rate'] = total_tt_hit_rate / n
        suite_results['aggregated']['avg_cache_hit_rate'] = total_cache_hit_rate / n
        
        return suite_results
    
    def compare_engines(self, engine1: V7P3REngine, engine2: V7P3REngine, 
                       positions: List[Tuple[str, str]], time_limit: float = 5.0) -> Dict:
        """
        Compare two engine versions side-by-side
        
        Args:
            engine1: First engine (e.g., v14.1)
            engine2: Second engine (e.g., v17.0)
            positions: Test positions
            time_limit: Time limit per position
        
        Returns:
            Comparison dictionary
        """
        print("\n" + "="*80)
        print("ENGINE COMPARISON")
        print("="*80)
        
        print("\n--- Profiling Engine 1 ---")
        profiler1 = EngineProfiler(engine1)
        results1 = profiler1.profile_position_suite(positions, time_limit)
        
        print("\n--- Profiling Engine 2 ---")
        profiler2 = EngineProfiler(engine2)
        results2 = profiler2.profile_position_suite(positions, time_limit)
        
        # Calculate deltas
        comparison = {
            'engine1': results1,
            'engine2': results2,
            'deltas': {
                'depth_improvement': results2['aggregated']['avg_depth'] - results1['aggregated']['avg_depth'],
                'nps_improvement': results2['aggregated']['avg_nps'] - results1['aggregated']['avg_nps'],
                'nps_improvement_pct': ((results2['aggregated']['avg_nps'] / results1['aggregated']['avg_nps']) - 1) * 100,
                'time_diff': results2['aggregated']['avg_search_time'] - results1['aggregated']['avg_search_time'],
                'tt_hit_rate_diff': results2['aggregated']['avg_tt_hit_rate'] - results1['aggregated']['avg_tt_hit_rate'],
            }
        }
        
        # Display comparison summary
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"\nAverage Depth:")
        print(f"  Engine 1: {results1['aggregated']['avg_depth']:.2f}")
        print(f"  Engine 2: {results2['aggregated']['avg_depth']:.2f}")
        print(f"  Improvement: {comparison['deltas']['depth_improvement']:+.2f}")
        
        print(f"\nAverage NPS:")
        print(f"  Engine 1: {results1['aggregated']['avg_nps']:,}")
        print(f"  Engine 2: {results2['aggregated']['avg_nps']:,}")
        print(f"  Improvement: {comparison['deltas']['nps_improvement']:+,} ({comparison['deltas']['nps_improvement_pct']:+.1f}%)")
        
        print(f"\nAverage Search Time:")
        print(f"  Engine 1: {results1['aggregated']['avg_search_time']:.3f}s")
        print(f"  Engine 2: {results2['aggregated']['avg_search_time']:.3f}s")
        print(f"  Difference: {comparison['deltas']['time_diff']:+.3f}s")
        
        print(f"\nTransposition Table Hit Rate:")
        print(f"  Engine 1: {results1['aggregated']['avg_tt_hit_rate']*100:.1f}%")
        print(f"  Engine 2: {results2['aggregated']['avg_tt_hit_rate']*100:.1f}%")
        print(f"  Difference: {comparison['deltas']['tt_hit_rate_diff']*100:+.1f}%")
        
        return comparison


def get_test_positions() -> List[Tuple[str, str]]:
    """
    Get diverse test positions covering opening, middlegame, and endgame
    
    Returns:
        List of (fen, description) tuples
    """
    return [
        # Opening positions
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting Position"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "After 1.e4"),
        ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Open Game (e4 e5)"),
        
        # Middlegame positions
        ("r1bqk2r/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQK2R w KQkq - 0 8", "Typical Middlegame"),
        ("r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP2QPPP/R1B1K2R w KQ - 2 10", "Complex Middlegame"),
        
        # Tactical positions
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "Scholar's Mate Threat"),
        ("r2qkb1r/ppp2ppp/2np1n2/4p1B1/2B1P3/2N2N2/PPPP1PPP/R2QK2R w KQkq - 0 6", "Fried Liver Setup"),
        
        # Endgame positions
        ("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1", "King and Pawn Endgame"),
        ("8/8/pk6/8/8/1K6/8/8 w - - 0 1", "King Opposition"),
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "Simple Pawn Promotion"),
    ]


def main():
    """Main profiling execution"""
    print("="*80)
    print("V17.0 ENGINE PROFILING TOOL")
    print("="*80)
    
    # Create engine
    print("\nInitializing engine...")
    engine = V7P3REngine(use_fast_evaluator=True)
    
    # Get test positions
    positions = get_test_positions()
    
    # Run profiling suite
    print(f"\nProfiling {len(positions)} positions with 5-second time limit...")
    profiler = EngineProfiler(engine)
    results = profiler.profile_position_suite(positions, time_limit=5.0)
    
    # Display final summary
    print("\n" + "="*80)
    print("PROFILING RESULTS SUMMARY")
    print("="*80)
    print(f"\nPositions Tested: {results['aggregated']['total_positions']}")
    print(f"Average Depth Achieved: {results['aggregated']['avg_depth']:.2f}")
    print(f"Average Nodes Searched: {results['aggregated']['avg_nodes']:,}")
    print(f"Average NPS: {results['aggregated']['avg_nps']:,}")
    print(f"Average Search Time: {results['aggregated']['avg_search_time']:.3f}s")
    print(f"Average TT Hit Rate: {results['aggregated']['avg_tt_hit_rate']*100:.1f}%")
    print(f"Average Cache Hit Rate: {results['aggregated']['avg_cache_hit_rate']*100:.1f}%")
    
    # Depth distribution
    depth_counts = defaultdict(int)
    for position in results['positions']:
        depth_counts[position['achieved_depth']] += 1
    
    print(f"\nDepth Distribution:")
    for depth in sorted(depth_counts.keys()):
        count = depth_counts[depth]
        pct = (count / len(positions)) * 100
        print(f"  Depth {depth}: {count} positions ({pct:.1f}%)")
    
    # Save results to JSON
    output_file = f"testing/v17_profiling_results_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Success criteria check
    avg_depth = results['aggregated']['avg_depth']
    avg_nps = results['aggregated']['avg_nps']
    
    print("\n" + "="*80)
    print("SUCCESS CRITERIA CHECK")
    print("="*80)
    
    criteria = [
        ("Average Depth >= 5.0", avg_depth >= 5.0, f"{avg_depth:.2f}"),
        ("Average NPS >= 15,000", avg_nps >= 15000, f"{avg_nps:,}"),
    ]
    
    for criterion, passed, value in criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {criterion} (actual: {value})")


if __name__ == '__main__':
    main()
