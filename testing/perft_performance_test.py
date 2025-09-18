#!/usr/bin/env python3
"""
V7P3R v11 Phase 1: Baseline Perft Performance Testing
Comprehensive perft test suite to establish performance benchmarks
Author: Pat Snyder
"""

import time
import chess
import sys
import os
from typing import Dict, List, Tuple

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine


class V7P3RPerftTester:
    """Comprehensive perft testing for performance baseline establishment"""
    
    def __init__(self):
        self.engine = V7P3REngine()
        self.test_positions = [
            # Starting position
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting Position"),
            
            # Kiwipete (complex middle game)
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", "Kiwipete"),
            
            # Position 3 (tactical)
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "Tactical Position"),
            
            # Position 4 (endgame)
            ("r3k2r/8/3Q4/8/8/8/8/R3K2R b KQkq - 0 1", "Endgame Position"),
            
            # Position 5 (promotion)
            ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", "Promotion Position"),
            
            # Position 6 (castling rights)
            ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", "Castling Position")
        ]
    
    def run_perft_test(self, fen: str, depth: int) -> Tuple[int, float]:
        """Run perft test and return (nodes, time_seconds)"""
        board = chess.Board(fen)
        start_time = time.time()
        nodes = self.engine.perft(board, depth)
        end_time = time.time()
        return nodes, end_time - start_time
    
    def benchmark_position(self, fen: str, position_name: str, max_depth: int = 4) -> Dict:
        """Benchmark a specific position across multiple depths"""
        print(f"\n=== BENCHMARKING: {position_name} ===")
        print(f"FEN: {fen}")
        
        results = {
            'position_name': position_name,
            'fen': fen,
            'depth_results': {},
            'total_nodes': 0,
            'total_time': 0.0,
            'average_nps': 0.0
        }
        
        for depth in range(1, max_depth + 1):
            try:
                nodes, time_taken = self.run_perft_test(fen, depth)
                nps = nodes / time_taken if time_taken > 0 else 0
                
                print(f"Depth {depth}: {nodes:,} nodes in {time_taken:.3f}s = {nps:,.0f} NPS")
                
                results['depth_results'][depth] = {
                    'nodes': nodes,
                    'time': time_taken,
                    'nps': nps
                }
                
                results['total_nodes'] += nodes
                results['total_time'] += time_taken
                
                # Stop if taking too long
                if time_taken > 10.0:
                    print(f"Stopping at depth {depth} - time limit reached")
                    break
                    
            except Exception as e:
                print(f"Error at depth {depth}: {e}")
                break
        
        if results['total_time'] > 0:
            results['average_nps'] = results['total_nodes'] / results['total_time']
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive benchmark across all test positions"""
        print("V7P3R v11 Phase 1: Comprehensive Perft Benchmark")
        print("=" * 60)
        
        all_results = {
            'test_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'engine_version': 'v10.9 (baseline for v11)',
            'positions': [],
            'summary': {}
        }
        
        total_nodes = 0
        total_time = 0.0
        position_count = 0
        
        for fen, name in self.test_positions:
            try:
                result = self.benchmark_position(fen, name, max_depth=4)
                all_results['positions'].append(result)
                
                total_nodes += result['total_nodes']
                total_time += result['total_time']
                position_count += 1
                
            except Exception as e:
                print(f"Error testing {name}: {e}")
                continue
        
        # Calculate summary statistics
        if total_time > 0 and position_count > 0:
            all_results['summary'] = {
                'total_positions_tested': position_count,
                'total_nodes_calculated': total_nodes,
                'total_time_seconds': total_time,
                'overall_average_nps': total_nodes / total_time,
                'fastest_position_nps': max(pos['average_nps'] for pos in all_results['positions']),
                'slowest_position_nps': min(pos['average_nps'] for pos in all_results['positions'])
            }
        
        return all_results
    
    def print_summary_report(self, results: Dict):
        """Print a comprehensive summary report"""
        print("\n" + "=" * 60)
        print("PERFT BENCHMARK SUMMARY REPORT")
        print("=" * 60)
        
        summary = results.get('summary', {})
        
        print(f"Engine Version: {results['engine_version']}")
        print(f"Test Date: {results['test_date']}")
        print(f"Positions Tested: {summary.get('total_positions_tested', 0)}")
        print(f"Total Nodes: {summary.get('total_nodes_calculated', 0):,}")
        print(f"Total Time: {summary.get('total_time_seconds', 0):.3f} seconds")
        print(f"Overall Average NPS: {summary.get('overall_average_nps', 0):,.0f}")
        print(f"Fastest Position NPS: {summary.get('fastest_position_nps', 0):,.0f}")
        print(f"Slowest Position NPS: {summary.get('slowest_position_nps', 0):,.0f}")
        
        print("\nPER-POSITION BREAKDOWN:")
        print("-" * 40)
        for pos in results['positions']:
            print(f"{pos['position_name']}: {pos['average_nps']:,.0f} NPS avg")
        
        # Phase 1 targets
        print("\n" + "=" * 60)
        print("V11 PHASE 1 TARGET ANALYSIS")
        print("=" * 60)
        
        current_nps = summary.get('overall_average_nps', 0)
        target_nps = 20000  # Target from development plan
        
        print(f"Current Performance: {current_nps:,.0f} NPS")
        print(f"Phase 1 Target: {target_nps:,} NPS")
        
        if current_nps > 0:
            improvement_needed = (target_nps / current_nps - 1) * 100
            print(f"Improvement Needed: {improvement_needed:.1f}%")
            
            if current_nps >= target_nps:
                print("✅ Already meeting Phase 1 performance targets!")
            else:
                print(f"📈 Need {improvement_needed:.1f}% improvement to meet targets")
        
        print("\nRECOMMENDATIONS:")
        if current_nps < 15000:
            print("- Focus on move generation optimization")
            print("- Implement more efficient bitboard operations")
            print("- Consider caching improvements")
        elif current_nps < 20000:
            print("- Implement Late Move Reduction (LMR)")
            print("- Optimize move ordering")
            print("- Enhance time management")
        else:
            print("- Performance baseline excellent")
            print("- Focus on search depth improvements")
            print("- Advanced evaluation enhancements")
    
    def save_results(self, results: Dict, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"perft_benchmark_{timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n📁 Results saved to: {filepath}")
        except Exception as e:
            print(f"\n❌ Error saving results: {e}")


def main():
    """Main perft testing function"""
    print("Initializing V7P3R Perft Benchmark System...")
    
    tester = V7P3RPerftTester()
    
    try:
        results = tester.run_comprehensive_benchmark()
        tester.print_summary_report(results)
        tester.save_results(results)
        
        print("\n🎯 Perft benchmark complete!")
        print("Results will be used as baseline for V11 Phase 1 optimization.")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()