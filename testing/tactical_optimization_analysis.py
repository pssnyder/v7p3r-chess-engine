#!/usr/bin/env python3
"""
V7P3R Performance Optimizer - V11.3 Tactical Integration Focus
Identifies optimization opportunities with tactical patterns enabled
"""

import sys
import time
import chess
import cProfile
import pstats
import io
from typing import Dict, List, Tuple
sys.path.append('src')

from v7p3r import V7P3REngine

class V7P3ROptimizer:
    """Performance optimizer focused on tactical integration efficiency"""
    
    def __init__(self):
        self.engine = V7P3REngine()
        
    def profile_tactical_overhead(self) -> Dict:
        """Measure the performance impact of tactical pattern detection"""
        
        test_position = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
        board = chess.Board(test_position)
        
        print("🔍 Profiling Tactical Pattern Overhead...")
        print("=" * 50)
        
        # Test with tactical patterns enabled
        print("\n1. Testing WITH tactical patterns (current state):")
        start_time = time.time()
        for _ in range(100):
            score = self.engine._evaluate_position(board)
        with_tactical_time = time.time() - start_time
        with_tactical_avg = with_tactical_time / 100
        
        print(f"   100 evaluations: {with_tactical_time:.3f}s")
        print(f"   Average per eval: {with_tactical_avg*1000:.2f}ms")
        
        # Temporarily disable tactical patterns for comparison
        print("\n2. Testing WITHOUT tactical patterns (comparison):")
        
        # Backup current detector
        original_detector = self.engine.tactical_pattern_detector
        
        # Create a mock detector that returns 0
        class MockTacticalDetector:
            def evaluate_tactical_patterns(self, board, color):
                return 0.0
        
        self.engine.tactical_pattern_detector = MockTacticalDetector()
        
        start_time = time.time()
        for _ in range(100):
            score = self.engine._evaluate_position(board)
        without_tactical_time = time.time() - start_time
        without_tactical_avg = without_tactical_time / 100
        
        # Restore original detector
        self.engine.tactical_pattern_detector = original_detector
        
        print(f"   100 evaluations: {without_tactical_time:.3f}s")
        print(f"   Average per eval: {without_tactical_avg*1000:.2f}ms")
        
        # Calculate overhead
        tactical_overhead = with_tactical_time - without_tactical_time
        overhead_percent = (tactical_overhead / without_tactical_time) * 100
        
        print(f"\n3. Tactical Pattern Overhead Analysis:")
        print(f"   Overhead: {tactical_overhead:.3f}s ({overhead_percent:.1f}%)")
        print(f"   Per evaluation: {(tactical_overhead/100)*1000:.2f}ms")
        
        return {
            'with_tactical_time': with_tactical_time,
            'without_tactical_time': without_tactical_time,
            'overhead_seconds': tactical_overhead,
            'overhead_percent': overhead_percent,
            'overhead_per_eval_ms': (tactical_overhead/100)*1000
        }
    
    def profile_search_bottlenecks(self, depth: int = 5) -> Dict:
        """Profile the search function to identify bottlenecks"""
        
        print(f"\n🔍 Profiling Search Bottlenecks (depth {depth})...")
        print("=" * 50)
        
        test_position = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
        board = chess.Board(test_position)
        
        # Create a profiler
        profiler = cProfile.Profile()
        
        # Profile the search
        profiler.enable()
        start_time = time.time()
        best_move, score = self.engine.search(board, depth, time_limit=5.0)
        search_time = time.time() - start_time
        profiler.disable()
        
        # Analyze the results
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        profile_output = stream.getvalue()
        
        print(f"Search completed in {search_time:.3f}s")
        print(f"Nodes searched: {self.engine.nodes_searched:,}")
        print(f"NPS: {self.engine.nodes_searched/max(search_time, 0.001):,.0f}")
        
        print("\nTop function calls by cumulative time:")
        print("-" * 40)
        
        # Parse and display key metrics
        lines = profile_output.split('\n')
        for line in lines[5:25]:  # Skip header, show top 20
            if line.strip() and 'function calls' not in line:
                print(f"  {line}")
        
        return {
            'search_time': search_time,
            'nodes_searched': self.engine.nodes_searched,
            'nps': self.engine.nodes_searched/max(search_time, 0.001),
            'profile_output': profile_output
        }
    
    def test_time_management_efficiency(self) -> Dict:
        """Test time management for different time controls"""
        
        print(f"\n🔍 Testing Time Management Efficiency...")
        print("=" * 50)
        
        time_controls = [
            ("10+5 Rapid", 600000, 5000),    # 10 minutes + 5 second increment
            ("2+1 Blitz", 120000, 1000),     # 2 minutes + 1 second increment
            ("60s Bullet", 60000, 0)         # 60 seconds, no increment
        ]
        
        test_position = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        board = chess.Board(test_position)
        
        results = {}
        
        for name, time_ms, increment_ms in time_controls:
            print(f"\n{name} ({time_ms//1000}s + {increment_ms//1000}s):")
            
            # Calculate time per move (rough estimate)
            expected_moves_remaining = 40
            time_per_move = (time_ms + increment_ms * expected_moves_remaining) / expected_moves_remaining / 1000
            
            print(f"  Expected time per move: {time_per_move:.1f}s")
            
            # Test 3 moves with this time allocation
            move_times = []
            depths_achieved = []
            
            for move_num in range(3):
                start_time = time.time()
                best_move, score = self.engine.search(board, depth=6, time_limit=time_per_move)
                actual_time = time.time() - start_time
                
                move_times.append(actual_time)
                # Estimate depth based on nodes (rough calculation)
                estimated_depth = min(6, max(3, int(self.engine.nodes_searched / 10000)))
                depths_achieved.append(estimated_depth)
                
                print(f"    Move {move_num+1}: {actual_time:.2f}s, ~depth {estimated_depth}, {self.engine.nodes_searched:,} nodes")
                
                # Make a random legal move to continue
                moves = list(board.legal_moves)
                if moves:
                    board.push(moves[0])
            
            avg_time = sum(move_times) / len(move_times)
            avg_depth = sum(depths_achieved) / len(depths_achieved)
            
            results[name] = {
                'time_per_move_target': time_per_move,
                'avg_actual_time': avg_time,
                'avg_depth': avg_depth,
                'time_efficiency': avg_time / time_per_move,
                'move_times': move_times
            }
            
            # Reset board
            board = chess.Board(test_position)
        
        return results
    
    def generate_optimization_recommendations(self, tactical_overhead: Dict, time_management: Dict) -> List[str]:
        """Generate specific optimization recommendations"""
        
        recommendations = []
        
        # Tactical overhead analysis
        if tactical_overhead['overhead_percent'] > 30:
            recommendations.append(
                f"🔥 HIGH PRIORITY: Tactical overhead is {tactical_overhead['overhead_percent']:.1f}% "
                f"- optimize tactical pattern detection"
            )
        elif tactical_overhead['overhead_percent'] > 15:
            recommendations.append(
                f"⚠️  MEDIUM: Tactical overhead is {tactical_overhead['overhead_percent']:.1f}% "
                f"- consider reducing tactical pattern complexity"
            )
        else:
            recommendations.append(
                f"✅ GOOD: Tactical overhead is {tactical_overhead['overhead_percent']:.1f}% "
                f"- tactical patterns are efficient"
            )
        
        # Time management analysis
        for name, data in time_management.items():
            if data['time_efficiency'] > 1.2:
                recommendations.append(
                    f"⚠️  {name}: Using {data['time_efficiency']:.1f}x target time - reduce search depth"
                )
            elif data['time_efficiency'] < 0.7:
                recommendations.append(
                    f"🚀 {name}: Using only {data['time_efficiency']:.1f}x target time - could search deeper"
                )
            else:
                recommendations.append(
                    f"✅ {name}: Good time usage ({data['time_efficiency']:.1f}x target)"
                )
        
        # Specific optimization suggestions
        if tactical_overhead['overhead_percent'] > 20:
            recommendations.extend([
                "💡 Optimize tactical pattern detection:",
                "   - Use time-budget limits for pattern detection",
                "   - Cache tactical evaluations",
                "   - Reduce pattern complexity in fast games"
            ])
        
        recommendations.extend([
            "💡 General optimizations:",
            "   - Improve transposition table hit rate",
            "   - Optimize move ordering",
            "   - Use iterative deepening more efficiently"
        ])
        
        return recommendations
    
    def run_optimization_analysis(self) -> Dict:
        """Run complete optimization analysis"""
        
        print("V7P3R v11.3 Performance Optimization Analysis")
        print("=" * 60)
        
        # Test tactical overhead
        tactical_results = self.profile_tactical_overhead()
        
        # Test search bottlenecks
        search_results = self.profile_search_bottlenecks()
        
        # Test time management
        time_mgmt_results = self.test_time_management_efficiency()
        
        # Generate recommendations
        recommendations = self.generate_optimization_recommendations(
            tactical_results, time_mgmt_results
        )
        
        print(f"\n📊 Optimization Recommendations:")
        print("=" * 50)
        for rec in recommendations:
            print(rec)
        
        # Save results
        import json
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results = {
            'timestamp': timestamp,
            'tactical_overhead': tactical_results,
            'search_profile': {
                'search_time': search_results['search_time'],
                'nodes_searched': search_results['nodes_searched'],
                'nps': search_results['nps']
            },
            'time_management': time_mgmt_results,
            'recommendations': recommendations
        }
        
        results_file = f"optimization_analysis_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Analysis saved to: {results_file}")
        
        return results

def main():
    """Run the optimization analysis"""
    optimizer = V7P3ROptimizer()
    results = optimizer.run_optimization_analysis()
    
    print(f"\n🎯 Next Steps:")
    print("Based on analysis results, implement the highest priority optimizations")
    print("Then re-run this analysis to measure improvements")

if __name__ == "__main__":
    main()