#!/usr/bin/env python3
"""
V7P3R v13.0 Performance Profiler
Analyzes engine performance to identify bottlenecks and optimization opportunities
"""

import cProfile
import pstats
import io
import time
import chess
import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine


class V7P3RProfiler:
    """Comprehensive profiling system for V7P3R engine"""
    
    def __init__(self):
        self.results = {}
        self.engine = V7P3REngine()
        
    def profile_search_depth(self, position_fen=None, depths=[1, 2, 3, 4]):
        """Profile search performance at different depths"""
        print("🔍 Profiling search depth performance...")
        
        board = chess.Board(position_fen) if position_fen else chess.Board()
        
        for depth in depths:
            print(f"  Testing depth {depth}...")
            
            # Profile this depth
            profiler = cProfile.Profile()
            start_time = time.time()
            
            profiler.enable()
            self.engine.default_depth = depth
            best_move = self.engine.search(board, time_limit=30.0)
            profiler.disable()
            
            elapsed = time.time() - start_time
            
            # Analyze results
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            
            self.results[f'depth_{depth}'] = {
                'elapsed_time': elapsed,
                'best_move': str(best_move),
                'profiling_data': s.getvalue()
            }
            
            print(f"    Depth {depth}: {elapsed:.2f}s")
    
    def profile_position_types(self):
        """Profile different types of chess positions"""
        print("🏁 Profiling different position types...")
        
        test_positions = {
            'opening': 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
            'middlegame': 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4',
            'endgame': '8/2k5/3p4/p2P1p2/P2P1P2/8/2K5/8 w - - 0 1',
            'tactical': 'r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4'
        }
        
        for pos_type, fen in test_positions.items():
            print(f"  Testing {pos_type} position...")
            
            board = chess.Board(fen)
            profiler = cProfile.Profile()
            start_time = time.time()
            
            profiler.enable()
            best_move = self.engine.search(board, time_limit=5.0)
            profiler.disable()
            
            elapsed = time.time() - start_time
            
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(15)
            
            self.results[f'position_{pos_type}'] = {
                'fen': fen,
                'elapsed_time': elapsed,
                'best_move': str(best_move),
                'profiling_data': s.getvalue()
            }
            
            print(f"    {pos_type.capitalize()}: {elapsed:.2f}s")
    
    def profile_move_generation(self):
        """Profile move generation and ordering"""
        print("♟️  Profiling move generation and ordering...")
        
        # Test position with many moves
        board = chess.Board('r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4')
        
        # Profile move generation
        profiler = cProfile.Profile()
        start_time = time.time()
        
        profiler.enable()
        for _ in range(1000):  # Generate moves many times
            moves = list(board.legal_moves)
        profiler.disable()
        
        elapsed = time.time() - start_time
        
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(10)
        
        self.results['move_generation'] = {
            'iterations': 1000,
            'elapsed_time': elapsed,
            'moves_per_position': len(list(board.legal_moves)),
            'profiling_data': s.getvalue()
        }
        
        print(f"    Move generation: {elapsed:.3f}s for 1000 iterations")
    
    def profile_evaluation_components(self):
        """Profile individual evaluation components"""
        print("⚖️  Profiling evaluation components...")
        
        board = chess.Board('r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4')
        
        # Test tactical detection
        if hasattr(self.engine, 'tactical_detector'):
            profiler = cProfile.Profile()
            start_time = time.time()
            
            profiler.enable()
            for _ in range(100):
                self.engine.tactical_detector.detect_all_tactical_patterns(board, True)
            profiler.disable()
            
            elapsed = time.time() - start_time
            
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(10)
            
            self.results['tactical_detection'] = {
                'iterations': 100,
                'elapsed_time': elapsed,
                'profiling_data': s.getvalue()
            }
            
            print(f"    Tactical detection: {elapsed:.3f}s for 100 iterations")
        
        # Test dynamic evaluation
        if hasattr(self.engine, 'dynamic_evaluator'):
            profiler = cProfile.Profile()
            start_time = time.time()
            
            profiler.enable()
            for _ in range(100):
                self.engine.dynamic_evaluator.evaluate_dynamic_position_value(board, True)
            profiler.disable()
            
            elapsed = time.time() - start_time
            
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(10)
            
            self.results['dynamic_evaluation'] = {
                'iterations': 100,
                'elapsed_time': elapsed,
                'profiling_data': s.getvalue()
            }
            
            print(f"    Dynamic evaluation: {elapsed:.3f}s for 100 iterations")
    
    def analyze_hot_spots(self):
        """Analyze profiling results to identify hot spots"""
        print("\n🔥 Analyzing hot spots...")
        
        hot_spots = []
        
        for test_name, data in self.results.items():
            if 'profiling_data' in data:
                lines = data['profiling_data'].split('\n')
                
                # Extract function call statistics
                for line in lines:
                    if 'cumtime' in line or 'tottime' in line:
                        continue
                    if line.strip() and not line.startswith(' '):
                        continue
                    if 'function calls' in line:
                        continue
                        
                    # Parse function statistics
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        try:
                            cumtime = float(parts[3])
                            if cumtime > 0.01:  # Only functions taking > 10ms
                                function_name = ' '.join(parts[5:])
                                hot_spots.append({
                                    'test': test_name,
                                    'function': function_name,
                                    'cumtime': cumtime,
                                    'calls': int(parts[0])
                                })
                        except (ValueError, IndexError):
                            continue
        
        # Sort by cumulative time
        hot_spots.sort(key=lambda x: x['cumtime'], reverse=True)
        
        print("\n📊 Top Performance Hot Spots:")
        print("-" * 80)
        print(f"{'Function':<50} {'Test':<20} {'Time (s)':<10} {'Calls':<10}")
        print("-" * 80)
        
        for i, spot in enumerate(hot_spots[:20]):  # Top 20
            print(f"{spot['function'][:49]:<50} {spot['test']:<20} {spot['cumtime']:<10.3f} {spot['calls']:<10}")
        
        return hot_spots
    
    def generate_optimization_report(self, hot_spots):
        """Generate optimization recommendations"""
        print("\n💡 Optimization Recommendations:")
        print("=" * 60)
        
        recommendations = []
        
        # Analyze patterns in hot spots
        evaluation_time = sum(s['cumtime'] for s in hot_spots if 'evaluate' in s['function'].lower())
        search_time = sum(s['cumtime'] for s in hot_spots if any(word in s['function'].lower() for word in ['search', 'minimax', 'alpha']))
        move_time = sum(s['cumtime'] for s in hot_spots if any(word in s['function'].lower() for word in ['move', 'legal', 'generate']))
        tactical_time = sum(s['cumtime'] for s in hot_spots if 'tactical' in s['function'].lower())
        
        print(f"🎯 Time Distribution Analysis:")
        print(f"  Evaluation functions: {evaluation_time:.2f}s")
        print(f"  Search functions: {search_time:.2f}s") 
        print(f"  Move generation: {move_time:.2f}s")
        print(f"  Tactical detection: {tactical_time:.2f}s")
        
        if evaluation_time > search_time * 0.5:
            recommendations.append("HIGH PRIORITY: Evaluation is taking too much time relative to search")
        
        if tactical_time > 0.1:
            recommendations.append("MEDIUM PRIORITY: Tactical detection may need caching/optimization")
            
        if move_time > search_time * 0.3:
            recommendations.append("MEDIUM PRIORITY: Move generation overhead is high")
        
        # Look for specific problematic functions
        for spot in hot_spots[:10]:
            func = spot['function'].lower()
            if 'detect_all_tactics' in func and spot['cumtime'] > 0.05:
                recommendations.append(f"OPTIMIZE: detect_all_tactics taking {spot['cumtime']:.3f}s - consider selective detection")
            elif 'get_piece_values' in func and spot['cumtime'] > 0.03:
                recommendations.append(f"OPTIMIZE: get_piece_values taking {spot['cumtime']:.3f}s - consider caching")
            elif 'legal_moves' in func and spot['calls'] > 1000:
                recommendations.append(f"OPTIMIZE: legal_moves called {spot['calls']} times - cache move generation")
        
        print(f"\n🚀 Specific Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return recommendations
    
    def save_results(self):
        """Save profiling results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"v7p3r_performance_profile_{timestamp}.json"
        
        # Prepare results for JSON (remove profiling_data which is too large)
        json_results = {}
        for key, value in self.results.items():
            json_results[key] = {k: v for k, v in value.items() if k != 'profiling_data'}
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename
    
    def run_full_profile(self):
        """Run complete profiling suite"""
        print("🏃‍♂️ Running V7P3R v13.0 Performance Profile...")
        print("=" * 60)
        
        # Run all profiling tests
        self.profile_search_depth(depths=[1, 2, 3])
        self.profile_position_types()
        self.profile_move_generation()
        self.profile_evaluation_components()
        
        # Analyze results
        hot_spots = self.analyze_hot_spots()
        recommendations = self.generate_optimization_report(hot_spots)
        
        # Save results
        filename = self.save_results()
        
        print(f"\n✅ Profiling complete! Check {filename} for detailed results.")
        return hot_spots, recommendations


def main():
    profiler = V7P3RProfiler()
    hot_spots, recommendations = profiler.run_full_profile()
    
    print(f"\n🎯 Next Steps:")
    print("1. Review hot spots and focus on optimizing the highest time consumers")
    print("2. Consider implementing caching for frequently called functions")
    print("3. Optimize move ordering to reduce search tree size")
    print("4. Test performance improvements incrementally")


if __name__ == "__main__":
    main()