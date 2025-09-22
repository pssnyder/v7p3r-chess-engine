#!/usr/bin/env python3
"""
V7P3R v11.4 Performance Profiler
Quick analysis of current performance characteristics
"""

import sys
import os
import time
import cProfile
import pstats
import io
from contextlib import redirect_stdout

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import chess
from v7p3r import V7P3REngine

class V11_4_Profiler:
    def __init__(self):
        self.engine = V7P3REngine()
        self.test_positions = [
            # Opening position
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            # Middlegame tactical
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            # Complex middlegame
            "r1bq1rk1/2p1bppp/p1n2n2/1p1pp3/3PP3/1BP2N2/PP3PPP/RNBQR1K1 w - - 0 9",
            # Endgame
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
        ]
    
    def profile_search_depth(self, fen, depth=4):
        """Profile a single search at specified depth"""
        board = chess.Board(fen)
        
        # Time the search - V11.5 FIX: Call with proper parameters
        start_time = time.time()
        try:
            # FIXED: Call search with depth parameter in correct position
            move, score, search_info = self.engine.search(board, time_limit=10.0, depth=depth)
            end_time = time.time()
            search_time = end_time - start_time
            
            nodes = search_info.get('nodes', 0) if search_info else 0
            nps = nodes / search_time if search_time > 0 else 0
            
            return {
                'move': str(move) if move else None,
                'score': score,
                'time': search_time,
                'nodes': nodes,
                'nps': nps,
                'depth': depth
            }
        except Exception as e:
            return {
                'error': str(e),
                'time': time.time() - start_time,
                'depth': depth
            }
    
    def detailed_function_profile(self, fen, depth=3):
        """Run cProfile on a single search"""
        board = chess.Board(fen)
        
        # Create profiler
        profiler = cProfile.Profile()
        
        # Profile the search - V11.5 FIX: Use correct parameters
        profiler.enable()
        try:
            move, score, search_info = self.engine.search(board, time_limit=5.0, depth=depth)
        except:
            pass
        profiler.disable()
        
        # Get statistics
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        return stats_buffer.getvalue()
    
    def run_comprehensive_analysis(self):
        """Run complete performance analysis"""
        print("=" * 60)
        print("V7P3R v11.4 PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Test different depths
        depths = [2, 3, 4, 5]
        position_names = ["Opening", "Tactical", "Complex", "Endgame"]
        
        results = {}
        
        for i, (pos_name, fen) in enumerate(zip(position_names, self.test_positions)):
            print(f"\n📍 {pos_name.upper()} POSITION:")
            print(f"FEN: {fen}")
            print("-" * 50)
            
            results[pos_name] = {}
            
            for depth in depths:
                result = self.profile_search_depth(fen, depth)
                results[pos_name][depth] = result
                
                if 'error' in result:
                    print(f"Depth {depth}: ERROR - {result['error']}")
                else:
                    print(f"Depth {depth}: {result['move']:8} | "
                          f"{result['time']:6.3f}s | "
                          f"{result['nodes']:8,} nodes | "
                          f"{result['nps']:8,.0f} NPS | "
                          f"Score: {result['score']}")
        
        # Calculate averages
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        total_nps = []
        total_times = []
        
        for pos_name, pos_results in results.items():
            pos_nps = []
            pos_times = []
            
            for depth, result in pos_results.items():
                if 'error' not in result:
                    pos_nps.append(result['nps'])
                    pos_times.append(result['time'])
                    total_nps.append(result['nps'])
                    total_times.append(result['time'])
            
            if pos_nps:
                avg_nps = sum(pos_nps) / len(pos_nps)
                avg_time = sum(pos_times) / len(pos_times)
                print(f"{pos_name:12}: {avg_nps:8,.0f} NPS avg | {avg_time:.3f}s avg")
        
        if total_nps:
            overall_avg_nps = sum(total_nps) / len(total_nps)
            overall_avg_time = sum(total_times) / len(total_times)
            print("-" * 60)
            print(f"{'OVERALL':12}: {overall_avg_nps:8,.0f} NPS avg | {overall_avg_time:.3f}s avg")
        
        # Detailed profiling on tactical position
        print("\n" + "=" * 60)
        print("DETAILED FUNCTION PROFILING (Tactical Position, Depth 3)")
        print("=" * 60)
        
        profile_output = self.detailed_function_profile(self.test_positions[1], depth=3)
        print(profile_output)
        
        return results

def main():
    profiler = V11_4_Profiler()
    results = profiler.run_comprehensive_analysis()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()