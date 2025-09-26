#!/usr/bin/env python3
"""
V7P3R v12.2 Evaluation Profiler
Target the specific evaluation bottlenecks identified
"""

import sys
import time
import chess
import cProfile
import pstats
import io

sys.path.append('src')
from v7p3r import V7P3REngine

def profile_evaluation_functions():
    """Profile just the evaluation functions to find bottlenecks"""
    
    print("V7P3R v12.2 Evaluation Function Profiler")
    print("=" * 50)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    print("Profiling individual evaluation components...")
    
    # Profile position evaluation
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run evaluation many times to get meaningful data
    for _ in range(1000):
        score = engine._evaluate_position(board)
    
    profiler.disable()
    
    # Get stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(15)
    
    output = stream.getvalue()
    print("\nTop evaluation function calls:")
    print("-" * 40)
    
    lines = output.split('\n')
    for line in lines:
        if ('fen(' in line or 'epd(' in line or 'board_fen' in line or 
            'evaluate' in line or 'king_safety' in line or 'piece_at' in line):
            print(line.strip())
    
    return output

def analyze_fen_usage():
    """Analyze where FEN strings are being generated"""
    
    print("\n" + "=" * 50)
    print("FEN USAGE ANALYSIS")
    print("=" * 50)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    print("Testing FEN generation performance...")
    
    # Time FEN generation
    start_time = time.perf_counter()
    for _ in range(10000):
        fen = board.fen()
    elapsed = time.perf_counter() - start_time
    
    print(f"10,000 FEN generations: {elapsed:.3f}s ({elapsed/10000*1000:.3f}ms each)")
    
    # Check where FEN is called in evaluation
    print("\nFEN usage in evaluation:")
    print("- Zobrist hashing: position keys")
    print("- Transposition table: position lookup") 
    print("- Position caching: evaluation cache")
    
    return elapsed

def suggest_optimizations():
    """Suggest specific optimizations based on profiling"""
    
    print("\n" + "=" * 50)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 50)
    
    print("🎯 HIGH IMPACT optimizations:")
    print("1. Cache FEN strings - avoid regenerating same position FEN")
    print("2. Use Zobrist hash directly for position keys instead of FEN")
    print("3. Simplify king safety evaluation - reduce piece_at() calls")
    print("4. Cache evaluation results more aggressively")
    
    print("\n🔧 IMPLEMENTATION suggestions:")
    print("1. Add position hash cache in _evaluate_position()")
    print("2. Replace FEN-based keys with Zobrist hash keys")
    print("3. Reduce king safety calculation frequency")
    print("4. Profile move generation next")

def main():
    """Run evaluation profiling analysis"""
    
    # Profile evaluation functions
    profile_output = profile_evaluation_functions()
    
    # Analyze FEN usage
    fen_time = analyze_fen_usage()
    
    # Suggest optimizations
    suggest_optimizations()

if __name__ == "__main__":
    main()