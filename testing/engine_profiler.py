#!/usr/bin/env python3
"""
V7P3R Code Profiler
Uses Python's cProfile to identify performance bottlenecks in the engine
"""

import cProfile
import pstats
import io
import sys
import os
import chess

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def profile_engine_search():
    """Profile the engine during a typical search"""
    print("🔍 Profiling V7P3R Engine Search Performance...")
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Initialize engine
    engine = V7P3REngine()
    
    # Set up test position
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    
    print(f"Position: {board.fen()}")
    print("Starting 5-second search...")
    
    # Profile the search
    profiler.enable()
    
    # Run a typical search (5 seconds)
    try:
        best_move = engine.search(board, time_limit=5.0)
        print(f"Best move: {best_move}, Score: {score}")
    except Exception as e:
        print(f"Search error: {e}")
    
    profiler.disable()
    
    # Analyze results
    analyze_profile_results(profiler)

def profile_engine_evaluation():
    """Profile just the evaluation function"""
    print("\n🔍 Profiling V7P3R Evaluation Function...")
    
    profiler = cProfile.Profile()
    engine = V7P3REngine()
    
    # Test various positions
    positions = [
        chess.Board(),  # Starting position
        chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),  # Kiwipete
        chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),  # Endgame
    ]
    
    print("Evaluating 1000 positions...")
    
    profiler.enable()
    
    # Evaluate positions multiple times
    for _ in range(1000):
        for board in positions:
            try:
                score = engine._evaluate_position(board)
            except Exception as e:
                print(f"Evaluation error: {e}")
                break
    
    profiler.disable()
    
    analyze_profile_results(profiler, "evaluation")

def analyze_profile_results(profiler, test_name="search"):
    """Analyze and display profiling results"""
    print(f"\n📊 PROFILING RESULTS - {test_name.upper()}")
    print("=" * 60)
    
    # Create stats object
    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream)
    
    # Sort by cumulative time (most expensive functions first)
    stats.sort_stats('cumulative')
    
    print("🔥 TOP FUNCTIONS BY CUMULATIVE TIME:")
    print("-" * 40)
    
    # Print top 15 functions
    stats.print_stats(15)
    
    # Get the string output
    output = stats_stream.getvalue()
    print(output)
    
    # Sort by total time per call
    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream)
    stats.sort_stats('tottime')
    
    print("\n⏱️ TOP FUNCTIONS BY TOTAL TIME:")
    print("-" * 40)
    stats.print_stats(10)
    
    output = stats_stream.getvalue()
    print(output)
    
    # Look for specific bottlenecks
    print("\n🎯 BOTTLENECK ANALYSIS:")
    print("-" * 25)
    
    # Check for common performance issues in the output
    if 'evaluate' in output:
        print("🔍 Found evaluation functions in top performers")
    if 'castling' in output:
        print("🏰 Found castling functions in top performers") 
    if 'nudge' in output:
        print("🧠 Found nudge functions in top performers")
    
    # Save detailed report
    save_profile_report(profiler, test_name)

def save_profile_report(profiler, test_name):
    """Save detailed profile report to file"""
    filename = f"profile_report_{test_name}.txt"
    
    with open(filename, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        f.write(f"V7P3R Engine Profile Report - {test_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("TOP 50 FUNCTIONS BY CUMULATIVE TIME:\n")
        f.write("-" * 40 + "\n")
        stats.sort_stats('cumulative')
        stats.print_stats(50)
        
        f.write("\n\nTOP 30 FUNCTIONS BY TOTAL TIME:\n")
        f.write("-" * 40 + "\n")
        stats.sort_stats('tottime')
        stats.print_stats(30)
    
    print(f"\n💾 Detailed report saved to: {filename}")

if __name__ == "__main__":
    print("V7P3R Performance Profiler")
    print("=" * 30)
    
    # Profile search performance
    profile_engine_search()
    
    # Profile evaluation performance  
    profile_engine_evaluation()
    
    print("\n✅ Profiling complete!")
    print("Check the generated .txt files for detailed function-level timing data.")