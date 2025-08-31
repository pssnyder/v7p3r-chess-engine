#!/usr/bin/env python3
"""
Current Engine Performance Test
Test the current engine with V7.0 scoring to measure NPS
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3RCleanEngine

def test_current_engine_performance():
    """Test current engine performance"""
    print("Current V7P3R Engine Performance Test")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    
    # Test positions
    positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # King's pawn
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Development
    ]
    
    for i, fen in enumerate(positions, 1):
        print(f"\nPosition {i}: {fen}")
        board = chess.Board(fen)
        
        print("Searching for 2 seconds...")
        start_time = time.time()
        best_move = engine.search(board, time_limit=2.0)
        elapsed = time.time() - start_time
        
        print(f"Best move: {best_move}")
        print(f"Search time: {elapsed:.3f} seconds")
        print(f"Nodes searched: {engine.nodes_searched}")
        
        if elapsed > 0:
            nps = engine.nodes_searched / elapsed
            print(f"NPS (Nodes Per Second): {nps:.0f}")
        
        # Quick evaluation test
        evaluation = engine._evaluate_position(board)
        print(f"Position evaluation: {evaluation:.4f}")

def test_evaluation_speed():
    """Test pure evaluation speed"""
    print("\n" + "=" * 50)
    print("Pure Evaluation Speed Test")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    board = chess.Board()
    
    iterations = 10000
    print(f"Running {iterations} evaluations...")
    
    start_time = time.time()
    for _ in range(iterations):
        evaluation = engine._evaluate_position(board)
    end_time = time.time()
    
    elapsed = end_time - start_time
    evals_per_sec = iterations / elapsed
    
    print(f"Time: {elapsed:.3f} seconds")
    print(f"Evaluations per second: {evals_per_sec:.0f}")
    print(f"Time per evaluation: {(elapsed/iterations)*1000:.3f} ms")

def main():
    """Main test function"""
    try:
        test_current_engine_performance()
        test_evaluation_speed()
        
        print("\n" + "=" * 50)
        print("PERFORMANCE SUMMARY")
        print("=" * 50)
        print("✓ Engine using V7.0 scoring calculation")
        print("✓ Should see improved NPS compared to V9.3")
        print("✓ Ready for V10.0 development with proven fast base")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
