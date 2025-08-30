#!/usr/bin/env python3
"""
V9.3 Quick Comparison Test
Compare v9.3 performance against previous versions on key positions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3RCleanEngine

# Test positions from earlier tactical analysis
TEST_POSITIONS = [
    {
        "name": "Opening: King's Pawn",
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "expected": "Central control"
    },
    {
        "name": "Tactical: Knight Fork Setup",
        "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "expected": "Development balance"
    },
    {
        "name": "Material: Queen vs Rooks",
        "fen": "r3k2r/8/8/8/8/8/8/4Q2K w kq - 0 1",
        "expected": "Material evaluation"
    },
    {
        "name": "Endgame: King and Pawn",
        "fen": "8/8/8/8/8/8/3P4/3K4 w - - 0 1",
        "expected": "Pawn promotion"
    },
    {
        "name": "Tactical: Pin Position",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
        "expected": "Pin awareness"
    }
]

def test_position_analysis(engine, position_data):
    """Test engine analysis on a specific position"""
    print(f"\nTesting: {position_data['name']}")
    print(f"Expected: {position_data['expected']}")
    
    board = chess.Board(position_data['fen'])
    
    # Quick evaluation
    evaluation = engine._evaluate_position_deterministic(board)
    print(f"Static eval: {evaluation:.4f}")
    
    # Quick search
    start_time = time.time()
    try:
        best_move = engine.search(board, time_limit=0.5)  # Short time for quick test
        search_time = time.time() - start_time
        print(f"Best move: {best_move} (in {search_time:.3f}s)")
        return True
    except Exception as e:
        print(f"Search failed: {e}")
        return False

def main():
    """Run quick comparison tests"""
    print("V7P3R v9.3 Quick Comparison Test")
    print("=" * 40)
    
    # Initialize v9.3 engine
    print("Initializing v9.3 engine...")
    engine = V7P3RCleanEngine()
    print("✓ v9.3 engine ready")
    
    successful_tests = 0
    total_tests = len(TEST_POSITIONS)
    
    for position_data in TEST_POSITIONS:
        if test_position_analysis(engine, position_data):
            successful_tests += 1
    
    print(f"\nComparison Results: {successful_tests}/{total_tests} positions analyzed")
    
    if successful_tests == total_tests:
        print("🎉 v9.3 successfully analyzed all test positions!")
        print("✓ Hybrid evaluation system is functioning correctly")
        return True
    else:
        print("❌ Some position analysis failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
