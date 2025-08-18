#!/usr/bin/env python3
"""
Test V7P3R Engine v4.3 Simplifications
Verify that the simplified engine works correctly and shows improved performance.
"""

import sys
import os
import time

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.abspath('..'))

# Import the engine modules
from src.v7p3r_config import load_config
from src.v7p3r_engine import V7P3REngine
from v7p3r_transposition import get_transposition_table
import chess

def test_simplified_engine():
    """Test the simplified engine functionality."""
    print("Testing V7P3R Engine v4.3 Simplifications")
    print("=" * 50)
    
    # Load configuration
    config = load_config("config_default.json")
    
    # Create engine instance
    engine = V7P3REngine(config)
    
    # Test positions
    test_positions = [
        ("Start position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Middle game", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"),
        ("Tactical position", "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    ]
    
    total_time = 0
    tt = get_transposition_table()
    
    for name, fen in test_positions:
        print(f"\nTesting: {name}")
        print(f"FEN: {fen}")
        
        board = chess.Board(fen)
        
        # Test with 3 second time limit
        start_time = time.time()
        best_move = engine.get_best_move(board, time_limit=3.0)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        print(f"Best move: {best_move}")
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"Nodes searched: {engine.search_controller.nodes_searched}")
        print(f"Cutoffs: {engine.search_controller.cutoffs}")
        
        # Clear transposition table for clean test
        tt.clear()
    
    print(f"\nTotal time for all positions: {total_time:.2f} seconds")
    print(f"Average time per position: {total_time/len(test_positions):.2f} seconds")
    
    # Test transposition table stats
    tt_stats = tt.get_stats()
    print(f"\nTransposition Table Stats: {tt_stats}")

def test_move_ordering():
    """Test the simplified move ordering."""
    print("\nTesting Simplified Move Ordering")
    print("=" * 30)
    
    from src.v7p3r_move_ordering import MoveOrdering
    
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    move_ordering = MoveOrdering()
    
    legal_moves = list(board.legal_moves)
    print(f"Legal moves: {len(legal_moves)}")
    
    ordered_moves = move_ordering.order_moves(board, legal_moves)
    print(f"Ordered moves: {len(ordered_moves)}")
    
    # Show top 5 moves with scores
    print("Top 5 moves:")
    for i, move in enumerate(ordered_moves[:5]):
        score = move_ordering._score_move(board, move)
        print(f"  {i+1}. {move}: {score}")

if __name__ == "__main__":
    try:
        test_simplified_engine()
        test_move_ordering()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
