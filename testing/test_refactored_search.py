#!/usr/bin/env python3
"""
Quick test of refactored V7P3R v9.5 search function
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine
import time

def test_refactored_search():
    """Test the new unified search function"""
    engine = V7P3REngine()
    
    # Test positions
    test_positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Middlegame", "r1bq1rk1/ppp2ppp/2n2n2/3p4/2PP4/3BPN2/PP3PPP/RNBQ1RK1 w - - 0 8"),
        ("Tactical position", "r4rk1/pp1bq1bp/3p1np1/2pPp3/2P1P3/2N2NQP/PP1B1PP1/2R2RK1 w - - 0 15")
    ]
    
    print("TESTING REFACTORED V7P3R SEARCH FUNCTION")
    print("=" * 50)
    
    for i, (name, fen) in enumerate(test_positions, 1):
        print(f"\nTest {i}: {name}")
        print(f"FEN: {fen}")
        
        board = chess.Board(fen)
        start_time = time.time()
        
        try:
            # Test with 5 second time limit
            best_move = engine.search(board, 5.0)
            elapsed = time.time() - start_time
            
            print(f"Result: {best_move}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Nodes: {engine.nodes_searched}")
            
            if best_move != chess.Move.null():
                print("✓ SUCCESS: Valid move returned")
            else:
                print("✗ FAILURE: No move returned")
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
            
        print("-" * 30)
    
    print("\nRefactoring test completed!")

if __name__ == "__main__":
    test_refactored_search()
