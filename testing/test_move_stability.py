#!/usr/bin/env python3
"""
Test Move Stability Issue
Reproduce the d2d4 -> g1h1 switch problem
"""

import chess
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3RCleanEngine

def test_problematic_position():
    """Test the exact position where move switches from d2d4 to g1h1"""
    
    print("🔍 MOVE STABILITY TEST")
    print("=" * 40)
    
    # This is the complex middle position from the performance test
    board = chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    
    print(f"Position: {board.fen()}")
    print(f"Legal moves: {[str(m) for m in board.legal_moves]}")
    
    engine = V7P3RCleanEngine()
    
    # Test each depth individually to see where the switch happens
    for depth in range(1, 6):
        print(f"\n📊 Testing depth {depth}:")
        
        # Reset engine state
        engine.nodes_searched = 0
        engine.evaluation_cache.clear()
        
        # Search at this specific depth
        score, move = engine._unified_search(board, depth, -99999, 99999)
        
        print(f"   Depth {depth}: score={score:.0f}, move={move}, nodes={engine.nodes_searched}")
        
        # Also test the moves individually to see their scores
        if depth <= 2:  # Don't do this for deep searches
            print(f"   Top move scores at depth {depth}:")
            legal_moves = list(board.legal_moves)
            move_scores = []
            
            for move in legal_moves[:8]:  # Test first 8 moves
                board.push(move)
                if depth == 1:
                    move_score = -engine._evaluate_position(board)
                else:
                    move_score, _ = engine._unified_search(board, depth-1, -99999, 99999)
                    move_score = -move_score
                board.pop()
                
                move_scores.append((move_score, move))
            
            # Sort by score and show top moves
            move_scores.sort(reverse=True)
            for i, (score, move) in enumerate(move_scores[:5]):
                print(f"     {i+1}. {move}: {score:.0f}")

if __name__ == "__main__":
    test_problematic_position()
