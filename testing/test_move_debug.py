#!/usr/bin/env python3
"""
Debug Move Change Issue
Test why d2d4 switches to g1h1 at depth 4
"""

import chess
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3RCleanEngine

def test_move_change():
    """Debug the specific move change issue"""
    
    print("🔍 DEBUGGING MOVE CHANGE: d2d4 -> g1h1")
    print("=" * 50)
    
    # Complex middle position where the switch happens
    board = chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    
    print(f"Position: {board.fen()}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    
    engine = V7P3RCleanEngine()
    
    # Test both moves at different depths
    moves_to_test = [
        chess.Move.from_uci("d2d4"),  # The move found at depths 1-3
        chess.Move.from_uci("g1h1"),  # The move found at depth 4
    ]
    
    print(f"\n📊 Comparing moves at different depths:")
    
    for depth in range(1, 5):
        print(f"\n--- DEPTH {depth} ---")
        
        for move in moves_to_test:
            if move in board.legal_moves:
                # Test this specific move
                board.push(move)
                
                if depth == 1:
                    score = -engine._evaluate_position(board)
                else:
                    score, _ = engine._unified_search(board, depth-1, -99999, 99999)
                    score = -score
                
                board.pop()
                
                print(f"{move}: {score:.0f}")
                
                # Reset node count for fair comparison
                engine.nodes_searched = 0
        
        # Also run full search to see what it picks
        print(f"\nFull search at depth {depth}:")
        engine.nodes_searched = 0
        score, best_move = engine._unified_search(board, depth, -99999, 99999)
        print(f"Best: {best_move} with score {score:.0f}")
        print(f"Nodes: {engine.nodes_searched}")

if __name__ == "__main__":
    test_move_change()
