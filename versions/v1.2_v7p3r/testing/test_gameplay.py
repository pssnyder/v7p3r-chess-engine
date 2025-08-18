#!/usr/bin/env python3
"""
Simple gameplay test for V7P3R Chess Engine v1.2
Tests the engine's ability to play actual moves.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r_engine import V7P3RChessEngine


def test_gameplay():
    """Test engine gameplay capabilities."""
    print("Testing engine gameplay...")
    
    # Initialize engine
    engine = V7P3RChessEngine()
    board = chess.Board()
    
    # Play a few moves
    moves_played = 0
    max_moves = 10
    
    print(f"Starting position: {board.fen()}")
    
    while moves_played < max_moves and not board.is_game_over():
        # Get engine move
        move = engine.get_best_move(board, time_limit=1.0)
        
        if move is None:
            print(f"Engine returned no move at position: {board.fen()}")
            break
            
        if move not in board.legal_moves:
            print(f"Engine returned illegal move: {move}")
            break
            
        # Make the move
        board.push(move)
        moves_played += 1
        
        print(f"Move {moves_played}: {move} -> {board.fen()}")
    
    print(f"Successfully played {moves_played} moves")
    assert moves_played > 0, "Engine should be able to play at least one move"
    print("✓ Gameplay test passed")


if __name__ == "__main__":
    test_gameplay()
