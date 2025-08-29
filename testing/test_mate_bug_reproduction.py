#!/usr/bin/env python3
"""
Test to reproduce the +M500 mate evaluation bug from the tournament PGN
"""

import sys
import os
sys.path.append('src')

import chess
from v7p3r import V7P3RCleanEngine


def test_mate_position():
    """Test the position where V7P3R showed +M500"""
    # Position after: 1. e3 e5 2. Bd3 Qg5 3. Qf3 Nc6 4. Nh3 Qh6 5. Nc3 Nb4 6. Ke2 Nxd3 7. Kxd3 Qa6+ 8. Ke4 Qc4+ 9. d4 d6
    # This is where V7P3R evaluated +M500/5
    
    moves = ["e3", "e5", "Bd3", "Qg5", "Qf3", "Nc6", "Nh3", "Qh6", "Nc3", "Nb4", "Ke2", "Nxd3", "Kxd3", "Qa6+", "Ke4", "Qc4+", "d4", "d6"]
    
    board = chess.Board()
    engine = V7P3RCleanEngine()
    
    print("Reproducing the +M500 bug position...")
    print("Starting position:", board.fen())
    
    # Play the moves to reach the problem position
    for move_san in moves:
        try:
            move = board.parse_san(move_san)
            board.push(move)
            print(f"After {move_san}: {board.fen()}")
        except Exception as e:
            print(f"Error parsing move {move_san}: {e}")
            break
    
    print(f"\nFinal position FEN: {board.fen()}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"Is checkmate: {board.is_checkmate()}")
    print(f"Is check: {board.is_check()}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    
    # Test the engine evaluation at this position
    print("\n=== Engine Analysis ===")
    best_move = engine.search(board, time_limit=2.0)
    print(f"Best move: {best_move}")
    
    # Test specific mate score formatting
    print("\n=== Testing mate score formatting ===")
    test_scores = [29999, 29998, 29997, 29990, 29500, 28000, 25000]
    for score in test_scores:
        formatted = engine._format_uci_score(score, 5)
        print(f"Score {score} -> {formatted}")


if __name__ == "__main__":
    test_mate_position()
