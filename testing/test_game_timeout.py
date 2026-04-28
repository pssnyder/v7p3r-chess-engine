#!/usr/bin/env python3
"""
Single game test with v19.5.2 playing itself
Strict time monitoring to catch timeout issues
"""

import sys
import time
import chess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from v7p3r import V7P3REngine

def play_test_game():
    """Play one game with 30s per move limit"""
    white = V7P3REngine()
    black = V7P3REngine()
    
    board = chess.Board()
    move_num = 1
    
    print("Playing v19.5.2 vs v19.5.2 with 30s per move limit\n")
    
    while not board.is_game_over() and move_num <= 50:  # Max 50 moves
        engine = white if board.turn == chess.WHITE else black
        color = "White" if board.turn == chess.WHITE else "Black"
        
        print(f"Move {move_num} ({color})...", end='', flush=True)
        
        start = time.time()
        move = engine.search(board, time_limit=30.0)
        elapsed = time.time() - start
        
        print(f" {move.uci()} ({elapsed:.2f}s)", end='')
        
        if elapsed > 32.0:  # 2s grace period
            print(f" ⚠️ TIMEOUT! ({elapsed:.2f}s > 30s limit)")
            print(f"\nFAILURE: Engine exceeded time limit on move {move_num}")
            print(f"Position: {board.fen()}")
            return False
        else:
            print(f" ✓")
        
        board.push(move)
        
        if board.turn == chess.BLACK:
            move_num += 1
    
    print(f"\nSUCCESS: Completed {move_num-1} moves without timeout!")
    print(f"Result: {board.result()}")
    return True

if __name__ == "__main__":
    success = play_test_game()
    sys.exit(0 if success else 1)
