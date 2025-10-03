#!/usr/bin/env python3
"""
V7P3R Opening Simulation Test
Tests the intelligent nudge system's influence on opening play
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import chess
import chess.engine
from v7p3r import V7P3R

def test_opening_simulation():
    """Test V7P3R opening play with intelligent nudge system"""
    print("V7P3R v12.5 Opening Simulation Test")
    print("="*50)
    
    # Initialize engine
    engine = V7P3R()
    board = chess.Board()
    
    print(f"🤖 V7P3R v{engine.version} initialized")
    print(f"🧠 Nudge system enabled: {hasattr(engine, 'ENABLE_NUDGE_SYSTEM') and engine.ENABLE_NUDGE_SYSTEM}")
    print(f"🎯 Intelligent nudges available: {hasattr(engine, 'intelligent_nudges') and engine.intelligent_nudges is not None}")
    print(f"📚 Legacy nudge database loaded: {hasattr(engine, 'nudge_database') and bool(engine.nudge_database)}")
    print()
    
    moves_played = []
    
    for move_number in range(1, 6):  # Play first 5 moves
        print(f"Move {move_number}:")
        print(f"Position: {board.fen()}")
        print(f"Board:\n{board}")
        
        # Get engine's move with timing
        import time
        start_time = time.time()
        result = engine.search(board, chess.engine.Limit(time=2.0))
        search_time = time.time() - start_time
        
        if result and result.move:
            move = result.move
            moves_played.append(move)
            
            # Get nudge bonus for this move
            nudge_bonus = engine._get_nudge_bonus(board, move)
            
            print(f"🎯 Selected move: {move} (nudge bonus: {nudge_bonus:+.1f})")
            print(f"⏱️  Search time: {search_time:.2f}s")
            
            board.push(move)
            print()
        else:
            print("❌ Engine failed to find a move")
            break
    
    print("Opening sequence played:")
    print(" ".join(str(move) for move in moves_played))
    print()
    print("Final position:")
    print(board)
    print(f"FEN: {board.fen()}")

if __name__ == "__main__":
    test_opening_simulation()