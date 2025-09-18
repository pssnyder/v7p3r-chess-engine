#!/usr/bin/env python3
"""
Debug the perspective issue with the king endgame position
"""

import chess
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine

def debug_endgame_position():
    """Debug the specific endgame position that's failing"""
    print("Debugging King Endgame Position")
    print("=" * 40)
    
    engine = V7P3REngine()
    
    # The failing position
    fen = "8/8/8/4k3/4K3/8/8/8 w - - 0 1"
    
    # Test with White to move
    board1 = chess.Board(fen)
    print(f"Position: {fen}")
    print(f"White to move: {board1.turn}")
    
    # Get component evaluations
    white_base = engine.bitboard_evaluator.calculate_score_optimized(board1, True)
    black_base = engine.bitboard_evaluator.calculate_score_optimized(board1, False)
    
    print(f"White base eval: {white_base}")
    print(f"Black base eval: {black_base}")
    
    eval1 = engine._evaluate_position(board1)
    print(f"Final White-to-move eval: {eval1}")
    
    print()
    
    # Test with Black to move
    board2 = chess.Board(fen)
    board2.turn = chess.BLACK
    print(f"Black to move: {board2.turn}")
    
    # Get component evaluations  
    white_base2 = engine.bitboard_evaluator.calculate_score_optimized(board2, True)
    black_base2 = engine.bitboard_evaluator.calculate_score_optimized(board2, False)
    
    print(f"White base eval: {white_base2}")
    print(f"Black base eval: {black_base2}")
    
    eval2 = engine._evaluate_position(board2)
    print(f"Final Black-to-move eval: {eval2}")
    
    print()
    print(f"Should be opposites: {eval1} vs {eval2}")
    print(f"Actual difference: {abs(eval1 + eval2)}")
    
    if abs(eval1 + eval2) < 0.01:
        print("✅ FIXED!")
    else:
        print("❌ Still broken")

if __name__ == "__main__":
    debug_endgame_position()