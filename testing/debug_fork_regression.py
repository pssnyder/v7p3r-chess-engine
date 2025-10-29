#!/usr/bin/env python3

"""
Debug the g6f4 regression specifically
"""

import sys
import chess

# Add v7p3r engine path
v7p3r_path = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\src"
sys.path.append(v7p3r_path)

def debug_fork_regression():
    try:
        from v7p3r import V7P3REngine
        engine = V7P3REngine()
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return
    
    # Fork position from Phase 1
    board = chess.Board('r6k/pp4pp/3pQ1n1/2pP1rq1/2N5/8/PPP2PPP/4RRK1 b - - 3 18')
    
    print("FORK REGRESSION DEBUG: g6f4")
    print("=" * 40)
    print(f"Position: {board.fen()}")
    
    # Test move ordering
    legal_moves = list(board.legal_moves)
    ordered_moves = engine._order_moves_advanced(board, legal_moves, 4)
    
    print(f"\nTop 10 moves:")
    target_move = chess.Move.from_uci('g6f4')
    
    for i, move in enumerate(ordered_moves[:10], 1):
        marker = "👈 TARGET" if move == target_move else ""
        print(f"  {i:2}. {str(move)} {marker}")
    
    # Test tactical analysis of g6f4 specifically
    print(f"\nTactical Analysis of g6f4:")
    tactical_analysis = engine._analyze_position_for_tactics(board)
    tactical_score = engine._calculate_tactical_move_score(board, target_move, tactical_analysis)
    
    print(f"Tactical score data: {tactical_score}")
    
    # Test what category it falls into
    print(f"\nMove Categorization:")
    
    # Test each category
    if board.gives_check(target_move):
        print(f"  Category: CHECK")
    elif tactical_score.get('attacks_multiple', False):
        print(f"  Category: MULTI-ATTACK")
        print(f"  Multi-attack bonus should be: 350 + {tactical_score['base_score']}")
    elif tactical_score.get('creates_threat', False):
        threat_value = tactical_score.get('threat_value', 0)
        if threat_value >= 900:
            bonus = 500.0
        elif threat_value >= 500:
            bonus = 400.0
        elif threat_value >= 300:
            bonus = 300.0
        else:
            bonus = 200.0
        print(f"  Category: THREAT (value: {threat_value})")
        print(f"  Threat bonus should be: {bonus} + {tactical_score['base_score']}")
    elif board.is_capture(target_move):
        print(f"  Category: CAPTURE")
    else:
        print(f"  Category: OTHER")

if __name__ == "__main__":
    debug_fork_regression()