#!/usr/bin/env python3
"""
Debug draw prevention interference with mate moves
"""

import chess
from v7p3r import V7P3REvaluationEngine

def debug_draw_prevention():
    """Check if draw prevention is interfering with mate moves"""
    
    # Position where Qf7 is mate
    fen = "rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 3"
    board = chess.Board(fen)
    
    print(f"Position: {fen}")
    
    engine = V7P3REvaluationEngine()
    mate_move = chess.Move.from_uci("h5f7")
    
    print(f"\nTesting draw prevention on mate move {mate_move}:")
    
    # Test what draw prevention does to the mate move
    result_move = engine._enforce_strict_draw_prevention(board, mate_move)
    
    print(f"Original move: {mate_move}")
    print(f"After draw prevention: {result_move}")
    print(f"Same move?: {mate_move == result_move}")
    
    # Check what happens when we apply the mate move
    temp_board = board.copy()
    temp_board.push(mate_move)
    print(f"\nAfter mate move:")
    print(f"  Is stalemate: {temp_board.is_stalemate()}")
    print(f"  Is insufficient material: {temp_board.is_insufficient_material()}")
    print(f"  Is fivefold repetition: {temp_board.is_fivefold_repetition()}")
    print(f"  Is threefold repetition: {temp_board.is_repetition(count=3)}")
    print(f"  Is checkmate: {temp_board.is_checkmate()}")
    temp_board.pop()
    
    # Test the full search process step by step
    print(f"\nStep-by-step search process:")
    
    # 1. Minimax root search
    best_move_root, best_score = engine._minimax_search_root(board, 4, -float('inf'), float('inf'), True)
    print(f"1. Root search found: {best_move_root}")
    
    # 2. Apply draw prevention
    after_draw_prevention = engine._enforce_strict_draw_prevention(board, best_move_root)
    print(f"2. After draw prevention: {after_draw_prevention}")
    
    # 3. Final safety check
    if not isinstance(after_draw_prevention, chess.Move) or not board.is_legal(after_draw_prevention):
        print(f"3. Failed safety check!")
        legal_moves = list(board.legal_moves)
        if legal_moves:
            final_move = random.choice(legal_moves)
            print(f"   Random fallback: {final_move}")
        else:
            final_move = chess.Move.null()
            print(f"   Null move fallback")
    else:
        final_move = after_draw_prevention
        print(f"3. Passed safety check: {final_move}")
    
    print(f"\nFinal result would be: {final_move}")

if __name__ == "__main__":
    import random
    debug_draw_prevention()
