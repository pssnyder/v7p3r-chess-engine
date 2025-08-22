#!/usr/bin/env python3
"""
Debug the search process for mate in 1
"""

import chess
from v7p3r import V7P3REvaluationEngine

def debug_search_for_mate():
    """Debug the search process step by step"""
    
    # Position where Qf7 is mate
    fen = "rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 3"
    board = chess.Board(fen)
    
    print(f"Position: {fen}")
    
    engine = V7P3REvaluationEngine()
    engine.depth = 2  # Use shallow depth for debugging
    
    # Test the main search directly
    print("\nTesting main search...")
    best_move = engine.search(board, chess.WHITE)
    print(f"Search returned: {best_move}")
    
    # Test the root minimax search directly
    print(f"\nTesting root minimax search...")
    best_move_root, best_score = engine._minimax_search_root(board, 2, -float('inf'), float('inf'), True)
    print(f"Root search returned: {best_move_root}, score: {best_score}")
    
    # Test if the mate move is being considered
    mate_move = chess.Move.from_uci("h5f7")
    print(f"\nTesting mate move {mate_move}:")
    
    # Simulate what happens when we evaluate this move
    board.push(mate_move)
    print(f"After mate move:")
    print(f"  Is checkmate: {board.is_checkmate()}")
    print(f"  Is check: {board.is_check()}")
    print(f"  Game over: {board.is_game_over()}")
    
    # Test evaluation from this position
    eval_score = engine.evaluate_position_from_perspective(board, chess.WHITE)
    print(f"  Evaluation from White's perspective: {eval_score}")
    board.pop()
    
    # Test with deeper search
    print(f"\nTesting with depth 4...")
    engine.depth = 4
    best_move_deep = engine.search(board, chess.WHITE)
    print(f"Deep search returned: {best_move_deep}")

if __name__ == "__main__":
    debug_search_for_mate()
