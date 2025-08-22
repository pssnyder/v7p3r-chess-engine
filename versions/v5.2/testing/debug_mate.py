#!/usr/bin/env python3
"""
Debug the mate in 1 position
"""

import chess
from v7p3r import V7P3REvaluationEngine

def debug_mate_position():
    """Debug the specific mate in 1 position"""
    
    # The position: r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4
    fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    board = chess.Board(fen)
    
    print(f"Position: {fen}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    
    # Check if position is already mate or stalemate
    print(f"Is checkmate: {board.is_checkmate()}")
    print(f"Is stalemate: {board.is_stalemate()}")
    print(f"Is check: {board.is_check()}")
    print(f"Is game over: {board.is_game_over()}")
    
    # List all legal moves
    legal_moves = list(board.legal_moves)
    print(f"\nAll legal moves ({len(legal_moves)}):")
    for i, move in enumerate(legal_moves):
        print(f"  {i+1}. {move}")
        
        # Check what happens after each move
        board.push(move)
        if board.is_checkmate():
            print(f"     -> Leads to checkmate for {'Black' if board.turn == chess.BLACK else 'White'}")
        elif board.is_check():
            print(f"     -> Gives check")
        board.pop()
    
    # Test the engine
    print(f"\nTesting engine...")
    engine = V7P3REvaluationEngine()
    
    # Set a low depth for debugging
    engine.depth = 3
    
    try:
        best_move = engine.find_best_move(board, time_limit=2.0)
        print(f"Engine found move: {best_move}")
        
        if best_move and best_move != chess.Move.null():
            print(f"Move is legal: {board.is_legal(best_move)}")
            
            # Analyze the move
            board.push(best_move)
            print(f"After move - Check: {board.is_check()}, Mate: {board.is_checkmate()}")
            board.pop()
        else:
            print("No move returned by engine")
            
    except Exception as e:
        print(f"Engine error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_mate_position()
