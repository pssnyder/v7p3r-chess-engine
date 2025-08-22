#!/usr/bin/env python3
"""
Debug move ordering for mate in 1
"""

import chess
from v7p3r import V7P3REvaluationEngine

def debug_move_ordering():
    """Debug why mate moves aren't being found"""
    
    # Position where Qf7 is mate
    fen = "rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 3"
    board = chess.Board(fen)
    
    print(f"Position: {fen}")
    
    engine = V7P3REvaluationEngine()
    
    # Get all legal moves and their scores
    legal_moves = list(board.legal_moves)
    print(f"\nAll legal moves ({len(legal_moves)}):")
    
    move_scores = []
    for move in legal_moves:
        score = engine._order_move_score(board, move, depth=4)
        move_scores.append((move, score))
        
        # Check if it's checkmate
        temp_board = board.copy()
        temp_board.push(move)
        is_mate = temp_board.is_checkmate()
        is_check = temp_board.is_check()
        temp_board.pop()
        
        status = ""
        if is_mate:
            status += " [MATE!]"
        elif is_check:
            status += " [CHECK]"
            
        print(f"  {move}: {score:.1f}{status}")
    
    # Sort by score
    move_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop moves by score:")
    for move, score in move_scores[:5]:
        temp_board = board.copy()
        temp_board.push(move)
        is_mate = temp_board.is_checkmate()
        temp_board.pop()
        status = " [MATE!]" if is_mate else ""
        print(f"  {move}: {score:.1f}{status}")
    
    # Test move ordering function
    print(f"\nTesting move ordering...")
    ordered_moves = engine.order_moves(board, legal_moves, depth=4)
    print(f"First 5 ordered moves:")
    for i, move in enumerate(ordered_moves[:5]):
        temp_board = board.copy()
        temp_board.push(move)
        is_mate = temp_board.is_checkmate()
        temp_board.pop()
        status = " [MATE!]" if is_mate else ""
        print(f"  {i+1}. {move}{status}")

    # Manually test the mate move
    mate_move = chess.Move.from_uci("h5f7")
    if mate_move in legal_moves:
        print(f"\nTesting mate move {mate_move}:")
        score = engine._order_move_score(board, mate_move, depth=4)
        print(f"  Score: {score}")
        
        temp_board = board.copy()
        temp_board.push(mate_move)
        print(f"  Is checkmate: {temp_board.is_checkmate()}")
        print(f"  Is check: {temp_board.is_check()}")
        temp_board.pop()

if __name__ == "__main__":
    debug_move_ordering()
