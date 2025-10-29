#!/usr/bin/env python3

"""
Debug pin detection logic
"""

import sys
import chess

# Add v7p3r engine path
v7p3r_path = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\src"
sys.path.append(v7p3r_path)

def debug_pin_detection():
    try:
        from v7p3r import V7P3REngine
        engine = V7P3REngine()
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return
    
    # Simple rook pin position: white rook pins black knight to black king
    board = chess.Board('3k4/8/8/8/3n4/8/8/3R4 w - - 0 1')
    
    print("ROOK PIN DEBUG")
    print("=" * 30)
    print(f"Position: {board.fen()}")
    print()
    
    # Manual analysis
    print("Manual Analysis:")
    print(f"White King: {chess.square_name(board.king(chess.WHITE)) if board.king(chess.WHITE) else 'None'}")
    print(f"Black King: {chess.square_name(board.king(chess.BLACK)) if board.king(chess.BLACK) else 'None'}")
    
    # Check white rook on d1
    white_rook_square = chess.D1
    black_king_square = board.king(chess.BLACK)
    
    print(f"White rook: {chess.square_name(white_rook_square)}")
    print(f"Black king: {chess.square_name(black_king_square)}")
    
    # Check if they're on same file
    white_rook_file = chess.square_file(white_rook_square)
    black_king_file = chess.square_file(black_king_square)
    
    print(f"Same file? {white_rook_file == black_king_file}")
    
    # Check squares between
    between = engine._get_squares_between(white_rook_square, black_king_square)
    print(f"Squares between: {[chess.square_name(sq) for sq in between]}")
    
    # Check pieces on those squares
    for sq in between:
        piece = board.piece_at(sq)
        if piece:
            print(f"Piece at {chess.square_name(sq)}: {piece.symbol()}")
    
    # Test pin detection
    print("\nPin Detection Result:")
    pin_data = engine._detect_pins(board)
    print(f"White pins: {len(pin_data['white_pins'])}")
    print(f"Black pins: {len(pin_data['black_pins'])}")
    
    if pin_data['white_pins']:
        for pin in pin_data['white_pins']:
            print(f"White pin: {chess.square_name(pin['pinning_square'])} pins {chess.square_name(pin['pinned_square'])}")

if __name__ == "__main__":
    debug_pin_detection()