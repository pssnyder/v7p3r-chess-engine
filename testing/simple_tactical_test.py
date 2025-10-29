#!/usr/bin/env python3

"""
Simple tactical move test to debug specific moves
"""

import sys
import chess

# Add v7p3r engine path
v7p3r_path = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\src"
sys.path.append(v7p3r_path)

def test_specific_moves():
    """Test specific problematic moves"""
    
    try:
        from v7p3r import V7P3REngine
        engine = V7P3REngine()
        print("✅ Engine initialized")
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return
    
    # Test the fork move g6f4
    print("FORK TEST: g6f4")
    print("=" * 30)
    board = chess.Board('r6k/pp4pp/3pQ1n1/2pP1rq1/2N5/8/PPP2PPP/4RRK1 b - - 3 18')
    move = chess.Move.from_uci('g6f4')
    
    print(f"Original position: {board.fen()}")
    print(f"Testing move: {move}")
    
    # Get tactical analysis
    tactical_analysis = engine._analyze_position_for_tactics(board)
    print(f"Valuable pieces: {[(p['square'], chess.square_name(p['square']), p['piece'].symbol(), p['value']) for p in tactical_analysis['valuable_pieces']]}")
    
    # Test the move
    tactical_score = engine._calculate_tactical_move_score(board, move, tactical_analysis)
    print(f"Tactical score: {tactical_score}")
    
    # Check what pieces are attacked after the move
    test_board = board.copy()
    test_board.push(move)
    print(f"After g6f4:")
    
    attacked_squares = []
    for square in chess.SQUARES:
        if test_board.is_attacked_by(board.turn, square):  # Black attacks
            piece = test_board.piece_at(square)
            if piece and piece.color != board.turn:  # Enemy piece
                attacked_squares.append((square, chess.square_name(square), piece.symbol()))
    
    print(f"Attacked enemy pieces: {attacked_squares}")
    
    print("\n")
    
    # Test the capture move f5e4
    print("CAPTURE TEST: f5e4")
    print("=" * 30)
    board2 = chess.Board('3r1rk1/pp1q1p1R/2n2p1Q/4pb2/4B3/2P2NP1/PP3PP1/R3K3 b Q - 8 20')
    move2 = chess.Move.from_uci('f5e4')
    
    print(f"Original position: {board2.fen()}")
    print(f"Testing move: {move2}")
    
    # Check what's being captured
    captured_piece = board2.piece_at(chess.E4)
    print(f"Captured piece: {captured_piece.symbol() if captured_piece else 'None'}")
    
    if captured_piece:
        piece_value = engine._get_dynamic_piece_value(board2, captured_piece.piece_type, captured_piece.color)
        print(f"Piece value: {piece_value}")
    
    # Get tactical analysis
    tactical_analysis2 = engine._analyze_position_for_tactics(board2)
    tactical_score2 = engine._calculate_tactical_move_score(board2, move2, tactical_analysis2)
    print(f"Tactical score: {tactical_score2}")

if __name__ == "__main__":
    test_specific_moves()