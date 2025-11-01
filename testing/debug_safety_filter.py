#!/usr/bin/env python3
"""
Debug V14.7 safety filter - understand why it's rejecting so many moves
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine


def debug_safety_filter(engine, fen, description):
    """Debug why safety filter rejects moves"""
    print(f"\n{'='*80}")
    print(f"DEBUG: {description}")
    print(f"FEN: {fen}")
    print(f"{'='*80}")
    
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    
    print(f"\nLegal moves: {len(legal_moves)}")
    print(f"Moves: {[str(m) for m in legal_moves[:10]]}...")  # First 10
    
    # Test each safety check individually
    for move in legal_moves[:5]:  # Test first 5 moves
        print(f"\n  Testing {move}:")
        
        # King safety
        king_safe = engine._is_king_safe_after_move(board, move)
        print(f"    King safe: {king_safe}")
        
        # Queen safety
        queen_safe = engine._is_queen_safe_after_move(board, move)
        print(f"    Queen safe: {queen_safe}")
        
        # Valuable pieces
        pieces_safe = engine._are_valuable_pieces_safe(board, move)
        print(f"    Pieces safe: {pieces_safe}")
        
        # Capture valid
        capture_valid = engine._is_capture_valid(board, move)
        print(f"    Capture valid: {capture_valid}")
        
        # Overall
        overall = king_safe and queen_safe and pieces_safe and capture_valid
        print(f"    OVERALL: {'SAFE' if overall else 'REJECTED'}")


if __name__ == "__main__":
    engine = V7P3REngine()
    
    # Test starting position
    debug_safety_filter(
        engine,
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Starting position - should have many safe moves"
    )
    
    # Test position from game
    debug_safety_filter(
        engine,
        "rnbqkb1r/pppppppp/8/8/4n3/5N2/PPPPPPPP/RNBQKB1R w KQkq - 1 3",
        "After 2...Ne4 - knight on e4"
    )
