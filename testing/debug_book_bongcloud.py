#!/usr/bin/env python3
"""Debug opening book moves"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

def trace_opening(moves_uci):
    """Trace opening moves and see what book suggests"""
    engine = V7P3REngine()
    board = chess.Board()
    
    print("="*60)
    print("Opening Book Trace")
    print("="*60)
    print()
    
    for i, move_uci in enumerate(moves_uci):
        # Show current position
        move_num = (i // 2) + 1
        side = "White" if board.turn == chess.WHITE else "Black"
        
        print(f"Move {move_num} ({side} to move):")
        print(f"Position: {board.fen()}")
        
        # Check what book suggests
        book_move = engine.opening_book.get_book_move(board)
        print(f"Book suggests: {book_move}")
        
        # Check what's in the book for this position
        key = chess.polyglot.zobrist_hash(board)
        if key in engine.opening_book.book_moves:
            entries = engine.opening_book.book_moves[key]
            print(f"Book entries: {entries}")
        else:
            print("Position not in book")
        
        # Make the move
        move = chess.Move.from_uci(move_uci)
        print(f"Actual move played: {move}")
        board.push(move)
        print()

# Test the sequence that led to bongcloud
print("\nTest 1: Standard opening")
trace_opening(["e2e4", "e7e5"])

print("\n" + "="*60)
print("\nTest 2: After 1.e4 e5 2.Nf3")
trace_opening(["e2e4", "e7e5", "g1f3"])

