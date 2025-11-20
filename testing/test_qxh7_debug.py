#!/usr/bin/env python3
"""
Debug Qxh7 evaluation
"""

import chess
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

# After Qxh7 (before Rxh7)
board = chess.Board("r1bqkbnr/ppppp1pQ/2n2p2/8/1nP5/5N2/PP1P1PPP/RNB1KB1R b KQkq - 0 1")

print("Position: After Qxh7")
print(board)
print()

# Count pieces
print("WHITE pieces:")
for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
    pieces = list(board.pieces(piece_type, chess.WHITE))
    print(f"  {chess.piece_name(piece_type)}s: {len(pieces)} at {[chess.square_name(sq) for sq in pieces]}")

print("\nBLACK pieces:")
for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
    pieces = list(board.pieces(piece_type, chess.BLACK))
    print(f"  {chess.piece_name(piece_type)}s: {len(pieces)} at {[chess.square_name(sq) for sq in pieces]}")

print()

# Check each piece for hanging
print("Checking WHITE pieces for hanging:")
for square in chess.SQUARES:
    piece = board.piece_at(square)
    if piece and piece.color == chess.WHITE:
        attacked = board.is_attacked_by(chess.BLACK, square)
        defended = board.is_attacked_by(chess.WHITE, square)
        is_hanging = attacked and not defended
        status = "HANGING!" if is_hanging else ("defended" if defended else "safe")
        print(f"  {piece.symbol()} on {chess.square_name(square)}: attacked={attacked}, defended={defended} → {status}")

print("\nChecking BLACK pieces for hanging:")
for square in chess.SQUARES:
    piece = board.piece_at(square)
    if piece and piece.color == chess.BLACK:
        attacked = board.is_attacked_by(chess.WHITE, square)
        defended = board.is_attacked_by(chess.BLACK, square)
        is_hanging = attacked and not defended
        status = "HANGING!" if is_hanging else ("defended" if defended else "safe")
        print(f"  {piece.symbol()} on {chess.square_name(square)}: attacked={attacked}, defended={defended} → {status}")

print()

# Material calculation (only safe pieces)
engine = V7P3REngine()
engine.board = board

PIECE_VALUES = {
    chess.QUEEN: 900,
    chess.ROOK: 500,
    chess.BISHOP: 300,
    chess.KNIGHT: 300,
    chess.PAWN: 100
}

white_safe_material = 0
black_safe_material = 0

for square in chess.SQUARES:
    piece = board.piece_at(square)
    if piece and piece.piece_type != chess.KING:
        attacked = board.is_attacked_by(not piece.color, square)
        defended = board.is_attacked_by(piece.color, square)
        is_hanging = attacked and not defended
        
        if not is_hanging:
            mat_val = PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_safe_material += mat_val
            else:
                black_safe_material += mat_val

print(f"White safe material: {white_safe_material} cp")
print(f"Black safe material: {black_safe_material} cp")
print(f"Material balance: {white_safe_material - black_safe_material:+d} cp")
print()

eval_score = engine._evaluate_position(board)
print(f"Engine eval: {eval_score:+.2f} cp")
