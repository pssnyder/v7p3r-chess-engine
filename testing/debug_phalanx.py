#!/usr/bin/env python3
"""Debug script to test phalanx detection"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r_fast_evaluator import V7P3RFastEvaluator

evaluator = V7P3RFastEvaluator()

# Test 1: Clear non-phalanx (pawns separated by empty file)
board1 = chess.Board("4k3/8/8/8/3P1P2/8/8/4K3 w - - 0 1")
print("Position 1 (d4 + f4, separated):")
print(board1)
score1 = evaluator.evaluate(board1)
print(f"Score: {score1}cp\n")

# Test 2: Clear phalanx (pawns side by side)
board2 = chess.Board("4k3/8/8/8/3PP3/8/8/4K3 w - - 0 1")
print("Position 2 (d4 + e4, phalanx):")
print(board2)
score2 = evaluator.evaluate(board2)
print(f"Score: {score2}cp\n")

print(f"Difference: {score2 - score1}cp (should be +5cp for phalanx bonus)")

# Manual check
print("\nManual phalanx check for position 2:")
for rank in range(8):
    for file in range(7):
        sq1 = chess.square(file, rank)
        sq2 = chess.square(file + 1, rank)
        p1 = board2.piece_at(sq1)
        p2 = board2.piece_at(sq2)
        if p1 and p2 and p1.piece_type == chess.PAWN and p2.piece_type == chess.PAWN:
            if p1.color == p2.color:
                print(f"PHALANX at rank {rank+1}: {chess.square_name(sq1)}-{chess.square_name(sq2)} (both {p1.color})")
