#!/usr/bin/env python3
"""
Quick test to compare evaluation components
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r_bitboard_evaluator import V7P3RBitboardEvaluator

piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

evaluator = V7P3RBitboardEvaluator(piece_values)

# Test starting position
board = chess.Board()
phase = evaluator.detect_game_phase(board)

print(f"Phase: {phase}")
print()

# Break down evaluation
white_base = evaluator.evaluate_bitboard(board, chess.WHITE)
black_base = evaluator.evaluate_bitboard(board, chess.BLACK)
print(f"Base material: White={white_base:.1f}, Black={black_base:.1f}, Diff={white_base-black_base:.1f}")

safety_data = evaluator.analyze_safety_bitboard(board)
white_safety = safety_data.get('white_safety_bonus', 0)
black_safety = safety_data.get('black_safety_bonus', 0)
print(f"Safety: White={white_safety:.1f}, Black={black_safety:.1f}, Diff={white_safety-black_safety:.1f}")

white_strategic, black_strategic = evaluator._evaluate_opening_strategy(board)
print(f"Opening strategy: White={white_strategic:.1f}, Black={black_strategic:.1f}, Diff={white_strategic-black_strategic:.1f}")

total = evaluator.evaluate_position_complete(board)
print(f"\nTotal: {total:.1f}")
