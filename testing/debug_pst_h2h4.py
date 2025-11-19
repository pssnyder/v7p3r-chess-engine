#!/usr/bin/env python3
"""
Debug PST evaluation for h2h4 move
"""

import sys
import os
sys.path.insert(0, "s:/Maker Stuff/Programming/Chess Engines/Chess Engine Playground/engine-tester/engines/V7P3R/V7P3R_v15.2/src")

import chess
from v7p3r import V7P3REngine

def analyze_move_evaluation(fen_before, move_uci, move_desc):
    """Analyze how a move changes the evaluation"""
    engine = V7P3REngine()
    
    print(f"\n{'='*60}")
    print(f"ANALYZING: {move_desc}")
    print(f"{'='*60}")
    
    board_before = chess.Board(fen_before)
    eval_before = engine._evaluate_position(board_before)
    
    print(f"Position before {move_uci}:")
    print(board_before)
    print(f"Evaluation: {eval_before:+d}")
    print()
    
    # Make the move
    move = chess.Move.from_uci(move_uci)
    board_after = board_before.copy()
    board_after.push(move)
    eval_after = engine._evaluate_position(board_after)
    
    print(f"Position after {move_uci}:")
    print(board_after)
    print(f"Evaluation: {eval_after:+d}")
    print()
    
    change = eval_after - eval_before
    print(f"Evaluation change: {change:+d}")
    
    if change > 0:
        print(f"✓ Move IMPROVES position (from White's perspective)")
    elif change < 0:
        print(f"✗ Move WORSENS position (from White's perspective)")
    else:
        print(f"= Move NEUTRAL")
    
    return change

def compare_opening_moves():
    """Compare different opening moves"""
    starting_pos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    moves_to_test = [
        ("e2e4", "1.e4 (King's pawn - central)"),
        ("d2d4", "1.d4 (Queen's pawn - central)"),
        ("g1f3", "1.Nf3 (Knight development)"),
        ("h2h4", "1.h4 (Edge pawn)"),
        ("h2h3", "1.h3 (Edge pawn)"),
        ("a2a4", "1.a4 (Edge pawn)"),
    ]
    
    print("="*60)
    print("WHITE OPENING MOVE COMPARISON")
    print("="*60)
    
    results = []
    for move_uci, move_desc in moves_to_test:
        change = analyze_move_evaluation(starting_pos, move_uci, move_desc)
        results.append((move_desc, change))
    
    print("\n" + "="*60)
    print("SUMMARY - Evaluation Changes")
    print("="*60)
    results.sort(key=lambda x: x[1], reverse=True)
    
    for desc, change in results:
        print(f"{desc:40s} {change:+4d}")
    
    best_move = results[0]
    print(f"\nBest move by PST evaluation: {best_move[0]}")

if __name__ == "__main__":
    compare_opening_moves()
