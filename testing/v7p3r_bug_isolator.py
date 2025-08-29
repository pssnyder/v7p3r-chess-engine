#!/usr/bin/env python3
"""
V7P3R Perspective Bug Isolator

This script directly tests the V7P3R evaluation function
to isolate exactly where the perspective issue occurs.
"""

import sys
import os
sys.path.append('src')

import chess
from v7p3r import V7P3RCleanEngine

def test_evaluation_consistency():
    """Test V7P3R evaluation function consistency"""
    
    print("🔍 V7P3R Evaluation Consistency Test")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    
    # Test positions from our diagnostic
    test_positions = [
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Post c3-d4 setup", "rnbqkbnr/ppp1pppp/8/3p4/3P4/2P2N2/PP2PPPP/RNBQKB1R b KQkq - 0 3"),
        ("Middlegame Test", "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 4 5"),
    ]
    
    for description, fen in test_positions:
        print(f"\n📍 {description}")
        print(f"FEN: {fen}")
        
        board = chess.Board(fen)
        
        # Test evaluation from both perspectives directly
        white_perspective = engine._evaluate_position(board, chess.WHITE)
        black_perspective = engine._evaluate_position(board, chess.BLACK)
        
        print(f"White Perspective: {white_perspective:+.2f}")
        print(f"Black Perspective: {black_perspective:+.2f}")
        print(f"Sum: {white_perspective + black_perspective:+.2f}")
        
        # Test the scoring calculator directly
        white_material = engine.scoring_calculator.calculate_score_optimized(board, chess.WHITE)
        black_material = engine.scoring_calculator.calculate_score_optimized(board, chess.BLACK)
        
        print(f"White Material Score: {white_material:+.2f}")
        print(f"Black Material Score: {black_material:+.2f}")
        
        # Test what happens in negamax search
        print("\n🔍 Testing negamax evaluation:")
        
        # When it's white's turn
        if board.turn == chess.WHITE:
            eval_white_turn = engine._evaluate_position(board, board.turn)
            print(f"Evaluation (White to move): {eval_white_turn:+.2f}")
        else:
            eval_black_turn = engine._evaluate_position(board, board.turn)
            print(f"Evaluation (Black to move): {eval_black_turn:+.2f}")
        
        # Check consistency
        expected_sum = 0.0
        actual_sum = white_perspective + black_perspective
        
        if abs(actual_sum) > 0.1:
            print(f"❌ PERSPECTIVE ISSUE: Sum should be ~0, got {actual_sum:+.2f}")
        else:
            print(f"✅ Perspective consistent: {actual_sum:+.2f}")

def flip_fen(fen: str) -> str:
    """Flip FEN position colors and orientation"""
    parts = fen.split()
    if len(parts) != 6:
        return fen
    
    # Flip piece placement
    ranks = parts[0].split('/')
    flipped_ranks = []
    
    for rank in reversed(ranks):
        flipped_rank = ""
        for char in rank:
            if char.isalpha():
                flipped_rank += char.swapcase()
            else:
                flipped_rank += char
        flipped_ranks.append(flipped_rank)
    
    # Flip active color
    active_color = "b" if parts[1] == "w" else "w"
    
    # Flip castling rights
    castling = parts[2]
    if castling != "-":
        new_castling = ""
        for char in castling:
            new_castling += char.swapcase()
        castling = new_castling
    
    # Flip en passant square
    en_passant = parts[3]
    if en_passant != "-":
        file = en_passant[0]
        rank = int(en_passant[1])
        new_rank = 9 - rank
        en_passant = f"{file}{new_rank}"
    
    return f"{'/'.join(flipped_ranks)} {active_color} {castling} {en_passant} {parts[4]} {parts[5]}"

def test_flipped_position_consistency():
    """Test that flipped positions evaluate consistently"""
    
    print("\n\n🔄 Flipped Position Consistency Test")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    
    test_positions = [
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Post c3-d4 setup", "rnbqkbnr/ppp1pppp/8/3p4/3P4/2P2N2/PP2PPPP/RNBQKB1R b KQkq - 0 3"),
    ]
    
    for description, fen in test_positions:
        print(f"\n📍 {description}")
        print(f"Original FEN: {fen}")
        
        # Original position
        board_orig = chess.Board(fen)
        eval_orig = engine._evaluate_position(board_orig, board_orig.turn)
        
        # Flipped position
        flipped_fen = flip_fen(fen)
        print(f"Flipped FEN:  {flipped_fen}")
        
        board_flipped = chess.Board(flipped_fen)
        eval_flipped = engine._evaluate_position(board_flipped, board_flipped.turn)
        
        print(f"Original Eval (from {board_orig.turn} perspective): {eval_orig:+.2f}")
        print(f"Flipped Eval (from {board_flipped.turn} perspective):  {eval_flipped:+.2f}")
        print(f"Sum: {eval_orig + eval_flipped:+.2f}")
        
        if abs(eval_orig + eval_flipped) > 0.1:
            print(f"❌ FLIPPED POSITION ISSUE: Evaluations should sum to ~0")
        else:
            print(f"✅ Flipped position consistent")

def test_search_consistency():
    """Test search behavior consistency"""
    
    print("\n\n🔍 Search Consistency Test")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    
    # Test with one problematic position
    fen = "rnbqkbnr/ppp1pppp/8/3p4/3P4/2P2N2/PP2PPPP/RNBQKB1R b KQkq - 0 3"
    board = chess.Board(fen)
    
    print(f"Position: {fen}")
    print(f"Turn: {board.turn}")
    
    # Test with different search depths
    for depth in [1, 2, 3]:
        move, score, pv = engine._search_best_move(board, depth)
        print(f"Depth {depth}: Move={move}, Score={score:+.2f}")
        
        # Now test the flipped position
        flipped_fen = flip_fen(fen)
        board_flipped = chess.Board(flipped_fen)
        move_flipped, score_flipped, pv_flipped = engine._search_best_move(board_flipped, depth)
        print(f"Depth {depth} (flipped): Move={move_flipped}, Score={score_flipped:+.2f}")
        
        print(f"Score sum: {score + score_flipped:+.2f}")
        print()

if __name__ == "__main__":
    test_evaluation_consistency()
    test_flipped_position_consistency()
    test_search_consistency()
