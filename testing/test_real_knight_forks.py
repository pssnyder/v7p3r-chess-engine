#!/usr/bin/env python3
"""
Test Real Knight Fork Positions
Create positions where knight forks actually exist and should be detected
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

def test_actual_knight_forks():
    """Test positions with actual knight forks"""
    print("=== TESTING ACTUAL KNIGHT FORK POSITIONS ===")
    
    engine = V7P3REngine()
    
    # Test 1: Knight on e5 can fork king on d7 and rook on f7
    fork_fen1 = "8/3kr3/8/4N3/8/8/8/4K3 w - - 0 1"
    print(f"\n1. Knight Fork: Ne5 forking king on d7 and rook on f7")
    print(f"   FEN: {fork_fen1}")
    
    board1 = chess.Board(fork_fen1)
    
    # The knight is already on e5, let's see what it's attacking
    knight_square = chess.E5
    attacks = engine.bitboard_evaluator.KNIGHT_ATTACKS[knight_square]
    
    enemy_pieces_attacked = []
    for sq in range(64):
        if attacks & (1 << sq):
            piece = board1.piece_at(sq)
            if piece and piece.color == chess.BLACK:
                enemy_pieces_attacked.append((chess.square_name(sq), piece.symbol()))
    
    print(f"   Knight on e5 attacks: {enemy_pieces_attacked}")
    
    # Test the fork detection directly
    knight_piece = chess.Piece(chess.KNIGHT, chess.WHITE)
    fork_bonus = engine._analyze_fork_bitboard(board1, knight_square, knight_piece, chess.BLACK)
    print(f"   Fork bonus: {fork_bonus}")
    
    # Test 2: Position where knight can move to create a fork
    setup_fen = "8/3kr3/8/8/4N3/8/8/4K3 w - - 0 1"
    print(f"\n2. Knight Move to Create Fork: Ne4-e5 creating fork")
    print(f"   FEN: {setup_fen}")
    
    board2 = chess.Board(setup_fen)
    fork_move = chess.Move(chess.E4, chess.E5)
    
    if fork_move in board2.legal_moves:
        tactical_bonus = engine._detect_bitboard_tactics(board2, fork_move)
        print(f"   Move Ne4-e5 tactical bonus: {tactical_bonus}")
    
    # Test 3: Classic knight fork pattern - fork king and queen
    royal_fork_fen = "4k3/8/8/4N3/8/8/4q3/4K3 w - - 0 1"
    print(f"\n3. Royal Fork: Knight on e5 forking king and queen")
    print(f"   FEN: {royal_fork_fen}")
    
    board3 = chess.Board(royal_fork_fen)
    
    # Check what the knight attacks in this position
    attacks3 = engine.bitboard_evaluator.KNIGHT_ATTACKS[chess.E5]
    enemy_pieces_attacked3 = []
    high_value_count = 0
    
    for sq in range(64):
        if attacks3 & (1 << sq):
            piece = board3.piece_at(sq)
            if piece and piece.color == chess.BLACK:
                enemy_pieces_attacked3.append((chess.square_name(sq), piece.symbol()))
                if piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                    high_value_count += 1
    
    print(f"   Knight attacks: {enemy_pieces_attacked3}")
    print(f"   High-value targets: {high_value_count}")
    
    fork_bonus3 = engine._analyze_fork_bitboard(board3, chess.E5, knight_piece, chess.BLACK)
    print(f"   Fork bonus: {fork_bonus3}")
    
    # Test 4: Create a position where a knight move leads to multiple piece attacks
    multi_fork_fen = "r2qk2r/8/8/8/4N3/8/8/4K3 w - - 0 1"
    print(f"\n4. Multiple Target Fork Test")
    print(f"   FEN: {multi_fork_fen}")
    
    board4 = chess.Board(multi_fork_fen)
    
    # Check all knight moves for tactical bonuses
    for move in board4.legal_moves:
        piece = board4.piece_at(move.from_square)
        if piece and piece.piece_type == chess.KNIGHT:
            tactical_bonus = engine._detect_bitboard_tactics(board4, move)
            if tactical_bonus > 0:
                print(f"   Move {move}: tactical bonus = {tactical_bonus}")

def analyze_fork_detection_requirements():
    """Analyze exactly what conditions trigger fork detection"""
    print(f"\n=== FORK DETECTION ANALYSIS ===")
    
    engine = V7P3REngine()
    
    # Create different scenarios and test
    scenarios = [
        ("1 enemy piece", "8/3k4/8/4N3/8/8/8/4K3 w - - 0 1"),
        ("2 enemy pieces", "8/3kr3/8/4N3/8/8/8/4K3 w - - 0 1"), 
        ("2 high-value", "8/3kq3/8/4N3/8/8/8/4K3 w - - 0 1"),
        ("3 enemy pieces", "8/2qkr3/8/4N3/8/8/8/4K3 w - - 0 1")
    ]
    
    for scenario_name, fen in scenarios:
        print(f"\n{scenario_name}: {fen}")
        board = chess.Board(fen)
        
        knight_square = chess.E5
        attacks = engine.bitboard_evaluator.KNIGHT_ATTACKS[knight_square]
        
        enemy_count = 0
        high_value_count = 0
        
        for sq in range(64):
            if attacks & (1 << sq):
                piece = board.piece_at(sq)
                if piece and piece.color == chess.BLACK:
                    enemy_count += 1
                    if piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                        high_value_count += 1
        
        knight_piece = chess.Piece(chess.KNIGHT, chess.WHITE)
        fork_bonus = engine._analyze_fork_bitboard(board, knight_square, knight_piece, chess.BLACK)
        
        print(f"  Enemy pieces attacked: {enemy_count}")
        print(f"  High-value targets: {high_value_count}")
        print(f"  Fork bonus: {fork_bonus}")
        
        # Calculate expected bonus
        if enemy_count >= 2:
            expected = 50.0 + (high_value_count * 25.0)
            print(f"  Expected bonus: {expected}")

if __name__ == "__main__":
    test_actual_knight_forks()
    analyze_fork_detection_requirements()