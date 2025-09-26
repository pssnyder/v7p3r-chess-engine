#!/usr/bin/env python3
"""
Debug Knight Fork Detection
Investigate why the knight fork detection isn't working as expected
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

def debug_knight_fork_logic():
    """Debug the knight fork detection step by step"""
    print("=== DEBUGGING KNIGHT FORK DETECTION ===")
    
    engine = V7P3REngine()
    
    # Create a position where a knight can clearly fork two pieces
    # Place knight on d5 where it can attack pieces on multiple squares
    debug_fen = "8/8/2k5/3N4/8/8/4r3/4K3 w - - 0 1"  # Knight on d5, king on c6, rook on e2
    print(f"Test position: {debug_fen}")
    print("Knight on d5 should be able to attack king on c6 and other squares")
    
    board = chess.Board(debug_fen)
    knight_square = chess.D5  # 35 in 0-63 notation
    
    # Check knight attacks manually
    attacks = engine.bitboard_evaluator.KNIGHT_ATTACKS[knight_square]
    print(f"\nKnight attacks bitboard from d5: {bin(attacks)}")
    
    # Show which squares are attacked
    attacked_squares = []
    for sq in range(64):
        if attacks & (1 << sq):
            attacked_squares.append(chess.square_name(sq))
    
    print(f"Knight on d5 attacks: {', '.join(attacked_squares)}")
    
    # Check what pieces are on those squares
    enemy_pieces_attacked = []
    high_value_targets = []
    
    for sq in range(64):
        if attacks & (1 << sq):
            piece = board.piece_at(sq)
            if piece and piece.color == chess.BLACK:  # Enemy pieces
                enemy_pieces_attacked.append((chess.square_name(sq), piece.piece_type, piece.symbol()))
                if piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                    high_value_targets.append(piece.piece_type)
    
    print(f"\nEnemy pieces attacked: {enemy_pieces_attacked}")
    print(f"High-value targets: {len(high_value_targets)}")
    print(f"Total enemy pieces attacked: {len(enemy_pieces_attacked)}")
    
    # Test the actual fork detection method
    knight_piece = chess.Piece(chess.KNIGHT, chess.WHITE)
    fork_bonus = engine._analyze_fork_bitboard(board, knight_square, knight_piece, chess.BLACK)
    print(f"\nFork bonus calculated: {fork_bonus}")
    
    # Test with the actual tactical detection
    move = chess.Move(chess.D4, chess.D5)  # Assume knight moves to d5
    if move in board.legal_moves:
        tactical_bonus = engine._detect_bitboard_tactics(board, move)
        print(f"Full tactical bonus for move {move}: {tactical_bonus}")

def debug_existing_position():
    """Debug the original position that didn't work"""
    print(f"\n=== DEBUGGING ORIGINAL POSITION ===")
    
    engine = V7P3REngine()
    
    # Original position that didn't detect forks
    fork_fen = "r3k3/8/8/8/4N3/8/8/R3K2R w - - 0 1"
    board = chess.Board(fork_fen)
    
    print(f"Original position: {fork_fen}")
    
    # Analyze the knight on e4
    knight_square = chess.E4  # 28 in 0-63 notation
    attacks = engine.bitboard_evaluator.KNIGHT_ATTACKS[knight_square]
    
    print(f"Knight on e4 attacks bitboard: {bin(attacks)}")
    
    # Check each knight move
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.KNIGHT:
            print(f"\nAnalyzing move: {move}")
            
            # Show what the knight would attack after the move
            to_square = move.to_square
            new_attacks = engine.bitboard_evaluator.KNIGHT_ATTACKS[to_square]
            
            attacked_squares = []
            for sq in range(64):
                if new_attacks & (1 << sq):
                    attacked_squares.append(chess.square_name(sq))
            
            print(f"  Knight on {chess.square_name(to_square)} would attack: {', '.join(attacked_squares)}")
            
            # Check what pieces would be attacked after the move
            board.push(move)  # Make the move temporarily
            
            enemy_pieces = []
            for sq in range(64):
                if new_attacks & (1 << sq):
                    piece_on_sq = board.piece_at(sq)
                    if piece_on_sq and piece_on_sq.color == chess.BLACK:
                        enemy_pieces.append((chess.square_name(sq), piece_on_sq.symbol()))
            
            print(f"  Enemy pieces that would be attacked: {enemy_pieces}")
            print(f"  Number of enemy pieces attacked: {len(enemy_pieces)}")
            
            board.pop()  # Undo the move
            
            # Get the tactical bonus
            tactical_bonus = engine._detect_bitboard_tactics(board, move)
            print(f"  Tactical bonus: {tactical_bonus}")

if __name__ == "__main__":
    debug_knight_fork_logic()
    debug_existing_position()