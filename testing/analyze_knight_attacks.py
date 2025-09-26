#!/usr/bin/env python3
"""
Knight Attack Pattern Analysis
Understand exactly which squares a knight attacks from each position
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

def analyze_knight_attack_pattern():
    """Analyze knight attack patterns from different squares"""
    print("=== KNIGHT ATTACK PATTERN ANALYSIS ===")
    
    engine = V7P3REngine()
    
    # Test knight on e4 (center square)
    knight_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
    
    for knight_sq in knight_squares:
        square_name = chess.square_name(knight_sq)
        print(f"\nKnight on {square_name} (square {knight_sq}):")
        
        attacks = engine.bitboard_evaluator.KNIGHT_ATTACKS[knight_sq]
        attacked_squares = []
        
        for sq in range(64):
            if attacks & (1 << sq):
                attacked_squares.append(chess.square_name(sq))
        
        print(f"  Attacks: {', '.join(sorted(attacked_squares))}")
        
        # Create a test position with pieces on those squares
        print(f"  Testing with enemy pieces on attacked squares...")
        
        # Place black king and rook on first two attacked squares (if available)
        if len(attacked_squares) >= 2:
            sq1, sq2 = attacked_squares[0], attacked_squares[1]
            
            # Create a FEN with knight and enemy pieces
            board = chess.Board("8/8/8/8/8/8/8/4K3 w - - 0 1")  # Empty board with white king
            board.set_piece_at(knight_sq, chess.Piece(chess.KNIGHT, chess.WHITE))
            board.set_piece_at(chess.parse_square(sq1), chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(chess.parse_square(sq2), chess.Piece(chess.ROOK, chess.BLACK))
            
            print(f"  Test FEN: {board.fen()}")
            
            # Test fork detection
            knight_piece = chess.Piece(chess.KNIGHT, chess.WHITE)
            fork_bonus = engine._analyze_fork_bitboard(board, knight_sq, knight_piece, chess.BLACK)
            print(f"  Fork bonus: {fork_bonus}")

def create_working_fork_positions():
    """Create positions where knight forks definitely work"""
    print(f"\n=== CREATING WORKING FORK POSITIONS ===")
    
    engine = V7P3REngine()
    
    # From the knight attack analysis, knight on e5 attacks:
    # c4, g4, c6, g6, d3, f3, d7, f7
    
    # Test 1: Knight on e5 with king on d7 and rook on f7
    print(f"\n1. Knight on e5 with pieces on d7 and f7:")
    board1 = chess.Board("8/8/8/8/8/8/8/4K3 w - - 0 1")
    board1.set_piece_at(chess.E5, chess.Piece(chess.KNIGHT, chess.WHITE))
    board1.set_piece_at(chess.D7, chess.Piece(chess.KING, chess.BLACK))
    board1.set_piece_at(chess.F7, chess.Piece(chess.ROOK, chess.BLACK))
    
    print(f"   FEN: {board1.fen()}")
    
    knight_piece = chess.Piece(chess.KNIGHT, chess.WHITE)
    fork_bonus = engine._analyze_fork_bitboard(board1, chess.E5, knight_piece, chess.BLACK)
    print(f"   Fork bonus: {fork_bonus}")
    
    # Test 2: Knight on e5 with king on c6 and queen on g6  
    print(f"\n2. Knight on e5 with pieces on c6 and g6:")
    board2 = chess.Board("8/8/8/8/8/8/8/4K3 w - - 0 1")
    board2.set_piece_at(chess.E5, chess.Piece(chess.KNIGHT, chess.WHITE))
    board2.set_piece_at(chess.C6, chess.Piece(chess.KING, chess.BLACK))
    board2.set_piece_at(chess.G6, chess.Piece(chess.QUEEN, chess.BLACK))
    
    print(f"   FEN: {board2.fen()}")
    
    fork_bonus2 = engine._analyze_fork_bitboard(board2, chess.E5, knight_piece, chess.BLACK)
    print(f"   Fork bonus: {fork_bonus2}")
    
    # Test 3: Create a move that leads to a fork
    print(f"\n3. Testing move that creates a fork:")
    setup_board = chess.Board("8/8/8/8/4N3/8/8/4K3 w - - 0 1")
    setup_board.set_piece_at(chess.D7, chess.Piece(chess.KING, chess.BLACK))
    setup_board.set_piece_at(chess.F7, chess.Piece(chess.ROOK, chess.BLACK))
    
    print(f"   Setup FEN: {setup_board.fen()}")
    
    fork_move = chess.Move(chess.E4, chess.E5)
    if fork_move in setup_board.legal_moves:
        tactical_bonus = engine._detect_bitboard_tactics(setup_board, fork_move)
        print(f"   Move Ne4-e5 tactical bonus: {tactical_bonus}")
    
    # Also test the move ordering
    moves = list(setup_board.legal_moves)
    ordered_moves = engine._order_moves_advanced(setup_board, moves, depth=1)
    
    print(f"   Move ordering:")
    for i, move in enumerate(ordered_moves[:5]):
        tactical_score = engine._detect_bitboard_tactics(setup_board, move)
        piece = setup_board.piece_at(move.from_square)
        print(f"     {i+1}. {piece.symbol()}{move} (tactical: {tactical_score:.1f})")

if __name__ == "__main__":
    analyze_knight_attack_pattern()
    create_working_fork_positions()