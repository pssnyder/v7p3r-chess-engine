#!/usr/bin/env python3
"""
V12.3 Castling Test - Verify that the engine now prefers castling moves
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

def test_castling_preference():
    """Test that the engine prefers castling in appropriate positions"""
    
    print("=" * 60)
    print("V7P3R v12.3 - CASTLING HEURISTIC TEST")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    test_positions = [
        {
            "name": "Basic Castling Position (White)",
            "fen": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            "description": "Both sides can castle - white to move"
        },
        {
            "name": "Development vs Castling (White)",
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "description": "Standard opening - should castle after development"
        },
        {
            "name": "Castling vs King Move",
            "fen": "r3k2r/ppp2ppp/2n1bn2/3p4/3P4/2N1BN2/PPP2PPP/R3K2R w KQkq - 0 8",
            "description": "Both sides developed - should prefer castling over Kf1"
        }
    ]
    
    for i, pos in enumerate(test_positions, 1):
        print(f"\nTest {i}: {pos['name']}")
        print(f"Description: {pos['description']}")
        print(f"FEN: {pos['fen']}")
        print("-" * 40)
        
        board = chess.Board(pos['fen'])
        print("Legal moves:", [str(move) for move in board.legal_moves])
        
        # Check if castling moves are available
        castling_moves = []
        for move in board.legal_moves:
            if engine._is_castling_move(board, move):
                castling_moves.append(move)
        
        if castling_moves:
            print(f"Available castling moves: {[str(move) for move in castling_moves]}")
        else:
            print("No castling moves available")
        
        # Test the engine's choice
        print("Searching for best move...")
        best_move = engine.search(board, time_limit=5.0)
        
        print(f"Engine chose: {best_move}")
        
        # Check if the engine chose a castling move
        is_castling = engine._is_castling_move(board, best_move)
        if is_castling:
            print("✅ SUCCESS: Engine chose a castling move!")
        elif castling_moves:
            print(f"❌ ISSUE: Engine chose {best_move} instead of castling")
            print(f"Available castling moves were: {[str(move) for move in castling_moves]}")
        else:
            print("ℹ️ INFO: No castling moves were available")
        
        print()

def test_move_ordering():
    """Test that castling moves get high priority in move ordering"""
    
    print("\n" + "=" * 60)
    print("MOVE ORDERING TEST")
    print("=" * 60)
    
    engine = V7P3REngine()
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    
    moves = list(board.legal_moves)
    print(f"Total legal moves: {len(moves)}")
    
    # Test move ordering
    ordered_moves = engine._order_moves_advanced(board, moves, depth=3)
    
    print("\nMove ordering (first 10 moves):")
    for i, move in enumerate(ordered_moves[:10]):
        is_castling = engine._is_castling_move(board, move)
        is_capture = board.is_capture(move)
        gives_check = board.gives_check(move)
        
        move_type = []
        if is_castling:
            move_type.append("CASTLING")
        if is_capture:
            move_type.append("CAPTURE")
        if gives_check:
            move_type.append("CHECK")
        if not move_type:
            move_type.append("QUIET")
        
        print(f"{i+1:2d}. {move} ({', '.join(move_type)})")
    
    # Check if castling moves are prioritized
    castling_positions = []
    for i, move in enumerate(ordered_moves):
        if engine._is_castling_move(board, move):
            castling_positions.append(i + 1)
    
    if castling_positions:
        print(f"\nCastling moves appear at positions: {castling_positions}")
        if min(castling_positions) <= 5:
            print("✅ SUCCESS: Castling moves are high priority")
        else:
            print("❌ ISSUE: Castling moves should have higher priority")
    else:
        print("No castling moves found in ordering")

if __name__ == "__main__":
    test_castling_preference()
    test_move_ordering()
    
    print("\n" + "=" * 60)
    print("CASTLING TEST COMPLETE")
    print("=" * 60)