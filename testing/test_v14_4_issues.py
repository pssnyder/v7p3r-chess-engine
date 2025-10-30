#!/usr/bin/env python3
"""
Quick diagnostic test for V14.4 issues
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

def test_v14_4_evaluation():
    """Test if evaluation is working correctly"""
    print("Testing V14.4 Evaluation Issues")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # Test position from the game where v14.4 played Nxd4 (move 5)
    # Position after 1.e3 Nc6 2.Nc3 Nf6 3.d4 Nd5 4.Nxd5 e6
    board = chess.Board("rnbqkb1r/pppp1ppp/4p3/8/3P4/4P3/PPP2PPP/R1BQKBNR w KQkq - 0 5")
    
    print(f"\nPosition after 4...e6:")
    print(board)
    print(f"FEN: {board.fen()}")
    
    # Check available moves
    legal_moves = list(board.legal_moves)
    print(f"\nLegal moves: {len(legal_moves)}")
    
    # Key moves to test:
    # Nc3 (normal development) vs Nxd4 (blundering knight to pawn)
    nc3 = chess.Move.from_uci("d5c3")
    
    # Test if the move exists
    if nc3 in legal_moves:
        print("\nNc3 is legal")
        
        # Evaluate position after Nc3
        board_after_nc3 = board.copy()
        board_after_nc3.push(nc3)
        score_nc3 = engine._evaluate_position(board_after_nc3)
        print(f"Position after Nc3: {score_nc3}")
    
    # Check what move the engine actually wants to play
    print("\n" + "="*60)
    print("Running engine search (3 seconds)...")
    best_move = engine.search(board, time_limit=3.0)
    print(f"Engine chose: {best_move}")
    
    # Look at the actual game move that was played
    print("\n" + "="*60)
    print("Checking the blunder from the actual game...")
    # In the game, Black played Nxd4 on move 5
    # Let's check the position after White's move 5: Nc3
    board_move_5 = chess.Board("rnbqkb1r/pppp1ppp/4p3/8/3P4/2N1P3/PPP2PPP/R1BQKB1R b KQkq - 1 5")
    print(f"\nPosition before Nxd4 blunder:")
    print(board_move_5)
    
    # Check if engine would play the blunder
    print("\nRunning engine search (3 seconds)...")
    black_move = engine.search(board_move_5, time_limit=3.0)
    print(f"Engine chose: {black_move}")
    
    # Test the blunder move
    nxd4 = chess.Move.from_uci("c6d4")
    if nxd4 in board_move_5.legal_moves:
        board_after_nxd4 = board_move_5.copy()
        board_after_nxd4.push(nxd4)
        score_nxd4 = engine._evaluate_position(board_after_nxd4)
        print(f"\nPosition after Nxd4 (blunder): {score_nxd4}")
        print("This should be significantly worse for Black!")

if __name__ == "__main__":
    test_v14_4_evaluation()
