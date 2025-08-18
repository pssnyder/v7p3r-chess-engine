#!/usr/bin/env python
# test_capture_escape.py
"""
Test script for the capture-to-escape-check functionality in V7P3R.
This script will load a specific position where capturing is needed to escape check,
and run the engine with the capture_escape_config.json to see if it makes the correct move.
"""

import chess
import argparse
from v7p3r_engine import V7P3REngine

def setup_position():
    """
    Setup a position where capturing is the best way to escape check.
    In this position, white is in check and can capture the checking piece (a queen).
    """
    # Set up a position where white is in check by a queen and can capture it
    fen = "r3kb1r/ppp2ppp/2n5/4q3/3Q4/5N2/PPP2PPP/R1B1K2R w KQkq - 0 1"
    board = chess.Board(fen)
    
    # Verify that white is in check
    assert board.is_check(), "Position should have white in check"
    
    # Verify that capturing the queen is a legal move
    capture_move = chess.Move.from_uci("d4e5")
    assert capture_move in board.legal_moves, "Capturing the queen should be a legal move"
    
    return board

def main():
    parser = argparse.ArgumentParser(description='Test capture-to-escape-check functionality')
    parser.add_argument('--config', '-c', default='capture_escape_config.json',
                       help='Configuration file to use (default: capture_escape_config.json)')
    
    args = parser.parse_args()
    
    # Setup the position
    board = setup_position()
    print(f"Testing position: {board.fen()}")
    print(board)
    
    # Initialize the engine with the specified config
    engine = V7P3REngine(args.config)
    
    # Find the best move
    best_move = engine.find_move(board)
    
    if best_move is None:
        print("\nFAILURE: Engine did not find any legal move!")
        return
    
    # Check if the engine found the capture move
    capture_move = chess.Move.from_uci("d4e5")
    if best_move == capture_move:
        print("\nSUCCESS: Engine found the capture move to escape check!")
    else:
        print(f"\nFAILURE: Engine did not find the capture move. Selected: {best_move}")
    
    # Print additional information
    print(f"\nBoard after move:")
    board.push(best_move)
    print(board)
    print(f"\nIs still in check: {board.is_check()}")
    
    # Check if white is still in check after the move
    if board.is_check():
        print("WARNING: White is still in check after the move!")

if __name__ == "__main__":
    main()
