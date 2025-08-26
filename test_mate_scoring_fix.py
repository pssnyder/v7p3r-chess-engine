#!/usr/bin/env python3
"""
Test script to verify the mate scoring fix in V7P3R
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import chess
from v7p3r import V7P3RCleanEngine


def test_mate_scoring():
    """Test mate scoring with known mate positions"""
    engine = V7P3RCleanEngine()
    
    # Test position 1: White to mate in 1
    # Scholar's mate position - Qh5 threatens mate on f7
    board1 = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 2 4")
    print("Test 1: White to mate in 1")
    print(f"Position: {board1.fen()}")
    
    try:
        move1 = engine.search(board1, time_limit=2.0)
        print(f"Best move: {move1}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Test position 2: Black in checkmate
    board2 = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 2")
    board2.push(chess.Move.from_uci("f8e7"))  # King moves
    board2.push(chess.Move.from_uci("d1h5"))  # Queen to h5, threatening mate
    board2.push(chess.Move.from_uci("g8f6"))  # Knight blocks
    board2.push(chess.Move.from_uci("h5f7"))  # Checkmate!
    
    print("Test 2: Checkmate position")
    print(f"Position: {board2.fen()}")
    print(f"Is checkmate: {board2.is_checkmate()}")
    print()
    
    # Test position 3: Simple 2-move mate
    board3 = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    print("Test 3: White has mate in 2")
    print(f"Position: {board3.fen()}")
    
    try:
        move3 = engine.search(board3, time_limit=3.0)
        print(f"Best move: {move3}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()
    
    # Test the score formatting function directly
    print("Test 4: Score formatting tests")
    print(f"Score 100 (cp): {engine._format_uci_score(1.0, 6)}")
    print(f"Score -200 (cp): {engine._format_uci_score(-2.0, 6)}")
    print(f"Score mate+3: {engine._format_uci_score(29997, 6)}")  # 30000 - 3
    print(f"Score mate-4: {engine._format_uci_score(-29996, 6)}")  # -30000 + 4
    print()


if __name__ == "__main__":
    test_mate_scoring()
