#!/usr/bin/env python3
"""
Debug test for immediate threat detection
"""

import os
import sys
import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from v7p3r import V7P3REngine

def debug_threat_detection():
    engine = V7P3REngine()
    
    # Test 1: Simple check position - Queen checking the king
    print("TEST 1: King in check")
    board = chess.Board("rnbqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQK2q w KQq - 0 1")
    print(f"FEN: {board.fen()}")
    print(f"Is in check: {board.is_check()}")
    
    immediate_threats = engine._detect_immediate_threats(board)
    print(f"Immediate threats: {immediate_threats}")
    print()
    
    # Test 2: Real Scholar's mate threat - after 1.e4 e5 2.Bc4 Nc6 3.Qh5
    print("TEST 2: Real Scholar's mate threat")
    board2 = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 2 4")
    print(f"FEN: {board2.fen()}")
    print(f"Is in check: {board2.is_check()}")
    
    # Check if opponent threatens Qxf7#
    print("Checking if White threatens Qxf7#...")
    
    # Create a test board to check the threat
    test_board = board2.copy()
    test_board.turn = chess.WHITE  # Switch to white's perspective
    
    # Check all White's legal moves for mate threats
    mate_found = False
    for move in test_board.legal_moves:
        temp_board = test_board.copy()
        temp_board.push(move)
        if temp_board.is_checkmate():
            print(f"White has mate in 1 with {test_board.san(move)}")
            mate_found = True
            break
    
    if not mate_found:
        print("No immediate mate found for White")
    
    immediate_threats2 = engine._detect_immediate_threats(board2)
    print(f"Immediate threats: {immediate_threats2}")
    print()
    
    # Test 3: Test mate threat detection directly
    print("TEST 3: Direct mate threat detection")
    mate_threats = engine._detect_mate_threats(board2)
    print(f"Mate threats: {mate_threats}")

if __name__ == "__main__":
    debug_threat_detection()