#!/usr/bin/env python3
"""
Test mate detection and scoring with known mate positions
"""

import sys
import os
sys.path.append('src')

import chess
from v7p3r import V7P3RCleanEngine


def test_known_mate_positions():
    """Test positions with known mates"""
    engine = V7P3RCleanEngine()
    
    # Test position 1: White to move, mate in 1
    print("=== Testing Mate in 1 ===")
    board1 = chess.Board("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4")
    print(f"Position: {board1.fen()}")
    print(f"Legal moves: {[str(m) for m in board1.legal_moves]}")
    
    # Check if Bxf7# is mate
    mate_move = chess.Move.from_uci("b5f7")
    if mate_move in board1.legal_moves:
        board1.push(mate_move)
        print(f"After Bxf7+: {board1.is_checkmate()}")
        board1.pop()
    
    best_move1 = engine.search(board1, time_limit=2.0)
    print(f"Engine's best move: {best_move1}")
    print()
    
    # Test position 2: Back rank mate in 1
    print("=== Testing Back Rank Mate in 1 ===")
    board2 = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    print(f"Position: {board2.fen()}")
    
    best_move2 = engine.search(board2, time_limit=2.0)
    print(f"Engine's best move: {best_move2}")
    print()
    
    # Test the actual position from the tournament where the mate happened
    print("=== Testing Tournament Final Position (Nf6# mate) ===")
    # Position right before 10...Nf6#
    board3 = chess.Board("r1b1kbnr/ppp2ppp/3p4/4p3/2qPK1N1/2N1PQ2/PPP2PPP/R1B4R b kq - 1 10")
    print(f"Position: {board3.fen()}")
    print(f"Legal moves: {[str(m) for m in board3.legal_moves]}")
    
    # Check if Nf6+ is mate
    if chess.Move.from_uci("g8f6") in board3.legal_moves:
        board3.push(chess.Move.from_uci("g8f6"))
        print(f"After Nf6+: {board3.is_checkmate()}")
        board3.pop()
    
    best_move3 = engine.search(board3, time_limit=2.0)
    print(f"Engine's best move: {best_move3}")


if __name__ == "__main__":
    test_known_mate_positions()
