#!/usr/bin/env python3
"""Test v18.0.0 Anti-Tactical Defense System"""

import chess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from v7p3r_move_safety import MoveSafetyChecker

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

def test_game2_pos1():
    print("\\n=== Test 1: Game 2 Position (Move 18) ===")
    fen = "r4rk1/pp3ppp/2nqb3/3n4/2BP4/1P6/PB3PPP/R3R1K1 b - - 3 18"
    board = chess.Board(fen)
    print(board)
    
    checker = MoveSafetyChecker(PIECE_VALUES)
    rf6 = chess.Move.from_uci("f8f6")
    safety = checker.evaluate_move_safety(board, rf6)
    
    print("\\n18...Rf6 safety: " + str(safety) + "cp")
    
    if safety < -100:
        print("PASS: Rf6 unsafe")
        return True
    else:
        print("FAIL: Rf6 should be unsafe")
        return False

def test_game2_pos2():
    print("\\n=== Test 2: Game 2 Position (Move 29) ===")
    fen = "8/pp3kpp/2nqb3/8/2B2R2/1P6/PB3PPP/6K1 b - - 2 29"
    board = chess.Board(fen)
    print(board)
    
    checker = MoveSafetyChecker(PIECE_VALUES)
    ke7 = chess.Move.from_uci("f7e7")
    safety = checker.evaluate_move_safety(board, ke7)
    
    print("\\n29...Ke7 safety: " + str(safety) + "cp")
    
    if safety < -50:
        print("PASS: Ke7 unsafe")
        return True
    else:
        print("FAIL: Ke7 should be unsafe")
        return False

def main():
    print("=" * 60)
    print("V7P3R v18.0.0 - Anti-Tactical Defense Tests")
    print("=" * 60)
    
    t1 = test_game2_pos1()
    t2 = test_game2_pos2()
    
    print("\\n" + "=" * 60)
    if t1 and t2:
        print("All tests PASSED")
        return 0
    else:
        print("Some tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
