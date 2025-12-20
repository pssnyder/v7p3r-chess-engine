#!/usr/bin/env python3
"""Benchmark v18.0.0 Performance Impact"""

import chess
import sys
import os
import time

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

def benchmark_safety_checker():
    print("=" * 60)
    print("V7P3R v18.0.0 - Move Safety Performance Benchmark")
    print("=" * 60)
    
    # Test position (middlegame with ~30 legal moves)
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5")
    checker = MoveSafetyChecker(PIECE_VALUES)
    moves = list(board.legal_moves)
    
    print("\\nTest position: Middlegame with {} legal moves".format(len(moves)))
    print("Running 1000 safety evaluations...")
    
    start = time.time()
    iterations = 1000
    
    for _ in range(iterations):
        for move in moves:
            checker.evaluate_move_safety(board, move)
    
    elapsed = time.time() - start
    total_checks = iterations * len(moves)
    checks_per_sec = total_checks / elapsed
    time_per_check = (elapsed / total_checks) * 1000000  # microseconds
    
    print("\\nResults:")
    print("  Total checks: {}".format(total_checks))
    print("  Time elapsed: {:.3f}s".format(elapsed))
    print("  Checks/second: {:.0f}".format(checks_per_sec))
    print("  Time/check: {:.1f} microseconds".format(time_per_check))
    
    # Estimate overhead in real search
    print("\\nEstimated overhead in search:")
    print("  At depth 4 (~1000 positions): {:.3f}s".format(1000 / checks_per_sec))
    print("  At depth 5 (~3000 positions): {:.3f}s".format(3000 / checks_per_sec))
    
    if checks_per_sec > 500:
        print("\\nPASS: Performance acceptable (>500 checks/sec)")
        return 0
    else:
        print("\\nWARN: Performance may impact search speed")
        return 1

if __name__ == "__main__":
    sys.exit(benchmark_safety_checker())
