#!/usr/bin/env python3
"""
Quick test to verify v19.5.2 timeout fix

Tests that engine respects time_limit parameter and doesn't
continue iterative deepening after timeout.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chess
from v7p3r import V7P3REngine

def test_timeout_respect():
    """Test that engine respects the time limit"""
    engine = V7P3REngine()
    
    # Complex endgame position that caused 135s timeout in v19.5.1
    # From Game 3, move 33
    board = chess.Board("4r3/5pkp/5np1/3p4/3N4/2P5/1R3PPP/6K1 w - - 0 33")
    
    print("Testing v19.5.2 timeout fix...")
    print(f"Position: {board.fen()}")
    print(f"Time limit: 10.0 seconds\n")
    
    start = time.time()
    move = engine.search(board, time_limit=10.0)
    elapsed = time.time() - start
    
    print(f"\nResult:")
    print(f"  Move: {move.uci()}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Usage: {elapsed/10.0*100:.1f}%")
    print(f"  Nodes: {engine.nodes_searched:,}")
    
    # Check if timeout was respected
    if elapsed <= 12.0:  # Allow 2s grace period for cleanup
        print(f"\n✓ PASS: Engine respected time limit!")
        if elapsed > 10.5:
            print(f"  (Warning: 5% overage, but acceptable)")
        return True
    else:
        print(f"\n✗ FAIL: Engine exceeded time limit by {elapsed-10.0:.2f}s!")
        return False

if __name__ == "__main__":
    success = test_timeout_respect()
    sys.exit(0 if success else 1)
