#!/usr/bin/env python3
"""Quick verification of time management multipliers"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine


def main():
    engine = V7P3REngine()
    
    # Test with short time control where multipliers are visible
    time_left = 30.0  # 30 seconds
    increment = 0.0
    
    positions = [
        ("Opening", "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2"),
        ("Middlegame", "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ - 0 9"),
        ("Endgame", "8/5k2/8/4K3/8/8/4R3/8 w - - 0 1")
    ]
    
    print("Time Management Verification")
    print("=" * 60)
    print(f"Time left: {time_left}s, Increment: {increment}s")
    print(f"Base time (30s / 20) = {30/20:.3f}s")
    print()
    
    for name, fen in positions:
        engine.board = chess.Board(fen)
        phase = engine._get_game_phase(engine.board)
        time_limit = engine._calculate_time_limit(time_left, increment)
        
        # Calculate what base time would be
        base_time = time_left / 20  # This matches the 60s > time > 1min bracket
        
        # Calculate multiplier
        multiplier = time_limit / base_time if base_time > 0 else 0
        
        print(f"{name} (phase={phase}):")
        print(f"  Time limit: {time_limit:.3f}s")
        print(f"  Base time: {base_time:.3f}s")
        print(f"  Multiplier: {multiplier:.2f}x")
        print()


if __name__ == "__main__":
    main()
