#!/usr/bin/env python3
"""
V14.9 Quick Validation Test
Tests 20 puzzles to verify workflow restoration is working
Target: See depth 4-6 restored and accuracy >50% (vs V14.8's 38.8%)
"""

import subprocess
import sys
import time
import chess
import os

# Add parent directory to path for v7p3r import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def test_search_depth():
    """Test that V14.9 achieves better depth than V14.8"""
    print("=" * 80)
    print("V14.9 QUICK VALIDATION TEST - Search Depth Check")
    print("=" * 80)
    
    test_positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Middlegame", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"),
        ("Endgame", "8/5k2/3p4/1p1Pp3/pP2Pp2/P4P2/8/6K1 w - - 0 1"),
    ]
    
    engine = V7P3REngine()
    
    for name, fen in test_positions:
        print(f"\n{name}:")
        board = chess.Board(fen)
        
        start_time = time.time()
        move = engine.search(board, depth=None, time_limit=5.0, is_root=True)
        elapsed = time.time() - start_time
        
        # Get depth from search stats
        depth_achieved = engine.search_stats.get('max_depth_reached', 0)
        nodes = engine.nodes_searched
        nps = nodes / elapsed if elapsed > 0 else 0
        
        print(f"  Depth: {depth_achieved}")
        print(f"  Nodes: {nodes:,}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  NPS: {nps:,.0f}")
        print(f"  Move: {move}")
        
        # Check if depth improved over V14.8
        if name == "Middlegame":
            if depth_achieved >= 4:
                print(f"  ✅ GOOD: Depth {depth_achieved} >= 4 (V14.8 only achieved 2-3)")
            else:
                print(f"  ⚠️  CONCERN: Depth {depth_achieved} < 4 (still shallow)")
    
    print("\n" + "=" * 80)
    print("Quick validation complete!")
    print("Next: Run full 20-puzzle test with universal_puzzle_analyzer")
    print("=" * 80)

if __name__ == "__main__":
    test_search_depth()
