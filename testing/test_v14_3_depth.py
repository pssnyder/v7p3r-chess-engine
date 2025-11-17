#!/usr/bin/env python3
"""
V14.3 Depth Test - Parse actual depth from search output
"""

import chess
import sys
import time
import re
from pathlib import Path
from io import StringIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v7p3r import V7P3REngine

def test_depth_at_time_control(seconds=5):
    """Test what depth V14.3 reaches in given time"""
    engine = V7P3REngine()
    
    test_positions = [
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Italian Game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"),
        ("Middlegame", "r2qkb1r/ppp2ppp/2n1pn2/3p4/2PP4/2N2NP1/PP2PP1P/R1BQKB1R w KQkq - 0 7"),
    ]
    
    print(f"V14.3 Depth Test @ {seconds}s per position")
    print("=" * 80)
    
    results = []
    
    # Capture stdout to parse depth
    import sys
    from io import StringIO
    
    for name, fen in test_positions:
        board = chess.Board(fen)
        
        print(f"\n{name}:")
        print(f"FEN: {fen[:60]}...")
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        start = time.time()
        best_move = engine.search(board, time_limit=seconds)
        elapsed = time.time() - start
        
        # Restore stdout
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        
        # Print the output
        print(output, end='')
        
        # Parse depth from output (last "info depth N" line)
        depths = re.findall(r'info depth (\d+)', output)
        depth = int(depths[-1]) if depths else 0
        
        nodes = engine.nodes_searched
        nps = int(nodes / elapsed) if elapsed > 0 else 0
        
        print(f"  Final Depth: {depth}")
        print(f"  Nodes: {nodes:,}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  NPS: {nps:,}")
        print(f"  Best move: {best_move}")
        
        results.append({
            'position': name,
            'depth': depth,
            'nodes': nodes,
            'nps': nps,
            'time': elapsed
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    avg_depth = sum(r['depth'] for r in results) / len(results)
    avg_nps = sum(r['nps'] for r in results) / len(results)
    
    print(f"Average depth: {avg_depth:.1f}")
    print(f"Average NPS: {avg_nps:,.0f}")
    
    print("\nTARGET COMPARISON:")
    print(f"  MaterialOpponent: depth 7-8, ~100K NPS")
    print(f"  V7P3R v14.3: depth {avg_depth:.1f}, ~{avg_nps:,.0f} NPS")
    
    if avg_depth >= 7:
        print("\n✓ DEPTH TARGET ACHIEVED!")
    else:
        print(f"\n⚠ Need {7 - avg_depth:.1f} more depth to match MaterialOpponent")
    
    if avg_nps >= 95000:
        print("✓ NPS TARGET ACHIEVED!")
    else:
        print(f"⚠ Need {95000 - avg_nps:,.0f} more NPS to match MaterialOpponent")

if __name__ == "__main__":
    test_depth_at_time_control(5)
