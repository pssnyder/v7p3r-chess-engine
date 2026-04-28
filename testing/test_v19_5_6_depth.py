#!/usr/bin/env python3
"""
Quick depth comparison: v19.5.6 vs v18.4

Tests both engines on the same position to verify v19.5.6 achieves
competitive search depth without the 67% threshold.
"""

import subprocess
import time
import chess
from pathlib import Path

V19_PATH = Path(__file__).parent.parent / "src" / "v7p3r_uci.py"
V18_PATH = Path(__file__).parent.parent / "lichess" / "engines" / "V7P3R_v18.4_20260417" / "src" / "v7p3r_uci.py"

def test_engine_depth(engine_path: Path, position_fen: str, time_limit: float = 5.0):
    """Test depth achieved by engine"""
    process = subprocess.Popen(
        ["python", str(engine_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Initialize
    process.stdin.write("uci\n")
    process.stdin.flush()
    
    engine_name = None
    while True:
        line = process.stdout.readline().strip()
        if not line:
            break
        if line == "uciok":
            break
        if line.startswith("id name"):
            engine_name = line.split("id name ")[1]
    
    process.stdin.write("isready\n")
    process.stdin.flush()
    while True:
        line = process.stdout.readline().strip()
        if line == "readyok":
            break
    
    # Search
    process.stdin.write(f"position fen {position_fen}\n")
    process.stdin.write(f"go movetime {int(time_limit * 1000)}\n")
    process.stdin.flush()
    
    start = time.time()
    max_depth = 0
    depth_times = []
    
    while True:
        line = process.stdout.readline().strip()
        if line.startswith("info depth"):
            parts = line.split()
            try:
                depth_idx = parts.index("depth")
                depth = int(parts[depth_idx + 1])
                
                time_idx = parts.index("time")
                time_ms = int(parts[time_idx + 1])
                
                if depth > max_depth:
                    max_depth = depth
                    depth_times.append((depth, time_ms / 1000))
            except (ValueError, IndexError):
                pass
        elif line.startswith("bestmove"):
            break
    
    elapsed = time.time() - start
    
    process.stdin.write("quit\n")
    process.stdin.flush()
    process.wait(timeout=2)
    
    return {
        "engine": engine_name,
        "max_depth": max_depth,
        "elapsed": elapsed,
        "depth_progression": depth_times
    }

if __name__ == "__main__":
    # Test position from PGN (complex middlegame)
    test_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
    time_limit = 5.0
    
    print("="*60)
    print("DEPTH COMPARISON TEST: v19.5.6 vs v18.4")
    print("="*60)
    print(f"Position: {test_fen}")
    print(f"Time limit: {time_limit}s\n")
    
    print("Testing v19.5.6...")
    v19_result = test_engine_depth(V19_PATH, test_fen, time_limit)
    
    print("Testing v18.4...")
    v18_result = test_engine_depth(V18_PATH, test_fen, time_limit)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\n{v19_result['engine']}:")
    print(f"  Max depth: {v19_result['max_depth']}")
    print(f"  Elapsed: {v19_result['elapsed']:.2f}s")
    print(f"  Depth progression:")
    for depth, time_sec in v19_result['depth_progression']:
        print(f"    depth {depth} @ {time_sec:.2f}s")
    
    print(f"\n{v18_result['engine']}:")
    print(f"  Max depth: {v18_result['max_depth']}")
    print(f"  Elapsed: {v18_result['elapsed']:.2f}s")
    print(f"  Depth progression:")
    for depth, time_sec in v18_result['depth_progression']:
        print(f"    depth {depth} @ {time_sec:.2f}s")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    depth_diff = v19_result['max_depth'] - v18_result['max_depth']
    
    if depth_diff >= 0:
        print(f"✓ v19.5.6 matches or exceeds v18.4 depth (+{depth_diff} plies)")
        print(f"✓ Time management restored successfully")
        print(f"\n→ READY FOR TOURNAMENT VALIDATION")
    else:
        print(f"✗ v19.5.6 still {abs(depth_diff)} plies behind v18.4")
        print(f"✗ Time management needs further adjustment")
        print(f"\n→ CONTINUE DEBUGGING")
