#!/usr/bin/env python3
"""
Quick depth comparison test: v19.5.5 (2.0x) vs v19.5.4 (2.5x)

Tests same position with 30s limit to see if 2.0x achieves better depth
while still respecting timeout.
"""

import subprocess
import time
import chess
from pathlib import Path

ENGINE_PATH = Path(__file__).parent.parent / "src" / "v7p3r_uci.py"

def test_depth(position_fen: str, time_limit: float = 30.0):
    """Test depth achieved in given time"""
    process = subprocess.Popen(
        ["python", str(ENGINE_PATH)],
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
        if line == "uciok":
            break
        if line.startswith("id name"):
            engine_name = line.split("id name ")[1]
    
    process.stdin.write("isready\n")
    process.stdin.flush()
    while process.stdout.readline().strip() != "readyok":
        pass
    
    # Search
    process.stdin.write(f"position fen {position_fen}\n")
    process.stdin.write(f"go movetime {int(time_limit * 1000)}\n")
    process.stdin.flush()
    
    start = time.time()
    max_depth = 0
    last_score = 0
    info_lines = []
    
    while True:
        line = process.stdout.readline().strip()
        if line.startswith("info depth"):
            parts = line.split()
            try:
                depth_idx = parts.index("depth")
                depth = int(parts[depth_idx + 1])
                max_depth = max(max_depth, depth)
                
                score_idx = parts.index("score")
                score_type = parts[score_idx + 1]
                score = int(parts[score_idx + 2])
                last_score = score
                
                time_idx = parts.index("time")
                time_ms = int(parts[time_idx + 1])
                
                info_lines.append(f"  depth {depth} @ {time_ms/1000:.2f}s")
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
        "score": last_score,
        "info": info_lines
    }

if __name__ == "__main__":
    # Test position: Complex middlegame with tactical opportunities
    # From earlier v19.5.3 timeout failure
    test_fen = "r7/pb2R3/2pkB3/1p6/8/7N/1P4P1/2K5 w - - 3 43"
    
    print("Depth Comparison Test: v19.5.5 (2.0x factor)")
    print(f"Position: {test_fen}")
    print(f"Time limit: 30s\n")
    
    result = test_depth(test_fen, 30.0)
    
    print(f"Engine: {result['engine']}")
    print(f"Max depth: {result['max_depth']}")
    print(f"Elapsed: {result['elapsed']:.2f}s")
    print(f"Final score: {result['score']}cp")
    print(f"\nDepth progression:")
    for line in result['info']:
        print(line)
    
    # Success criteria
    print("\n" + "="*60)
    timeout = result['elapsed'] > 32.0
    good_depth = result['max_depth'] >= 5
    
    print(f"Timeout check (≤32s): {'PASS' if not timeout else 'FAIL'} ({result['elapsed']:.2f}s)")
    print(f"Depth check (≥5): {'PASS' if good_depth else 'FAIL'} (depth {result['max_depth']})")
    
    if not timeout and good_depth:
        print("\n✓ Ready for tournament validation")
    else:
        print("\n✗ Needs adjustment")
