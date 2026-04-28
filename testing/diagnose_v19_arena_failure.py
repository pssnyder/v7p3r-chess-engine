#!/usr/bin/env python3
"""
Critical Analysis: Why is v19.5.6 performing so poorly in Arena tournaments?

Arena Results:
- Tournament 1 (5min+3s): 2.0/10 (20%)
- Tournament 2 (3min+2s): 1.0/10 (10%)
- Combined: 3.0/20 (15%)

Our validation script reported: 3.0/6 (50%)

Something is very wrong. Let's diagnose:
"""

import subprocess
from pathlib import Path
import chess

V19_PATH = Path(__file__).parent.parent / "src" / "v7p3r_uci.py"
V18_PATH = Path(__file__).parent.parent / "lichess" / "engines" / "V7P3R_v18.4_20260417" / "src" / "v7p3r_uci.py"

def test_version_identification():
    """Verify we're actually testing v19.5.6"""
    print("="*80)
    print("VERSION IDENTIFICATION CHECK")
    print("="*80)
    
    for path, label in [(V19_PATH, "Current"), (V18_PATH, "Baseline")]:
        process = subprocess.Popen(
            ["python", str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        process.stdin.write("uci\n")
        process.stdin.flush()
        
        version = None
        while True:
            line = process.stdout.readline().strip()
            if not line:
                break
            if line.startswith("id name"):
                version = line.split("id name ")[1]
            if line == "uciok":
                break
        
        print(f"\n{label} engine: {version}")
        print(f"  Path: {path}")
        
        process.stdin.write("quit\n")
        process.stdin.flush()
        process.wait(timeout=2)

def check_search_behavior():
    """Test if engine is making reasonable moves"""
    print("\n" + "="*80)
    print("SEARCH BEHAVIOR CHECK")
    print("="*80)
    
    # Test opening position
    board = chess.Board()
    
    process = subprocess.Popen(
        ["python", str(V19_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Init
    process.stdin.write("uci\n")
    process.stdin.flush()
    while True:
        line = process.stdout.readline().strip()
        if line == "uciok":
            break
    
    process.stdin.write("isready\n")
    process.stdin.flush()
    while True:
        line = process.stdout.readline().strip()
        if line == "readyok":
            break
    
    # Search
    process.stdin.write(f"position startpos\n")
    process.stdin.write(f"go movetime 5000\n")
    process.stdin.flush()
    
    best_move = None
    info_lines = []
    
    while True:
        line = process.stdout.readline().strip()
        if line.startswith("info"):
            info_lines.append(line)
        if line.startswith("bestmove"):
            best_move = line.split()[1]
            break
    
    process.stdin.write("quit\n")
    process.stdin.flush()
    process.wait(timeout=2)
    
    print(f"\nOpening position (5s search):")
    print(f"  Best move: {best_move}")
    print(f"  Info lines: {len(info_lines)}")
    
    if len(info_lines) > 0:
        print(f"  Last info: {info_lines[-1][:100]}...")
    
    # Check if move is legal
    try:
        move = chess.Move.from_uci(best_move)
        if move in board.legal_moves:
            print(f"  ✓ Move is legal")
        else:
            print(f"  ✗ ILLEGAL MOVE!")
    except:
        print(f"  ✗ INVALID MOVE FORMAT!")

def check_time_management():
    """Verify time management is working correctly"""
    print("\n" + "="*80)
    print("TIME MANAGEMENT CHECK")
    print("="*80)
    
    import time
    
    board = chess.Board()
    board.set_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
    
    process = subprocess.Popen(
        ["python", str(V19_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Init
    process.stdin.write("uci\n")
    process.stdin.flush()
    while True:
        line = process.stdout.readline().strip()
        if line == "uciok":
            break
    
    process.stdin.write("isready\n")
    process.stdin.flush()
    while True:
        line = process.stdout.readline().strip()
        if line == "readyok":
            break
    
    # Test with 10s limit
    process.stdin.write(f"position fen {board.fen()}\n")
    process.stdin.write(f"go movetime 10000\n")
    process.stdin.flush()
    
    start = time.time()
    
    while True:
        line = process.stdout.readline().strip()
        if line.startswith("bestmove"):
            break
    
    elapsed = time.time() - start
    
    process.stdin.write("quit\n")
    process.stdin.flush()
    process.wait(timeout=2)
    
    print(f"\n10s search test:")
    print(f"  Actual time: {elapsed:.2f}s")
    print(f"  Target: 9.0s (90% of 10s)")
    print(f"  Max: 10.0s (100%)")
    
    if elapsed > 10.5:
        print(f"  ✗ TIMEOUT! Exceeded limit by {elapsed - 10.0:.2f}s")
    elif elapsed > 10.0:
        print(f"  ⚠️  Over limit by {elapsed - 10.0:.2f}s")
    elif elapsed < 8.0:
        print(f"  ⚠️  TOO CONSERVATIVE! Only used {elapsed:.2f}s")
    else:
        print(f"  ✓ Good time usage ({elapsed/10.0*100:.1f}%)")

def analyze_arena_discrepancy():
    """Analyze why Arena results differ from our validation"""
    print("\n" + "="*80)
    print("ARENA vs VALIDATION DISCREPANCY ANALYSIS")
    print("="*80)
    
    print("""
Arena Tournaments (User's system):
  - Tournament 1 (5min+3s): 2.0/10 (20%)
  - Tournament 2 (3min+2s): 1.0/10 (10%)
  - Combined: 3.0/20 (15%)

Our Validation Script:
  - 6 games (5min+4s): 3.0/6 (50%)

POSSIBLE CAUSES:
1. Different engine versions tested
   - Arena may be using BAT file that points to different code
   - Need to verify V7P3R_v19_current.bat points to src/ directory

2. Time control differences
   - Arena used 5min+3s and 3min+2s
   - We tested 5min+4s
   - Shorter increment might cause issues

3. GUI differences
   - Arena may handle UCI differently than our script
   - Possible timing or move input issues

4. Color imbalance not caught
   - Need to check Arena results by color
   - v19.5.6 may have severe Black-side weakness

5. Validation script bugs
   - Our script may have miscounted results
   - May have run wrong engine versions

IMMEDIATE ACTIONS NEEDED:
1. Verify BAT file points to correct src/ directory
2. Check Arena PGN for color-based patterns
3. Re-test with Arena-style time controls
4. Fix whatever is causing 85% loss rate!

DO NOT DEPLOY v19.5.6 - it's catastrophically broken!
""")

if __name__ == "__main__":
    print("V19.5.6 CRITICAL DIAGNOSTIC")
    print("="*80)
    print()
    
    test_version_identification()
    check_search_behavior()
    check_time_management()
    analyze_arena_discrepancy()
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
Based on Arena tournament results showing 15% win rate:

v19.5.6 is NOT ready for deployment.

Next steps:
1. Run this diagnostic to identify root cause
2. Check BAT file configuration
3. Analyze Arena PGN games for patterns
4. Compare v19.5.6 code vs v18.4 for regressions
5. DO NOT DEPLOY until issues are resolved
""")
