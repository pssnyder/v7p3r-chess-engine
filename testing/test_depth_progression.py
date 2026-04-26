"""
Compare v19.5 depth progression at different time allocations.
Test if PVS + full time usage can reach depth 8-10.
"""

import sys
sys.path.insert(0, 'src')

from v7p3r import V7P3REngine
import chess
import time

# Test position (Italian Game)
board = chess.Board('r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1')

# Create engine
engine = V7P3REngine()

# Test at different time allocations
time_allocations = [5, 10, 15, 20]

print("\n" + "=" * 80)
print("v19.5 DEPTH PROGRESSION TEST (PVS + No Predictive Timing)")
print("=" * 80)

for time_limit in time_allocations:
    # Create fresh engine for each test (clear TT between runs)
    engine = V7P3REngine()
    
    # Run search
    start = time.time()
    move = engine.search(board, time_limit=float(time_limit))
    elapsed = time.time() - start
    
    # Get stats
    nodes = engine.nodes_searched
    nps = nodes / elapsed if elapsed > 0 else 0
    
    # Extract depth from info output (last depth before time ran out)
    depth = "unknown"
    # We can extract this from the info string output, but for now just report nodes
    
    print(f"\nTime: {time_limit}s | Used: {elapsed:.2f}s ({elapsed/time_limit*100:.1f}%) | "
          f"Nodes: {nodes:,} | NPS: {nps:,.0f} | Move: {move}")

print("\n" + "=" * 80)
print("Note: Depth information extracted from info output above")
print("Goal: Reach depth 8-10 at 15-20 second time controls")
print("=" * 80)
