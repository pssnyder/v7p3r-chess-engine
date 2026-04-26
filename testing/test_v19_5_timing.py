"""
Test v19.5 timing without predictive completion logic.
Should use full 90% of allocated time (9s out of 10s).
"""

import sys
sys.path.insert(0, 'src')

from v7p3r import V7P3REngine
import chess
import time

# Test position
board = chess.Board('r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1')

# Create engine
engine = V7P3REngine()

# Run search with 10s time limit
print("Starting 10-second search test...")
start = time.time()
move = engine.search(board, time_limit=10.0)
elapsed = time.time() - start

# Get stats
nodes = engine.nodes_searched
depth = engine.search_stats.get('depth', 'unknown')

# Results
print("\n" + "=" * 60)
print("v19.5 NO PREDICTIVE TIMING TEST")
print("=" * 60)
print(f"Time Used:      {elapsed:.2f}s (allocated: 10s, target: 9s)")
print(f"Time Usage:     {(elapsed/10.0)*100:.1f}%")
print(f"Best Move:      {move}")
print(f"Depth Reached:  {depth}")
print(f"Nodes:          {nodes:,}")
print(f"NPS:            {nodes/elapsed:,.0f}")
print("=" * 60)

# Validation
if elapsed >= 8.5:
    print("✓ SUCCESS: Engine used full time budget (>8.5s)")
else:
    print(f"✗ ISSUE: Engine stopped early at {elapsed:.2f}s")
