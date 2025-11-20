"""Debug why engine stops at depth 2 instead of reaching max_depth"""
import sys
sys.path.insert(0, 'src')
from v7p3r import V7P3REngine
import chess
import time

print("="*60)
print("DEPTH INVESTIGATION - V16.1")
print("="*60)

engine = V7P3REngine(max_depth=10, tt_size_mb=256, tablebase_path='')

# Position outside opening book
fen = 'r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 10'
board = chess.Board(fen)
engine.board = board

print(f"\nConfiguration:")
print(f"  max_depth: {engine.max_depth}")
print(f"  TT size: 256 MB")
print(f"\nPosition: {fen}")
print(f"Position is OUTSIDE opening book\n")

# Manually run the search loop with debug output
engine.start_time = time.time()
engine.time_limit = engine._calculate_time_limit(0, 0)
engine.nodes_searched = 0
engine.age += 1

print(f"Time configuration:")
print(f"  time_left: 0 (infinite)")
print(f"  time_limit: {engine.time_limit}")
print(f"  _is_time_up() before loop: {engine._is_time_up()}\n")

best_move = None
best_value = -float('inf')

print("Starting iterative deepening loop...\n")

for depth in range(1, engine.max_depth + 1):
    print(f"[LOOP] Starting depth {depth} / {engine.max_depth}")
    
    # Check time BEFORE search
    if engine._is_time_up():
        print(f"[LOOP] TIME UP before search at depth {depth}!")
        break
    
    search_start = time.time()
    value, move = engine._search(engine.board, depth, -float('inf'), float('inf'), 0)
    search_time = time.time() - search_start
    
    print(f"[LOOP] Depth {depth} complete: value={value}, move={move}, time={search_time:.3f}s")
    
    if move is not None:
        best_move = move
        best_value = value
        
        nps = int(engine.nodes_searched / max(search_time, 0.001))
        print(f"info depth {depth} score cp {value} nodes {engine.nodes_searched} "
              f"nps {nps} time {int(search_time * 1000)} pv {move.uci()}")
        sys.stdout.flush()
    
    # Check time AFTER search
    if engine._is_time_up():
        print(f"[LOOP] TIME UP after search at depth {depth}!")
        break
    
    print()  # Blank line between depths

total_time = time.time() - engine.start_time
print(f"\n{'='*60}")
print(f"Search completed:")
print(f"  Total time: {total_time:.3f}s")
print(f"  Total nodes: {engine.nodes_searched}")
print(f"  Best move: {board.san(best_move) if best_move else 'None'}")
print(f"  Best value: {best_value}")
print(f"  Depths completed: ???")
print(f"{'='*60}")
