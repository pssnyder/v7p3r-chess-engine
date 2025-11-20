"""Debug TT behavior during iterative deepening"""
import sys
sys.path.insert(0, 'src')
from v7p3r import V7P3REngine
import chess
import time

print("="*70)
print("TRANSPOSITION TABLE DEBUG")
print("="*70)

# Monkey-patch the TT functions to add debug output
original_probe_tt = V7P3REngine._probe_tt
original_store_tt = V7P3REngine._store_tt_entry

def debug_probe_tt(self, zobrist_key, depth, alpha, beta):
    result_value, result_move = original_probe_tt(self, zobrist_key, depth, alpha, beta)
    entry = self.transposition_table.get(zobrist_key)
    
    if entry:
        print(f"  [TT PROBE] key={zobrist_key % 10000}, req_depth={depth}, "
              f"stored_depth={entry.depth}, stored_value={entry.value}, "
              f"stored_move={entry.best_move}, "
              f"returns=({result_value}, {result_move})")
    
    return result_value, result_move

def debug_store_tt(self, zobrist_key, depth, value, node_type, best_move):
    print(f"  [TT STORE] key={zobrist_key % 10000}, depth={depth}, value={value}, "
          f"type={node_type.name}, move={best_move}")
    return original_store_tt(self, zobrist_key, depth, value, node_type, best_move)

V7P3REngine._probe_tt = debug_probe_tt
V7P3REngine._store_tt_entry = debug_store_tt

# Now run a search
engine = V7P3REngine(max_depth=4, tt_size_mb=256, tablebase_path='')

fen = 'r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 10'
board = chess.Board(fen)
engine.board = board

print(f"\nPosition: {fen}")
print(f"Max depth: {engine.max_depth}\n")

engine.start_time = time.time()
engine.time_limit = 0
engine.nodes_searched = 0
engine.age += 1

root_zobrist = engine._get_zobrist_key(engine.board)
print(f"ROOT zobrist key (last 4 digits): {root_zobrist % 10000}\n")

for depth in range(1, engine.max_depth + 1):
    print(f"{'='*70}")
    print(f"DEPTH {depth}")
    print(f"{'='*70}")
    
    value, move = engine._search(engine.board, depth, -float('inf'), float('inf'), 0)
    
    print(f"\n  RESULT: value={value}, move={move}")
    print()

print(f"{'='*70}")
print(f"FINAL TT STATE")
print(f"{'='*70}")
print(f"TT size: {len(engine.transposition_table)} entries")
print(f"\nRoot position entry:")
entry = engine.transposition_table.get(root_zobrist)
if entry:
    print(f"  Depth: {entry.depth}")
    print(f"  Value: {entry.value}")
    print(f"  Move: {entry.best_move}")
    print(f"  Type: {entry.node_type.name}")
else:
    print("  NOT FOUND!")
