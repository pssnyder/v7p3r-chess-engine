"""Check if ordered_moves is empty at depth 3"""
import sys
sys.path.insert(0, 'src')
from v7p3r import V7P3REngine
import chess

# Monkey-patch _search to add debug
original_search = V7P3REngine._search

def debug_search(self, board, depth, alpha, beta, ply, do_null_move=True):
    if ply == 0:  # Root level only
        legal_moves = list(board.legal_moves)
        zobrist_key = self._get_zobrist_key(board)
        tt_value, tt_move = self._probe_tt(zobrist_key, depth, alpha, beta)
        ordered_moves = self._filter_and_order_moves(board, legal_moves, ply, tt_move)
        
        print(f"\n[ROOT SEARCH DEBUG] depth={depth}")
        print(f"  Legal moves: {len(legal_moves)}")
        print(f"  TT move hint: {tt_move}")
        print(f"  Ordered moves: {len(ordered_moves)}")
        if len(ordered_moves) == 0:
            print(f"  ^^^ EMPTY! This is the bug!")
    
    return original_search(self, board, depth, alpha, beta, ply, do_null_move)

V7P3REngine._search = debug_search

engine = V7P3REngine(max_depth=4, tt_size_mb=256, tablebase_path='')
fen = 'r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 10'
board = chess.Board(fen)
engine.board = board

print("Testing depths 1-4...")
best_move = engine.get_best_move(time_left=0, increment=0)
print(f"\nFinal: {board.san(best_move) if best_move else 'None'}")
