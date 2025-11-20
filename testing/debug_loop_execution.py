#!/usr/bin/env python3
"""Debug the actual loop execution at depth 3"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

# Patch the _search function to add loop debug
original_search = V7P3REngine._search

iteration_count = 0

def debug_search(self, board, depth, alpha, beta, ply, do_null_move=True):
    """Wrapped search with loop debug output"""
    global iteration_count
    
    if ply == 0 and depth >= 3:  # Only debug root at depth 3+
        print(f"\n[DEPTH {depth} ROOT SEARCH STARTING]")
        print(f"  alpha={alpha}, beta={beta}")
        
        # Get zobrist and check for early exits
        zobrist_key = self._get_zobrist_key(board)
        
        # Check game over
        if board.is_checkmate():
            print(f"  [EARLY EXIT] Checkmate detected")
            return -30000 + ply, None
        if board.is_stalemate():
            print(f"  [EARLY EXIT] Stalemate detected")
            return 0, None
        
        # Check depth
        if depth <= 0:
            print(f"  [EARLY EXIT] Depth <= 0, going to quiescence")
            return self._quiescence_search(board, alpha, beta), None
        
        # Check time
        if self._is_time_up():
            print(f"  [EARLY EXIT] Time is up")
            return self._evaluate_position(board), None
        
        # Check TT
        entry = self.transposition_table.get(zobrist_key)
        if entry and entry.depth >= depth:
            print(f"  [EARLY EXIT] TT hit with sufficient depth")
            if entry.node_type == 0:  # EXACT
                return entry.value, entry.best_move
        
        # Generate moves
        legal_moves = list(board.legal_moves)
        print(f"  Legal moves: {len(legal_moves)}")
        
        tt_move = entry.best_move if entry else None
        ordered_moves = self._filter_and_order_moves(board, legal_moves, ply, tt_move)
        print(f"  Filtered moves: {len(ordered_moves)}")
        
        # Check null move pruning
        if do_null_move and depth >= 3 and not board.is_check():
            print(f"  [NULL MOVE] Trying null move pruning...")
            board.push(chess.Move.null())
            null_score, _ = self._search(board, depth - 3, -beta, -beta + 1, ply + 1, False)
            null_score = -null_score
            board.pop()
            print(f"  [NULL MOVE] Result: {null_score}, beta={beta}")
            
            if null_score >= beta:
                print(f"  [EARLY EXIT] Null move pruning cutoff!")
                return beta, None
        
        print(f"  [LOOP] Starting iteration through {len(ordered_moves)} moves...")
        iteration_count = 0
    
    # Call original
    result = original_search(self, board, depth, alpha, beta, ply, do_null_move)
    
    if ply == 0 and depth >= 3:
        print(f"  [DEPTH {depth} COMPLETE] value={result[0]}, move={result[1]}")
    
    return result

# Patch it
V7P3REngine._search = debug_search

print("Testing depth 3+ with detailed loop debugging...\n")

engine = V7P3REngine(max_depth=10, tt_size_mb=256, tablebase_path="")

# Test position outside opening book
fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5"
board = chess.Board(fen)
engine.board = board

print(f"Position: {fen}\n")

best_move = engine.get_best_move(time_left=0)

print(f"\n\nFinal: {board.san(best_move) if best_move else 'None'}")
