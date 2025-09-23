#!/usr/bin/env python3
"""
V7P3R Chess Engine v11.1 - EMERGENCY PERFORMANCE PATCH
Simplified version focused on core functionality and reliable performance
Based on v11 with critical complexity removed

Author: Pat Snyder
"""

import time
import chess
import sys
import random
import json
import os
from typing import Optional, Tuple, List, Dict, Any
from collections import defaultdict

# Core evaluation components
from v7p3r_bitboard_evaluator import V7P3RScoringCalculationBitboard

# Simplified components for v11.1
from v7p3r_simple_time_manager import V7P3RSimpleTimeManager
from v7p3r_simple_move_orderer import V7P3RSimpleMoveOrderer


# Simplified data structures
class TranspositionEntry:
    def __init__(self, depth, score, best_move, node_type, zobrist_hash):
        self.depth = depth
        self.score = score
        self.best_move = best_move
        self.node_type = node_type
        self.zobrist_hash = zobrist_hash


class KillerMoves:
    def __init__(self):
        self.killers = defaultdict(list)
    
    def get_killers(self, depth):
        return self.killers.get(depth, [])
    
    def store_killer(self, move, depth):
        if move not in self.killers[depth]:
            self.killers[depth].insert(0, move)
            if len(self.killers[depth]) > 2:
                self.killers[depth].pop()


class HistoryHeuristic:
    def __init__(self):
        self.history = defaultdict(int)
    
    def get_history_score(self, move):
        return self.history.get(move.uci(), 0)
    
    def update_history(self, move, depth):
        self.history[move.uci()] += depth * depth


class SimpleZobrist:
    def __init__(self):
        self.piece_keys = {}
        self.castling_keys = {}
        self.en_passant_keys = {}
        self.turn_key = random.getrandbits(64)
        
        # Initialize random keys for all pieces and squares
        for square in range(64):
            for piece_type in range(1, 7):
                for color in [True, False]:
                    key = (square, piece_type, color)
                    self.piece_keys[key] = random.getrandbits(64)
    
    def hash_position(self, board):
        hash_value = 0
        
        # Hash pieces
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                key = (square, piece.piece_type, piece.color)
                if key in self.piece_keys:
                    hash_value ^= self.piece_keys[key]
        
        # Hash turn
        if not board.turn:
            hash_value ^= self.turn_key
            
        return hash_value


class V7P3REngineSimple:
    """V7P3R v11.1 - Simplified for Performance Recovery"""
    
    def __init__(self):
        # Basic piece values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300, 
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        # Search settings
        self.default_depth = 6
        self.nodes_searched = 0
        self.search_start_time = 0.0
        
        # Core evaluator (fast only for v11.1)
        self.fast_evaluator = V7P3RScoringCalculationBitboard(self.piece_values)
        
        # Simplified systems
        self.time_manager = V7P3RSimpleTimeManager()
        self.move_orderer = V7P3RSimpleMoveOrderer()
        
        # Basic search infrastructure
        self.evaluation_cache = {}
        self.transposition_table = {}
        self.max_tt_entries = 100000
        self.zobrist = SimpleZobrist()
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        
        # Stats
        self.search_stats = {
            'nodes_searched': 0,
            'cache_hits': 0,
            'tt_hits': 0,
            'evaluation_calls': 0
        }
    
    def search(self, board: chess.Board, time_limit: float = 3.0) -> chess.Move:
        """Simplified search with reliable time management"""
        self.nodes_searched = 0
        self.search_start_time = time.time()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        # Get time allocation
        allocated_time, target_depth = self.time_manager.calculate_time_allocation(board, time_limit)
        
        # Iterative deepening with simplified time management
        best_move = legal_moves[0]
        best_score = -99999
        
        for depth in range(1, min(target_depth + 1, self.default_depth + 1)):
            try:
                # Check time before starting iteration
                elapsed = time.time() - self.search_start_time
                if elapsed > allocated_time * 0.8:
                    break
                
                score, move = self._search_recursive(board, depth, -99999, 99999)
                
                if move and move != chess.Move.null():
                    best_move = move
                    best_score = score
                    
                    # Print search info
                    elapsed_ms = int(elapsed * 1000)
                    nps = int(self.nodes_searched / max(elapsed, 0.001))
                    print(f"info depth {depth} score cp {int(score)} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {move}")
                    sys.stdout.flush()
                
                # Stop if time is running out
                if elapsed > allocated_time * 0.7:
                    break
                    
            except Exception as e:
                print(f"info string Search error at depth {depth}: {e}")
                break
        
        # Update search stats
        self.search_stats['nodes_searched'] = self.nodes_searched
        
        return best_move
    
    def _search_recursive(self, board: chess.Board, depth: int, alpha: float, beta: float) -> Tuple[float, Optional[chess.Move]]:
        """Simplified alpha-beta search"""
        self.nodes_searched += 1
        
        # Time check every 1000 nodes
        if self.nodes_searched % 1000 == 0:
            elapsed = time.time() - self.search_start_time
            if elapsed > 10.0:  # Emergency timeout
                return self._evaluate_position(board), None
        
        # Terminal conditions
        if depth == 0:
            return self._quiescence_search(board, alpha, beta, 3), None
        
        if board.is_game_over():
            if board.is_checkmate():
                return -29000.0 + (self.default_depth - depth), None
            else:
                return 0.0, None
        
        # Transposition table probe
        tt_hit, tt_score, tt_move = self._probe_tt(board, depth, alpha, beta)
        if tt_hit:
            return tt_score, tt_move
        
        # Move generation and ordering
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0, None
        
        ordered_moves = self.move_orderer.order_moves(board, legal_moves, depth, tt_move)
        
        # Search moves
        best_score = -99999.0
        best_move = None
        original_alpha = alpha
        
        for move in ordered_moves:
            board.push(move)
            score, _ = self._search_recursive(board, depth - 1, -beta, -alpha)
            score = -score
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            if alpha >= beta:
                # Store killer move for non-captures
                if not board.is_capture(move):
                    self.move_orderer.store_killer_move(move, depth)
                break
        
        # Store in transposition table
        self._store_tt(board, depth, best_score, best_move, original_alpha, beta)
        
        return best_score, best_move
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """Simplified evaluation using fast evaluator only"""
        cache_key = board.fen()
        
        if cache_key in self.evaluation_cache:
            self.search_stats['cache_hits'] += 1
            return self.evaluation_cache[cache_key]
        
        # Use bitboard evaluation in v11.1
        try:
            # Use the correct method from V7P3RScoringCalculationBitboard
            white_score = self.fast_evaluator.calculate_score_optimized(board, chess.WHITE)
            black_score = self.fast_evaluator.calculate_score_optimized(board, chess.BLACK)
            
            if board.turn:  # White to move
                score = white_score - black_score
            else:  # Black to move
                score = black_score - white_score
        except Exception as e:
            print(f"Bitboard evaluation error: {e}")
            # Fallback to material count
            score = self._material_evaluation(board)
        
        self.evaluation_cache[cache_key] = score
        self.search_stats['evaluation_calls'] += 1
        
        return score
    
    def _material_evaluation(self, board: chess.Board) -> float:
        """Simple material evaluation fallback"""
        score = 0.0
        
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                piece_value = self.piece_values.get(piece.piece_type, 0)
                if piece.color:  # White
                    score += piece_value
                else:  # Black
                    score -= piece_value
        
        return score
    
    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int) -> float:
        """Simple quiescence search for tactical stability"""
        self.nodes_searched += 1
        
        stand_pat = self._evaluate_position(board)
        
        if stand_pat >= beta:
            return beta
        
        alpha = max(alpha, stand_pat)
        
        if depth <= 0:
            return stand_pat
        
        # Search captures only
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        
        for move in captures:
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha, depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            
            alpha = max(alpha, score)
        
        return alpha
    
    def _probe_tt(self, board: chess.Board, depth: int, alpha: float, beta: float) -> Tuple[bool, float, Optional[chess.Move]]:
        """Simple transposition table probe"""
        zobrist_hash = self.zobrist.hash_position(board)
        
        if zobrist_hash in self.transposition_table:
            entry = self.transposition_table[zobrist_hash]
            if entry.depth >= depth:
                self.search_stats['tt_hits'] += 1
                if entry.node_type == 'exact':
                    return True, float(entry.score), entry.best_move
                elif entry.node_type == 'lowerbound' and entry.score >= beta:
                    return True, float(entry.score), entry.best_move
                elif entry.node_type == 'upperbound' and entry.score <= alpha:
                    return True, float(entry.score), entry.best_move
        
        return False, 0.0, None
    
    def _store_tt(self, board: chess.Board, depth: int, score: float, best_move: Optional[chess.Move], alpha: float, beta: float):
        """Simple transposition table storage"""
        zobrist_hash = self.zobrist.hash_position(board)
        
        # Determine node type
        if score <= alpha:
            node_type = 'upperbound'
        elif score >= beta:
            node_type = 'lowerbound'
        else:
            node_type = 'exact'
        
        # Simple replacement (clear when full)
        if len(self.transposition_table) >= self.max_tt_entries:
            self.transposition_table.clear()
        
        entry = TranspositionEntry(depth, int(score), best_move, node_type, zobrist_hash)
        self.transposition_table[zobrist_hash] = entry
    
    def new_game(self):
        """Reset for new game"""
        self.evaluation_cache.clear()
        self.transposition_table.clear()
        self.move_orderer.clear_killers()
        self.nodes_searched = 0
        
        for key in self.search_stats:
            self.search_stats[key] = 0


# Export the simplified engine for testing
V7P3REngine = V7P3REngineSimple