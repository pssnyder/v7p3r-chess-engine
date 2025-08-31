#!/usr/bin/env python3
"""
V7P3R Chess Engine v10.0 - Unified Search Architecture
Clean engine with ONE unified search function containing all advanced features
Author: Pat Snyder
"""

import time
import chess
import sys
import random
from typing import Optional, Tuple, List, Dict
from collections import defaultdict
from v7p3r_scoring_calculation import V7P3RScoringCalculationClean


class TranspositionEntry:
    """Entry in the transposition table"""
    def __init__(self, depth: int, score: int, best_move: Optional[chess.Move], 
                 node_type: str, zobrist_hash: int):
        self.depth = depth
        self.score = score
        self.best_move = best_move
        self.node_type = node_type  # 'exact', 'lowerbound', 'upperbound'
        self.zobrist_hash = zobrist_hash


class KillerMoves:
    """Killer move storage - 2 killer moves per depth"""
    def __init__(self):
        self.killers: Dict[int, List[chess.Move]] = defaultdict(list)
        
    def store_killer(self, move: chess.Move, depth: int):
        """Store a killer move at a specific depth"""
        if depth not in self.killers:
            self.killers[depth] = []
        
        # Don't store duplicates
        if move in self.killers[depth]:
            return
            
        # Add new killer, keep only 2 most recent
        self.killers[depth].insert(0, move)
        if len(self.killers[depth]) > 2:
            self.killers[depth] = self.killers[depth][:2]
    
    def get_killers(self, depth: int) -> List[chess.Move]:
        """Get killer moves for a specific depth"""
        return self.killers.get(depth, [])
    
    def is_killer(self, move: chess.Move, depth: int) -> bool:
        """Check if move is a killer move at this depth"""
        return move in self.killers.get(depth, [])


class HistoryHeuristic:
    """History heuristic for move ordering"""
    def __init__(self):
        self.history: Dict[Tuple[int, int], int] = defaultdict(int)
        self.max_history = 1000  # Prevent overflow
        
    def update_history(self, move: chess.Move, depth: int):
        """Update history score for a move that caused a cutoff"""
        key = (move.from_square, move.to_square)
        self.history[key] += depth * depth  # Deeper searches get higher scores
        
        # Prevent overflow
        if self.history[key] > self.max_history:
            self._scale_down_history()
    
    def get_history_score(self, move: chess.Move) -> int:
        """Get history score for a move"""
        key = (move.from_square, move.to_square)
        return self.history.get(key, 0)
    
    def _scale_down_history(self):
        """Scale down all history scores to prevent overflow"""
        for key in self.history:
            self.history[key] //= 2


class ZobristHashing:
    """Simple Zobrist hashing for transposition table"""
    def __init__(self):
        # Initialize random numbers for Zobrist hashing
        random.seed(12345)  # Fixed seed for reproducible results
        self.piece_keys = {}
        
        # Generate random keys for each piece on each square
        for square in range(64):
            for piece_type in range(1, 7):  # Pawn=1 to King=6
                for color in [True, False]:  # White/Black
                    key = random.getrandbits(64)
                    self.piece_keys[(square, piece_type, color)] = key
        
        # Additional keys
        self.black_to_move = random.getrandbits(64)
        self.castling_rights = [random.getrandbits(64) for _ in range(16)]
        self.en_passant_files = [random.getrandbits(64) for _ in range(8)]
    
    def hash_position(self, board: chess.Board) -> int:
        """Calculate Zobrist hash for a position"""
        hash_value = 0
        
        # Hash pieces
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                key = (square, piece.piece_type, piece.color)
                if key in self.piece_keys:
                    hash_value ^= self.piece_keys[key]
        
        # Hash side to move
        if not board.turn:  # Black to move
            hash_value ^= self.black_to_move
        
        # Hash castling rights (simplified)
        if board.has_kingside_castling_rights(chess.WHITE):
            hash_value ^= self.castling_rights[0]
        if board.has_queenside_castling_rights(chess.WHITE):
            hash_value ^= self.castling_rights[1]
        if board.has_kingside_castling_rights(chess.BLACK):
            hash_value ^= self.castling_rights[2]
        if board.has_queenside_castling_rights(chess.BLACK):
            hash_value ^= self.castling_rights[3]
        
        # Hash en passant (simplified)
        if board.ep_square is not None:
            file_idx = chess.square_file(board.ep_square)
            hash_value ^= self.en_passant_files[file_idx]
        
        return hash_value


class V7P3RCleanEngine:
    """V10.0 - Unified Search Architecture with ALL Advanced Features"""
    
    def __init__(self):
        # Basic configuration
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300, 
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0  # King safety handled separately
        }
        
        # Search configuration
        self.default_depth = 6
        self.nodes_searched = 0
        
        # Evaluation components
        self.scoring_calculator = V7P3RScoringCalculationClean(self.piece_values)
        
        # Simple evaluation cache for speed
        self.evaluation_cache = {}  # position_hash -> evaluation
        
        # Advanced search infrastructure
        self.transposition_table: Dict[int, TranspositionEntry] = {}
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        self.zobrist = ZobristHashing()
        
        # Configuration
        self.max_tt_entries = 50000  # Reasonable size for testing
        
        # Performance monitoring
        self.search_stats = {
            'nodes_per_second': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'tt_hits': 0,
            'tt_stores': 0,
            'killer_hits': 0,
        }
    
    def search(self, board: chess.Board, time_limit: float = 3.0) -> chess.Move:
        """
        UNIFIED SEARCH - The ONE search function with ALL advanced features:
        - Iterative deepening
        - Transposition table with Zobrist hashing
        - Killer moves (2 per depth)
        - History heuristic
        - Advanced move ordering
        - Alpha-beta pruning
        - Proper time management
        """
        print("info string Starting search...", flush=True)
        sys.stdout.flush()
        
        self.nodes_searched = 0
        start_time = time.time()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        # Iterative deepening
        best_move = legal_moves[0]
        target_time = min(time_limit * 0.8, 10.0)
        
        for depth in range(1, self.default_depth + 1):
            iteration_start = time.time()
            
            try:
                score, move = self._unified_search(board, depth, -99999, 99999)
                
                if move:
                    best_move = move
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    nps = int(self.nodes_searched / max(elapsed_ms / 1000, 0.001))
                    
                    # Include TT stats in output
                    print(f"info depth {depth} score cp {int(score)} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {move}")
                    sys.stdout.flush()
                
                # Time management
                elapsed = time.time() - start_time
                iteration_time = time.time() - iteration_start
                
                if elapsed > target_time or iteration_time > time_limit * 0.3:
                    break
                    
            except:
                break
                
        return best_move
    
    def _unified_search(self, board: chess.Board, depth: int, alpha: float, beta: float) -> Tuple[float, Optional[chess.Move]]:
        """
        THE UNIFIED SEARCH FUNCTION - Contains ALL advanced features:
        - Alpha-beta pruning (negamax framework)
        - Transposition table with proper bounds
        - Killer moves (non-captures that cause beta cutoffs)
        - History heuristic (move success tracking)
        - Advanced move ordering (TT move > captures > killers > history > quiet)
        - Proper mate scoring
        """
        self.nodes_searched += 1
        
        # 1. TRANSPOSITION TABLE PROBE
        tt_hit, tt_score, tt_move = self._probe_transposition_table(board, depth, int(alpha), int(beta))
        if tt_hit:
            return float(tt_score), tt_move
        
        # 2. TERMINAL CONDITIONS
        if depth == 0:
            score = self._evaluate_position(board)
            self._store_transposition_table(board, depth, int(score), None, int(alpha), int(beta))
            return score, None
            
        if board.is_game_over():
            if board.is_checkmate():
                score = -29000.0 + (self.default_depth - depth)  # Prefer quicker mates
            else:
                score = 0.0  # Stalemate
            self._store_transposition_table(board, depth, int(score), None, int(alpha), int(beta))
            return score, None
        
        # 3. MOVE GENERATION AND ORDERING
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0, None
        
        ordered_moves = self._order_moves_advanced(board, legal_moves, depth, tt_move)
        
        # 4. MAIN SEARCH LOOP (NEGAMAX WITH ALPHA-BETA)
        best_score = -99999.0
        best_move = ordered_moves[0]
        original_alpha = alpha
        
        for move in ordered_moves:
            board.push(move)
            score, _ = self._unified_search(board, depth - 1, -beta, -alpha)
            score = -score  # Negamax
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            if alpha >= beta:
                # 5. BETA CUTOFF - UPDATE HEURISTICS
                if not board.is_capture(move):
                    self.killer_moves.store_killer(move, depth)
                    self.history_heuristic.update_history(move, depth)
                break
        
        # 6. TRANSPOSITION TABLE STORE
        self._store_transposition_table(board, depth, int(best_score), best_move, int(original_alpha), int(beta))
        
        return best_score, best_move
    
    def _order_moves_advanced(self, board: chess.Board, moves: List[chess.Move], depth: int, 
                              tt_move: Optional[chess.Move] = None) -> List[chess.Move]:
        """Advanced move ordering with ALL heuristics"""
        if len(moves) <= 2:
            return moves
        
        move_scores = []
        
        for move in moves:
            score = 0
            
            # 1. Transposition table move (highest priority)
            if tt_move and move == tt_move:
                score += 10000
            
            # 2. Captures (MVV-LVA simplified)
            elif board.is_capture(move):
                victim = board.piece_at(move.to_square)
                if victim:
                    score += 5000 + self.piece_values.get(victim.piece_type, 0)
            
            # 3. Killer moves
            elif self.killer_moves.is_killer(move, depth):
                score += 1000
                self.search_stats['killer_hits'] += 1
            
            # 4. History heuristic
            else:
                score += self.history_heuristic.get_history_score(move)
            
            move_scores.append((score, move))
        
        # Sort by score (highest first)
        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in move_scores]
    
    def _probe_transposition_table(self, board: chess.Board, depth: int, alpha: int, beta: int) -> Tuple[bool, int, Optional[chess.Move]]:
        """Probe transposition table for this position"""
        zobrist_hash = self.zobrist.hash_position(board)
        
        if zobrist_hash in self.transposition_table:
            entry = self.transposition_table[zobrist_hash]
            
            # Only use if searched to sufficient depth
            if entry.depth >= depth:
                self.search_stats['tt_hits'] += 1
                
                # Check if we can use the score
                if entry.node_type == 'exact':
                    return True, entry.score, entry.best_move
                elif entry.node_type == 'lowerbound' and entry.score >= beta:
                    return True, entry.score, entry.best_move
                elif entry.node_type == 'upperbound' and entry.score <= alpha:
                    return True, entry.score, entry.best_move
        
        return False, 0, None
    
    def _store_transposition_table(self, board: chess.Board, depth: int, score: int, 
                                   best_move: Optional[chess.Move], alpha: int, beta: int):
        """Store result in transposition table"""
        zobrist_hash = self.zobrist.hash_position(board)
        
        # Determine node type
        if score <= alpha:
            node_type = 'upperbound'
        elif score >= beta:
            node_type = 'lowerbound'
        else:
            node_type = 'exact'
        
        # Clear table if too large (simple replacement strategy)
        if len(self.transposition_table) >= self.max_tt_entries:
            self.transposition_table.clear()
        
        entry = TranspositionEntry(depth, score, best_move, node_type, zobrist_hash)
        self.transposition_table[zobrist_hash] = entry
        self.search_stats['tt_stores'] += 1
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """Position evaluation with caching"""
        # Create cache key
        cache_key = board.fen()
        
        if cache_key in self.evaluation_cache:
            self.search_stats['cache_hits'] += 1
            return self.evaluation_cache[cache_key]
        
        self.search_stats['cache_misses'] += 1
        
        # Use V7.0-style scoring calculation
        white_score = self.scoring_calculator.calculate_score_optimized(board, True)
        black_score = self.scoring_calculator.calculate_score_optimized(board, False)
        
        if board.turn:  # White to move
            final_score = white_score - black_score
        else:  # Black to move
            final_score = black_score - white_score
        
        # Cache the result
        self.evaluation_cache[cache_key] = final_score
        
        return final_score
    
    def new_game(self):
        """Reset for a new game"""
        self.evaluation_cache.clear()
        self.transposition_table.clear()
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        self.nodes_searched = 0
        self.search_stats = {
            'nodes_per_second': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'tt_hits': 0,
            'tt_stores': 0,
            'killer_hits': 0,
        }


# Backward compatibility alias
V7P3REngineV81 = V7P3RCleanEngine
