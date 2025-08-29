#!/usr/bin/env python3
"""
V7P3R Chess Engine v8.1 - "Consistency First"
Unified Architecture with Deterministic Evaluation
Rolls back async evaluation while preserving architectural improvements
Author: Pat Snyder
"""

import chess
import chess.engine
import time
import sys
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from v7p3r_scoring_calculation import V7P3RScoringCalculationClean


@dataclass
class SearchOptions:
    """Configuration options for the unified search"""
    return_pv: bool = True
    use_killer_moves: bool = True
    use_history_heuristic: bool = True
    use_late_move_reduction: bool = True
    use_null_move_pruning: bool = False


class V7P3REngineV81:
    """V8.1 - Unified Architecture with Consistent Deterministic Evaluation"""
    
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
        
        # Unified search optimizations (kept from V8.0)
        self.killer_moves = {}  # killer_moves[ply] = [move1, move2]
        self.history_scores = {}  # history_scores[move_key] = score
        
        # Simple evaluation cache (deterministic, no async issues)
        self.evaluation_cache = {}  # position_hash -> evaluation
        
        # Performance monitoring (simplified)
        self.search_stats = {
            'nodes_per_second': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
        
        # Transposition table for opening guidance only
        self.transposition_table: Dict[str, Dict[str, Any]] = {}
        self._inject_opening_knowledge()
    
    def search(self, board: chess.Board, time_limit: float = 3.0) -> chess.Move:
        """Main search entry point with consistent evaluation"""
        print("info string Starting search...", flush=True)
        sys.stdout.flush()
        
        self.nodes_searched = 0
        start_time = time.time()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
            
        # Enhanced time management (kept from V8.0)
        target_time = min(time_limit * 0.6, 8.0)
        max_time = min(time_limit * 0.8, 12.0)
        
        # Configure search options based on available time
        search_options = self._configure_search_options(time_limit)
        
        # Unified iterative deepening (V8.0 architecture, V8.1 consistency)
        best_move = legal_moves[0]
        best_pv = [best_move]
        depth = 1
        
        while depth <= self.default_depth:
            iteration_start = time.time()
            try:
                move, score, pv = self._unified_search_root(board, depth, search_options)
                iteration_time = time.time() - iteration_start
                
                if move:
                    best_move = move
                    best_pv = pv
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    nps = int(self.nodes_searched / max(elapsed_ms / 1000, 0.001))
                    pv_str = " ".join(str(m) for m in pv[:depth])
                    
                    # UCI score from side-to-move perspective (fixed in V8.0)
                    score_str = self._format_uci_score(score, depth)
                    
                    print(f"info depth {depth} score {score_str} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {pv_str}")
                    sys.stdout.flush()
                
                # Time management (kept from V8.0, removed confidence system)
                elapsed = time.time() - start_time
                
                if iteration_time > target_time * 0.4 or elapsed > target_time:
                    break
                    
                if elapsed + (iteration_time * 2.5) > max_time:
                    break
                    
                depth += 1
            except:
                break
                
        return best_move
    
    def _unified_search_root(self, board: chess.Board, depth: int, options: SearchOptions) -> Tuple[Optional[chess.Move], float, list]:
        """Unified root search with all enhancements"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0.0, []
            
        # Enhanced move ordering (kept from V8.0)
        legal_moves = self._order_moves_enhanced(board, legal_moves, 0, options)
        
        best_move = legal_moves[0]
        best_score = -99999.0
        best_pv = [best_move]
        alpha = -99999.0
        beta = 99999.0
        
        for move in legal_moves:
            board.push(move)
            score, pv = self._unified_negamax(board, depth - 1, -beta, -alpha, 1, options)
            score = -score
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
                best_pv = [move] + pv if options.return_pv else [move]
                alpha = max(alpha, score)
        
        return best_move, best_score, best_pv
    
    def _unified_negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, 
                        ply: int, options: SearchOptions) -> Tuple[float, list]:
        """
        Unified negamax function with deterministic evaluation.
        V8.1: Removed async evaluation, restored consistent scoring.
        """
        self.nodes_searched += 1
        
        # Terminal depth - use deterministic evaluation
        if depth == 0:
            evaluation = self._evaluate_position_deterministic(board)
            pv = [] if options.return_pv else []
            return evaluation, pv
            
        # Terminal positions
        if board.is_game_over():
            if board.is_checkmate():
                return -29000.0 + ply, []
            return 0.0, []  # Stalemate
        
        # Null move pruning (if enabled and conditions are met)
        if (options.use_null_move_pruning and depth > 2 and 
            not board.is_check() and self._has_non_pawn_material(board)):
            # TODO: Implement null move pruning
            pass
        
        # Get and order legal moves
        legal_moves = list(board.legal_moves)
        legal_moves = self._order_moves_enhanced(board, legal_moves, ply, options)
        
        best_score = -99999.0
        best_pv = []
        
        for i, move in enumerate(legal_moves):
            # Late move reduction (if enabled)
            reduced_depth = depth - 1
            if (options.use_late_move_reduction and depth > 2 and i > 3 and 
                not board.is_capture(move) and not board.is_check()):
                reduced_depth = max(0, depth - 2)
            
            board.push(move)
            score, pv = self._unified_negamax(board, reduced_depth, -beta, -alpha, ply + 1, options)
            score = -score
            board.pop()
            
            if score > best_score:
                best_score = score
                if options.return_pv:
                    best_pv = [move] + pv
            
            alpha = max(alpha, score)
            
            if alpha >= beta:
                # Alpha-beta cutoff
                if options.use_killer_moves:
                    self._store_killer_move(move, ply)
                if options.use_history_heuristic:
                    self._update_history_score(move, depth)
                break
        
        return best_score, best_pv
    
    def _evaluate_position_deterministic(self, board: chess.Board) -> float:
        """
        V8.1: Deterministic evaluation - always returns the same score for the same position.
        Restored original V7.2 logic with V8.0 caching benefits.
        """
        position_hash = str(board.board_fen())
        cache_key = f"{position_hash}_{board.turn}"
        
        # Check evaluation cache first
        if cache_key in self.evaluation_cache:
            self.search_stats['cache_hits'] += 1
            return self.evaluation_cache[cache_key]
        
        self.search_stats['cache_misses'] += 1
        
        # Use the original, proven evaluation logic (from V7.2)
        white_score = self.scoring_calculator.calculate_score_optimized(board, chess.WHITE)
        black_score = self.scoring_calculator.calculate_score_optimized(board, chess.BLACK)
        
        # Return from current side-to-move perspective (negamax requirement)
        if board.turn == chess.WHITE:
            final_score = white_score - black_score
        else:
            final_score = black_score - white_score
        
        # Cache the result - deterministic, no async corruption
        self.evaluation_cache[cache_key] = final_score
        
        return final_score
    
    def _configure_search_options(self, time_limit: float) -> SearchOptions:
        """Configure search options based on available time (simplified from V8.0)"""
        if time_limit > 5.0:
            # Plenty of time - use all features
            return SearchOptions(
                return_pv=True,
                use_killer_moves=True,
                use_history_heuristic=True,
                use_late_move_reduction=True
            )
        elif time_limit > 1.0:
            # Moderate time - reduce some overhead
            return SearchOptions(
                return_pv=True,
                use_killer_moves=True,
                use_history_heuristic=False,
                use_late_move_reduction=True
            )
        else:
            # Time pressure - minimize overhead
            return SearchOptions(
                return_pv=False,
                use_killer_moves=False,
                use_history_heuristic=False,
                use_late_move_reduction=False
            )
    
    def _has_non_pawn_material(self, board: chess.Board) -> bool:
        """Check if the current player has non-pawn material"""
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            if board.pieces(piece_type, board.turn):
                return True
        return False
    
    # Enhanced move ordering (kept from V8.0 - this is deterministic and beneficial)
    def _order_moves_enhanced(self, board: chess.Board, moves: List[chess.Move], 
                            ply: int, options: SearchOptions) -> List[chess.Move]:
        """Enhanced move ordering with all heuristics (V8.0 preserved)"""
        if len(moves) <= 1:
            return moves
            
        scored_moves = []
        
        for move in moves:
            score = self._score_move_enhanced(board, move, ply, options)
            scored_moves.append((move, score))
            
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in scored_moves]
    
    def _score_move_enhanced(self, board: chess.Board, move: chess.Move, 
                           ply: int, options: SearchOptions) -> float:
        """Enhanced move scoring with all heuristics (V8.0 preserved)"""
        score = 0.0
        
        # 1. Captures (MVV-LVA) - highest priority
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                victim_value = self.piece_values.get(victim.piece_type, 0)
                attacker_value = self.piece_values.get(attacker.piece_type, 0)
                score += 100000 + (victim_value * 10) - attacker_value
        
        # 2. Promotions
        if move.promotion:
            score += 90000 + self.piece_values.get(move.promotion, 0)
        
        # 3. Killer moves (if enabled)
        if options.use_killer_moves and ply in self.killer_moves:
            if move in self.killer_moves[ply]:
                score += 80000 - self.killer_moves[ply].index(move) * 1000
        
        # 4. Checks
        board.push(move)
        if board.is_check():
            score += 5000
        board.pop()
        
        # 5. History heuristic (if enabled)
        if options.use_history_heuristic:
            move_key = f"{move.from_square}_{move.to_square}"
            if move_key in self.history_scores:
                score += min(self.history_scores[move_key], 4000)
        
        # 6. Positional bonuses
        if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
            score += 100
            
        return score
    
    # Helper methods (preserved from V8.0 but simplified)
    def new_game(self):
        """Reset search tables for a new game"""
        self.killer_moves.clear()
        self.history_scores.clear()
        self.evaluation_cache.clear()
        self.nodes_searched = 0
        
        # Reset performance stats
        self.search_stats = {
            'nodes_per_second': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
    
    def _store_killer_move(self, move: chess.Move, ply: int):
        """Store a killer move that caused a cutoff"""
        if ply not in self.killer_moves:
            self.killer_moves[ply] = []
        
        if move in self.killer_moves[ply]:
            self.killer_moves[ply].remove(move)
        
        self.killer_moves[ply].insert(0, move)
        
        if len(self.killer_moves[ply]) > 2:
            self.killer_moves[ply] = self.killer_moves[ply][:2]
    
    def _update_history_score(self, move: chess.Move, depth: int):
        """Update history heuristic score for a move"""
        move_key = f"{move.from_square}_{move.to_square}"
        bonus = depth * depth
        
        if move_key in self.history_scores:
            self.history_scores[move_key] += bonus
        else:
            self.history_scores[move_key] = bonus
        
        if self.history_scores[move_key] > 10000:
            self.history_scores[move_key] = 10000
    
    def _format_uci_score(self, score: float, search_depth: int) -> str:
        """Format score for UCI output - properly handle mate scores"""
        if abs(score) >= 28500:
            if score > 0:
                if score >= 29000:
                    depth_to_mate = 29000 - score + search_depth
                    mate_in = max(1, int((depth_to_mate + 1) / 2))
                else:
                    mate_in = max(1, int((29000 - score) / 2))
                return f"mate {mate_in}"
            else:
                if score <= -29000:
                    depth_to_mate = abs(score) - 29000 + search_depth
                    mate_in = max(1, int((depth_to_mate + 1) / 2))
                else:
                    mate_in = max(1, int((abs(score) - 29000) / 2))
                return f"mate -{mate_in}"
        else:
            return f"cp {int(score * 100)}"
    
    def _inject_opening_knowledge(self):
        """Inject basic opening knowledge into transposition table"""
        # TODO: Implement opening book or opening principles
        pass
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get current search statistics"""
        return self.search_stats.copy()


# Backward compatibility alias
V7P3RCleanEngine = V7P3REngineV81
