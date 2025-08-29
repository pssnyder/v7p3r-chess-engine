#!/usr/bin/env python3
"""
V7P3R Chess Engine v8.0 - "Sports Car to Supercar"
Unified Search Architecture with Progressive Asynchronous Evaluation
Author: Pat Snyder
"""

import chess
import chess.engine
import time
import sys
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Optional, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
from v7p3r_scoring_calculation import V7P3RScoringCalculationClean


class EvaluationPriority(Enum):
    """Evaluation priority levels for progressive evaluation system"""
    CRITICAL = 1    # Must complete before move selection
    PRIMARY = 2     # High priority, asynchronous
    SECONDARY = 3   # Background evaluation, nice-to-have


@dataclass
class EvaluationTask:
    """Represents an evaluation task for asynchronous processing"""
    name: str
    priority: EvaluationPriority
    weight: float
    evaluator: Callable
    timeout_ms: int
    default_value: float = 0.0


@dataclass
class EvaluationResult:
    """Result from an evaluation task"""
    name: str
    value: float
    completed: bool
    execution_time_ms: int


@dataclass
class SearchOptions:
    """Configuration options for the unified search"""
    return_pv: bool = True
    use_killer_moves: bool = True
    use_history_heuristic: bool = True
    use_late_move_reduction: bool = True
    use_null_move_pruning: bool = False
    confidence_threshold: float = 0.75
    max_evaluation_threads: int = 4


class V7P3REngineV8:
    """V8.0 - Unified Architecture with Progressive Asynchronous Evaluation"""
    
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
        self.root_color = chess.WHITE
        
        # Evaluation components
        self.scoring_calculator = V7P3RScoringCalculationClean(self.piece_values)
        
        # Unified search optimizations
        self.killer_moves = {}  # killer_moves[ply] = [move1, move2]
        self.history_scores = {}  # history_scores[move_key] = score
        self.evaluation_cache = {}  # position_hash -> evaluation
        
        # Progressive evaluation system
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.confidence_threshold = 0.75  # UCI configurable strength
        
        # Performance monitoring
        self.search_stats = {
            'nodes_per_second': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'evaluation_timeouts': 0,
            'confidence_exits': 0
        }
        
        # Transposition table for opening guidance only
        self.transposition_table: Dict[str, Dict[str, Any]] = {}
        self._inject_opening_knowledge()
    
    def search(self, board: chess.Board, time_limit: float = 3.0) -> chess.Move:
        """Main search entry point with enhanced time management"""
        print("info string Starting search...", flush=True)
        sys.stdout.flush()
        
        self.nodes_searched = 0
        start_time = time.time()
        self.root_color = board.turn
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
            
        # Enhanced time management
        target_time = min(time_limit * 0.6, 8.0)
        max_time = min(time_limit * 0.8, 12.0)
        
        # Configure search options based on available time
        search_options = self._configure_search_options(time_limit)
        
        # Unified iterative deepening with progressive evaluation
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
                    
                    # UCI score is already from side-to-move perspective
                    score_str = self._format_uci_score(score, depth)
                    
                    print(f"info depth {depth} score {score_str} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {pv_str}")
                    sys.stdout.flush()
                
                # Enhanced time management with confidence consideration
                elapsed = time.time() - start_time
                
                # Early exit if high confidence and reasonable depth
                if depth >= 4 and self._has_high_confidence(score, depth):
                    self.search_stats['confidence_exits'] += 1
                    break
                
                # Time-based stopping conditions
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
            
        # Enhanced move ordering
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
        Unified negamax function with all heuristics and progressive evaluation.
        Replaces both _negamax() and _negamax_with_pv().
        """
        self.nodes_searched += 1
        
        # Terminal depth - use progressive evaluation
        if depth == 0:
            evaluation = self._progressive_evaluate(board, options)
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
        moves_searched = 0
        
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
            
            moves_searched += 1
            
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
    
    def _progressive_evaluate(self, board: chess.Board, options: SearchOptions) -> float:
        """
        Progressive asynchronous evaluation system.
        FIXED: Properly handles perspective by calculating for both sides and taking difference.
        """
        position_hash = str(board.board_fen())
        
        # Check evaluation cache first
        cache_key = f"{position_hash}_{board.turn}"  # Include turn in cache key
        if cache_key in self.evaluation_cache:
            self.search_stats['cache_hits'] += 1
            return self.evaluation_cache[cache_key]
        
        self.search_stats['cache_misses'] += 1
        
        # Use the original evaluation logic that properly handles perspective
        white_score = self.scoring_calculator.calculate_score_optimized(board, chess.WHITE)
        black_score = self.scoring_calculator.calculate_score_optimized(board, chess.BLACK)
        
        # Return from current side-to-move perspective (negamax requirement)
        if board.turn == chess.WHITE:
            final_score = white_score - black_score
        else:
            final_score = black_score - white_score
        
        # Cache the result with turn-specific key
        self.evaluation_cache[cache_key] = final_score
        
        return final_score
    
    def _create_evaluation_tasks(self, board: chess.Board) -> List[EvaluationTask]:
        """Create evaluation tasks based on position characteristics"""
        tasks = [
            # Critical evaluations (must complete)
            EvaluationTask("material", EvaluationPriority.CRITICAL, 0.4, 
                          self._evaluate_material, 50, 0.0),
            EvaluationTask("king_safety", EvaluationPriority.CRITICAL, 0.3, 
                          self._evaluate_king_safety, 50, 0.0),
            EvaluationTask("immediate_threats", EvaluationPriority.CRITICAL, 0.3, 
                          self._evaluate_immediate_threats, 50, 0.0),
            
            # Primary evaluations (high priority async)
            EvaluationTask("piece_activity", EvaluationPriority.PRIMARY, 0.2, 
                          self._evaluate_piece_activity, 20, 0.0),
            EvaluationTask("center_control", EvaluationPriority.PRIMARY, 0.15, 
                          self._evaluate_center_control, 20, 0.0),
            
            # Secondary evaluations (background)
            EvaluationTask("pawn_structure", EvaluationPriority.SECONDARY, 0.1, 
                          self._evaluate_pawn_structure, 10, 0.0),
            EvaluationTask("endgame_factors", EvaluationPriority.SECONDARY, 0.1, 
                          self._evaluate_endgame_factors, 10, 0.0),
        ]
        
        return tasks
    
    def _safe_evaluate(self, task: EvaluationTask, board: chess.Board, perspective: chess.Color) -> float:
        """Safely execute an evaluation task with error handling"""
        try:
            return task.evaluator(board, perspective)
        except:
            return task.default_value
    
    # Evaluation functions for progressive system
    def _evaluate_material(self, board: chess.Board, perspective: chess.Color) -> float:
        """Critical: Material balance evaluation"""
        white_material = 0
        black_material = 0
        
        for piece_type, value in self.piece_values.items():
            if piece_type != chess.KING:
                white_material += len(board.pieces(piece_type, chess.WHITE)) * value
                black_material += len(board.pieces(piece_type, chess.BLACK)) * value
        
        if perspective == chess.WHITE:
            return white_material - black_material
        else:
            return black_material - white_material
    
    def _evaluate_king_safety(self, board: chess.Board, perspective: chess.Color) -> float:
        """Critical: King safety evaluation"""
        king_square = board.king(perspective)
        if not king_square:
            return -1000.0
        
        safety_score = 0.0
        
        # Basic king safety checks
        if self._is_king_exposed(board, perspective, king_square):
            safety_score -= 50.0
        
        # Castling bonus
        if perspective == chess.WHITE and king_square in [chess.G1, chess.C1]:
            safety_score += 20.0
        elif perspective == chess.BLACK and king_square in [chess.G8, chess.C8]:
            safety_score += 20.0
        
        return safety_score
    
    def _evaluate_immediate_threats(self, board: chess.Board, perspective: chess.Color) -> float:
        """Critical: Immediate tactical threats"""
        threat_score = 0.0
        
        # Check for checks
        if board.is_check():
            if board.turn == perspective:
                threat_score -= 25.0  # We're in check
            else:
                threat_score += 25.0  # We're giving check
        
        # Check for immediate captures
        for move in board.legal_moves:
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                if victim:
                    value = self.piece_values.get(victim.piece_type, 0)
                    if board.turn == perspective:
                        threat_score += value * 0.1  # We can capture
                    else:
                        threat_score -= value * 0.1  # We can be captured
        
        return threat_score
    
    def _evaluate_piece_activity(self, board: chess.Board, perspective: chess.Color) -> float:
        """Primary: Piece development and activity"""
        return self.scoring_calculator._piece_development(board, perspective)
    
    def _evaluate_center_control(self, board: chess.Board, perspective: chess.Color) -> float:
        """Primary: Center control evaluation"""
        return self.scoring_calculator._center_control(board, perspective)
    
    def _evaluate_pawn_structure(self, board: chess.Board, perspective: chess.Color) -> float:
        """Secondary: Pawn structure analysis"""
        # TODO: Implement detailed pawn structure evaluation
        return 0.0
    
    def _evaluate_endgame_factors(self, board: chess.Board, perspective: chess.Color) -> float:
        """Secondary: Endgame-specific evaluation"""
        if self.scoring_calculator._is_endgame(board):
            return self.scoring_calculator._endgame_logic(board, perspective)
        return 0.0
    
    def _configure_search_options(self, time_limit: float) -> SearchOptions:
        """Configure search options based on available time"""
        if time_limit > 5.0:
            # Plenty of time - use all features
            return SearchOptions(
                return_pv=True,
                use_killer_moves=True,
                use_history_heuristic=True,
                use_late_move_reduction=True,
                confidence_threshold=self.confidence_threshold,
                max_evaluation_threads=4
            )
        elif time_limit > 1.0:
            # Moderate time - reduce some overhead
            return SearchOptions(
                return_pv=True,
                use_killer_moves=True,
                use_history_heuristic=False,
                use_late_move_reduction=True,
                confidence_threshold=max(0.6, self.confidence_threshold - 0.1),
                max_evaluation_threads=2
            )
        else:
            # Time pressure - minimize overhead
            return SearchOptions(
                return_pv=False,
                use_killer_moves=False,
                use_history_heuristic=False,
                use_late_move_reduction=False,
                confidence_threshold=0.5,
                max_evaluation_threads=1
            )
    
    def _has_high_confidence(self, score: float, depth: int) -> bool:
        """Determine if we have high confidence in the current evaluation"""
        # TODO: Implement confidence calculation based on score stability,
        # evaluation completeness, and search depth
        return depth >= 6 and abs(score) > 100  # Simple heuristic for now
    
    def _has_non_pawn_material(self, board: chess.Board) -> bool:
        """Check if the current player has non-pawn material"""
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            if board.pieces(piece_type, board.turn):
                return True
        return False
    
    def _is_king_exposed(self, board: chess.Board, color: chess.Color, king_square: int) -> bool:
        """Check if king is dangerously exposed"""
        if color == chess.WHITE:
            return chess.square_rank(king_square) > 2
        else:
            return chess.square_rank(king_square) < 5
    
    # Enhanced move ordering
    def _order_moves_enhanced(self, board: chess.Board, moves: List[chess.Move], 
                            ply: int, options: SearchOptions) -> List[chess.Move]:
        """Enhanced move ordering with all heuristics"""
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
        """Enhanced move scoring with all heuristics"""
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
    
    # Existing helper methods (preserved for compatibility)
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
            'evaluation_timeouts': 0,
            'confidence_exits': 0
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
    
    def set_strength(self, strength_percent: int):
        """Set engine strength (UCI configurable)"""
        self.confidence_threshold = max(0.5, min(0.95, strength_percent / 100.0))
        print(f"info string Strength set to {strength_percent}% (confidence {self.confidence_threshold:.2f})")
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get current search statistics"""
        return self.search_stats.copy()


# Backward compatibility alias
V7P3RCleanEngine = V7P3REngineV8
