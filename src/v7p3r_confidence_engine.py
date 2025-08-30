#!/usr/bin/env python3
"""
V7P3R Chess Engine - Confidence-Based Multithreaded Evaluation System
Part of v9.1 Enhancement: Transposition Table Confidence Weighting

This module implements the confidence calculation framework for multithreaded
evaluation with mate/critical move priority preservation.
"""

import time
import threading
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import chess


class MoveCategory(Enum):
    """Categories for move classification with confidence impact"""
    MATE = "mate"
    CRITICAL = "critical"  
    CAPTURE = "capture"
    DEVELOPMENT = "development"
    POSITIONAL = "positional"
    QUIET = "quiet"


@dataclass
class EvaluationMetrics:
    """Metrics collected during evaluation to calculate confidence"""
    search_depth: int = 0
    nodes_searched: int = 0
    time_allocated: float = 0.0
    time_spent: float = 0.0
    thread_count: int = 1
    beta_cutoffs: int = 0
    move_ordering_hits: int = 0
    move_ordering_attempts: int = 0
    alpha_improvements: int = 0
    
    def calculate_time_utilization(self) -> float:
        """Calculate how effectively allocated time was used"""
        if self.time_allocated <= 0:
            return 0.0
        return min(1.0, self.time_spent / self.time_allocated)
    
    def calculate_move_ordering_success(self) -> float:
        """Calculate move ordering effectiveness"""
        if self.move_ordering_attempts <= 0:
            return 0.0
        return self.move_ordering_hits / self.move_ordering_attempts
    
    def calculate_search_efficiency(self) -> float:
        """Calculate beta cutoff efficiency"""
        if self.nodes_searched <= 0:
            return 0.0
        return min(1.0, self.beta_cutoffs / (self.nodes_searched / 10))


@dataclass 
class ConfidenceWeightedEvaluation:
    """Evaluation result with confidence weighting"""
    raw_evaluation: float
    confidence_weight: float
    final_evaluation: float
    move_category: MoveCategory
    metrics: EvaluationMetrics
    is_mate: bool = False
    is_critical: bool = False
    mate_distance: Optional[int] = None
    
    @classmethod
    def create(cls, raw_eval: float, metrics: EvaluationMetrics, 
               move_category: MoveCategory, board: chess.Board, move: chess.Move):
        """Create confidence-weighted evaluation with automatic classification"""
        instance = cls(
            raw_evaluation=raw_eval,
            confidence_weight=0.0,
            final_evaluation=0.0,
            move_category=move_category,
            metrics=metrics
        )
        
        # Detect mates and critical positions
        instance._classify_position(board, move)
        
        # Calculate confidence weight
        instance.confidence_weight = instance._calculate_confidence_weight()
        
        # Apply confidence weighting
        instance.final_evaluation = instance._apply_confidence_weighting()
        
        return instance
    
    def _classify_position(self, board: chess.Board, move: chess.Move):
        """Classify if this is a mate or critical position"""
        # Check for mate
        if abs(self.raw_evaluation) > 20000:  # Mate score threshold
            self.is_mate = True
            self.move_category = MoveCategory.MATE
            self.mate_distance = int((30000 - abs(self.raw_evaluation)) / 100)
        
        # Check for critical positions (high material advantage, forcing moves, etc.)
        elif (abs(self.raw_evaluation) > 300 or  # Material advantage
              board.is_check() or  # Check
              board.is_capture(move) or  # Capture
              self._is_forcing_move(board, move)):  # Forcing move
            self.is_critical = True
            if self.move_category not in [MoveCategory.MATE]:
                self.move_category = MoveCategory.CRITICAL
    
    def _is_forcing_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Detect if move is forcing (checks, captures, threats)"""
        board.push(move)
        is_forcing = (board.is_check() or 
                     len(list(board.legal_moves)) < 10 or  # Limited responses
                     board.is_attacked_by(board.turn, move.to_square))  # Attack
        board.pop()
        return is_forcing
    
    def _calculate_confidence_weight(self) -> float:
        """
        Calculate confidence weight based on evaluation quality metrics
        
        Confidence Framework:
        - Base confidence for mates/critical: 50.9% (guaranteed above 50% threshold)
        - Variable confidence: 49.1% allocated based on search quality
        - Total possible confidence: 100%
        """
        
        # Base confidence for mates and critical moves (51% to ensure safety)
        if self.is_mate:
            base_confidence = 0.70  # 70% for mates
        elif self.is_critical:
            base_confidence = 0.509  # 50.9% for critical moves
        else:
            base_confidence = 0.20  # 20% base for normal moves
        
        # Variable confidence factors (remaining percentage)
        max_variable_confidence = 1.0 - base_confidence
        
        # Calculate quality factors (each 0.0 to 1.0)
        depth_factor = min(1.0, self.metrics.search_depth / 8.0)
        time_factor = self.metrics.calculate_time_utilization()
        ordering_factor = self.metrics.calculate_move_ordering_success()
        efficiency_factor = self.metrics.calculate_search_efficiency()
        thread_factor = min(1.0, self.metrics.thread_count / 4.0)  # Assume max 4 threads
        
        # Weight the factors
        quality_weights = {
            'depth': 0.30,
            'time': 0.25, 
            'ordering': 0.20,
            'efficiency': 0.15,
            'threads': 0.10
        }
        
        # Calculate weighted quality score
        quality_score = (
            depth_factor * quality_weights['depth'] +
            time_factor * quality_weights['time'] +
            ordering_factor * quality_weights['ordering'] +
            efficiency_factor * quality_weights['efficiency'] +
            thread_factor * quality_weights['threads']
        )
        
        # Apply variable confidence
        variable_confidence = quality_score * max_variable_confidence
        
        # Final confidence
        final_confidence = base_confidence + variable_confidence
        
        # Ensure mates and critical moves stay above 50% threshold
        if self.is_mate or self.is_critical:
            final_confidence = max(final_confidence, 0.51)
        
        return min(1.0, final_confidence)
    
    def _apply_confidence_weighting(self) -> float:
        """Apply confidence weighting to raw evaluation"""
        # For mates, preserve mate distance but adjust confidence
        if self.is_mate:
            mate_dist = self.mate_distance or 0  # Handle None case
            if self.raw_evaluation > 0:
                return 20000 + (10000 * self.confidence_weight) + (100 - mate_dist)
            else:
                return -20000 - (10000 * self.confidence_weight) - (100 - mate_dist)
        
        # For other moves, apply confidence weighting
        return self.raw_evaluation * self.confidence_weight


class MultithreadedEvaluationEngine:
    """
    Multithreaded evaluation engine with confidence-based result compilation
    """
    
    def __init__(self, max_threads: int = 4):
        self.max_threads = max_threads
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_threads)
        self.evaluation_cache = {}
        self.lock = threading.RLock()
        
        # Statistics
        self.total_evaluations = 0
        self.cache_hits = 0
        self.thread_utilization = 0.0
    
    def evaluate_position_multithreaded(self, board: chess.Board, move: chess.Move,
                                      base_evaluator, depth: int = 3, 
                                      time_allocation: float = 1.0) -> ConfidenceWeightedEvaluation:
        """
        Evaluate a position using multiple threads with confidence calculation
        """
        start_time = time.time()
        
        # Check cache first
        position_key = self._get_position_key(board, move)
        with self.lock:
            if position_key in self.evaluation_cache:
                self.cache_hits += 1
                return self.evaluation_cache[position_key]
        
        # Prepare evaluation tasks for multiple threads
        futures = []
        
        # Thread 1: Quick shallow evaluation
        futures.append(
            self.thread_pool.submit(
                self._evaluate_shallow, board, move, base_evaluator, depth=2
            )
        )
        
        # Thread 2: Medium depth evaluation
        futures.append(
            self.thread_pool.submit(
                self._evaluate_medium, board, move, base_evaluator, depth=depth
            )
        )
        
        # Thread 3: Deep targeted evaluation (if time allows)
        if time_allocation > 0.5:
            futures.append(
                self.thread_pool.submit(
                    self._evaluate_deep, board, move, base_evaluator, depth=depth+2
                )
            )
        
        # Thread 4: Specialized evaluation (tactics, endgame, etc.)
        futures.append(
            self.thread_pool.submit(
                self._evaluate_specialized, board, move, base_evaluator
            )
        )
        
        # Collect results with timeout
        results = []
        threads_completed = 0
        
        for future in concurrent.futures.as_completed(futures, timeout=time_allocation):
            try:
                result = future.result()
                results.append(result)
                threads_completed += 1
            except concurrent.futures.TimeoutError:
                break
            except Exception as e:
                print(f"info string Thread evaluation error: {e}")
                continue
        
        # Calculate metrics
        time_spent = time.time() - start_time
        metrics = EvaluationMetrics(
            search_depth=depth,
            time_allocated=time_allocation,
            time_spent=time_spent,
            thread_count=threads_completed,
            nodes_searched=sum(r.get('nodes', 0) for r in results),
            beta_cutoffs=sum(r.get('cutoffs', 0) for r in results),
            move_ordering_hits=sum(r.get('ordering_hits', 0) for r in results),
            move_ordering_attempts=sum(r.get('ordering_attempts', 0) for r in results)
        )
        
        # Compile best evaluation with confidence weighting
        best_eval = self._compile_threaded_results(results, metrics, board, move)
        
        # Cache result
        with self.lock:
            self.evaluation_cache[position_key] = best_eval
            self.total_evaluations += 1
        
        return best_eval
    
    def _get_position_key(self, board: chess.Board, move: chess.Move) -> str:
        """Generate cache key for position+move"""
        return f"{board.board_fen()}_{move.uci()}"
    
    def _evaluate_shallow(self, board: chess.Board, move: chess.Move, 
                         evaluator, depth: int) -> Dict[str, Any]:
        """Quick shallow evaluation for immediate tactical assessment"""
        board.push(move)
        try:
            # First check for immediate checkmate
            if board.is_checkmate():
                eval_score = 29000  # Checkmate found
            else:
                eval_score = evaluator._evaluate_position_deterministic(board)
            
            nodes = depth * 10  # Estimate
            return {
                'evaluation': eval_score,
                'depth': depth,
                'nodes': nodes,
                'type': 'shallow',
                'cutoffs': 0,
                'ordering_hits': 0,
                'ordering_attempts': 1
            }
        finally:
            board.pop()
    
    def _evaluate_medium(self, board: chess.Board, move: chess.Move,
                        evaluator, depth: int) -> Dict[str, Any]:
        """Medium depth evaluation with standard search"""
        board.push(move)
        try:
            # Simulate medium-depth search with some tactical analysis
            eval_score = evaluator._evaluate_position_deterministic(board)
            
            # Check for checkmate
            if board.is_checkmate():
                eval_score = 29000  # High mate score
            # Add depth bonus/penalty based on position complexity
            elif board.is_check():
                eval_score += 50  # Bonus for giving check
            
            nodes = depth * 25
            cutoffs = max(0, depth - 2)
            
            return {
                'evaluation': eval_score,
                'depth': depth,
                'nodes': nodes,
                'type': 'medium',
                'cutoffs': cutoffs,
                'ordering_hits': cutoffs,
                'ordering_attempts': depth
            }
        finally:
            board.pop()
    
    def _evaluate_deep(self, board: chess.Board, move: chess.Move,
                      evaluator, depth: int) -> Dict[str, Any]:
        """Deep evaluation for complex positions"""
        board.push(move)
        try:
            eval_score = evaluator._evaluate_position_deterministic(board)
            
            # Deep search would include more sophisticated analysis
            # For now, simulate with position complexity bonus
            complexity_bonus = len(list(board.legal_moves)) * 2
            eval_score += complexity_bonus if board.turn == chess.WHITE else -complexity_bonus
            
            nodes = depth * 50
            cutoffs = max(0, depth - 1)
            
            return {
                'evaluation': eval_score,
                'depth': depth,
                'nodes': nodes,
                'type': 'deep',
                'cutoffs': cutoffs,
                'ordering_hits': cutoffs + 2,
                'ordering_attempts': depth + 1
            }
        finally:
            board.pop()
    
    def _evaluate_specialized(self, board: chess.Board, move: chess.Move,
                            evaluator) -> Dict[str, Any]:
        """Specialized evaluation for tactics, endgames, etc."""
        board.push(move)
        try:
            eval_score = evaluator._evaluate_position_deterministic(board)
            
            # Add specialized knowledge
            if self._is_endgame(board):
                eval_score = self._adjust_for_endgame(board, eval_score)
            elif self._has_tactical_motifs(board):
                eval_score = self._adjust_for_tactics(board, eval_score)
            
            return {
                'evaluation': eval_score,
                'depth': 3,
                'nodes': 30,
                'type': 'specialized',
                'cutoffs': 1,
                'ordering_hits': 2,
                'ordering_attempts': 3
            }
        finally:
            board.pop()
    
    def _compile_threaded_results(self, results: List[Dict[str, Any]], 
                                 metrics: EvaluationMetrics, board: chess.Board,
                                 move: chess.Move) -> ConfidenceWeightedEvaluation:
        """Compile results from multiple threads into confidence-weighted evaluation"""
        if not results:
            # Fallback evaluation
            return ConfidenceWeightedEvaluation.create(
                0.0, metrics, MoveCategory.QUIET, board, move
            )
        
        # Weight evaluations by their depth and type
        weights = {'shallow': 1.0, 'medium': 2.0, 'deep': 3.0, 'specialized': 1.5}
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for result in results:
            weight = weights.get(result['type'], 1.0) * (result['depth'] / 3.0)
            weighted_sum += result['evaluation'] * weight
            total_weight += weight
        
        # Calculate final weighted evaluation
        final_eval = weighted_sum / max(total_weight, 1.0)
        
        # Determine move category
        move_category = self._classify_move(board, move, final_eval)
        
        return ConfidenceWeightedEvaluation.create(
            final_eval, metrics, move_category, board, move
        )
    
    def _classify_move(self, board: chess.Board, move: chess.Move, evaluation: float) -> MoveCategory:
        """Classify move based on position and evaluation"""
        if abs(evaluation) > 20000:
            return MoveCategory.MATE
        elif board.is_capture(move) or board.is_check() or abs(evaluation) > 300:
            return MoveCategory.CRITICAL
        elif board.is_capture(move):
            return MoveCategory.CAPTURE
        else:
            return MoveCategory.POSITIONAL
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """Detect if position is in endgame"""
        piece_count = len(board.piece_map())
        return piece_count <= 10
    
    def _has_tactical_motifs(self, board: chess.Board) -> bool:
        """Detect tactical motifs in position"""
        return board.is_check() or len(list(board.legal_moves)) < 10
    
    def _adjust_for_endgame(self, board: chess.Board, eval_score: float) -> float:
        """Apply endgame-specific adjustments"""
        # Simple king activity bonus in endgame
        king_square = board.king(board.turn)
        if king_square:
            center_distance = abs(chess.square_file(king_square) - 3.5) + abs(chess.square_rank(king_square) - 3.5)
            activity_bonus = (7 - center_distance) * 5
            return eval_score + activity_bonus
        return eval_score
    
    def _adjust_for_tactics(self, board: chess.Board, eval_score: float) -> float:
        """Apply tactical adjustments"""
        # Bonus for limiting opponent options
        if len(list(board.legal_moves)) < 5:
            return eval_score + 100
        return eval_score
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        cache_hit_rate = self.cache_hits / max(self.total_evaluations, 1)
        return {
            'total_evaluations': self.total_evaluations,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'max_threads': self.max_threads
        }
    
    def cleanup(self):
        """Clean up thread pool"""
        self.thread_pool.shutdown(wait=True)
