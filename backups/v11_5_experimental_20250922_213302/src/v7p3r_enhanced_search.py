#!/usr/bin/env python3
"""
V7P3R Enhanced Search with Adaptive Pruning
Integrates with dynamic move ordering and positional intelligence
Author: Pat Snyder
"""

import chess
import time
import sys
from typing import Optional, Tuple, List, Dict, Any
from v7p3r_enhanced_move_orderer import (
    V7P3REnhancedMoveOrderer, MoveClass, ClassifiedMove, PositionTone, SearchFeedback
)


class AdaptivePruningParameters:
    """Dynamic pruning parameters that adapt based on position characteristics"""
    
    def __init__(self, position_tone: PositionTone):
        self.position_tone = position_tone
        
        # Base pruning parameters
        self.base_null_move_depth = 2
        self.base_late_move_reduction = 2
        self.base_futility_margin = 100
        
        # Adaptive parameters based on position tone
        self.calculate_adaptive_parameters()
    
    def calculate_adaptive_parameters(self):
        """Calculate pruning parameters based on position characteristics"""
        tone = self.position_tone
        
        # Tactical positions need less aggressive pruning
        if tone.tactical_complexity > 0.7:
            self.null_move_depth_reduction = 1  # Less null move pruning
            self.late_move_reduction_factor = 0.5  # Less LMR
            self.futility_margin = 150  # Higher futility margin
            self.allow_tactical_extensions = True
        else:
            self.null_move_depth_reduction = 2
            self.late_move_reduction_factor = 1.0
            self.futility_margin = 100
            self.allow_tactical_extensions = False
        
        # King safety critical positions need different pruning
        if tone.king_safety_urgency > 0.6:
            self.prioritize_checks = True
            self.reduce_defensive_pruning = True
        else:
            self.prioritize_checks = False
            self.reduce_defensive_pruning = False
        
        # Endgame positions can use more aggressive pruning for non-king moves
        if tone.endgame_factor > 0.7:
            self.endgame_pruning_bonus = True
            self.king_activity_bonus = True
        else:
            self.endgame_pruning_bonus = False
            self.king_activity_bonus = False


class V7P3REnhancedSearch:
    """Enhanced search engine with adaptive pruning and positional intelligence"""
    
    def __init__(self, evaluator, time_manager):
        self.evaluator = evaluator
        self.time_manager = time_manager
        self.move_orderer = V7P3REnhancedMoveOrderer()
        
        # Search state
        self.nodes_searched = 0
        self.search_start_time = 0.0
        self.current_position_tone = None
        self.pruning_params = None
        
        # Search tracking for feedback
        self.current_move_evaluations = {}
        self.previous_best_evaluation = 0.0
        
        # Search statistics
        self.pruning_stats = {
            'null_moves': 0,
            'late_move_reductions': 0,
            'futility_prunes': 0,
            'adaptive_prunes': 0,
            'tactical_extensions': 0
        }
    
    def search(self, board: chess.Board, time_limit: float = 3.0) -> chess.Move:
        """Main search entry point with enhanced move ordering and adaptive pruning"""
        self.nodes_searched = 0
        self.search_start_time = time.time()
        
        # Reset for new search
        self.move_orderer.reset_for_new_position()
        self.current_move_evaluations = {}
        self.previous_best_evaluation = 0.0
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        # Analyze position tone once for the entire search
        self.current_position_tone = self.move_orderer.analyze_position_tone(board)
        self.move_orderer.current_position_tone = self.current_position_tone
        
        # Set up adaptive pruning parameters
        self.pruning_params = AdaptivePruningParameters(self.current_position_tone)
        
        # Get time allocation
        allocated_time, target_depth = self.time_manager.calculate_time_allocation(board, time_limit)
        
        # Iterative deepening with enhanced intelligence
        best_move = legal_moves[0]
        best_score = -99999
        
        print(f"info string Position tone: tactical={self.current_position_tone.tactical_complexity:.2f} "
              f"safety={self.current_position_tone.king_safety_urgency:.2f} "
              f"endgame={self.current_position_tone.endgame_factor:.2f}")
        
        for depth in range(1, min(target_depth + 1, 8)):
            try:
                # Check time before starting iteration
                elapsed = time.time() - self.search_start_time
                if elapsed > allocated_time * 0.8:
                    break
                
                score, move = self._search_with_adaptive_pruning(board, depth, -99999, 99999, True)
                
                if move and move != chess.Move.null():
                    best_move = move
                    best_score = score
                    
                    # Print search info with enhanced details
                    elapsed_ms = int(elapsed * 1000)
                    nps = int(self.nodes_searched / max(elapsed, 0.001))
                    print(f"info depth {depth} score cp {int(score)} nodes {self.nodes_searched} "
                          f"time {elapsed_ms} nps {nps} pv {move}")
                    
                    # Print pruning statistics
                    if depth >= 3:
                        print(f"info string Pruning: null={self.pruning_stats['null_moves']} "
                              f"lmr={self.pruning_stats['late_move_reductions']} "
                              f"fut={self.pruning_stats['futility_prunes']} "
                              f"adapt={self.pruning_stats['adaptive_prunes']}")
                    
                    sys.stdout.flush()
                
                # Stop if time is running out
                if elapsed > allocated_time * 0.7:
                    break
                    
            except Exception as e:
                print(f"info string Enhanced search error at depth {depth}: {e}")
                break
        
        return best_move
    
    def _search_with_adaptive_pruning(self, board: chess.Board, depth: int, alpha: float, 
                                    beta: float, is_pv_node: bool = False) -> Tuple[float, Optional[chess.Move]]:
        """Enhanced search with adaptive pruning based on position characteristics"""
        self.nodes_searched += 1
        
        # Time check every 1000 nodes
        if self.nodes_searched % 1000 == 0:
            elapsed = time.time() - self.search_start_time
            if elapsed > 10.0:  # Emergency timeout
                return self.evaluator.evaluate(board), None
        
        # Terminal conditions
        if depth == 0:
            return self._enhanced_quiescence_search(board, alpha, beta, 4), None
        
        if board.is_game_over():
            if board.is_checkmate():
                return -29000.0 + (8 - depth), None
            else:
                return 0.0, None
        
        # Transposition table probe (if available)
        # tt_hit, tt_score, tt_move = self._probe_tt(board, depth, alpha, beta)
        # if tt_hit:
        #     return tt_score, tt_move
        
        # Null move pruning with adaptive parameters
        if (depth >= self.pruning_params.base_null_move_depth and 
            not is_pv_node and 
            not board.is_check() and
            self._can_do_null_move(board) and
            self.pruning_params is not None):
            
            null_reduction = self.pruning_params.null_move_depth_reduction
            
            board.push(chess.Move.null())
            null_score, _ = self._search_with_adaptive_pruning(
                board, depth - 1 - null_reduction, -beta, -beta + 1, False
            )
            null_score = -null_score
            board.pop()
            
            if null_score >= beta:
                self.pruning_stats['null_moves'] += 1
                return null_score, None
        
        # Move generation with enhanced ordering
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0, None
        
        # Get classified and ordered moves
        classified_moves = self.move_orderer.classify_and_order_moves(board, legal_moves, depth)
        
        # Search moves with adaptive techniques
        best_score = -99999.0
        best_move = None
        original_alpha = alpha
        moves_searched = 0
        
        for classified_move in classified_moves:
            move = classified_move.move
            move_class = classified_move.primary_class
            
            # Apply adaptive pruning based on move classification and position
            if self._should_prune_move(classified_move, depth, moves_searched, alpha, beta):
                continue
            
            # Determine search depth modifications
            extension = self._calculate_extensions(board, classified_move, depth)
            reduction = self._calculate_reductions(classified_move, depth, moves_searched, is_pv_node)
            
            new_depth = depth - 1 + extension - reduction
            new_depth = max(new_depth, 0)
            
            board.push(move)
            
            # Search with appropriate window
            if moves_searched == 0:
                # First move gets full window
                score, _ = self._search_with_adaptive_pruning(board, new_depth, -beta, -alpha, is_pv_node)
            else:
                # Try null window first
                score, _ = self._search_with_adaptive_pruning(board, new_depth, -alpha - 1, -alpha, False)
                
                # Re-search if necessary
                if score > alpha and (is_pv_node or reduction > 0):
                    score, _ = self._search_with_adaptive_pruning(board, depth - 1, -beta, -alpha, is_pv_node)
            
            score = -score
            board.pop()
            
            # Record move result for adaptive learning
            improved = score > self.previous_best_evaluation
            self.move_orderer.record_search_result(move, move_class, score, improved)
            
            moves_searched += 1
            
            if score > best_score:
                best_score = score
                best_move = move
                self.previous_best_evaluation = score
            
            alpha = max(alpha, score)
            if alpha >= beta:
                # Beta cutoff - store move for future ordering
                if not board.is_capture(move):
                    self.move_orderer.store_killer_move(move, depth)
                break
        
        # Store in transposition table (if available)
        # self._store_tt(board, depth, best_score, best_move, original_alpha, beta)
        
        return best_score, best_move
    
    def _should_prune_move(self, classified_move: ClassifiedMove, depth: int, 
                          moves_searched: int, alpha: float, beta: float) -> bool:
        """Determine if a move should be pruned based on adaptive criteria"""
        
        # Never prune first few moves
        if moves_searched < 3:
            return False
        
        # Never prune in tactical positions if move is tactical
        if (self.current_position_tone and 
            self.current_position_tone.tactical_complexity > 0.7 and 
            classified_move.primary_class == MoveClass.TACTICAL):
            return False
        
        # Never prune when in check or giving check
        if "check" in classified_move.tactical_features:
            return False
        
        # Futility pruning with adaptive margins
        if depth <= 3 and moves_searched > 8:
            margin = self.pruning_params.futility_margin
            
            # Adjust margin based on move type and position
            if classified_move.primary_class == MoveClass.DEFENSIVE:
                margin *= 1.5  # Give defensive moves more chance
            elif classified_move.primary_class == MoveClass.OFFENSIVE:
                margin *= 0.8  # Prune offensive moves more aggressively if failing
            
            # Use search feedback to adjust pruning
            if self.move_orderer.search_feedback.should_deprioritize_move_type(classified_move.primary_class):
                self.pruning_stats['adaptive_prunes'] += 1
                return True
        
        # Late move reductions eligibility (not actual pruning)
        return False
    
    def _calculate_extensions(self, board: chess.Board, classified_move: ClassifiedMove, depth: int) -> int:
        """Calculate search extensions based on move characteristics"""
        extension = 0
        
        # Check extension
        if board.gives_check(classified_move.move):
            extension += 1
        
        # Tactical extensions in tactical positions
        if (self.pruning_params.allow_tactical_extensions and 
            classified_move.primary_class == MoveClass.TACTICAL):
            extension += 1
            self.pruning_stats['tactical_extensions'] += 1
        
        # Recapture extension
        if (board.is_capture(classified_move.move) and 
            "capture" in classified_move.tactical_features):
            extension += 1
        
        # King safety extensions
        if (self.current_position_tone.king_safety_urgency > 0.6 and
            classified_move.primary_class == MoveClass.DEFENSIVE):
            extension += 1
        
        return min(extension, 2)  # Limit total extensions
    
    def _calculate_reductions(self, classified_move: ClassifiedMove, depth: int, 
                            moves_searched: int, is_pv_node: bool) -> int:
        """Calculate late move reductions based on move characteristics"""
        
        # No reduction for first few moves or in PV
        if moves_searched < 4 or is_pv_node:
            return 0
        
        # No reduction for tactical moves
        if classified_move.primary_class == MoveClass.TACTICAL:
            return 0
        
        # No reduction for captures or checks
        if ("capture" in classified_move.tactical_features or 
            "check" in classified_move.tactical_features):
            return 0
        
        # Base reduction
        reduction = int(self.pruning_params.late_move_reduction_factor)
        
        # Increase reduction for moves of types that have been failing
        if self.move_orderer.search_feedback.should_deprioritize_move_type(classified_move.primary_class):
            reduction += 1
            self.pruning_stats['late_move_reductions'] += 1
        
        # Reduce reduction in tactical positions
        if self.current_position_tone.tactical_complexity > 0.7:
            reduction = max(reduction - 1, 0)
        
        return reduction
    
    def _enhanced_quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int) -> float:
        """Enhanced quiescence search with move classification"""
        self.nodes_searched += 1
        
        if depth <= 0:
            return self.evaluator.evaluate(board)
        
        # Stand pat evaluation
        stand_pat = self.evaluator.evaluate(board)
        
        if stand_pat >= beta:
            return beta
        
        alpha = max(alpha, stand_pat)
        
        # Generate and classify captures and checks
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        checks = [move for move in board.legal_moves if board.gives_check(move)]
        
        # Combine and classify moves
        qsearch_moves = list(set(captures + checks))
        classified_moves = self.move_orderer.classify_and_order_moves(board, qsearch_moves, 0)
        
        # Search promising moves only
        for classified_move in classified_moves:
            move = classified_move.move
            
            # Delta pruning with enhanced evaluation
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    captured_value = self.move_orderer.piece_values[captured_piece.piece_type]
                    if stand_pat + captured_value + 200 < alpha:
                        continue  # Delta pruning
            
            board.push(move)
            score = -self._enhanced_quiescence_search(board, -beta, -alpha, depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            
            alpha = max(alpha, score)
        
        return alpha
    
    def _can_do_null_move(self, board: chess.Board) -> bool:
        """Determine if null move is safe in current position"""
        # Don't do null move in endgames or when material is low
        if self.current_position_tone.endgame_factor > 0.7:
            return False
        
        # Don't do null move when in king safety crisis
        if self.current_position_tone.king_safety_urgency > 0.6:
            return False
        
        # Don't do null move in highly tactical positions
        if self.current_position_tone.tactical_complexity > 0.8:
            return False
        
        return True
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics"""
        return {
            'nodes_searched': self.nodes_searched,
            'pruning_stats': self.pruning_stats.copy(),
            'position_tone': {
                'tactical_complexity': self.current_position_tone.tactical_complexity if self.current_position_tone else 0,
                'king_safety_urgency': self.current_position_tone.king_safety_urgency if self.current_position_tone else 0,
                'endgame_factor': self.current_position_tone.endgame_factor if self.current_position_tone else 0,
            },
            'move_type_confidence': {
                move_class.value: self.move_orderer.search_feedback.get_move_type_confidence(move_class)
                for move_class in MoveClass
            } if self.move_orderer else {}
        }