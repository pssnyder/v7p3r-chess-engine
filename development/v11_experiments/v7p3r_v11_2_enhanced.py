#!/usr/bin/env python3
"""
V7P3R Chess Engine v11.2 - Enhanced with Dynamic Positional Intelligence
Integrates sophisticated move classification, adaptive search, and tactical awareness
Based on v11.1 with enhanced move ordering and search pruning

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
from enum import Enum
from dataclasses import dataclass

# Core evaluation components
from v7p3r_bitboard_evaluator import V7P3RScoringCalculationBitboard
from v7p3r_enhanced_evaluation_fixed import V7P3REnhancedEvaluation

# Simplified time manager from v11.1
from v7p3r_simple_time_manager import V7P3RSimpleTimeManager


# Move classification and intelligence components
class MoveClass(Enum):
    """Move classification for tactical and positional understanding"""
    OFFENSIVE = "offensive"           # Captures, attacks, checks
    DEFENSIVE = "defensive"           # Piece defense, blocking, retreat  
    DEVELOPMENTAL = "developmental"   # Piece development, castling, central control
    TACTICAL = "tactical"            # Complex tactical motifs (forks, pins, etc.)
    ENDGAME = "endgame"              # King activity, pawn promotion, centralization


@dataclass
class PositionTone:
    """Analysis of position's strategic character"""
    tactical_complexity: float = 0.0      # 0.0 = quiet, 1.0 = highly tactical
    king_safety_urgency: float = 0.0      # 0.0 = safe, 1.0 = critical danger
    development_priority: float = 0.0     # 0.0 = developed, 1.0 = needs development  
    endgame_factor: float = 0.0           # 0.0 = opening/middle, 1.0 = pure endgame
    aggression_viability: float = 0.5     # 0.0 = defensive, 1.0 = attacking chances
    
    # Dynamic factors updated during search
    offensive_success_rate: float = 0.5
    defensive_success_rate: float = 0.5  
    developmental_success_rate: float = 0.5


@dataclass
class ClassifiedMove:
    """Move with classification and scoring data"""
    move: chess.Move
    primary_class: MoveClass
    score: int
    tactical_features: List[str]


class SearchFeedback:
    """Tracks move type performance during search"""
    
    def __init__(self):
        self.move_type_attempts = {move_class: 0 for move_class in MoveClass}
        self.move_type_successes = {move_class: 0 for move_class in MoveClass}
        
    def record_move_result(self, move_class: MoveClass, improved: bool):
        """Record the result of trying a move of this type"""
        self.move_type_attempts[move_class] += 1
        if improved:
            self.move_type_successes[move_class] += 1
    
    def get_move_type_confidence(self, move_class: MoveClass) -> float:
        """Get confidence that this move type is working"""
        attempts = self.move_type_attempts[move_class]
        if attempts == 0:
            return 0.5  # Neutral confidence
        
        successes = self.move_type_successes[move_class]
        return successes / attempts
    
    def should_deprioritize_move_type(self, move_class: MoveClass, threshold: float = 0.3) -> bool:
        """Determine if we should deprioritize this move type"""
        attempts = self.move_type_attempts[move_class]
        if attempts < 3:  
            return False
        
        confidence = self.get_move_type_confidence(move_class)
        return confidence < threshold


# Simplified data structures from v11.1
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


class SimpleZobrist:
    def __init__(self):
        self.piece_keys = {}
        self.turn_key = random.getrandbits(64)
        
        # Initialize random keys
        for square in range(64):
            for piece_type in range(1, 7):
                for color in [True, False]:
                    key = (square, piece_type, color)
                    self.piece_keys[key] = random.getrandbits(64)
    
    def hash_position(self, board):
        hash_value = 0
        
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                key = (square, piece.piece_type, piece.color)
                if key in self.piece_keys:
                    hash_value ^= self.piece_keys[key]
        
        if not board.turn:
            hash_value ^= self.turn_key
            
        return hash_value


class EnhancedMoveOrderer:
    """Enhanced move ordering with dynamic intelligence"""
    
    def __init__(self):
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330, 
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 10000
        }
        
        self.killer_moves = {}
        self.search_feedback = SearchFeedback()
        self.current_position_tone = None
    
    def analyze_position_tone(self, board: chess.Board) -> PositionTone:
        """Analyze the strategic character of the position"""
        # Material analysis
        white_material = self._calculate_material(board, chess.WHITE)
        black_material = self._calculate_material(board, chess.BLACK)
        total_material = white_material + black_material
        
        # Position metrics
        tactical_complexity = self._assess_tactical_complexity(board)
        king_safety_urgency = self._assess_king_safety(board)
        development_priority = self._assess_development_needs(board)
        endgame_factor = self._assess_endgame_factor(total_material)
        aggression_viability = self._assess_aggression_viability(board)
        
        return PositionTone(
            tactical_complexity=tactical_complexity,
            king_safety_urgency=king_safety_urgency,
            development_priority=development_priority,
            endgame_factor=endgame_factor,
            aggression_viability=aggression_viability
        )
    
    def order_moves(self, board: chess.Board, moves: List[chess.Move], 
                   depth: int = 0, tt_move: Optional[chess.Move] = None) -> List[chess.Move]:
        """Order moves with enhanced intelligence"""
        if not moves:
            return moves
        
        # Analyze position tone if not already done
        if self.current_position_tone is None:
            self.current_position_tone = self.analyze_position_tone(board)
        
        # Classify and score moves
        classified_moves = []
        for move in moves:
            classified_move = self._classify_move(board, move, depth, tt_move)
            classified_moves.append(classified_move)
        
        # Apply dynamic prioritization
        self._apply_dynamic_prioritization(classified_moves)
        
        # Sort by score
        classified_moves.sort(key=lambda x: x.score, reverse=True)
        
        return [cm.move for cm in classified_moves]
    
    def _classify_move(self, board: chess.Board, move: chess.Move, depth: int, 
                      tt_move: Optional[chess.Move]) -> ClassifiedMove:
        """Classify a move and determine its characteristics"""
        
        # Get base score from traditional ordering
        base_score = self._get_traditional_score(board, move, depth, tt_move)
        
        # Identify tactical features
        tactical_features = self._identify_tactical_features(board, move)
        
        # Determine classification
        primary_class = self._determine_move_class(board, move, tactical_features)
        
        return ClassifiedMove(
            move=move,
            primary_class=primary_class,
            score=base_score,
            tactical_features=tactical_features
        )
    
    def _get_traditional_score(self, board: chess.Board, move: chess.Move, 
                             depth: int, tt_move: Optional[chess.Move]) -> int:
        """Get traditional move ordering score"""
        score = 0
        
        # TT move gets highest priority
        if tt_move and move == tt_move:
            return 10000
        
        # Captures (MVV-LVA)
        if board.is_capture(move):
            victim_value = 0
            attacker_value = 0
            
            captured_piece = board.piece_at(move.to_square)
            moving_piece = board.piece_at(move.from_square)
            
            if captured_piece:
                victim_value = self.piece_values[captured_piece.piece_type]
            if moving_piece:
                attacker_value = self.piece_values[moving_piece.piece_type]
            
            mvv_lva_score = victim_value - (attacker_value // 10)
            
            if mvv_lva_score > 0:
                score = 8000 + mvv_lva_score
            else:
                score = 6000 + mvv_lva_score
        
        # Checks
        elif board.gives_check(move):
            score = 7000
        
        # Killer moves
        elif self._is_killer_move(move, depth):
            score = 5000
        
        # Promotions
        elif move.promotion:
            if move.promotion == chess.QUEEN:
                score = 4500
            else:
                score = 4000
        
        # Castling
        elif board.is_castling(move):
            score = 3500
        
        # Basic positional bonus
        else:
            score = self._get_basic_positional_bonus(board, move)
        
        return score
    
    def _identify_tactical_features(self, board: chess.Board, move: chess.Move) -> List[str]:
        """Identify tactical patterns in the move"""
        features = []
        
        # Check if move gives check
        if board.gives_check(move):
            features.append("check")
        
        # Check if move is a capture
        if board.is_capture(move):
            features.append("capture")
        
        # Simple fork detection for knights
        moving_piece = board.piece_at(move.from_square)
        if moving_piece and moving_piece.piece_type == chess.KNIGHT:
            board.push(move)
            attacks = board.attacks(move.to_square)
            enemy_high_value = 0
            for attacked_square in attacks:
                attacked_piece = board.piece_at(attacked_square)
                if (attacked_piece and attacked_piece.color != moving_piece.color and 
                    attacked_piece.piece_type in {chess.KING, chess.QUEEN, chess.ROOK}):
                    enemy_high_value += 1
            if enemy_high_value >= 2:
                features.append("fork")
            board.pop()
        
        return features
    
    def _determine_move_class(self, board: chess.Board, move: chess.Move, 
                            tactical_features: List[str]) -> MoveClass:
        """Determine primary classification for the move"""
        
        # Tactical moves take priority
        if "fork" in tactical_features:
            return MoveClass.TACTICAL
        
        # Captures and checks are offensive
        if board.is_capture(move) or "check" in tactical_features:
            return MoveClass.OFFENSIVE
        
        # Development moves
        if board.is_castling(move) or self._is_development_move(board, move):
            return MoveClass.DEVELOPMENTAL
        
        # Endgame moves
        if self.current_position_tone and self.current_position_tone.endgame_factor > 0.7:
            return MoveClass.ENDGAME
        
        # Default to developmental
        return MoveClass.DEVELOPMENTAL
    
    def _apply_dynamic_prioritization(self, classified_moves: List[ClassifiedMove]):
        """Apply dynamic prioritization based on search feedback"""
        if not self.current_position_tone:
            return
        
        tone = self.current_position_tone
        
        for classified_move in classified_moves:
            move_class = classified_move.primary_class
            bonus = 0
            
            # Apply position tone bonuses
            if move_class == MoveClass.OFFENSIVE:
                if tone.aggression_viability > 0.6:
                    bonus += 200
                if tone.tactical_complexity > 0.7:
                    bonus += 150
            
            elif move_class == MoveClass.DEFENSIVE:
                if tone.king_safety_urgency > 0.6:
                    bonus += 250
            
            elif move_class == MoveClass.DEVELOPMENTAL:
                if tone.development_priority > 0.6:
                    bonus += 200
            
            elif move_class == MoveClass.TACTICAL:
                if tone.tactical_complexity > 0.5:
                    bonus += 300
            
            elif move_class == MoveClass.ENDGAME:
                if tone.endgame_factor > 0.7:
                    bonus += 200
            
            # Apply search feedback penalties
            if self.search_feedback.should_deprioritize_move_type(move_class):
                bonus -= 400
            
            classified_move.score += bonus
    
    def _is_development_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move develops a piece"""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        
        # Simple development check: piece moving from back rank
        if piece.color == chess.WHITE:
            return chess.square_rank(move.from_square) <= 1 and chess.square_rank(move.to_square) > 1
        else:
            return chess.square_rank(move.from_square) >= 6 and chess.square_rank(move.to_square) < 6
    
    # Position analysis helper methods
    def _calculate_material(self, board: chess.Board, color: bool) -> int:
        """Calculate total material for a color"""
        material = 0
        for piece_type, value in self.piece_values.items():
            if piece_type != chess.KING:
                material += len(board.pieces(piece_type, color)) * value
        return material
    
    def _assess_tactical_complexity(self, board: chess.Board) -> float:
        """Assess how tactically complex the position is"""
        complexity = 0.0
        
        # More pieces = more tactical
        total_pieces = len(board.piece_map())
        complexity += min(total_pieces / 32.0, 1.0) * 0.3
        
        # Checks increase complexity
        if board.is_check():
            complexity += 0.4
        
        # Available captures
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        complexity += min(len(captures) / 10.0, 1.0) * 0.3
        
        return min(complexity, 1.0)
    
    def _assess_king_safety(self, board: chess.Board) -> float:
        """Assess king safety urgency"""
        urgency = 0.0
        
        if board.is_check():
            urgency += 0.6
        
        # Count attackers near king
        our_color = board.turn
        king_square = board.king(our_color)
        
        if king_square is not None:
            # Simple king safety: count enemy attacks around king
            attackers = 0
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            
            for file_offset in [-1, 0, 1]:
                for rank_offset in [-1, 0, 1]:
                    new_file = king_file + file_offset
                    new_rank = king_rank + rank_offset
                    if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
                        square = chess.square(new_file, new_rank)
                        if board.is_attacked_by(not our_color, square):
                            attackers += 1
            
            urgency += min(attackers / 8.0, 0.4)
        
        return min(urgency, 1.0)
    
    def _assess_development_needs(self, board: chess.Board) -> float:
        """Assess how much development is needed"""
        our_color = board.turn
        development_score = 0.0
        
        # Check minor pieces on back rank
        back_rank = 0 if our_color == chess.WHITE else 7
        
        for file in range(8):
            square = chess.square(file, back_rank)
            piece = board.piece_at(square)
            if piece and piece.color == our_color and piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
                development_score += 0.25
        
        # Check castling
        if our_color == chess.WHITE:
            if not (board.has_castling_rights(chess.WHITE) or 
                   board.king(chess.WHITE) in {chess.G1, chess.C1}):
                development_score += 0.3
        else:
            if not (board.has_castling_rights(chess.BLACK) or 
                   board.king(chess.BLACK) in {chess.G8, chess.C8}):
                development_score += 0.3
        
        return min(development_score, 1.0)
    
    def _assess_endgame_factor(self, total_material: int) -> float:
        """Assess how much we're in the endgame"""
        opening_material = 7800
        endgame_threshold = 2000
        
        if total_material <= endgame_threshold:
            return 1.0
        elif total_material >= opening_material:
            return 0.0
        else:
            return 1.0 - ((total_material - endgame_threshold) / (opening_material - endgame_threshold))
    
    def _assess_aggression_viability(self, board: chess.Board) -> float:
        """Assess whether aggressive moves are likely to succeed"""
        viability = 0.5
        
        our_color = board.turn
        our_material = self._calculate_material(board, our_color)
        enemy_material = self._calculate_material(board, not our_color)
        
        # Material advantage affects aggression viability
        if our_material > enemy_material:
            advantage = (our_material - enemy_material) / max(enemy_material, 1)
            viability += min(advantage, 0.3)
        
        return min(max(viability, 0.0), 1.0)
    
    def _is_killer_move(self, move: chess.Move, depth: int) -> bool:
        """Check if move is a killer move"""
        if depth not in self.killer_moves:
            return False
        return move in self.killer_moves[depth]
    
    def store_killer_move(self, move: chess.Move, depth: int):
        """Store a killer move"""
        if depth not in self.killer_moves:
            self.killer_moves[depth] = []
        
        if move not in self.killer_moves[depth]:
            self.killer_moves[depth].insert(0, move)
            if len(self.killer_moves[depth]) > 2:
                self.killer_moves[depth].pop()
    
    def _get_basic_positional_bonus(self, board: chess.Board, move: chess.Move) -> int:
        """Basic positional bonus for quiet moves"""
        bonus = 100
        
        piece = board.piece_at(move.from_square)
        if not piece:
            return bonus
        
        # Center control
        center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        if move.to_square in center_squares:
            bonus += 50
        
        # Piece-specific bonuses
        if piece.piece_type == chess.KNIGHT:
            if move.to_square in {chess.C3, chess.F3, chess.C6, chess.F6}:
                bonus += 30
        elif piece.piece_type == chess.BISHOP:
            if move.to_square in {chess.C4, chess.F4, chess.C5, chess.F5}:
                bonus += 25
        
        return bonus
    
    def record_search_result(self, move: chess.Move, move_class: MoveClass, improved: bool):
        """Record search result for learning"""
        self.search_feedback.record_move_result(move_class, improved)
    
    def reset_for_new_position(self):
        """Reset for new position"""
        self.current_position_tone = None
        self.search_feedback = SearchFeedback()
    
    def clear_killers(self):
        """Clear killer moves"""
        self.killer_moves.clear()
        self.reset_for_new_position()


class V7P3REngineEnhanced:
    """V7P3R v11.2 - Enhanced with Dynamic Positional Intelligence"""
    
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
        
        # Core evaluator
        self.fast_evaluator = V7P3RScoringCalculationBitboard(self.piece_values)
        
        # Enhanced evaluation system
        self.enhanced_evaluator = V7P3REnhancedEvaluation()
        
        # Enhanced systems
        self.time_manager = V7P3RSimpleTimeManager()
        self.move_orderer = EnhancedMoveOrderer()
        
        # Basic search infrastructure
        self.evaluation_cache = {}
        self.transposition_table = {}
        self.max_tt_entries = 100000
        self.zobrist = SimpleZobrist()
        self.killer_moves = KillerMoves()
        
        # Stats
        self.search_stats = {
            'nodes_searched': 0,
            'cache_hits': 0,
            'tt_hits': 0,
            'evaluation_calls': 0,
            'tactical_extensions': 0,
            'adaptive_prunes': 0
        }
    
    def search(self, board: chess.Board, time_limit: float = 3.0) -> chess.Move:
        """Enhanced search with dynamic positional intelligence"""
        self.nodes_searched = 0
        self.search_start_time = time.time()
        
        # Reset move orderer for new search
        self.move_orderer.reset_for_new_position()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        # Get time allocation
        allocated_time, target_depth = self.time_manager.calculate_time_allocation(board, time_limit)
        
        # Iterative deepening with enhanced intelligence
        best_move = legal_moves[0]
        best_score = -99999
        previous_score = 0
        
        # Analyze position tone once
        position_tone = self.move_orderer.analyze_position_tone(board)
        print(f"info string Position analysis: tactical={position_tone.tactical_complexity:.2f} "
              f"safety={position_tone.king_safety_urgency:.2f} "
              f"endgame={position_tone.endgame_factor:.2f}")
        
        for depth in range(1, min(target_depth + 1, self.default_depth + 1)):
            try:
                # Check time before starting iteration
                elapsed = time.time() - self.search_start_time
                if elapsed > allocated_time * 0.8:
                    break
                
                score, move = self._enhanced_search_recursive(board, depth, -99999, 99999, previous_score)
                
                if move and move != chess.Move.null():
                    best_move = move
                    best_score = score
                    previous_score = score
                    
                    # Print search info
                    elapsed_ms = int(elapsed * 1000)
                    nps = int(self.nodes_searched / max(elapsed, 0.001))
                    print(f"info depth {depth} score cp {int(score)} nodes {self.nodes_searched} "
                          f"time {elapsed_ms} nps {nps} pv {move}")
                    sys.stdout.flush()
                
                # Stop if time is running out
                if elapsed > allocated_time * 0.7:
                    break
                    
            except Exception as e:
                print(f"info string Enhanced search error at depth {depth}: {e}")
                break
        
        # Update search stats
        self.search_stats['nodes_searched'] = self.nodes_searched
        
        return best_move
    
    def _enhanced_search_recursive(self, board: chess.Board, depth: int, alpha: float, beta: float, 
                                 previous_score: float) -> Tuple[float, Optional[chess.Move]]:
        """Enhanced recursive search with adaptive pruning"""
        self.nodes_searched += 1
        
        # Time check every 1000 nodes
        if self.nodes_searched % 1000 == 0:
            elapsed = time.time() - self.search_start_time
            if elapsed > 10.0:  # Emergency timeout
                return self._evaluate_position(board), None
        
        # Terminal conditions
        if depth == 0:
            return self._enhanced_quiescence_search(board, alpha, beta, 3), None
        
        if board.is_game_over():
            if board.is_checkmate():
                return -29000.0 + (self.default_depth - depth), None
            else:
                return 0.0, None
        
        # Transposition table probe
        tt_hit, tt_score, tt_move = self._probe_tt(board, depth, alpha, beta)
        if tt_hit:
            return tt_score, tt_move
        
        # Move generation with enhanced ordering
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0, None
        
        ordered_moves = self.move_orderer.order_moves(board, legal_moves, depth, tt_move)
        
        # Search moves with enhanced intelligence
        best_score = -99999.0
        best_move = None
        original_alpha = alpha
        moves_searched = 0
        
        for move in ordered_moves:
            # Get move classification for feedback
            move_class = self._get_move_class(board, move)
            
            # Determine if we should extend search
            extension = 0
            if board.gives_check(move):
                extension = 1
                self.search_stats['tactical_extensions'] += 1
            
            # Apply late move reductions for quiet moves
            reduction = 0
            if (moves_searched >= 4 and 
                not board.is_capture(move) and 
                not board.gives_check(move)):
                reduction = 1
            
            new_depth = depth - 1 + extension - reduction
            new_depth = max(new_depth, 0)
            
            board.push(move)
            
            if moves_searched == 0:
                # First move gets full window
                score, _ = self._enhanced_search_recursive(board, new_depth, -beta, -alpha, -previous_score)
            else:
                # Try null window first
                score, _ = self._enhanced_search_recursive(board, new_depth, -alpha - 1, -alpha, -previous_score)
                
                # Re-search if necessary
                if score > alpha and reduction > 0:
                    score, _ = self._enhanced_search_recursive(board, depth - 1, -beta, -alpha, -previous_score)
            
            score = -score
            board.pop()
            
            # Record result for adaptive learning
            improved = score > previous_score
            self.move_orderer.record_search_result(move, move_class, improved)
            
            moves_searched += 1
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            if alpha >= beta:
                # Beta cutoff
                if not board.is_capture(move):
                    self.move_orderer.store_killer_move(move, depth)
                break
        
        # Store in transposition table
        self._store_tt(board, depth, best_score, best_move, original_alpha, beta)
        
        return best_score, best_move
    
    def _get_move_class(self, board: chess.Board, move: chess.Move) -> MoveClass:
        """Quick move classification for feedback"""
        if board.is_capture(move) or board.gives_check(move):
            return MoveClass.OFFENSIVE
        elif board.is_castling(move):
            return MoveClass.DEVELOPMENTAL
        else:
            return MoveClass.DEVELOPMENTAL
    
    def _enhanced_quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int) -> float:
        """Enhanced quiescence search"""
        self.nodes_searched += 1
        
        if depth <= 0:
            return self._evaluate_position(board)
        
        # Stand pat
        stand_pat = self._evaluate_position(board)
        
        if stand_pat >= beta:
            return beta
        
        alpha = max(alpha, stand_pat)
        
        # Only search captures and checks
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        checks = [move for move in board.legal_moves if board.gives_check(move)]
        
        # Use enhanced ordering for quiescence moves
        qsearch_moves = list(set(captures + checks))
        ordered_qmoves = self.move_orderer.order_moves(board, qsearch_moves, 0)
        
        for move in ordered_qmoves:
            # Delta pruning
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    captured_value = self.piece_values[captured_piece.piece_type]
                    if stand_pat + captured_value + 200 < alpha:
                        continue
            
            board.push(move)
            score = -self._enhanced_quiescence_search(board, -beta, -alpha, depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            
            alpha = max(alpha, score)
        
        return alpha
    
    def _basic_material_evaluation(self, board: chess.Board) -> float:
        """Basic material evaluation"""
        score = 0.0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        
        # Adjust score based on current player
        if board.turn == chess.BLACK:
            score = -score
        
        return score
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """Enhanced position evaluation"""
        cache_key = board.fen()
        
        if cache_key in self.evaluation_cache:
            self.search_stats['cache_hits'] += 1
            return self.evaluation_cache[cache_key]
        
        self.search_stats['evaluation_calls'] += 1
        
        # Use enhanced evaluation system
        try:
            score = self.enhanced_evaluator.evaluate_position(board)
        except Exception as e:
            print(f"info string Enhanced evaluator error: {e}")
            # Fallback to basic material evaluation
            score = self._basic_material_evaluation(board)
        
        # Add tactical bonuses based on position tone
        if self.move_orderer.current_position_tone:
            tone = self.move_orderer.current_position_tone
            
            # Bonus for tactical complexity recognition
            if tone.tactical_complexity > 0.7:
                # Give slight bonus for finding tactics
                if board.is_check():
                    score += 10
                
                # Bonus for piece activity in tactical positions
                our_color = board.turn
                piece_activity = 0
                for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    for piece_square in board.pieces(piece_type, our_color):
                        piece_activity += len(board.attacks(piece_square))
                
                score += piece_activity * 2
        
        # Cache the evaluation
        if len(self.evaluation_cache) < 10000:
            self.evaluation_cache[cache_key] = score
        
        return score
    
    def _probe_tt(self, board: chess.Board, depth: int, alpha: float, beta: float) -> Tuple[bool, float, Optional[chess.Move]]:
        """Probe transposition table"""
        zobrist_hash = self.zobrist.hash_position(board)
        
        if zobrist_hash in self.transposition_table:
            entry = self.transposition_table[zobrist_hash]
            if entry.depth >= depth and entry.zobrist_hash == zobrist_hash:
                self.search_stats['tt_hits'] += 1
                return True, entry.score, entry.best_move
        
        return False, 0.0, None
    
    def _store_tt(self, board: chess.Board, depth: int, score: float, best_move: Optional[chess.Move], 
                 alpha: float, beta: float):
        """Store in transposition table"""
        zobrist_hash = self.zobrist.hash_position(board)
        
        # Determine node type
        if score <= alpha:
            node_type = "UPPERBOUND"
        elif score >= beta:
            node_type = "LOWERBOUND"
        else:
            node_type = "EXACT"
        
        # Store entry
        entry = TranspositionEntry(depth, score, best_move, node_type, zobrist_hash)
        self.transposition_table[zobrist_hash] = entry
        
        # Limit table size
        if len(self.transposition_table) > self.max_tt_entries:
            # Remove oldest entries (simple replacement)
            keys_to_remove = list(self.transposition_table.keys())[:1000]
            for key in keys_to_remove:
                del self.transposition_table[key]
    
    def new_game(self):
        """Reset for new game"""
        self.evaluation_cache.clear()
        self.transposition_table.clear()
        self.move_orderer.clear_killers()
        self.killer_moves = KillerMoves()
        
        # Reset stats
        self.search_stats = {
            'nodes_searched': 0,
            'cache_hits': 0,
            'tt_hits': 0,
            'evaluation_calls': 0,
            'tactical_extensions': 0,
            'adaptive_prunes': 0
        }
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information"""
        return {
            'name': 'V7P3R v11.2 Enhanced',
            'version': '11.2',
            'author': 'Pat Snyder',
            'features': [
                'Dynamic Positional Intelligence',
                'Adaptive Move Ordering',
                'Enhanced Tactical Recognition',
                'Position Tone Analysis',
                'Search Feedback Learning'
            ],
            'search_stats': self.search_stats
        }


# Main execution for direct testing
if __name__ == "__main__":
    print("V7P3R Chess Engine v11.2 - Enhanced with Dynamic Positional Intelligence")
    print("Direct-to-code testing mode")
    
    engine = V7P3REngineEnhanced()
    board = chess.Board()
    
    # Test position
    print("\nTesting basic position...")
    start_time = time.time()
    best_move = engine.search(board, 2.0)
    elapsed = time.time() - start_time
    
    print(f"\nResult: {best_move}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Nodes: {engine.nodes_searched}")
    print(f"NPS: {int(engine.nodes_searched / max(elapsed, 0.001))}")
    
    print("\nEngine info:", engine.get_engine_info())