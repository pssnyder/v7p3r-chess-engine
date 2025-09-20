#!/usr/bin/env python3
"""
V7P3R Chess Engine v11.1 - EMERGENCY PERFORMANCE FIXES
Built from v11 with critical simplifications for performance recovery
Phase 1 Fixes: Simplified time management, basic move ordering, consistent evaluation

VERSION LINEAGE:
- v10.6: Tournament baseline (19.5/30 points)
- v10.7: Failed tactical patterns (70% performance loss)  
- v10.8: Recovery baseline for v11 development (but had critical perspective bug)
- v10.9: CRITICAL PERSPECTIVE FIX - resolves 42% performance gap between White/Black play
- v11.0: Advanced features but severe performance regression (4.0/21 points)
- v11.1: EMERGENCY FIXES - simplified systems for performance recovery

CRITICAL FIXES IN v11.1:
- Simplified time management (reliable 70% allocation)
- Basic move ordering (tactical priorities only)
- Consistent evaluation (fast evaluator only)
- Reduced search complexity (minimal pruning)

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
from v7p3r_bitboard_evaluator import V7P3RScoringCalculationBitboard
from v7p3r_advanced_pawn_evaluator import V7P3RAdvancedPawnEvaluator
from v7p3r_king_safety_evaluator import V7P3RKingSafetyEvaluator
from v7p3r_strategic_database import V7P3RStrategicDatabase
from v7p3r_tactical_pattern_detector import TimeControlAdaptiveTacticalDetector  # V10.9 PHASE 3B: TIME-ADAPTIVE TACTICAL PATTERNS
from v7p3r_simple_time_manager import V7P3RSimpleTimeManager  # V11.1 SIMPLIFIED TIME MANAGEMENT
from v7p3r_simple_move_orderer import V7P3RSimpleMoveOrderer  # V11.1 SIMPLIFIED MOVE ORDERING


# ==============================================================================
# V11 CONSOLIDATED MODULES: Position Posture Assessment
# ==============================================================================

class PositionVolatility(Enum):
    """Position volatility levels"""
    STABLE = "stable"          # Quiet, positional game
    TACTICAL = "tactical"      # Some tactics, moderate complexity
    VOLATILE = "volatile"      # High tactics, many threats
    CRITICAL = "critical"      # Immediate danger, forcing moves


class GamePosture(Enum):
    """Current game posture for evaluation priorities"""
    OFFENSIVE = "offensive"    # We have initiative, prioritize attacks
    BALANCED = "balanced"      # Equal position, standard evaluation
    DEFENSIVE = "defensive"    # Under pressure, prioritize defense
    EMERGENCY = "emergency"    # Immediate threats, survival mode


class V7P3RPostureAssessment:
    """Fast position posture assessment for adaptive evaluation"""
    
    def __init__(self):
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 10000
        }
        
        # Cached assessments for performance
        self.posture_cache = {}
        self.assessment_stats = {
            'calls': 0,
            'cache_hits': 0,
            'total_time': 0.0
        }
    
    def assess_position_posture(self, board: chess.Board) -> Tuple[PositionVolatility, GamePosture]:
        """
        Main entry point: assess position volatility and determine game posture
        Returns: (volatility_level, game_posture)
        """
        self.assessment_stats['calls'] += 1
        
        # Quick cache check
        cache_key = self._generate_cache_key(board)
        if cache_key in self.posture_cache:
            self.assessment_stats['cache_hits'] += 1
            return self.posture_cache[cache_key]
        
        # Perform fast assessment
        volatility = self._assess_volatility(board)
        posture = self._determine_posture(board, volatility)
        
        # Cache result
        result = (volatility, posture)
        self.posture_cache[cache_key] = result
        
        return result
    
    def _generate_cache_key(self, board: chess.Board) -> str:
        """Generate lightweight cache key for position"""
        key_parts = []
        
        # Material counts for both sides
        white_material = sum(len(board.pieces(piece_type, True)) * value 
                           for piece_type, value in self.piece_values.items())
        black_material = sum(len(board.pieces(piece_type, False)) * value 
                           for piece_type, value in self.piece_values.items())
        
        key_parts.append(f"mat_{white_material}_{black_material}")
        key_parts.append(f"turn_{board.turn}")
        
        # King positions (critical for threat assessment)
        if board.king(True):
            key_parts.append(f"wk_{board.king(True)}")
        if board.king(False):
            key_parts.append(f"bk_{board.king(False)}")
        
        # Castling rights (affects king safety)
        key_parts.append(f"castle_{board.castling_rights}")
        
        return "_".join(key_parts)
    
    def _assess_volatility(self, board: chess.Board) -> PositionVolatility:
        """Assess position volatility level"""
        volatility_score = 0
        
        # Factor 1: Pieces under attack
        attacked_pieces = self._count_attacked_pieces(board)
        volatility_score += attacked_pieces * 8
        
        # Factor 2: Undefended pieces
        undefended_pieces = self._count_undefended_pieces(board)
        volatility_score += undefended_pieces * 12
        
        # Factor 3: Pins and skewers
        tactical_threats = self._count_tactical_patterns(board)
        volatility_score += tactical_threats * 15
        
        # Factor 4: King exposure
        king_danger = self._assess_king_exposure(board)
        volatility_score += king_danger * 18
        
        # Factor 5: Material imbalance
        material_imbalance = self._assess_material_imbalance(board)
        if abs(material_imbalance) >= 300:
            volatility_score += abs(material_imbalance) // 150
        
        # Classify volatility
        if volatility_score <= 10:
            return PositionVolatility.STABLE
        elif volatility_score <= 30:
            return PositionVolatility.TACTICAL
        elif volatility_score <= 60:
            return PositionVolatility.VOLATILE
        else:
            return PositionVolatility.CRITICAL
    
    def _determine_posture(self, board: chess.Board, volatility: PositionVolatility) -> GamePosture:
        """Determine game posture based on position assessment"""
        
        # Emergency posture for critical positions
        if volatility == PositionVolatility.CRITICAL:
            return GamePosture.EMERGENCY
        
        # Assess threat balance
        our_threats = self._count_our_threats(board)
        their_threats = self._count_their_threats(board)
        
        # Assess material advantage
        material_advantage = self._get_material_advantage(board)
        
        # Assess initiative
        initiative_factor = our_threats - their_threats
        
        # Combine factors to determine posture
        posture_score = initiative_factor + (material_advantage // 100)
        
        # Factor in volatility
        if volatility == PositionVolatility.VOLATILE:
            if posture_score <= 1:
                return GamePosture.DEFENSIVE
        
        # Determine posture
        if their_threats >= our_threats + 2:
            return GamePosture.DEFENSIVE
        elif their_threats > our_threats and volatility in [PositionVolatility.VOLATILE, PositionVolatility.CRITICAL]:
            return GamePosture.DEFENSIVE
        elif our_threats >= their_threats + 2:
            return GamePosture.OFFENSIVE
        elif material_advantage >= 200:
            return GamePosture.OFFENSIVE
        elif material_advantage <= -200:
            return GamePosture.DEFENSIVE
        else:
            return GamePosture.BALANCED
    
    def _count_attacked_pieces(self, board: chess.Board) -> int:
        """Count pieces under attack for current player"""
        attacked_count = 0
        current_color = board.turn
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == current_color:
                if board.is_attacked_by(not current_color, square):
                    attacked_count += 1
        
        return attacked_count
    
    def _count_undefended_pieces(self, board: chess.Board) -> int:
        """Count undefended pieces for current player"""
        undefended_count = 0
        current_color = board.turn
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == current_color and piece.piece_type != chess.KING:
                if (board.is_attacked_by(not current_color, square) and 
                    not board.is_attacked_by(current_color, square)):
                    undefended_count += 1
        
        return undefended_count
    
    def _count_tactical_patterns(self, board: chess.Board) -> int:
        """Count basic tactical patterns (pins, skewers, forks)"""
        tactical_count = 0
        current_color = board.turn
        
        # Look for potential forks
        for move in board.legal_moves:
            board.push(move)
            
            # Count how many enemy pieces this move would attack
            attacking_count = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color != current_color:
                    if board.is_attacked_by(current_color, square):
                        attacking_count += 1
            
            if attacking_count >= 2:
                tactical_count += 1
            
            board.pop()
            
            if tactical_count >= 5:
                break
        
        return min(tactical_count, 3)
    
    def _assess_king_exposure(self, board: chess.Board) -> int:
        """Assess king exposure for current player"""
        current_color = board.turn
        king_square = board.king(current_color)
        
        if king_square is None:
            return 10
        
        danger_score = 0
        
        # Count attackers on king
        if board.is_attacked_by(not current_color, king_square):
            danger_score += 3
        
        # Check squares around king
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        for file_offset in [-1, 0, 1]:
            for rank_offset in [-1, 0, 1]:
                if file_offset == 0 and rank_offset == 0:
                    continue
                
                new_file = king_file + file_offset
                new_rank = king_rank + rank_offset
                
                if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
                    square = chess.square(new_file, new_rank)
                    if board.is_attacked_by(not current_color, square):
                        danger_score += 1
        
        return danger_score
    
    def _assess_material_imbalance(self, board: chess.Board) -> int:
        """Assess material imbalance from current player's perspective"""
        white_material = sum(len(board.pieces(piece_type, True)) * value 
                           for piece_type, value in self.piece_values.items())
        black_material = sum(len(board.pieces(piece_type, False)) * value 
                           for piece_type, value in self.piece_values.items())
        
        if board.turn:
            return white_material - black_material
        else:
            return black_material - white_material
    
    def _count_our_threats(self, board: chess.Board) -> int:
        """Count threats we can create"""
        threat_count = 0
        current_color = board.turn
        
        for move in list(board.legal_moves)[:20]:
            board.push(move)
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color != current_color:
                    if board.is_attacked_by(current_color, square):
                        threat_count += 1
            
            board.pop()
        
        return threat_count
    
    def _count_their_threats(self, board: chess.Board) -> int:
        """Count threats opponent can create on their turn"""
        current_color = board.turn
        
        threat_count = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == current_color:
                if board.is_attacked_by(not current_color, square):
                    threat_count += 1
        
        return threat_count
    
    def _get_material_advantage(self, board: chess.Board) -> int:
        """Get material advantage for current player"""
        return self._assess_material_imbalance(board)
    
    def get_cache_stats(self) -> Dict:
        """Get performance statistics"""
        hit_rate = 0.0
        if self.assessment_stats['calls'] > 0:
            hit_rate = (self.assessment_stats['cache_hits'] / self.assessment_stats['calls']) * 100
        
        return {
            'calls': self.assessment_stats['calls'],
            'cache_hits': self.assessment_stats['cache_hits'],
            'hit_rate_percent': hit_rate,
            'cache_size': len(self.posture_cache)
        }


# ==============================================================================
# V11 CONSOLIDATED MODULES: Fast Evaluation System
# ==============================================================================

class V7P3RFastEvaluator:
    """Lightweight evaluator for performance-critical search nodes"""
    
    def __init__(self):
        # Standard piece values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 10000
        }
        
        # Simple positional tables for piece-square values
        self.pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        self.knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
        
        self.bishop_table = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]
        
        self.rook_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ]
        
        self.queen_table = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]
        
        self.king_table = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]
        
        # Simple evaluation cache
        self.evaluation_cache: Dict[str, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def evaluate_position_fast(self, board: chess.Board) -> float:
        """Fast evaluation for performance-critical nodes"""
        # Check cache first
        position_hash = str(board.board_fen())
        if position_hash in self.evaluation_cache:
            self.cache_hits += 1
            return self.evaluation_cache[position_hash]
        
        self.cache_misses += 1
        
        # Quick material + piece-square evaluation
        white_score = 0
        black_score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Material value
                material_value = self.piece_values[piece.piece_type]
                
                # Piece-square table value
                positional_value = self._get_piece_square_value(piece, square)
                
                total_value = material_value + positional_value
                
                if piece.color:
                    white_score += total_value
                else:
                    black_score += total_value
        
        # Simple mobility bonus
        mobility_bonus = 0
        if not board.is_game_over():
            try:
                num_moves = len(list(board.legal_moves))
                mobility_bonus = min(num_moves * 5, 100)
            except:
                mobility_bonus = 0
        
        # Apply mobility to current player
        if board.turn:
            white_score += mobility_bonus
        else:
            black_score += mobility_bonus
        
        # Calculate final score from current player's perspective
        if board.turn:
            final_score = (white_score - black_score) / 100.0
        else:
            final_score = (black_score - white_score) / 100.0
        
        # Cache the result
        self.evaluation_cache[position_hash] = final_score
        
        return final_score
    
    def _get_piece_square_value(self, piece: chess.Piece, square: int) -> int:
        """Get piece-square table value for piece at square"""
        # Convert square to table index (flip for black pieces)
        if piece.color:
            table_index = square
        else:
            table_index = square ^ 56
        
        # Select appropriate table
        if piece.piece_type == chess.PAWN:
            return self.pawn_table[table_index]
        elif piece.piece_type == chess.KNIGHT:
            return self.knight_table[table_index]
        elif piece.piece_type == chess.BISHOP:
            return self.bishop_table[table_index]
        elif piece.piece_type == chess.ROOK:
            return self.rook_table[table_index]
        elif piece.piece_type == chess.QUEEN:
            return self.queen_table[table_index]
        elif piece.piece_type == chess.KING:
            return self.king_table[table_index]
        
        return 0
    
    def clear_cache(self):
        """Clear evaluation cache"""
        self.evaluation_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / max(total_requests, 1)) * 100
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.evaluation_cache)
        }


# ==============================================================================
# V11 CONSOLIDATED MODULES: Dynamic Move Selection
# ==============================================================================

class V7P3RDynamicMoveSelector:
    """V11 ENHANCEMENT: Dynamic move selection based on search depth"""
    
    def __init__(self):
        self.depth_thresholds = {
            1: 50,   # Depth 1-2: Consider most moves
            2: 50,
            3: 25,   # Depth 3-4: Moderate pruning
            4: 18,   # Slightly less aggressive
            5: 12,   # Depth 5-6: Selective
            6: 8,    # Less aggressive for depth 6
            7: 5,    # Depth 7+: Very selective
            8: 3
        }
    
    def get_move_limit_for_depth(self, depth: int) -> int:
        """Get maximum number of moves to consider at given depth"""
        if depth <= 2:
            return self.depth_thresholds[1]
        elif depth <= 4:
            return self.depth_thresholds[min(depth, 4)]
        elif depth <= 6:
            return self.depth_thresholds[min(depth, 6)]
        else:
            return self.depth_thresholds[8]
    
    def should_prune_move(self, board: chess.Board, move: chess.Move, depth: int, move_index: int) -> bool:
        """Determine if a move should be pruned based on depth and position"""
        # Never prune in shallow depths
        if depth <= 2:
            return False
        
        # Get move limit for this depth
        move_limit = self.get_move_limit_for_depth(depth)
        
        # Always keep moves within the limit based on their ordering
        if move_index < move_limit:
            return False
        
        # For deeper searches, be very aggressive about pruning
        if depth >= 5:
            # Always keep captures and checks
            if board.is_capture(move) or board.gives_check(move):
                return False
            
            # Always keep moves that escape check
            if board.is_check():
                return False
            
            # Prune quiet moves beyond the limit
            return True
        
        # For medium depths (3-4), prune less aggressively
        if depth >= 3:
            # Keep tactical moves
            if board.is_capture(move) or board.gives_check(move):
                return False
            
            # Keep castling
            if board.is_castling(move):
                return False
            
            # Prune quiet moves beyond limit
            return move_index >= move_limit
        
        return False
    
    def filter_moves_by_depth(self, board: chess.Board, ordered_moves: List[chess.Move], depth: int) -> List[chess.Move]:
        """Filter moves based on search depth - more aggressive pruning at deeper levels"""
        if depth <= 2:
            return ordered_moves  # No pruning for shallow depths
        
        filtered_moves = []
        move_limit = self.get_move_limit_for_depth(depth)
        
        # Categorize moves for depth-based selection
        critical_moves = []
        tactical_moves = []
        positional_moves = []
        quiet_moves = []
        
        for move in ordered_moves:
            if board.is_check() or board.is_capture(move) or board.gives_check(move):
                critical_moves.append(move)
            elif board.is_castling(move) or self._is_promotion(move):
                tactical_moves.append(move)
            elif self._is_central_move(board, move) or self._is_development_move(board, move):
                positional_moves.append(move)
            else:
                quiet_moves.append(move)
        
        # Select moves based on depth
        if depth >= 6:
            # Depth 6+: Only critical moves
            filtered_moves = critical_moves[:move_limit]
        elif depth >= 5:
            # Depth 5: Critical + very few tactical
            filtered_moves = critical_moves[:int(move_limit * 0.8)]
            remaining = move_limit - len(filtered_moves)
            filtered_moves.extend(tactical_moves[:remaining])
        elif depth >= 4:
            # Depth 4: Critical + some tactical
            filtered_moves = critical_moves[:int(move_limit * 0.7)]
            remaining = move_limit - len(filtered_moves)
            filtered_moves.extend(tactical_moves[:remaining])
        else:
            # Depth 3: Critical + tactical + some positional
            filtered_moves = critical_moves[:int(move_limit * 0.6)]
            remaining = move_limit - len(filtered_moves)
            filtered_moves.extend(tactical_moves[:int(remaining * 0.7)])
            remaining = move_limit - len(filtered_moves)
            filtered_moves.extend(positional_moves[:remaining])
        
        # Ensure we have at least one move
        if not filtered_moves and ordered_moves:
            filtered_moves = [ordered_moves[0]]
        
        return filtered_moves
    
    def _is_promotion(self, move: chess.Move) -> bool:
        """Check if move is a promotion"""
        return move.promotion is not None
    
    def _is_central_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move involves central squares"""
        central_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        return move.to_square in central_squares or move.from_square in central_squares
    
    def _is_development_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move develops a piece"""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        
        # Check if piece is moving from back rank for first time
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            back_rank = 0 if piece.color == chess.BLACK else 7
            return chess.square_rank(move.from_square) == back_rank
        
        return False


# ==============================================================================
# V11 CONSOLIDATED MODULES: Lightweight Defense Analysis
# ==============================================================================

class V7P3RLightweightDefense:
    """Fast defensive analysis with performance safeguards"""
    
    def __init__(self):
        self.defense_cache = {}
        self.performance_stats = {
            'calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'avg_time_ms': 0.0
        }
        
        # Piece values for defense calculation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
    
    def quick_defensive_assessment(self, board: chess.Board) -> float:
        """Fast defensive assessment (target: <2ms)"""
        start_time = time.time()
        self.performance_stats['calls'] += 1
        
        # Create cache key
        cache_key = self._create_position_key(board)
        
        # Check cache first
        if cache_key in self.defense_cache:
            self.performance_stats['cache_hits'] += 1
            return self.defense_cache[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        
        try:
            defense_score = 0.0
            
            # Quick piece safety assessment
            defense_score += self._evaluate_piece_safety(board) * 0.4
            
            # Basic king shelter evaluation
            defense_score += self._evaluate_king_shelter(board) * 0.3
            
            # Simple piece coordination
            defense_score += self._evaluate_piece_coordination(board) * 0.3
            
            # Cache the result (with size limit)
            if len(self.defense_cache) < 1000:
                self.defense_cache[cache_key] = defense_score
            
            # Update performance stats
            elapsed_time = time.time() - start_time
            self.performance_stats['total_time'] += elapsed_time
            self.performance_stats['avg_time_ms'] = (
                self.performance_stats['total_time'] / self.performance_stats['calls'] * 1000
            )
            
            return defense_score
            
        except Exception as e:
            return 0.0
    
    def _create_position_key(self, board: chess.Board) -> str:
        """Create simplified position key for caching"""
        key_parts = []
        
        # Material count by piece type
        material_counts = {}
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            white_count = len(board.pieces(piece_type, True))
            black_count = len(board.pieces(piece_type, False))
            material_counts[piece_type] = (white_count, black_count)
        
        # King positions
        white_king = board.king(True)
        black_king = board.king(False)
        
        # Create stable cache key
        key_parts.append(f"mat_{material_counts}")
        key_parts.append(f"wk_{white_king}")
        key_parts.append(f"bk_{black_king}")
        key_parts.append(f"turn_{board.turn}")
        
        return "_".join(key_parts)
    
    def _evaluate_piece_safety(self, board: chess.Board) -> float:
        """Quick evaluation of piece safety"""
        safety_score = 0.0
        
        for color in [True, False]:
            color_multiplier = 1.0 if color == board.turn else -1.0
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == color:
                    # Check if piece is defended
                    if self._is_piece_defended(board, square, color):
                        safety_score += self.piece_values[piece.piece_type] * 0.01 * color_multiplier
                    
                    # Check if piece is attacked
                    if board.is_attacked_by(not color, square):
                        # Penalty for undefended attacked pieces
                        if not self._is_piece_defended(board, square, color):
                            safety_score -= self.piece_values[piece.piece_type] * 0.02 * color_multiplier
        
        return safety_score / 100.0
    
    def _is_piece_defended(self, board: chess.Board, square: int, color: bool) -> bool:
        """Quick check if piece is defended"""
        return board.is_attacked_by(color, square)
    
    def _evaluate_king_shelter(self, board: chess.Board) -> float:
        """Basic king shelter evaluation"""
        shelter_score = 0.0
        
        for color in [True, False]:
            color_multiplier = 1.0 if color == board.turn else -1.0
            king_square = board.king(color)
            
            if king_square is None:
                continue
            
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            
            # Check pawn shelter in front of king
            pawn_shelter = 0
            
            if color:  # White king
                shelter_squares = [
                    chess.square(f, king_rank + 1) for f in range(max(0, king_file-1), min(8, king_file+2))
                    if king_rank < 7
                ]
            else:  # Black king
                shelter_squares = [
                    chess.square(f, king_rank - 1) for f in range(max(0, king_file-1), min(8, king_file+2))
                    if king_rank > 0
                ]
            
            for square in shelter_squares:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    pawn_shelter += 1
            
            # Bonus for pawn shelter
            shelter_score += pawn_shelter * 0.2 * color_multiplier
            
            # Penalty for open files near king
            open_files_near_king = 0
            for file_offset in [-1, 0, 1]:
                check_file = king_file + file_offset
                if 0 <= check_file < 8:
                    file_has_pawn = False
                    for rank in range(8):
                        piece = board.piece_at(chess.square(check_file, rank))
                        if piece and piece.piece_type == chess.PAWN:
                            file_has_pawn = True
                            break
                    if not file_has_pawn:
                        open_files_near_king += 1
            
            shelter_score -= open_files_near_king * 0.1 * color_multiplier
        
        return shelter_score
    
    def _evaluate_piece_coordination(self, board: chess.Board) -> float:
        """Simple piece coordination evaluation"""
        coordination_score = 0.0
        
        for color in [True, False]:
            color_multiplier = 1.0 if color == board.turn else -1.0
            
            # Count pieces defending each other
            defended_pieces = 0
            total_pieces = 0
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == color and piece.piece_type != chess.KING:
                    total_pieces += 1
                    if board.is_attacked_by(color, square):
                        defended_pieces += 1
            
            if total_pieces > 0:
                coordination_ratio = defended_pieces / total_pieces
                coordination_score += coordination_ratio * 0.3 * color_multiplier
        
        return coordination_score
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for monitoring"""
        if self.performance_stats['calls'] > 0:
            cache_hit_rate = (self.performance_stats['cache_hits'] / 
                            self.performance_stats['calls'] * 100)
        else:
            cache_hit_rate = 0.0
        
        return {
            'calls': self.performance_stats['calls'],
            'cache_hit_rate': cache_hit_rate,
            'avg_time_ms': self.performance_stats['avg_time_ms'],
            'cache_size': len(self.defense_cache)
        }
    
    def clear_cache(self):
        """Clear defense cache to free memory"""
        self.defense_cache.clear()
    
    def reset_performance_stats(self):
        """Reset performance tracking"""
        self.performance_stats = {
            'calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'avg_time_ms': 0.0
        }


class PVTracker:
    """Tracks principal variation using predicted board states for instant move recognition"""
    
    def __init__(self):
        self.predicted_position_fen = None  # FEN of position we expect after opponent move
        self.next_our_move = None          # Our move to play if prediction hits
        self.remaining_pv_queue = []       # Remaining moves in PV [opp_move, our_move, opp_move, our_move, ...]
        self.following_pv = False          # Whether we're actively following PV
        self.pv_display_string = ""        # PV string for display purposes
        self.original_pv = []              # Store original PV for display
        
    def store_pv_from_search(self, starting_board: chess.Board, pv_moves: List[chess.Move]):
        """Store PV from search results, setting up for following"""
        self.original_pv = pv_moves.copy()  # Keep original for display
        
        if len(pv_moves) < 3:  # Need at least: our_move, opp_move, our_next_move
            self.clear()
            return
            
        # We're about to play pv_moves[0], so prepare for opponent response
        temp_board = starting_board.copy()
        temp_board.push(pv_moves[0])  # Make our first move
        
        if len(pv_moves) >= 2:
            # Predict position after opponent plays pv_moves[1]
            temp_board.push(pv_moves[1])  # Opponent's expected response
            self.predicted_position_fen = temp_board.fen()
            
            if len(pv_moves) >= 3:
                self.next_our_move = pv_moves[2]  # Our response to their response
                self.remaining_pv_queue = pv_moves[3:]  # Rest of PV
                self.pv_display_string = ' '.join(str(m) for m in pv_moves[2:])
                self.following_pv = True
            else:
                self.clear()
        else:
            self.clear()
    
    def clear(self):
        """Clear all PV following state"""
        self.predicted_position_fen = None
        self.next_our_move = None
        self.remaining_pv_queue = []
        self.following_pv = False
        self.pv_display_string = ""
        # Keep original_pv for display even after clearing following state
    
    def check_position_for_instant_move(self, current_board: chess.Board) -> Optional[chess.Move]:
        """Check if current position matches prediction - if so, return instant move"""
        if not self.following_pv or not self.predicted_position_fen:
            return None
            
        current_fen = current_board.fen()
        if current_fen == self.predicted_position_fen:
            # Position matches prediction - return instant move
            move_to_play = self.next_our_move
            
            # Clean UCI output for PV following
            remaining_pv_str = self.pv_display_string if self.pv_display_string else str(move_to_play)
            print(f"info depth PV score cp 0 nodes 0 time 0 pv {remaining_pv_str}")
            
            # Set up for next prediction if we have more moves
            self._setup_next_prediction(current_board)
            
            return move_to_play
        else:
            # Position doesn't match - opponent broke PV, clear following
            self.clear()
            return None
    
    def _setup_next_prediction(self, current_board: chess.Board):
        """Set up prediction for next opponent move"""
        if len(self.remaining_pv_queue) < 2:  # Need at least opp_move, our_move
            self.clear()
            return
            
        # Make our move that we're about to play
        temp_board = current_board.copy()
        if self.next_our_move:  # Safety check
            temp_board.push(self.next_our_move)
        
        # Predict position after next opponent move
        next_opp_move = self.remaining_pv_queue[0]
        temp_board.push(next_opp_move)
        
        # Set up for next iteration
        self.predicted_position_fen = temp_board.fen()
        self.next_our_move = self.remaining_pv_queue[1] if len(self.remaining_pv_queue) >= 2 else None
        self.remaining_pv_queue = self.remaining_pv_queue[2:]  # Remove used moves
        self.pv_display_string = ' '.join(str(m) for m in [self.next_our_move] + self.remaining_pv_queue) if self.next_our_move else ""
        
        if not self.next_our_move:
            self.clear()  # No more moves to follow


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
        """Store a killer move at the given depth"""
        if move not in self.killers[depth]:
            self.killers[depth].insert(0, move)
            if len(self.killers[depth]) > 2:
                self.killers[depth].pop()
    
    def get_killers(self, depth: int) -> List[chess.Move]:
        """Get killer moves for the given depth"""
        return self.killers[depth]
    
    def is_killer(self, move: chess.Move, depth: int) -> bool:
        """Check if a move is a killer at the given depth"""
        return move in self.killers[depth]


class HistoryHeuristic:
    """History heuristic for move ordering"""
    def __init__(self):
        self.history: Dict[str, int] = defaultdict(int)
    
    def update_history(self, move: chess.Move, depth: int):
        """Update history score for a move"""
        move_key = f"{move.from_square}-{move.to_square}"
        self.history[move_key] += depth * depth
    
    def get_history_score(self, move: chess.Move) -> int:
        """Get history score for a move"""
        move_key = f"{move.from_square}-{move.to_square}"
        return self.history[move_key]


class ZobristHashing:
    """Zobrist hashing for transposition table"""
    def __init__(self):
        random.seed(12345)  # Deterministic for reproducibility
        self.piece_square_table = {}
        self.side_to_move = random.getrandbits(64)
        
        # Generate random numbers for each piece on each square
        for square in range(64):
            for piece_type in range(1, 7):  # PAWN to KING
                for color in [chess.WHITE, chess.BLACK]:
                    key = (square, piece_type, color)
                    self.piece_square_table[key] = random.getrandbits(64)
    
    def hash_position(self, board: chess.Board) -> int:
        """Generate Zobrist hash for the position"""
        hash_value = 0
        
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                key = (square, piece.piece_type, piece.color)
                hash_value ^= self.piece_square_table[key]
        
        if board.turn == chess.BLACK:
            hash_value ^= self.side_to_move
            
        return hash_value


# ==============================================================================
# V11 CONSOLIDATED MODULES: Adaptive Evaluation Framework
# ==============================================================================

class V7P3RAdaptiveEvaluation:
    """Adaptive evaluation system that adapts based on position posture"""
    
    def __init__(self, posture_assessor: V7P3RPostureAssessment):
        self.posture_assessor = posture_assessor
        
        # Piece values for basic material evaluation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 10000
        }
        
        # Evaluation component weights for different postures
        self.evaluation_weights = {
            GamePosture.EMERGENCY: {
                'material': 1.0,
                'king_safety': 2.0,
                'piece_safety': 1.5,
                'mobility': 0.3,
                'positional': 0.1,
                'development': 0.2,
                'strategic': 0.0
            },
            GamePosture.DEFENSIVE: {
                'material': 1.0,
                'king_safety': 1.5,
                'piece_safety': 1.3,
                'mobility': 0.6,
                'positional': 0.4,
                'development': 0.5,
                'strategic': 0.2
            },
            GamePosture.BALANCED: {
                'material': 1.0,
                'king_safety': 1.0,
                'piece_safety': 1.0,
                'mobility': 1.0,
                'positional': 1.0,
                'development': 0.8,
                'strategic': 0.6
            },
            GamePosture.OFFENSIVE: {
                'material': 1.0,
                'king_safety': 0.8,
                'piece_safety': 0.7,
                'mobility': 1.2,
                'positional': 1.1,
                'development': 1.0,
                'strategic': 1.0
            }
        }
        
        # Performance tracking
        self.evaluation_stats = {
            'calls': 0,
            'posture_breakdown': {
                'emergency': 0,
                'defensive': 0,
                'balanced': 0,
                'offensive': 0
            },
            'total_time': 0.0,
            'component_times': {
                'posture_assessment': 0.0,
                'material': 0.0,
                'king_safety': 0.0,
                'piece_safety': 0.0,
                'mobility': 0.0,
                'positional': 0.0,
                'strategic': 0.0
            }
        }
    
    def evaluate_position(self, board: chess.Board, depth: int = 0) -> float:
        """Main evaluation function that adapts based on position posture"""
        start_time = time.time()
        
        self.evaluation_stats['calls'] += 1
        
        # Step 1: Assess position posture
        posture_start = time.time()
        volatility, posture = self.posture_assessor.assess_position_posture(board)
        self.evaluation_stats['component_times']['posture_assessment'] += time.time() - posture_start
        
        # Track posture statistics
        self.evaluation_stats['posture_breakdown'][posture.value] += 1
        
        # Step 2: Get evaluation weights for this posture
        weights = self.evaluation_weights[posture]
        
        # Step 3: Run evaluation components based on weights
        total_score = 0.0
        
        # Material evaluation (always run)
        if weights['material'] > 0:
            component_start = time.time()
            material_score = self._evaluate_material(board) * weights['material']
            total_score += material_score
            self.evaluation_stats['component_times']['material'] += time.time() - component_start
        
        # King safety
        if weights['king_safety'] > 0:
            component_start = time.time()
            king_safety_score = self._evaluate_king_safety(board) * weights['king_safety']
            total_score += king_safety_score
            self.evaluation_stats['component_times']['king_safety'] += time.time() - component_start
        
        # Piece safety
        if weights['piece_safety'] > 0:
            component_start = time.time()
            piece_safety_score = self._evaluate_piece_safety(board) * weights['piece_safety']
            total_score += piece_safety_score
            self.evaluation_stats['component_times']['piece_safety'] += time.time() - component_start
        
        # Mobility
        if weights['mobility'] > 0:
            component_start = time.time()
            mobility_score = self._evaluate_mobility(board) * weights['mobility']
            total_score += mobility_score
            self.evaluation_stats['component_times']['mobility'] += time.time() - component_start
        
        # Positional factors
        if weights['positional'] > 0:
            component_start = time.time()
            positional_score = self._evaluate_positional(board) * weights['positional']
            total_score += positional_score
            self.evaluation_stats['component_times']['positional'] += time.time() - component_start
        
        # Strategic analysis
        if weights['strategic'] > 0:
            component_start = time.time()
            strategic_score = self._evaluate_strategic(board) * weights['strategic']
            total_score += strategic_score
            self.evaluation_stats['component_times']['strategic'] += time.time() - component_start
        
        # Posture-specific bonuses
        posture_bonus = self._get_posture_bonus(board, posture, volatility)
        total_score += posture_bonus
        
        self.evaluation_stats['total_time'] += time.time() - start_time
        
        return total_score
    
    def _evaluate_material(self, board: chess.Board) -> float:
        """Basic material evaluation"""
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color:
                    white_material += value
                else:
                    black_material += value
        
        # Return from current player's perspective
        if board.turn:
            return (white_material - black_material) / 100.0
        else:
            return (black_material - white_material) / 100.0
    
    def _evaluate_king_safety(self, board: chess.Board) -> float:
        """Evaluate king safety for current player"""
        score = 0.0
        current_color = board.turn
        king_square = board.king(current_color)
        
        if king_square is None:
            return -100.0
        
        # Count attackers on king
        attackers = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != current_color:
                if board.is_attacked_by(not current_color, king_square):
                    attackers += 1
        
        score -= attackers * 0.5
        
        # King shelter evaluation
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Check for pawn shelter
        shelter_bonus = 0
        for file_offset in [-1, 0, 1]:
            for rank_offset in [1, 2]:
                new_file = king_file + file_offset
                new_rank = king_rank + (rank_offset if current_color else -rank_offset)
                
                if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
                    square = chess.square(new_file, new_rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == current_color:
                        shelter_bonus += 0.1
        
        score += shelter_bonus
        
        return score
    
    def _evaluate_piece_safety(self, board: chess.Board) -> float:
        """Evaluate safety of our pieces"""
        score = 0.0
        current_color = board.turn
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == current_color:
                piece_value = self.piece_values[piece.piece_type] / 1000.0
                
                # Penalty for undefended pieces
                if board.is_attacked_by(not current_color, square):
                    if not board.is_attacked_by(current_color, square):
                        score -= piece_value * 0.5
                    else:
                        score -= piece_value * 0.1
                
                # Bonus for defended pieces
                elif board.is_attacked_by(current_color, square):
                    score += piece_value * 0.05
        
        return score
    
    def _evaluate_mobility(self, board: chess.Board) -> float:
        """Evaluate piece mobility and control"""
        our_mobility = len(list(board.legal_moves))
        
        # Switch sides to count opponent mobility
        board.push(chess.Move.null())
        try:
            their_mobility = len(list(board.legal_moves))
        except:
            their_mobility = 0
        finally:
            board.pop()
        
        # Mobility advantage
        mobility_advantage = our_mobility - their_mobility
        return mobility_advantage / 100.0
    
    def _evaluate_positional(self, board: chess.Board) -> float:
        """Basic positional evaluation"""
        score = 0.0
        current_color = board.turn
        
        # Center control
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        for square in center_squares:
            if board.is_attacked_by(current_color, square):
                score += 0.1
            if board.is_attacked_by(not current_color, square):
                score -= 0.1
        
        # Development bonus
        back_rank = 0 if current_color else 7
        developed_pieces = 0
        total_pieces = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == current_color:
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.QUEEN]:
                    total_pieces += 1
                    if chess.square_rank(square) != back_rank:
                        developed_pieces += 1
        
        if total_pieces > 0:
            development_ratio = developed_pieces / total_pieces
            score += development_ratio * 0.2
        
        return score
    
    def _evaluate_strategic(self, board: chess.Board) -> float:
        """Strategic pattern evaluation (simplified)"""
        score = 0.0
        
        # Pawn structure basics
        current_color = board.turn
        our_pawns = board.pieces(chess.PAWN, current_color)
        their_pawns = board.pieces(chess.PAWN, not current_color)
        
        # Passed pawn bonus
        for square in our_pawns:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            # Check if pawn is passed (simplified)
            blocked = False
            for enemy_square in their_pawns:
                enemy_file = chess.square_file(enemy_square)
                enemy_rank = chess.square_rank(enemy_square)
                
                if abs(enemy_file - file) <= 1:
                    if current_color:
                        if enemy_rank > rank:
                            blocked = True
                            break
                    else:
                        if enemy_rank < rank:
                            blocked = True
                            break
            
            if not blocked:
                distance_to_promotion = 7 - rank if current_color else rank
                score += (8 - distance_to_promotion) * 0.05
        
        return score
    
    def _get_posture_bonus(self, board: chess.Board, posture: GamePosture, volatility: PositionVolatility) -> float:
        """Get posture-specific evaluation bonus"""
        bonus = 0.0
        
        if posture == GamePosture.EMERGENCY:
            bonus -= 0.5
        elif posture == GamePosture.DEFENSIVE:
            bonus -= 0.2
        elif posture == GamePosture.OFFENSIVE:
            bonus += 0.3
        
        # Volatility adjustment
        if volatility == PositionVolatility.CRITICAL:
            bonus -= 0.3
        elif volatility == PositionVolatility.STABLE:
            bonus += 0.1
        
        return bonus
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get performance and usage statistics"""
        stats = self.evaluation_stats.copy()
        
        if stats['calls'] > 0:
            # Add average times
            stats['avg_total_time'] = stats['total_time'] / stats['calls']
            stats['avg_component_times'] = {}
            for component, total_time in stats['component_times'].items():
                stats['avg_component_times'][component] = total_time / stats['calls']
            
            # Add posture percentages
            stats['posture_percentages'] = {}
            for posture, count in stats['posture_breakdown'].items():
                stats['posture_percentages'][posture] = (count / stats['calls']) * 100
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.evaluation_stats = {
            'calls': 0,
            'posture_breakdown': {
                'emergency': 0,
                'defensive': 0,
                'balanced': 0,
                'offensive': 0
            },
            'total_time': 0.0,
            'component_times': {
                'posture_assessment': 0.0,
                'material': 0.0,
                'king_safety': 0.0,
                'piece_safety': 0.0,
                'mobility': 0.0,
                'positional': 0.0,
                'strategic': 0.0
            }
        }


# ==============================================================================
# V11 CONSOLIDATED MODULES: Adaptive Move Ordering
# ==============================================================================

class V7P3RAdaptiveMoveOrdering:
    """Adaptive move ordering system that prioritizes moves based on position posture"""
    
    def __init__(self, posture_assessor: V7P3RPostureAssessment):
        self.posture_assessor = posture_assessor
        
        # Move type priorities for different postures
        self.posture_priorities = {
            GamePosture.EMERGENCY: {
                'escape_moves': 1000,
                'blocking_moves': 900,
                'defensive_captures': 800,
                'defensive_moves': 700,
                'other_captures': 200,
                'other_moves': 100
            },
            GamePosture.DEFENSIVE: {
                'defensive_captures': 900,
                'defensive_moves': 800,
                'escape_moves': 700,
                'development': 600,
                'good_captures': 500,
                'other_captures': 300,
                'other_moves': 200
            },
            GamePosture.BALANCED: {
                'good_captures': 900,
                'tactical_moves': 800,
                'development': 700,
                'defensive_moves': 600,
                'other_captures': 500,
                'castling': 400,
                'other_moves': 300
            },
            GamePosture.OFFENSIVE: {
                'tactical_moves': 1000,
                'good_captures': 900,
                'attacking_moves': 800,
                'development': 700,
                'other_captures': 600,
                'defensive_moves': 400,
                'other_moves': 300
            }
        }
        
        # Piece values for capture evaluation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 10000
        }
    
    def order_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[Tuple[chess.Move, int]]:
        """Order moves based on current position posture"""
        if not moves:
            return []
        
        # Assess current position posture
        volatility, posture = self.posture_assessor.assess_position_posture(board)
        
        # Get priority scheme for this posture
        priorities = self.posture_priorities[posture]
        
        # Score each move
        scored_moves = []
        for move in moves:
            move_type, base_score = self._classify_move(board, move, posture, volatility)
            priority = priorities.get(move_type, 100)
            
            # Final score combines priority and base score
            final_score = priority + base_score
            scored_moves.append((move, final_score))
        
        # Sort by score (highest first)
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        return scored_moves
    
    def _classify_move(self, board: chess.Board, move: chess.Move, 
                      posture: GamePosture, volatility: PositionVolatility) -> Tuple[str, int]:
        """Classify a move and assign a base score"""
        base_score = 0
        
        # Check basic move properties
        is_capture = board.is_capture(move)
        is_check = board.gives_check(move)
        piece = board.piece_at(move.from_square)
        
        if piece is None:
            return ('other_moves', 0)
        
        # Handle captures
        if is_capture:
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                capture_value = self.piece_values[captured_piece.piece_type] - self.piece_values[piece.piece_type]
                
                # Check if this capture defends us
                if self._is_defensive_capture(board, move):
                    base_score = capture_value + 50
                    return ('defensive_captures', base_score)
                elif capture_value > 0:
                    base_score = capture_value
                    return ('good_captures', base_score)
                else:
                    base_score = capture_value
                    return ('other_captures', base_score)
        
        # Check for special move types based on posture
        if posture in [GamePosture.EMERGENCY, GamePosture.DEFENSIVE]:
            if self._is_escape_move(board, move):
                return ('escape_moves', base_score + 100)
            elif self._is_blocking_move(board, move):
                return ('blocking_moves', base_score + 80)
            elif self._is_defensive_move(board, move):
                return ('defensive_moves', base_score + 60)
        
        elif posture == GamePosture.OFFENSIVE:
            if self._is_tactical_move(board, move):
                return ('tactical_moves', base_score + 100)
            elif self._creates_threats(board, move):
                return ('attacking_moves', base_score + 80)
        
        # Check for development moves
        if self._is_development_move(board, move):
            return ('development', base_score + 40)
        
        # Check for castling
        if board.is_castling(move):
            return ('castling', base_score + 30)
        
        # Default classification
        return ('other_moves', base_score)
    
    def _is_defensive_capture(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if capture removes a piece that was attacking us"""
        target_square = move.to_square
        
        # Check if the captured piece was attacking any of our pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                if board.is_attacked_by(not board.turn, square):
                    # Check if the piece being captured was one of the attackers
                    board.push(move)
                    still_attacked = board.is_attacked_by(not board.turn, square)
                    board.pop()
                    
                    if not still_attacked:
                        return True
        
        return False
    
    def _is_escape_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move escapes a piece from attack"""
        from_square = move.from_square
        
        # Check if the piece was under attack and moves to safety
        if board.is_attacked_by(not board.turn, from_square):
            board.push(move)
            safe = not board.is_attacked_by(not board.turn, move.to_square)
            board.pop()
            return safe
        
        return False
    
    def _is_blocking_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move blocks an attack on our pieces"""
        board.push(move)
        
        # Count how many of our pieces are under attack after the move
        our_attacked_after = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                if board.is_attacked_by(not board.turn, square):
                    our_attacked_after += 1
        
        board.pop()
        
        # Count how many were under attack before
        our_attacked_before = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                if board.is_attacked_by(not board.turn, square):
                    our_attacked_before += 1
        
        # If fewer pieces are attacked after the move, it might be blocking
        return our_attacked_after < our_attacked_before
    
    def _is_defensive_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move defends one of our pieces"""
        to_square = move.to_square
        
        # Check if the destination square defends any of our pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                # See if this move would defend that piece
                board.push(move)
                defended = board.is_attacked_by(board.turn, square)
                board.pop()
                
                if defended and board.is_attacked_by(not board.turn, square):
                    return True
        
        return False
    
    def _is_tactical_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move creates tactical threats"""
        board.push(move)
        
        # Count enemy pieces under attack after the move
        enemy_attacked = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                if board.is_attacked_by(board.turn, square):
                    enemy_attacked += 1
        
        board.pop()
        
        # If we attack 2+ enemy pieces, it's tactical
        return enemy_attacked >= 2
    
    def _creates_threats(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move creates new threats"""
        # Count threats before move
        threats_before = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                if board.is_attacked_by(board.turn, square):
                    threats_before += 1
        
        # Count threats after move
        board.push(move)
        threats_after = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                if board.is_attacked_by(board.turn, square):
                    threats_after += 1
        board.pop()
        
        return threats_after > threats_before
    
    def _is_development_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move develops a piece"""
        piece = board.piece_at(move.from_square)
        if piece is None:
            return False
        
        # Check if piece is moving from back rank (development)
        back_rank = 0 if piece.color else 7
        from_rank = chess.square_rank(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        
        # Simple heuristic: moving from back rank towards center
        if from_rank == back_rank and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            return abs(to_rank - 3.5) < abs(from_rank - 3.5)
        
        return False
    
    def get_best_moves(self, board: chess.Board, count: int = 10) -> List[chess.Move]:
        """Get the best moves according to current posture assessment"""
        legal_moves = list(board.legal_moves)
        scored_moves = self.order_moves(board, legal_moves)
        
        # Return top 'count' moves
        return [move for move, score in scored_moves[:count]]
    
    def get_move_classification(self, board: chess.Board, move: chess.Move) -> str:
        """Get the classification of a specific move for debugging"""
        volatility, posture = self.posture_assessor.assess_position_posture(board)
        move_type, base_score = self._classify_move(board, move, posture, volatility)
        return f"{move_type} (posture: {posture.value}, score: {base_score})"


class V7P3REngine:
    """V7P3R Chess Engine v9.5 - Bitboard-powered version"""
    
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
        
        # Evaluation components - V10 BITBOARD POWERED + V11 PHASE 3A ADVANCED + V10.9 PHASE 3B TIME-ADAPTIVE TACTICAL
        self.bitboard_evaluator = V7P3RScoringCalculationBitboard(self.piece_values)
        self.advanced_pawn_evaluator = V7P3RAdvancedPawnEvaluator()  # V11 PHASE 3A
        self.king_safety_evaluator = V7P3RKingSafetyEvaluator()      # V11 PHASE 3A
        self.tactical_pattern_detector = TimeControlAdaptiveTacticalDetector()  # V10.9 PHASE 3B: TIME-ADAPTIVE TACTICAL PATTERNS
        
        # Simple evaluation cache for speed
        self.evaluation_cache = {}  # position_hash -> evaluation
        
        # Advanced search infrastructure
        self.transposition_table: Dict[int, TranspositionEntry] = {}
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        self.zobrist = ZobristHashing()
        
        # V11 PHASE 2: Nudge System Integration
        self.nudge_database = {}
        self.nudge_stats = {
            'hits': 0,
            'misses': 0,
            'moves_boosted': 0,
            'positions_matched': 0,
            'instant_moves': 0,      # V11 PHASE 2 ENHANCEMENT
            'instant_time_saved': 0.0  # V11 PHASE 2 ENHANCEMENT
        }
        
        # V11 PHASE 2 ENHANCEMENT: Nudge threshold configuration
        self.nudge_instant_config = {
            'min_frequency': 8,        # Move must be played at least 8 times
            'min_eval': 0.4,          # Move must have eval improvement >= 0.4
            'confidence_threshold': 12.0  # Combined confidence score threshold
        }
        
        self._load_nudge_database()
        
        # V11 PHASE 2: Enhanced Strategic Database
        self.strategic_database = V7P3RStrategicDatabase()
        
        # V11 PHASE 3B: New Adaptive Evaluation System
        self.posture_assessor = V7P3RPostureAssessment()
        self.adaptive_evaluator = V7P3RAdaptiveEvaluation(self.posture_assessor)
        self.adaptive_move_orderer = V7P3RAdaptiveMoveOrdering(self.posture_assessor)
        
        # V11 PHASE 3A: Lightweight Defense System
        self.lightweight_defense = V7P3RLightweightDefense()
        
        # V11 PERFORMANCE: Fast evaluator for non-critical nodes
        self.fast_evaluator = V7P3RFastEvaluator()
        
        # V11 PERFORMANCE: Dynamic move selector for depth-based pruning
        self.dynamic_move_selector = V7P3RDynamicMoveSelector()
        
        # Configuration
        self.max_tt_entries = 50000  # Reasonable size for testing
        
        # V10.9 PHASE 3B: Time control tracking for tactical pattern detector
        self.current_time_remaining_ms = 600000  # Default 10 minutes
        self.current_moves_played = 0
        self.search_start_time = 0.0
        
        # Performance monitoring
        self.search_stats = {
            'nodes_per_second': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'tt_hits': 0,
            'tt_stores': 0,
            'killer_hits': 0,
            'nudge_hits': 0,        # V11 PHASE 2
            'nudge_positions': 0,   # V11 PHASE 2
        }
        
        # PV Following System - V10 OPTIMIZATION
        self.pv_tracker = PVTracker()
        
        # V11.1 SIMPLIFIED SYSTEMS FOR PERFORMANCE RECOVERY
        self.simple_move_orderer = V7P3RSimpleMoveOrderer()  # Replace complex move ordering
        
        # V11 PHASE 1: Advanced Time Management System
        self.time_manager = V7P3RSimpleTimeManager(base_time=300.0, increment=3.0)  # V11.1 SIMPLIFIED TIME CONTROL
    
    def search(self, board: chess.Board, time_limit: float = 3.0, depth: Optional[int] = None, 
               alpha: float = -99999, beta: float = 99999, is_root: bool = True) -> chess.Move:
        """
        UNIFIED SEARCH - Single function with ALL advanced features:
        - Iterative deepening with stable best move handling (root level)
        - Alpha-beta pruning with negamax framework (recursive level)  
        - Transposition table with Zobrist hashing
        - Killer moves and history heuristic
        - Advanced move ordering with tactical detection
        - Proper time management with periodic checks
        - Full PV extraction and following
        - Quiescence search for tactical stability
        """
        
        # ROOT LEVEL: Iterative deepening with time management
        if is_root:
            self.nodes_searched = 0
            self.search_start_time = time.time()
            
            # V10.9 PHASE 3B: Update game state for tactical pattern detector
            self.current_moves_played = len(board.move_stack)
            
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return chess.Move.null()

            # PV FOLLOWING OPTIMIZATION - check if current position triggers instant move
            pv_move = self.pv_tracker.check_position_for_instant_move(board)
            if pv_move:
                return pv_move
            
            # V11 PHASE 2 ENHANCEMENT: Check for instant nudge moves (high confidence)
            instant_nudge_move = self._check_instant_nudge_move(board)
            if instant_nudge_move:
                # Calculate time saved
                time_saved = time_limit * 0.8  # Estimate time that would have been used
                self.nudge_stats['instant_time_saved'] += time_saved
                
                # Output instant move info
                print(f"info depth NUDGE score cp 50 nodes 0 time 0 pv {instant_nudge_move}")
                print(f"info string Instant nudge move: {instant_nudge_move} (high confidence)")
                
                return instant_nudge_move
            
            # V11 PHASE 1: Advanced time management with complexity analysis
            time_remaining = time_limit  # For now, assume time_limit is our remaining time
            allocated_time, target_depth = self.time_manager.calculate_time_allocation(board, time_remaining)
            
            # Update time control info if available
            self.time_manager.update_time_info(time_remaining, len(board.move_stack))
            
            target_time = allocated_time
            max_time = min(allocated_time * 1.5, time_limit * 0.9)  # Allow 50% overage but cap at 90% of limit
            
            # Iterative deepening
            best_move = legal_moves[0]
            best_score = -99999
            
            for current_depth in range(1, self.default_depth + 1):
                iteration_start = time.time()
                
                # V11 ENHANCEMENT: Improved time checking with adaptive limits
                elapsed = time.time() - self.search_start_time
                if elapsed > target_time:
                    break
                
                # Predict if next iteration will exceed max time
                if current_depth > 1:
                    last_iteration_time = time.time() - iteration_start
                    if elapsed + (last_iteration_time * 2) > max_time:
                        break
                
                try:
                    # Store previous best in case iteration fails
                    previous_best = best_move
                    previous_score = best_score
                    
                    # Call recursive search for this depth
                    score, move = self._recursive_search(board, current_depth, -99999, 99999, time_limit)
                    
                    # Update best move if we got a valid result
                    if move and move != chess.Move.null():
                        best_move = move
                        best_score = score
                        
                        elapsed_ms = int((time.time() - self.search_start_time) * 1000)
                        nps = int(self.nodes_searched / max(elapsed_ms / 1000, 0.001))
                        
                        # Extract and display PV
                        pv_line = self._extract_pv(board, current_depth)
                        pv_string = " ".join([str(m) for m in pv_line])
                        
                        # Store PV for following optimization
                        if current_depth >= 4 and len(pv_line) >= 3:
                            self.pv_tracker.store_pv_from_search(board, pv_line)
                        
                        print(f"info depth {current_depth} score cp {int(score)} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {pv_string}")
                        sys.stdout.flush()
                    else:
                        # Restore previous best if iteration failed
                        best_move = previous_best
                        best_score = previous_score
                    
                    # V11 ENHANCEMENT: Better time management for next iteration
                    elapsed = time.time() - self.search_start_time
                    iteration_time = time.time() - iteration_start
                    
                    if elapsed > target_time:
                        break
                    elif elapsed + (iteration_time * 1.5) > max_time:
                        break  # Don't start next iteration if likely to exceed max time
                        
                except Exception as e:
                    print(f"info string Search interrupted at depth {current_depth}: {e}")
                    break
                    
            return best_move
        
        # This should never be called directly with is_root=False from external code
        else:
            # Fallback - call the recursive search method
            score, move = self._recursive_search(board, depth or 1, alpha, beta, time_limit)
            return move if move else chess.Move.null()
    
    def _recursive_search(self, board: chess.Board, search_depth: int, alpha: float, beta: float, time_limit: float) -> Tuple[float, Optional[chess.Move]]:
        """
        Recursive alpha-beta search with all advanced features
        Returns (score, best_move) tuple
        """
        self.nodes_searched += 1
        
        # CRITICAL: Time checking during recursive search to prevent timeouts
        if hasattr(self, 'search_start_time') and self.nodes_searched % 1000 == 0:
            elapsed = time.time() - self.search_start_time
            if elapsed > time_limit:
                # Emergency return with current best evaluation
                return self._evaluate_position(board), None
        
        # 1. TRANSPOSITION TABLE PROBE
        tt_hit, tt_score, tt_move = self._probe_transposition_table(board, search_depth, int(alpha), int(beta))
        if tt_hit:
            return float(tt_score), tt_move
        
        # 2. TERMINAL CONDITIONS
        if search_depth == 0:
            # Enter quiescence search for tactical stability
            score = self._quiescence_search(board, alpha, beta, 4)
            return score, None
            
        if board.is_game_over():
            if board.is_checkmate():
                score = -29000.0 + (self.default_depth - search_depth)  # Prefer quicker mates
            else:
                score = 0.0  # Stalemate
            return score, None
        
        # 3. NULL MOVE PRUNING
        if (search_depth >= 3 and not board.is_check() and 
            self._has_non_pawn_pieces(board) and beta - alpha > 1):
            
            board.turn = not board.turn
            null_score, _ = self._recursive_search(board, search_depth - 2, -beta, -beta + 1, time_limit)
            null_score = -null_score
            board.turn = not board.turn
            
            if null_score >= beta:
                return null_score, None
        
        # 4. MOVE GENERATION AND ORDERING
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0, None
        
        ordered_moves = self._order_moves_advanced(board, legal_moves, search_depth, tt_move)
        
        # V11 PERFORMANCE: Dynamic move pruning based on search depth
        # Progressively reduce move count as depth increases for speed
        if search_depth >= 3:
            ordered_moves = self.dynamic_move_selector.filter_moves_by_depth(board, ordered_moves, search_depth)
        
        # 5. MAIN SEARCH LOOP (NEGAMAX WITH ALPHA-BETA)
        best_score = -99999.0
        best_move = None
        original_alpha = alpha
        moves_searched = 0
        
        for move in ordered_moves:
            board.push(move)
            
            # V11 ENHANCEMENT: Enhanced Late Move Reduction
            reduction = self._calculate_lmr_reduction(move, moves_searched, search_depth, board)
            
            # Search with possible reduction
            if reduction > 0:
                score, _ = self._recursive_search(board, search_depth - 1 - reduction, -beta, -alpha, time_limit)
                score = -score
                
                # Re-search at full depth if reduced search failed high
                if score > alpha:
                    score, _ = self._recursive_search(board, search_depth - 1, -beta, -alpha, time_limit)
                    score = -score
            else:
                score, _ = self._recursive_search(board, search_depth - 1, -beta, -alpha, time_limit)
                score = -score
            
            board.pop()
            moves_searched += 1
            
            # Update best move
            if best_move is None or score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            if alpha >= beta:
                # Beta cutoff - update heuristics
                if not board.is_capture(move):
                    self.killer_moves.store_killer(move, search_depth)
                    self.history_heuristic.update_history(move, search_depth)
                    self.search_stats['killer_hits'] += 1
                break
        
        # 7. TRANSPOSITION TABLE STORE
        self._store_transposition_table(board, search_depth, int(best_score), best_move, int(original_alpha), int(beta))
        
        return best_score, best_move
    
    def _order_moves_advanced(self, board: chess.Board, moves: List[chess.Move], depth: int, 
                              tt_move: Optional[chess.Move] = None) -> List[chess.Move]:
        """V11.1 SIMPLIFIED: Use basic tactical move ordering for reliable performance"""
        return self.simple_move_orderer.order_moves(board, moves, depth, tt_move)
        quiet_moves = []
        tactical_moves = []  # NEW: Bitboard tactical moves
        nudge_moves = []     # V11 PHASE 2: Nudge system moves
        tt_moves = []
        
        killer_set = set(self.killer_moves.get_killers(depth))
        
        # Check if current position has nudge data
        position_has_nudges = self._get_position_key(board) in self.nudge_database
        if position_has_nudges:
            self.nudge_stats['positions_matched'] += 1
        
        for move in moves:
            # Calculate nudge bonus for this move (V11 PHASE 2)
            nudge_bonus = self._get_nudge_bonus(board, move)
            
            # 1. Transposition table move (highest priority)
            if tt_move and move == tt_move:
                tt_moves.append(move)
            
            # 2. Nudge moves (second highest priority - V11 PHASE 2)
            elif nudge_bonus > 0:
                nudge_moves.append((nudge_bonus, move))
            
            # 3. Captures (will be sorted by MVV-LVA)
            elif board.is_capture(move):
                victim = board.piece_at(move.to_square)
                victim_value = self.piece_values.get(victim.piece_type, 0) if victim else 0
                attacker = board.piece_at(move.from_square)
                attacker_value = self.piece_values.get(attacker.piece_type, 0) if attacker else 0
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                mvv_lva_score = victim_value * 100 - attacker_value
                
                # Add tactical bonus using bitboards
                tactical_bonus = self._detect_bitboard_tactics(board, move)
                total_score = mvv_lva_score + tactical_bonus
                
                captures.append((total_score, move))
            
            # 4. Checks (high priority for tactical play)
            elif board.gives_check(move):
                # Add tactical bonus for checking moves too
                tactical_bonus = self._detect_bitboard_tactics(board, move)
                checks.append((tactical_bonus, move))
            
            # 5. Killer moves
            elif move in killer_set:
                killers.append(move)
                self.search_stats['killer_hits'] += 1
            
            # 6. Check for tactical patterns in quiet moves
            else:
                history_score = self.history_heuristic.get_history_score(move)
                tactical_bonus = self._detect_bitboard_tactics(board, move)
                
                if tactical_bonus > 20.0:  # Significant tactical move
                    tactical_moves.append((tactical_bonus + history_score, move))
                else:
                    quiet_moves.append((history_score, move))
        
        # Sort all move categories by their scores
        captures.sort(key=lambda x: x[0], reverse=True)
        checks.sort(key=lambda x: x[0], reverse=True)
        tactical_moves.sort(key=lambda x: x[0], reverse=True)
        quiet_moves.sort(key=lambda x: x[0], reverse=True)
        nudge_moves.sort(key=lambda x: x[0], reverse=True)  # V11 PHASE 2
        
        # Combine in V11 PHASE 2 ENHANCED order
        ordered = []
        ordered.extend(tt_moves)  # TT move first
        ordered.extend([move for _, move in nudge_moves])  # Then nudge moves (V11 PHASE 2)
        ordered.extend([move for _, move in captures])  # Then captures (with tactical bonus)
        ordered.extend([move for _, move in checks])  # Then checks (with tactical bonus)
        ordered.extend([move for _, move in tactical_moves])  # Then tactical patterns
        ordered.extend(killers)  # Then killers
        ordered.extend([move for _, move in quiet_moves])  # Then quiet moves
        
        return ordered
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """V11 PERFORMANCE: Optimized evaluation with fast path for most nodes"""
        # Create cache key
        cache_key = board.fen()
        
        if cache_key in self.evaluation_cache:
            self.search_stats['cache_hits'] += 1
            return self.evaluation_cache[cache_key]
        
        self.search_stats['cache_misses'] += 1
        
        # V11 PERFORMANCE OPTIMIZATION: Drastically limit adaptive evaluation usage
        # Only use expensive adaptive evaluator for truly critical positions
        is_critical_position = board.is_check() or len(list(board.legal_moves)) < 6
        is_early_search = self.nodes_searched < 50  # Very early in search only
        
        use_adaptive_evaluation = (
            is_early_search and is_critical_position  # Only critical positions very early in search
        )
        
        if use_adaptive_evaluation:
            # Use expensive but accurate adaptive evaluation for critical positions only
            try:
                evaluation_score = self.adaptive_evaluator.evaluate_position(board, depth=0)
                
                # V11 PHASE 2: Add strategic database evaluation bonus (always for adaptive eval)
                try:
                    strategic_bonus = self.strategic_database.get_strategic_evaluation_bonus(board)
                    evaluation_score += strategic_bonus * 0.1  # Strategic input
                except Exception as e:
                    # Ignore strategic bonus if there's an error
                    pass
                    
            except Exception as e:
                # Fallback to fast evaluation if adaptive system fails
                evaluation_score = self.fast_evaluator.evaluate_position_fast(board)
        else:
            # Use fast evaluation for 99%+ of nodes (performance optimization)
            evaluation_score = self.fast_evaluator.evaluate_position_fast(board)
            
            # V11 INTEGRATION: Add lightweight strategic input even for fast evaluation
            try:
                if self.nodes_searched % 10 == 0:  # Every 10th node
                    strategic_bonus = self.strategic_database.get_strategic_evaluation_bonus(board)
                    evaluation_score += strategic_bonus * 0.05  # Light strategic influence
            except Exception as e:
                pass
            evaluation_score = self.fast_evaluator.evaluate_position_fast(board)
        
        # Cache and return result
        self.evaluation_cache[cache_key] = evaluation_score
        return evaluation_score
    
    def _probe_transposition_table(self, board: chess.Board, depth: int, alpha: int, beta: int) -> Tuple[bool, int, Optional[chess.Move]]:
        """Probe transposition table for this position - PHASE 1 FEATURE"""
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
        """Store result in transposition table - PHASE 1 FEATURE"""
        zobrist_hash = self.zobrist.hash_position(board)
        
        # Determine node type
        if score <= alpha:
            node_type = 'upperbound'
        elif score >= beta:
            node_type = 'lowerbound'
        else:
            node_type = 'exact'
        
        # Simple replacement strategy for performance
        if len(self.transposition_table) >= self.max_tt_entries:
            # Clear 25% of entries when full (simple aging)
            entries = list(self.transposition_table.items())
            entries.sort(key=lambda x: x[1].depth, reverse=True)
            self.transposition_table = dict(entries[:int(self.max_tt_entries * 0.75)])
        
        entry = TranspositionEntry(depth, score, best_move, node_type, zobrist_hash)
        self.transposition_table[zobrist_hash] = entry
        self.search_stats['tt_stores'] += 1

    def _has_non_pawn_pieces(self, board: chess.Board) -> bool:
        """Check if the current side has non-pawn pieces (for null move pruning)"""
        current_color = board.turn
        for square in range(64):
            piece = board.piece_at(square)
            if piece and piece.color == current_color and piece.piece_type != chess.PAWN:
                return True
        return False
    
    def _extract_pv(self, board: chess.Board, max_depth: int) -> List[chess.Move]:
        """Extract principal variation from transposition table"""
        pv = []
        temp_board = board.copy()
        
        for depth in range(max_depth, 0, -1):
            zobrist_hash = self.zobrist.hash_position(temp_board)
            
            if zobrist_hash in self.transposition_table:
                entry = self.transposition_table[zobrist_hash]
                if entry.best_move and entry.best_move in temp_board.legal_moves:
                    pv.append(entry.best_move)
                    temp_board.push(entry.best_move)
                else:
                    break
            else:
                break
        
        return pv
    
    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int) -> float:
        """
        Quiescence search for tactical stability - V10 PHASE 2
        Only search captures and checks to avoid horizon effects
        """
        self.nodes_searched += 1
        
        # Stand pat evaluation
        stand_pat = self._evaluate_position(board)
        
        # Beta cutoff on stand pat
        if stand_pat >= beta:
            return beta
        
        # Update alpha if stand pat is better
        if stand_pat > alpha:
            alpha = stand_pat
        
        # Depth limit reached
        if depth <= 0:
            return stand_pat
        
        # Generate and search tactical moves only
        legal_moves = list(board.legal_moves)
        tactical_moves = []
        
        for move in legal_moves:
            # Only consider captures and checks for quiescence
            if board.is_capture(move) or board.gives_check(move):
                tactical_moves.append(move)
        
        # If no tactical moves, return stand pat
        if not tactical_moves:
            return stand_pat
        
        # Sort tactical moves by MVV-LVA for better ordering
        capture_scores = []
        for move in tactical_moves:
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                victim_value = self.piece_values.get(victim.piece_type, 0) if victim else 0
                attacker = board.piece_at(move.from_square)
                attacker_value = self.piece_values.get(attacker.piece_type, 0) if attacker else 0
                mvv_lva = victim_value * 100 - attacker_value
                capture_scores.append((mvv_lva, move))
            else:
                # Check moves get lower priority
                capture_scores.append((0, move))
        
        # Sort by MVV-LVA score
        capture_scores.sort(key=lambda x: x[0], reverse=True)
        ordered_tactical = [move for _, move in capture_scores]
        
        # Search tactical moves
        best_score = stand_pat
        for move in ordered_tactical:
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha, depth - 1)
            board.pop()
            
            if score > best_score:
                best_score = score
            
            if score > alpha:
                alpha = score
            
            if alpha >= beta:
                break  # Beta cutoff
        
        return best_score
    
    def _detect_bitboard_tactics(self, board: chess.Board, move: chess.Move) -> float:
        """
        V11 PHASE 3B ENHANCED: Detect tactical patterns using advanced pattern detector
        Returns a bonus score for tactical moves (pins, forks, skewers, discovered attacks)
        """
        tactical_bonus = 0.0
        
        # Make the move to analyze the resulting position
        board.push(move)
        
        try:
            our_color = not board.turn  # We just moved, so it's opponent's turn
            
            # V10.6 ROLLBACK: Use legacy tactical analysis only  
            # Advanced tactical patterns disabled due to 70% performance degradation
            tactical_bonus += 0  # V10.6: Disabled Phase 3B advanced tactical detection
            
            # Legacy bitboard tactics for additional analysis
            moving_piece = board.piece_at(move.to_square)
            if moving_piece:
                fork_bonus = self._analyze_fork_bitboard(board, move.to_square, moving_piece, board.turn)
                tactical_bonus += fork_bonus
                
                # Analyze for pins and skewers using ray attacks
                if moving_piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    pin_skewer_bonus = self._analyze_pins_skewers_bitboard(board, move.to_square, moving_piece, board.turn)
                    tactical_bonus += pin_skewer_bonus
            
        except Exception:
            # If analysis fails, return 0 bonus
            pass
        finally:
            board.pop()
        
        return tactical_bonus
    
    def _analyze_fork_bitboard(self, board: chess.Board, square: int, piece: chess.Piece, enemy_color: chess.Color) -> float:
        """Analyze fork patterns using bitboards"""
        if piece.piece_type == chess.KNIGHT:
            # Knight fork detection
            attacks = self.bitboard_evaluator.bitboard_evaluator.KNIGHT_ATTACKS[square]
            enemy_pieces = 0
            high_value_targets = 0
            
            for target_sq in range(64):
                if attacks & (1 << target_sq):
                    target_piece = board.piece_at(target_sq)
                    if target_piece and target_piece.color == enemy_color:
                        enemy_pieces += 1
                        if target_piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                            high_value_targets += 1
            
            # Knight forking 2+ pieces gets bonus, more for high-value targets
            if enemy_pieces >= 2:
                return 50.0 + (high_value_targets * 25.0)
        
        return 0.0
    
    def _analyze_pins_skewers_bitboard(self, board: chess.Board, square: int, piece: chess.Piece, enemy_color: chess.Color) -> float:
        """Analyze pin and skewer patterns using ray attacks"""
        # This is a simplified version - full implementation would need sliding piece attack generation
        # For now, just give a small bonus for pieces that could create pins/skewers
        
        if piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            # Look for aligned enemy pieces that could be pinned/skewered
            bonus = 0.0
            
            # Check if we're attacking towards the enemy king
            enemy_king_sq = None
            for sq in range(64):
                p = board.piece_at(sq)
                if p and p.piece_type == chess.KING and p.color == enemy_color:
                    enemy_king_sq = sq
                    break
            
            if enemy_king_sq is not None:
                # Simple heuristic: if we're on the same rank/file/diagonal as enemy king
                sq_rank, sq_file = divmod(square, 8)
                king_rank, king_file = divmod(enemy_king_sq, 8)
                
                if (sq_rank == king_rank or sq_file == king_file or 
                    abs(sq_rank - king_rank) == abs(sq_file - king_file)):
                    bonus += 15.0  # Potential pin/skewer bonus
            
            return bonus
        
        return 0.0
    
    def notify_move_played(self, move: chess.Move, board_before_move: chess.Board):
        """Notify engine that a move was played (for PV following)
        
        Args:
            move: The move that was played
            board_before_move: The board position BEFORE the move was made
        """
        # We don't need this method anymore - position checking is automatic
        # The new approach checks positions directly when search() is called
        pass

    def new_game(self):
        """Reset for a new game"""
        self.evaluation_cache.clear()
        self.transposition_table.clear()
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        self.nodes_searched = 0
        
        # Clear PV following data
        self.pv_tracker.clear()
        
        # V10.9 PHASE 3B: Reset time control tracking
        self.current_time_remaining_ms = 600000  # Default 10 minutes
        self.current_moves_played = 0
    
    def update_time_control_info(self, time_remaining_ms: int):
        """V10.9 PHASE 3B: Update time control information for tactical pattern detector"""
        self.current_time_remaining_ms = time_remaining_ms
        
        # Reset stats
        for key in self.search_stats:
            self.search_stats[key] = 0

    @property
    def principal_variation(self) -> List[chess.Move]:
        """Get the current principal variation"""
        # Return the original PV from the last search for display purposes
        return self.pv_tracker.original_pv.copy() if self.pv_tracker.original_pv else []

    def perft(self, board: chess.Board, depth: int, divide: bool = False, root_call: bool = True) -> int:
        """
        V11 ENHANCEMENT: Perft (Performance Test) - counts nodes at given depth
        
        This is essential for move generation validation and testing.
        Standard chess engine requirement that was missing in V10.2.
        
        Args:
            board: Current chess position
            depth: Depth to search (number of plies)
            divide: If True, show per-move breakdown at root
            root_call: Internal flag for root level tracking
            
        Returns:
            Total number of leaf nodes at specified depth
        """
        if depth == 0:
            return 1
        
        if root_call:
            self.perft_start_time = time.time()
            self.perft_nodes = 0
        
        legal_moves = list(board.legal_moves)
        total_nodes = 0
        
        if divide and depth == 1:
            # For divide, show each move's contribution
            for move in legal_moves:
                board.push(move)
                nodes = 1  # At depth 1, each legal move contributes 1 node
                board.pop()
                total_nodes += nodes
                if root_call:
                    print(f"{move}: {nodes}")
        else:
            # Normal perft counting
            for move in legal_moves:
                board.push(move)
                nodes = self.perft(board, depth - 1, False, False)
                board.pop()
                total_nodes += nodes
                
                if divide and root_call:
                    print(f"{move}: {nodes}")
        
        if root_call:
            elapsed = time.time() - self.perft_start_time
            nps = int(total_nodes / max(elapsed, 0.001))
            print(f"\nNodes searched: {total_nodes}")
            print(f"Time: {elapsed:.3f}s")
            print(f"Nodes per second: {nps}")
        
        return total_nodes

    def _calculate_adaptive_time_allocation(self, board: chess.Board, base_time_limit: float) -> Tuple[float, float]:
        """
        V11 ENHANCEMENT: Adaptive time management based on position complexity
        
        Returns: (target_time, max_time)
        """
        moves_played = len(board.move_stack)
        legal_moves = list(board.legal_moves)
        num_legal_moves = len(legal_moves)
        
        # Base time factor
        time_factor = 1.0
        
        # Game phase adjustment
        if moves_played < 15:  # Opening
            time_factor *= 0.8  # Faster in opening
        elif moves_played < 40:  # Middle game
            time_factor *= 1.2  # More time in complex middle game
        else:  # Endgame
            time_factor *= 0.9  # Moderate time in endgame
        
        # Position complexity factors
        if board.is_check():
            time_factor *= 1.3  # More time when in check
        
        if num_legal_moves <= 5:
            time_factor *= 0.7  # Less time with few options
        elif num_legal_moves >= 35:
            time_factor *= 1.4  # More time with many options
        
        # Material balance consideration
        our_material = self._count_material(board, board.turn)
        their_material = self._count_material(board, not board.turn)
        material_diff = our_material - their_material
        
        if material_diff < -300:  # We're behind
            time_factor *= 1.2  # Take more time when behind
        elif material_diff > 300:  # We're ahead
            time_factor *= 0.9  # Play faster when ahead
        
        # Calculate final times
        target_time = min(base_time_limit * time_factor * 0.8, base_time_limit * 0.9)
        max_time = min(base_time_limit * time_factor, base_time_limit)
        
        return target_time, max_time
    
    def _count_material(self, board: chess.Board, color: bool) -> int:
        """Count total material for a color"""
        total = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            pieces = board.pieces(piece_type, color)
            total += len(pieces) * self.piece_values[piece_type]
        return total

    def _calculate_lmr_reduction(self, move: chess.Move, moves_searched: int, search_depth: int, board: chess.Board) -> int:
        """
        V11 PHASE 1 ENHANCED: Advanced Late Move Reduction for 30-50% node reduction
        
        Returns reduction amount (0 = no reduction, 1+ = reduction plies)
        """
        # LMR configuration for optimal performance
        lmr_threshold = 4  # Start reducing after 4 moves (vs previous 3)
        min_depth_for_lmr = 3  # Only apply LMR at depth 3+
        max_reduction = 3  # Maximum reduction amount
        
        # No reduction for first few moves or at shallow depths
        if moves_searched < lmr_threshold or search_depth < min_depth_for_lmr:
            return 0
        
        # No reduction for important moves (expanded conditions)
        if (board.is_capture(move) or 
            board.gives_check(move) or 
            board.is_check() or
            self.killer_moves.is_killer(move, search_depth)):
            return 0
        
        # No reduction for pawn moves near promotion
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(move.to_square)
            if to_rank in [0, 1, 6, 7]:  # Near promotion ranks
                return 0
        
        # No reduction for moves that escape from attack
        if board.is_attacked_by(not board.turn, move.from_square):
            return 0
        
        # Calculate progressive reduction
        base_reduction = 1
        
        # Increase reduction based on move ordering position
        move_factor = (moves_searched - lmr_threshold) // 4
        reduction = base_reduction + move_factor
        
        # Depth-based scaling
        if search_depth >= 6:
            reduction += 1
        if search_depth >= 9:
            reduction += 1
        
        # History heuristic adjustment
        history_score = self.history_heuristic.get_history_score(move)
        if history_score > 100:  # Very good history
            reduction = max(0, reduction - 1)
        elif history_score < 10:  # Poor history
            reduction += 1
        
        # Apply bounds and safety limits
        reduction = max(1, min(reduction, max_reduction))
        
        # Never reduce more than half the remaining depth
        max_safe_reduction = max(1, search_depth // 2)
        reduction = min(reduction, max_safe_reduction)
        
        return reduction
        
        return reduction

    # V11 PHASE 2: NUDGE SYSTEM METHODS
    
    def _load_nudge_database(self):
        """Load the nudge database from JSON file"""
        try:
            # Construct path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            nudge_db_path = os.path.join(current_dir, 'v7p3r_nudge_database.json')
            
            if os.path.exists(nudge_db_path):
                with open(nudge_db_path, 'r', encoding='utf-8') as f:
                    self.nudge_database = json.load(f)
                print(f"info string Loaded {len(self.nudge_database)} nudge positions")
            else:
                print(f"info string Nudge database not found at {nudge_db_path}")
                self.nudge_database = {}
        except Exception as e:
            print(f"info string Error loading nudge database: {e}")
            self.nudge_database = {}
    
    def _get_position_key(self, board: chess.Board) -> str:
        """Generate position key for nudge database lookup"""
        # Use FEN without halfmove and fullmove clocks for broader matching
        fen_parts = board.fen().split(' ')
        if len(fen_parts) >= 4:
            # Keep: position, turn, castling, en passant
            key_fen = ' '.join(fen_parts[:4])
        else:
            key_fen = board.fen()
        
        # Generate hash key similar to nudge database format
        import hashlib
        return hashlib.md5(key_fen.encode()).hexdigest()[:12]
    
    def _get_nudge_bonus(self, board: chess.Board, move: chess.Move) -> float:
        """Calculate nudge bonus for a move in current position"""
        try:
            position_key = self._get_position_key(board)
            
            # Check if position exists in nudge database
            if position_key not in self.nudge_database:
                self.nudge_stats['misses'] += 1
                
                # V11 PHASE 2: Try strategic database for similar positions
                try:
                    strategic_bonus = self.strategic_database.get_strategic_move_bonus(board, move)
                    return strategic_bonus * 25.0  # Scale to match nudge bonus range
                except:
                    return 0.0
            
            position_data = self.nudge_database[position_key]
            move_uci = move.uci()
            
            # Check if move exists in nudge data
            if 'moves' not in position_data or move_uci not in position_data['moves']:
                # Try strategic database bonus even if not in nudge data
                try:
                    strategic_bonus = self.strategic_database.get_strategic_move_bonus(board, move)
                    return strategic_bonus * 25.0  # Scale to match nudge bonus range
                except:
                    return 0.0
            
            move_data = position_data['moves'][move_uci]
            
            # Calculate bonus based on frequency and evaluation
            frequency = move_data.get('frequency', 1)
            evaluation = move_data.get('eval', 0.0)
            
            # Base bonus for nudge moves
            base_bonus = 50.0
            
            # Frequency multiplier (more frequent = higher bonus, capped at 3x)
            frequency_multiplier = min(frequency / 2.0, 3.0)
            
            # Evaluation multiplier (better evaluation = higher bonus)
            eval_multiplier = max(evaluation / 0.5, 1.0) if evaluation > 0 else 1.0
            
            total_bonus = base_bonus * frequency_multiplier * eval_multiplier
            
            # V11 PHASE 2: Add strategic database bonus
            try:
                strategic_bonus = self.strategic_database.get_strategic_move_bonus(board, move)
                total_bonus += strategic_bonus * 25.0  # Additional strategic bonus
            except:
                pass
            
            # Update statistics
            self.nudge_stats['hits'] += 1
            self.nudge_stats['moves_boosted'] += 1
            
            return total_bonus
            
        except Exception as e:
            # Silently handle errors to avoid disrupting search
            return 0.0

    def _check_instant_nudge_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        V11 PHASE 2 ENHANCEMENT: Check for instant nudge moves that bypass search
        Returns move if confidence is high enough, None otherwise
        """
        try:
            position_key = self._get_position_key(board)
            
            # Check if position exists in nudge database
            if position_key not in self.nudge_database:
                return None
            
            position_data = self.nudge_database[position_key]
            moves_data = position_data.get('moves', {})
            
            if not moves_data:
                return None
            
            best_move = None
            best_confidence = 0.0
            
            # Evaluate all nudge moves for instant play criteria
            for move_uci, move_data in moves_data.items():
                try:
                    move = chess.Move.from_uci(move_uci)
                    
                    # Verify move is legal
                    if move not in board.legal_moves:
                        continue
                    
                    frequency = move_data.get('frequency', 0)
                    evaluation = move_data.get('eval', 0.0)
                    
                    # Check minimum thresholds
                    if (frequency < self.nudge_instant_config['min_frequency'] or 
                        evaluation < self.nudge_instant_config['min_eval']):
                        continue
                    
                    # Calculate confidence score (frequency + eval bonus)
                    confidence = frequency + (evaluation * 10)  # Scale eval to match frequency range
                    
                    # Track best candidate
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_move = move
                
                except:
                    continue
            
            # Check if best move meets confidence threshold
            if (best_move and 
                best_confidence >= self.nudge_instant_config['confidence_threshold']):
                
                # Update statistics
                self.nudge_stats['instant_moves'] += 1
                
                return best_move
            
            return None
            
        except Exception as e:
            # Silently handle errors to avoid disrupting search
            return None
