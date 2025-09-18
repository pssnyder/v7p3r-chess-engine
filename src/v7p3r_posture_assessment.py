#!/usr/bin/env python3
"""
V7P3R v11 Phase 3B: Position Posture Assessment System
Intelligent detection of position "tone" to drive evaluation priorities
Author: Pat Snyder
"""

import chess
from typing import Dict, List, Tuple, Optional
from enum import Enum


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
        # Use material imbalance, king positions, and turn
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
        
        # Factor 1: Pieces under attack (reduced weight)
        attacked_pieces = self._count_attacked_pieces(board)
        volatility_score += attacked_pieces * 8  # Reduced from 15
        
        # Factor 2: Undefended pieces (reduced weight)
        undefended_pieces = self._count_undefended_pieces(board)
        volatility_score += undefended_pieces * 12  # Increased from 10 but still reasonable
        
        # Factor 3: Pins and skewers (reduced weight)
        tactical_threats = self._count_tactical_patterns(board)
        volatility_score += tactical_threats * 15  # Reduced from 20
        
        # Factor 4: King exposure (reduced weight)
        king_danger = self._assess_king_exposure(board)
        volatility_score += king_danger * 18  # Reduced from 25
        
        # Factor 5: Material imbalance (increased sensitivity to major imbalances)
        material_imbalance = self._assess_material_imbalance(board)
        if abs(material_imbalance) >= 300:  # Significant material difference
            volatility_score += abs(material_imbalance) // 150  # Reduced impact
        
        # Classify volatility with adjusted thresholds
        if volatility_score <= 10:  # More conservative thresholds
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
        
        # Assess initiative (who moved last created more threats?)
        initiative_factor = our_threats - their_threats
        
        # Combine factors to determine posture
        posture_score = initiative_factor + (material_advantage // 100)
        
        # Factor in volatility
        if volatility == PositionVolatility.VOLATILE:
            # In volatile positions, prefer defensive if we're not clearly ahead
            if posture_score <= 1:
                return GamePosture.DEFENSIVE
        
        # Determine posture with improved logic
        if their_threats >= our_threats + 2:  # Clear threat disadvantage
            return GamePosture.DEFENSIVE
        elif their_threats > our_threats and volatility in [PositionVolatility.VOLATILE, PositionVolatility.CRITICAL]:
            return GamePosture.DEFENSIVE  # Be defensive in volatile positions when behind in threats
        elif our_threats >= their_threats + 2:  # Clear threat advantage
            return GamePosture.OFFENSIVE
        elif material_advantage >= 200:  # Significant material advantage
            return GamePosture.OFFENSIVE
        elif material_advantage <= -200:  # Significant material disadvantage
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
                # Check if attacked and not defended
                if (board.is_attacked_by(not current_color, square) and 
                    not board.is_attacked_by(current_color, square)):
                    undefended_count += 1
        
        return undefended_count
    
    def _count_tactical_patterns(self, board: chess.Board) -> int:
        """Count basic tactical patterns (pins, skewers, forks)"""
        # Simplified tactical pattern detection
        # This is a placeholder for more sophisticated pattern recognition
        
        tactical_count = 0
        current_color = board.turn
        
        # Look for potential forks (one piece attacking multiple pieces)
        for move in board.legal_moves:
            board.push(move)
            
            # Count how many enemy pieces this move would attack
            attacking_count = 0
            move_square = move.to_square
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color != current_color:
                    if board.is_attacked_by(current_color, square):
                        attacking_count += 1
            
            if attacking_count >= 2:  # Potential fork
                tactical_count += 1
            
            board.pop()
            
            # Limit analysis to avoid performance impact
            if tactical_count >= 5:
                break
        
        return min(tactical_count, 3)  # Cap for performance
    
    def _assess_king_exposure(self, board: chess.Board) -> int:
        """Assess king exposure for current player"""
        current_color = board.turn
        king_square = board.king(current_color)
        
        if king_square is None:
            return 10  # High danger if no king
        
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
        
        if board.turn:  # White to move
            return white_material - black_material
        else:  # Black to move
            return black_material - white_material
    
    def _count_our_threats(self, board: chess.Board) -> int:
        """Count threats we can create"""
        threat_count = 0
        current_color = board.turn
        
        for move in list(board.legal_moves)[:20]:  # Limit for performance
            board.push(move)
            
            # Count enemy pieces we'd be attacking after this move
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color != current_color:
                    if board.is_attacked_by(current_color, square):
                        threat_count += 1
            
            board.pop()
        
        return threat_count
    
    def _count_their_threats(self, board: chess.Board) -> int:
        """Count threats opponent can create on their turn"""
        # Switch sides to evaluate opponent threats
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