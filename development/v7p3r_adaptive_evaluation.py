#!/usr/bin/env python3
"""
V7P3R v11 Phase 3B: Adaptive Evaluation Framework
Intelligent evaluation system that adapts based on position posture
Author: Pat Snyder
"""

import chess
from typing import Dict, Tuple, Optional, Any
from v7p3r_posture_assessment import V7P3RPostureAssessment, PositionVolatility, GamePosture


class V7P3RAdaptiveEvaluation:
    """
    Adaptive evaluation system that runs different evaluation components 
    based on position posture for maximum efficiency and focus
    """
    
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
                'material': 1.0,           # Material is always important
                'king_safety': 2.0,        # Critical in emergency
                'piece_safety': 1.5,       # Protect pieces under attack
                'mobility': 0.3,           # Less important when in danger
                'positional': 0.1,         # Lowest priority
                'development': 0.2,        # Limited importance
                'strategic': 0.0           # Skip strategic analysis
            },
            GamePosture.DEFENSIVE: {
                'material': 1.0,
                'king_safety': 1.5,        # Still important
                'piece_safety': 1.3,       # High priority
                'mobility': 0.6,           # Moderate importance
                'positional': 0.4,         # Some positional play
                'development': 0.5,        # Modest importance
                'strategic': 0.2           # Limited strategic focus
            },
            GamePosture.BALANCED: {
                'material': 1.0,
                'king_safety': 1.0,        # Normal importance
                'piece_safety': 1.0,       # Normal importance
                'mobility': 1.0,           # Full evaluation
                'positional': 1.0,         # Full positional analysis
                'development': 0.8,        # Good importance
                'strategic': 0.6           # Moderate strategic analysis
            },
            GamePosture.OFFENSIVE: {
                'material': 1.0,
                'king_safety': 0.8,        # Slightly less critical
                'piece_safety': 0.7,       # Accept some risk
                'mobility': 1.2,           # Extra mobility for attacks
                'positional': 1.1,         # Aggressive positioning
                'development': 1.0,        # Aggressive development
                'strategic': 1.0           # Full strategic analysis
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
        """
        Main evaluation function that adapts based on position posture
        Returns evaluation score from current player's perspective
        """
        import time
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
        
        # King safety (critical in defensive postures)
        if weights['king_safety'] > 0:
            component_start = time.time()
            king_safety_score = self._evaluate_king_safety(board) * weights['king_safety']
            total_score += king_safety_score
            self.evaluation_stats['component_times']['king_safety'] += time.time() - component_start
        
        # Piece safety (important in defensive postures)
        if weights['piece_safety'] > 0:
            component_start = time.time()
            piece_safety_score = self._evaluate_piece_safety(board) * weights['piece_safety']
            total_score += piece_safety_score
            self.evaluation_stats['component_times']['piece_safety'] += time.time() - component_start
        
        # Mobility (important for balanced/offensive play)
        if weights['mobility'] > 0:
            component_start = time.time()
            mobility_score = self._evaluate_mobility(board) * weights['mobility']
            total_score += mobility_score
            self.evaluation_stats['component_times']['mobility'] += time.time() - component_start
        
        # Positional factors (less important in emergency)
        if weights['positional'] > 0:
            component_start = time.time()
            positional_score = self._evaluate_positional(board) * weights['positional']
            total_score += positional_score
            self.evaluation_stats['component_times']['positional'] += time.time() - component_start
        
        # Strategic analysis (skip in emergency, limited in defensive)
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
            return -100.0  # No king is very bad
        
        # Count attackers on king
        attackers = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != current_color:
                if board.is_attacked_by(not current_color, king_square):
                    attackers += 1
        
        score -= attackers * 0.5  # Penalty for king attackers
        
        # King shelter evaluation
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Check for pawn shelter
        shelter_bonus = 0
        for file_offset in [-1, 0, 1]:
            for rank_offset in [1, 2]:  # In front of king
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
                piece_value = self.piece_values[piece.piece_type] / 1000.0  # Normalize
                
                # Penalty for undefended pieces
                if board.is_attacked_by(not current_color, square):
                    if not board.is_attacked_by(current_color, square):
                        score -= piece_value * 0.5  # Undefended and attacked
                    else:
                        score -= piece_value * 0.1  # Defended but attacked
                
                # Bonus for defended pieces
                elif board.is_attacked_by(current_color, square):
                    score += piece_value * 0.05  # Small bonus for defense
        
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
        return mobility_advantage / 100.0  # Normalize
    
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
        
        # Development bonus (simplified)
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
        # This would integrate with existing strategic database
        # For now, return a small positional bonus
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
                
                if abs(enemy_file - file) <= 1:  # Adjacent or same file
                    if current_color:  # White
                        if enemy_rank > rank:  # Enemy pawn ahead
                            blocked = True
                            break
                    else:  # Black
                        if enemy_rank < rank:  # Enemy pawn ahead
                            blocked = True
                            break
            
            if not blocked:
                distance_to_promotion = 7 - rank if current_color else rank
                score += (8 - distance_to_promotion) * 0.05  # Closer = better
        
        return score
    
    def _get_posture_bonus(self, board: chess.Board, posture: GamePosture, volatility: PositionVolatility) -> float:
        """Get posture-specific evaluation bonus"""
        bonus = 0.0
        
        if posture == GamePosture.EMERGENCY:
            # In emergency, heavily penalize being in this position
            bonus -= 0.5
            
        elif posture == GamePosture.DEFENSIVE:
            # Small penalty for being on defensive
            bonus -= 0.2
            
        elif posture == GamePosture.OFFENSIVE:
            # Small bonus for having initiative
            bonus += 0.3
        
        # Volatility adjustment
        if volatility == PositionVolatility.CRITICAL:
            bonus -= 0.3  # Dangerous positions are bad
        elif volatility == PositionVolatility.STABLE:
            bonus += 0.1  # Stable positions are good
        
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