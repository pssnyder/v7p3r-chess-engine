"""
Enhanced evaluation system for V7P3R v11.2
Combines tactical pattern recognition with strategic assessment
"""

import chess
from typing import Dict, List, Tuple
import sys
import os

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from v7p3r_scoring_calculation_bitboard import V7P3RScoringCalculationBitboard
except ImportError:
    print("Warning: Could not import V7P3RScoringCalculationBitboard, using fallback evaluation")


class V7P3REnhancedEvaluation:
    """Enhanced evaluation combining bitboard scoring with tactical pattern recognition"""
    
    def __init__(self):
        """Initialize the enhanced evaluation system"""
        # Try to load the bitboard evaluator
        try:
            self.bitboard_evaluator = V7P3RScoringCalculationBitboard()
            self.has_bitboard = True
            print("info string Enhanced evaluation: Bitboard evaluator loaded")
        except:
            self.has_bitboard = False
            print("info string Enhanced evaluation: Using tactical-only mode")
        
        # Piece values for fallback
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        # Position square tables for tactical awareness
        self.pawn_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5, 5, 10, 25, 25, 10, 5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, -5, -10, 0, 0, -10, -5, 5,
            5, 10, 10, -20, -20, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
        
        self.knight_table = [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
        ]
        
        self.bishop_table = [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ]
        
        self.king_table_middlegame = [
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            20, 20, 0, 0, 0, 0, 20, 20,
            20, 30, 10, 0, 0, 10, 30, 20
        ]
        
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Main evaluation function
        Returns score from current player's perspective
        """
        try:
            if self.has_bitboard:
                # Use bitboard evaluator for strategic assessment
                white_score = self.bitboard_evaluator.calculate_score_optimized(board, chess.WHITE)
                black_score = self.bitboard_evaluator.calculate_score_optimized(board, chess.BLACK)
                
                # Get base strategic score
                if board.turn:  # White to move
                    strategic_score = white_score - black_score
                else:  # Black to move
                    strategic_score = black_score - white_score
            else:
                # Fallback to enhanced material + positional evaluation
                strategic_score = self._enhanced_material_evaluation(board)
            
            # Add tactical pattern recognition
            tactical_bonus = self._evaluate_tactical_patterns(board)
            
            # Combine strategic and tactical scores
            total_score = strategic_score + tactical_bonus
            
            return total_score
            
        except Exception as e:
            print(f"info string Evaluation error: {e}")
            # Ultimate fallback to simple material
            return self._simple_material_evaluation(board)
    
    def _enhanced_material_evaluation(self, board: chess.Board) -> float:
        """Enhanced material evaluation with positional bonuses"""
        score = 0.0
        
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                piece_value = self.piece_values.get(piece.piece_type, 0)
                positional_bonus = self._get_positional_bonus(piece, square)
                
                total_value = piece_value + positional_bonus
                
                if piece.color == board.turn:  # Current player's pieces
                    score += total_value
                else:  # Opponent's pieces
                    score -= total_value
        
        return score
    
    def _get_positional_bonus(self, piece: chess.Piece, square: int) -> float:
        """Get positional bonus for piece placement"""
        # Convert square to rank/file for table lookup
        rank = square // 8
        file = square % 8
        
        # Flip rank for black pieces
        if not piece.color:  # Black
            rank = 7 - rank
        
        table_index = rank * 8 + file
        
        if piece.piece_type == chess.PAWN:
            return self.pawn_table[table_index] * 0.01
        elif piece.piece_type == chess.KNIGHT:
            return self.knight_table[table_index] * 0.01
        elif piece.piece_type == chess.BISHOP:
            return self.bishop_table[table_index] * 0.01
        elif piece.piece_type == chess.KING:
            return self.king_table_middlegame[table_index] * 0.01
        
        return 0.0
    
    def _evaluate_tactical_patterns(self, board: chess.Board) -> float:
        """Evaluate tactical patterns and threats"""
        tactical_score = 0.0
        
        # Check for basic tactical motifs
        tactical_score += self._evaluate_piece_attacks(board)
        tactical_score += self._evaluate_king_safety(board)
        tactical_score += self._evaluate_piece_coordination(board)
        tactical_score += self._evaluate_center_control(board)
        
        return tactical_score
    
    def _evaluate_piece_attacks(self, board: chess.Board) -> float:
        """Evaluate piece attack patterns"""
        score = 0.0
        
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                # Count attackers and defenders
                attackers = board.attackers(not piece.color, square)
                defenders = board.attackers(piece.color, square)
                
                # Bonus for attacking valuable pieces
                if len(attackers) > 0:
                    attack_value = self.piece_values.get(piece.piece_type, 0) * 0.1
                    if piece.color == board.turn:  # Opponent attacking our pieces
                        score -= attack_value
                    else:  # We're attacking opponent pieces
                        score += attack_value
                
                # Penalty for undefended pieces
                if len(defenders) == 0 and len(attackers) > 0:
                    undefended_penalty = self.piece_values.get(piece.piece_type, 0) * 0.05
                    if piece.color == board.turn:  # Our undefended piece
                        score -= undefended_penalty
                    else:  # Their undefended piece
                        score += undefended_penalty
        
        return score
    
    def _evaluate_king_safety(self, board: chess.Board) -> float:
        """Evaluate king safety"""
        score = 0.0
        
        # Find kings
        our_king = board.king(board.turn)
        their_king = board.king(not board.turn)
        
        if our_king is not None:
            # Penalty for checks
            if board.is_check():
                score -= 50.0
            
            # Bonus for castling rights
            if board.turn:
                if board.has_kingside_castling_rights(chess.WHITE):
                    score += 10.0
                if board.has_queenside_castling_rights(chess.WHITE):
                    score += 5.0
            else:
                if board.has_kingside_castling_rights(chess.BLACK):
                    score += 10.0
                if board.has_queenside_castling_rights(chess.BLACK):
                    score += 5.0
        
        if their_king is not None:
            # Bonus for attacking their king
            their_king_attackers = board.attackers(board.turn, their_king)
            score += len(their_king_attackers) * 5.0
        
        return score
    
    def _evaluate_piece_coordination(self, board: chess.Board) -> float:
        """Evaluate piece coordination and development"""
        score = 0.0
        
        # Bonus for developed pieces (knights and bishops not on back rank)
        for square in range(64):
            piece = board.piece_at(square)
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                rank = square // 8
                
                # Check if piece is developed
                if piece.color == chess.WHITE and rank > 0:
                    if piece.color == board.turn:
                        score += 5.0
                    else:
                        score -= 5.0
                elif piece.color == chess.BLACK and rank < 7:
                    if piece.color == board.turn:
                        score += 5.0
                    else:
                        score -= 5.0
        
        return score
    
    def _evaluate_center_control(self, board: chess.Board) -> float:
        """Evaluate center control"""
        score = 0.0
        
        # Central squares
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        
        for square in center_squares:
            # Count attackers of central squares
            our_attackers = board.attackers(board.turn, square)
            their_attackers = board.attackers(not board.turn, square)
            
            score += len(our_attackers) * 2.0
            score -= len(their_attackers) * 2.0
            
            # Bonus for occupying center
            piece = board.piece_at(square)
            if piece:
                if piece.color == board.turn:
                    score += 10.0
                else:
                    score -= 10.0
        
        return score
    
    def _simple_material_evaluation(self, board: chess.Board) -> float:
        """Simple material count fallback"""
        score = 0.0
        
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                piece_value = self.piece_values.get(piece.piece_type, 0)
                if piece.color == board.turn:
                    score += piece_value
                else:
                    score -= piece_value
        
        return score