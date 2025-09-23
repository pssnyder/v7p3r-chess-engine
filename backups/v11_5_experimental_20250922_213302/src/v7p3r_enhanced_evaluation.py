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
        
        return score\n    \n    def _get_positional_bonus(self, piece: chess.Piece, square: int) -> float:\n        \"\"\"Get positional bonus for piece placement\"\"\"\n        # Convert square to rank/file for table lookup\n        rank = square // 8\n        file = square % 8\n        \n        # Flip rank for black pieces\n        if not piece.color:  # Black\n            rank = 7 - rank\n        \n        table_index = rank * 8 + file\n        \n        if piece.piece_type == chess.PAWN:\n            return self.pawn_table[table_index] * 0.01\n        elif piece.piece_type == chess.KNIGHT:\n            return self.knight_table[table_index] * 0.01\n        elif piece.piece_type == chess.BISHOP:\n            return self.bishop_table[table_index] * 0.01\n        elif piece.piece_type == chess.KING:\n            return self.king_table_middlegame[table_index] * 0.01\n        \n        return 0.0\n    \n    def _evaluate_tactical_patterns(self, board: chess.Board) -> float:\n        \"\"\"Evaluate tactical patterns and threats\"\"\"\n        tactical_score = 0.0\n        \n        # Check for basic tactical motifs\n        tactical_score += self._evaluate_piece_attacks(board)\n        tactical_score += self._evaluate_king_safety(board)\n        tactical_score += self._evaluate_piece_coordination(board)\n        tactical_score += self._evaluate_center_control(board)\n        \n        return tactical_score\n    \n    def _evaluate_piece_attacks(self, board: chess.Board) -> float:\n        \"\"\"Evaluate piece attack patterns\"\"\"\n        score = 0.0\n        \n        for square in range(64):\n            piece = board.piece_at(square)\n            if piece:\n                # Count attackers and defenders\n                attackers = board.attackers(not piece.color, square)\n                defenders = board.attackers(piece.color, square)\n                \n                # Bonus for attacking valuable pieces\n                if len(attackers) > 0:\n                    attack_value = self.piece_values.get(piece.piece_type, 0) * 0.1\n                    if piece.color == board.turn:  # Opponent attacking our pieces\n                        score -= attack_value\n                    else:  # We're attacking opponent pieces\n                        score += attack_value\n                \n                # Penalty for undefended pieces\n                if len(defenders) == 0 and len(attackers) > 0:\n                    undefended_penalty = self.piece_values.get(piece.piece_type, 0) * 0.05\n                    if piece.color == board.turn:  # Our undefended piece\n                        score -= undefended_penalty\n                    else:  # Their undefended piece\n                        score += undefended_penalty\n        \n        return score\n    \n    def _evaluate_king_safety(self, board: chess.Board) -> float:\n        \"\"\"Evaluate king safety\"\"\"\n        score = 0.0\n        \n        # Find kings\n        our_king = board.king(board.turn)\n        their_king = board.king(not board.turn)\n        \n        if our_king is not None:\n            # Penalty for checks\n            if board.is_check():\n                score -= 50.0\n            \n            # Bonus for castling rights\n            if board.turn:\n                if board.has_kingside_castling_rights(chess.WHITE):\n                    score += 10.0\n                if board.has_queenside_castling_rights(chess.WHITE):\n                    score += 5.0\n            else:\n                if board.has_kingside_castling_rights(chess.BLACK):\n                    score += 10.0\n                if board.has_queenside_castling_rights(chess.BLACK):\n                    score += 5.0\n        \n        if their_king is not None:\n            # Bonus for attacking their king\n            their_king_attackers = board.attackers(board.turn, their_king)\n            score += len(their_king_attackers) * 5.0\n        \n        return score\n    \n    def _evaluate_piece_coordination(self, board: chess.Board) -> float:\n        \"\"\"Evaluate piece coordination and development\"\"\"\n        score = 0.0\n        \n        # Bonus for developed pieces (knights and bishops not on back rank)\n        for square in range(64):\n            piece = board.piece_at(square)\n            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:\n                rank = square // 8\n                \n                # Check if piece is developed\n                if piece.color == chess.WHITE and rank > 0:\n                    if piece.color == board.turn:\n                        score += 5.0\n                    else:\n                        score -= 5.0\n                elif piece.color == chess.BLACK and rank < 7:\n                    if piece.color == board.turn:\n                        score += 5.0\n                    else:\n                        score -= 5.0\n        \n        return score\n    \n    def _evaluate_center_control(self, board: chess.Board) -> float:\n        \"\"\"Evaluate center control\"\"\"\n        score = 0.0\n        \n        # Central squares\n        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]\n        \n        for square in center_squares:\n            # Count attackers of central squares\n            our_attackers = board.attackers(board.turn, square)\n            their_attackers = board.attackers(not board.turn, square)\n            \n            score += len(our_attackers) * 2.0\n            score -= len(their_attackers) * 2.0\n            \n            # Bonus for occupying center\n            piece = board.piece_at(square)\n            if piece:\n                if piece.color == board.turn:\n                    score += 10.0\n                else:\n                    score -= 10.0\n        \n        return score\n    \n    def _simple_material_evaluation(self, board: chess.Board) -> float:\n        \"\"\"Simple material count fallback\"\"\"\n        score = 0.0\n        \n        for square in range(64):\n            piece = board.piece_at(square)\n            if piece:\n                piece_value = self.piece_values.get(piece.piece_type, 0)\n                if piece.color == board.turn:\n                    score += piece_value\n                else:\n                    score -= piece_value\n        \n        return score