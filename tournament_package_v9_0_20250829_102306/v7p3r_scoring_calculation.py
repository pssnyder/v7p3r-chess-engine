#!/usr/bin/env python3
"""
V7P3R Scoring Calculation v7.0 - Clean Slate Edition
Clean evaluation module aligned with C0BR4's proven heuristics
Author: Pat Snyder
"""

import chess
from typing import Dict


class V7P3RScoringCalculationClean:
    """Clean, simple scoring calculation following C0BR4's evaluation principles"""
    
    def __init__(self, piece_values: Dict):
        self.piece_values = piece_values
    
    def calculate_score_optimized(self, board: chess.Board, color: chess.Color, endgame_factor: float = 0.0) -> float:
        """
        C0BR4-aligned scoring with only proven heuristics.
        Returns positive values for good positions for the given color.
        """
        score = 0.0
        
        # 1. Material evaluation (C0BR4 core)
        score += self._material_score(board, color)
        
        # 2. King Safety (C0BR4 core) 
        score += self._king_safety(board, color)
        
        # 3. Development/Piece-Square Tables equivalent
        score += self._piece_development(board, color)
        
        # 4. Castling evaluation (C0BR4 castling incentive/rights)
        score += self._castling_bonus(board, color)
        
        # 5. Rook coordination (C0BR4 core)
        score += self._rook_coordination(board, color)
        
        # 6. Center control (basic positional)
        score += self._center_control(board, color)
        
        # 7. Endgame logic
        if self._is_endgame(board):
            score += self._endgame_logic(board, color)
            
        return score
    
    def _material_score(self, board: chess.Board, color: chess.Color) -> float:
        """Material count for given color - C0BR4 style"""
        score = 0.0
        for piece_type, value in self.piece_values.items():
            if piece_type != chess.KING:  # King safety handled separately
                piece_count = len(board.pieces(piece_type, color))
                score += piece_count * value
        return score
    
    def _king_safety(self, board: chess.Board, color: chess.Color) -> float:
        """Basic king safety - C0BR4 style"""
        score = 0.0
        king_square = board.king(color)
        if not king_square:
            return -1000.0  # No king = very bad
            
        # Penalty for exposed king
        if self._is_king_exposed(board, color, king_square):
            score -= 50.0
            
        return score
    
    def _piece_development(self, board: chess.Board, color: chess.Color) -> float:
        """Piece development bonus - C0BR4 PST equivalent"""
        score = 0.0
        
        # Bonus for developed knights and bishops
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for square in board.pieces(piece_type, color):
                # Bonus for pieces not on back rank
                if color == chess.WHITE and chess.square_rank(square) > 0:
                    score += 5.0
                elif color == chess.BLACK and chess.square_rank(square) < 7:
                    score += 5.0
                    
        return score
    
    def _castling_bonus(self, board: chess.Board, color: chess.Color) -> float:
        """Castling bonus - C0BR4 style"""
        score = 0.0
        king_square = board.king(color)
        
        if king_square:
            # Bonus for castled king
            if color == chess.WHITE and king_square in [chess.G1, chess.C1]:
                score += 20.0
            elif color == chess.BLACK and king_square in [chess.G8, chess.C8]:
                score += 20.0
        
        return score
    
    def _rook_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Rook coordination - C0BR4 style"""
        score = 0.0
        rooks = list(board.pieces(chess.ROOK, color))
        
        if len(rooks) >= 2:
            # Simple bonus for having both rooks
            score += 10.0
            
        return score
    
    def _center_control(self, board: chess.Board, color: chess.Color) -> float:
        """Center control bonus"""
        score = 0.0
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        
        for square in center_squares:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                if piece.piece_type == chess.PAWN:
                    score += 10.0  # Pawn in center
                else:
                    score += 5.0   # Other piece in center
                    
        return score
    
    def _endgame_logic(self, board: chess.Board, color: chess.Color) -> float:
        """Basic endgame logic"""
        score = 0.0
        
        # King activity in endgame
        king_square = board.king(color)
        if king_square:
            # Bonus for centralized king
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            center_distance = abs(king_file - 3.5) + abs(king_rank - 3.5)
            score += (7 - center_distance) * 2  # Closer to center = better
            
        return score
    
    def _is_king_exposed(self, board: chess.Board, color: chess.Color, king_square: int) -> bool:
        """Check if king is dangerously exposed"""
        # Simple check: king on starting rank = safer
        if color == chess.WHITE:
            return chess.square_rank(king_square) > 2
        else:
            return chess.square_rank(king_square) < 5
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """Detect endgame phase"""
        # Count major pieces
        major_pieces = 0
        for color in [chess.WHITE, chess.BLACK]:
            major_pieces += len(board.pieces(chess.QUEEN, color)) * 2
            major_pieces += len(board.pieces(chess.ROOK, color))
            
        return major_pieces <= 6  # Arbitrary threshold
