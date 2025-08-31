#!/usr/bin/env python3
"""
V7P3R v9.4-alpha Scoring Calculation
Simplified evaluation focused on beating v7.0 through strategic clarity
Author: Pat Snyder

DESIGN GOALS:
- Match v7.0's simplicity while addressing its key weaknesses
- Remove v9.3's heuristic conflicts and over-complexity
- Focus on 4 core areas: Material, Safety, Development, Tactics
"""

import chess
from typing import Dict


class V7P3RScoringCalculationV94Alpha:
    """v9.4-alpha: Simplified evaluation focused on beating v7.0"""
    
    def __init__(self, piece_values: Dict):
        self.piece_values = piece_values
        self._init_evaluation_tables()
    
    def _init_evaluation_tables(self):
        """Initialize focused evaluation tables"""
        
        # Simplified piece-square tables (key squares only)
        self.center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        self.extended_center = {chess.C3, chess.C4, chess.C5, chess.C6,
                               chess.D3, chess.D6, chess.E3, chess.E6,
                               chess.F3, chess.F4, chess.F5, chess.F6}
        
        # Development target squares (where pieces should go)
        self.knight_good_squares = {
            chess.WHITE: {chess.F3, chess.C3, chess.D2, chess.E2, chess.F6, chess.C6},
            chess.BLACK: {chess.F6, chess.C6, chess.D7, chess.E7, chess.F3, chess.C3}
        }
        
        self.bishop_good_squares = {
            chess.WHITE: {chess.C4, chess.F4, chess.E2, chess.D3, chess.B5, chess.G5},
            chess.BLACK: {chess.C5, chess.F5, chess.E7, chess.D6, chess.B4, chess.G4}
        }
    
    def calculate_score_optimized(self, board: chess.Board, color: chess.Color, endgame_factor: float = 0.0) -> float:
        """
        Simplified evaluation focusing on beating v7.0
        Returns positive values for good positions for the given color.
        """
        score = 0.0
        
        # 1. Material evaluation (same as v7.0 - proven)
        score += self._material_score(board, color)
        
        # 2. King safety (enhanced from v7.0)
        score += self._king_safety_enhanced(board, color)
        
        # 3. Development (simplified from v9.3, better than v7.0)
        score += self._development_focused(board, color)
        
        # 4. Tactical opportunities (v7.0 weakness we exploit)
        score += self._basic_tactics(board, color)
        
        # 5. Endgame improvements (better than v7.0's basic logic)
        if self._is_endgame(board):
            score += self._endgame_enhanced(board, color)
            
        return score
    
    def _material_score(self, board: chess.Board, color: chess.Color) -> float:
        """Material count - same as v7.0 (proven effective)"""
        score = 0.0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            piece_count = len(board.pieces(piece_type, color))
            score += piece_count * self.piece_values.get(piece_type, 0)
        return score
    
    def _king_safety_enhanced(self, board: chess.Board, color: chess.Color) -> float:
        """Enhanced king safety - builds on v7.0's foundation"""
        score = 0.0
        king_square = board.king(color)
        if not king_square:
            return -1000.0
            
        # v7.0's basic exposure check
        if self._is_king_exposed(board, color, king_square):
            score -= 50.0
            
        # Enhancement: Castling status (more detailed than v7.0)
        if board.has_castling_rights(color):
            score += 15.0  # Castling rights available
        elif self._is_castled(board, color, king_square):
            score += 30.0  # Successfully castled
            
        # Enhancement: Basic attack detection around king
        attack_count = self._count_attacks_near_king(board, color, king_square)
        score -= attack_count * 8.0  # Penalty for enemy attacks near king
            
        return score
    
    def _development_focused(self, board: chess.Board, color: chess.Color) -> float:
        """Focused development - better than v7.0, simpler than v9.3"""
        score = 0.0
        
        # Knights development (key to good openings)
        for square in board.pieces(chess.KNIGHT, color):
            if square in self.knight_good_squares[color]:
                score += 15.0
            # Penalty for knights still on back rank
            elif (color == chess.WHITE and chess.square_rank(square) == 0) or \
                 (color == chess.BLACK and chess.square_rank(square) == 7):
                score -= 10.0
                
        # Bishops development
        for square in board.pieces(chess.BISHOP, color):
            if square in self.bishop_good_squares[color]:
                score += 12.0
            # Penalty for trapped bishops
            elif (color == chess.WHITE and chess.square_rank(square) == 0) or \
                 (color == chess.BLACK and chess.square_rank(square) == 7):
                score -= 8.0
                
        # Center control (fundamental advantage)
        score += self._center_control_enhanced(board, color)
        
        return score
    
    def _basic_tactics(self, board: chess.Board, color: chess.Color) -> float:
        """Basic tactical awareness - v7.0's main weakness"""
        score = 0.0
        
        # Fork detection (simple but effective)
        score += self._detect_forks(board, color)
        
        # Pin detection (basic)
        score += self._detect_pins(board, color)
        
        # Hanging pieces (major tactical flaw)
        score -= self._count_hanging_pieces(board, color) * 25.0
        
        return score
    
    def _center_control_enhanced(self, board: chess.Board, color: chess.Color) -> float:
        """Enhanced center control - improvement over v7.0"""
        score = 0.0
        
        # Core center squares (most important)
        for square in self.center_squares:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                if piece.piece_type == chess.PAWN:
                    score += 20.0  # Pawn in center (very strong)
                else:
                    score += 10.0  # Other piece in center
            # Control without occupation
            elif board.is_attacked_by(color, square):
                score += 3.0
                
        # Extended center (supporting squares)
        for square in self.extended_center:
            if board.is_attacked_by(color, square):
                score += 1.0
                
        return score
    
    def _detect_forks(self, board: chess.Board, color: chess.Color) -> float:
        """Simple fork detection - major tactical advantage over v7.0"""
        score = 0.0
        
        for square in board.pieces(chess.KNIGHT, color):
            # Count valuable targets a knight could fork
            attacks = board.attacks(square)
            valuable_targets = 0
            for target_square in attacks:
                target_piece = board.piece_at(target_square)
                if target_piece and target_piece.color != color:
                    if target_piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                        valuable_targets += 1
            
            if valuable_targets >= 2:
                score += 25.0  # Potential fork opportunity
                
        return score
    
    def _detect_pins(self, board: chess.Board, color: chess.Color) -> float:
        """Basic pin detection"""
        score = 0.0
        
        # Simple pin detection for bishops and rooks
        for piece_type in [chess.BISHOP, chess.ROOK]:
            for square in board.pieces(piece_type, color):
                # Check if this piece pins an enemy piece
                attacks = board.attacks(square)
                for attacked_square in attacks:
                    attacked_piece = board.piece_at(attacked_square)
                    if attacked_piece and attacked_piece.color != color:
                        # Simple pin check - if removing the piece exposes a more valuable piece
                        test_board = board.copy()
                        test_board.remove_piece_at(attacked_square)
                        if test_board.is_attacked_by(color, attacked_square):
                            score += 8.0  # Potential pin
                            
        return score
    
    def _count_hanging_pieces(self, board: chess.Board, color: chess.Color) -> float:
        """Count undefended pieces - critical tactical weakness"""
        hanging_count = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color and piece.piece_type != chess.KING:
                # Check if piece is attacked and undefended
                if board.is_attacked_by(not color, square) and \
                   not board.is_attacked_by(color, square):
                    hanging_count += 1
                    
        return hanging_count
    
    def _endgame_enhanced(self, board: chess.Board, color: chess.Color) -> float:
        """Enhanced endgame logic - improvement over v7.0"""
        score = 0.0
        
        # King activity (same as v7.0 but more weight)
        king_square = board.king(color)
        if king_square:
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            center_distance = abs(king_file - 3.5) + abs(king_rank - 3.5)
            score += (7 - center_distance) * 4  # Double v7.0's weight
            
        # Pawn promotion threats (major v7.0 weakness)
        for square in board.pieces(chess.PAWN, color):
            if color == chess.WHITE:
                distance_to_promotion = 7 - chess.square_rank(square)
            else:
                distance_to_promotion = chess.square_rank(square)
                
            if distance_to_promotion <= 2:  # Close to promotion
                score += 30.0 * (3 - distance_to_promotion)
                
        return score
    
    # Helper methods (similar to v7.0)
    def _is_king_exposed(self, board: chess.Board, color: chess.Color, king_square: int) -> bool:
        """Check if king is dangerously exposed - same as v7.0"""
        if color == chess.WHITE:
            return chess.square_rank(king_square) > 2
        else:
            return chess.square_rank(king_square) < 5
    
    def _is_castled(self, board: chess.Board, color: chess.Color, king_square: int) -> bool:
        """Check if king has castled"""
        if color == chess.WHITE:
            return king_square in [chess.G1, chess.C1]
        else:
            return king_square in [chess.G8, chess.C8]
    
    def _count_attacks_near_king(self, board: chess.Board, color: chess.Color, king_square: int) -> int:
        """Count enemy attacks near the king"""
        attack_count = 0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Check squares around king
        for file_offset in [-1, 0, 1]:
            for rank_offset in [-1, 0, 1]:
                if file_offset == 0 and rank_offset == 0:
                    continue
                    
                check_file = king_file + file_offset
                check_rank = king_rank + rank_offset
                
                if 0 <= check_file <= 7 and 0 <= check_rank <= 7:
                    check_square = chess.square(check_file, check_rank)
                    if board.is_attacked_by(not color, check_square):
                        attack_count += 1
                        
        return attack_count
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """Detect endgame phase - same logic as v7.0"""
        major_pieces = 0
        for color in [chess.WHITE, chess.BLACK]:
            major_pieces += len(board.pieces(chess.QUEEN, color)) * 2
            major_pieces += len(board.pieces(chess.ROOK, color))
        return major_pieces <= 6
