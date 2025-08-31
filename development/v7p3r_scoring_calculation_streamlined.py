#!/usr/bin/env python3
"""
V7P3R Chess Engine Streamlined Scoring Calculation
Performance-optimized version without complex tactical detection
"""

import chess
from typing import Dict, List, Optional, Tuple


class V7P3RScoringCalculationClean:
    """
    Streamlined scoring calculation for V10 performance
    Keeps: Basic evaluation, piece defense, endgame improvements
    Removes: Complex tactical pattern detection (pin, fork, skewer, etc.)
    """
    
    def __init__(self, piece_values: Dict[int, int]):
        """Initialize with piece values"""
        self.piece_values = piece_values
        
        # Center control squares
        self.center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        self.extended_center = [
            chess.C3, chess.C4, chess.C5, chess.C6,
            chess.D3, chess.D4, chess.D5, chess.D6,
            chess.E3, chess.E4, chess.E5, chess.E6,
            chess.F3, chess.F4, chess.F5, chess.F6
        ]
        
        # Edge squares for endgame
        self.edge_squares = [
            chess.A1, chess.A2, chess.A3, chess.A4, chess.A5, chess.A6, chess.A7, chess.A8,
            chess.H1, chess.H2, chess.H3, chess.H4, chess.H5, chess.H6, chess.H7, chess.H8,
            chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1,
            chess.B8, chess.C8, chess.D8, chess.E8, chess.F8, chess.G8
        ]
    
    def calculate_score_optimized(self, board: chess.Board, color: chess.Color, endgame_factor: float = 0.0) -> float:
        """
        Streamlined evaluation - fast and effective
        """
        score = 0.0
        
        # Core evaluation components
        score += self._material_score(board, color)
        score += self._king_safety(board, color)
        score += self._piece_development(board, color)
        score += self._castling_bonus(board, color)
        score += self._rook_coordination(board, color)
        score += self._center_control(board, color)
        
        # Lightweight piece defense
        score += self._piece_defense_light(board, color)
        
        # Enhanced endgame
        if self._is_endgame(board):
            score += self._endgame_enhanced(board, color)
        else:
            score += self._endgame_logic(board, color)
            
        return score
    
    def _material_score(self, board: chess.Board, color: chess.Color) -> float:
        """Material count for given color"""
        score = 0.0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            count = len(board.pieces(piece_type, color))
            score += count * self.piece_values[piece_type]
        return score
    
    def _king_safety(self, board: chess.Board, color: chess.Color) -> float:
        """King safety evaluation"""
        score = 0.0
        king_square = board.king(color)
        
        if king_square is None:
            return -1000.0  # No king = very bad
        
        # Basic king safety - avoid early king moves
        if not self._is_endgame(board):
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            
            # Prefer castled king positions
            if color == chess.WHITE:
                if king_rank == 0:  # Back rank
                    if king_file in [1, 2, 6, 7]:  # Castled positions
                        score += 25.0
            else:
                if king_rank == 7:  # Back rank
                    if king_file in [1, 2, 6, 7]:  # Castled positions
                        score += 25.0
        
        return score
    
    def _piece_development(self, board: chess.Board, color: chess.Color) -> float:
        """Piece development scoring"""
        score = 0.0
        
        # Knights developed
        for knight_square in board.pieces(chess.KNIGHT, color):
            if knight_square in self.extended_center:
                score += 15.0
            # Avoid edge development
            if chess.square_file(knight_square) in [0, 7] or chess.square_rank(knight_square) in [0, 7]:
                score -= 10.0
        
        # Bishops developed
        for bishop_square in board.pieces(chess.BISHOP, color):
            # Long diagonals
            if bishop_square in [chess.A1, chess.H8, chess.A8, chess.H1]:
                score += 10.0
            elif bishop_square in self.extended_center:
                score += 12.0
        
        return score
    
    def _castling_bonus(self, board: chess.Board, color: chess.Color) -> float:
        """Castling rights and execution bonus"""
        score = 0.0
        
        # Bonus for having castling rights
        if board.has_kingside_castling_rights(color):
            score += 15.0
        if board.has_queenside_castling_rights(color):
            score += 10.0
        
        # Bonus for having castled
        king_square = board.king(color)
        if king_square:
            if color == chess.WHITE:
                if king_square in [chess.G1, chess.C1]:  # Castled
                    score += 20.0
            else:
                if king_square in [chess.G8, chess.C8]:  # Castled
                    score += 20.0
        
        return score
    
    def _rook_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Rook coordination and activity"""
        score = 0.0
        rooks = list(board.pieces(chess.ROOK, color))
        
        # Connected rooks
        if len(rooks) >= 2:
            for i, rook1 in enumerate(rooks):
                for rook2 in rooks[i+1:]:
                    # Same rank or file
                    if (chess.square_rank(rook1) == chess.square_rank(rook2) or
                        chess.square_file(rook1) == chess.square_file(rook2)):
                        # Check if they can see each other
                        if self._can_pieces_see_each_other(board, rook1, rook2):
                            score += 15.0
        
        # Open files
        for rook_square in rooks:
            file_idx = chess.square_file(rook_square)
            if self._is_open_file(board, file_idx, color):
                score += 20.0
            elif self._is_semi_open_file(board, file_idx, color):
                score += 10.0
        
        return score
    
    def _center_control(self, board: chess.Board, color: chess.Color) -> float:
        """Center control evaluation"""
        score = 0.0
        
        for square in self.center_squares:
            attackers = board.attackers(color, square)
            score += len(attackers) * 5.0
        
        for square in self.extended_center:
            attackers = board.attackers(color, square)
            score += len(attackers) * 2.0
        
        return score
    
    def _piece_defense_light(self, board: chess.Board, color: chess.Color) -> float:
        """Lightweight piece defense heuristics"""
        score = 0.0
        
        # Simple defense bonus
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                defenders = len(board.attackers(color, square))
                if defenders > 0:
                    score += defenders * 2.0
                else:
                    # Penalty for undefended pieces
                    if piece.piece_type != chess.PAWN:
                        score -= 5.0
        
        return score
    
    def _endgame_enhanced(self, board: chess.Board, color: chess.Color) -> float:
        """Enhanced endgame evaluation"""
        score = 0.0
        
        our_king = board.king(color)
        enemy_king = board.king(not color)
        
        if our_king is None or enemy_king is None:
            return score
        
        # King centralization
        king_file = chess.square_file(our_king)
        king_rank = chess.square_rank(our_king)
        center_distance = abs(king_file - 3.5) + abs(king_rank - 3.5)
        score += (7 - center_distance) * 3.0
        
        # Drive enemy king to edge
        enemy_file = chess.square_file(enemy_king)
        enemy_rank = chess.square_rank(enemy_king)
        enemy_edge_distance = min(enemy_file, 7 - enemy_file, enemy_rank, 7 - enemy_rank)
        score += (3 - enemy_edge_distance) * 10.0
        
        # King proximity (good in endgame)
        king_distance = max(abs(chess.square_file(our_king) - chess.square_file(enemy_king)),
                           abs(chess.square_rank(our_king) - chess.square_rank(enemy_king)))
        score += (8 - king_distance) * 5.0
        
        return score
    
    def _endgame_logic(self, board: chess.Board, color: chess.Color) -> float:
        """Basic endgame logic for non-endgame positions"""
        score = 0.0
        
        # Pawn advancement
        for pawn_square in board.pieces(chess.PAWN, color):
            rank = chess.square_rank(pawn_square)
            if color == chess.WHITE:
                advancement = rank - 1
            else:
                advancement = 6 - rank
            score += advancement * 2.0
        
        return score
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """Determine if position is endgame"""
        # Count material
        total_pieces = 0
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            total_pieces += len(board.pieces(piece_type, chess.WHITE))
            total_pieces += len(board.pieces(piece_type, chess.BLACK))
        
        # Endgame if fewer than 8 pieces or no queens
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        return total_pieces <= 8 or queens == 0
    
    def _can_pieces_see_each_other(self, board: chess.Board, square1: int, square2: int) -> bool:
        """Check if two pieces can see each other (no pieces between)"""
        file1, rank1 = chess.square_file(square1), chess.square_rank(square1)
        file2, rank2 = chess.square_file(square2), chess.square_rank(square2)
        
        # Must be on same rank, file, or diagonal
        if file1 != file2 and rank1 != rank2 and abs(file1 - file2) != abs(rank1 - rank2):
            return False
        
        # Check for pieces between
        file_dir = 0 if file1 == file2 else (1 if file2 > file1 else -1)
        rank_dir = 0 if rank1 == rank2 else (1 if rank2 > rank1 else -1)
        
        current_file, current_rank = file1 + file_dir, rank1 + rank_dir
        
        while current_file != file2 or current_rank != rank2:
            check_square = chess.square(current_file, current_rank)
            if board.piece_at(check_square) is not None:
                return False
            current_file += file_dir
            current_rank += rank_dir
        
        return True
    
    def _is_open_file(self, board: chess.Board, file_idx: int, color: chess.Color) -> bool:
        """Check if file is open (no pawns)"""
        for rank in range(8):
            square = chess.square(file_idx, rank)
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                return False
        return True
    
    def _is_semi_open_file(self, board: chess.Board, file_idx: int, color: chess.Color) -> bool:
        """Check if file is semi-open (no our pawns)"""
        for rank in range(8):
            square = chess.square(file_idx, rank)
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                return False
        return True
