#!/usr/bin/env python3
"""
V7P3R v9.3 Enhanced Scoring Calculation
Combines v7.0's proven chess knowledge with v9.2's infrastructure improvements
Author: Pat Snyder

TOURNAMENT VALIDATION:
- v7.0 achieved 79.5% (17.5/22) in Engine Battle 20250830
- v7.0 dominated v9.2 4-0 in head-to-head matches  
- Based on C0BR4's proven piece-square table approach
"""

import chess
from typing import Dict


class V7P3RScoringCalculationV93:
    """v9.3 Evaluation: v7.0's tournament-winning chess knowledge + v9.2 infrastructure"""
    
    def __init__(self, piece_values: Dict):
        self.piece_values = piece_values
        self._init_piece_square_tables()
    
    def _init_piece_square_tables(self):
        """Initialize piece-square tables based on tournament-winning v7.0/C0BR4 approach"""
        
        # Pawn table - emphasizes central control and safe advancement
        self.pawn_table = [
             0,  0,  0,  0,  0,  0,  0,  0,  # 8th rank (promotion)
            50, 50, 50, 50, 50, 50, 50, 50,  # 7th rank - promotion threats
            10, 10, 20, 30, 30, 20, 10, 10,  # 6th rank - advanced pawns
             5,  5, 10, 25, 25, 10,  5,  5,  # 5th rank - central control
             0,  0,  0, 20, 20,  0,  0,  0,  # 4th rank - center pawns valuable
             5, -5,-10,  0,  0,-10, -5,  5,  # 3rd rank - avoid early moves
             5, 10, 10,-20,-20, 10, 10,  5,  # 2nd rank - stay protected
             0,  0,  0,  0,  0,  0,  0,  0   # 1st rank
        ]
        
        # Knight table - rewards centralization (C0BR4 style)
        self.knight_table = [
           -50,-40,-30,-30,-30,-30,-40,-50,  # 8th rank - poor squares
           -40,-20,  0,  0,  0,  0,-20,-40,  # 7th rank
           -30,  0, 10, 15, 15, 10,  0,-30,  # 6th rank
           -30,  5, 15, 20, 20, 15,  5,-30,  # 5th rank - excellent centralization
           -30,  0, 15, 20, 20, 15,  0,-30,  # 4th rank
           -30,  5, 10, 15, 15, 10,  5,-30,  # 3rd rank
           -40,-20,  0,  5,  5,  0,-20,-40,  # 2nd rank
           -50,-40,-30,-30,-30,-30,-40,-50   # 1st rank - poor development
        ]
        
        # Bishop table - emphasizes long diagonals
        self.bishop_table = [
           -20,-10,-10,-10,-10,-10,-10,-20,  # 8th rank
           -10,  0,  0,  0,  0,  0,  0,-10,  # 7th rank
           -10,  0,  5, 10, 10,  5,  0,-10,  # 6th rank
           -10,  5,  5, 10, 10,  5,  5,-10,  # 5th rank
           -10,  0, 10, 10, 10, 10,  0,-10,  # 4th rank - central control
           -10, 10, 10, 10, 10, 10, 10,-10,  # 3rd rank - long diagonals
           -10,  5,  0,  0,  0,  0,  5,-10,  # 2nd rank
           -20,-10,-10,-10,-10,-10,-10,-20   # 1st rank
        ]
        
        # Rook table - encourages open files and back rank activity
        self.rook_table = [
             0,  0,  0,  0,  0,  0,  0,  0,  # 8th rank - back rank control
            10, 10, 10, 10, 10, 10, 10, 10,  # 7th rank - 7th rank invasion
             0,  0,  0,  0,  0,  0,  0,  0,  # 6th rank
             0,  0,  0,  0,  0,  0,  0,  0,  # 5th rank
             0,  0,  0,  0,  0,  0,  0,  0,  # 4th rank
             0,  0,  0,  0,  0,  0,  0,  0,  # 3rd rank
             0,  0,  0,  0,  0,  0,  0,  0,  # 2nd rank
             5,  5,  5,  5,  5,  5,  5,  5   # 1st rank - back rank activity
        ]
        
        # Queen table - balanced central activity with safety
        self.queen_table = [
           -20,-10,-10, -5, -5,-10,-10,-20,  # 8th rank
           -10,  0,  0,  0,  0,  0,  0,-10,  # 7th rank
           -10,  0,  5,  5,  5,  5,  0,-10,  # 6th rank
            -5,  0,  5,  5,  5,  5,  0, -5,  # 5th rank
             0,  0,  5,  5,  5,  5,  0, -5,  # 4th rank
           -10,  5,  5,  5,  5,  5,  0,-10,  # 3rd rank
           -10,  0,  5,  0,  0,  0,  0,-10,  # 2nd rank - avoid early development
           -20,-10,-10, -5, -5,-10,-10,-20   # 1st rank
        ]
        
        # King middlegame table - emphasizes safety (castled positions)
        self.king_table_middlegame = [
           -30,-40,-40,-50,-50,-40,-40,-30,  # 8th rank
           -30,-40,-40,-50,-50,-40,-40,-30,  # 7th rank
           -30,-40,-40,-50,-50,-40,-40,-30,  # 6th rank
           -30,-40,-40,-50,-50,-40,-40,-30,  # 5th rank
           -20,-30,-30,-40,-40,-30,-30,-20,  # 4th rank
           -10,-20,-20,-20,-20,-20,-20,-10,  # 3rd rank
            20, 20,  0,  0,  0,  0, 20, 20,  # 2nd rank - encouraged to stay
            20, 30, 10,  0,  0, 10, 30, 20   # 1st rank - castling positions
        ]
        
        # King endgame table - encourages centralization
        self.king_table_endgame = [
           -50,-40,-30,-20,-20,-30,-40,-50,  # 8th rank
           -30,-20,-10,  0,  0,-10,-20,-30,  # 7th rank
           -30,-10, 20, 30, 30, 20,-10,-30,  # 6th rank - active king
           -30,-10, 30, 40, 40, 30,-10,-30,  # 5th rank - centralized
           -30,-10, 30, 40, 40, 30,-10,-30,  # 4th rank
           -30,-10, 20, 30, 30, 20,-10,-30,  # 3rd rank
           -30,-30,  0,  0,  0,  0,-30,-30,  # 2nd rank
           -50,-30,-30,-30,-30,-30,-30,-50   # 1st rank
        ]
    
    def calculate_score_optimized(self, board: chess.Board, color: chess.Color, endgame_factor: float = 0.0) -> float:
        """
        v9.3 Evaluation: Tournament-proven scoring with proper centipawn scaling
        Returns evaluation in proper centipawn range (-1000 to +1000)
        """
        score = 0.0
        
        # Detect game phase for king table selection
        is_endgame = self._is_endgame(board)
        
        # 1. Material evaluation with piece-square table bonuses
        score += self._material_and_position_score(board, color, is_endgame)
        
        # 2. King safety (critical for tournament play)
        score += self._king_safety(board, color, is_endgame)
        
        # 3. Castling evaluation (v7.0 strength)
        score += self._castling_bonus(board, color)
        
        # 4. Pawn structure (central control emphasis)
        score += self._pawn_structure(board, color)
        
        # 5. Piece development and coordination
        score += self._piece_coordination(board, color)
        
        # 6. Center control (tournament-proven importance)
        score += self._center_control(board, color)
        
        # 7. Endgame-specific logic
        if is_endgame:
            score += self._endgame_logic(board, color)
        
        # CRITICAL: Normalize to proper centipawn range for UCI compliance
        return max(-999, min(999, score))
    
    def _material_and_position_score(self, board: chess.Board, color: chess.Color, is_endgame: bool) -> float:
        """Combined material and positional evaluation using piece-square tables"""
        score = 0.0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                # Material value
                if piece.piece_type in self.piece_values:
                    score += self.piece_values[piece.piece_type]
                
                # Positional bonus from piece-square tables
                flipped_square = square if color == chess.WHITE else chess.square_mirror(square)
                
                if piece.piece_type == chess.PAWN:
                    score += self.pawn_table[flipped_square] / 100.0  # Scale for centipawns
                elif piece.piece_type == chess.KNIGHT:
                    score += self.knight_table[flipped_square] / 100.0
                elif piece.piece_type == chess.BISHOP:
                    score += self.bishop_table[flipped_square] / 100.0
                elif piece.piece_type == chess.ROOK:
                    score += self.rook_table[flipped_square] / 100.0
                elif piece.piece_type == chess.QUEEN:
                    score += self.queen_table[flipped_square] / 100.0
                elif piece.piece_type == chess.KING:
                    if is_endgame:
                        score += self.king_table_endgame[flipped_square] / 100.0
                    else:
                        score += self.king_table_middlegame[flipped_square] / 100.0
        
        return score
    
    def _king_safety(self, board: chess.Board, color: chess.Color, is_endgame: bool) -> float:
        """Enhanced king safety evaluation"""
        score = 0.0
        king_square = board.king(color)
        
        if not king_square:
            return -500.0  # Severe penalty for missing king
        
        if not is_endgame:
            # Middlegame: Penalty for exposed king
            if self._is_king_exposed(board, color, king_square):
                score -= 30.0
            
            # Bonus for castling rights preserved
            if board.has_kingside_castling_rights(color):
                score += 5.0
            if board.has_queenside_castling_rights(color):
                score += 3.0
        
        return score
    
    def _castling_bonus(self, board: chess.Board, color: chess.Color) -> float:
        """Tournament-proven castling evaluation (v7.0 strength)"""
        score = 0.0
        king_square = board.king(color)
        
        if king_square:
            # Significant bonus for castled king (v7.0 style)
            if color == chess.WHITE:
                if king_square in [chess.G1, chess.C1]:
                    score += 25.0  # Castled king bonus
            else:
                if king_square in [chess.G8, chess.C8]:
                    score += 25.0
        
        return score
    
    def _pawn_structure(self, board: chess.Board, color: chess.Color) -> float:
        """Enhanced pawn structure evaluation"""
        score = 0.0
        pawns = list(board.pieces(chess.PAWN, color))
        
        # Doubled pawn penalty
        files_with_pawns = [chess.square_file(pawn) for pawn in pawns]
        for file in range(8):
            pawn_count = files_with_pawns.count(file)
            if pawn_count > 1:
                score -= 10.0 * (pawn_count - 1)  # Penalty for each extra pawn
        
        # Isolated pawn penalty
        for pawn in pawns:
            file = chess.square_file(pawn)
            adjacent_files = [file - 1, file + 1]
            has_support = any(
                chess.square_file(other_pawn) in adjacent_files 
                for other_pawn in pawns if other_pawn != pawn
            )
            if not has_support:
                score -= 15.0  # Isolated pawn penalty
        
        return score
    
    def _piece_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Piece coordination and development bonuses"""
        score = 0.0
        
        # Rook coordination (v7.0 style)
        rooks = list(board.pieces(chess.ROOK, color))
        if len(rooks) >= 2:
            score += 10.0  # Both rooks bonus
            
            # Connected rooks bonus
            if len(rooks) == 2:
                rook1, rook2 = rooks
                if chess.square_rank(rook1) == chess.square_rank(rook2):
                    score += 15.0  # Rooks on same rank
                elif chess.square_file(rook1) == chess.square_file(rook2):
                    score += 10.0  # Rooks on same file
        
        # Bishop pair bonus
        bishops = list(board.pieces(chess.BISHOP, color))
        if len(bishops) >= 2:
            score += 20.0  # Bishop pair is strong
        
        # Knight-bishop coordination
        knights = list(board.pieces(chess.KNIGHT, color))
        if len(knights) >= 1 and len(bishops) >= 1:
            score += 5.0  # Minor piece coordination
        
        return score
    
    def _center_control(self, board: chess.Board, color: chess.Color) -> float:
        """Enhanced center control evaluation (tournament-critical)"""
        score = 0.0
        
        # Central squares (v7.0 priority)
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        extended_center = [chess.C3, chess.C4, chess.C5, chess.C6,
                          chess.D3, chess.D6, chess.E3, chess.E6,
                          chess.F3, chess.F4, chess.F5, chess.F6]
        
        # Central occupation bonus
        for square in center_squares:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                if piece.piece_type == chess.PAWN:
                    score += 15.0  # Central pawn very valuable
                elif piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    score += 10.0  # Centralized minor pieces
                else:
                    score += 5.0   # Other pieces in center
        
        # Extended center control
        for square in extended_center:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    score += 3.0  # Extended center control
        
        return score
    
    def _endgame_logic(self, board: chess.Board, color: chess.Color) -> float:
        """Enhanced endgame evaluation"""
        score = 0.0
        
        # King activity bonus (already handled in piece-square tables)
        
        # Passed pawn evaluation
        pawns = list(board.pieces(chess.PAWN, color))
        for pawn in pawns:
            if self._is_passed_pawn(board, pawn, color):
                rank = chess.square_rank(pawn)
                if color == chess.WHITE:
                    advancement = rank  # 0-7, higher is more advanced
                else:
                    advancement = 7 - rank
                score += advancement * 5.0  # Passed pawn bonus increases with advancement
        
        return score
    
    def _is_king_exposed(self, board: chess.Board, color: chess.Color, king_square: int) -> bool:
        """Check if king is dangerously exposed"""
        king_rank = chess.square_rank(king_square)
        
        # King too far from back rank is exposed
        if color == chess.WHITE:
            return king_rank > 2  # White king beyond 3rd rank
        else:
            return king_rank < 5  # Black king beyond 6th rank
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """Detect endgame phase (v7.0 logic)"""
        # Count major pieces (queens and rooks)
        major_pieces = 0
        for color in [chess.WHITE, chess.BLACK]:
            major_pieces += len(board.pieces(chess.QUEEN, color)) * 2
            major_pieces += len(board.pieces(chess.ROOK, color))
        
        return major_pieces <= 6  # Endgame threshold
    
    def _is_passed_pawn(self, board: chess.Board, pawn_square: int, color: chess.Color) -> bool:
        """Check if pawn is passed (no enemy pawns blocking)"""
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Check files: same file and adjacent files
        files_to_check = [f for f in [file - 1, file, file + 1] if 0 <= f <= 7]
        
        enemy_pawns = list(board.pieces(chess.PAWN, not color))
        
        if color == chess.WHITE:
            # Check squares ahead of white pawn
            blocking_squares = [
                chess.square(f, r) for f in files_to_check 
                for r in range(rank + 1, 8)
            ]
        else:
            # Check squares ahead of black pawn
            blocking_squares = [
                chess.square(f, r) for f in files_to_check 
                for r in range(0, rank)
            ]
        
        # If any enemy pawn blocks this pawn's path, it's not passed
        return not any(pawn in blocking_squares for pawn in enemy_pawns)
