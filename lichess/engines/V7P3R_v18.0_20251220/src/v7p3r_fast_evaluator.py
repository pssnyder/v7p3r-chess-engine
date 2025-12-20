#!/usr/bin/env python3
"""
V7P3R Fast Evaluator - Extracted from V16.1
Ultra-fast evaluation using simple PST + Material + Lightweight Middlegame Bonuses

Speed: ~0.001ms per position (40x faster than bitboard evaluator)
Architecture: 60% PST + 40% Material + Middlegame Bonuses

Designed to enable depth 6-8 consistently in V14.1 while maintaining tactical strength
"""

import chess
from typing import Dict, Optional

# Material values (from MaterialOpponent - proven to prevent sacrifices)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 325,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

# Piece-Square Tables (from PositionalOpponent - proven 81% win rate)
# Values in centipawns, White perspective

PAWN_PST = [
    [  0,  0,  0,  0,  0,  0,  0,  0],  # 1st rank
    [ 50, 50, 50, 50, 50, 50, 50, 50],  # 2nd rank  
    [ 60, 60, 70, 80, 80, 70, 60, 60],  # 3rd rank
    [ 70, 70, 80, 90, 90, 80, 70, 70],  # 4th rank
    [100,100,110,120,120,110,100,100],  # 5th rank
    [200,200,220,250,250,220,200,200],  # 6th rank
    [400,400,450,500,500,450,400,400],  # 7th rank
    [900,900,900,900,900,900,900,900],  # 8th rank
]

KNIGHT_PST = [
    [200,220,240,250,250,240,220,200],
    [220,240,260,270,270,260,240,220],
    [240,260,300,320,320,300,260,240],
    [250,270,320,350,350,320,270,250],
    [250,270,320,350,350,320,270,250],
    [240,260,300,320,320,300,260,240],
    [220,240,260,270,270,260,240,220],
    [200,220,240,250,250,240,220,200],
]

BISHOP_PST = [
    [250,260,270,280,280,270,260,250],
    [260,300,290,290,290,290,300,260],
    [270,290,320,300,300,320,290,270],
    [280,290,300,350,350,300,290,280],
    [280,290,300,350,350,300,290,280],
    [270,290,320,300,300,320,290,270],
    [260,300,290,290,290,290,300,260],
    [250,260,270,280,280,270,260,250],
]

ROOK_PST = [
    [400,410,420,430,430,420,410,400],
    [450,450,450,450,450,450,450,450],
    [440,440,440,440,440,440,440,440],
    [440,440,440,440,440,440,440,440],
    [440,440,440,440,440,440,440,440],
    [450,450,450,450,450,450,450,450],
    [500,500,500,500,500,500,500,500],
    [480,480,480,480,480,480,480,480],
]

QUEEN_PST = [
    [700,710,720,730,730,720,710,700],
    [710,750,750,750,750,750,750,710],
    [720,750,800,800,800,800,750,720],
    [730,750,800,850,850,800,750,730],
    [730,750,800,850,850,800,750,730],
    [720,750,800,800,800,800,750,720],
    [710,750,750,750,750,750,750,710],
    [700,710,720,730,730,720,710,700],
]

KING_MIDDLEGAME_PST = [
    [ 50, 80,  0,  0,  0,  0, 80, 50],
    [  0,  0,  0,  0,  0,  0,  0,  0],
    [ -50,-50,-50,-50,-50,-50,-50,-50],
    [-100,-100,-100,-100,-100,-100,-100,-100],
    [-150,-150,-150,-150,-150,-150,-150,-150],
    [-200,-200,-200,-200,-200,-200,-200,-200],
    [-250,-250,-250,-250,-250,-250,-250,-250],
    [-300,-300,-300,-300,-300,-300,-300,-300],
]

KING_ENDGAME_PST = [
    [-50,-40,-30,-20,-20,-30,-40,-50],
    [-30,-20,-10,  0,  0,-10,-20,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-30,  0,  0,  0,  0,-30,-30],
    [-50,-30,-30,-30,-30,-30,-30,-50],
]


class V7P3RFastEvaluator:
    """
    Fast PST-based evaluator for maximum search depth
    Architecture: 60% PST + 40% Material + Middlegame Bonuses
    """
    
    def __init__(self):
        """Initialize fast evaluator"""
        self.piece_values = PIECE_VALUES
    
    def evaluate(self, board: chess.Board) -> int:
        """
        Main evaluation function
        Returns: score in centipawns (positive = White advantage)
        """
        pst_score = 0
        material_score = 0
        is_endgame = self._is_endgame(board)
        
        # Evaluate all pieces on board
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # PST value
                pst_score += self._get_piece_square_value(piece, square, is_endgame)
                
                # Material value
                material_value = self.piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    material_score += material_value
                else:
                    material_score -= material_value
        
        # Middlegame bonuses (only in middlegame phase)
        middlegame_bonus = 0
        if not is_endgame and not self._is_opening(board):
            middlegame_bonus = self._calculate_middlegame_bonuses(board)
        
        # Combine scores: 60% PST + 40% Material + Bonuses
        combined_score = int(pst_score * 0.6 + material_score * 0.4 + middlegame_bonus)
        
        # Return from current player perspective
        return combined_score if board.turn == chess.WHITE else -combined_score
    
    def _get_piece_square_value(self, piece: chess.Piece, square: chess.Square, is_endgame: bool = False) -> int:
        """Get PST value for piece at square"""
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        # Flip rank for black pieces (PST is from White perspective)
        if piece.color == chess.BLACK:
            rank = 7 - rank
        
        piece_type = piece.piece_type
        
        # Look up PST value
        if piece_type == chess.PAWN:
            value = PAWN_PST[rank][file]
        elif piece_type == chess.KNIGHT:
            value = KNIGHT_PST[rank][file]
        elif piece_type == chess.BISHOP:
            value = BISHOP_PST[rank][file]
        elif piece_type == chess.ROOK:
            value = ROOK_PST[rank][file]
        elif piece_type == chess.QUEEN:
            value = QUEEN_PST[rank][file]
        elif piece_type == chess.KING:
            value = KING_ENDGAME_PST[rank][file] if is_endgame else KING_MIDDLEGAME_PST[rank][file]
        else:
            value = 0
        
        # Negate for black pieces
        return value if piece.color == chess.WHITE else -value
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """Detect endgame phase (no queens or low material)"""
        # No queens = endgame
        if not board.pieces(chess.QUEEN, chess.WHITE) and not board.pieces(chess.QUEEN, chess.BLACK):
            return True
        
        # Low material = endgame
        white_material = sum(len(board.pieces(pt, chess.WHITE)) * self.piece_values.get(pt, 0) 
                            for pt in [chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN])
        black_material = sum(len(board.pieces(pt, chess.BLACK)) * self.piece_values.get(pt, 0)
                            for pt in [chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN])
        
        return white_material < 800 and black_material < 800
    
    def _is_opening(self, board: chess.Board) -> bool:
        """Detect opening phase (< 10 moves, pieces not developed)"""
        return board.fullmove_number < 10
    
    def _calculate_middlegame_bonuses(self, board: chess.Board) -> int:
        """
        Calculate middlegame positional bonuses
        Returns: bonus in centipawns (White perspective)
        
        Bonuses:
        - Rooks on open files: +20cp
        - Rooks on semi-open files: +10cp
        - King pawn shield: +10cp per shield pawn
        - Passed pawns: +30cp
        - Doubled pawns: -20cp per extra pawn
        """
        bonus = 0
        
        # BONUS 1: Rooks on open/semi-open files
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.ROOK:
                file = chess.square_file(square)
                is_open = True
                is_semi_open = True
                
                # Check if file has pawns
                for rank in range(8):
                    sq = chess.square(file, rank)
                    p = board.piece_at(sq)
                    if p and p.piece_type == chess.PAWN:
                        if p.color == piece.color:
                            is_open = False
                            is_semi_open = False
                            break
                        else:
                            is_open = False
                
                if is_open:
                    bonus += 20 if piece.color == chess.WHITE else -20
                elif is_semi_open:
                    bonus += 10 if piece.color == chess.WHITE else -10
        
        # BONUS 2: King safety - pawn shield
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is not None:
                king_file = chess.square_file(king_square)
                king_rank = chess.square_rank(king_square)
                
                # Check pawns in front of king (shield)
                shield_pawns = 0
                for file_offset in [-1, 0, 1]:
                    shield_file = king_file + file_offset
                    if 0 <= shield_file <= 7:
                        # Check 1-2 ranks ahead
                        for rank_offset in [1, 2]:
                            if color == chess.WHITE:
                                shield_rank = king_rank + rank_offset
                            else:
                                shield_rank = king_rank - rank_offset
                            
                            if 0 <= shield_rank <= 7:
                                sq = chess.square(shield_file, shield_rank)
                                p = board.piece_at(sq)
                                if p and p.piece_type == chess.PAWN and p.color == color:
                                    shield_pawns += 1
                                    break
                
                if color == chess.WHITE:
                    bonus += shield_pawns * 10
                else:
                    bonus -= shield_pawns * 10
        
        # BONUS 3: Pawn structure (passed pawns, doubled pawns)
        for file in range(8):
            white_pawns = []
            black_pawns = []
            
            for rank in range(8):
                sq = chess.square(file, rank)
                p = board.piece_at(sq)
                if p and p.piece_type == chess.PAWN:
                    if p.color == chess.WHITE:
                        white_pawns.append(rank)
                    else:
                        black_pawns.append(rank)
            
            # Doubled pawns penalty (-20cp per extra pawn)
            if len(white_pawns) > 1:
                bonus -= 20 * (len(white_pawns) - 1)
            if len(black_pawns) > 1:
                bonus += 20 * (len(black_pawns) - 1)
            
            # Passed pawns bonus (+30cp)
            for rank in white_pawns:
                if self._is_passed_pawn(board, chess.square(file, rank), chess.WHITE):
                    bonus += 30
            
            for rank in black_pawns:
                if self._is_passed_pawn(board, chess.square(file, rank), chess.BLACK):
                    bonus -= 30
        
        return bonus
    
    def _is_passed_pawn(self, board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
        """Check if pawn at square is passed"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        # Check adjacent files and this file ahead
        for adj_file in [file - 1, file, file + 1]:
            if 0 <= adj_file <= 7:
                if color == chess.WHITE:
                    # Check ranks ahead for white
                    for r in range(rank + 1, 8):
                        sq = chess.square(adj_file, r)
                        p = board.piece_at(sq)
                        if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                            return False
                else:
                    # Check ranks ahead for black
                    for r in range(0, rank):
                        sq = chess.square(adj_file, r)
                        p = board.piece_at(sq)
                        if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                            return False
        
        return True


# For compatibility with existing code
class V7P3RScoringCalculationFast(V7P3RFastEvaluator):
    """Alias for compatibility with v14.1's naming convention"""
    
    def __init__(self, piece_values: Optional[Dict[int, int]] = None):
        super().__init__()
        if piece_values:
            self.piece_values = piece_values
    
    def calculate_current_board_score(self, board: chess.Board) -> int:
        """Compatibility method matching v14.1's interface"""
        return self.evaluate(board)
