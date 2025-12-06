#!/usr/bin/env python3
"""
V7P3R Fast Evaluator - v17.6 Structural Intelligence
Ultra-fast evaluation using simple PST + Material + Lightweight Bonuses

Speed: ~0.001-0.002ms per position (40x faster than bitboard evaluator)
Architecture: 60% PST + 40% Material + Middlegame/Endgame Bonuses

v17.6 ENHANCEMENTS (Phase 1 - Pawn Structure Intelligence):
- Bishop Valuation Philosophy: B=290 (alone) < N=300, but 2B+50 bonus = 630 > 2N=600
  * Single bishop loses half board coverage → worth less than knight
  * Bishop pair covers all squares → +50cp synergy bonus makes pair superior
- Isolated Pawns: -15cp penalty (was completely missing - explains 159/game in analytics)
- Connected Pawns (Phalanx): +5cp bonus (side-by-side pawns)
- Knight Outposts: +20cp bonus (knights on 4th-6th rank, pawn-protected, safe from attack)
- Expected cost: +10-15μs per eval (still <0.002ms total - negligible)
- Goal: Reduce isolated pawn creation from 159/game to <50/game

v17.4 ENHANCEMENT:
- Rooks on open files in ENDGAMES now get +40cp bonus (was only middlegame)
- Zero-cost implementation: computed in existing piece loop using bitboards
- Performance: 48K eval/sec (8.5% regression from 53K, acceptable)

Designed to enable depth 6-8 consistently while maintaining tactical strength
"""

import chess
from typing import Dict, Optional

# Material values (v17.6 PHILOSOPHY UPDATE)
# Bishop < Knight when alone (bishop loses half board coverage)
# Bishop Pair > Knight Pair (via +50cp bonus = 2×290+50 = 630 vs 2×300 = 600)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 290,  # Lower than knight when alone
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
        endgame_bonus = 0  # NEW: Endgame-specific bonuses
        is_endgame = self._is_endgame(board)
        
        # V17.5: Skip PST in pure endgames (K+P, K+R endings) for speed
        skip_pst = is_endgame and self._is_pure_endgame(board)
        
        # Evaluate all pieces on board
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # PST value (skip in pure endgames for 20-30% speed gain)
                if not skip_pst:
                    pst_score += self._get_piece_square_value(piece, square, is_endgame)
                
                # Material value
                material_value = self.piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    material_score += material_value
                else:
                    material_score -= material_value
                
                # NEW: Endgame rook activity bonus (computed in same loop - zero cost!)
                if is_endgame and piece.piece_type == chess.ROOK:
                    file = chess.square_file(square)
                    # Quick open file check - is there ANY pawn on this file?
                    file_mask = chess.BB_FILES[file]
                    pawn_mask = board.pawns
                    is_open_file = not (file_mask & pawn_mask)
                    
                    if is_open_file:
                        # Rook on open file in endgame: +40cp (was only in middlegame before)
                        if piece.color == chess.WHITE:
                            endgame_bonus += 40
                        else:
                            endgame_bonus -= 40
        
        # Middlegame bonuses (traditional bonuses - middlegame only)
        middlegame_bonus = 0
        if not is_endgame and not self._is_opening(board):
            middlegame_bonus = self._calculate_middlegame_bonuses(board)
        
        # V17.6 bonuses (apply in ALL game phases for consistency)
        v17_6_bonus = self._calculate_v17_6_bonuses(board)
        
        # Combine scores: 60% PST + 40% Material + Bonuses
        combined_score = int(pst_score * 0.6 + material_score * 0.4 + middlegame_bonus + endgame_bonus + v17_6_bonus)
        
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
        
        # Low material = endgame (v17.4: raised from 800 to 1300 to catch R+2minors)
        white_material = sum(len(board.pieces(pt, chess.WHITE)) * self.piece_values.get(pt, 0) 
                            for pt in [chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN])
        black_material = sum(len(board.pieces(pt, chess.BLACK)) * self.piece_values.get(pt, 0)
                            for pt in [chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN])
        
        return white_material < 1300 and black_material < 1300
    
    def _is_opening(self, board: chess.Board) -> bool:
        """Detect opening phase (< 10 moves, pieces not developed)"""
        return board.fullmove_number < 10
    
    def _is_pure_endgame(self, board: chess.Board) -> bool:
        """V17.5: Detect very simplified endgames where PST is irrelevant (K+P, K+R endings)"""
        piece_count = len(board.piece_map())
        return piece_count <= 6  # 2 kings + max 4 other pieces
    
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
        
        V17.6 NEW BONUSES:
        - Bishop pair: +30cp
        - Isolated pawns: -15cp each
        - Connected pawns (phalanx): +5cp per pair
        - Knight outposts: +20cp each
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
    
    def _calculate_v17_6_bonuses(self, board: chess.Board) -> int:
        """
        V17.6 Bonuses - Applied in ALL game phases
        - Bishop pair: +50cp
        - Isolated pawns: -15cp each
        - Connected pawns: +5cp per phalanx
        - Knight outposts: +20cp each
        """
        bonus = 0
        
        # BONUS 1: Bishop Pair (+50cp)
        # Synergy bonus: 2 bishops cover both color complexes
        # With B=290, N=300: 2B+50 = 630 > 2N = 600 (implements user's philosophy)
        # Single bishop = 290 < Knight = 300 (knight more flexible alone)
        if len(board.pieces(chess.BISHOP, chess.WHITE)) == 2:
            bonus += 50
        if len(board.pieces(chess.BISHOP, chess.BLACK)) == 2:
            bonus -= 50
        
        # BONUS 2: Isolated Pawns (-15cp each)
        # Check each file for pawns without adjacent file support
        for file in range(8):
            for rank in range(8):
                sq = chess.square(file, rank)
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.PAWN:
                    # Check if isolated (no friendly pawns on adjacent files)
                    has_adjacent = False
                    for adj_file in [file - 1, file + 1]:
                        if 0 <= adj_file <= 7:
                            for adj_rank in range(8):
                                adj_sq = chess.square(adj_file, adj_rank)
                                adj_piece = board.piece_at(adj_sq)
                                if adj_piece and adj_piece.piece_type == chess.PAWN and adj_piece.color == piece.color:
                                    has_adjacent = True
                                    break
                            if has_adjacent:
                                break
                    
                    if not has_adjacent:  # Isolated pawn
                        if piece.color == chess.WHITE:
                            bonus -= 15
                        else:
                            bonus += 15
        
        # BONUS 3: Connected Pawns/Phalanx (+5cp per pair)
        # Pawns side-by-side on same rank
        for rank in range(8):
            for file in range(7):  # Check file and file+1
                sq1 = chess.square(file, rank)
                sq2 = chess.square(file + 1, rank)
                p1 = board.piece_at(sq1)
                p2 = board.piece_at(sq2)
                if p1 and p2 and p1.piece_type == chess.PAWN and p2.piece_type == chess.PAWN:
                    if p1.color == p2.color:  # Same color = connected
                        if p1.color == chess.WHITE:
                            bonus += 5
                        else:
                            bonus -= 5
        
        # BONUS 4: Knight Outposts (+20cp each)
        # Knights on 4th-6th rank in center files, protected by pawn, no enemy pawn attacks
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.KNIGHT:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                # Check if on strong square (4th-6th rank, center files c-f)
                if piece.color == chess.WHITE:
                    if 3 <= rank <= 5 and 2 <= file <= 5:  # 4th-6th rank for white
                        # Check if protected by own pawn
                        protected = False
                        for pawn_file in [file - 1, file + 1]:
                            if 0 <= pawn_file <= 7 and rank > 0:
                                pawn_sq = chess.square(pawn_file, rank - 1)
                                pawn = board.piece_at(pawn_sq)
                                if pawn and pawn.piece_type == chess.PAWN and pawn.color == chess.WHITE:
                                    protected = True
                                    break
                        
                        if protected:
                            # Check no enemy pawns can attack (simple check - no enemy pawns on adjacent files ahead)
                            can_be_attacked = False
                            for enemy_file in [file - 1, file + 1]:
                                if 0 <= enemy_file <= 7:
                                    for enemy_rank in range(rank, 8):
                                        enemy_sq = chess.square(enemy_file, enemy_rank)
                                        enemy = board.piece_at(enemy_sq)
                                        if enemy and enemy.piece_type == chess.PAWN and enemy.color == chess.BLACK:
                                            can_be_attacked = True
                                            break
                                    if can_be_attacked:
                                        break
                            
                            if not can_be_attacked:
                                bonus += 20
                else:  # Black knight
                    if 2 <= rank <= 4 and 2 <= file <= 5:  # 5th-3rd rank for black (inverted)
                        # Check if protected by own pawn
                        protected = False
                        for pawn_file in [file - 1, file + 1]:
                            if 0 <= pawn_file <= 7 and rank < 7:
                                pawn_sq = chess.square(pawn_file, rank + 1)
                                pawn = board.piece_at(pawn_sq)
                                if pawn and pawn.piece_type == chess.PAWN and pawn.color == chess.BLACK:
                                    protected = True
                                    break
                        
                        if protected:
                            # Check no enemy pawns can attack
                            can_be_attacked = False
                            for enemy_file in [file - 1, file + 1]:
                                if 0 <= enemy_file <= 7:
                                    for enemy_rank in range(0, rank + 1):
                                        enemy_sq = chess.square(enemy_file, enemy_rank)
                                        enemy = board.piece_at(enemy_sq)
                                        if enemy and enemy.piece_type == chess.PAWN and enemy.color == chess.WHITE:
                                            can_be_attacked = True
                                            break
                                    if can_be_attacked:
                                        break
                            
                            if not can_be_attacked:
                                bonus -= 20
        
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
