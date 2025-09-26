#!/usr/bin/env python3
"""
=== PAT'S V7P3R BITBOARD EVALUATION SYSTEM v12.3 ===
Ultra-fast evaluation using bitboard operations for tens of thousands of NPS
Integrated king safety and advanced pawn evaluation for comprehensive positional assessment

PERFORMANCE TARGET: 20,000+ evaluations per second (10x faster than piece-square tables)

=== KEY EVALUATION COMPONENTS (Pat's Quick Reference) ===
1. MATERIAL BALANCE: Standard piece values with fast bit counting
2. CENTER CONTROL: Pawn and piece presence in center squares  
3. PIECE DEVELOPMENT: Opening development bonuses/penalties
4. KING SAFETY: Pawn shield, castling rights, king exposure (lines 200-350)
5. ADVANCED PAWN STRUCTURE: Passed, doubled, isolated, backward pawns (lines 400-600)
6. ENDGAME FACTORS: King activity, passed pawn promotion potential

=== BITBOARD MAGIC (Pat's Performance Secrets) ===
- Pre-calculated attack tables for instant piece mobility
- Bitwise operations replace loops (AND, OR, XOR, shifts)
- Population count (bin().count('1')) for instant piece counting
- Mask-based pattern detection (files, ranks, diagonals)

=== TUNING PARAMETERS (Pat's Adjustment Points) ===
- Piece activity bonuses: lines 180-220
- King safety factors: lines 50-80  
- Pawn structure penalties: lines 60-90
- Center control weights: lines 160-180
- Development penalties: lines 200-240
"""

import chess
from typing import Dict, Tuple


class V7P3RBitboardEvaluator:
    """
    === PAT'S HIGH-PERFORMANCE BITBOARD EVALUATION SYSTEM ===
    High-performance bitboard-based evaluation system with integrated advanced features
    Uses bitwise operations for 10x+ speed improvement over traditional evaluation
    
    ARCHITECTURE:
    - Bitboard constants: Pre-calculated masks for ranks, files, centers, etc.
    - Attack tables: Knight, king, pawn attacks for every square
    - Pattern detection: Fast bitwise operations for tactical/positional patterns
    - Integrated features: All evaluation in one pass for maximum speed
    """
    
    def __init__(self, piece_values: Dict[int, int]):
        self.piece_values = piece_values
        
        # Pre-calculate bitboard masks for ultra-fast evaluation
        self._init_bitboard_constants()
        self._init_attack_tables()
        
        # V12.3: Integrated advanced evaluation constants
        self._init_advanced_evaluation_constants()
    
    def _init_advanced_evaluation_constants(self):
        """
        === PAT'S ADVANCED EVALUATION CONSTANTS ===
        V12.3: Initialize constants for integrated king safety and pawn evaluation
        """
        
        # === PAT'S KING SAFETY TUNING PARAMETERS ===
        self.pawn_shelter_bonus = [0, 8, 16, 24, 32]  # By number of shelter pawns (TUNE: increase for safer play)
        self.castling_rights_bonus = 25               # Bonus for having castling rights (TUNE: 20-35 range)
        self.king_exposure_penalty = 20               # Penalty for exposed king (TUNE: 15-30 range)
        self.escape_square_bonus = 6                  # Bonus for king escape squares (TUNE: 4-10 range)
        self.king_activity_bonus = 4                  # For endgame king activity (TUNE: 2-8 range)
        
        # === PAT'S ADVANCED PAWN STRUCTURE CONSTANTS ===
        # MAJOR TUNING AREA: These values significantly impact playing style
        self.passed_pawn_bonus = [0, 15, 25, 40, 65, 100, 150, 200]  # By rank (TUNE: more aggressive = higher values)
        self.isolated_pawn_penalty = 12                               # Penalty for pawns without neighbors (TUNE: 8-20)
        self.doubled_pawn_penalty = 18                                # Penalty for doubled pawns (TUNE: 12-25)
        self.backward_pawn_penalty = 10                               # Penalty for backward pawns (TUNE: 6-15)
        self.connected_pawn_bonus = 6                                 # Bonus for connected pawns (TUNE: 4-10)
        self.pawn_chain_bonus = 4                                     # Bonus for pawn chains (TUNE: 2-8)
        
        # Pawn storm/shelter patterns
        self.pawn_storm_bonus = 8
        self.enemy_pawn_storm_penalty = 12
        
        # King safety zone masks (3x3 around each square)
        self.KING_ZONES = [0] * 64
        for sq in range(64):
            self.KING_ZONES[sq] = self._calc_king_zone_mask(sq)
    
    def _calc_king_zone_mask(self, square: int) -> int:
        """Calculate 3x3 zone around a square for king safety analysis"""
        mask = 0
        rank, file = divmod(square, 8)
        
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                new_rank, new_file = rank + dr, file + df
                if 0 <= new_rank < 8 and 0 <= new_file < 8:
                    mask |= (1 << (new_rank * 8 + new_file))
        
        return mask
    
    def _init_bitboard_constants(self):
        """Initialize constant bitboard masks"""
        
        # Rank masks
        self.RANK_1 = 0x00000000000000FF
        self.RANK_2 = 0x000000000000FF00
        self.RANK_3 = 0x0000000000FF0000
        self.RANK_4 = 0x00000000FF000000
        self.RANK_5 = 0x000000FF00000000
        self.RANK_6 = 0x0000FF0000000000
        self.RANK_7 = 0x00FF000000000000
        self.RANK_8 = 0xFF00000000000000
        
        # File masks
        self.FILE_A = 0x0101010101010101
        self.FILE_B = 0x0202020202020202
        self.FILE_C = 0x0404040404040404
        self.FILE_D = 0x0808080808080808
        self.FILE_E = 0x1010101010101010
        self.FILE_F = 0x2020202020202020
        self.FILE_G = 0x4040404040404040
        self.FILE_H = 0x8080808080808080
        
        # V12.3: File masks array for indexed access
        self.FILE_MASKS = [
            self.FILE_A, self.FILE_B, self.FILE_C, self.FILE_D,
            self.FILE_E, self.FILE_F, self.FILE_G, self.FILE_H
        ]
        
        # Center squares
        self.CENTER = 0x0000001818000000  # d4, d5, e4, e5
        self.EXTENDED_CENTER = 0x00003C3C3C3C0000  # c3-f3 to c6-f6
        
        # Edge squares for endgame king driving
        self.EDGES = (self.RANK_1 | self.RANK_8 | self.FILE_A | self.FILE_H)
        
        # King safety masks
        self.WHITE_KINGSIDE_CASTLE = 0x0000000000000060  # f1, g1
        self.WHITE_QUEENSIDE_CASTLE = 0x000000000000000E  # b1, c1, d1
        self.BLACK_KINGSIDE_CASTLE = 0x6000000000000000  # f8, g8
        self.BLACK_QUEENSIDE_CASTLE = 0x0E00000000000000  # b8, c8, d8
        
        # Pawn structure masks
        self.WHITE_PASSED_PAWN_MASKS = self._generate_passed_pawn_masks(True)
        self.BLACK_PASSED_PAWN_MASKS = self._generate_passed_pawn_masks(False)
        
        # Development squares
        self.KNIGHT_OUTPOSTS = 0x0000240000240000  # c4, c5, f4, f5
        self.BISHOP_DIAGONALS = 0x8040201008040201 | 0x0102040810204080
    
    def _init_attack_tables(self):
        """Initialize pre-calculated attack tables for super-fast lookups"""
        
        # Knight attack patterns from each square
        self.KNIGHT_ATTACKS = [0] * 64
        for sq in range(64):
            self.KNIGHT_ATTACKS[sq] = self._calc_knight_attacks(sq)
        
        # King attack patterns
        self.KING_ATTACKS = [0] * 64
        for sq in range(64):
            self.KING_ATTACKS[sq] = self._calc_king_attacks(sq)
        
        # Pawn attack patterns
        self.WHITE_PAWN_ATTACKS = [0] * 64
        self.BLACK_PAWN_ATTACKS = [0] * 64
        for sq in range(64):
            self.WHITE_PAWN_ATTACKS[sq] = self._calc_white_pawn_attacks(sq)
            self.BLACK_PAWN_ATTACKS[sq] = self._calc_black_pawn_attacks(sq)
    
    def _calc_knight_attacks(self, square: int) -> int:
        """Calculate knight attack bitboard for a square"""
        attacks = 0
        rank, file = divmod(square, 8)
        
        # All 8 possible knight moves
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), 
                       (1, -2), (1, 2), (2, -1), (2, 1)]
        
        for dr, df in knight_moves:
            new_rank, new_file = rank + dr, file + df
            if 0 <= new_rank < 8 and 0 <= new_file < 8:
                attacks |= (1 << (new_rank * 8 + new_file))
        
        return attacks
    
    def _calc_king_attacks(self, square: int) -> int:
        """Calculate king attack bitboard for a square"""
        attacks = 0
        rank, file = divmod(square, 8)
        
        # All 8 possible king moves
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                if dr == 0 and df == 0:
                    continue
                new_rank, new_file = rank + dr, file + df
                if 0 <= new_rank < 8 and 0 <= new_file < 8:
                    attacks |= (1 << (new_rank * 8 + new_file))
        
        return attacks
    
    def _calc_white_pawn_attacks(self, square: int) -> int:
        """Calculate white pawn attack bitboard"""
        attacks = 0
        rank, file = divmod(square, 8)
        
        if rank < 7:  # Can attack forward
            if file > 0:  # Can attack left
                attacks |= (1 << ((rank + 1) * 8 + file - 1))
            if file < 7:  # Can attack right
                attacks |= (1 << ((rank + 1) * 8 + file + 1))
        
        return attacks
    
    def _calc_black_pawn_attacks(self, square: int) -> int:
        """Calculate black pawn attack bitboard"""
        attacks = 0
        rank, file = divmod(square, 8)
        
        if rank > 0:  # Can attack forward
            if file > 0:  # Can attack left
                attacks |= (1 << ((rank - 1) * 8 + file - 1))
            if file < 7:  # Can attack right
                attacks |= (1 << ((rank - 1) * 8 + file + 1))
        
        return attacks
    
    def _generate_passed_pawn_masks(self, is_white: bool) -> list:
        """Generate passed pawn masks for fast passed pawn detection"""
        masks = [0] * 64
        
        for square in range(64):
            rank, file = divmod(square, 8)
            mask = 0
            
            # Add files to check (own file and adjacent files)
            for check_file in [file - 1, file, file + 1]:
                if 0 <= check_file < 8:
                    if is_white:
                        # For white, check ranks ahead
                        for check_rank in range(rank + 1, 8):
                            mask |= (1 << (check_rank * 8 + check_file))
                    else:
                        # For black, check ranks ahead (down)
                        for check_rank in range(0, rank):
                            mask |= (1 << (check_rank * 8 + check_file))
            
            masks[square] = mask
        
        return masks
    
    def evaluate_bitboard(self, board: chess.Board, color: chess.Color) -> float:
        """
        === PAT'S MAIN EVALUATION FUNCTION ===
        V12.3 ENHANCED: Ultra-fast bitboard evaluation with integrated advanced features
        Target: 15,000+ NPS with comprehensive positional evaluation
        
        EVALUATION FLOW:
        1. Convert chess.Board to bitboards (fastest representation)
        2. Game phase detection (opening/middle/endgame)
        3. Material counting (ultra-fast bit population)
        4. Center control and piece activity
        5. Integrated king safety evaluation  
        6. Advanced pawn structure analysis
        7. Endgame-specific adjustments
        8. Draw detection and prevention
        
        PERFORMANCE NOTES:
        - All piece bitboards extracted once at start
        - Game phase affects feature weights
        - Bitwise operations throughout for maximum speed
        """
        
        # Convert chess.Board to bitboards for fast processing
        white_pawns = int(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = int(board.pieces(chess.PAWN, chess.BLACK))
        white_knights = int(board.pieces(chess.KNIGHT, chess.WHITE))
        black_knights = int(board.pieces(chess.KNIGHT, chess.BLACK))
        white_bishops = int(board.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = int(board.pieces(chess.BISHOP, chess.BLACK))
        white_rooks = int(board.pieces(chess.ROOK, chess.WHITE))
        black_rooks = int(board.pieces(chess.ROOK, chess.BLACK))
        white_queens = int(board.pieces(chess.QUEEN, chess.WHITE))
        black_queens = int(board.pieces(chess.QUEEN, chess.BLACK))
        white_king = int(board.pieces(chess.KING, chess.WHITE))
        black_king = int(board.pieces(chess.KING, chess.BLACK))
        
        white_pieces = white_pawns | white_knights | white_bishops | white_rooks | white_queens | white_king
        black_pieces = black_pawns | black_knights | black_bishops | black_rooks | black_queens | black_king
        all_pieces = white_pieces | black_pieces
        
        # Game phase detection for feature weighting
        total_material = self._popcount(all_pieces & ~(white_pawns | black_pawns))
        is_endgame = total_material < 12  # Endgame threshold
        is_opening = total_material >= 20  # Opening threshold
        
        score = 0.0
        
        # === 1. PAT'S MATERIAL EVALUATION (Ultra-fast bit counting) ===
        # This is the foundation - material balance using population count
        score += self._popcount(white_pawns) * self.piece_values[chess.PAWN]       # White material
        score += self._popcount(white_knights) * self.piece_values[chess.KNIGHT]
        score += self._popcount(white_bishops) * self.piece_values[chess.BISHOP]
        score += self._popcount(white_rooks) * self.piece_values[chess.ROOK]
        score += self._popcount(white_queens) * self.piece_values[chess.QUEEN]
        
        score -= self._popcount(black_pawns) * self.piece_values[chess.PAWN]       # Black material
        score -= self._popcount(black_knights) * self.piece_values[chess.KNIGHT]
        score -= self._popcount(black_bishops) * self.piece_values[chess.BISHOP]
        score -= self._popcount(black_rooks) * self.piece_values[chess.ROOK]
        score -= self._popcount(black_queens) * self.piece_values[chess.QUEEN]
        
        # === 2. PAT'S CENTER CONTROL EVALUATION ===
        # Enhanced for opening aggression - controlling center is crucial
        white_center_pawns = white_pawns & self.CENTER      # d4, d5, e4, e5 squares
        black_center_pawns = black_pawns & self.CENTER
        score += self._popcount(white_center_pawns) * 12    # TUNE: Increase for more aggressive center play
        score -= self._popcount(black_center_pawns) * 12
        
        white_extended_center = white_pawns & self.EXTENDED_CENTER
        black_extended_center = black_pawns & self.EXTENDED_CENTER
        score += self._popcount(white_extended_center) * 6
        score -= self._popcount(black_extended_center) * 6
        
        # Opening phase center control bonus for pieces
        if is_opening:
            white_center_pieces = (white_knights | white_bishops) & self.CENTER
            black_center_pieces = (black_knights | black_bishops) & self.CENTER
            score += self._popcount(white_center_pieces) * 18
            score -= self._popcount(black_center_pieces) * 18
            
            white_extended_pieces = (white_knights | white_bishops) & self.EXTENDED_CENTER
            black_extended_pieces = (black_knights | black_bishops) & self.EXTENDED_CENTER
            score += self._popcount(white_extended_pieces) * 10
            score -= self._popcount(black_extended_pieces) * 10
        
        # 3. PIECE DEVELOPMENT AND ACTIVITY
        white_knight_outposts = white_knights & self.KNIGHT_OUTPOSTS
        black_knight_outposts = black_knights & self.KNIGHT_OUTPOSTS
        score += self._popcount(white_knight_outposts) * 15
        score -= self._popcount(black_knight_outposts) * 15
        
        # Opening development penalties
        if is_opening:
            white_undeveloped = 0
            black_undeveloped = 0
            
            # Count undeveloped pieces on starting squares
            if white_knights & (1 << 1): white_undeveloped += 1  # b1
            if white_knights & (1 << 6): white_undeveloped += 1  # g1
            if black_knights & (1 << 57): black_undeveloped += 1  # b8
            if black_knights & (1 << 62): black_undeveloped += 1  # g8
            
            if white_bishops & (1 << 2): white_undeveloped += 1  # c1
            if white_bishops & (1 << 5): white_undeveloped += 1  # f1
            if black_bishops & (1 << 58): black_undeveloped += 1  # c8
            if black_bishops & (1 << 61): black_undeveloped += 1  # f8
            
            score -= white_undeveloped * 15
            score += black_undeveloped * 15
        
        # Piece activity bonus (avoid back ranks in middlegame)
        if not is_endgame and not is_opening:  # Middlegame
            white_back_pieces = (white_knights | white_bishops | white_rooks | white_queens) & (self.RANK_1 | self.RANK_2)
            black_back_pieces = (black_knights | black_bishops | black_rooks | black_queens) & (self.RANK_7 | self.RANK_8)
            
            score -= self._popcount(white_back_pieces) * 4
            score += self._popcount(black_back_pieces) * 4
        
        # === 4. PAT'S INTEGRATED KING SAFETY EVALUATION ===
        # V12.3: Phase-dependent king evaluation
        if not is_endgame:
            # MIDDLE GAME: King safety is paramount
            white_king_square = self._get_king_square(white_king)
            black_king_square = self._get_king_square(black_king)
            
            if white_king_square >= 0:
                king_safety_white = self._evaluate_king_safety_fast(
                    board, white_king_square, True, white_pawns, black_pawns, all_pieces)
                score += king_safety_white
                
            if black_king_square >= 0:
                king_safety_black = self._evaluate_king_safety_fast(
                    board, black_king_square, False, black_pawns, white_pawns, all_pieces)
                score -= king_safety_black
        else:
            # ENDGAME: King activity is important (help with pawn promotion)
            white_king_square = self._get_king_square(white_king)
            black_king_square = self._get_king_square(black_king)
            
            if white_king_square >= 0:
                score += self._evaluate_king_activity_fast(white_king_square, True)
            if black_king_square >= 0:
                score -= self._evaluate_king_activity_fast(black_king_square, False)
        
        # === 5. PAT'S INTEGRATED ADVANCED PAWN EVALUATION ===
        # V12.3: Comprehensive pawn structure analysis
        # This is a MAJOR evaluation component - pawn structure often determines the game
        white_pawn_score = self._evaluate_advanced_pawn_structure_fast(
            board, white_pawns, black_pawns, True, is_endgame)
        black_pawn_score = self._evaluate_advanced_pawn_structure_fast(
            board, black_pawns, white_pawns, False, is_endgame)
        
        score += white_pawn_score
        score -= black_pawn_score
        
        # 6. TRADITIONAL KING SAFETY BONUSES (for completed castling)
        if white_king & self.WHITE_KINGSIDE_CASTLE:
            score += 35  # Completed kingside castling
        if white_king & self.WHITE_QUEENSIDE_CASTLE:
            score += 28  # Completed queenside castling
        if black_king & self.BLACK_KINGSIDE_CASTLE:
            score -= 35
        if black_king & self.BLACK_QUEENSIDE_CASTLE:
            score -= 28
        
        # 7. ENDGAME CONSIDERATIONS  
        if is_endgame:
            # Drive enemy king to edge
            black_king_on_edge = black_king & self.EDGES
            white_king_on_edge = white_king & self.EDGES
            score += self._popcount(black_king_on_edge) * 12
            score -= self._popcount(white_king_on_edge) * 12
        
        # 8. DRAW PREVENTION AND ACTIVITY ENCOURAGEMENT
        # Fifty-move rule awareness
        if board.halfmove_clock > 30:
            draw_penalty = (board.halfmove_clock - 30) * 2.0
            score -= draw_penalty if color == chess.WHITE else -draw_penalty
        
        # Basic repetition detection
        if hasattr(board, 'move_stack') and len(board.move_stack) >= 4:
            try:
                board_copy = board.copy()
                recent_positions = []
                
                for _ in range(min(4, len(board.move_stack))):
                    if board_copy.move_stack:
                        recent_positions.append(board_copy.fen().split(' ')[0])
                        board_copy.pop()
                
                current_pos = board.fen().split(' ')[0]
                repetition_count = recent_positions.count(current_pos)
                
                if repetition_count >= 1:
                    repetition_penalty = repetition_count * 20.0
                    score -= repetition_penalty if color == chess.WHITE else -repetition_penalty
            except:
                pass
        
        return score if color == chess.WHITE else -score
    
    def _get_king_square(self, king_bitboard: int) -> int:
        """Extract king square from king bitboard."""
        if king_bitboard == 0:
            return -1  # Invalid square
        # Get the least significant bit (king position)
        return (king_bitboard & -king_bitboard).bit_length() - 1
    
    def _get_king_zone_mask(self, king_square: int, is_white: bool) -> int:
        """Get the king safety zone mask for a given king position."""
        if king_square < 0 or king_square >= 64:
            return 0
        
        # Use pre-calculated king zone mask
        return self.KING_ZONES[king_square]
    
    def _evaluate_king_safety_fast(self, board: chess.Board, king_square: int, is_white: bool, 
                                 own_pawns: int, enemy_pawns: int, all_pieces: int) -> float:
        """
        V12.3: Ultra-fast king safety evaluation using bitboards
        Integrates key patterns from v7p3r_king_safety_evaluator
        """
        score = 0.0
        
        # Get king zone mask
        king_zone = self._get_king_zone_mask(king_square, is_white)
        
        # 1. PAWN SHIELD EVALUATION
        own_shield_pawns = own_pawns & king_zone
        shield_count = self._popcount(own_shield_pawns)
        
        if shield_count >= 3:
            score += 25.0  # Full pawn shield
        elif shield_count == 2:
            score += 15.0  # Partial shield
        elif shield_count == 1:
            score += 5.0   # Minimal shield
        else:
            score -= 20.0  # No shield - dangerous
        
        # 2. PAWN STORM DETECTION
        enemy_storm_pawns = enemy_pawns & king_zone
        storm_count = self._popcount(enemy_storm_pawns)
        score -= storm_count * 12.0  # Each attacking pawn is dangerous
        
        # 3. OPEN FILES NEAR KING
        king_file = king_square % 8
        adjacent_files = []
        
        if king_file > 0:
            adjacent_files.append(king_file - 1)
        adjacent_files.append(king_file)
        if king_file < 7:
            adjacent_files.append(king_file + 1)
        
        open_files_near_king = 0
        for file_idx in adjacent_files:
            file_mask = self.FILE_MASKS[file_idx]
            if not (own_pawns & file_mask):  # No own pawns on this file
                open_files_near_king += 1
                if not (enemy_pawns & file_mask):  # Completely open
                    open_files_near_king += 1
        
        score -= open_files_near_king * 8.0  # Open files are dangerous
        
        # 4. CASTLING SAFETY BONUS
        if is_white:
            if king_square == 6:  # Kingside castled (g1)
                score += 20.0
            elif king_square == 2:  # Queenside castled (c1)
                score += 15.0
        else:
            if king_square == 62:  # Kingside castled (g8)
                score += 20.0
            elif king_square == 58:  # Queenside castled (c8)
                score += 15.0
        
        # 5. KING EXPOSURE PENALTY
        if is_white and king_square >= 32:  # White king too advanced
            score -= 15.0
        elif not is_white and king_square < 32:  # Black king too advanced
            score -= 15.0
        
        return score
    
    def _evaluate_king_activity_fast(self, king_square: int, is_white: bool) -> float:
        """Endgame king activity evaluation."""
        score = 0.0
        
        # Center control bonus in endgame
        king_rank = king_square // 8
        king_file = king_square % 8
        
        # Distance from center
        center_distance = max(abs(king_rank - 3.5), abs(king_file - 3.5))
        score += (4.0 - center_distance) * 3.0  # Closer to center is better
        
        # Seventh rank bonus for attacking enemy pawns
        if is_white and king_rank == 6:  # White king on 7th rank
            score += 10.0
        elif not is_white and king_rank == 1:  # Black king on 2nd rank
            score += 10.0
        
        return score
    
    def _evaluate_advanced_pawn_structure_fast(self, board: chess.Board, own_pawns: int, 
                                             enemy_pawns: int, is_white: bool, is_endgame: bool) -> float:
        """
        V12.3: Ultra-fast advanced pawn structure evaluation
        Integrates key patterns from v7p3r_advanced_pawn_evaluator
        """
        score = 0.0
        
        # 1. PASSED PAWNS (enhanced)
        passed_pawn_count = self._count_passed_pawns(own_pawns, enemy_pawns, is_white)
        if is_endgame:
            score += passed_pawn_count * 40.0  # Passed pawns are crucial in endgame
        else:
            score += passed_pawn_count * 25.0
        
        # 2. DOUBLED PAWNS
        doubled_penalty = 0
        for file_idx in range(8):
            file_mask = self.FILE_MASKS[file_idx]
            pawns_on_file = self._popcount(own_pawns & file_mask)
            if pawns_on_file > 1:
                doubled_penalty += (pawns_on_file - 1) * 8.0
        score -= doubled_penalty
        
        # 3. ISOLATED PAWNS
        isolated_penalty = 0
        for file_idx in range(8):
            file_mask = self.FILE_MASKS[file_idx]
            if own_pawns & file_mask:  # Has pawn on this file
                has_support = False
                
                # Check adjacent files for pawn support
                if file_idx > 0 and (own_pawns & self.FILE_MASKS[file_idx - 1]):
                    has_support = True
                if file_idx < 7 and (own_pawns & self.FILE_MASKS[file_idx + 1]):
                    has_support = True
                
                if not has_support:
                    isolated_penalty += 12.0
        score -= isolated_penalty
        
        # 4. BACKWARD PAWNS (simplified check)
        backward_penalty = 0
        for file_idx in range(8):
            file_mask = self.FILE_MASKS[file_idx]
            own_file_pawns = own_pawns & file_mask
            
            if own_file_pawns:
                # Check if this pawn can advance safely
                pawn_square = (own_file_pawns & -own_file_pawns).bit_length() - 1  # Get LSB
                advance_square = pawn_square + (8 if is_white else -8)
                
                if 0 <= advance_square < 64:
                    advance_mask = 1 << advance_square
                    
                    # Check if advance square is attacked by enemy pawns
                    if is_white:
                        enemy_attacks = ((enemy_pawns & ~self.FILE_MASKS[0]) >> 9) | \
                                      ((enemy_pawns & ~self.FILE_MASKS[7]) >> 7)
                    else:
                        enemy_attacks = ((enemy_pawns & ~self.FILE_MASKS[0]) << 7) | \
                                      ((enemy_pawns & ~self.FILE_MASKS[7]) << 9)
                    
                    if advance_mask & enemy_attacks:
                        # Can't advance safely - check if it's backward
                        has_pawn_support = False
                        if file_idx > 0:
                            left_file_pawns = own_pawns & self.FILE_MASKS[file_idx - 1]
                            if left_file_pawns:
                                left_pawn_sq = (left_file_pawns & -left_file_pawns).bit_length() - 1
                                if is_white and left_pawn_sq < pawn_square:
                                    has_pawn_support = True
                                elif not is_white and left_pawn_sq > pawn_square:
                                    has_pawn_support = True
                        
                        if file_idx < 7:
                            right_file_pawns = own_pawns & self.FILE_MASKS[file_idx + 1]
                            if right_file_pawns:
                                right_pawn_sq = (right_file_pawns & -right_file_pawns).bit_length() - 1
                                if is_white and right_pawn_sq < pawn_square:
                                    has_pawn_support = True
                                elif not is_white and right_pawn_sq > pawn_square:
                                    has_pawn_support = True
                        
                        if not has_pawn_support:
                            backward_penalty += 10.0
        
        score -= backward_penalty
        
        # 5. PAWN CHAINS (bonus for connected pawns)
        chain_bonus = 0
        for file_idx in range(7):  # Don't check the last file
            file_mask = self.FILE_MASKS[file_idx]
            next_file_mask = self.FILE_MASKS[file_idx + 1]
            
            if (own_pawns & file_mask) and (own_pawns & next_file_mask):
                # Adjacent files both have pawns - check if they support each other
                chain_bonus += 6.0
        
        score += chain_bonus
        
        # 6. ADVANCED PAWNS BONUS
        advanced_bonus = 0
        if is_white:
            # White pawns on ranks 5, 6, 7
            advanced_pawns = own_pawns & (self.RANK_5 | self.RANK_6 | self.RANK_7)
            rank_5_pawns = self._popcount(own_pawns & self.RANK_5)
            rank_6_pawns = self._popcount(own_pawns & self.RANK_6)
            rank_7_pawns = self._popcount(own_pawns & self.RANK_7)
            
            advanced_bonus += rank_5_pawns * 5.0
            advanced_bonus += rank_6_pawns * 10.0
            advanced_bonus += rank_7_pawns * 20.0
        else:
            # Black pawns on ranks 4, 3, 2
            rank_4_pawns = self._popcount(own_pawns & self.RANK_4)
            rank_3_pawns = self._popcount(own_pawns & self.RANK_3)
            rank_2_pawns = self._popcount(own_pawns & self.RANK_2)
            
            advanced_bonus += rank_4_pawns * 5.0
            advanced_bonus += rank_3_pawns * 10.0
            advanced_bonus += rank_2_pawns * 20.0
        
        score += advanced_bonus
        
        return score

    def _popcount(self, bitboard: int) -> int:
        """Ultra-fast population count (number of 1 bits)"""
        return bin(bitboard).count('1')
    
    def _count_passed_pawns(self, our_pawns: int, enemy_pawns: int, is_white: bool) -> int:
        """Count passed pawns using pre-calculated masks"""
        passed_count = 0
        pawns = our_pawns
        
        while pawns:
            # Get least significant bit (first pawn)
            pawn_square = (pawns & -pawns).bit_length() - 1
            
            # Check if it's passed using pre-calculated mask
            if is_white:
                mask = self.WHITE_PASSED_PAWN_MASKS[pawn_square]
            else:
                mask = self.BLACK_PASSED_PAWN_MASKS[pawn_square]
            
            if not (enemy_pawns & mask):
                passed_count += 1
            
            # Remove this pawn and continue
            pawns &= pawns - 1
        
        return passed_count


class V7P3RScoringCalculationBitboard:
    """
    Drop-in replacement for the slow scoring calculator
    Uses bitboards for ultra-high performance
    """
    
    def __init__(self, piece_values: Dict[int, int]):
        self.piece_values = piece_values
        self.bitboard_evaluator = V7P3RBitboardEvaluator(piece_values)
    
    def calculate_score_optimized(self, board: chess.Board, color: chess.Color, endgame_factor: float = 0.0) -> float:
        """
        Ultra-fast evaluation using bitboards
        Target: 20,000+ NPS
        """
        return self.bitboard_evaluator.evaluate_bitboard(board, color)
