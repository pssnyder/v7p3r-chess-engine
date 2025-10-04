#!/usr/bin/env python3
"""
V7P3R Bitboard-Based Evaluation System
Ultra-fast evaluation using bitboard operations for tens of thousands of NPS

Enhanced with Intelligent Nudge System v2.0 for better opening play and center control
"""

import chess
from typing import Dict, Tuple


class V7P3RBitboardEvaluator:
    """
    High-performance bitboard-based evaluation system
    Uses bitwise operations for 10x+ speed improvement
    Enhanced with intelligent nudge system for better opening play
    """
    
    def __init__(self, piece_values: Dict[int, int]):
        self.piece_values = piece_values
        
        # Initialize intelligent nudge system
        self.nudges = None
        self._try_init_nudges()
        
        # Pre-calculate bitboard masks for ultra-fast evaluation
        self._init_bitboard_constants()
        self._init_attack_tables()
    
    def _try_init_nudges(self):
        """Try to initialize the intelligent nudge system"""
        try:
            from .v7p3r_intelligent_nudges import V7P3RIntelligentNudges
            self.nudges = V7P3RIntelligentNudges()
        except (ImportError, ModuleNotFoundError):
            # Try without relative import
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(__file__))
                from v7p3r_intelligent_nudges import V7P3RIntelligentNudges
                self.nudges = V7P3RIntelligentNudges()
            except (ImportError, ModuleNotFoundError):
                print("⚠️  Nudge system not available - using base evaluation")
                self.nudges = None
        except Exception as e:
            print(f"⚠️  Nudge system initialization error: {e}")
            self.nudges = None
    
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
        Ultra-fast bitboard evaluation using bitwise operations
        This should give us 20,000+ NPS
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
        
        # V12.1: Calculate material count for game phase detection
        total_material = self._popcount(all_pieces & ~(white_pawns | black_pawns))
        
        score = 0.0
        
        # 1. MATERIAL (ultra-fast bit counting)
        score += self._popcount(white_pawns) * self.piece_values[chess.PAWN]
        score += self._popcount(white_knights) * self.piece_values[chess.KNIGHT]
        score += self._popcount(white_bishops) * self.piece_values[chess.BISHOP]
        score += self._popcount(white_rooks) * self.piece_values[chess.ROOK]
        score += self._popcount(white_queens) * self.piece_values[chess.QUEEN]
        
        score -= self._popcount(black_pawns) * self.piece_values[chess.PAWN]
        score -= self._popcount(black_knights) * self.piece_values[chess.KNIGHT]
        score -= self._popcount(black_bishops) * self.piece_values[chess.BISHOP]
        score -= self._popcount(black_rooks) * self.piece_values[chess.ROOK]
        score -= self._popcount(black_queens) * self.piece_values[chess.QUEEN]
        
        # 2. CENTER CONTROL (V12.1: Enhanced for opening aggression)
        white_center_pawns = white_pawns & self.CENTER
        black_center_pawns = black_pawns & self.CENTER
        score += self._popcount(white_center_pawns) * 10
        score -= self._popcount(black_center_pawns) * 10
        
        white_extended_center = white_pawns & self.EXTENDED_CENTER
        black_extended_center = black_pawns & self.EXTENDED_CENTER
        score += self._popcount(white_extended_center) * 5
        score -= self._popcount(black_extended_center) * 5
        
        # V12.1: Opening phase center control bonus for pieces (not just pawns)
        if total_material >= 20:  # Opening/early middlegame
            white_center_pieces = (white_knights | white_bishops) & self.CENTER
            black_center_pieces = (black_knights | black_bishops) & self.CENTER
            score += self._popcount(white_center_pieces) * 15  # Bonus for pieces on center
            score -= self._popcount(black_center_pieces) * 15
            
            white_extended_pieces = (white_knights | white_bishops) & self.EXTENDED_CENTER
            black_extended_pieces = (black_knights | black_bishops) & self.EXTENDED_CENTER
            score += self._popcount(white_extended_pieces) * 8  # Bonus for pieces near center
            score -= self._popcount(black_extended_pieces) * 8
        
        # 3. PIECE DEVELOPMENT (V12.1: Enhanced development evaluation)
        white_knight_outposts = white_knights & self.KNIGHT_OUTPOSTS
        black_knight_outposts = black_knights & self.KNIGHT_OUTPOSTS
        score += self._popcount(white_knight_outposts) * 15
        score -= self._popcount(black_knight_outposts) * 15
        
        # V12.1: Opening development penalty - punish undeveloped pieces
        if total_material >= 18:  # Opening phase
            # Count pieces still on starting squares
            white_undeveloped = 0
            black_undeveloped = 0
            
            # Knights on starting squares (b1, g1 for white; b8, g8 for black)
            if white_knights & (1 << 1):  # b1
                white_undeveloped += 1
            if white_knights & (1 << 6):  # g1
                white_undeveloped += 1
            if black_knights & (1 << 57):  # b8
                black_undeveloped += 1
            if black_knights & (1 << 62):  # g8
                black_undeveloped += 1
                
            # Bishops on starting squares (c1, f1 for white; c8, f8 for black)
            if white_bishops & (1 << 2):  # c1
                white_undeveloped += 1
            if white_bishops & (1 << 5):  # f1
                white_undeveloped += 1
            if black_bishops & (1 << 58):  # c8
                black_undeveloped += 1
            if black_bishops & (1 << 61):  # f8
                black_undeveloped += 1
            
            # Apply development penalties
            score -= white_undeveloped * 12  # Penalty for undeveloped White pieces
            score += black_undeveloped * 12  # Penalty for undeveloped Black pieces
        
        # 4. ENHANCED KING SAFETY & CASTLING EVALUATION (V12.4)
        score += self._evaluate_enhanced_castling(board, color)
        
        # 5. PAWN STRUCTURE (passed pawns - ultra-fast)
        score += self._count_passed_pawns(white_pawns, black_pawns, True) * 20
        score -= self._count_passed_pawns(black_pawns, white_pawns, False) * 20
        
        # 6. ENDGAME CONSIDERATIONS  
        if total_material <= 8:  # Endgame
            # Drive enemy king to edge (always from White's perspective)
            black_king_on_edge = black_king & self.EDGES
            white_king_on_edge = white_king & self.EDGES
            score += self._popcount(black_king_on_edge) * 10  # Good for White if Black king on edge
            score -= self._popcount(white_king_on_edge) * 10  # Bad for White if White king on edge
        
        # 7. V12.1: STRICTER DRAW PREVENTION
        # Encourage aggressive play and discourage repetitive/passive positions
        
        # Fifty-move rule awareness: stronger penalty as we approach limit
        if board.halfmove_clock > 30:
            draw_penalty = (board.halfmove_clock - 30) * 2.0  # Escalating penalty
            score -= draw_penalty if color == chess.WHITE else -draw_penalty
        
        # Repetition penalty: discourage position repetition
        if hasattr(board, 'move_stack') and len(board.move_stack) >= 4:
            try:
                # Check for 2-fold repetition in recent moves
                board_copy = board.copy()
                recent_positions = []
                
                # Collect last 4 positions
                for _ in range(min(4, len(board.move_stack))):
                    if board_copy.move_stack:
                        recent_positions.append(board_copy.fen().split(' ')[0])  # Position only
                        board_copy.pop()
                
                # Count repetitions
                current_pos = board.fen().split(' ')[0]
                repetition_count = recent_positions.count(current_pos)
                
                if repetition_count >= 1:
                    repetition_penalty = repetition_count * 25.0  # Strong penalty for repetition
                    score -= repetition_penalty if color == chess.WHITE else -repetition_penalty
                    
            except:
                pass  # Ignore errors in repetition detection
        
        # Encourage piece activity: penalty for pieces on back ranks in middlegame
        if total_material >= 12:  # Middlegame
            white_back_rank_pieces = (white_knights | white_bishops | white_rooks | white_queens) & (self.RANK_1 | self.RANK_2)
            black_back_rank_pieces = (black_knights | black_bishops | black_rooks | black_queens) & (self.RANK_7 | self.RANK_8)
            
            activity_penalty = (self._popcount(white_back_rank_pieces) - self._popcount(black_back_rank_pieces)) * 3
            score -= activity_penalty if color == chess.WHITE else -activity_penalty

        # V12.5: INTELLIGENT NUDGE SYSTEM v2.0 - Enhanced piece-square evaluations
        if self.nudges is not None:
            nudge_bonus = self._evaluate_nudge_enhancements(board, white_pieces, black_pieces, 
                                                           white_pawns, black_pawns,
                                                           white_knights, black_knights, 
                                                           white_bishops, black_bishops,
                                                           white_rooks, black_rooks,
                                                           white_queens, black_queens, total_material)
            score += nudge_bonus if color == chess.WHITE else -nudge_bonus

        return score if color == chess.WHITE else -score
    
    def _evaluate_nudge_enhancements(self, board: chess.Board, white_pieces: int, black_pieces: int,
                                   white_pawns: int, black_pawns: int, white_knights: int, black_knights: int,
                                   white_bishops: int, black_bishops: int, white_rooks: int, black_rooks: int,
                                   white_queens: int, black_queens: int, total_material: int) -> float:
        """
        V12.5: Intelligent Nudge System v2.0 - Performance-optimized piece-square enhancements
        Uses pre-computed adjustments based on historical game data and opening theory
        """
        nudge_score = 0.0
        
        # 1. ENHANCED PIECE-SQUARE EVALUATION
        # Apply nudge-based piece square adjustments for better opening play
        if total_material >= 16:  # Opening and early middlegame
            # Pawns - encourage center control based on nudge data
            nudge_score += self._evaluate_piece_nudges(white_pawns, chess.PAWN, chess.WHITE)
            nudge_score -= self._evaluate_piece_nudges(black_pawns, chess.PAWN, chess.BLACK)
            
            # Knights - prioritize center and outpost squares from nudge database
            nudge_score += self._evaluate_piece_nudges(white_knights, chess.KNIGHT, chess.WHITE)
            nudge_score -= self._evaluate_piece_nudges(black_knights, chess.KNIGHT, chess.BLACK)
            
            # Bishops - encourage center control and long diagonals
            nudge_score += self._evaluate_piece_nudges(white_bishops, chess.BISHOP, chess.WHITE)
            nudge_score -= self._evaluate_piece_nudges(black_bishops, chess.BISHOP, chess.BLACK)
        
        # 2. OPENING PREFERENCES
        # Apply opening move bonuses for early game center control
        if board.fullmove_number <= 8:
            move_number = board.fullmove_number
            
            # Check recent moves for opening preferences
            if len(board.move_stack) > 0:
                last_move = board.move_stack[-1]
                # Create a temporary board to get the position before the last move
                temp_board = board.copy()
                temp_board.pop()
                opening_bonus = self.nudges.get_opening_bonus(last_move.uci(), move_number, temp_board.fen())
                if opening_bonus > 0:
                    # Apply bonus based on whose move it was
                    move_was_white = (len(board.move_stack) % 2 == 1)
                    if move_was_white:
                        nudge_score += opening_bonus * 0.5  # Scale down to avoid overwhelming
                    else:
                        nudge_score -= opening_bonus * 0.5
        
        # 3. CENTER AGGRESSION ENHANCEMENT
        # Extra bonuses for center control from Caro-Kann and other openings
        if total_material >= 18:  # Early opening
            # Enhanced d4/d5/e4/e5 control
            center_control_bonus = 0.0
            
            # Check for pieces attacking/controlling center squares
            for square in [chess.D4, chess.D5, chess.E4, chess.E5]:
                white_attackers = len(board.attackers(chess.WHITE, square))
                black_attackers = len(board.attackers(chess.BLACK, square))
                
                # Bonus for controlling center squares
                control_diff = white_attackers - black_attackers
                center_control_bonus += control_diff * 3.0
            
            nudge_score += center_control_bonus
        
        # Scale nudge influence to avoid overwhelming base evaluation
        return nudge_score * 0.3  # 30% influence to maintain balance
    
    def _evaluate_piece_nudges(self, piece_bitboard: int, piece_type: chess.PieceType, color: chess.Color) -> float:
        """Evaluate nudge-based piece square bonuses for a specific piece type"""
        if piece_bitboard == 0:
            return 0.0
        
        total_bonus = 0.0
        
        # Iterate through all pieces of this type
        temp_bitboard = piece_bitboard
        while temp_bitboard:
            # Find the least significant bit (rightmost 1)
            square = (temp_bitboard & -temp_bitboard).bit_length() - 1
            
            # Get nudge adjustment for this piece and square
            adjustment = self.nudges.get_piece_square_adjustment(piece_type, square)
            total_bonus += adjustment
            
            # Remove this bit and continue
            temp_bitboard &= temp_bitboard - 1
        
        return total_bonus
    
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
    
    def _evaluate_enhanced_castling(self, board: chess.Board, color: chess.Color) -> float:
        """
        Enhanced castling evaluation for V12.4
        Rewards actual castling, penalizes wasted castling rights
        ALWAYS returns score from White's perspective (positive = good for White)
        """
        score = 0.0
        
        # Determine if we're in opening phase
        opening_phase = len(board.move_stack) < 20
        
        # WHITE evaluation
        white_has_castled = self._has_castled(board, chess.WHITE)
        
        if white_has_castled:
            # Reward successful castling for White
            score += 50.0
            king_square = board.king(chess.WHITE)
            if king_square in [chess.G1, chess.C1]:
                score += 25.0  # Safety bonus for White
        else:
            # Check White castling availability
            can_castle_kingside = board.has_kingside_castling_rights(chess.WHITE)
            can_castle_queenside = board.has_queenside_castling_rights(chess.WHITE)
            
            if opening_phase:
                if can_castle_kingside:
                    score += 30.0  # Good for White to have castling rights
                if can_castle_queenside:
                    score += 20.0
                
                # Penalty for White moving king without castling
                king_square = board.king(chess.WHITE)
                if king_square != chess.E1 and not white_has_castled:
                    score -= 50.0  # Bad for White
            else:
                # Mild penalty for unused castling in middlegame
                if can_castle_kingside or can_castle_queenside:
                    score -= 10.0

        # BLACK evaluation (opposite perspective)
        black_has_castled = self._has_castled(board, chess.BLACK)
        
        if black_has_castled:
            # Penalize successful castling for Black (good for Black = bad for White)
            score -= 50.0
            king_square = board.king(chess.BLACK)
            if king_square in [chess.G8, chess.C8]:
                score -= 25.0  # Safety bonus for Black = penalty for White
        else:
            can_castle_kingside = board.has_kingside_castling_rights(chess.BLACK)
            can_castle_queenside = board.has_queenside_castling_rights(chess.BLACK)
            
            if opening_phase:
                if can_castle_kingside:
                    score -= 30.0  # Bad for White if Black has castling rights
                if can_castle_queenside:
                    score -= 20.0
                
                # CRITICAL FIX: Reward White when Black moves king without castling
                king_square = board.king(chess.BLACK)
                if king_square != chess.E8 and not black_has_castled:
                    score += 50.0  # GOOD for White when Black blunders king move!
            else:
                if can_castle_kingside or can_castle_queenside:
                    score += 10.0  # Good for White if Black wastes castling rights
        
        return score
    
    def _has_castled(self, board: chess.Board, color: chess.Color) -> bool:
        """Check if the specified color has already castled"""
        king_square = board.king(color)
        
        if color == chess.WHITE:
            if king_square == chess.G1:
                rook_on_f1 = board.piece_at(chess.F1)
                return (rook_on_f1 is not None and 
                       rook_on_f1.piece_type == chess.ROOK and 
                       rook_on_f1.color == chess.WHITE)
            elif king_square == chess.C1:
                rook_on_d1 = board.piece_at(chess.D1)
                return (rook_on_d1 is not None and 
                       rook_on_d1.piece_type == chess.ROOK and 
                       rook_on_d1.color == chess.WHITE)
        else:  # BLACK
            if king_square == chess.G8:
                rook_on_f8 = board.piece_at(chess.F8)
                return (rook_on_f8 is not None and 
                       rook_on_f8.piece_type == chess.ROOK and 
                       rook_on_f8.color == chess.BLACK)
            elif king_square == chess.C8:
                rook_on_d8 = board.piece_at(chess.D8)
                return (rook_on_d8 is not None and 
                       rook_on_d8.piece_type == chess.ROOK and 
                       rook_on_d8.color == chess.BLACK)
        
        # Check move history for explicit castling moves
        for move in board.move_stack:
            if board.is_castling(move):
                from_square = move.from_square
                if color == chess.WHITE and from_square == chess.E1:
                    return True
                elif color == chess.BLACK and from_square == chess.E8:
                    return True
        
        return False


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
