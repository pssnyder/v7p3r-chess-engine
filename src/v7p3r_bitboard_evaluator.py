#!/usr/bin/env python3
"""
V7P3R Bitboard-Based Evaluation System
Ultra-fast evaluation using bitboard operations for maximum performance

Optimized for tournament play with no nudge system overhead
"""

import chess
from typing import Dict, Tuple, List


class V7P3RBitboardEvaluator:
    """
    High-performance bitboard-based evaluation system
    Uses bitwise operations for 10x+ speed improvement
    Optimized for maximum performance without nudge system overhead
    """
    
    def __init__(self, piece_values: Dict[int, int], enable_nudges: bool = False):
        self.piece_values = piece_values
        
        # Pre-calculate bitboard masks for ultra-fast evaluation
        self._init_bitboard_constants()
        self._init_attack_tables()
    
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
        # V14.2: Removed knight outposts (3.9% overhead) - tactical analysis detects fork opportunities
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
        
        # 4. PAWN STRUCTURE (passed pawns - ultra-fast)
        # V14.2: Removed castling evaluation (41.7% overhead) - castling handled by move ordering
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
        
        # The repetition detection was calling board.fen() multiple times per evaluation,
        # causing massive performance degradation. Commenting out for tournament performance.
        # TODO: Implement fast repetition detection using zobrist hashing
        
        # V14.2: Removed activity penalties (5.7% overhead) - depth will find active moves naturally

        return score if color == chess.WHITE else -score
    
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

    def detect_bitboard_tactics(self, board: chess.Board, move: chess.Move) -> float:
        """
        V12.6 CONSOLIDATED: Detect tactical patterns using bitboard operations
        Returns a bonus score for tactical moves (pins, forks, skewers, discovered attacks)
        """
        tactical_bonus = 0.0
        
        # Make the move to analyze the resulting position
        board.push(move)
        
        try:
            our_color = not board.turn  # We just moved, so it's opponent's turn
            
            # Legacy bitboard tactics for additional analysis
            moving_piece = board.piece_at(move.to_square)
            if moving_piece:
                fork_bonus = self._analyze_fork_bitboard(board, move.to_square, moving_piece, board.turn)
                tactical_bonus += fork_bonus
                
                # Analyze for pins and skewers using ray attacks
                if moving_piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    pin_skewer_bonus = self._analyze_pins_skewers_bitboard(board, move.to_square, moving_piece, board.turn)
                    tactical_bonus += pin_skewer_bonus
            
        except Exception:
            # If analysis fails, return 0 bonus
            pass
        finally:
            board.pop()
        
        return tactical_bonus
    
    def _analyze_fork_bitboard(self, board: chess.Board, square: int, piece: chess.Piece, enemy_color: chess.Color) -> float:
        """Analyze fork patterns using bitboards"""
        if piece.piece_type == chess.KNIGHT:
            # Knight fork detection
            attacks = self.KNIGHT_ATTACKS[square]
            enemy_pieces = 0
            high_value_targets = 0
            
            for target_sq in range(64):
                if attacks & (1 << target_sq):
                    target_piece = board.piece_at(target_sq)
                    if target_piece and target_piece.color == enemy_color:
                        enemy_pieces += 1
                        if target_piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                            high_value_targets += 1
            
            # Knight forking 2+ pieces gets bonus, more for high-value targets
            if enemy_pieces >= 2:
                return 50.0 + (high_value_targets * 25.0)
        
        return 0.0
    
    def _analyze_pins_skewers_bitboard(self, board: chess.Board, square: int, piece: chess.Piece, enemy_color: chess.Color) -> float:
        """Analyze pin and skewer patterns using ray attacks"""
        # This is a simplified version - full implementation would need sliding piece attack generation
        # For now, just give a small bonus for pieces that could create pins/skewers
        
        if piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            # Look for aligned enemy pieces that could be pinned/skewered
            bonus = 0.0
            
            # Check if we're attacking towards the enemy king
            enemy_king_sq = None
            for sq in range(64):
                p = board.piece_at(sq)
                if p and p.piece_type == chess.KING and p.color == enemy_color:
                    enemy_king_sq = sq
                    break
            
            if enemy_king_sq is not None:
                # Simple heuristic: if we're on the same rank/file/diagonal as enemy king
                sq_rank, sq_file = divmod(square, 8)
                king_rank, king_file = divmod(enemy_king_sq, 8)
                
                if (sq_rank == king_rank or sq_file == king_file or 
                    abs(sq_rank - king_rank) == abs(sq_file - king_file)):
                    bonus += 15.0  # Potential pin/skewer bonus
            
            return bonus
        
        return 0.0

    def evaluate_pawn_structure(self, board: chess.Board, color: bool) -> float:
        """
        V12.6 CONSOLIDATED: Comprehensive pawn structure evaluation using bitboards
        Returns score from the perspective of the given color
        """
        total_score = 0.0
        
        # Get all pawns for this color as bitboard
        pawns = board.pieces(chess.PAWN, color)
        
        # Evaluate each pawn and overall structure
        total_score += self._evaluate_passed_pawns_bitboard(board, pawns, color)
        total_score += self._evaluate_isolated_pawns_bitboard(board, pawns, color)
        total_score += self._evaluate_doubled_pawns_bitboard(board, pawns, color)
        total_score += self._evaluate_backward_pawns_bitboard(board, pawns, color)
        total_score += self._evaluate_connected_pawns_bitboard(board, pawns, color)
        total_score += self._evaluate_pawn_chains_bitboard(board, pawns, color)
        total_score += self._evaluate_pawn_storms_bitboard(board, pawns, color)
        
        return total_score
    
    def _evaluate_passed_pawns_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate passed pawns using bitboard operations"""
        score = 0.0
        passed_pawn_bonus = [0, 20, 30, 50, 80, 120, 180, 250]  # By rank
        advanced_passed_bonus = 30
        
        for pawn_square in pawns:
            if self._is_passed_pawn_bitboard(board, pawn_square, color):
                rank = chess.square_rank(pawn_square)
                if not color:  # Black pawns
                    rank = 7 - rank
                
                # Base passed pawn bonus
                bonus = passed_pawn_bonus[rank]
                
                # Advanced passed pawn (6th rank or higher)
                if rank >= 5:
                    bonus += advanced_passed_bonus
                
                # Connected passed pawns get extra bonus
                if self._has_connected_passed_pawn_bitboard(board, pawn_square, color):
                    bonus += 20
                
                score += bonus
        
        return score
    
    def _evaluate_isolated_pawns_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate isolated pawns using bitboard operations"""
        score = 0.0
        isolated_pawn_penalty = 15
        
        for pawn_square in pawns:
            if self._is_isolated_pawn_bitboard(board, pawn_square, color):
                penalty = isolated_pawn_penalty
                
                # Isolated pawns on open files are worse
                if self._is_on_open_file_bitboard(board, pawn_square):
                    penalty += 10
                
                score -= penalty
        
        return score
    
    def _evaluate_doubled_pawns_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate doubled pawns using bitboard operations"""
        score = 0.0
        doubled_pawn_penalty = 25
        file_counts = {}
        
        # Count pawns per file
        for pawn_square in pawns:
            file_idx = chess.square_file(pawn_square)
            file_counts[file_idx] = file_counts.get(file_idx, 0) + 1
        
        # Penalize multiple pawns on same file
        for file_idx, count in file_counts.items():
            if count > 1:
                penalty = doubled_pawn_penalty * (count - 1)
                score -= penalty
        
        return score
    
    def _evaluate_backward_pawns_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate backward pawns using bitboard operations"""
        score = 0.0
        backward_pawn_penalty = 12
        
        for pawn_square in pawns:
            if self._is_backward_pawn_bitboard(board, pawn_square, color):
                penalty = backward_pawn_penalty
                
                # Backward pawns on semi-open files are worse
                if self._is_on_semi_open_file_bitboard(board, pawn_square, color):
                    penalty += 8
                
                score -= penalty
        
        return score
    
    def _evaluate_connected_pawns_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate connected pawns using bitboard operations"""
        score = 0.0
        connected_pawn_bonus = 8
        
        for pawn_square in pawns:
            if self._has_pawn_support_bitboard(board, pawn_square, color):
                score += connected_pawn_bonus
        
        return score
    
    def _evaluate_pawn_chains_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate pawn chains using bitboard operations"""
        score = 0.0
        pawn_chain_bonus = 5
        chain_lengths = self._find_pawn_chains_bitboard(board, pawns, color)
        
        for length in chain_lengths:
            if length >= 2:
                score += pawn_chain_bonus * length
        
        return score
    
    def _evaluate_pawn_storms_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate pawn storms using bitboard operations"""
        score = 0.0
        pawn_storm_bonus = 10
        
        # Find enemy king
        enemy_king_square = board.king(not color)
        if enemy_king_square is None:
            return 0.0
        
        enemy_king_file = chess.square_file(enemy_king_square)
        
        for pawn_square in pawns:
            pawn_file = chess.square_file(pawn_square)
            pawn_rank = chess.square_rank(pawn_square)
            
            # Check if pawn is advancing toward enemy king
            if abs(pawn_file - enemy_king_file) <= 1:  # Adjacent or same file
                if color and pawn_rank >= 4:  # White pawn advanced
                    score += pawn_storm_bonus
                elif not color and pawn_rank <= 3:  # Black pawn advanced
                    score += pawn_storm_bonus
        
        return score
    
    # Helper methods for bitboard pawn analysis
    
    def _is_passed_pawn_bitboard(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn is passed using bitboard operations"""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        # Get enemy pawns
        enemy_pawns = board.pieces(chess.PAWN, not color)
        
        # Check files that could block this pawn (same file + adjacent files)
        blocking_files = [pawn_file]
        if pawn_file > 0:
            blocking_files.append(pawn_file - 1)
        if pawn_file < 7:
            blocking_files.append(pawn_file + 1)
        
        # Check if any enemy pawns can block
        for file_idx in blocking_files:
            file_mask = self.FILE_A << file_idx
            enemy_pawns_on_file = enemy_pawns & file_mask
            
            if enemy_pawns_on_file:
                for enemy_square in enemy_pawns_on_file:
                    enemy_rank = chess.square_rank(enemy_square)
                    
                    # Check if enemy pawn is ahead of our pawn
                    if color and enemy_rank > pawn_rank:  # White pawn
                        return False
                    elif not color and enemy_rank < pawn_rank:  # Black pawn
                        return False
        
        return True
    
    def _is_isolated_pawn_bitboard(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn is isolated using bitboard operations"""
        pawn_file = chess.square_file(pawn_square)
        our_pawns = board.pieces(chess.PAWN, color)
        
        # Check adjacent files for friendly pawns
        adjacent_files = []
        if pawn_file > 0:
            adjacent_files.append(pawn_file - 1)
        if pawn_file < 7:
            adjacent_files.append(pawn_file + 1)
        
        for file_idx in adjacent_files:
            file_mask = self.FILE_A << file_idx
            if our_pawns & file_mask:
                return False
        
        return True
    
    def _is_backward_pawn_bitboard(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn is backward using bitboard operations"""
        # Simplified backward pawn detection
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        our_pawns = board.pieces(chess.PAWN, color)
        
        # Check if adjacent pawns are ahead
        adjacent_files = [pawn_file - 1, pawn_file + 1]
        
        for file_idx in adjacent_files:
            if 0 <= file_idx <= 7:
                file_mask = self.FILE_A << file_idx
                adjacent_pawns = our_pawns & file_mask
                
                for adj_square in adjacent_pawns:
                    adj_rank = chess.square_rank(adj_square)
                    
                    # If adjacent pawn is ahead and we can't advance safely
                    if (color and adj_rank > pawn_rank) or (not color and adj_rank < pawn_rank):
                        return True
        
        return False
    
    def _has_pawn_support_bitboard(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn has support using bitboard operations"""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        our_pawns = board.pieces(chess.PAWN, color)
        
        # Check diagonal squares behind for supporting pawns
        support_squares = []
        if color:  # White
            if pawn_rank > 0:
                if pawn_file > 0:
                    support_squares.append(chess.square(pawn_file - 1, pawn_rank - 1))
                if pawn_file < 7:
                    support_squares.append(chess.square(pawn_file + 1, pawn_rank - 1))
        else:  # Black
            if pawn_rank < 7:
                if pawn_file > 0:
                    support_squares.append(chess.square(pawn_file - 1, pawn_rank + 1))
                if pawn_file < 7:
                    support_squares.append(chess.square(pawn_file + 1, pawn_rank + 1))
        
        for support_square in support_squares:
            if support_square in our_pawns:
                return True
        
        return False
    
    def _find_pawn_chains_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> List[int]:
        """Find pawn chains using bitboard operations"""
        # Simplified chain detection - count connected groups
        chains = []
        visited = set()
        
        for pawn_square in pawns:
            if pawn_square not in visited:
                chain_length = self._count_chain_length_bitboard(board, pawn_square, color, visited, pawns)
                if chain_length > 0:
                    chains.append(chain_length)
        
        return chains
    
    def _count_chain_length_bitboard(self, board: chess.Board, start_square: int, color: bool, visited: set, pawns: chess.SquareSet) -> int:
        """Count chain length recursively using bitboard operations"""
        if start_square in visited or start_square not in pawns:
            return 0
        
        visited.add(start_square)
        length = 1
        
        # Check connected pawns
        pawn_file = chess.square_file(start_square)
        pawn_rank = chess.square_rank(start_square)
        
        # Check diagonal connections
        connections = []
        if color:  # White
            if pawn_rank < 7:
                if pawn_file > 0:
                    connections.append(chess.square(pawn_file - 1, pawn_rank + 1))
                if pawn_file < 7:
                    connections.append(chess.square(pawn_file + 1, pawn_rank + 1))
        else:  # Black
            if pawn_rank > 0:
                if pawn_file > 0:
                    connections.append(chess.square(pawn_file - 1, pawn_rank - 1))
                if pawn_file < 7:
                    connections.append(chess.square(pawn_file + 1, pawn_rank - 1))
        
        for connected_square in connections:
            length += self._count_chain_length_bitboard(board, connected_square, color, visited, pawns)
        
        return length
    
    def _has_connected_passed_pawn_bitboard(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if passed pawn has connected passed pawn using bitboard operations"""
        pawn_file = chess.square_file(pawn_square)
        adjacent_files = [pawn_file - 1, pawn_file + 1]
        
        for file_idx in adjacent_files:
            if 0 <= file_idx <= 7:
                file_mask = self.FILE_A << file_idx
                our_pawns = board.pieces(chess.PAWN, color)
                adjacent_pawns = our_pawns & file_mask
                
                for adj_square in adjacent_pawns:
                    if self._is_passed_pawn_bitboard(board, adj_square, color):
                        return True
        
        return False
    
    def _is_on_open_file_bitboard(self, board: chess.Board, pawn_square: int) -> bool:
        """Check if pawn is on open file using bitboard operations"""
        pawn_file = chess.square_file(pawn_square)
        file_mask = self.FILE_A << pawn_file
        
        # Check if any other pawns exist on this file
        all_pawns = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
        other_pawns = all_pawns & file_mask
        
        return len(other_pawns) <= 1  # Only our pawn on the file
    
    def _is_on_semi_open_file_bitboard(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn is on semi-open file using bitboard operations"""
        pawn_file = chess.square_file(pawn_square)
        file_mask = self.FILE_A << pawn_file
        
        # Check if enemy has no pawns on this file
        enemy_pawns = board.pieces(chess.PAWN, not color)
        enemy_pawns_on_file = enemy_pawns & file_mask
        
        return len(enemy_pawns_on_file) == 0

    def evaluate_king_safety(self, board: chess.Board, color: bool) -> float:
        """
        V12.6 CONSOLIDATED: Comprehensive king safety evaluation using bitboards
        Returns score from the perspective of the given color
        """
        total_score = 0.0
        
        king_square = board.king(color)
        if king_square is None:
            return -1000.0  # King missing - critical error
        
        # Determine game phase for king safety vs activity balance
        material_count = self._count_material_bitboard(board)
        is_endgame = material_count < 2000  # Rough endgame threshold
        
        if is_endgame:
            # Endgame: King activity is important
            total_score += self._evaluate_king_activity_bitboard(board, king_square, color)
        else:
            # Opening/Middlegame: King safety is paramount
            total_score += self._evaluate_pawn_shelter_bitboard(board, king_square, color)
            total_score += self._evaluate_castling_rights_bitboard(board, color)
            total_score += self._evaluate_king_exposure_bitboard(board, king_square, color)
            total_score += self._evaluate_escape_squares_bitboard(board, king_square, color)
            total_score += self._evaluate_attack_zone_bitboard(board, king_square, color)
            total_score += self._evaluate_enemy_pawn_storms_bitboard(board, king_square, color)
        
        return total_score
    
    def _count_material_bitboard(self, board: chess.Board) -> int:
        """Count total material on board using bitboard operations"""
        material = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            white_pieces = board.pieces(piece_type, chess.WHITE)
            black_pieces = board.pieces(piece_type, chess.BLACK)
            piece_value = [100, 320, 330, 500, 900][piece_type - 1]
            material += (len(white_pieces) + len(black_pieces)) * piece_value
        return material
    
    def _evaluate_pawn_shelter_bitboard(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate pawn shelter around the king using bitboards"""
        score = 0.0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        friendly_pawns = board.pieces(chess.PAWN, color)
        shelter_pawns = 0
        pawn_shelter_bonus = [0, 5, 10, 15, 20]  # By number of shelter pawns
        
        # Check files around the king (king file and adjacent files)
        for file_offset in [-1, 0, 1]:
            check_file = king_file + file_offset
            if 0 <= check_file <= 7:
                file_mask = self.FILE_A << check_file
                pawns_on_file = friendly_pawns & file_mask
                
                # Look for pawn shelter in front of king
                shelter_found = False
                for pawn_square in pawns_on_file:
                    pawn_rank = chess.square_rank(pawn_square)
                    
                    # Check if pawn is providing shelter
                    if color and pawn_rank > king_rank:  # White king
                        if pawn_rank - king_rank <= 2:
                            shelter_pawns += 1
                            shelter_found = True
                        break
                    elif not color and pawn_rank < king_rank:  # Black king
                        if king_rank - pawn_rank <= 2:
                            shelter_pawns += 1
                            shelter_found = True
                        break
                
                # Penalty for missing pawn shelter
                if not shelter_found:
                    score -= 10
        
        # Bonus for pawn shelter
        if shelter_pawns < len(pawn_shelter_bonus):
            score += pawn_shelter_bonus[shelter_pawns]
        else:
            score += pawn_shelter_bonus[-1]
        
        return score
    
    def _evaluate_castling_rights_bitboard(self, board: chess.Board, color: bool) -> float:
        """Evaluate castling rights value using bitboard operations"""
        score = 0.0
        castling_rights_bonus = 25
        
        if color:  # White
            if board.has_kingside_castling_rights(chess.WHITE):
                score += castling_rights_bonus
            if board.has_queenside_castling_rights(chess.WHITE):
                score += castling_rights_bonus * 0.8  # Queenside slightly less valuable
        else:  # Black
            if board.has_kingside_castling_rights(chess.BLACK):
                score += castling_rights_bonus
            if board.has_queenside_castling_rights(chess.BLACK):
                score += castling_rights_bonus * 0.8
        
        return score
    
    def _evaluate_king_exposure_bitboard(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate king exposure to enemy attacks using bitboards"""
        score = 0.0
        king_exposure_penalty = 30
        
        # Check if king is on an open file or rank
        if self._is_on_open_file_bitboard(board, king_square):
            score -= king_exposure_penalty
        
        if self._is_on_open_rank_bitboard(board, king_square):
            score -= king_exposure_penalty * 0.5  # Less dangerous than open file
        
        # Check for enemy pieces attacking king vicinity
        enemy_attacks = self._count_enemy_attacks_near_king_bitboard(board, king_square, color)
        score -= enemy_attacks * 5
        
        return score
    
    def _evaluate_escape_squares_bitboard(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate available escape squares for the king using bitboards"""
        score = 0.0
        escape_squares = 0
        escape_square_bonus = 8
        
        # Check all adjacent squares
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                if rank_offset == 0 and file_offset == 0:
                    continue  # Skip king's current square
                
                target_square = king_square + rank_offset * 8 + file_offset
                
                if 0 <= target_square <= 63:
                    # Check if square is safe and accessible
                    if self._is_safe_escape_square_bitboard(board, target_square, color):
                        escape_squares += 1
        
        score += escape_squares * escape_square_bonus
        
        # Penalty for having very few escape squares
        if escape_squares <= 1:
            score -= 20
        
        return score
    
    def _evaluate_attack_zone_bitboard(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate enemy control of squares around the king using bitboards"""
        score = 0.0
        attack_zone_penalty = 12
        
        # Define attack zone (3x3 squares around king)
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        enemy_controlled = 0
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                target_file = king_file + file_offset
                target_rank = king_rank + rank_offset
                
                if 0 <= target_file <= 7 and 0 <= target_rank <= 7:
                    target_square = target_rank * 8 + target_file
                    if self._is_square_attacked_by_enemy_bitboard(board, target_square, color):
                        enemy_controlled += 1
        
        score -= enemy_controlled * attack_zone_penalty
        return score
    
    def _evaluate_enemy_pawn_storms_bitboard(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate enemy pawn storms using bitboards"""
        score = 0.0
        enemy_pawn_storm_penalty = 15
        
        king_file = chess.square_file(king_square)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        
        # Check for advancing enemy pawns near king
        for pawn_square in enemy_pawns:
            pawn_file = chess.square_file(pawn_square)
            pawn_rank = chess.square_rank(pawn_square)
            
            # Check if pawn is advancing toward our king
            if abs(pawn_file - king_file) <= 1:  # Adjacent or same file
                if not color and pawn_rank >= 4:  # Enemy white pawn advanced
                    score -= enemy_pawn_storm_penalty
                elif color and pawn_rank <= 3:  # Enemy black pawn advanced
                    score -= enemy_pawn_storm_penalty
        
        return score
    
    def _evaluate_king_activity_bitboard(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate king activity in endgame using bitboards"""
        score = 0.0
        king_activity_bonus = 5
        
        # King centralization in endgame
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Distance from center
        center_distance = max(abs(king_file - 3.5), abs(king_rank - 3.5))
        centralization_bonus = [12, 8, 4, 2][min(int(center_distance), 3)]
        
        score += centralization_bonus
        
        # King mobility in endgame
        mobility = 0
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                if rank_offset == 0 and file_offset == 0:
                    continue
                
                target_square = king_square + rank_offset * 8 + file_offset
                if 0 <= target_square <= 63:
                    target_piece = board.piece_at(target_square)
                    if target_piece is None or target_piece.color != color:
                        mobility += 1
        
        score += mobility * king_activity_bonus
        
        return score
    
    def _is_on_open_rank_bitboard(self, board: chess.Board, square: int) -> bool:
        """Check if square is on open rank using bitboards"""
        rank = chess.square_rank(square)
        rank_mask = self.RANK_1 << (rank * 8)
        
        all_pieces = board.occupied
        pieces_on_rank = all_pieces & rank_mask
        
        return bin(pieces_on_rank).count('1') <= 2  # Only kings on the rank
    
    def _count_enemy_attacks_near_king_bitboard(self, board: chess.Board, king_square: int, color: bool) -> int:
        """Count enemy attacks near king using bitboards"""
        attacks = 0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Check 3x3 area around king
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                target_file = king_file + file_offset
                target_rank = king_rank + rank_offset
                
                if 0 <= target_file <= 7 and 0 <= target_rank <= 7:
                    target_square = target_rank * 8 + target_file
                    if self._is_square_attacked_by_enemy_bitboard(board, target_square, color):
                        attacks += 1
        
        return attacks
    
    def _is_safe_escape_square_bitboard(self, board: chess.Board, square: int, color: bool) -> bool:
        """Check if square is safe escape square using bitboards"""
        # Check if square is occupied by our piece
        piece = board.piece_at(square)
        if piece and piece.color == color:
            return False
        
        # Check if square is attacked by enemy
        if self._is_square_attacked_by_enemy_bitboard(board, square, color):
            return False
        
        return True
    
    def _is_square_attacked_by_enemy_bitboard(self, board: chess.Board, square: int, our_color: bool) -> bool:
        """Check if square is attacked by enemy using bitboards"""
        # This is a simplified version - full implementation would check all enemy piece attacks
        enemy_pieces = board.occupied_co[not our_color]
        
        # Quick check for pawn attacks
        enemy_pawns = board.pieces(chess.PAWN, not our_color)
        for pawn_square in enemy_pawns:
            if self._pawn_attacks_square_bitboard(pawn_square, square, not our_color):
                return True
        
        # Check for knight attacks
        enemy_knights = board.pieces(chess.KNIGHT, not our_color)
        for knight_square in enemy_knights:
            knight_attacks = self.KNIGHT_ATTACKS[knight_square]
            if knight_attacks & (1 << square):
                return True
        
        return False
    
    def _pawn_attacks_square_bitboard(self, pawn_square: int, target_square: int, pawn_color: bool) -> bool:
        """Check if pawn attacks target square using bitboards"""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        target_file = chess.square_file(target_square)
        target_rank = chess.square_rank(target_square)
        
        # Check diagonal pawn attacks
        if abs(pawn_file - target_file) == 1:
            if pawn_color and target_rank == pawn_rank + 1:  # White pawn
                return True
            elif not pawn_color and target_rank == pawn_rank - 1:  # Black pawn
                return True
        
        return False


class V7P3RScoringCalculationBitboard:
    """
    Drop-in replacement for the slow scoring calculator
    Uses bitboards for ultra-high performance
    """
    
    def __init__(self, piece_values: Dict[int, int], enable_nudges: bool = False):
        self.piece_values = piece_values
        self.bitboard_evaluator = V7P3RBitboardEvaluator(piece_values, enable_nudges=enable_nudges)
    
    def calculate_score_optimized(self, board: chess.Board, color: chess.Color, endgame_factor: float = 0.0) -> float:
        """
        Ultra-fast evaluation using bitboards
        Target: 20,000+ NPS
        """
        return self.bitboard_evaluator.evaluate_bitboard(board, color)
    
    def detect_bitboard_tactics(self, board: chess.Board, move: chess.Move) -> float:
        """
        V12.6 CONSOLIDATED: Detect tactical patterns using bitboard operations
        Delegate to the bitboard evaluator for consistency
        """
        return self.bitboard_evaluator.detect_bitboard_tactics(board, move)
    
    def evaluate_pawn_structure(self, board: chess.Board, color: bool) -> float:
        """
        V12.6 CONSOLIDATED: Evaluate pawn structure using bitboard operations
        Delegate to the bitboard evaluator for consistency
        """
        return self.bitboard_evaluator.evaluate_pawn_structure(board, color)
    
    def evaluate_king_safety(self, board: chess.Board, color: bool) -> float:
        """
        V12.6 CONSOLIDATED: Evaluate king safety using bitboard operations
        Delegate to the bitboard evaluator for consistency
        """
        return self.bitboard_evaluator.evaluate_king_safety(board, color)
