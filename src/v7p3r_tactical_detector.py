#!/usr/bin/env python3
"""
V7P3R Tactical Pattern Detection System
Ultra-fast bitboard-based tactical pattern recognition for pins, forks, skewers, and more

Inspired by Mikhail Tal's tactical genius: "Chess is 99% tactics"
Uses bitboard operations for maximum performance while detecting critical tactical motifs
"""

import chess
from typing import List, Dict, Tuple, Set, NamedTuple, Optional
from dataclasses import dataclass


@dataclass
class TacticalPattern:
    """Represents a detected tactical pattern"""
    pattern_type: str  # 'pin', 'fork', 'skewer', 'discovered_attack', etc.
    attacker_square: int
    target_squares: List[int]
    victim_pieces: List[int]  # chess.Piece values
    tactical_value: float  # Estimated centipawn value
    forcing_level: int  # 1=suggestion, 2=strong, 3=forcing, 4=critical


class V7P3RTacticalDetector:
    """
    High-performance tactical pattern detection using bitboard operations
    Focuses on patterns that fire frequently and provide measurable value
    """
    
    def __init__(self):
        # Tactical pattern values in centipawns
        self.PATTERN_VALUES = {
            'pin': 15,              # Pin lower value piece to higher value
            'skewer': 20,           # Higher value piece must move, exposing lower
            'fork': 25,             # One piece attacks two+ pieces
            'discovered_attack': 20, # Piece moves revealing attack
            # REMOVED: patterns that never fired in profiling
            # 'deflection': 10,       # Force piece to move from important duty
            # 'removing_guard': 15,   # Capture defender to attack protected piece
            # 'tactical_overload': 12, # Piece defending too many things
        }
        
        # Pre-calculate attack patterns for fast lookup
        self._init_attack_patterns()
        
        # Profiling counters - remove patterns that never fire
        self.pattern_counters = {pattern: 0 for pattern in self.PATTERN_VALUES.keys()}
        
        # OPTIMIZATION: Pattern detection cache
        self.pattern_cache = {}  # position_hash -> List[TacticalPattern]
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _init_attack_patterns(self):
        """Initialize pre-calculated attack patterns for each piece type"""
        
        # Knight attack patterns
        self.KNIGHT_ATTACKS = [0] * 64
        for sq in range(64):
            self.KNIGHT_ATTACKS[sq] = self._calc_knight_attacks(sq)
            
        # King attack patterns  
        self.KING_ATTACKS = [0] * 64
        for sq in range(64):
            self.KING_ATTACKS[sq] = self._calc_king_attacks(sq)
            
        # Direction masks for sliding pieces
        self.RANK_MASKS = [0] * 64
        self.FILE_MASKS = [0] * 64
        self.DIAGONAL_MASKS = [0] * 64
        self.ANTI_DIAGONAL_MASKS = [0] * 64
        
        for sq in range(64):
            rank = sq // 8
            file = sq % 8
            self.RANK_MASKS[sq] = 0xFF << (rank * 8)
            self.FILE_MASKS[sq] = 0x0101010101010101 << file
            self.DIAGONAL_MASKS[sq] = self._calc_diagonal_mask(sq)
            self.ANTI_DIAGONAL_MASKS[sq] = self._calc_anti_diagonal_mask(sq)
    
    def _calc_knight_attacks(self, square: int) -> int:
        """Calculate knight attack bitboard for given square"""
        rank, file = divmod(square, 8)
        attacks = 0
        
        # All possible knight moves
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        for dr, df in knight_moves:
            new_rank, new_file = rank + dr, file + df
            if 0 <= new_rank < 8 and 0 <= new_file < 8:
                attacks |= 1 << (new_rank * 8 + new_file)
                
        return attacks
    
    def _calc_king_attacks(self, square: int) -> int:
        """Calculate king attack bitboard for given square"""
        rank, file = divmod(square, 8)
        attacks = 0
        
        # All possible king moves
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                if dr == 0 and df == 0:
                    continue
                new_rank, new_file = rank + dr, file + df
                if 0 <= new_rank < 8 and 0 <= new_file < 8:
                    attacks |= 1 << (new_rank * 8 + new_file)
                    
        return attacks
    
    def _calc_diagonal_mask(self, square: int) -> int:
        """Calculate diagonal mask for given square"""
        rank, file = divmod(square, 8)
        mask = 0
        
        # Positive diagonal (up-right, down-left)
        for i in range(8):
            r, f = rank + i, file + i
            if 0 <= r < 8 and 0 <= f < 8:
                mask |= 1 << (r * 8 + f)
            r, f = rank - i, file - i
            if 0 <= r < 8 and 0 <= f < 8:
                mask |= 1 << (r * 8 + f)
                
        return mask
    
    def _calc_anti_diagonal_mask(self, square: int) -> int:
        """Calculate anti-diagonal mask for given square"""
        rank, file = divmod(square, 8)
        mask = 0
        
        # Negative diagonal (up-left, down-right)
        for i in range(8):
            r, f = rank + i, file - i
            if 0 <= r < 8 and 0 <= f < 8:
                mask |= 1 << (r * 8 + f)
            r, f = rank - i, file + i
            if 0 <= r < 8 and 0 <= f < 8:
                mask |= 1 << (r * 8 + f)
                
        return mask
    
    def detect_all_tactical_patterns(self, board: chess.Board, for_white: bool) -> List[TacticalPattern]:
        """
        Detect all tactical patterns for the given side
        Returns list of patterns sorted by tactical value (highest first)
        """
        patterns = []
        
        # Detect each pattern type
        patterns.extend(self.detect_pins(board, for_white))
        patterns.extend(self.detect_forks(board, for_white))
        patterns.extend(self.detect_skewers(board, for_white))
        patterns.extend(self.detect_discovered_attacks(board, for_white))
        
        # Sort by tactical value (most valuable first)
        patterns.sort(key=lambda p: p.tactical_value, reverse=True)
        
        return patterns
    
    def detect_pins(self, board: chess.Board, for_white: bool) -> List[TacticalPattern]:
        """
        Detect pin patterns using bitboard operations
        A pin occurs when a piece cannot move without exposing a more valuable piece
        """
        patterns = []
        enemy_color = not for_white
        
        # Get our sliding pieces (queen, rook, bishop) that can create pins
        our_queens = board.pieces(chess.QUEEN, for_white)
        our_rooks = board.pieces(chess.ROOK, for_white)  
        our_bishops = board.pieces(chess.BISHOP, for_white)
        
        # Check each of our sliding pieces for potential pins
        for piece_type, piece_squares in [(chess.QUEEN, our_queens), 
                                         (chess.ROOK, our_rooks), 
                                         (chess.BISHOP, our_bishops)]:
            
            for attacker_sq in piece_squares:
                pin_patterns = self._find_pins_from_square(board, attacker_sq, piece_type, for_white)
                patterns.extend(pin_patterns)
                
        self.pattern_counters['pin'] += len(patterns)
        return patterns
    
    def _find_pins_from_square(self, board: chess.Board, attacker_sq: int, piece_type: int, for_white: bool) -> List[TacticalPattern]:
        """Find all pins created by a piece from a specific square"""
        patterns = []
        enemy_color = not for_white
        
        # Get attack directions based on piece type
        if piece_type in [chess.QUEEN, chess.ROOK]:
            # Horizontal and vertical lines
            attack_masks = [self.RANK_MASKS[attacker_sq], self.FILE_MASKS[attacker_sq]]
        if piece_type in [chess.QUEEN, chess.BISHOP]:
            # Diagonal lines
            attack_masks = [self.DIAGONAL_MASKS[attacker_sq], self.ANTI_DIAGONAL_MASKS[attacker_sq]]
            
        for mask in attack_masks:
            # Find pieces on this line
            pieces_on_line = []
            for sq in range(64):
                if (mask & (1 << sq)) and board.piece_at(sq):
                    pieces_on_line.append((sq, board.piece_at(sq)))
                    
            # Sort by distance from attacker
            pieces_on_line.sort(key=lambda x: abs(x[0] - attacker_sq))
            
            # Look for pin pattern: [Attacker] -> [Enemy Piece] -> [Valuable Enemy Piece]
            for i in range(len(pieces_on_line) - 1):
                first_piece_sq, first_piece = pieces_on_line[i]
                second_piece_sq, second_piece = pieces_on_line[i + 1]
                
                # Skip if first piece is ours
                if first_piece.color == for_white:
                    continue
                    
                # Check if we have a pin: enemy piece blocking attack on more valuable enemy piece
                if (first_piece.color == enemy_color and 
                    second_piece.color == enemy_color and
                    self._get_piece_value(second_piece.piece_type) > self._get_piece_value(first_piece.piece_type)):
                    
                    patterns.append(TacticalPattern(
                        pattern_type='pin',
                        attacker_square=attacker_sq,
                        target_squares=[first_piece_sq, second_piece_sq],
                        victim_pieces=[first_piece.piece_type, second_piece.piece_type],
                        tactical_value=self.PATTERN_VALUES['pin'] + 
                                     (self._get_piece_value(second_piece.piece_type) - 
                                      self._get_piece_value(first_piece.piece_type)) * 0.1,
                        forcing_level=3 if second_piece.piece_type == chess.KING else 2
                    ))
                    
        return patterns
    
    def detect_forks(self, board: chess.Board, for_white: bool) -> List[TacticalPattern]:
        """
        Detect fork patterns - one piece attacking two+ enemy pieces
        Focus on knights and pawns as most common fork pieces
        """
        patterns = []
        enemy_color = not for_white
        
        # Check knight forks
        our_knights = board.pieces(chess.KNIGHT, for_white)
        for knight_sq in our_knights:
            fork_pattern = self._check_knight_fork(board, knight_sq, for_white)
            if fork_pattern:
                patterns.append(fork_pattern)
                
        # Check pawn forks  
        our_pawns = board.pieces(chess.PAWN, for_white)
        for pawn_sq in our_pawns:
            fork_pattern = self._check_pawn_fork(board, pawn_sq, for_white)
            if fork_pattern:
                patterns.append(fork_pattern)
                
        # Check queen forks on multiple pieces
        our_queens = board.pieces(chess.QUEEN, for_white)
        for queen_sq in our_queens:
            fork_pattern = self._check_queen_fork(board, queen_sq, for_white)
            if fork_pattern:
                patterns.append(fork_pattern)
                
        self.pattern_counters['fork'] += len(patterns)
        return patterns
    
    def _check_knight_fork(self, board: chess.Board, knight_sq: int, for_white: bool) -> Optional[TacticalPattern]:
        """Check if knight can fork two or more enemy pieces"""
        enemy_color = not for_white
        knight_attacks = self.KNIGHT_ATTACKS[knight_sq]
        
        # Find enemy pieces under attack
        attacked_pieces = []
        for sq in range(64):
            if (knight_attacks & (1 << sq)) and board.piece_at(sq):
                piece = board.piece_at(sq)
                if piece and piece.color == enemy_color:
                    attacked_pieces.append((sq, piece))
                    
        # Fork requires attacking 2+ pieces
        if len(attacked_pieces) >= 2:
            total_value = sum(self._get_piece_value(piece.piece_type) for _, piece in attacked_pieces)
            
            return TacticalPattern(
                pattern_type='fork',
                attacker_square=knight_sq,
                target_squares=[sq for sq, _ in attacked_pieces],
                victim_pieces=[piece.piece_type for _, piece in attacked_pieces],
                tactical_value=self.PATTERN_VALUES['fork'] + total_value * 0.05,
                forcing_level=4 if any(piece.piece_type == chess.KING for _, piece in attacked_pieces) else 3
            )
            
        return None
    
    def _check_pawn_fork(self, board: chess.Board, pawn_sq: int, for_white: bool) -> Optional[TacticalPattern]:
        """Check if pawn can fork enemy pieces"""
        enemy_color = not for_white
        rank, file = divmod(pawn_sq, 8)
        
        # Pawn attack squares
        direction = 1 if for_white else -1
        attack_squares = []
        
        if 0 <= file - 1 < 8:
            attack_squares.append((rank + direction) * 8 + (file - 1))
        if 0 <= file + 1 < 8:
            attack_squares.append((rank + direction) * 8 + (file + 1))
            
        # Find attacked enemy pieces
        attacked_pieces = []
        for sq in attack_squares:
            if 0 <= sq < 64 and board.piece_at(sq):
                piece = board.piece_at(sq)
                if piece and piece.color == enemy_color:
                    attacked_pieces.append((sq, piece))
                    
        # Pawn fork requires attacking 2 pieces
        if len(attacked_pieces) == 2:
            total_value = sum(self._get_piece_value(piece.piece_type) for _, piece in attacked_pieces)
            
            return TacticalPattern(
                pattern_type='fork',
                attacker_square=pawn_sq,
                target_squares=[sq for sq, _ in attacked_pieces],
                victim_pieces=[piece.piece_type for _, piece in attacked_pieces],
                tactical_value=self.PATTERN_VALUES['fork'] + total_value * 0.08,
                forcing_level=3
            )
            
        return None
    
    def _check_queen_fork(self, board: chess.Board, queen_sq: int, for_white: bool) -> Optional[TacticalPattern]:
        """Check if queen can fork multiple high-value enemy pieces"""
        enemy_color = not for_white
        
        # Get all squares queen attacks
        queen_attacks = board.attacks(queen_sq)
        
        # Find high-value enemy pieces under attack
        attacked_pieces = []
        for sq in queen_attacks:
            piece = board.piece_at(sq)
            if piece and piece.color == enemy_color and piece.piece_type in [chess.ROOK, chess.QUEEN, chess.KING]:
                attacked_pieces.append((sq, piece))
                
        # Queen fork is valuable when attacking 2+ high-value pieces
        if len(attacked_pieces) >= 2:
            total_value = sum(self._get_piece_value(piece.piece_type) for _, piece in attacked_pieces)
            
            return TacticalPattern(
                pattern_type='fork',
                attacker_square=queen_sq,
                target_squares=[sq for sq, _ in attacked_pieces],
                victim_pieces=[piece.piece_type for _, piece in attacked_pieces],
                tactical_value=self.PATTERN_VALUES['fork'] + total_value * 0.03,
                forcing_level=4 if any(piece.piece_type == chess.KING for _, piece in attacked_pieces) else 2
            )
            
        return None
    
    def detect_skewers(self, board: chess.Board, for_white: bool) -> List[TacticalPattern]:
        """
        Detect skewer patterns - attack high-value piece that must move, exposing lower-value piece
        """
        patterns = []
        enemy_color = not for_white
        
        # Our sliding pieces can create skewers
        sliding_pieces = [(chess.QUEEN, board.pieces(chess.QUEEN, for_white)),
                         (chess.ROOK, board.pieces(chess.ROOK, for_white)),
                         (chess.BISHOP, board.pieces(chess.BISHOP, for_white))]
        
        for piece_type, piece_squares in sliding_pieces:
            for attacker_sq in piece_squares:
                skewer_patterns = self._find_skewers_from_square(board, attacker_sq, piece_type, for_white)
                patterns.extend(skewer_patterns)
                
        self.pattern_counters['skewer'] += len(patterns)
        return patterns
    
    def _find_skewers_from_square(self, board: chess.Board, attacker_sq: int, piece_type: int, for_white: bool) -> List[TacticalPattern]:
        """Find skewers created by piece from specific square"""
        patterns = []
        enemy_color = not for_white
        
        # Get attack lines for this piece type
        attack_lines = []
        if piece_type in [chess.QUEEN, chess.ROOK]:
            attack_lines.extend([self.RANK_MASKS[attacker_sq], self.FILE_MASKS[attacker_sq]])
        if piece_type in [chess.QUEEN, chess.BISHOP]:
            attack_lines.extend([self.DIAGONAL_MASKS[attacker_sq], self.ANTI_DIAGONAL_MASKS[attacker_sq]])
            
        for line_mask in attack_lines:
            # Find enemy pieces on this line
            enemy_pieces_on_line = []
            for sq in range(64):
                if (line_mask & (1 << sq)) and board.piece_at(sq):
                    piece = board.piece_at(sq)
                    if piece.color == enemy_color:
                        enemy_pieces_on_line.append((sq, piece))
                        
            # Sort by distance from attacker
            enemy_pieces_on_line.sort(key=lambda x: abs(x[0] - attacker_sq))
            
            # Look for skewer: [Attacker] -> [High Value Enemy] -> [Lower Value Enemy]
            for i in range(len(enemy_pieces_on_line) - 1):
                first_sq, first_piece = enemy_pieces_on_line[i]
                second_sq, second_piece = enemy_pieces_on_line[i + 1]
                
                first_value = self._get_piece_value(first_piece.piece_type)
                second_value = self._get_piece_value(second_piece.piece_type)
                
                # Skewer: high value piece must move, exposing lower value piece
                if first_value > second_value:
                    patterns.append(TacticalPattern(
                        pattern_type='skewer',
                        attacker_square=attacker_sq,
                        target_squares=[first_sq, second_sq],
                        victim_pieces=[first_piece.piece_type, second_piece.piece_type],
                        tactical_value=self.PATTERN_VALUES['skewer'] + (first_value - second_value) * 0.1,
                        forcing_level=4 if first_piece.piece_type == chess.KING else 3
                    ))
                    
        return patterns
    
    def detect_discovered_attacks(self, board: chess.Board, for_white: bool) -> List[TacticalPattern]:
        """
        Detect discovered attack patterns - piece moves revealing attack from behind
        OPTIMIZED: Only check pieces that could realistically create discovered attacks
        """
        patterns = []
        
        # OPTIMIZATION: Only check pieces that could move to create discovered attacks
        # Focus on minor pieces and pawns that could uncover sliding piece attacks
        our_pieces = []
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP]:  # Skip rook/queen for performance
            for sq in board.pieces(piece_type, for_white):
                our_pieces.append((sq, piece_type))
                
        # OPTIMIZATION: Limit to first 10 pieces to avoid performance hit
        for piece_sq, piece_type in our_pieces[:10]:
            discovered_patterns = self._check_discovered_attack_from_square(board, piece_sq, for_white)
            patterns.extend(discovered_patterns)
            
        self.pattern_counters['discovered_attack'] += len(patterns)
        return patterns[:3]  # OPTIMIZATION: Limit to top 3 to avoid overflow
    
    def _check_discovered_attack_from_square(self, board: chess.Board, piece_sq: int, for_white: bool) -> List[TacticalPattern]:
        """Check if moving piece from square creates discovered attack"""
        patterns = []
        enemy_color = not for_white
        
        # Temporarily remove the piece and see what attacks are revealed
        piece = board.piece_at(piece_sq)
        if not piece:
            return patterns
            
        # Create a temporary board with piece removed
        temp_board = board.copy()
        temp_board.remove_piece_at(piece_sq)
        
        # Check all our sliding pieces to see if removing this piece reveals attacks
        sliding_pieces = [(chess.QUEEN, temp_board.pieces(chess.QUEEN, for_white)),
                         (chess.ROOK, temp_board.pieces(chess.ROOK, for_white)),
                         (chess.BISHOP, temp_board.pieces(chess.BISHOP, for_white))]
        
        for slider_type, slider_squares in sliding_pieces:
            for slider_sq in slider_squares:
                # Check if this slider now attacks valuable enemy pieces
                attacks = temp_board.attacks(slider_sq)
                
                valuable_targets = []
                for target_sq in attacks:
                    target_piece = temp_board.piece_at(target_sq)
                    if (target_piece and target_piece.color == enemy_color and 
                        target_piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]):
                        valuable_targets.append((target_sq, target_piece))
                        
                if valuable_targets:
                    total_value = sum(self._get_piece_value(tp.piece_type) for _, tp in valuable_targets)
                    patterns.append(TacticalPattern(
                        pattern_type='discovered_attack',
                        attacker_square=piece_sq,  # The piece that moves
                        target_squares=[sq for sq, _ in valuable_targets],
                        victim_pieces=[tp.piece_type for _, tp in valuable_targets],
                        tactical_value=self.PATTERN_VALUES['discovered_attack'] + total_value * 0.05,
                        forcing_level=3
                    ))
                    
        return patterns
    
    def _get_piece_value(self, piece_type: int) -> int:
        """Get standard piece value in centipawns"""
        values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0  # King value handled separately
        }
        return values.get(piece_type, 0)
    
    def get_tactical_score(self, board: chess.Board, for_white: bool) -> float:
        """
        Calculate total tactical score for position
        OPTIMIZED: Uses caching to avoid recalculating same positions
        """
        # Create cache key (simple hash of board state)
        cache_key = (str(board.board_fen()), for_white)
        
        if cache_key in self.pattern_cache:
            self.cache_hits += 1
            patterns = self.pattern_cache[cache_key]
        else:
            self.cache_misses += 1
            patterns = self.detect_all_tactical_patterns(board, for_white)
            
            # Cache result if cache isn't too large
            if len(self.pattern_cache) < 1000:
                self.pattern_cache[cache_key] = patterns
        
        # Weight patterns by forcing level and frequency
        total_score = 0.0
        for pattern in patterns[:6]:  # OPTIMIZATION: Limit to top 6 patterns for performance
            # Weight by forcing level: critical patterns get full value
            forcing_multiplier = pattern.forcing_level / 4.0
            total_score += pattern.tactical_value * forcing_multiplier
            
        return total_score
    
    def get_profiling_stats(self) -> Dict[str, int]:
        """Get pattern detection statistics for profiling"""
        stats = self.pattern_counters.copy()
        stats['cache_hits'] = self.cache_hits
        stats['cache_misses'] = self.cache_misses
        stats['cache_size'] = len(self.pattern_cache)
        if self.cache_hits + self.cache_misses > 0:
            stats['cache_hit_rate'] = int(self.cache_hits / (self.cache_hits + self.cache_misses) * 100)
        else:
            stats['cache_hit_rate'] = 0
        return stats
    
    def reset_profiling_stats(self):
        """Reset profiling counters"""
        self.pattern_counters = {pattern: 0 for pattern in self.PATTERN_VALUES.keys()}