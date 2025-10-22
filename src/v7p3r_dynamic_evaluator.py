#!/usr/bin/env python3
"""
V7P3R Dynamic Piece Value System v13.0
Context-dependent piece evaluation based on position, tactical patterns, and Tal's principles

Mikhail Tal insight: "A piece's value is not fixed - it depends on the position"
"""

import chess
from typing import Dict, List, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from v7p3r_tactical_detector import V7P3RTacticalDetector, TacticalPattern
else:
    # Runtime imports - handle gracefully if modules not available
    try:
        from v7p3r_tactical_detector import V7P3RTacticalDetector, TacticalPattern
    except ImportError:
        V7P3RTacticalDetector = None
        TacticalPattern = None


class V7P3RDynamicEvaluator:
    """
    Dynamic piece evaluation system that adjusts piece values based on:
    1. Position context (center, outposts, trapped pieces)
    2. Tactical potential (creating/defending tactical patterns)
    3. Mobility and activity
    4. Game phase (opening, middlegame, endgame)
    """
    
    def __init__(self, tactical_detector: V7P3RTacticalDetector):
        self.tactical_detector = tactical_detector
        
        # Base piece values (traditional starting point)
        self.BASE_VALUES = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0  # Special handling
        }
        
        # Activity bonuses based on mobility
        self.MOBILITY_BONUS = {
            chess.KNIGHT: 4,   # +4cp per move beyond base mobility
            chess.BISHOP: 3,   # +3cp per move beyond base mobility  
            chess.ROOK: 2,     # +2cp per move beyond base mobility
            chess.QUEEN: 1,    # +1cp per move beyond base mobility
        }
        
        # Base mobility expectations (piece should have at least this many moves)
        self.BASE_MOBILITY = {
            chess.KNIGHT: 4,   # Knights should have 4+ moves
            chess.BISHOP: 6,   # Bishops should have 6+ moves
            chess.ROOK: 8,     # Rooks should have 8+ moves  
            chess.QUEEN: 12,   # Queens should have 12+ moves
        }
        
        # Positional modifiers
        self.CENTER_BONUS = 15      # Bonus for pieces in center
        self.OUTPOST_BONUS = 20     # Bonus for pieces on outposts
        self.TRAPPED_PENALTY = -50  # Penalty for trapped pieces
        self.COORDINATION_BONUS = 8 # Bonus for pieces working together
        
        # Profiling stats
        self.evaluation_count = 0
        self.dynamic_adjustments = 0
        
    def evaluate_dynamic_position_value(self, board: chess.Board, for_white: bool) -> float:
        """
        Calculate dynamic position value incorporating tactical patterns and piece activity
        This is the main entry point for V13 evaluation
        """
        self.evaluation_count += 1
        
        # Get tactical patterns for context
        tactical_patterns = self.tactical_detector.detect_all_tactical_patterns(board, for_white)
        
        # Calculate dynamic piece values
        total_dynamic_value = 0.0
        
        # Evaluate each piece type dynamically
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            piece_squares = board.pieces(piece_type, for_white)
            for square in piece_squares:
                dynamic_value = self.calculate_piece_dynamic_value(
                    board, square, piece_type, for_white, tactical_patterns
                )
                total_dynamic_value += dynamic_value
                
        # Add tactical pattern bonuses
        tactical_bonus = self._calculate_tactical_bonus(tactical_patterns)
        total_dynamic_value += tactical_bonus
        
        # Add position complexity bonus (Tal's "dark forest")
        complexity_bonus = self._calculate_position_complexity_bonus(board, for_white)
        total_dynamic_value += complexity_bonus
        
        return total_dynamic_value
    
    def calculate_piece_dynamic_value(self, board: chess.Board, square: int, piece_type: int, 
                                    for_white: bool, tactical_patterns: List[TacticalPattern]) -> float:
        """Calculate dynamic value for a specific piece"""
        base_value = self.BASE_VALUES[piece_type]
        dynamic_adjustments = 0.0
        
        # 1. Activity bonus based on mobility
        mobility_bonus = self._calculate_mobility_bonus(board, square, piece_type, for_white)
        dynamic_adjustments += mobility_bonus
        
        # 2. Positional bonuses
        positional_bonus = self._calculate_positional_bonus(board, square, piece_type, for_white)
        dynamic_adjustments += positional_bonus
        
        # 3. Tactical involvement bonus
        tactical_bonus = self._calculate_piece_tactical_bonus(square, tactical_patterns)
        dynamic_adjustments += tactical_bonus
        
        # 4. Game phase adjustments
        phase_adjustment = self._calculate_game_phase_adjustment(board, square, piece_type, for_white)
        dynamic_adjustments += phase_adjustment
        
        # Track significant adjustments for profiling
        if abs(dynamic_adjustments) > 10:
            self.dynamic_adjustments += 1
            
        return base_value + dynamic_adjustments
    
    def _calculate_mobility_bonus(self, board: chess.Board, square: int, piece_type: int, for_white: bool) -> float:
        """Calculate mobility bonus based on piece movement options"""
        if piece_type not in self.MOBILITY_BONUS:
            return 0.0
            
        # Count legal moves from this square
        mobility = len([move for move in board.legal_moves if move.from_square == square])
        
        # Calculate bonus/penalty based on mobility vs expectations
        base_mobility = self.BASE_MOBILITY.get(piece_type, 0)
        mobility_diff = mobility - base_mobility
        
        # Apply mobility bonus/penalty
        mobility_value = mobility_diff * self.MOBILITY_BONUS[piece_type]
        
        # Special case: severely restricted pieces get heavy penalty
        if mobility <= 1 and piece_type != chess.PAWN:
            mobility_value += self.TRAPPED_PENALTY
            
        return mobility_value
    
    def _calculate_positional_bonus(self, board: chess.Board, square: int, piece_type: int, for_white: bool) -> float:
        """Calculate positional bonuses for piece placement"""
        bonus = 0.0
        rank, file = divmod(square, 8)
        
        # Center control bonus
        if self._is_center_square(square):
            bonus += self.CENTER_BONUS
        elif self._is_extended_center_square(square):
            bonus += self.CENTER_BONUS * 0.5
            
        # Piece-specific positional bonuses
        if piece_type == chess.KNIGHT:
            # Knights love outposts and central squares
            if self._is_knight_outpost(board, square, for_white):
                bonus += self.OUTPOST_BONUS
                
        elif piece_type == chess.BISHOP:
            # Bishops prefer long diagonals
            if self._is_long_diagonal(square):
                bonus += 10
            # Bishop pair bonus handled elsewhere
            
        elif piece_type == chess.ROOK:
            # Rooks prefer open files and 7th rank
            if self._is_open_file(board, square):
                bonus += 15
            elif self._is_semi_open_file(board, square, for_white):
                bonus += 8
                
            if self._is_seventh_rank(square, for_white):
                bonus += 20
                
        elif piece_type == chess.QUEEN:
            # Queens prefer central outposts but not too early
            material_count = self._count_total_material(board)
            if material_count > 2500:  # Opening/early middlegame
                # Penalize early queen development
                if rank in [3, 4, 5] if for_white else [2, 3, 4]:
                    bonus -= 15
            else:
                # Middlegame/endgame - centralize queen
                if self._is_center_square(square):
                    bonus += 25
                    
        return bonus
    
    def _calculate_piece_tactical_bonus(self, square: int, tactical_patterns: List[TacticalPattern]) -> float:
        """Calculate bonus for piece involvement in tactical patterns"""
        bonus = 0.0
        
        # Check if this piece is involved in any tactical patterns
        for pattern in tactical_patterns:
            if square == pattern.attacker_square:
                # This piece creates a tactical pattern
                bonus += pattern.tactical_value * 0.3  # 30% of pattern value
            elif square in pattern.target_squares:
                # This piece is targeted by a tactical pattern (defensive consideration)
                bonus -= pattern.tactical_value * 0.1  # Small penalty for being attacked
                
        return bonus
    
    def _calculate_game_phase_adjustment(self, board: chess.Board, square: int, piece_type: int, for_white: bool) -> float:
        """Adjust piece values based on game phase"""
        material_count = self._count_total_material(board)
        adjustment = 0.0
        
        if material_count > 2500:
            # Opening phase - prioritize development
            if piece_type in [chess.KNIGHT, chess.BISHOP]:
                if self._is_developed_square(square, piece_type, for_white):
                    adjustment += 12
                else:
                    adjustment -= 8  # Penalty for undeveloped pieces
                    
        elif material_count < 1500:
            # Endgame phase - kings and pawns more valuable
            if piece_type == chess.PAWN:
                # Passed pawns much more valuable
                if self._is_passed_pawn(board, square, for_white):
                    rank = square // 8
                    promotion_distance = (7 - rank) if for_white else rank
                    adjustment += (7 - promotion_distance) * 15
                    
            elif piece_type == chess.KING:
                # Active king bonus in endgame
                mobility = len([move for move in board.legal_moves if move.from_square == square])
                adjustment += mobility * 3
                
        return adjustment
    
    def _calculate_tactical_bonus(self, tactical_patterns: List[TacticalPattern]) -> float:
        """Calculate overall tactical bonus from detected patterns"""
        total_bonus = 0.0
        
        for pattern in tactical_patterns:
            # Weight by forcing level and type
            pattern_value = pattern.tactical_value
            forcing_multiplier = pattern.forcing_level / 4.0
            
            # Adjust by pattern type importance
            if pattern.pattern_type in ['fork', 'skewer']:
                pattern_value *= 1.2  # These patterns are especially valuable
            elif pattern.pattern_type == 'pin':
                pattern_value *= 1.1
                
            total_bonus += pattern_value * forcing_multiplier
            
        return total_bonus
    
    def _calculate_position_complexity_bonus(self, board: chess.Board, for_white: bool) -> float:
        """
        Calculate Tal's "position complexity" bonus
        In complex positions, favor tactical activity over material safety
        """
        complexity_factors = 0
        
        # Factor 1: Number of pieces in contact (attacking/defending)
        contact_count = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == for_white:
                # Count attacks on enemy pieces and defenses of our pieces
                attackers = board.attackers(not for_white, square)
                defenders = board.attackers(for_white, square)
                contact_count += len(attackers) + len(defenders)
                
        if contact_count > 12:
            complexity_factors += 1
            
        # Factor 2: Multiple piece types on central squares
        center_pieces = 0
        for square in [chess.D4, chess.D5, chess.E4, chess.E5]:
            if board.piece_at(square):
                center_pieces += 1
        if center_pieces >= 2:
            complexity_factors += 1
            
        # Factor 3: Kings not castled (tactical potential)
        if not board.has_castling_rights(chess.WHITE) and not board.has_castling_rights(chess.BLACK):
            complexity_factors += 1
            
        # In complex positions, add bonus for having tactical threats
        complexity_bonus = complexity_factors * 8  # +8cp per complexity factor
        
        return complexity_bonus
    
    # Helper methods for positional evaluation
    def _is_center_square(self, square: int) -> bool:
        """Check if square is in the center (d4, d5, e4, e5)"""
        return square in [chess.D4, chess.D5, chess.E4, chess.E5]
    
    def _is_extended_center_square(self, square: int) -> bool:
        """Check if square is in extended center"""
        rank, file = divmod(square, 8)
        return 2 <= rank <= 5 and 2 <= file <= 5
    
    def _is_knight_outpost(self, board: chess.Board, square: int, for_white: bool) -> bool:
        """Check if knight is on a strong outpost"""
        # Basic implementation - knight on strong square not attackable by enemy pawns
        rank, file = divmod(square, 8)
        
        # Check for enemy pawns that could attack this square
        enemy_pawn_direction = 1 if for_white else -1
        enemy_pawn_ranks = [rank - enemy_pawn_direction]
        
        for enemy_rank in enemy_pawn_ranks:
            if 0 <= enemy_rank < 8:
                for enemy_file in [file - 1, file + 1]:
                    if 0 <= enemy_file < 8:
                        enemy_square = enemy_rank * 8 + enemy_file
                        piece = board.piece_at(enemy_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color != for_white:
                            return False  # Enemy pawn can attack
                            
        return True
    
    def _is_long_diagonal(self, square: int) -> bool:
        """Check if square is on a long diagonal"""
        return square in [chess.A1, chess.B2, chess.C3, chess.D4, chess.E5, chess.F6, chess.G7, chess.H8,
                         chess.A8, chess.B7, chess.C6, chess.D5, chess.E4, chess.F3, chess.G2, chess.H1]
    
    def _is_open_file(self, board: chess.Board, square: int) -> bool:
        """Check if file is open (no pawns)"""
        file = square % 8
        for rank in range(8):
            check_square = rank * 8 + file
            piece = board.piece_at(check_square)
            if piece and piece.piece_type == chess.PAWN:
                return False
        return True
    
    def _is_semi_open_file(self, board: chess.Board, square: int, for_white: bool) -> bool:
        """Check if file is semi-open (no friendly pawns)"""
        file = square % 8
        for rank in range(8):
            check_square = rank * 8 + file
            piece = board.piece_at(check_square)
            if piece and piece.piece_type == chess.PAWN and piece.color == for_white:
                return False
        return True
    
    def _is_seventh_rank(self, square: int, for_white: bool) -> bool:
        """Check if piece is on 7th rank (relative to player)"""
        rank = square // 8
        return rank == 6 if for_white else rank == 1
    
    def _is_developed_square(self, square: int, piece_type: int, for_white: bool) -> bool:
        """Check if piece is on a developed square"""
        rank = square // 8
        
        if piece_type == chess.KNIGHT:
            starting_squares = [chess.B1, chess.G1] if for_white else [chess.B8, chess.G8]
            return square not in starting_squares
        elif piece_type == chess.BISHOP:
            starting_squares = [chess.C1, chess.F1] if for_white else [chess.C8, chess.F8]
            return square not in starting_squares
            
        return True
    
    def _is_passed_pawn(self, board: chess.Board, square: int, for_white: bool) -> bool:
        """Check if pawn is passed"""
        rank, file = divmod(square, 8)
        direction = 1 if for_white else -1
        
        # Check if any enemy pawns block this pawn's path
        for check_rank in range(rank + direction, 8 if for_white else -1, direction):
            for check_file in [file - 1, file, file + 1]:
                if 0 <= check_file < 8:
                    check_square = check_rank * 8 + check_file
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color != for_white:
                        return False
                        
        return True
    
    def _count_total_material(self, board: chess.Board) -> int:
        """Count total material on board"""
        total = 0
        for piece_type, value in self.BASE_VALUES.items():
            if piece_type != chess.KING:
                total += len(board.pieces(piece_type, chess.WHITE)) * value
                total += len(board.pieces(piece_type, chess.BLACK)) * value
        return total
    
    def get_profiling_stats(self) -> Dict[str, Union[int, float]]:
        """Get profiling statistics"""
        return {
            'evaluations': self.evaluation_count,
            'dynamic_adjustments': self.dynamic_adjustments,
            'adjustment_rate': self.dynamic_adjustments / max(self.evaluation_count, 1)
        }