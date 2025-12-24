#!/usr/bin/env python3
"""
V7P3R Position Context Calculator

Calculates position characteristics ONCE before search, persists through entire tree.

This module provides the foundation for modular evaluation by determining
what type of position we're in and what evaluation modules should be active.

Author: Pat Snyder
Created: 2025-12-23 (v18.2 Modular Evaluation System)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Set
import chess


class GamePhase(Enum):
    """Unified game phase classification (single source of truth)"""
    OPENING = "opening"                    # Move < 12, pieces ≥ 12
    MIDDLEGAME_COMPLEX = "middlegame_complex"  # Material 1300-2500cp, pieces 7-11
    MIDDLEGAME_SIMPLE = "middlegame_simple"    # Material 1300-2500cp, pieces 4-6
    ENDGAME_COMPLEX = "endgame_complex"        # Material < 1300cp, pieces 3-6
    ENDGAME_SIMPLE = "endgame_simple"          # Material < 800cp, pieces ≤ 2


class MaterialBalance(Enum):
    """Material imbalance classification"""
    EQUAL = "equal"                # |diff| < 100cp
    SLIGHT_ADVANTAGE = "slight"    # 100-300cp
    ADVANTAGE = "advantage"        # 300-500cp
    WINNING = "winning"            # 500-900cp
    CRUSHING = "crushing"          # > 900cp


class TacticalFlags(Enum):
    """Binary tactical indicators (hints for module selection)"""
    KING_EXPOSED = "king_exposed"              # King has ≤2 pawn shield
    PIECES_HANGING = "pieces_hanging"          # Undefended pieces exist (needs verification)
    CHECKS_AVAILABLE = "checks_available"      # Can give check (queen/rook near enemy king)
    PINS_PRESENT = "pins_present"              # Pin opportunities exist
    FORKS_PRESENT = "forks_present"            # Fork opportunities exist
    BACK_RANK_WEAK = "back_rank_weak"         # Back rank mate threat


@dataclass
class PositionContext:
    """
    Immutable position characteristics calculated once per search.
    
    This context is passed to ALL evaluation modules and persists
    through the entire search tree (not recalculated per node).
    
    Design Principle: Calculate expensive checks ONCE, use O(1) lookups everywhere else.
    """
    # Time management
    time_remaining: float        # Seconds left on clock
    time_per_move: float         # Allocated time for this move
    time_pressure: bool          # < 30 seconds remaining
    
    # Game phase (single source of truth)
    game_phase: GamePhase        # Authoritative phase classification
    move_number: int             # Full move count (1-based)
    
    # Material
    material_balance: MaterialBalance  # Who's winning materially
    material_diff_cp: int        # Centipawn difference (+ = we're winning)
    total_material: int          # Combined material on board
    
    # Piece inventory (for module activation)
    piece_types: Set[chess.PieceType]  # {PAWN, KNIGHT, BISHOP, ROOK, QUEEN}
    white_pieces: int            # Count of white pieces (excluding king)
    black_pieces: int            # Count of black pieces (excluding king)
    
    # Positional flags (quick checks for module relevance)
    queens_on_board: bool        # At least one queen present
    bishops_on_board: bool       # At least one bishop present
    opposite_bishops: bool       # Each side has 1+ bishops (bishop pair relevant)
    rooks_on_board: bool         # At least one rook present
    
    # Tactical indicators (hints, not full tactical analysis)
    tactical_flags: Set[TacticalFlags]  # Active tactical themes
    king_safety_critical: bool   # King exposure detected
    
    # Endgame specifics
    pawn_endgame: bool          # Only kings + pawns
    pure_piece_endgame: bool    # No pawns, only pieces
    theoretical_draw: bool       # Known drawn material (K vs K, K+B vs K, etc)
    
    # Search context
    depth_target: int           # Planned search depth based on time
    use_fast_profile: bool      # Force fast evaluation (time pressure)


class PositionContextCalculator:
    """
    Calculates position context once before search.
    
    Design Principles:
    - O(1) or O(64) complexity (single board scan, no move generation)
    - No recursive analysis
    - Cache-friendly (single object creation)
    - Fast enough to call every root search (~0.1ms target)
    """
    
    # Material values (standard)
    MATERIAL_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900
    }
    
    def calculate(self, board: chess.Board, time_remaining: float = 300.0, 
                  time_per_move: float = 5.0) -> PositionContext:
        """
        Main entry point: Calculate all position characteristics.
        
        Args:
            board: Current chess position
            time_remaining: Seconds left on clock
            time_per_move: Allocated time for this move
            
        Returns:
            PositionContext with all calculated fields
            
        Time Complexity: O(64) - single board scan
        Space Complexity: O(1) - fixed-size dataclass
        """
        # Material calculation (O(64))
        material_info = self._calculate_material(board)
        
        # Piece inventory (O(64))
        piece_info = self._calculate_piece_inventory(board)
        
        # Game phase (O(1) - uses material_info)
        game_phase = self._determine_game_phase(
            board, material_info, piece_info
        )
        
        # Tactical flags (O(64) - simple board scan, no move gen)
        tactical_flags = self._detect_tactical_flags(board, piece_info)
        
        # Time pressure detection (O(1))
        time_pressure = time_remaining < 30.0
        use_fast_profile = time_pressure or time_per_move < 5.0
        
        # Depth target based on time (O(1))
        depth_target = self._calculate_depth_target(
            time_per_move, game_phase, time_pressure
        )
        
        return PositionContext(
            # Time
            time_remaining=time_remaining,
            time_per_move=time_per_move,
            time_pressure=time_pressure,
            
            # Phase
            game_phase=game_phase,
            move_number=board.fullmove_number,
            
            # Material
            material_balance=material_info['balance'],
            material_diff_cp=material_info['diff_cp'],
            total_material=material_info['total'],
            
            # Pieces
            piece_types=piece_info['types'],
            white_pieces=piece_info['white_count'],
            black_pieces=piece_info['black_count'],
            
            # Flags
            queens_on_board=chess.QUEEN in piece_info['types'],
            bishops_on_board=chess.BISHOP in piece_info['types'],
            opposite_bishops=piece_info['opposite_bishops'],
            rooks_on_board=chess.ROOK in piece_info['types'],
            
            # Tactical
            tactical_flags=tactical_flags,
            king_safety_critical=TacticalFlags.KING_EXPOSED in tactical_flags,
            
            # Endgame
            pawn_endgame=piece_info['pawn_endgame'],
            pure_piece_endgame=piece_info['pure_piece_endgame'],
            theoretical_draw=material_info['theoretical_draw'],
            
            # Search
            depth_target=depth_target,
            use_fast_profile=use_fast_profile
        )
    
    def _calculate_material(self, board: chess.Board) -> dict:
        """
        Calculate material counts and balance (O(64))
        
        Returns:
            dict with 'diff_cp', 'total', 'balance', 'theoretical_draw'
        """
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                value = self.MATERIAL_VALUES[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        # Calculate from our perspective (positive = we're winning)
        diff_cp = white_material - black_material
        if not board.turn:  # Black to move
            diff_cp = -diff_cp
        
        # Determine balance category
        abs_diff = abs(diff_cp)
        if abs_diff < 100:
            balance = MaterialBalance.EQUAL
        elif abs_diff < 300:
            balance = MaterialBalance.SLIGHT_ADVANTAGE
        elif abs_diff < 500:
            balance = MaterialBalance.ADVANTAGE
        elif abs_diff < 900:
            balance = MaterialBalance.WINNING
        else:
            balance = MaterialBalance.CRUSHING
        
        # Theoretical draw detection
        total = white_material + black_material
        theoretical_draw = (
            total == 0 or  # K vs K
            total <= 330   # K+B vs K, K+N vs K, or K+B vs K+B (endgame tables)
        )
        
        return {
            'diff_cp': diff_cp,
            'total': total,
            'balance': balance,
            'theoretical_draw': theoretical_draw
        }
    
    def _calculate_piece_inventory(self, board: chess.Board) -> dict:
        """
        Count pieces and determine endgame types (O(64))
        
        Returns:
            dict with piece counts and flags
        """
        piece_types = set()
        white_count = 0
        black_count = 0
        white_bishops = 0
        black_bishops = 0
        has_pawns = False
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                piece_types.add(piece.piece_type)
                
                if piece.color == chess.WHITE:
                    white_count += 1
                    if piece.piece_type == chess.BISHOP:
                        white_bishops += 1
                else:
                    black_count += 1
                    if piece.piece_type == chess.BISHOP:
                        black_bishops += 1
                
                if piece.piece_type == chess.PAWN:
                    has_pawns = True
        
        return {
            'types': piece_types,
            'white_count': white_count,
            'black_count': black_count,
            'opposite_bishops': white_bishops > 0 and black_bishops > 0,
            'pawn_endgame': piece_types == {chess.PAWN},
            'pure_piece_endgame': len(piece_types) > 0 and not has_pawns
        }
    
    def _determine_game_phase(self, board: chess.Board, 
                              material_info: dict, piece_info: dict) -> GamePhase:
        """
        Unified game phase detection (single source of truth).
        
        Logic:
        1. Opening: move < 12 AND pieces ≥ 12
        2. Endgame: material < 1300cp OR (pieces ≤ 4 AND no queens)
        3. Middlegame: everything else
        4. Complex vs Simple: based on piece count
        
        This replaces all inconsistent thresholds across the codebase.
        """
        move_num = board.fullmove_number
        total_material = material_info['total']
        total_pieces = piece_info['white_count'] + piece_info['black_count']
        has_queens = chess.QUEEN in piece_info['types']
        
        # Opening
        if move_num < 12 and total_pieces >= 12:
            return GamePhase.OPENING
        
        # Endgame
        if total_material < 1300 or (total_pieces <= 4 and not has_queens):
            if total_pieces <= 2:
                return GamePhase.ENDGAME_SIMPLE
            else:
                return GamePhase.ENDGAME_COMPLEX
        
        # Middlegame
        if total_pieces <= 6:
            return GamePhase.MIDDLEGAME_SIMPLE
        else:
            return GamePhase.MIDDLEGAME_COMPLEX
    
    def _detect_tactical_flags(self, board: chess.Board, 
                               piece_info: dict) -> Set[TacticalFlags]:
        """
        Quick tactical flag detection (no move generation).
        
        Note: These are HINTS for evaluation selection, not full tactical analysis.
        Full tactical checks done by selected evaluation modules.
        """
        flags = set()
        
        # King exposure (simple pawn shield check)
        our_king = board.king(board.turn)
        if our_king is not None:
            pawn_shield_count = self._count_pawn_shield(board, our_king, board.turn)
            if pawn_shield_count <= 2:
                flags.add(TacticalFlags.KING_EXPOSED)
        
        # Checks available (heuristic: queen or rook near enemy king)
        if chess.QUEEN in piece_info['types'] or chess.ROOK in piece_info['types']:
            enemy_king = board.king(not board.turn)
            if enemy_king is not None:
                # Check if we have queen/rook (could potentially give check)
                flags.add(TacticalFlags.CHECKS_AVAILABLE)
        
        return flags
    
    def _count_pawn_shield(self, board: chess.Board, king_square: int, color: chess.Color) -> int:
        """
        Count pawns in front of king (pawn shield).
        
        Returns: Number of friendly pawns protecting king (0-3)
        """
        king_rank = chess.square_rank(king_square)
        king_file = chess.square_file(king_square)
        
        pawn_shield_count = 0
        
        if color == chess.WHITE and king_rank < 2:
            # Check squares in front of white king
            for file_offset in [-1, 0, 1]:
                check_file = king_file + file_offset
                if 0 <= check_file <= 7:
                    check_square = chess.square(check_file, king_rank + 1)
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                        pawn_shield_count += 1
        
        elif color == chess.BLACK and king_rank > 5:
            # Check squares in front of black king
            for file_offset in [-1, 0, 1]:
                check_file = king_file + file_offset
                if 0 <= check_file <= 7:
                    check_square = chess.square(check_file, king_rank - 1)
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                        pawn_shield_count += 1
        
        return pawn_shield_count
    
    def _calculate_depth_target(self, time_per_move: float, 
                                game_phase: GamePhase, 
                                time_pressure: bool) -> int:
        """
        Determine target search depth based on available time.
        
        Fast profiles can search deeper due to lower per-node cost.
        
        Args:
            time_per_move: Allocated time in seconds
            game_phase: Current game phase
            time_pressure: Whether in time pressure
            
        Returns:
            Target depth (4-8)
        """
        if time_pressure:
            return 4  # Emergency mode
        elif time_per_move < 5.0:
            return 5  # Blitz fast mode
        elif time_per_move < 15.0:
            return 6  # Blitz/rapid normal
        elif time_per_move < 60.0:
            return 7  # Rapid deep search
        else:
            return 8  # Long time control
