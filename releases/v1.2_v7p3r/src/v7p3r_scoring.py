# v7p3r_scoring.py

"""V7P3R Static Evaluation
Lightweight static evaluation system for V7P3R Chess Engine v1.2.
Combines material, positional, and strategic evaluation factors.
"""

import chess
from typing import Dict, Any

from v7p3r_pst import V7P3RPST
from v7p3r_config import V7P3RConfig


class V7P3RScoring:
    """Static position evaluation system."""
    
    def __init__(self, piece_values: Dict[int, float], pst: V7P3RPST, config: V7P3RConfig):
        """Initialize scoring system.
        
        Args:
            piece_values: Material values for pieces
            pst: Piece-square table evaluator
            config: Engine configuration
        """
        self.piece_values = piece_values
        self.pst = pst
        self.config = config
        
        # Load evaluation weights using consistent config approach
        self.material_weight = config.get_setting('evaluation_config', 'material_weight', 1.0)
        self.positional_weight = config.get_setting('evaluation_config', 'positional_weight', 0.3)
        self.mobility_weight = config.get_setting('evaluation_config', 'mobility_weight', 0.1)
        self.king_safety_weight = config.get_setting('evaluation_config', 'king_safety_weight', 0.2)
        self.pawn_structure_weight = config.get_setting('evaluation_config', 'pawn_structure_weight', 0.15)
    
    def evaluate_board(self, board: chess.Board, endgame_factor: float = 0.0) -> float:
        """Evaluate position from white's perspective.
        
        Args:
            board: Current board position
            endgame_factor: 0.0 (middlegame) to 1.0 (endgame)
            
        Returns:
            Evaluation score (positive favors white)
        """
        if board.is_checkmate():
            return -9999.0 if board.turn == chess.WHITE else 9999.0
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        
        score = 0.0
        
        # Material evaluation
        score += self.material_weight * self._evaluate_material(board)
        
        # Positional evaluation (piece-square tables)
        score += self.positional_weight * self.pst.evaluate_position(board, endgame_factor)
        
        # Mobility evaluation
        score += self.mobility_weight * self._evaluate_mobility(board)
        
        # King safety evaluation
        score += self.king_safety_weight * self._evaluate_king_safety(board, endgame_factor)
        
        # Pawn structure evaluation
        score += self.pawn_structure_weight * self._evaluate_pawn_structure(board)
        
        return score
    
    def _evaluate_material(self, board: chess.Board) -> float:
        """Evaluate material balance."""
        white_material = 0.0
        black_material = 0.0
        
        for piece_type, value in self.piece_values.items():
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            white_material += white_count * value
            black_material += black_count * value
        
        return white_material - black_material
    
    def _evaluate_mobility(self, board: chess.Board) -> float:
        """Evaluate piece mobility (legal moves available)."""
        original_turn = board.turn
        
        # Count white mobility
        board.turn = chess.WHITE
        white_mobility = len(list(board.legal_moves))
        
        # Count black mobility
        board.turn = chess.BLACK
        black_mobility = len(list(board.legal_moves))
        
        # Restore original turn
        board.turn = original_turn
        
        # Normalize mobility score
        mobility_diff = white_mobility - black_mobility
        return mobility_diff / 50.0  # Scale to reasonable range
    
    def _evaluate_king_safety(self, board: chess.Board, endgame_factor: float) -> float:
        """Evaluate king safety (more important in middlegame)."""
        if endgame_factor > 0.7:  # In endgame, king safety is less critical
            return 0.0
        
        score = 0.0
        
        # Check if kings have castled
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        if white_king_square and black_king_square:
            # Penalty for king in center during middlegame
            white_king_file = chess.square_file(white_king_square)
            black_king_file = chess.square_file(black_king_square)
            
            # Penalty for king on central files (d, e)
            if white_king_file in [3, 4]:  # d or e file
                score -= 0.5
            if black_king_file in [3, 4]:  # d or e file
                score += 0.5
            
            # Check for pawn shield (pawns in front of king)
            score += self._evaluate_pawn_shield(board, chess.WHITE, white_king_square)
            score -= self._evaluate_pawn_shield(board, chess.BLACK, black_king_square)
        
        return score * (1.0 - endgame_factor)  # Less important in endgame
    
    def _evaluate_pawn_shield(self, board: chess.Board, color: chess.Color, king_square: int) -> float:
        """Evaluate pawn shield in front of king."""
        shield_value = 0.0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Check files around king (king file and adjacent files)
        for file_offset in [-1, 0, 1]:
            check_file = king_file + file_offset
            if 0 <= check_file <= 7:  # Valid file
                # Look for pawns 1-2 ranks ahead of king
                direction = 1 if color == chess.WHITE else -1
                for rank_offset in [1, 2]:
                    check_rank = king_rank + (direction * rank_offset)
                    if 0 <= check_rank <= 7:  # Valid rank
                        check_square = chess.square(check_file, check_rank)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            shield_value += 0.1  # Small bonus per shield pawn
                            break  # Only count closest pawn on this file
        
        return shield_value
    
    def _evaluate_pawn_structure(self, board: chess.Board) -> float:
        """Evaluate pawn structure (passed pawns, doubled pawns, etc.)."""
        score = 0.0
        
        # Evaluate for both colors
        score += self._evaluate_color_pawn_structure(board, chess.WHITE)
        score -= self._evaluate_color_pawn_structure(board, chess.BLACK)
        
        return score
    
    def _evaluate_color_pawn_structure(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate pawn structure for one color."""
        score = 0.0
        pawns = list(board.pieces(chess.PAWN, color))
        
        if not pawns:
            return 0.0
        
        # Check for doubled pawns (penalty)
        pawn_files = [chess.square_file(pawn) for pawn in pawns]
        for file in range(8):
            count = pawn_files.count(file)
            if count > 1:
                score -= 0.2 * (count - 1)  # Penalty for doubled pawns
        
        # Check for passed pawns (bonus)
        for pawn_square in pawns:
            if self._is_passed_pawn(board, pawn_square, color):
                pawn_rank = chess.square_rank(pawn_square)
                # More valuable the closer to promotion
                if color == chess.WHITE:
                    advancement = pawn_rank  # 0-7, higher is better
                else:
                    advancement = 7 - pawn_rank  # 7-0, lower is better
                score += 0.1 + (advancement * 0.05)  # Bonus increases with advancement
        
        # Check for isolated pawns (penalty)
        for pawn_square in pawns:
            if self._is_isolated_pawn(board, pawn_square, color):
                score -= 0.15  # Penalty for isolated pawn
        
        return score
    
    def _is_passed_pawn(self, board: chess.Board, pawn_square: int, color: chess.Color) -> bool:
        """Check if pawn is passed (no enemy pawns can stop it)."""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        opponent_color = not color
        
        # Check if any opponent pawns can block this pawn's path
        if color == chess.WHITE:
            # Check ranks ahead of white pawn
            for rank in range(pawn_rank + 1, 8):
                for file in [pawn_file - 1, pawn_file, pawn_file + 1]:
                    if 0 <= file <= 7:
                        check_square = chess.square(file, rank)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == opponent_color:
                            return False
        else:
            # Check ranks ahead of black pawn
            for rank in range(pawn_rank - 1, -1, -1):
                for file in [pawn_file - 1, pawn_file, pawn_file + 1]:
                    if 0 <= file <= 7:
                        check_square = chess.square(file, rank)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == opponent_color:
                            return False
        
        return True
    
    def _is_isolated_pawn(self, board: chess.Board, pawn_square: int, color: chess.Color) -> bool:
        """Check if pawn is isolated (no friendly pawns on adjacent files)."""
        pawn_file = chess.square_file(pawn_square)
        
        # Check adjacent files for friendly pawns
        for file_offset in [-1, 1]:
            check_file = pawn_file + file_offset
            if 0 <= check_file <= 7:
                # Check entire file for friendly pawns
                for rank in range(8):
                    check_square = chess.square(check_file, rank)
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        return False  # Found friendly pawn on adjacent file
        
        return True  # No friendly pawns on adjacent files
