# v7p3r_score.py

"""V7P3R Scoring Module
This module handles position evaluation using material count, material scores,
and piece square tables.
"""

import chess
from v7p3r_config import v7p3rConfig
from v7p3r_mvv_lva import v7p3rMVVLVA

class v7p3rScore:
    """Class for position evaluation in the V7P3R chess engine."""
    
    def __init__(self, pst):
        """Initialize scoring module.
        
        Args:
            pst: Piece square table module for position evaluation
        """
        self.config = v7p3rConfig()
        self.engine_config = self.config.get_engine_config()
        self.mvv_lva = v7p3rMVVLVA()  # For piece values
        self.pst = pst  # For piece square tables
        
        # Get weights from config
        self.material_count_weight = self.engine_config.get('material_count_weight', 10)
        self.material_score_weight = self.engine_config.get('material_score_weight', 10)
        self.pst_weight = self.engine_config.get('piece_square_tables_weight', 10)

    def evaluate_material_count(self, board: chess.Board) -> float:
        """Evaluate material count difference between sides.
        
        Args:
            board: Current board position
            
        Returns:
            float: Material count score (positive favors white)
        """
        score = 0.0
        
        # Count pieces using MVV-LVA's piece values
        for piece_type in chess.PIECE_TYPES:
            value = self.mvv_lva.get_piece_value(piece_type)
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            score += (white_count - black_count) * value
            
        return score * self.material_count_weight / 100

    def evaluate_material_score(self, board: chess.Board) -> float:
        """Evaluate material score considering piece values and basic positioning.
        
        Args:
            board: Current board position
            
        Returns:
            float: Material score (positive favors white)
        """
        score = 0.0
        
        # Evaluate each piece using MVV-LVA values
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.mvv_lva.get_piece_value(piece.piece_type)
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
                    
        return score * self.material_score_weight / 100

    def evaluate_pst(self, board: chess.Board) -> float:
        """Evaluate position using piece square tables.
        
        Args:
            board: Current board position
            
        Returns:
            float: Piece square table score (positive favors white)
        """
        score = 0.0
        
        # Let PST module handle the evaluation
        score = self.pst.evaluate_position(board)
        
        return score * self.pst_weight / 100

    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate the current position using all enabled scoring components.
        
        Args:
            board: Current board position
            
        Returns:
            float: Position score (positive favors white)
        """
        score = 0.0
        
        # Material count (if enabled)
        if self.engine_config.get('use_material_count', True):
            score += self.evaluate_material_count(board)
        
        # Material score (if enabled)
        if self.engine_config.get('use_material_score', True):
            score += self.evaluate_material_score(board)
        
        # Piece square tables (if enabled)
        if self.engine_config.get('use_piece_square_tables', True):
            score += self.evaluate_pst(board)
            
        return score