# v7p3r_ordering.py

"""Move ordering for the V7P3R chess engine.
This module provides move ordering using MVV-LVA scores and basic promotion handling.
"""

import chess
from typing import List
from v7p3r_config import v7p3rConfig
from v7p3r_mvv_lva import v7p3rMVVLVA

class v7p3rOrdering:
    """Class for ordering moves in the V7P3R chess engine."""

    # Simple promotion bonus (less than a queen capture but significant)
    PROMOTION_BONUS = 800
    
    def __init__(self, mvv_lva: v7p3rMVVLVA | None = None):
        """Initialize move ordering with MVV-LVA scorer.
        
        Args:
            mvv_lva: Optional MVV-LVA scorer instance
        """
        self.mvv_lva = mvv_lva or v7p3rMVVLVA()

    def score_move(self, move: chess.Move, board: chess.Board) -> int:
        """Score a move using MVV-LVA for captures and basic promotion bonus.
        
        Args:
            move: The move to score
            board: Current board position
            
        Returns:
            int: Move score
        """
        score = 0
        
        # Use MVV-LVA module for capture scoring
        if board.is_capture(move):
            score += self.mvv_lva.score_capture(move, board)
            
        # Simple promotion bonus
        if move.promotion:
            score += self.PROMOTION_BONUS
            
        return score

    def order_moves(self, moves: List[chess.Move], board: chess.Board, max_moves: int = 0) -> List[chess.Move]:
        """Order moves by score, optionally limiting the number returned.
        
        Args:
            moves: List of moves to order
            board: Current board position
            max_moves: Maximum number of moves to return (0 for all)
            
        Returns:
            List[chess.Move]: Ordered list of moves
        """
        # Score all moves
        scored_moves = [(move, self.score_move(move, board)) for move in moves]
        
        # Sort by score, highest first
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        ordered = [move for move, _ in scored_moves]
        
        # Limit number of moves if requested
        if max_moves > 0:
            ordered = ordered[:max_moves]
            
        return ordered
