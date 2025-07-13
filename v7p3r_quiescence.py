# v7p3r_quiescence.py

"""V7P3R Quiescence Search Module
This module provides basic quiescence search to prevent horizon effect.
"""

import chess
from v7p3r_config import v7p3rConfig
from v7p3r_ordering import v7p3rOrdering

class v7p3rQuiescence:
    """Class for quiescence search in the V7P3R chess engine."""
    
    def __init__(self, scoring):
        """Initialize quiescence search.
        
        Args:
            scoring: Scoring module for position evaluation
        """
        self.scoring = scoring
        self.ordering = v7p3rOrdering()
        
        # Load config
        config = v7p3rConfig()
        self.engine_config = config.get_engine_config()
        
        # Search parameters
        self.max_depth = 4  # Reasonable depth for quiescence
        self.delta = 900  # Queen value for delta pruning
        
    def get_capture_moves(self, board: chess.Board) -> list[chess.Move]:
        """Get all capturing moves in a position.
        
        Args:
            board: Current board position
            
        Returns:
            list[chess.Move]: List of capturing moves
        """
        captures = []
        
        # Only consider captures for quiescence
        for move in board.legal_moves:
            if board.is_capture(move):
                captures.append(move)
                
        return captures
        
    def search(self, board: chess.Board, alpha: float, beta: float, depth: int = 0) -> float:
        """Perform quiescence search on a position.
        
        Args:
            board: Current board position
            alpha: Alpha bound
            beta: Beta bound
            depth: Current depth (for depth limiting)
            
        Returns:
            float: Position score after quiescence search
        """
        # Base case - evaluate if too deep
        if depth >= self.max_depth:
            return self.scoring.evaluate_position(board)
        
        # Get stand-pat score
        stand_pat = self.scoring.evaluate_position(board)
        
        # Beta cutoff
        if stand_pat >= beta:
            return beta
            
        # Delta pruning - if even a queen can't raise alpha
        if stand_pat < alpha - self.delta:
            return alpha
            
        # Update alpha if standing pat is better
        if stand_pat > alpha:
            alpha = stand_pat
            
        # Get capture moves
        captures = self.get_capture_moves(board)
        
        # No captures - position is quiet
        if not captures:
            return stand_pat
            
        # Order captures by MVV-LVA if enabled
        if self.engine_config.get('use_mvv_lva', True):
            captures = self.ordering.order_moves(captures, board)
            
        # Search captures
        for move in captures:
            # Make move
            board.push(move)
            
            # Recursively search
            score = -self.search(board, -beta, -alpha, depth + 1)
            
            # Unmake move
            board.pop()
            
            # Beta cutoff
            if score >= beta:
                return beta
                
            # Update alpha
            if score > alpha:
                alpha = score
                
        return alpha
