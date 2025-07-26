# v7p3r_search.py

"""V7P3R Search Module
This module implements minimax search with alpha-beta pruning and quiescence.
"""

import chess
from typing import Optional, Tuple
from v7p3r_config import v7p3rConfig
from v7p3r_ordering import v7p3rOrdering

class v7p3rSearch:
    """Class for search functionality in the V7P3R chess engine."""
    
    def __init__(self, scoring, quiescence):
        """Initialize search module.
        
        Args:
            scoring: Scoring module for position evaluation
            quiescence: Quiescence search module
        """
        # Load config
        self.config = v7p3rConfig()
        self.engine_config = self.config.get_engine_config()
        
        # Required modules
        self.scoring = scoring
        self.quiescence = quiescence
        self.ordering = v7p3rOrdering()
        
        # Search parameters
        self.depth = self.engine_config.get('depth', 6)
        self.max_ordered_moves = self.engine_config.get('max_ordered_moves', 10)
        
        # Track best move/score
        self.best_move = None
        self.best_score = float('-inf')
        
    def search(self, board: chess.Board) -> Optional[chess.Move]:
        """Find the best move in the current position.
        
        Args:
            board: Current board position
            
        Returns:
            Optional[chess.Move]: Best move found or None
        """
        # Reset tracking
        self.best_move = None
        self.best_score = float('-inf')
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        # Order moves
        if self.engine_config.get('use_move_ordering', True):
            ordered = self.ordering.order_moves(legal_moves, board, self.max_ordered_moves)
        else:
            ordered = legal_moves
            
        # Search each move
        alpha = float('-inf')
        beta = float('inf')
        is_white = board.turn == chess.WHITE
        
        for move in ordered:
            # Make move
            board.push(move)
            
            # Search position
            if self.engine_config.get('use_quiescence', True):
                score = -self._alphabeta_quiescence(board, self.depth - 1, -beta, -alpha, not is_white)
            else:
                score = -self._alphabeta(board, self.depth - 1, -beta, -alpha, not is_white)
            
            # Unmake move
            board.pop()
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_move = move
                alpha = max(alpha, score)
                
            # Prune if possible
            if alpha >= beta:
                break
                
        return self.best_move
        
    def _alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float, 
                   is_white: bool) -> float:
        """Alpha-beta minimax search.
        
        Args:
            board: Current board position
            depth: Remaining depth to search
            alpha: Alpha bound
            beta: Beta bound
            is_white: Whether white is to move
            
        Returns:
            float: Best score found
        """
        # Base case
        if depth == 0:
            return self.scoring.evaluate_position(board)
            
        # Get and order moves
        moves = list(board.legal_moves)
        if self.engine_config.get('use_move_ordering', True):
            moves = self.ordering.order_moves(moves, board, self.max_ordered_moves)
            
        # Search moves
        best_score = float('-inf')
        
        for move in moves:
            # Make move
            board.push(move)
            
            # Search
            score = -self._alphabeta(board, depth - 1, -beta, -alpha, not is_white)
            
            # Unmake move
            board.pop()
            
            # Update best
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            
            # Prune if possible
            if alpha >= beta:
                break
                
        return best_score
        
    def _alphabeta_quiescence(self, board: chess.Board, depth: int, alpha: float, beta: float,
                             is_white: bool) -> float:
        """Alpha-beta minimax with quiescence search.
        
        Args:
            board: Current board position
            depth: Remaining depth to search
            alpha: Alpha bound
            beta: Beta bound
            is_white: Whether white is to move
            
        Returns:
            float: Best score found
        """
        # Base case - evaluate or go to quiescence
        if depth == 0:
            return self.quiescence.search(board, alpha, beta)
            
        # Get and order moves
        moves = list(board.legal_moves)
        if self.engine_config.get('use_move_ordering', True):
            moves = self.ordering.order_moves(moves, board, self.max_ordered_moves)
            
        # Search moves
        best_score = float('-inf')
        
        for move in moves:
            # Make move
            board.push(move)
            
            # Search
            score = -self._alphabeta_quiescence(board, depth - 1, -beta, -alpha, not is_white)
            
            # Unmake move
            board.pop()
            
            # Update best
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            
            # Prune if possible
            if alpha >= beta:
                break
                
        return best_score
