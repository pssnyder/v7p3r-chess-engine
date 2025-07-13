# v7p3r_engine.py

"""V7P3R Engine
This module coordinates the chess engine components for move generation.
"""

import chess
from typing import Optional
from v7p3r_search import v7p3rSearch
from v7p3r_score import v7p3rScore
from v7p3r_quiescence import v7p3rQuiescence
from v7p3r_pst import v7p3rPST
from v7p3r_config import v7p3rConfig
from v7p3r_rules import v7p3rRules

class v7p3rEngine:
    """Main engine class coordinating all components."""
    
    def __init__(self, engine_config=None):
        """Initialize engine with configuration.
        
        Args:
            engine_config: Optional engine configuration override
        """
        # Load configuration
        self.config = v7p3rConfig()
        self.engine_config = engine_config or self.config.get_engine_config()
        
        # Initialize components
        self.pst = v7p3rPST()  # Piece square tables
        self.rules = v7p3rRules()  # Game rules
        self.score = v7p3rScore(self.pst)  # Position evaluation
        self.quiescence = v7p3rQuiescence(self.score)  # Quiescence search
        self.search = v7p3rSearch(self.score, self.quiescence)  # Main search
        
        # Engine settings
        self.depth = self.engine_config.get('depth', 6)
        
        # State tracking
        self.nodes_searched = 0
        self.current_score = 0.0
        
    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get the best move in the current position.
        
        Args:
            board: Current board position
            
        Returns:
            Optional[chess.Move]: Best move found, or None if no moves available
        """
        # Get legal moves
        moves = list(board.legal_moves)
        if not moves:
            return None
            
        # Check for immediate checkmate
        if self.engine_config.get('use_checkmate_detection', True):
            for move in moves:
                board.push(move)
                is_mate = board.is_checkmate()
                board.pop()
                if is_mate:
                    return move
                    
        # Do full search
        best_move = self.search.search(board)
        if best_move:
            return best_move
            
        # Fallback to first legal move
        return moves[0]
        
    def make_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Make a move in the given position.
        
        Args:
            board: Current board position
            
        Returns:
            Optional[chess.Move]: Move made, or None if no moves available
        """
        move = self.get_move(board)
        if move:
            board.push(move)
        return move
