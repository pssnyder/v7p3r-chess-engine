# v7p3r_quiescence.py

"""v7p3r Quiescence Search Module
This module handles quiescence search to identify and evaluate loud positions.
It helps prevent the horizon effect by examining captures and checks until a quiet position is reached.
"""

import chess
from typing import Optional, Dict, List, Tuple
from v7p3r_config import v7p3rConfig
from v7p3r_mvv_lva import v7p3rMVVLVA

class v7p3rQuiescence:
    def __init__(self, scoring_calculator, move_organizer, rules_manager, engine_config: Optional[Dict] = None):
        """Initialize the quiescence search module"""
        self.scoring_calculator = scoring_calculator
        self.move_organizer = move_organizer
        self.rules_manager = rules_manager
        self.config = engine_config or v7p3rConfig().get_engine_config()
        self.mvv_lva = v7p3rMVVLVA(rules_manager)
        
        # Configuration
        self.use_quiescence = self.config.get('use_quiescence', True)
        self.max_quiescence_depth = 4  # Limit quiescence search depth
        
        # Statistics
        self.nodes_searched = 0
        
    def is_quiet_position(self, board: chess.Board) -> bool:
        """Determine if a position is quiet (no captures or checks available)"""
        if board.is_check():
            return False
            
        # Look for any captures
        for move in board.legal_moves:
            if board.is_capture(move):
                return False
                
        return True
        
    def get_loud_moves(self, board: chess.Board) -> List[chess.Move]:
        """Get a list of capturing moves and checks"""
        loud_moves = []
        for move in board.legal_moves:
            # Include captures and checks
            if board.is_capture(move):
                loud_moves.append(move)
            else:
                # Test for check
                board.push(move)
                gives_check = board.is_check()
                board.pop()
                if gives_check:
                    loud_moves.append(move)
                    
        return loud_moves
        
    def quiescence_search(self, board: chess.Board, alpha: float, beta: float, color: chess.Color, depth: int = 0) -> float:
        """Perform quiescence search to evaluate tactical sequences"""
        if not self.use_quiescence:
            return self.scoring_calculator.calculate_score(board, color)
            
        # Base case: maximum depth reached or quiet position
        if depth >= self.max_quiescence_depth or self.is_quiet_position(board):
            return self.scoring_calculator.calculate_score(board, color)
            
        # Stand pat - evaluate current position
        stand_pat = self.scoring_calculator.calculate_score(board, color)
        
        # Beta cutoff
        if stand_pat >= beta:
            return beta
            
        # Update alpha if standing pat is better
        alpha = max(alpha, stand_pat)
        
        # Get capture moves and checks
        loud_moves = self.get_loud_moves(board)
        
        # Order moves by MVV-LVA if available
        if self.config.get('use_mvv_lva', True):
            loud_moves = self.move_organizer.order_moves(board, loud_moves, [])
        
        # Search loud moves
        for move in loud_moves:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, not color, depth + 1)
            board.pop()
            
            if score >= beta:
                return beta
            alpha = max(alpha, score)
            
        return alpha

    def search(self, board: chess.Board, alpha: float = float('-inf'), beta: float = float('inf'), depth: Optional[int] = None) -> float:
        """Perform quiescence search to evaluate tactical sequences
        
        Args:
            board: Current chess position
            alpha: Alpha bound for alpha-beta pruning
            beta: Beta bound for alpha-beta pruning
            depth: Maximum depth for quiescence search (uses self.max_quiescence_depth if None)
            
        Returns:
            float: Score for the position after quiescence search
        """
        # Count this node
        self.nodes_searched += 1
        
        # Use class's max depth if none provided
        if depth is None:
            depth = self.max_quiescence_depth
            
        # Stop if we've gone too deep
        if depth <= 0:
            return self.scoring_calculator.evaluate_position(board)
        # 1. Stand-pat score
        stand_pat = self.scoring_calculator.evaluate_position(board)
        
        # Fail-high check
        if stand_pat >= beta:
            return beta
            
        # Delta pruning - if even the best possible capture won't improve alpha
        delta = 900  # Value of a queen
        if stand_pat < alpha - delta:
            return alpha
            
        # Update alpha if stand-pat is better
        if stand_pat > alpha:
            alpha = stand_pat

        # 2. Generate capture moves and checks
        capture_moves = []
        for move in board.legal_moves:
            if board.is_capture(move) or board.gives_check(move):
                capture_moves.append(move)
                
        # No captures/checks available - return stand-pat score
        if not capture_moves:
            return stand_pat

        # 3. Order moves by MVV-LVA if available
        if self.mvv_lva and len(capture_moves) > 1:
            capture_moves.sort(key=lambda m: self.mvv_lva.calculate_mvv_lva_score(m, board), reverse=True)

        # 4. Search capture moves
        for move in capture_moves:
            board.push(move)
            score = -self.search(board, -beta, -alpha, depth - 1)  # Negamax framework with depth limit
            board.pop()
            
            if score >= beta:
                return beta
                
            if score > alpha:
                alpha = score

        return alpha
