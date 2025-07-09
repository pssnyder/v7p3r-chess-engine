# v7p3r_ordering.py
""" Move ordering for the V7P3R chess engine.
This module provides functionality to order moves based on their potential effectiveness"""
import os
import sys
import chess
from v7p3r_config import v7p3rConfig
from v7p3r_mvv_lva import v7p3rMVVLVA

# Ensure the parent directory is in sys.path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class v7p3rOrdering:
    """Class for move ordering in the V7P3R chess engine."""
    def __init__(self, scoring_calculator):
        # Engine Configuration
        self.config_manager = v7p3rConfig()
        self.engine_config = self.config_manager.get_engine_config()  # Ensure it's always a dictionary
        
        # Required Engine Modules
        self.scoring_calculator = scoring_calculator
        self.mvv_lva = v7p3rMVVLVA(scoring_calculator.rules_manager)
        
        # Move Ordering Settings
        self.move_ordering_enabled = self.engine_config.get('move_ordering_enabled', True)
        self.max_ordered_moves = self.engine_config.get('max_ordered_moves', 10)  # Default to 10 moves if not set

    def order_moves(self, board: chess.Board, moves, depth: int = 0, cutoff: int = 0) -> list:
        """Order moves for better alpha-beta pruning efficiency"""
        # Safety check to prevent empty move list issues
        if not moves:
            return []
            
        move_scores = []
        for move in moves:
            if not board.is_legal(move):
                continue
            
            score = self._order_move_score(board, move)
            move_scores.append((move, score))
            if score > 10000: # shortcut to instantly return a move that scores over 10k
                return [move]

        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get max_ordered_moves setting from engine config if cutoff is not specified
        max_ordered_moves = cutoff if cutoff > 0 else self.max_ordered_moves
        if max_ordered_moves > 0 and len(move_scores) > max_ordered_moves:
            move_scores = move_scores[:max_ordered_moves]
        
        return [move for move, _ in move_scores]

    def _order_move_score(self, board: chess.Board, move: chess.Move) -> float:
        """Calculate a score for a move for ordering purposes.
        Focus on checkmates, checks, captures, and promotions."""
        score = 0.0

        temp_board = board.copy()
        temp_board.push(move)
        if temp_board.is_checkmate():
            temp_board.pop()
            return 99999.0 # Score over 10K for checkmate moves
        
        if temp_board.is_check(): # Check after move is made
            score += 9999.0

        temp_board.pop()
        if board.is_capture(move):
            # Use MVV-LVA for capture scoring
            mvv_lva_score = self.mvv_lva.calculate_mvv_lva_score(move, board)
            score += 900.0 + mvv_lva_score

        if move.promotion:
            score += 90.0
            if move.promotion == chess.QUEEN:
                score += 9.0
        return score
