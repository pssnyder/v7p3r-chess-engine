# v7p3r_ordering.py
""" Move ordering for the V7P3R chess engine.
This module provides functionality to order moves based on their potential effectiveness"""
import os
import sys
import chess
from v7p3r_config import v7p3rConfig
from v7p3r_debug import v7p3rLogger, v7p3rUtilities

# Ensure the parent directory is in sys.path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup centralized logging for this module
v7p3r_ordering_logger = v7p3rLogger.setup_logger("v7p3r_ordering")

class v7p3rOrdering:
    """Class for move ordering in the V7P3R chess engine."""
    def __init__(self, scoring_calculator):
        # Engine Configuration
        self.config_manager = v7p3rConfig()
        self.engine_config = self.config_manager.get_engine_config()  # Ensure it's always a dictionary
        
        # Required Engine Modules
        self.scoring_calculator = scoring_calculator
        
        # Logging Setup
        self.logger = v7p3r_ordering_logger
        self.monitoring_enabled = self.engine_config.get('monitoring_enabled', True)
        self.verbose_output_enabled = self.engine_config.get('verbose_output', False)
        
        # Move Ordering Settings
        self.move_ordering_enabled = self.engine_config.get('move_ordering_enabled', True)
        self.max_moves = self.engine_config.get('max_moves', 10)  # Default to 10 moves if not set

    def order_moves(self, board: chess.Board, moves, depth: int = 0, cutoff: int = 0) -> list:
        """Order moves for better alpha-beta pruning efficiency"""
        # Safety check to prevent empty move list issues
        if not moves:
            return []
            
        move_scores = []
        for move in moves:
            if not board.is_legal(move):
                if self.monitoring_enabled and self.logger:
                    self.logger.error(f"[Error] Illegal move passed to order_moves: {move} | FEN: {board.fen()}")
                continue
            
            score = self._order_move_score(board, move)
            move_scores.append((move, score))
            if score > 1000000: # shortcut to instantly return a move that scores over 1 Mil
                return [move]

        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get max_moves setting from engine config if cutoff is not specified
        max_moves = cutoff if cutoff > 0 else self.max_moves
        if max_moves > 0 and len(move_scores) > max_moves:
            original_count = len(move_scores)
            move_scores = move_scores[:max_moves]
            # Only log truncation for significant reductions to avoid excessive logging
            if original_count - max_moves > 5:
                log_msg = f"MOVE_TRUNCATION: Truncated move list from {original_count} to {max_moves} moves at depth {depth}"
                if self.monitoring_enabled and self.logger:
                    self.logger.info(log_msg)
        
        # Reduce logging frequency to improve performance
        if self.monitoring_enabled and self.verbose_output_enabled and self.logger:
            self.logger.info(f"Ordered moves at depth {depth}: {[f'{move} ({score:.2f})' for move, score in move_scores[:3]]}... (total: {len(move_scores)}) | FEN: {board.fen()}")
        
        return [move for move, _ in move_scores]

    def _order_move_score(self, board: chess.Board, move: chess.Move) -> float:
        """Calculate a score for a move for ordering purposes."""
        score = 0.0

        temp_board = board.copy()
        temp_board.push(move)
        if temp_board.is_checkmate():
            temp_board.pop()
            # Get checkmate move bonus from the current ruleset
            return self.scoring_calculator.ruleset.get('checkmate_order_modifier', 999999999.0)
        
        if temp_board.is_check(): # Check after move is made
            score += self.scoring_calculator.ruleset.get('check_order_modifier', 99999.0)

        temp_board.pop()
        if temp_board.is_capture(move):
            score += self.scoring_calculator.ruleset.get('capture_order_modifier', 5000.0)
            victim_type = temp_board.piece_type_at(move.to_square)
            aggressor_type = temp_board.piece_type_at(move.from_square)
            if victim_type and aggressor_type:
                score += (self.engine_config.get('piece_values', {}).get(victim_type, 0) * 10) - self.engine_config.get('piece_values', {}).get(aggressor_type, 0)

        if move.promotion:
            score += self.scoring_calculator.ruleset.get('promotion_order_modifier', 3000.0)
            if move.promotion == chess.QUEEN:
                score += self.engine_config.get('piece_values', {}).get(chess.QUEEN, 9.0) * 100 # Ensure piece_values is used
        return score
