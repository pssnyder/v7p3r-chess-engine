# # v7p3r_engine/v7p3r_ordering.py
""" Move ordering for the V7P3R chess engine.
This module provides functionality to order moves based on their potential effectiveness"""
import os
import sys
import chess
import logging
import datetime

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base = getattr(sys, '_MEIPASS', None)
    if base:
        return os.path.join(base, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def get_log_file_path():
    # Optional timestamp for log file name
    timestamp = get_timestamp()
    log_dir = "logging"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"v7p3r_evaluation_engine.log")
v7p3r_engine_logger = logging.getLogger("v7p3r_evaluation_engine")
v7p3r_engine_logger.setLevel(logging.DEBUG)
_init_status = globals().get("_init_status", {})
if not _init_status.get("initialized", False):
    log_file_path = get_log_file_path()
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10*1024*1024,
        backupCount=3,
        delay=True
    )
    formatter = logging.Formatter(
        '%(asctime)s | %(funcName)-15s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    v7p3r_engine_logger.addHandler(file_handler)
    v7p3r_engine_logger.propagate = False
    _init_status["initialized"] = True
    # Store the log file path for later use (e.g., to match with PGN/config)
    _init_status["log_file_path"] = log_file_path

class v7p3rOrdering:
    """Class for move ordering in the V7P3R chess engine."""
    def __init__(self, engine_config: dict, scoring_calculator, logger: logging.Logger):
        self.engine_config = engine_config
        self.scoring_calculator = scoring_calculator
        self.logger = logger or logging.getLogger('v7p3r_engine_logger')
        self.monitoring_enabled = engine_config.get('monitoring_enabled', False)
        self.verbose_output_enabled = engine_config.get('verbose_output', False)

    def order_moves(self, board: chess.Board, moves, depth: int = 0) -> list:
        """Order moves for better alpha-beta pruning efficiency"""
        move_scores = []
        for move in moves:
            if not board.is_legal(move):
                if self.logger and self.monitoring_enabled:
                    self.logger.error(f"[Error] Illegal move passed to order_moves: {move} | FEN: {board.fen()}")
                continue
            
            score = self._order_move_score(board, move, depth)
            move_scores.append((move, score))

        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
            self.logger.info(f"Ordered moves at depth {depth}: {[f'{move} ({score:.2f})' for move, score in move_scores]} | FEN: {board.fen()}")
        
        return [move for move, _ in move_scores]

    def _order_move_score(self, board: chess.Board, move: chess.Move, depth: int = 0) -> float:
        """Calculate a score for a move for ordering purposes."""
        score = 0.0

        temp_board = board.copy()
        temp_board.push(move)
        if temp_board.is_checkmate():
            temp_board.pop()
            # Get checkmate move bonus from the current ruleset
            return self.scoring_calculator.rules('checkmate_move_bonus', 999999999.0)
        
        if temp_board.is_check(): # Check after move is made
            score += self.scoring_calculator.rules('check_move_bonus', 99999.0)

        temp_board.pop()
        if temp_board.is_capture(move):
            score += self.scoring_calculator.rules('capture_move_bonus', 5000.0)
            victim_type = temp_board.piece_type_at(move.to_square)
            aggressor_type = temp_board.piece_type_at(move.from_square)
            if victim_type and aggressor_type:
                score += (self.engine_config.get('piece_values', {}).get(victim_type, 0) * 10) - self.engine_config.get('piece_values', {}).get(aggressor_type, 0)

        if move.promotion:
            score += self.scoring_calculator.rules('promotion_move_bonus', 3000.0)
            if move.promotion == chess.QUEEN:
                score += self.engine_config.get('piece_values', {}).get(chess.QUEEN, 9.0) * 100 # Ensure piece_values is used
        return score
