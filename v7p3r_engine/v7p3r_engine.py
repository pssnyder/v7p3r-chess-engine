# v7p3r_engine/v7p3r_engine.py

""" v7p3r Engine
This module implements the core engine for the v7p3r chess AI.
It provides handler functionality for search algorithms, evaluation functions, and move ordering
"""

import chess
import yaml
import logging
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import datetime
from v7p3r_pst import v7p3rPST
from v7p3r_time import v7p3rTime
from v7p3r_book import v7p3rBook
from v7p3r_score import v7p3rScore
from v7p3r_search import v7p3rSearch

# =====================================
# ========== LOGGING SETUP ============
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
v7p3r_engine_logger = logging.getLogger("v7p3r_evaluation_engine")
v7p3r_engine_logger.setLevel(logging.DEBUG)
if not v7p3r_engine_logger.handlers:
    if not os.path.exists('logging'):
        os.makedirs('logging', exist_ok=True)
    from logging.handlers import RotatingFileHandler
    # Use a timestamped log file for each engine run
    timestamp = get_timestamp()
    log_file_path = f"logging/v7p3r_evaluation_engine_{timestamp}.log"
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

# =====================================
# ========== ENGINE CLASS =============
class v7p3rEngine:
    def __init__(self, board: chess.Board = chess.Board(), player: chess.Color = chess.WHITE):
         # Load Configuration Files
        try:
            with open("config/v7p3r_config.yaml") as f:
                self.v7p3r_config = yaml.safe_load(f) or {}
        except Exception as e:
            v7p3r_engine_logger.error(f"Error loading v7p3r or game settings YAML files: {e}")
            self.v7p3r_config = {}
            self.chess_game_config = {}
        
        self.board = board
        self.current_player = player
        self.time_manager = v7p3rTime()
        self.opening_book = v7p3rBook()
        self.pst = v7p3rPST()
        self.scoring_calculator = v7p3rScore(self.v7p3r_config.get('v7p3r', {}), self.v7p3r_config)
        self.search_engine = v7p3rSearch(self.v7p3r_config)
        self.time_control = {'infinite': True}  # Default to infinite time control

        self.nodes_searched = 0
        self.killer_moves = [[None, None] for _ in range(50)] 
        self.history_table = {}
        self.counter_moves = {}

        # Default Piece Values
        self.piece_values = {
            chess.KING: 0.0,
            chess.QUEEN: 9.0,
            chess.ROOK: 5.0,
            chess.BISHOP: 3.25,
            chess.KNIGHT: 3.0,
            chess.PAWN: 1.0
        }

        # Engine Details
        self.engine_name = self.v7p3r_config.get('engine_name', 'v7p3r')
        self.engine_version = self.v7p3r_config.get('engine_version', '1.0.0')
        self.engine_color = 'white' if board.turn else 'black'

        # Monitoring Config
        self.monitoring_config = self.chess_game_config.get('monitoring', {})
        if self.monitoring_config is not None:
            self.logging_enabled = self.monitoring_config.get('enable_logging', False)
            self.evaluation_display_enabled = self.monitoring_config.get('show_evaluation', False)
            self.show_thoughts = self.monitoring_config.get('show_thinking', False)
            self.logger = v7p3r_engine_logger
            if not self.logging_enabled:
                self.show_thoughts = False
            if self.logging_enabled:
                self.logger.debug(f"Logging enabled for {player} v7p3rEngine")
        else:
            self.logging_enabled = False
            self.evaluation_display_enabled = False
            self.show_thoughts = False
            self.logger = None

        # Dynamically fetch the config for this engine
        self.engine_config = self.v7p3r_config.get(self.engine_name, {})  # Dynamically load configuration based on engine_name
        if self.engine_config is not None:
            # Set up config values for evaluator and scoring
            self.ruleset = self.engine_config.get('ruleset')
            self.search_algorithm = self.engine_config.get('search_algorithm', 'random')
            self.depth = self.engine_config.get('depth', 3)
            self.max_depth = self.engine_config.get('max_depth', 4)
            self.pst_weight = self.engine_config.get('pst_weight', 1.0)
            self.scoring_modifier = self.engine_config.get('scoring_modifier', 1.0)

        # Initialize a scoring setup for this engine config
        if self.logging_enabled and self.logger:
            self.logger.debug(f"Configuring v7p3r AI for {self.engine_color} via: {self.engine_config}")
        self.scoring_calculator = v7p3rScore(self.engine_config, self.v7p3r_config)
        if self.show_thoughts and self.logger:
            self.logger.debug(f"AI configured for {self.engine_color}: type={self.search_algorithm} depth={self.depth}, ruleset={self.ruleset}")

    def close(self):
        self.reset()
        if self.show_thoughts and self.logger:
            self.logger.debug("v7p3rEngine closed and resources cleaned up.")

    def reset(self):
        if self.board is None:
            board = chess.Board()
        else:
            board = self.board
        self.current_player = chess.WHITE if board.turn else chess.BLACK
        self.nodes_searched = 0
        self.killer_moves = [[None, None] for _ in range(50)]
        self.history_table.clear()
        self.counter_moves.clear()
        if self.show_thoughts and self.logger:
            self.logger.debug(f"v7p3rEngine for {self.engine_color} reset to initial state.")

    def _is_draw_condition(self, board):
        if board.can_claim_threefold_repetition():
            return True
        if board.can_claim_fifty_moves():
            return True
        if board.is_seventyfive_moves():
            return True
        return False

    def _get_game_phase_factor(self, board: chess.Board) -> float:        
        total_material = 0
        for piece_type, value in self.piece_values.items():
            if piece_type != chess.KING:
                total_material += len(board.pieces(piece_type, chess.WHITE)) * value
                total_material += len(board.pieces(piece_type, chess.BLACK)) * value

        QUEEN_ROOK_MATERIAL = self.piece_values[chess.QUEEN] + self.piece_values[chess.ROOK]
        TWO_ROOK_MATERIAL = self.piece_values[chess.ROOK] * 2
        KNIGHT_BISHOP_MATERIAL = self.piece_values[chess.KNIGHT] + self.piece_values[chess.BISHOP]

        if total_material >= (QUEEN_ROOK_MATERIAL * 2) + (KNIGHT_BISHOP_MATERIAL * 2):
            return 0.0
        if total_material < (TWO_ROOK_MATERIAL + KNIGHT_BISHOP_MATERIAL * 2) and total_material > (KNIGHT_BISHOP_MATERIAL * 2):
            return 0.5
        if total_material <= (KNIGHT_BISHOP_MATERIAL * 2):
            return 1.0
        
        return 0.0