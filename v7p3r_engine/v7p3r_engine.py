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
import datetime
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
    def __init__(self, board: chess.Board = chess.Board()):
        self.logger = v7p3r_engine_logger
        # Load Configuration Files
        try:
            with open("config/v7p3r_config.yaml") as f:
                self.v7p3r_config = yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"Error loading v7p3r YAML file: {e}")
            self.engine_config = {}

        # Overrides
        self.time_control = {'infinite': True}  # Default to infinite time control

        # Engine Configuration
        self.engine_name = "v7p3r"
        self.engine_version = "1.0.0"
        self.engine_config = {
            "engine_color": "white",
            "engine_ruleset": "default_evaluation",         # Name of the evaluation rule set to use, see below for available options
            "engine_search_algorithm": "minimax",           # Move search type for White (see search_algorithms for options)
            "engine_depth": 5,                            # Depth of search for AI, 1 for random, 2 for simple search, 3+ for more complex searches
            "engine_max_depth": 8,                        # Max depth of search for AI, 1 for random, 2 for simple search, 3+ for more complex searches
            "monitoring_enabled": True,                   # Enable or disable monitoring features
            "engine_logger": "v7p3r_engine_logger",
            "max_game_count": 1,               # Number of games to play in AI vs AI mode
            "starting_position": "default",   # Default starting position name (or FEN string)
            "white_player": "v7p3r",          # Name of the engine being used (e.g., 'v7p3r', 'stockfish'), this value is a direct reference to the engine configuration values in their respective config files
            "black_player": "stockfish"      # sets this colors engine configuration name, same as above, important note that if the engines are set the same then only whites metrics will be collected to prevent negation in win loss metrics

        }

        # Required Engine Modules
        self.search_engine = v7p3rSearch(board, self.engine_config)

        # Hard-coded base piece values
        self.piece_values = {
            chess.KING: 0.0,
            chess.QUEEN: 9.0,
            chess.ROOK: 5.0,
            chess.BISHOP: 3.25,
            chess.KNIGHT: 3.0,
            chess.PAWN: 1.0
        }      

        # Initialize Board
        if board is None or not isinstance(board, chess.Board):
            self.board = chess.Board()
        else:
            self.board = board.copy()

    def close(self):
        self.reset()
        if self.logger:
            self.logger.debug("v7p3rEngine closed and resources cleaned up.")

    def reset(self):
        if self.board is None:
            board = chess.Board()
        else:
            board = self.board
        self.current_player = chess.WHITE if board.turn else chess.BLACK
        self.nodes_searched = 0
        if self.logger:
            self.logger.debug(f"v7p3rEngine for {self.engine_config.get('engine_color')} reset to initial state.")