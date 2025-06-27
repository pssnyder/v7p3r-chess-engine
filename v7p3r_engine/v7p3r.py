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
from v7p3r_score import v7p3rScore
from v7p3r_ordering import v7p3rOrdering
from v7p3r_time import v7p3rTime
from v7p3r_book import v7p3rBook
from v7p3r_pst import v7p3rPST

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
    def __init__(self):
        self.logger = v7p3r_engine_logger
        # Load Configuration Files
        try:
            with open("config/v7p3r_config.yaml") as f:
                self.v7p3r_config = yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"Error loading v7p3r YAML file: {e}")
            self.engine_config = {}

        # Overrides
        self.time_control = {    # Default to infinite time control
            'infinite': True
        }  
        self.piece_values = {    # Hard-coded base piece values
            chess.KING: 0.0,
            chess.QUEEN: 9.0,
            chess.ROOK: 5.0,
            chess.BISHOP: 3.25,
            chess.KNIGHT: 3.0,
            chess.PAWN: 1.0
        }

        # Engine Configuration
        self.engine_config = {
            "name": "v7p3r",                     # Name of the engine, used for identification and logging
            "version": "1.0.0",                  # Version of the engine, used for identification and logging
            "color": "white",                    # Color of the engine, either 'white' or 'black'
            "ruleset": "default_evaluation",     # Name of the evaluation rule set to use, see below for available options
            "search_algorithm": "minimax",       # Move search type for White (see search_algorithms for options)
            "depth": 5,                          # Depth of search for AI, 1 for random, 2 for simple search, 3+ for more complex searches
            "max_depth": 8,                      # Max depth of search for AI, 1 for random, 2 for simple search, 3+ for more complex searches
            "monitoring_enabled": True,                 # Enable or disable monitoring features
            "logger": "v7p3r_engine_logger",     # Logger name for the engine, used for logging engine-specific events
            "max_game_count": 1,                        # Number of games to play in AI vs AI mode
            "starting_position": "default",             # Default starting position name (or FEN string)
            "white_player": "v7p3r",                    # Name of the engine being used (e.g., 'v7p3r', 'stockfish'), this value is a direct reference to the engine configuration values in their respective config files
            "black_player": "stockfish",                # sets this colors engine configuration name, same as above, important note that if the engines are set the same then only whites metrics will be collected to prevent negation in win loss metrics
            "piece_values": self.piece_values           # Hard-coded base piece values
        }

        # Load engine config and default values
        self.name = self.engine_config.get("name", "v7p3r")
        self.version = self.engine_config.get("version", "1.0.0")
        self.color = self.engine_config.get("color", "white")
        self.ruleset = self.engine_config.get("ruleset", "default_evaluation")
        self.search_algorithm = self.engine_config.get("search_algorithm", "minimax")
        self.depth = self.engine_config.get("depth", 5)
        self.max_depth = self.engine_config.get("max_depth", 8)
        self.monitoring_enabled = self.engine_config.get("monitoring_enabled", True)
        self.max_game_count = self.engine_config.get("max_game_count", 1)
        self.starting_position = self.engine_config.get("starting_position", "default")
        self.white_player = self.engine_config.get("white_player", "v7p3r")
        self.black_player = self.engine_config.get("black_player", "stockfish")

        # Required Engine Modules
        self.pst = v7p3rPST()
        self.scoring_calculator = v7p3rScore(self.engine_config, self.pst, self.logger)
        self.move_organizer = v7p3rOrdering(self.engine_config, self.scoring_calculator, self.logger)
        self.time_manager = v7p3rTime()
        self.opening_book = v7p3rBook()
        self.search_engine = v7p3rSearch(self.engine_config, self.scoring_calculator, self.move_organizer, self.time_manager, self.opening_book, self.logger)