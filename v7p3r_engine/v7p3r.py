# v7p3r_engine/v7p3r_engine.py

""" v7p3r Engine
This module implements the core engine for the v7p3r chess AI.
It provides handler functionality for search algorithms, evaluation functions, and move ordering
"""

import chess
import logging
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import datetime
from v7p3r_search import v7p3rSearch
from v7p3r_engine.v7p3r_score_v2 import v7p3rScore
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
    def __init__(self, engine_config=None):
        self.logger = v7p3r_engine_logger

        # Overrides
        self.time_control = {    # Default to infinite time control
            'infinite': True
        }

        # Engine Configuration
        if engine_config is not None:
            self.engine_config = engine_config
        else:
            if self.logger:
                self.logger.info("No engine configuration provided, using v7p3r's inbuilt configuration.")

        # Load engine config and default values
        self.name = self.engine_config.get("name", "v7p3r")
        self.version = self.engine_config.get("version", "1.0.0")
        self.color = self.engine_config.get("color", "white")
        self.ruleset = self.engine_config.get("ruleset", "default_evaluation")
        self.search_algorithm = self.engine_config.get("search_algorithm", "lookahead")
        self.depth = self.engine_config.get("depth", 3)
        self.max_depth = self.engine_config.get("max_depth", 4)
        self.use_game_phase = self.engine_config.get("use_game_phase", False)
        self.monitoring_enabled = self.engine_config.get("monitoring_enabled", False)
        self.verbose_output = self.engine_config.get("verbose_output", False)
        self.game_count = self.engine_config.get("game_count", 1)
        self.starting_position = self.engine_config.get("starting_position", "default")
        self.white_player = self.engine_config.get("white_player", "v7p3r")
        self.black_player = self.engine_config.get("black_player", "stockfish")
        self.piece_values = self.engine_config.get("piece_values", {
            chess.KING: 0.0,
            chess.QUEEN: 9.0,
            chess.ROOK: 5.0,
            chess.BISHOP: 3.25,
            chess.KNIGHT: 3.0,
            chess.PAWN: 1.0
        })  
        # Required Engine Modules
        self.pst = v7p3rPST(self.logger)
        self.scoring_calculator = v7p3rScore(self.engine_config, self.pst, self.logger)
        self.move_organizer = v7p3rOrdering(self.engine_config, self.scoring_calculator, self.logger)
        self.time_manager = v7p3rTime()
        self.opening_book = v7p3rBook()
        self.search_engine = v7p3rSearch(self.engine_config, self.scoring_calculator, self.move_organizer, self.time_manager, self.opening_book, self.logger)