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
from v7p3r_score import v7p3rScore
from v7p3r_ordering import v7p3rOrdering
from v7p3r_time import v7p3rTime
from v7p3r_book import v7p3rBook
from v7p3r_pst import v7p3rPST
from v7p3r_config import v7p3rConfig
from v7p3r_rules import v7p3rRules

# =====================================
# ========== LOGGING SETUP ============
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create logging directory relative to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
log_dir = os.path.join(project_root, 'logging')
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# Setup individual logger for this file
timestamp = get_timestamp()
log_filename = f"v7p3r_{timestamp}.log"
log_file_path = os.path.join(log_dir, log_filename)

v7p3r_logger = logging.getLogger(f"v7p3r_{timestamp}")
v7p3r_logger.setLevel(logging.DEBUG)

if not v7p3r_logger.handlers:
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
    v7p3r_logger.addHandler(file_handler)
    v7p3r_logger.propagate = False

# =====================================
# ========== ENGINE CLASS =============
class v7p3rEngine:
    def __init__(self, engine_config=None):
        self.logger = v7p3r_logger

        # Overrides
        self.time_control = {    # Default to infinite time control
            'infinite': True
        }

        # Base Engine Configuration
        self.config_manager = v7p3rConfig()
        if engine_config is not None:
            self.engine_config = engine_config
        else:
            self.engine_config = self.config_manager.get_engine_config()  # Ensure it's always a dictionary
            if self.logger:
                self.logger.info("No engine configuration provided, using v7p3r's default_config.")

        # Load engine config and default values
        self.name = self.engine_config.get("name", "v7p3r")
        self.version = self.engine_config.get("version", "0.0.0")
        self.ruleset_name = self.engine_config.get("ruleset", "default_ruleset")
        self.search_algorithm = self.engine_config.get("search_algorithm", "simple")
        self.depth = self.engine_config.get("depth", 5)
        self.max_depth = self.engine_config.get("max_depth", 8)
        self.use_game_phase = self.engine_config.get("use_game_phase", True)
        self.piece_values = self.engine_config.get("piece_values", {
            chess.KING: 0.0,
            chess.QUEEN: 9.0,
            chess.ROOK: 5.0,
            chess.BISHOP: 3.25,
            chess.KNIGHT: 3.0,
            chess.PAWN: 1.0
        })  
        
        # Base Ruleset
        self.ruleset = self.config_manager.get_ruleset()
        

        # Required Engine Modules
        self.pst = v7p3rPST(self.logger)
        self.rules_manager = v7p3rRules(ruleset=self.ruleset, pst=self.pst)
        self.scoring_calculator = v7p3rScore(rules_manager=self.rules_manager, pst=self.pst)
        self.move_organizer = v7p3rOrdering(self.scoring_calculator)
        self.time_manager = v7p3rTime()
        self.opening_book = v7p3rBook()
        self.search_engine = v7p3rSearch(self.scoring_calculator, self.move_organizer, self.time_manager, self.opening_book)

        # Debug: Check if all components are properly initialized
        if self.logger:
            self.logger.info(f"pst: {type(self.pst)} | scoring: {type(self.scoring_calculator)} | ordering: {type(self.move_organizer)} | time: {type(self.time_manager)} | book: {type(self.opening_book)} | search: {type(self.search_engine)}")
            self.logger.info(f"search_engine.search method: {type(getattr(self.search_engine, 'search', 'NOT_FOUND'))}")
            
        print(f"pst: {type(self.pst)} | scoring: {type(self.scoring_calculator)} | ordering: {type(self.move_organizer)} | time: {type(self.time_manager)} | book: {type(self.opening_book)} | search: {type(self.search_engine)}")
