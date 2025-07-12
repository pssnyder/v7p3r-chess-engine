# v7p3r_engine.py

""" v7p3r Engine
This module impleme        # Engine components initialization in dependency order
        # 1. Base components without dependencies
        self.pst = v7p3rPST()
        self.opening_book = v7p3rBook()
        self.time_manager = v7p3rTime()
        
        # 2. Rules system
        self.rules = v7p3rRules(self.ruleset_name, self.pst)
        
        # 3. Scoring system
        self.scoring_calculator = v7p3rScore(self.rules, self.pst)
        
        # 4. Move organization
        self.move_organizer = v7p3rOrdering(self.scoring_calculator)
        
        # 5. Main search engine with all dependencies
        self.search_engine = v7p3rSearch(
            self.scoring_calculator,
            self.move_organizer,
            self.time_manager,
            self.opening_book,
            self.rules,
            self.engine_config
        )gine for the v7p3r chess AI.
It provides handler functionality for search algorithms, evaluation functions, and move ordering
"""

import chess

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from v7p3r_search import v7p3rSearch
from v7p3r_score import v7p3rScore
from v7p3r_ordering import v7p3rOrdering
from v7p3r_time import v7p3rTime
from v7p3r_book import v7p3rBook
from v7p3r_pst import v7p3rPST
from v7p3r_config import v7p3rConfig
from v7p3r_rules import v7p3rRules

# =====================================
# ========== ENGINE CLASS =============
class v7p3rEngine:
    def __init__(self, engine_config=None):
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
        self.pst = v7p3rPST()
        self.rules_manager = v7p3rRules(self.ruleset_name, self.pst)
        self.scoring_calculator = v7p3rScore(self.rules_manager, self.pst)
        self.ordering = v7p3rOrdering(scoring_calculator=self.scoring_calculator)
        self.time_manager = v7p3rTime()
        self.opening_book = v7p3rBook()
        self.search_engine = v7p3rSearch(
            self.scoring_calculator,
            self.time_manager
        )
        
        # State tracking
        self.current_evaluation = 0.0
        self.nodes_searched = 0
        self.last_move_time = 0.0
        self.last_pv_line = []

    def get_move(self, board: chess.Board, color: chess.Color) -> chess.Move:
        """Get the best move for the current position following the evaluation hierarchy."""
        import time
        start_time = time.time()
        
        try:
            # 1. Try book moves first
            book_move = self.opening_book.get_book_move(board)
            if book_move and board.is_legal(book_move):
                self._update_state(0.0, 1, time.time() - start_time)
                return book_move
                
            # 2. Check for immediate checkmate
            mate_move = self.rules_manager.find_checkmate_in_n(board, 5, color)
            if mate_move and board.is_legal(mate_move):
                self._update_state(float('inf'), 1, time.time() - start_time)
                return mate_move
                
            # 3. If we're in trouble, look for saving moves
            if board.is_check():
                # Search deeper in check positions
                self.search_engine.depth = min(self.depth + 2, self.max_depth)
            else:
                self.search_engine.depth = self.depth
                
            # 4. Get move from search engine
            move = self.search_engine.search(board, color)
            
            # 5. Update engine state
            self._update_state(
                self.scoring_calculator.evaluate_position(board),
                self.search_engine.nodes_searched,
                time.time() - start_time
            )
            
            return move if move else chess.Move.null()
            
        except Exception as e:
            print(f"Error in get_move: {str(e)}")
            # Emergency fallback - return first legal move
            legal_moves = list(board.legal_moves)
            return legal_moves[0] if legal_moves else chess.Move.null()
            
    def _update_state(self, evaluation: float, nodes: int, time_taken: float):
        """Update engine state after a move calculation"""
        self.current_evaluation = evaluation
        self.nodes_searched = nodes
        self.last_move_time = time_taken
        
    def get_current_evaluation(self) -> float:
        """Get the evaluation of the current position"""
        return self.current_evaluation
        
    def get_nodes_searched(self) -> int:
        """Get the number of nodes searched in the last search"""
        return self.nodes_searched
        
    def get_last_move_time(self) -> float:
        """Get the time taken for the last move calculation"""
        return self.last_move_time
