# chess_game.py

import os
import sys
import pygame
import chess
import chess.pgn
import random
import yaml
import datetime
import logging
import math
import socket
import time # Import time for measuring move duration
from io import StringIO
import hashlib

# Define the maximum frames per second for the game loop
MAX_FPS = 60
from v7p3r_engine.v7p3r import v7p3rEvaluationEngine # Corrected import for v7p3rEvaluationEngine
from v7p3r_nn_engine.v7p3r_nn import v7p3rNeuralNetwork # Import Neural Network engine
from engine_utilities.stockfish_handler import StockfishHandler # Corrected import path and name
from metrics.metrics_store import MetricsStore # Import MetricsStore

# Resource path config for distro
def resource_path(relative_path):
    # Use getattr to avoid attribute error
    base = getattr(sys, '_MEIPASS', None)
    if base:
        return os.path.join(base, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# At module level, define a single logger for this file
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def get_log_file_path():
    # Use a timestamped log file for each game session
    timestamp = get_timestamp()
    log_dir = "logging"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"chess_game.log")

chess_game_logger = logging.getLogger("chess_game")
chess_game_logger.setLevel(logging.DEBUG)
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
    chess_game_logger.addHandler(file_handler)
    chess_game_logger.propagate = False
    _init_status["initialized"] = True
    # Store the log file path for later use (e.g., to match with PGN/config)
    _init_status["log_file_path"] = log_file_path

class ChessGame:
    def __init__(self, config):
        """
        config: ChessGameConfig object containing all configuration parameters.
        """
        # Always define human_color to avoid attribute errors
        self.human_color = None

        # Extract configuration from the ChessGameConfig object
        self.starting_position = config.fen_position if hasattr(config, 'fen_position') else None
        self.data_collector = getattr(config, 'data_collector', None)
        self.game_config_data = getattr(config, 'game_config', {})
        self.v7p3r_config_data = getattr(config, 'v7p3r_config', {})
        self.stockfish_handler_data = getattr(config, 'stockfish_handler', {})

        # Initialize Pygame (even in headless mode, for internal timing)
        pygame.init()
        self.clock = pygame.time.Clock()
        # Enable logging
        self.logging_enabled = self.game_config_data.get('monitoring', {}).get('enable_logging', True)
        self.show_thoughts = self.game_config_data.get('monitoring', {}).get('show_thinking', True)
        self.logger = chess_game_logger
        if not self.logging_enabled:
            self.show_thoughts = False
        if self.logging_enabled:
            self.logger.debug("Logging enabled for ChessGame")
        # Enable logging
        self.logging_enabled = self.game_config_data.get('monitoring', {}).get('enable_logging', True) # Adjusted path
        self.show_thoughts = self.game_config_data.get('monitoring', {}).get('show_thinking', True) # Adjusted path
        self.logger = chess_game_logger # Use the module-level logger
        if not self.logging_enabled:
            self.show_thoughts = False
        if self.logging_enabled:
            self.logger.debug("Logging enabled for ChessGame")
    

        # Initialize game settings
        self.human_color_pref = self.game_config_data.get('game_config', {}).get('human_color', 'random') # Adjusted path
        if self.starting_position == None:
            self.starting_position = self.game_config_data.get('game_config', {}).get(
                'starting_position',
                'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
            ) # Adjusted path
        print("Initializing headless AI vs AI mode. No chess GUI will be shown.")
        print("Press Ctrl+C in the terminal to stop the game early.")

        # Initialize piece fallback values (primarily for v7p3r's evaluation)
        self.piece_values = {
            chess.KING: 0,
            chess.QUEEN: 9,
            chess.ROOK: 5,
            chess.BISHOP: 3.25,
            chess.KNIGHT: 3,
            chess.PAWN: 1
        }
 
        # Initialize AI configurations
        self.game_count = self.game_config_data.get('game_config', {}).get('ai_game_count', 0) # Adjusted path
        self.ai_vs_ai = self.game_config_data.get('game_config', {}).get('ai_vs_ai', False) # Adjusted path
        self.search_algorithms = self.game_config_data.get('search_algorithms', []) # Adjusted path

        self.white_engine_config = self.game_config_data.get('white_engine_config', {}) # Adjusted path
        self.black_engine_config = self.game_config_data.get('black_engine_config', {}) # Adjusted path
        
        # Set 'engine' keys in AI configs
        self.white_engine_config['engine'] = self.white_engine_config.get('engine', 'v7p3r')
        self.black_engine_config['engine'] = self.black_engine_config.get('engine', 'v7p3r')
        self.white_eval_engine = self.white_engine_config['engine']
        self.black_eval_engine = self.black_engine_config['engine']

        # Compose engine description for headers
        def engine_desc(engine_config, v7p3r_config_data, color):
            engine = engine_config.get('engine', 'Unknown')
            # Determine which v7p3r config to use for evaluation
            if engine == 'v7p3r':
                eval_name = v7p3r_config_data.get('v7p3r', {}).get('ruleset', 'default_evaluation')
                search_algo = v7p3r_config_data.get('v7p3r', {}).get('search_algorithm', 'deepsearch')
            elif engine == 'v7p3r_opponent':
                eval_name = v7p3r_config_data.get('v7p3r_opponent', {}).get('ruleset', 'simple_evaluation')
                search_algo = v7p3r_config_data.get('v7p3r_opponent', {}).get('search_algorithm', 'random')
            else:
                eval_name = engine_config.get('ruleset', engine_config.get('evaluation', ''))
                search_algo = engine_config.get('search_algorithm', '')
            # Allow override from engine_config
            if 'ruleset' in engine_config:
                eval_name = engine_config['ruleset']
            if 'search_algorithm' in engine_config:
                search_algo = engine_config['search_algorithm']
            return f"{engine} ({eval_name}) via {search_algo}"

        self.white_engine_desc = engine_desc(self.white_engine_config, self.v7p3r_config_data, 'white')
        self.black_engine_desc = engine_desc(self.black_engine_config, self.v7p3r_config_data, 'black')

        # Set exclude_from_metrics attributes for later use
        self.exclude_white_from_metrics = self.white_engine_config.get('exclude_from_metrics', False)
        self.exclude_black_from_metrics = self.black_engine_config.get('exclude_from_metrics', False)

        if self.logging_enabled and self.logger:
            self.logger.debug(f"Initializing ChessGame with {self.starting_position} position")
            self.logger.debug(f"White Engine: {self.white_engine_desc}")
            self.logger.debug(f"Black Engine: {self.black_engine_desc}")
        
        # Debug settings
        self.show_eval = self.game_config_data.get('monitoring', {}).get('show_evaluation', False) # Adjusted path for debug settings if they were moved, assuming they are in 'monitoring' or similar in chess_game_config.yaml
        
        # Initialize MetricsStore
        self.metrics_store = MetricsStore()
        self._move_metrics_batch = []  # in-memory store of move metrics
        self.game_start_timestamp = get_timestamp()
        self.current_game_db_id = f"eval_game_{self.game_start_timestamp}.pgn"
        
        # Initialize board and new game
        # Add rated flag from config
        self.rated = self.game_config_data.get('game_config', {}).get('rated', True)
        self.new_game(self.starting_position)
        
        # Set headers
        
    # ================================
    # ====== GAME CONFIGURATION ======
    
    def new_game(self, fen_position=None):
        """Reset the game state for a new game"""
        fen_to_use = fen_position
        if fen_position and not fen_position.count('/') == 7: # Corrected FEN check
            fen_to_use = self.game_config_data.get('starting_positions', {}).get(fen_position, None) # Adjusted path
            if fen_to_use is None:
                fen_to_use = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        self.board = chess.Board(fen=fen_to_use) if fen_to_use else chess.Board()
        self.game = chess.pgn.Game()
        self.game_node = self.game
        self.selected_square = chess.SQUARES[0]
        self.last_ai_move = chess.Move.null()
        self.current_eval = 0.0
        self.current_player = chess.WHITE if self.board.turn else chess.BLACK
        self.last_move = chess.Move.null()
        self.move_history = []
        
        # Initialize/Reset AI engines based on config
        self._initialize_ai_engines()

        # Reset PGN headers and file
        self.set_headers()
        self.quick_save_pgn("logging/active_game.pgn")
        
        if self.logging_enabled and self.logger:
            self.logger.info(f"Starting new game: {self.current_game_db_id}.")

    def _initialize_ai_engines(self):
        """Initializes or re-initializes AI engines based on config."""
        # Stockfish specific settings from stockfish_config.yaml
        stockfish_path = self.stockfish_handler_data.get('stockfish_handler', {}).get('path')
        stockfish_elo = self.stockfish_handler_data.get('stockfish_handler', {}).get('elo_rating')
        stockfish_skill = self.stockfish_handler_data.get('stockfish_handler', {}).get('skill_level')
        debug_stockfish = self.stockfish_handler_data.get('stockfish_handler', {}).get('debug_stockfish', False)

        # v7p3r engine general settings from v7p3r_config.yaml
        v7p3r_ruleset = self.v7p3r_config_data.get('v7p3r', {}).get('ruleset', 'default_evaluation')
        v7p3r_depth = self.v7p3r_config_data.get('v7p3r', {}).get('depth', 3)        # White Engine
        if self.white_engine_config.get('engine', '').lower() == 'v7p3r':
            # Pass relevant parts of v7p3r_config_data and game_config_data (white_engine_config)
            v7p3r_engine_config = {**self.v7p3r_config_data.get('v7p3r', {}), **self.white_engine_config}
            self.white_engine = v7p3rEvaluationEngine(self.board, chess.WHITE, engine_config=v7p3r_engine_config)
        elif self.white_engine_config.get('engine', '').lower() == 'v7p3r_nn':
            # Initialize Neural Network engine
            self.white_engine = v7p3rNeuralNetwork(config_path="config/v7p3r_nn_config.yaml")
            self.logger.info("Initialized v7p3r Neural Network engine for White")
        elif self.white_engine_config.get('engine', '').lower() == 'stockfish':
            if not stockfish_path or not os.path.exists(stockfish_path):
                self.logger.error(f"Stockfish executable not found at: {stockfish_path}. White AI defaulting to v7p3r.")
                self.white_engine = v7p3rEvaluationEngine(self.board, chess.WHITE, engine_config=self.white_engine_config)
                self.white_engine_config['engine'] = 'v7p3r' # Force engine name change
                self.white_engine_config['search_algorithm'] = 'random' # Force type change as Stockfish type invalid
            else:
                self.white_engine = StockfishHandler(
                    stockfish_path=stockfish_path,
                    elo_rating=stockfish_elo,
                    skill_level=stockfish_skill,
                    debug_mode=debug_stockfish
                )
        else: # Default to v7p3r if unknown engine
            self.logger.warning(f"Unknown engine for White AI: {self.white_engine_config['engine']}. Defaulting to v7p3r.")
            self.white_engine = v7p3rEvaluationEngine(self.board, chess.WHITE, engine_config=self.white_engine_config)
            self.white_engine_config['engine'] = 'v7p3r'        # Black Engine
        if self.black_engine_config.get('engine', '').lower() == 'v7p3r':
            # Pass relevant parts of v7p3r_config_data and game_config_data (black_engine_config)
            v7p3r_engine_config = {**self.v7p3r_config_data.get('v7p3r', {}), **self.black_engine_config}
            self.black_engine = v7p3rEvaluationEngine(self.board, chess.BLACK, engine_config=v7p3r_engine_config)
        elif self.black_engine_config.get('engine', '').lower() == 'v7p3r_nn':
            # Initialize Neural Network engine
            self.black_engine = v7p3rNeuralNetwork(config_path="config/v7p3r_nn_config.yaml")
            self.logger.info("Initialized v7p3r Neural Network engine for Black")
        elif self.black_engine_config.get('engine', '').lower() == 'stockfish':
            if not stockfish_path or not os.path.exists(stockfish_path):
                self.logger.error(f"Stockfish executable not found at: {stockfish_path}. Black AI defaulting to v7p3r.")
                self.black_engine = v7p3rEvaluationEngine(self.board, chess.BLACK, engine_config=self.black_engine_config)
                self.black_engine_config['engine'] = 'v7p3r' # Force engine name change
                self.black_engine_config['search_algorithm'] = 'random' # Force type change as Stockfish type invalid
            else:
                self.black_engine = StockfishHandler(
                    stockfish_path=stockfish_path,
                    elo_rating=stockfish_elo,
                    skill_level=stockfish_skill,
                    debug_mode=debug_stockfish
                )
        else: # Default to v7p3r if unknown engine
            self.logger.warning(f"Unknown engine for Black AI: {self.black_engine_config['engine']}. Defaulting to v7p3r.")
            self.black_engine = v7p3rEvaluationEngine(self.board, chess.BLACK, engine_config=self.black_engine_config)
            self.black_engine_config['engine'] = 'v7p3r'

        # Reset and configure engines for the new game board
        self.white_engine.reset()
        self.black_engine.reset()

    def set_headers(self):
        # Set initial PGN headers
        white_depth = self.white_engine_config.get('depth') # Depth might come from white_engine_config in chess_game_config.yaml
        if white_depth is None and (self.white_engine_config.get('engine','').lower() == 'v7p3r' or self.white_engine_config.get('engine','').lower() == 'v7p3r_nn'): # Or from v7p3r_config.yaml if v7p3r
            white_depth = self.v7p3r_config_data.get('v7p3r', {}).get('depth', '#')
        elif white_depth is None and (self.white_engine_config.get('engine','').lower() == 'v7p3r_opponent' or self.white_engine_config.get('engine','').lower() == 'v7p3r_nn_opponent'):
            white_depth = self.v7p3r_config_data.get('v7p3r_opponent', {}).get('depth', '#')
        elif white_depth is None and self.white_engine_config.get('engine','').lower() == 'stockfish': # Or from stockfish.yaml if Stockfish
            white_depth = self.stockfish_handler_data.get('stockfish', {}).get('depth', '#') # Fallback for Stockfish depth
        else:
            white_depth = '#'


        black_depth = self.black_engine_config.get('depth') # Depth might come from black_engine_config in chess_game_config.yaml
        if black_depth is None and (self.black_engine_config.get('engine','').lower() == 'v7p3r' or self.black_engine_config.get('engine','').lower() == 'v7p3r_nn'): # Or from v7p3r_config.yaml if v7p3r
            black_depth = self.v7p3r_config_data.get('v7p3r', {}).get('depth', '#')
        elif black_depth is None and (self.black_engine_config.get('engine','').lower() == 'v7p3r_opponent' or self.black_engine_config.get('engine','').lower() == 'v7p3r_nn_opponent'):
            black_depth = self.v7p3r_config_data.get('v7p3r_opponent', {}).get('depth', '#')
        elif black_depth is None and self.black_engine_config.get('engine','').lower() == 'stockfish': # Or from stockfish.yaml if Stockfish
            black_depth = self.stockfish_handler_data.get('stockfish', {}).get('depth', '#') # Fallback for Stockfish depth
        else:
            black_depth = '#'

        white_engine_name = self.white_engine_config.get('engine', 'Unknown')
        black_engine_name = self.black_engine_config.get('engine', 'Unknown')

        # Use the new engine description for headers
        if self.ai_vs_ai:
            self.game.headers["Event"] = "AI vs. AI Game"
            self.game.headers["White"] = f"AI: {self.white_engine_desc}"
            self.game.headers["Black"] = f"AI: {self.black_engine_desc}"
        elif not self.ai_vs_ai and self.human_color_pref:
            self.game.headers["Event"] = "Human vs. AI Game"
            if self.human_color == chess.WHITE:
                self.game.headers["White"] = "Human"
                self.game.headers["Black"] = f"AI: {self.black_engine_desc}"
            else:
                self.game.headers["White"] = f"AI: {self.white_engine_desc}"
                self.game.headers["Black"] = "Human"

        self.game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        self.game.headers["Site"] = socket.gethostbyname(socket.gethostname())
        self.game.headers["Round"] = "#"
        self.game.headers["Rated"] = str(self.rated)
        
    def set_colors(self):
        if self.ai_vs_ai:
            self.flip_board = False # White on bottom for AI vs AI
            self.human_color = None
            self.ai_color = None
        else: # Human vs AI mode
            if self.human_color_pref.lower() in ['white', 'w']:
                user_color = 'w'
            elif self.human_color_pref.lower() in ['black', 'b']:
                user_color = 'b'
            else:
                user_color = random.choice(['w', 'b'])

            self.flip_board = (user_color == 'b')

            self.human_color = chess.WHITE if user_color == 'w' else chess.BLACK
            self.ai_color = chess.WHITE if self.human_color == chess.BLACK else chess.BLACK

    def _is_draw_condition(self, board):
        """Check if the current board position is a draw condition"""
        if board.can_claim_threefold_repetition():
            return True
        if board.can_claim_fifty_moves():
            return True
        if board.is_seventyfive_moves():
            return True
        return False
    
    def strict_draw_prevention(self):
        """If strict_draw_prevention is enabled, try to select a legal move that avoids an immediate draw
        (threefold repetition, 50-move, insufficient material, or 75-move rule) and return it. Otherwise, return None."""
        if not self.game_config_data.get('game_config', {}).get('strict_draw_prevention', False):
            return None
        for move in self.board.legal_moves:
            board_copy = self.board.copy(stack=False)
            board_copy.push(move)
            if not (board_copy.can_claim_threefold_repetition() or
                    board_copy.can_claim_fifty_moves() or
                    board_copy.is_insufficient_material() or
                    board_copy.is_seventyfive_moves()):
                return move
        return None

    def get_board_result(self):
        """Return the result string for the current board state."""
        # Explicitly handle all draw and win/loss cases, fallback to "*"
        if self.board.is_checkmate():
            # The side to move is checkmated, so the other side wins
            return "1-0" if self.board.turn == chess.BLACK else "0-1"
        # Explicit draw conditions
        if (
            self.board.is_stalemate()
            or self.board.is_insufficient_material()
            or self.board.can_claim_fifty_moves()
            or self.board.can_claim_threefold_repetition()
            or self.board.is_seventyfive_moves()
            or self.board.is_fivefold_repetition()
            or self.board.is_variant_draw()
        ):
            return "1/2-1/2"
        # If the game is over but not by checkmate or above draws, fallback to chess.Board.result()
        if self.board.is_game_over():
            result = self.board.result()
            # Defensive: If result is not a valid string, force draw string
            if result not in ("1-0", "0-1", "1/2-1/2"):
                return "1/2-1/2"
            return result
        return "*"

    def handle_game_end(self):
        # Before ending the game due to draw conditions, attempt strict draw prevention if enabled.
        if self.board.is_game_over(claim_draw=self._is_draw_condition(self.board)):
            # If a draw is about to occur but an alternative move exists, play it instead.
            alt_move = self.strict_draw_prevention()
            if alt_move is not None:
                if self.logging_enabled and self.logger:
                    self.logger.info(f"Strict draw prevention triggered, playing alternative move: {alt_move}")
                self.push_move(alt_move)
                return False
            # Otherwise add the final move (if not already added) and conclude the game.
            # Ensure the result is set in the PGN headers and game node
            result = self.get_board_result()
            self.game.headers["Result"] = result
            self.game_node = self.game.end()
            self.save_game_data()
            print(f"\nGame over: {result}")
            return True
        
        if self.board.is_seventyfive_moves():
            result = "1/2-1/2"
            if self.logging_enabled and self.logger:
                self.logger.info("\nGame automatically drawn by seventy-five move rule!")
            self.game.headers["Result"] = result
            self.game_node = self.game.end()
            self.save_game_data()
            print(f"Game over: {result}")
            return True

        return False
    
    def record_evaluation(self):
        """Record evaluation score in PGN comments"""
        current_player_color = chess.WHITE if self.board.turn else chess.BLACK
        engine_for_eval = self.white_engine if current_player_color == chess.WHITE else self.black_engine
        score = engine_for_eval.evaluate_position_from_perspective(self.board, current_player_color)
        self.current_eval = score
        
        if self.game_node.move:
            self.game_node.comment = f"Eval: {score:.2f}"
        else:
            self.game.comment = f"Initial Eval: {score:.2f}"
            
    def save_game_data(self):
        """Save the game data to local files and database only."""
        games_dir = "games"
        os.makedirs(games_dir, exist_ok=True)
        timestamp = self.game_start_timestamp
        game_id = f"eval_game_{timestamp}"

        # Finalize game and headers
        result = self.get_board_result()
        self.game.headers["Result"] = result
        self.game_node = self.game.end()

        # --- Get PGN text ---
        buf = StringIO()
        exporter = chess.pgn.FileExporter(buf)
        self.game.accept(exporter)
        pgn_text = buf.getvalue()

        # Prepare game result data for metrics_store (ensure all fields are present)
        metrics_data = {
            "game_id": game_id,
            "timestamp": timestamp,
            "winner": result,
            "game_pgn": pgn_text,
            "white_player": self.game.headers.get("White"),
            "black_player": self.game.headers.get("Black"),
            "game_length": self.board.fullmove_number,
            "white_engine_config": self.white_engine_config,
            "black_engine_config": self.black_engine_config,
            # Add extra fields for full schema coverage:
            "white_engine_id": hashlib.md5(self.white_engine_config.get('engine', 'unknown').encode()).hexdigest(),
            "black_engine_id": hashlib.md5(self.black_engine_config.get('engine', 'unknown').encode()).hexdigest(),
            "white_engine_name": self.white_engine_config.get('engine', 'unknown'),
            "black_engine_name": self.black_engine_config.get('engine', 'unknown'),
            "white_engine_version": self.white_engine_config.get('engine_version', self.white_engine_config.get('version', '1.0')),
            "black_engine_version": self.black_engine_config.get('engine_version', self.black_engine_config.get('version', '1.0')),
            "exclude_white_from_metrics": int(self.white_engine_config.get('exclude_from_metrics', False)),
            "exclude_black_from_metrics": int(self.black_engine_config.get('exclude_from_metrics', False)),
            "white_search_algorithm": self.white_engine_config.get('search_algorithm', ''),
            "black_search_algorithm": self.black_engine_config.get('search_algorithm', ''),
            "white_depth": self.white_engine_config.get('depth', 0),
            "black_depth": self.black_engine_config.get('depth', 0)
        }
        # Save locally using metrics_store
        self.metrics_store.add_game_result(**metrics_data)
        
        # Save locally into pgn file
        pgn_filepath = f"games/eval_game_{timestamp}.pgn"
        with open(pgn_filepath, "w") as f:
            exporter = chess.pgn.FileExporter(f)
            self.game.accept(exporter)
            # Always append the result string at the end for compatibility
            if result != "*":
                f.write(f"\n{result}\n")
        if self.logging_enabled and self.logger:
            self.logger.info(f"Game PGN saved to {pgn_filepath}")

        # Save YAML config file for metrics processing
        config_filepath = f"games/eval_game_{timestamp}.yaml"
        # Save a combined config for this specific game, including relevant parts of all loaded configs
        game_specific_config = {
            "game_settings": self.game_config_data,
            "v7p3r_settings": self.v7p3r_config_data,
            "stockfish_settings": self.stockfish_handler_data,
            "white_actual_config": self.white_engine_config, # The specific config used by white AI for this game
            "black_actual_config": self.black_engine_config  # The specific config used by black AI for this game
        }
        with open(config_filepath, "w") as f:
            yaml.dump(game_specific_config, f)
        if self.logging_enabled and self.logger:
            self.logger.info(f"Game-specific combined configuration saved to {config_filepath}")

        log_filepath = f"games/eval_game_{timestamp}.log"
        eval_log_dir = "logging"
        
        log_files_to_copy = []
        for f_name in os.listdir(eval_log_dir):
            if f_name.startswith("v7p3r_evaluation_engine.log") or \
               f_name.startswith("v7p3r_scoring_calculation.log") or \
               f_name.startswith("chess_game.log") or \
               f_name.startswith("stockfish_handler.log"):
                log_files_to_copy.append(os.path.join(eval_log_dir, f_name))
        log_files_to_copy.sort()
        
        with open(log_filepath, "w") as outfile:
            for log_file in log_files_to_copy:
                try:
                    with open(log_file, "r") as infile:
                        outfile.write(f"\n--- {os.path.basename(log_file)} ---\n")
                        outfile.write(infile.read())
                except Exception as e:
                    if self.logging_enabled and self.logger:
                        self.logger.warning(f"Could not read {log_file}: {e}")
        if self.logging_enabled and self.logger:
            self.logger.info(f"Combined logs saved to {log_filepath}")

        # Prepare complete game result data for data collector
        game_result_data = {
            "game_id": game_id,
            "timestamp": timestamp,
            "winner": result,
            "game_pgn": pgn_text,
            "white_player": self.game.headers.get("White"),
            "black_player": self.game.headers.get("Black"),
            "game_length": self.board.fullmove_number,
            "white_engine_id": self.white_engine_config.get('engine_id', 'unknown'),
            "black_engine_id": self.black_engine_config.get('engine_id', 'unknown'),
            "white_engine_name": self.white_eval_engine,
            "black_engine_name": self.black_eval_engine,
            "white_engine_version": self.white_engine_config.get('engine_version', '1.0'),
            "black_engine_version": self.black_engine_config.get('engine_version', '1.0'),
            "exclude_white_from_metrics": self.exclude_white_from_metrics,
            "exclude_black_from_metrics": self.exclude_black_from_metrics
        }

        # Save the game result to a file for instant analysis
        self.quick_save_pgn(f"games/{game_id}.pgn")  

        # Use the data collector if available
        if self.data_collector:
            try:
                self.data_collector('game_result', game_result_data)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to upload game data to cloud: {e}")                # If cloud upload fails, save locally
                self.save_local_game_files(game_id, pgn_text)
            

    def quick_save_pgn_to_file(self, filename):
        """Quick save the current game to a PGN file"""
        # Inject or update Result header so the PGN shows the game outcome
        if self.board.is_game_over():
            self.game.headers["Result"] = self.get_board_result()
            self.game_node = self.game.end()
        else:
            self.game.headers["Result"] = "*"
        
        with open(filename, "w") as f:
            exporter = chess.pgn.FileExporter(f)
            self.game.accept(exporter)

    def quick_save_pgn(self, filename):
        """Save PGN to local file."""
        try:
            # Ensure games directory exists
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                # Get PGN text
                buf = StringIO()
                exporter = chess.pgn.FileExporter(buf)
                self.game.accept(exporter)
                f.write(buf.getvalue())
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save PGN to {filename}: {e}")

    def save_local_game_files(self, game_id, pgn_text):
        """Save both PGN and YAML config files locally for metrics processing."""
        try:
            import os
            import yaml
            
            # Ensure games directory exists
            os.makedirs("games", exist_ok=True)
            
            # Save PGN file
            pgn_path = f"games/{game_id}.pgn"
            with open(pgn_path, 'w', encoding='utf-8') as f:
                f.write(pgn_text)
            
            # Save YAML config file for metrics processing
            yaml_path = f"games/{game_id}.yaml"
            config_data = {
                'game_id': game_id,
                'timestamp': self.game_start_timestamp,
                'white_engine_config': self.white_engine_config,
                'black_engine_config': self.black_engine_config,
                'game_settings': self.game_config_data,
                'v7p3r_settings': self.v7p3r_config_data,
                'stockfish_settings': self.stockfish_handler_data
            }
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            if self.logger:
                self.logger.debug(f"Saved local files: {pgn_path}, {yaml_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save local game files for {game_id}: {e}")

    def import_fen(self, fen_string):
        """Import a position from FEN notation"""
        try:
            new_board = chess.Board(fen_string)
            
            if not new_board.is_valid():
                if self.logging_enabled and self.logger:
                    self.logger.error(f"Invalid FEN position: {fen_string}")
                return False

            self.board = new_board
            self.game = chess.pgn.Game()
            self.game.setup(new_board)
            
            self.game_node = self.game

            self._initialize_ai_engines()
            
            self.selected_square = None
            
            self.game.headers["Event"] = "Custom Position Game"
            self.game.headers["SetUp"] = "1"
            self.game.headers["FEN"] = fen_string

            if self.logging_enabled and self.logger:
                self.logger.info(f"Successfully imported FEN: {fen_string}")
            return True

        except Exception as e:
            if self.logging_enabled and self.logger:
                self.logger.error(f"Unexpected problem importing FEN: {e}")
            return False

    # ===================================
    # ========= MOVE HANDLERS ===========
    
    def process_ai_move(self):
        """Process AI move for the current player"""

        ai_move = chess.Move.null()
        self.current_eval = 0.0
        
        current_player_color = chess.WHITE if self.board.turn else chess.BLACK
        current_engine_config = self.white_engine_config if current_player_color == chess.WHITE else self.black_engine_config
        current_ai_engine = self.white_engine if current_player_color == chess.WHITE else self.black_engine
        
        if self.logging_enabled and self.logger:
            self.logger.info(f"Processing AI move for {'White' if current_player_color == chess.WHITE else 'Black'} using {current_engine_config.get('engine', 'Unknown')} engine.")
        
        if self.show_thoughts:
            print(f"AI ({'White' if current_player_color == chess.WHITE else 'Black'}) is thinking...")

        move_start_time = time.perf_counter()
        
        nodes_before_search = 0
        if isinstance(current_ai_engine, v7p3rEvaluationEngine): # Changed from EvaluationEngine
            nodes_before_search = current_ai_engine.nodes_searched

        try:
            ai_move = current_ai_engine.search(self.board, current_player_color, engine_config=current_engine_config)
            
            move_end_time = time.perf_counter()
            self.move_duration = move_end_time - move_start_time
            
            nodes_this_move = 0
            pv_line_info = ""
            
            if isinstance(current_ai_engine, v7p3rEvaluationEngine): # Changed from EvaluationEngine
                nodes_after_search = current_ai_engine.nodes_searched
                nodes_this_move = nodes_after_search - nodes_before_search
            elif isinstance(current_ai_engine, StockfishHandler):
                stockfish_info = current_ai_engine.get_last_search_info()
                nodes_this_move = stockfish_info.get('nodes', 0)
                pv_line_info = stockfish_info.get('pv', '')
                self.current_eval = stockfish_info.get('score', 0.0)
                if current_player_color == chess.BLACK:
                    self.current_eval = -self.current_eval
            
            if isinstance(ai_move, chess.Move) and self.board.is_legal(ai_move):
                fen_before_move = self.board.fen()
                move_number = self.board.fullmove_number
                self.push_move(ai_move)
                
                if self.logging_enabled and self.logger:
                    self.logger.info(f"AI ({current_player_color}) played: {ai_move} (Eval: {self.current_eval:.2f})")
                self.last_ai_move = ai_move

                if self.rated:
                    # Ensure all move metric fields are present
                    metric = {
                        'game_id': self.current_game_db_id,
                        'move_number': move_number,
                        'player_color': 'w' if current_player_color == chess.WHITE else 'b',
                        'move_uci': ai_move.uci(),
                        'fen_before': fen_before_move,
                        'evaluation': self.current_eval,
                        'search_algorithm': current_engine_config.get('search_algorithm', current_engine_config.get('search_algorithm', 'unknown')),
                        'depth': current_engine_config.get('depth', 0),
                        'nodes_searched': nodes_this_move,
                        'time_taken': self.move_duration,
                        'pv_line': pv_line_info,
                        # Full schema coverage:
                        'engine_id': hashlib.md5(current_engine_config.get('engine', 'unknown').encode()).hexdigest(),
                        'engine_name': current_engine_config.get('engine', 'unknown'),
                        'engine_version': current_engine_config.get('engine_version', current_engine_config.get('version', '1.0')),
                        'exclude_from_metrics': int(current_engine_config.get('exclude_from_metrics', False))
                    }
                    self.metrics_store.add_move_metric(**metric)
                    self._move_metrics_batch.append(metric)
                    
                    # Use data collector if available
                    if self.data_collector:
                        self.data_collector('move_metric', metric)
                        
                    if self.logging_enabled and self.logger:
                        self.logger.debug(f"Move metrics for {ai_move.uci()} added to MetricsStore.")

            elif ai_move == chess.Move.null():
                if self.board.is_game_over():
                    if self.logging_enabled and self.logger:
                        self.logger.info(f"AI ({current_player_color}) received null move, game likely over.")
                else:
                    if self.logging_enabled and self.logger:
                        self.logger.error(f"AI ({current_player_color}) returned null move unexpectedly. | FEN: {self.board.fen()}")
            else:
                if self.logging_enabled and self.logger:
                    self.logger.error(f"AI ({current_player_color}) returned an invalid object type or illegal move: {ai_move}. Forcing random move. | FEN: {self.board.fen()}")
                
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    fallback_move = random.choice(legal_moves)
                    fen_before_move = self.board.fen()
                    move_number = self.board.fullmove_number
                    self.push_move(fallback_move)
                    if self.logging_enabled and self.logger:
                        self.logger.info(f"AI ({current_player_color}) played fallback move: {fallback_move} (Eval: {self.current_eval:.2f})")
                    self.last_ai_move = fallback_move

                    color_str = 'White' if current_player_color == chess.WHITE else 'Black'
                    if self.logger:
                        self.logger.info(f"{color_str} AI excluded from metrics this game due to invalid move.")
                    
                    if self.rated:
                        self.metrics_store.add_move_metric(
                            game_id=self.current_game_db_id,
                            move_number=move_number,
                            player_color='w' if current_player_color == chess.WHITE else 'b',
                            move_uci=fallback_move.uci(),
                            fen_before=fen_before_move,
                            evaluation=self.current_eval,
                            search_algorithm=current_engine_config.get('search_algorithm', 'unknown') + "_FALLBACK",
                            depth=0,
                            nodes_searched=0,
                            time_taken=0.0,
                            pv_line="FALLBACK: AI returned invalid move"
                        )
                else:
                    if self.logging_enabled and self.logger:
                        self.logger.warning(f"No legal moves for fallback. Game might be over or stalled. | FEN: {self.board.fen()}")

        except Exception as e:
            if self.logging_enabled and self.logger:
                self.logger.error(f"-- Hardstop Error -- Cannot process any AI moves: {e}. Forcing random move. | FEN: {self.board.fen()}")
            print(f"-- Hardstop Error -- Cannot process any AI moves: {e}")
            
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                fallback_move = random.choice(legal_moves)
                fen_before_move = self.board.fen()
                move_number = self.board.fullmove_number
                self.push_move(fallback_move)
                if self.logging_enabled and self.logger:
                    self.logger.info(f"AI ({current_player_color}) played emergency fallback move: {fallback_move} (Eval: {self.current_eval:.2f})")
                self.last_ai_move = fallback_move
                
                color_str = 'White' if current_player_color == chess.WHITE else 'Black'
                if self.logger:
                    self.logger.info(f"{color_str} AI excluded from metrics this game due to critical error.")
                
                if self.rated:
                    self.metrics_store.add_move_metric(
                        game_id=self.current_game_db_id,
                        move_number=move_number,
                        player_color='w' if current_player_color == chess.WHITE else 'b',
                        move_uci=fallback_move.uci(),
                        fen_before=fen_before_move,
                        evaluation=self.current_eval,
                        search_algorithm=current_engine_config.get('search_algorithm', 'unknown') + "_CRITICAL_FALLBACK",
                        depth=0,
                        nodes_searched=0,
                        time_taken=0.0,
                        pv_line=f"CRITICAL FALLBACK: {e}"
                    )
            else:
                if self.logging_enabled and self.logger:
                    self.logger.warning(f"No legal moves for emergency fallback. Game might be over or stalled. | FEN: {self.board.fen()}")

        if self.show_eval:
            print(f"AI ({'White' if current_player_color == chess.WHITE else 'Black'}) plays: {ai_move} (Eval: {self.current_eval:.2f})")
        else:
            print(f"AI ({'White' if current_player_color == chess.WHITE else 'Black'}) plays: {ai_move}")
        if self.logging_enabled and self.logger:
            self.logger.info(f"AI ({current_player_color}) played: {ai_move} (Eval: {self.current_eval:.2f}) | Time: {self.move_duration:.4f}s | Nodes: {nodes_this_move}")

    def push_move(self, move):
        """ Test and push a move to the board and game node """
        if not self.board.is_valid():
            if self.logging_enabled and self.logger:
                self.logger.error(f"Invalid board state detected! | FEN: {self.board.fen()}")
            return False
        
        if self.logging_enabled and self.logger:
            self.logger.info(f"Attempting to push move from {'White AI' if self.board.turn == chess.WHITE else 'Black AI'}: {move} | FEN: {self.board.fen()}")
        
        if isinstance(move, str):
            try:
                move = chess.Move.from_uci(move)
                if self.logging_enabled and self.logger:
                    self.logger.info(f"Converted to chess.Move move from UCI string before push: {move}")
            except ValueError:
                if self.logging_enabled and self.logger:
                    self.logger.error(f"Invalid UCI string received: {move}")
                return False
        
        if not self.board.is_legal(move):
            if self.logging_enabled and self.logger:
                self.logger.info(f"Illegal move blocked from being pushed: {move}")
            return False
        
        try:
            if self.logging_enabled and self.logger:
                self.logger.info(f"No remaining checks, pushing move: {move} | FEN: {self.board.fen()}")
            
            self.board.push(move)
            self.game_node = self.game_node.add_variation(move)
            self.last_move = move
            if self.logging_enabled and self.logger:
                self.logger.info(f"Move pushed successfully: {move} | FEN: {self.board.fen()}")
            
            self.current_player = chess.WHITE if self.board.turn else chess.BLACK
            
            current_engine_name = self.white_engine_config['engine'] if self.current_player == chess.BLACK else self.black_engine_config['engine']
            if current_engine_name.lower() != 'stockfish':
                self.record_evaluation()
            
            # If the move ends the game, set the result header and end the game node
            if self.board.is_game_over():
                result = self.get_board_result()
                self.game.headers["Result"] = result
                self.game_node = self.game.end()
            else:
                self.game.headers["Result"] = "*"
            
            self.quick_save_pgn("logging/active_game.pgn")
            
            return True
        except ValueError as e:
            if self.logging_enabled and self.logger:
                self.logger.error(f"ValueError pushing move {move}: {e}. Dumping PGN to error_dump.pgn")
            self.quick_save_pgn("games/game_error_dump.pgn")
            return False

    # =============================================
    # ============ MAIN GAME LOOP =================

    def run(self):
        running = True
        game_count_remaining = self.game_count
        
        print(f"White Engine: {self.white_eval_engine} vs Black Engine: {self.black_eval_engine}")
        if self.logging_enabled and self.logger:
            self.logger.info(f"White Engine: {self.white_eval_engine} vs Black Engine: {self.black_eval_engine}")
        
        self._initialize_ai_engines()


        while running and ((self.ai_vs_ai and game_count_remaining >= 1)):
            if self.logging_enabled and self.logger:
                self.logger.info(f"Running chess game loop: {self.game_count - game_count_remaining}/{self.game_count} completed.")
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if not self.board.is_game_over(claim_draw=self._is_draw_condition(self.board)) and self.board.is_valid():
                self.process_ai_move()
            else:
                if self.handle_game_end():
                    game_count_remaining -= 1
                    if game_count_remaining == 0:
                        running = False
                        if self.logging_enabled and self.logger:
                            self.logger.info(f'All {self.game_count} games complete, exiting...')
                    else:
                        if self.logging_enabled and self.logger:
                            self.logger.info(f'Game {self.game_count - game_count_remaining}/{self.game_count} complete, starting next...')
                        self.game_start_timestamp = get_timestamp()
                        self.current_game_db_id = f"eval_game_{self.game_start_timestamp}.pgn"
                        self.new_game(self.starting_position)

            self.clock.tick(MAX_FPS)
            
        # Cleanup
        if pygame.get_init():
            pygame.quit()
        if hasattr(self, 'white_engine') and self.white_engine:
            if isinstance(self.white_engine, StockfishHandler):
                self.white_engine.quit()
        if hasattr(self, 'black_engine') and self.black_engine:
            if isinstance(self.black_engine, StockfishHandler):
                self.black_engine.quit()

if __name__ == "__main__":
    # Load configuration from YAML files
    import yaml
    
    class ChessGameConfig:
        def __init__(self):
            self.fen_position = None
            self.data_collector = None
            
            # Load chess game configuration
            try:
                with open('config/chess_game_config.yaml', 'r') as f:
                    chess_config = yaml.safe_load(f)
                self.game_config = chess_config
            except FileNotFoundError:
                print("Warning: chess_game_config.yaml not found, using default settings")
                self.game_config = {
                    'monitoring': {
                        'enable_logging': True,
                        'show_thinking': True
                    },
                    'game_config': {
                        'human_color': 'random',
                        'starting_position': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                        'ai_game_count': 1,
                        'ai_vs_ai': True,
                        'rated': True
                    },
                    'white_engine_config': {
                        'engine': 'v7p3r',
                        'depth': 4,
                        'exclude_from_metrics': False
                    },
                    'black_engine_config': {
                        'engine': 'stockfish',
                        'depth': 3,
                        'exclude_from_metrics': True
                    }
                }
            
            # Load v7p3r configuration
            try:
                with open('config/v7p3r_config.yaml', 'r') as f:
                    self.v7p3r_config = yaml.safe_load(f)
            except FileNotFoundError:
                print("Warning: v7p3r_config.yaml not found, using default settings")
                self.v7p3r_config = {
                    'v7p3r': {
                        'ruleset': 'default_evaluation',
                        'depth': 3
                    }
                }
            
            # Load Stockfish configuration
            try:
                with open('config/stockfish_config.yaml', 'r') as f:
                    self.stockfish_handler = yaml.safe_load(f)
            except FileNotFoundError:
                print("Warning: stockfish_config.yaml not found, using default settings")
                self.stockfish_handler = {
                    'stockfish_handler': {
                        'path': None,
                        'elo_rating': None,
                        'skill_level': None,
                        'debug_stockfish': False
                    }
                }

    config = ChessGameConfig()
    game = ChessGame(config)
    game.run()
    game.metrics_store.close()