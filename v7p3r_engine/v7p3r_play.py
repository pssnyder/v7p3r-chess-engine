# v7p3r_engine/play_v7p3r.py

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pygame
import chess
import chess.pgn
import random
import yaml
import datetime
import logging
import socket
from typing import Optional
import time # Import time for measuring move duration
from io import StringIO
import hashlib

config = {
    "game_config": {
        "game_count": 1,                     # Number of games to play
        "starting_position": "default",      # Default starting position name (or FEN string)
        "white_player": "v7p3r",             # Name of the engine being used (e.g., 'v7p3r', 'stockfish'), this value is a direct reference to the engine configuration values in their respective config files
        "black_player": "stockfish",         # sets this colors engine configuration name, same as above, important note that if the engines are set the same then only whites metrics will be collected to prevent negation in win loss metrics
    },
    "engine_config": {
        "name": "v7p3r",                     # Name of the engine, used for identification and logging
        "version": "1.0.0",                  # Version of the engine, used for identification and logging
        "color": "white",                    # Color of the engine, either 'white' or 'black'
        "ruleset": "default_evaluation",    # Name of the evaluation rule set to use, see below for available options
        "search_algorithm": "minimax",       # Move search type for White (see search_algorithms for options)
        "depth": 3,                          # Depth of search for AI, 1 for random, 2 for simple search, 3+ for more complex searches
        "max_depth": 4,                      # Max depth of search for AI, 1 for random, 2 for simple search, 3+ for more complex searches
        "use_game_phase": False,             # Use game phase evaluation
        "monitoring_enabled": True,          # Enable or disable monitoring features
        "verbose_output": True,             # Enable or disable verbose output for debugging
        "logger": "v7p3r_engine_logger",     # Logger name for the engine, used for logging engine-specific events
    },
    "stockfish_config": {
        "stockfish_path": "engine_utilities/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe",
        "elo_rating": 100,
        "skill_level": 1,
        "debug_mode": False,
        "depth": 2,
        "max_depth": 2,
        "movetime": 500,  # Time in milliseconds for Stockfish to think
    },
}

# Define the maximum frames per second for the game loop
MAX_FPS = 60
# Import necessary modules from v7p3r_engine
from v7p3r_engine.v7p3r import v7p3rEngine # Corrected import for v7p3rEngine
from metrics.metrics_store import MetricsStore # Import MetricsStore
from v7p3r_engine.stockfish_handler import StockfishHandler

# Set engine availability values
RL_ENGINE_AVAILABLE = False
GA_ENGINE_AVAILABLE = False
NN_ENGINE_AVAILABLE = False

# At module level, define a single logger for this file
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def get_log_file_path():
    # Use a timestamped log file for each game session
    timestamp = get_timestamp()
    log_dir = "logging"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"chess_game_{timestamp}.log")

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
        # Initialize Pygame (even in headless mode, for internal timing)
        pygame.init()
        self.clock = pygame.time.Clock()

        # Enable logging
        self.logger = chess_game_logger
        
        # Initialize Engines
        self.engine_config = config.get("engine_config", {})
        self.engine = v7p3rEngine(self.engine_config)
        self.stockfish_config = config.get("stockfish_config", {})
        self.stockfish = StockfishHandler(self.stockfish_config)
        
        # Set logging level
        self.monitoring_enabled = self.engine_config.get("monitoring_enabled", True)
        self.verbose_output_enabled = self.engine_config.get("verbose_output", True)

        # Initialize new engines based on availability and configuration
        self.engines = {
            'v7p3r': self.engine,
            'stockfish': self.stockfish
        }
        
        # Game Settings
        self.game_config = config.get("game_config", {})
        self.game_count = self.game_config.get("game_count", 1)
        self.white_player = self.game_config.get("white_player", "v7p3r")
        self.black_player = self.game_config.get("black_player", "stockfish")
        self.starting_position = self.game_config.get("starting_position", "default")

        # Access global engine availability variables
        global RL_ENGINE_AVAILABLE, GA_ENGINE_AVAILABLE, NN_ENGINE_AVAILABLE

        if 'v7p3r_rl' in self.engines:
            # Import new engines
            try:
                from v7p3r_rl_engine.v7p3r_rl import v7p3rRLEngine
                RL_ENGINE_AVAILABLE = True
            except ImportError:
                RL_ENGINE_AVAILABLE = False
                print("Warning: v7p3r_rl_engine not available")
        
        if 'v7p3r_ga' in self.engines:
            try:
                from v7p3r_ga_engine.v7p3r_ga import v7p3rGeneticAlgorithm
                GA_ENGINE_AVAILABLE = True
            except ImportError:
                GA_ENGINE_AVAILABLE = False
                print("Warning: v7p3r_ga_engine not available")

        if 'v7p3r_nn' in self.engines:
            try:
                from v7p3r_nn_engine.v7p3r_nn import v7p3rNeuralNetwork
                NN_ENGINE_AVAILABLE = True
            except ImportError:
                NN_ENGINE_AVAILABLE = False
                print("Warning: v7p3r_nn_engine not available")

        # Initialize RL engine if available
        if RL_ENGINE_AVAILABLE and 'v7p3r_rl' in self.engines:
            try:
                rl_config_path = config.get('rl_config_path', 'config/v7p3r_rl_config.yaml')
                self.rl_engine = v7p3rRLEngine(rl_config_path)
                self.engines['v7p3r_rl'] = self.rl_engine
                print("✓ v7p3r RL engine initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize RL engine: {e}")
        
        # Initialize GA engine if available
        if GA_ENGINE_AVAILABLE and 'v7p3r_ga' in self.engines:
            try:
                from v7p3r_ga_engine.v7p3r_ga import v7p3rGeneticAlgorithm
                ga_config_path = config.get('ga_config_path', 'config/v7p3r_ga_config.yaml')
                self.ga_engine = v7p3rGeneticAlgorithm(ga_config_path)
                self.engines['v7p3r_ga'] = self.ga_engine
                print("✓ v7p3r GA engine initialized for gameplay")
            except Exception as e:
                print(f"Warning: Failed to initialize GA engine: {e}")
        
        # Initialize NN engine if available
        if NN_ENGINE_AVAILABLE and 'v7p3r_nn' in self.engines:
            try:
                nn_config_path = config.get('nn_config_path', 'config/v7p3r_nn_config.yaml')
                self.nn_engine = self._create_nn_engine_wrapper(nn_config_path)
                if self.nn_engine:
                    self.engines['v7p3r_nn'] = self.nn_engine
                    print("✓ v7p3r NN engine wrapper initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize NN engine: {e}")

        # Initialize MetricsStore
        self.metrics_store = MetricsStore()
        self._move_metrics_batch = []  # in-memory store of move metrics
        self.game_start_timestamp = get_timestamp()
        self.current_game_db_id = f"eval_game_{self.game_start_timestamp}.pgn"
        
        # Initialize board and new game
        self.new_game()
    
    def new_game(self):
        """Reset the game state for a new game"""
        self.board = chess.Board()
        if self.starting_position != "default":
            self.board.set_fen(self.starting_position)
        self.game = chess.pgn.Game()
        self.game_node = self.game
        self.selected_square = chess.SQUARES[0]
        self.last_engine_move = chess.Move.null()
        self.current_eval = 0.0
        self.current_player = chess.WHITE if self.board.turn else chess.BLACK
        self.last_move = chess.Move.null()
        self.move_history = []
        self.move_start_time = 0
        self.move_end_time = 0
        self.move_duration = 0

        # Reset PGN headers and file
        self.set_headers()
        self.quick_save_pgn("logging/active_game.pgn")
        
        if self.logger and self.monitoring_enabled:
            self.logger.info(f"Starting new game: {self.current_game_db_id}.")

    def set_headers(self):
        self.game.headers["Event"] = "v7p3r Engine Chess Game"
        self.game.headers["White"] = f"{self.white_player}"
        self.game.headers["Black"] = f"{self.black_player}"
        self.game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        self.game.headers["Site"] = socket.gethostbyname(socket.gethostname())
        self.game.headers["Round"] = "#"

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
        """Check if the game is over and handle end conditions."""
        if self.board.is_game_over():
            # Ensure the result is set in the PGN headers and game node
            result = self.get_board_result()
            self.game.headers["Result"] = result
            self.game_node = self.game.end()
            self.save_game_data()
            print(f"\nGame over: {result}")
            return True
        return False
    
    def record_evaluation(self):
        """Record evaluation score in PGN comments"""
        score = self.engine.scoring_calculator.evaluate_position(self.board)
        self.current_eval = score
        self.game_node.comment = f"Eval: {score:.2f}"
            
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
        if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
            self.logger.info(f"Game PGN saved to {pgn_filepath}")

        # Save YAML config file for metrics processing
        config_filepath = f"games/eval_game_{timestamp}.yaml"
        
        # Save a combined config for this specific game, including relevant parts of all loaded configs
        game_specific_config = {
            "game_settings": self.game_config,
            "engine_settings": self.engine.engine_config,
        }
        with open(config_filepath, "w") as f:
            yaml.dump(game_specific_config, f)
        if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
            self.logger.info(f"Game-specific combined configuration saved to {config_filepath}")

        # Save the game result to a file for instant analysis
        self.quick_save_pgn(f"games/{game_id}.pgn")

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
            if self.logger and self.monitoring_enabled:
                self.logger.error(f"[Error] Failed to save PGN to {filename}: {e}")

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
                'engine_config': self.engine.engine_config,
            }
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.debug(f"Saved local files: {pgn_path}, {yaml_path}")
                
        except Exception as e:
            if self.logger and self.monitoring_enabled:
                self.logger.error(f"[Error] Failed to save local game files for {game_id}: {e}")

    def import_fen(self, fen_string):
        """Import a position from FEN notation"""
        try:
            new_board = chess.Board(fen_string)
            
            if not new_board.is_valid():
                if self.logger and self.monitoring_enabled:
                    self.logger.error(f"[Error] Invalid FEN position: {fen_string}")
                return False

            self.board = new_board
            self.game = chess.pgn.Game()
            self.game.setup(new_board)
            self.game_node = self.game            
            self.selected_square = None
            self.game.headers["FEN"] = fen_string

            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"Successfully imported FEN: {fen_string}")
            return True

        except Exception as e:
            if self.logger and self.monitoring_enabled:
                self.logger.error(f"[Error] Unexpected problem importing FEN: {e}")
            return False

    # ===================================
    # ========= MOVE HANDLERS ===========
    
    def process_engine_move(self):
        """Process move for the engine"""
        engine_move = chess.Move.null()
        self.current_eval = 0.0
        current_player_color = chess.WHITE if self.board.turn else chess.BLACK
        self.move_start_time = time.time()  # Start timing the move
        self.move_end_time = 0
        self.move_duration = 0
        if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
            self.logger.info(f"Processing move for {self.white_player if current_player_color == chess.WHITE else self.black_player} using {self.engine.name} engine.")
        print(f"{self.white_player if current_player_color == chess.WHITE else self.black_player} is thinking...")

        # Send the move request to the appropriate engine or human interface
        if (self.white_player.lower() == 'human' and self.board.turn) or (self.black_player.lower() == 'human' and not self.board.turn):
            # Handle human player input (not implemented here, placeholder)
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info("Waiting for human player input...")
            return
        else:
            # Determine current player engine
            current_engine_name = self.white_player.lower() if self.board.turn else self.black_player.lower()
            
            # Handle different engine types
            if current_engine_name == 'v7p3r':
                try:
                    # Use the v7p3r engine for the current player
                    engine_move = self.engine.search_engine.search(self.board, current_player_color)
                except Exception as e:
                    if self.logger and self.monitoring_enabled:
                        self.logger.error(f"[HARDSTOP Error] Cannot find move via v7p3rSearch: {e}. | FEN: {self.board.fen()}")
                    print(f"HARDSTOP ERROR: Cannot find move via v7p3rSearch: {e}. | FEN: {self.board.fen()}")
                    return
                    
            elif current_engine_name == 'stockfish':
                try:
                    # Use the Stockfish engine for the current player
                    if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                        self.logger.info("Using Stockfish engine for move processing.")
                    stockfish_handler = StockfishHandler(self.stockfish_config)
                    engine_move = stockfish_handler.search(self.board, current_player_color, self.stockfish_config)
                except Exception as e:
                    if self.logger and self.monitoring_enabled:
                        self.logger.error(f"[HARDSTOP Error] Cannot find move via Stockfish: {e}. | FEN: {self.board.fen()}")
                    print(f"HARDSTOP ERROR: Cannot find move via Stockfish: {e}. | FEN: {self.board.fen()}")
                    return
            
            elif current_engine_name == 'v7p3r_rl' and 'v7p3r_rl' in self.engines:
                try:
                    # Use the RL engine for the current player
                    if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                        self.logger.info("Using v7p3r RL engine for move processing.")
                    engine_move = self.engines['v7p3r_rl'].search(self.board, current_player_color)
                except Exception as e:
                    if self.logger and self.monitoring_enabled:
                        self.logger.error(f"[HARDSTOP Error] Cannot find move via v7p3rReinforcementLearning: {e}. | FEN: {self.board.fen()}")
                    print(f"HARDSTOP ERROR: Cannot find move via v7p3rReinforcementLearning: {e}. | FEN: {self.board.fen()}")
                    return
                    
            elif current_engine_name == 'v7p3r_ga' and 'v7p3r_ga' in self.engines:
                try:
                    # Use the GA engine
                    if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                        self.logger.info("Using v7p3r GA engine for move processing.")
                    engine_move = self.engines['v7p3r_ga'].search(self.board, current_player_color)
                except Exception as e:
                    if self.logger and self.monitoring_enabled:
                        self.logger.error(f"[HARDSTOP Error] Cannot find move via v7p3rGeneticAlgorithm: {e}. | FEN: {self.board.fen()}")
                    print(f"HARDSTOP ERROR: Cannot find move via v7p3r_ga: {e}. | FEN: {self.board.fen()}")
                    return
                    
            elif current_engine_name == 'v7p3r_nn' and 'v7p3r_nn' in self.engines:
                try:
                    # Use the NN engine
                    if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                        self.logger.info("Using v7p3r NN engine for move processing.")
                    engine_move = self.engines['v7p3r_nn'].search(self.board, current_player_color)
                except Exception as e:
                    if self.logger and self.monitoring_enabled:
                        self.logger.error(f"[HARDSTOP Error] Cannot find move via v7p3rNeuralNetwork: {e}. | FEN: {self.board.fen()}")
                    print(f"HARDSTOP ERROR: Cannot find move via v7p3rNeuralNetwork: {e}. | FEN: {self.board.fen()}")
            else:
                if self.logger and self.monitoring_enabled:
                    self.logger.error(f"[HARDSTOP Error] No valid engine in configuration: {e}. | FEN: {self.board.fen()}")
                print(f"HARDSTOP ERROR: No valid engine in configuration: {e}. | FEN: {self.board.fen()}")
                return

            # Check and Push the move
            if not isinstance(engine_move, chess.Move):
                return # Move invalid
            try:
                if self.board.is_legal(engine_move):
                    fen_before_move = self.board.fen()
                    move_number = self.board.fullmove_number
                    self.push_move(engine_move)
                    self.last_engine_move = engine_move
                    self.move_end_time = time.time()  # End timing the move
                    self.move_duration = self.move_end_time - self.move_start_time
                    self.pv_line = ""
            except Exception as e:
                if self.logger and self.monitoring_enabled:
                    self.logger.error(f"[HARDSTOP Error] Move Invalid: {e}. | Move: {engine_move} | FEN: {self.board.fen()}")
                print(f"HARDSTOP ERROR: Move Invalid: {e}. | Move: {engine_move} | FEN: {self.board.fen()}")
                return
            
            try: 
                # Ensure all move metric fields are present
                metric = {
                    'game_id': self.current_game_db_id,
                    'move_number': move_number,
                    'player_color': 'w' if current_player_color == chess.WHITE else 'b',
                    'move_uci': engine_move.uci(),
                    'fen_before': fen_before_move,
                    'evaluation': self.current_eval,
                    'search_algorithm': self.engine.search_algorithm,
                    'depth': self.engine.depth,
                    'engine_id': hashlib.md5(self.engine.name.encode()).hexdigest(),
                    'engine_name': self.engine.name,
                    'engine_version': self.engine.version,
                    'nodes_searched': self.engine.search_engine.nodes_searched,
                    'time_taken': self.move_duration,
                    'pv_line': self.pv_line
                }
                # Add metrics record
                self.metrics_store.add_move_metric(**metric)
                self._move_metrics_batch.append(metric)
                
                if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                    self.logger.debug(f"Move metrics for {engine_move} added to MetricsStore.")
            except Exception as e:
                if self.logger and self.monitoring_enabled:
                    self.logger.error(f"[HARDSTOP Error] Move Invalid: {e}. | Move: {engine_move} | FEN: {self.board.fen()}")
                print(f"HARDSTOP ERROR: Move Invalid: {e}. | Move: {engine_move} | FEN: {self.board.fen()}")
                return
    
            print(f"{self.white_player if current_player_color == chess.WHITE else self.black_player} played: {engine_move} after {self.move_duration:.4f}s (Eval: {self.current_eval:.2f})")
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                nodes_searched = self.engine.search_engine.nodes_searched
                self.logger.info(f"{self.white_player if current_player_color == chess.WHITE else self.black_player} played: {engine_move} (Eval: {self.current_eval:.2f}) | Time: {self.move_duration:.4f}s | Nodes: {nodes_searched}")

    def push_move(self, move):
        """ Test and push a move to the board and game node """
        if not self.board.is_valid():
            if self.logger and self.monitoring_enabled:
                self.logger.error(f"[Error] Invalid board state detected! | FEN: {self.board.fen()}")
            return False
        
        if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
            self.logger.info(f"Attempting to push move from {'White Engine' if self.board.turn == chess.WHITE else 'Black Engine'}: {move} | FEN: {self.board.fen()}")

        if isinstance(move, str):
            try:
                move = chess.Move.from_uci(move)
                if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                    self.logger.info(f"Converted to chess.Move move from UCI string before push: {move}")
            except ValueError:
                if self.logger and self.monitoring_enabled:
                    self.logger.error(f"[Error] Invalid UCI string received: {move}")
                return False
        
        if not self.board.is_legal(move):
            if self.logger and self.monitoring_enabled:
                self.logger.info(f"[Error] Illegal move blocked from being pushed: {move}")
            return False
        
        try:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"No remaining checks, pushing move: {move} | FEN: {self.board.fen()}")
            
            self.board.push(move)
            self.game_node = self.game_node.add_variation(move)
            self.last_move = move
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"Move pushed successfully: {move} | FEN: {self.board.fen()}")
            
            self.current_player = chess.WHITE if self.board.turn else chess.BLACK
            
            if self.engine.name.lower() == 'v7p3r':
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
            if self.logger and self.monitoring_enabled:
                self.logger.error(f"[Error] ValueError pushing move {move}: {e}. Dumping PGN to error_dump.pgn")
            self.quick_save_pgn("games/game_error_dump.pgn")
            return False
            
        except Exception as e:
            print(f"Failed to create GA engine wrapper: {e}")
            return None
    
    def _create_nn_engine_wrapper(self, nn_config_path):
        """Create a wrapper for NN engine if it doesn't have proper game interface."""
        try:
            # Import the NN engine class here to avoid undefined reference
            from v7p3r_nn_engine.v7p3r_nn import v7p3rNeuralNetwork
            
            # Try to create NN engine, but it might not have search interface
            nn_engine = v7p3rNeuralNetwork(nn_config_path)
            
            # Check if it has search method
            if hasattr(nn_engine, 'search'):
                return nn_engine
            else:
                # Create a wrapper
                class NNEngineWrapper:
                    def __init__(self, nn_engine):
                        self.nn_engine = nn_engine
                        self.v7p3r_engine = v7p3rEngine()
                    
                    def search(self, board, player_color, engine_config=None):
                        """Search using NN-assisted evaluation."""
                        # Fallback to v7p3r engine for now
                        return self.v7p3r_engine.search_engine.search(board, player_color)
                    
                    def cleanup(self):
                        """Cleanup resources."""
                        if hasattr(self.nn_engine, 'cleanup'):
                            self.nn_engine.cleanup()
                
                return NNEngineWrapper(nn_engine)
                
        except Exception as e:
            print(f"Failed to create NN engine wrapper: {e}")
            return None

    # =============================================
    # ============ MAIN GAME LOOP =================

    def run(self, debug_mode: Optional[bool] = False):
        running = True
        game_count_remaining = self.game_count
        
        print(f"White: {self.white_player} vs Black: {self.black_player}")
        if self.logger and self.monitoring_enabled:
            self.logger.info(f"White: {self.white_player} vs Black: {self.black_player}")

        while running and game_count_remaining >= 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if not self.board.is_game_over() and self.board.is_valid():
                self.process_engine_move()
            else:
                if self.handle_game_end():
                    game_count_remaining -= 1
                    if game_count_remaining == 0:
                        running = False
                        if self.logger and self.monitoring_enabled:
                            self.logger.info(f'All {self.game_count} games complete, exiting...')
                        print(f'All {self.game_count} games complete, exiting...')
                    else:
                        if self.logger and self.monitoring_enabled:
                            self.logger.info(f'Game {self.game_count - game_count_remaining}/{self.game_count} complete, starting next...')
                        print(f'Game {self.game_count - game_count_remaining}/{self.game_count} complete, starting next...')
                        self.game_start_timestamp = get_timestamp()
                        self.current_game_db_id = f"eval_game_{self.game_start_timestamp}.pgn"
                        self.new_game()

            # Update the game state and render if necessary
            self.clock.tick(MAX_FPS)

            # Debug mode: pause the game state for examination
            if debug_mode:
                print("Debug mode active. Press any key to continue...")
                pygame.event.wait()    

        # Cleanup engines and resources
        self.cleanup_engines()
        if pygame.get_init():
            pygame.quit()

    def cleanup_engines(self):
        """Clean up engine resources."""
        try:
            # Cleanup RL engine
            if hasattr(self, 'rl_engine') and self.rl_engine:
                self.rl_engine.cleanup()
                
            # Cleanup GA engine
            if hasattr(self, 'ga_engine') and self.ga_engine and hasattr(self.ga_engine, 'cleanup'):
                self.ga_engine.cleanup()
                
            # Cleanup NN engine
            if hasattr(self, 'nn_engine') and self.nn_engine and hasattr(self.nn_engine, 'cleanup'):
                self.nn_engine.cleanup()
                
            # Cleanup Stockfish
            if hasattr(self, 'stockfish') and self.stockfish:
                self.stockfish.quit()
                
        except Exception as e:
            if self.logger and self.monitoring_enabled:
                self.logger.error(f"[Error] Failed during engine cleanup: {e}")
            else:
                print(f"Error during engine cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup_engines()
        except:
            pass

if __name__ == "__main__":
    game = ChessGame(config)
    game.run()
    game.metrics_store.close()
