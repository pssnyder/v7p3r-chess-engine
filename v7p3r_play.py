# v7p3r_play.py

import os
import sys
import pygame
import chess
import chess.pgn
import datetime
import socket
from typing import Optional
import time
from io import StringIO
from v7p3r_config import v7p3rConfig
from v7p3r_debug import v7p3rLogger, v7p3rUtilities
from metrics.v7p3r_chess_metrics import get_metrics_instance, GameMetric

CONFIG_NAME = "custom_config"

# Define the maximum frames per second for the game loop
MAX_FPS = 60

# Set engine availability values
RL_ENGINE_AVAILABLE = False
GA_ENGINE_AVAILABLE = False
NN_ENGINE_AVAILABLE = False

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup centralized logging for this module
v7p3r_play_logger = v7p3rLogger.setup_logger("v7p3r_play")

# Import necessary modules from v7p3r_engine
from v7p3r import v7p3rEngine # Corrected import for v7p3rEngine
from v7p3r_stockfish_handler import StockfishHandler

class v7p3rChess:
    def __init__(self, config_name: Optional[str] = None):
        """
        config: ChessGameConfig object containing all configuration parameters.
        """
        # Initialize Pygame (even in headless mode, for internal timing)
        pygame.init()
        self.clock = pygame.time.Clock()

        # Enable logging
        self.logger = v7p3r_play_logger

        # Load configuration first
        try:
            if config_name is None:
                self.config_manager = v7p3rConfig()
                self.config = self.config_manager.get_config()
                self.game_config = self.config_manager.get_game_config()
                self.engine_config = self.config_manager.get_engine_config()
                self.stockfish_config = self.config_manager.get_stockfish_config()
                self.puzzle_config = self.config_manager.get_puzzle_config()
                if self.logger:
                    self.logger.info("No configuration provided, using default v7p3r configuration.")
            else:
                self.config_name = config_name
                self.config_manager = v7p3rConfig(config_path=os.path.join('configs',f"{self.config_name}.json"))
                self.game_config = self.config_manager.get_game_config()
                self.engine_config = self.config_manager.get_engine_config()
                self.stockfish_config = self.config_manager.get_stockfish_config()
                if self.logger:
                    self.logger.info("Configuration provided, using custom settings.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading configuration: {e}")
            raise

        # Set logging and output levels after config is loaded
        self.monitoring_enabled = self.engine_config.get("monitoring_enabled", True)
        self.verbose_output_enabled = self.engine_config.get("verbose_output", False)

        # Initialize Engines
        self.engine = v7p3rEngine()
        
        # Initialize metrics system
        self.metrics = get_metrics_instance()
        self.current_game_id = None
        if self.logger:
            self.logger.info("Metrics system initialized")
        
        # Debug stockfish config before passing to handler
        if self.logger:
            self.logger.info(f"Stockfish config being passed to handler: {self.stockfish_config}")
            self.logger.info(f"Stockfish path in config: {self.stockfish_config.get('stockfish_path', 'NOT_FOUND')}")
        
        try:
            self.stockfish = StockfishHandler(stockfish_config=self.stockfish_config)
            if self.logger:
                self.logger.info("StockfishHandler initialized successfully")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize StockfishHandler: {e}")
            print(f"ERROR: Failed to initialize Stockfish: {e}")
            if self.verbose_output_enabled:
                print("Falling back to DummyStockfish for testing purposes")
            # Create a dummy stockfish handler that always returns null moves
            class DummyStockfish:
                def __init__(self):
                    self.process = None
                    self.nodes_searched = 0
                    self.last_search_info = {}
                
                def search(self, board, player, config):
                    return chess.Move.null()
                
                def cleanup(self):
                    pass
                    
                def quit(self):
                    pass
                    
                def get_last_search_info(self):
                    return {}
            
            self.stockfish = DummyStockfish()
        
        # Check if stockfish process is running
        if hasattr(self.stockfish, 'process') and self.stockfish.process is None:
            if self.verbose_output_enabled:
                print("WARNING: Stockfish process is None - engine failed to start!")

        # Initialize new engines based on availability and configuration
        self.engines = {
            'v7p3r': self.engine,
            'stockfish': self.stockfish
        }
        
        # Game Settings
        self.game_count = self.game_config.get("game_count", 1)
        self.white_player = self.game_config.get("white_player", "v7p3r")
        self.black_player = self.game_config.get("black_player", "stockfish")
        self.starting_position = self.game_config.get("starting_position", "default")
        
        # Prepare engine configurations for metrics
        self.white_engine_config = self._get_engine_config_for_player(self.white_player)
        self.black_engine_config = self._get_engine_config_for_player(self.black_player)

        # Access global engine availability variables
        global RL_ENGINE_AVAILABLE, GA_ENGINE_AVAILABLE, NN_ENGINE_AVAILABLE

        if 'v7p3r_rl' in self.engines:
            # Import new engines
            try:
                from v7p3r_rl import v7p3rRLEngine
                RL_ENGINE_AVAILABLE = True
            except ImportError:
                RL_ENGINE_AVAILABLE = False
                if self.verbose_output_enabled:
                    print("Warning: v7p3r_rl_engine not available")
        
        if 'v7p3r_ga' in self.engines:
            try:
                from v7p3r_ga import v7p3rGeneticAlgorithm
                GA_ENGINE_AVAILABLE = True
            except ImportError:
                GA_ENGINE_AVAILABLE = False
                if self.verbose_output_enabled:
                    print("Warning: v7p3r_ga_engine not available")

        if 'v7p3r_nn' in self.engines:
            try:
                from v7p3r_nn import v7p3rNeuralNetwork
                NN_ENGINE_AVAILABLE = True
            except ImportError:
                NN_ENGINE_AVAILABLE = False
                if self.verbose_output_enabled:
                    print("Warning: v7p3r_nn_engine not available")

        # Initialize RL engine if available
        if RL_ENGINE_AVAILABLE and 'v7p3r_rl' in self.engines:
            try:
                # Use centralized config instead of separate config file
                self.rl_engine = v7p3rRLEngine(self.config_manager)
                self.engines['v7p3r_rl'] = self.rl_engine
                if self.verbose_output_enabled:
                    print("✓ v7p3r RL engine initialized")
            except Exception as e:
                if self.verbose_output_enabled:
                    print(f"Warning: Failed to initialize RL engine: {e}")
        
        # Initialize GA engine if available
        if GA_ENGINE_AVAILABLE and 'v7p3r_ga' in self.engines:
            try:
                from v7p3r_ga import v7p3rGeneticAlgorithm
                # Use centralized config instead of separate config file
                self.ga_engine = v7p3rGeneticAlgorithm(self.config_manager)
                self.engines['v7p3r_ga'] = self.ga_engine
                if self.verbose_output_enabled:
                    print("✓ v7p3r GA engine initialized for gameplay")
            except Exception as e:
                if self.verbose_output_enabled:
                    print(f"Warning: Failed to initialize GA engine: {e}")
        
        # Initialize NN engine if available
        if NN_ENGINE_AVAILABLE and 'v7p3r_nn' in self.engines:
            try:
                # Use centralized config instead of separate config file
                self.nn_engine = self._create_nn_engine_wrapper(self.config_manager)
                if self.nn_engine:
                    self.engines['v7p3r_nn'] = self.nn_engine
                    if self.verbose_output_enabled:
                        print("✓ v7p3r NN engine wrapper initialized")
            except Exception as e:
                if self.verbose_output_enabled:
                    print(f"Warning: Failed to initialize NN engine: {e}")

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
        self.current_player = self.board.turn
        self.last_move = chess.Move.null()
        self.move_history = []
        self.move_start_time = 0
        self.move_end_time = 0
        self.move_duration = 0
        self.game_start_time = time.time()  # Track overall game timing
        self.game_start_timestamp = v7p3rUtilities.get_timestamp()

        # Reset PGN headers and file
        self.set_headers()
        self.quick_save_pgn("active_game.pgn")
        
        # Initialize metrics for new game
        self.current_game_id = f"eval_game_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Record game start in metrics (use a safer approach for async)
        try:
            game_metric = GameMetric(
                game_id=self.current_game_id,
                timestamp=datetime.datetime.now().isoformat(),
                v7p3r_color='white' if self.white_player == 'v7p3r' else 'black',
                opponent=self.black_player if self.white_player == 'v7p3r' else self.white_player,
                result='pending',
                total_moves=0,
                game_duration=0.0
            )
            # Use legacy compatibility function to avoid async issues
            from metrics.v7p3r_chess_metrics import add_game_result
            add_game_result(
                game_id=self.current_game_id,
                timestamp=datetime.datetime.now().isoformat(),
                winner='pending',
                game_pgn='',
                white_player=self.white_player,
                black_player=self.black_player,
                game_length=0,
                white_engine_config=str(self.white_engine_config),
                black_engine_config=str(self.black_engine_config)
            )
            if self.logger:
                self.logger.info(f"Game metrics initialized for: {self.current_game_id}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to initialize game metrics: {e}")
        
        if self.monitoring_enabled and self.logger:
            self.logger.info(f"Starting new game: {self.current_game_id}.")

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
        # Use standard white perspective evaluation (white_score - black_score)
        score = self.engine.scoring_calculator.evaluate_position(self.board)
        self.current_eval = score
        self.game_node.comment = f"Eval: {score:+.2f}"
        
        # Display move with evaluation if verbose output is enabled
        if self.verbose_output_enabled:
            current_player = self.board.turn
            player_name = "White" if current_player == chess.WHITE else "Black"
            print(f"Position evaluation: {score:+.2f} (White perspective)")
        
        if self.monitoring_enabled and self.logger:
            self.logger.info(f"Evaluation recorded: {score:+.2f} (White perspective)")
            
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
        
        # Update game completion metrics
        if self.current_game_id:
            try:
                # Calculate final game metrics
                game_duration = time.time() - self.game_start_time
                total_moves = len(list(self.game.mainline_moves()))
                
                # Update database with final game result
                from metrics.v7p3r_chess_metrics import get_metrics_instance
                metrics = get_metrics_instance()
                
                # Determine winner for database format
                if result == "1-0":
                    db_result = "white_win"
                elif result == "0-1":
                    db_result = "black_win"  
                elif result == "1/2-1/2":
                    db_result = "draw"
                else:
                    db_result = "incomplete"
                
                # Get final position
                final_fen = self.board.fen()
                
                # Get termination reason
                if self.board.is_checkmate():
                    termination = "checkmate"
                elif self.board.is_stalemate():
                    termination = "stalemate"
                elif self.board.is_insufficient_material():
                    termination = "insufficient_material"
                elif self.board.is_seventyfive_moves():
                    termination = "75_move_rule"
                elif self.board.is_fivefold_repetition():
                    termination = "5_fold_repetition"
                else:
                    termination = "normal"
                
                # Update game result in database
                try:
                    import sqlite3
                    with sqlite3.connect(metrics.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE games 
                            SET result = ?, total_moves = ?, game_duration = ?,
                                final_position_fen = ?, termination_reason = ?
                            WHERE game_id = ?
                        """, (db_result, total_moves, game_duration, final_fen, termination, self.current_game_id))
                        conn.commit()
                    if self.monitoring_enabled and self.logger:
                        self.logger.info(f"Game {self.current_game_id} result updated in database: {db_result}")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to update game result in database: {e}")
                
                # Log completion
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Game {self.current_game_id} completed: {result} in {total_moves} moves ({game_duration:.1f}s)")
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to record game completion metrics: {e}")
                # Fallback: Save JSON config file for metrics processing
                config_filepath = f"games/eval_game_{timestamp}.json"
                
                # Save a combined config for this specific game, including relevant parts of all loaded configs
                game_specific_config = {
                    "game_settings": self.game_config,
                    "engine_settings": self.engine.engine_config,
                    "stockfish_settings": self.stockfish_config,
                    "white_player": self.white_player,
                    "black_player": self.black_player,
                    "game_id": game_id,
                    "ruleset": self.config_manager.ruleset
                }
                with open(config_filepath, "w") as f:
                    import json
                    json.dump(game_specific_config, f, indent=4)
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Game-specific combined configuration saved to {config_filepath}")
        
            # Save locally into pgn file
            pgn_filepath = f"games/eval_game_{timestamp}.pgn"
            with open(pgn_filepath, "w") as f:
                exporter = chess.pgn.FileExporter(f)
                self.game.accept(exporter)
                # Always append the result string at the end for compatibility
                if result != "*":
                    f.write(f"\n{result}\n")
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"Game PGN saved to {pgn_filepath}")
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
            with open(filename, 'w', encoding='utf-8') as f:
                # Get PGN text
                buf = StringIO()
                exporter = chess.pgn.FileExporter(buf)
                self.game.accept(exporter)
                f.write(buf.getvalue())
        except Exception as e:
            if self.monitoring_enabled and self.logger:
                self.logger.error(f"[Error] Failed to save PGN to {filename}: {e}")

    def save_local_game_files(self, game_id, pgn_text):
        """Save both PGN and JSON config files locally for metrics processing."""
        try:
            import os
            import json
            
            # Ensure games directory exists
            os.makedirs("games", exist_ok=True)
            
            # Save PGN file
            pgn_path = f"games/{game_id}.pgn"
            with open(pgn_path, 'w', encoding='utf-8') as f:
                f.write(pgn_text)
            
            # Save JSON config file for metrics processing
            json_path = f"games/{game_id}.json"
            config_data = {
                'game_id': game_id,
                'timestamp': self.game_start_timestamp,
                'engine_config': self.engine.engine_config,
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4)
            
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"Saved local files: {pgn_path}, {json_path}")
                
        except Exception as e:
            if self.monitoring_enabled and self.logger:
                self.logger.error(f"[Error] Failed to save local game files for {game_id}: {e}")

    def import_fen(self, fen_string):
        """Import a position from FEN notation"""
        try:
            new_board = chess.Board(fen_string)
            
            if not new_board.is_valid():
                if self.monitoring_enabled and self.logger:
                    self.logger.error(f"[Error] Invalid FEN position: {fen_string}")
                return False

            self.board = new_board
            self.game = chess.pgn.Game()
            self.game.setup(new_board)
            self.game_node = self.game            
            self.selected_square = None
            self.game.headers["FEN"] = fen_string

            if self.monitoring_enabled and self.logger:
                self.logger.info(f"Successfully imported FEN: {fen_string}")
            return True

        except Exception as e:
            if self.monitoring_enabled and self.logger:
                self.logger.error(f"[Error] Unexpected problem importing FEN: {e}")
            return False

    # ===================================
    # ========= MOVE HANDLERS ===========
    
    def process_engine_move(self):
        """Process move for the engine"""
        engine_move = chess.Move.null()
        self.current_eval = 0.0
        self.current_player = self.board.turn
        self.move_start_time = time.time()  # Start timing the move
        self.move_end_time = 0
        self.move_duration = 0
        if self.monitoring_enabled and self.logger:
            self.logger.info(f"Processing move for {self.white_player if self.current_player == chess.WHITE else self.black_player} using {self.engine.name} engine.")
        print(f"{self.white_player if self.current_player == chess.WHITE else self.black_player} is thinking...")
        
        if self.monitoring_enabled and self.logger:
            current_player_name = "White" if self.current_player == chess.WHITE else "Black"
            self.logger.info(f"{current_player_name} ({self.white_player if self.current_player == chess.WHITE else self.black_player}) is thinking...")

        # Send the move request to the appropriate engine or human interface
        if (self.white_player.lower() == 'human' and self.current_player == chess.WHITE) or (self.black_player.lower() == 'human' and self.current_player == chess.BLACK):
            # Handle human player input (not implemented here, placeholder)
            if self.monitoring_enabled and self.logger:
                self.logger.info("Waiting for human player input...")
            return
        else:
            # Determine current player engine
            current_engine_name = self.white_player.lower() if self.current_player == chess.WHITE else self.black_player.lower()

            # Handle different engine types
            if current_engine_name == 'v7p3r':
                try:
                    # Debug logging to track the issue (always log this)
                    if self.monitoring_enabled and self.logger:
                        self.logger.info(f"Engine type: {type(self.engine)}")
                        self.logger.info(f"Search engine type: {type(self.engine.search_engine)}")
                        self.logger.info(f"Search method type: {type(self.engine.search_engine.search)}")
                        self.logger.info(f"Search method is callable: {callable(getattr(self.engine.search_engine, 'search', None))}")
                        if hasattr(self.engine.search_engine, 'search'):
                            search_attr = getattr(self.engine.search_engine, 'search')
                            if isinstance(search_attr, dict):
                                self.logger.error(f"FOUND THE ISSUE! search attribute is a dict: {search_attr}")
                            self.logger.info(f"Search attribute: {search_attr}")
                    
                    # Use the v7p3r engine for the current player
                    engine_move = self.engine.search_engine.search(self.board, self.current_player)
                except Exception as e:
                    if self.monitoring_enabled and self.logger:
                        self.logger.error(f"[HARDSTOP Error] Cannot find move via v7p3rSearch: {e}. | FEN: {self.board.fen()}")
                        self.logger.error(f"Engine type: {type(self.engine)}")
                        self.logger.error(f"Search engine type: {type(self.engine.search_engine)}")
                        self.logger.error(f"Search engine search attr: {type(getattr(self.engine.search_engine, 'search', 'NOT_FOUND'))}")
                    print(f"ERROR: v7p3rSearch failed")
                    if self.verbose_output_enabled:
                        print(f"HARDSTOP ERROR: Cannot find move via v7p3rSearch: {e}. | FEN: {self.board.fen()}")
                    return
                    
            elif current_engine_name == 'stockfish':
                try:
                    # Debug: Check if Stockfish process is running
                    if self.stockfish.process is None:
                        if self.verbose_output_enabled:
                            print("DEBUG: Stockfish process is None! Cannot search.")
                        return
                    # Use the Stockfish engine for the current player
                    if self.monitoring_enabled and self.logger:
                        self.logger.info("Using Stockfish engine for move processing.")
                    engine_move = self.stockfish.search(self.board, chess.WHITE, self.stockfish_config)
                except Exception as e:
                    if self.monitoring_enabled and self.logger:
                        self.logger.error(f"[HARDSTOP Error] Cannot find move via Stockfish: {e}. | FEN: {self.board.fen()}")
                    print(f"ERROR: Stockfish search failed")
                    if self.verbose_output_enabled:
                        print(f"HARDSTOP ERROR: Cannot find move via Stockfish: {e}. | FEN: {self.board.fen()}")
                    return
            
            elif current_engine_name == 'v7p3r_rl' and 'v7p3r_rl' in self.engines:
                try:
                    # Use the RL engine for the current player
                    if self.monitoring_enabled and self.logger:
                        self.logger.info("Using v7p3r RL engine for move processing.")
                    engine_move = self.engines['v7p3r_rl'].search(self.board, self.current_player)
                except Exception as e:
                    if self.monitoring_enabled and self.logger:
                        self.logger.error(f"[HARDSTOP Error] Cannot find move via v7p3rReinforcementLearning: {e}. | FEN: {self.board.fen()}")
                    print(f"ERROR: RL engine search failed")
                    if self.verbose_output_enabled:
                        print(f"HARDSTOP ERROR: Cannot find move via v7p3rReinforcementLearning: {e}. | FEN: {self.board.fen()}")
                    return
                    
            elif current_engine_name == 'v7p3r_ga' and 'v7p3r_ga' in self.engines:
                try:
                    # Use the GA engine
                    if self.monitoring_enabled and self.logger:
                        self.logger.info("Using v7p3r GA engine for move processing.")
                    engine_move = self.engines['v7p3r_ga'].search(self.board, self.current_player)
                except Exception as e:
                    if self.monitoring_enabled and self.logger:
                        self.logger.error(f"[HARDSTOP Error] Cannot find move via v7p3rGeneticAlgorithm: {e}. | FEN: {self.board.fen()}")
                    print(f"ERROR: GA engine search failed")
                    if self.verbose_output_enabled:
                        print(f"HARDSTOP ERROR: Cannot find move via v7p3r_ga: {e}. | FEN: {self.board.fen()}")
                    return
                    
            elif current_engine_name == 'v7p3r_nn' and 'v7p3r_nn' in self.engines:
                try:
                    # Use the NN engine
                    if self.monitoring_enabled and self.logger:
                        self.logger.info("Using v7p3r NN engine for move processing.")
                    engine_move = self.engines['v7p3r_nn'].search(self.board, self.current_player)
                except Exception as e:
                    if self.monitoring_enabled and self.logger:
                        self.logger.error(f"[HARDSTOP Error] Cannot find move via v7p3rNeuralNetwork: {e}. | FEN: {self.board.fen()}")
                    print(f"ERROR: NN engine search failed")
                    if self.verbose_output_enabled:
                        print(f"HARDSTOP ERROR: Cannot find move via v7p3rNeuralNetwork: {e}. | FEN: {self.board.fen()}")
                    return
            else:
                if self.monitoring_enabled and self.logger:
                    self.logger.error(f"[HARDSTOP Error] No valid engine in configuration: {current_engine_name}. | FEN: {self.board.fen()}")
                print(f"ERROR: No valid engine configured: {current_engine_name}")
                if self.verbose_output_enabled:
                    print(f"HARDSTOP ERROR: No valid engine in configuration: {current_engine_name}. | FEN: {self.board.fen()}")
                return

            # Check and Push the move
            if not isinstance(engine_move, chess.Move):
                return # Move invalid
                
            # Initialize variables before try block to ensure they're always defined
            fen_before_move = self.board.fen()
            move_number = self.board.fullmove_number
            
            try:
                if self.board.is_legal(engine_move):
                    # Calculate timing BEFORE push_move so it's available for display
                    self.move_end_time = time.time()  # End timing the move
                    self.move_duration = self.move_end_time - self.move_start_time
                    
                    self.push_move(engine_move)
                    self.last_engine_move = engine_move
                    self.pv_line = ""
                else:
                    if self.monitoring_enabled and self.logger:
                        self.logger.error(f"[HARDSTOP Error] Illegal move: {engine_move} | FEN: {self.board.fen()}")
                    print(f"ERROR: Illegal move attempted")
                    if self.verbose_output_enabled:
                        print(f"HARDSTOP ERROR: Illegal move: {engine_move} | FEN: {self.board.fen()}")
                    return
            except Exception as e:
                if self.monitoring_enabled and self.logger:
                    self.logger.error(f"[HARDSTOP Error] Move Invalid: {e}. | Move: {engine_move} | FEN: {self.board.fen()}")
                print(f"ERROR: Move validation failed")
                if self.verbose_output_enabled:
                    print(f"HARDSTOP ERROR: Move Invalid: {e}. | Move: {engine_move} | FEN: {self.board.fen()}")
                return

            # Move display is now handled by display_move_made() in the push_move() method
            # This old print statement is no longer needed
            
            # Record move metrics
            if self.current_game_id:
                try:
                    current_player_name = self.white_player if self.current_player == chess.BLACK else self.black_player
                    
                    # Determine nodes evaluated (try to get from appropriate engine)
                    nodes_evaluated = 0
                    search_depth = 0
                    
                    if current_player_name == 'v7p3r' and hasattr(self.engine.search_engine, 'nodes_searched'):
                        nodes_evaluated = getattr(self.engine.search_engine, 'nodes_searched', 0)
                        search_depth = getattr(self.engine.search_engine, 'depth', 0)
                    elif current_player_name == 'stockfish' and hasattr(self.stockfish, 'nodes_searched'):
                        nodes_evaluated = getattr(self.stockfish, 'nodes_searched', 0)
                        search_depth = getattr(self.stockfish, 'depth', 0)
                    
                    # Use legacy compatibility function to avoid async issues
                    from metrics.v7p3r_chess_metrics import add_move_metric
                    add_move_metric(
                        game_id=self.current_game_id,
                        move_number=move_number,
                        player_color='white' if self.current_player == chess.BLACK else 'black',
                        move_uci=str(engine_move),
                        fen_before=fen_before_move,
                        evaluation=self.current_eval,
                        search_algorithm=current_player_name,
                        depth=search_depth,
                        nodes_searched=nodes_evaluated,
                        time_taken=self.move_duration,
                        pv_line=getattr(self, 'pv_line', '')
                    )
                    
                    if self.monitoring_enabled and self.logger:
                        self.logger.debug(f"Move metrics recorded for {current_player_name}")
                        
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to record move metrics: {e}")
            
            # Initialize logged_nodes
            logged_nodes = 0
            
            if self.monitoring_enabled and self.logger:
                # Use the nodes_searched from the appropriate engine
                current_engine_name = self.white_player.lower() if self.current_player == chess.WHITE else self.black_player.lower()
                try:
                    if current_engine_name == 'v7p3r' and hasattr(self.engine.search_engine, 'nodes_searched'):
                        logged_nodes = self.engine.search_engine.nodes_searched
                    else:
                        logged_nodes = 0
                except:
                    logged_nodes = 0
                    
                self.logger.info(f"{self.white_player if self.current_player == chess.WHITE else self.black_player} played: {engine_move} (Eval: {self.current_eval:.2f}) | Time: {self.move_duration:.6f}s | Nodes: {logged_nodes}")

    def push_move(self, move):
        """ Test and push a move to the board and game node """
        if not self.board.is_valid():
            if self.monitoring_enabled and self.logger:
                self.logger.error(f"[Error] Invalid board state detected! | FEN: {self.board.fen()}")
            return False
        
        if self.monitoring_enabled and self.logger:
            self.logger.info(f"Attempting to push move from {'White Engine' if self.board.turn == chess.WHITE else 'Black Engine'}: {move} | FEN: {self.board.fen()}")

        if isinstance(move, str):
            try:
                move = chess.Move.from_uci(move)
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Converted to chess.Move move from UCI string before push: {move}")
            except ValueError:
                if self.monitoring_enabled and self.logger:
                    self.logger.error(f"[Error] Invalid UCI string received: {move}")
                return False
        
        if not self.board.is_legal(move):
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Error] Illegal move blocked from being pushed: {move}")
            return False
        
        try:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"No remaining checks, pushing move: {move} | FEN: {self.board.fen()}")
            
            self.board.push(move)
            self.game_node = self.game_node.add_variation(move)
            self.last_move = move
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"Move pushed successfully: {move} | FEN: {self.board.fen()}")
            
            self.current_player = self.board.turn
            
            # Calculate move time
            move_time = self.move_duration if hasattr(self, 'move_duration') and self.move_duration > 0 else 0.0
            
            # Display the move that was made
            self.display_move_made(move, move_time)
            
            if self.engine.name.lower() == 'v7p3r':
                self.record_evaluation()
            
            # If the move ends the game, set the result header and end the game node
            if self.board.is_game_over():
                result = self.get_board_result()
                self.game.headers["Result"] = result
                self.game_node = self.game.end()
            else:
                self.game.headers["Result"] = "*"
            
            self.quick_save_pgn("active_game.pgn")
            
            return True
        except ValueError as e:
            if self.monitoring_enabled and self.logger:
                self.logger.error(f"[Error] ValueError pushing move {move}: {e}. Dumping PGN to error_dump.pgn")
            self.quick_save_pgn("games/game_error_dump.pgn")
            return False
            
        except Exception as e:
            if self.monitoring_enabled and self.logger:
                self.logger.error(f"[Error] Exception pushing move {move}: {e}. Dumping PGN to error_dump.pgn")
            self.quick_save_pgn("games/game_error_dump.pgn")
            return False
    
    def _create_nn_engine_wrapper(self, nn_config_path):
        """Create a wrapper for NN engine if it doesn't have proper game interface."""
        try:
            # Import the NN engine class here to avoid undefined reference
            from v7p3r_nn import v7p3rNeuralNetwork
            
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

    def _get_engine_config_for_player(self, player_name: str) -> dict:
        """
        Get the appropriate engine configuration for a given player
        """
        player_lower = player_name.lower()
        
        if player_lower == 'v7p3r':
            return {
                'name': getattr(self.engine, 'name', 'v7p3r'),
                'version': getattr(self.engine, 'version', 'unknown'),
                'search_algorithm': getattr(self.engine, 'search_algorithm', 'minimax'),
                'depth': getattr(self.engine, 'depth', 2),
                'max_depth': getattr(self.engine, 'max_depth', 2),
                'ruleset': getattr(self.config_manager, 'ruleset', 'standard') if hasattr(self, 'config_manager') else 'standard',
                'use_game_phase': getattr(self.engine, 'use_game_phase', True)
            }
        elif player_lower == 'stockfish':
            return {
                'name': 'stockfish',
                'version': 'unknown',
                'search_algorithm': 'stockfish',
                'depth': self.stockfish_config.get('depth', 2),
                'max_depth': self.stockfish_config.get('max_depth', 2),
                'elo_rating': self.stockfish_config.get('elo_rating', 1000),
                'skill_level': self.stockfish_config.get('skill_level', 5),
                'movetime': self.stockfish_config.get('movetime', 1000)
            }
        elif player_lower in ['v7p3r_rl', 'v7p3r_ga', 'v7p3r_nn'] and player_lower in self.engines:
            # Get configuration from specialized engines
            engine_obj = self.engines[player_lower]
            return {
                'name': getattr(engine_obj, 'name', player_lower),
                'version': getattr(engine_obj, 'version', 'unknown'),
                'search_algorithm': getattr(engine_obj, 'search_algorithm', player_lower),
                'depth': getattr(engine_obj, 'depth', 0),
                'max_depth': getattr(engine_obj, 'max_depth', 0)
            }
        else:
            # Default configuration for unknown engines
            return {
                'name': player_name,
                'version': 'unknown',
                'search_algorithm': 'unknown',
                'depth': 0,
                'max_depth': 0
            }

    # =============================================
    # ============ MAIN GAME LOOP =================

    def run(self, debug_mode: Optional[bool] = False):
        running = True
        game_count_remaining = self.game_count
        
        print(f"White: {self.white_player} vs Black: {self.black_player}")
        if self.monitoring_enabled and self.logger:
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
                        if self.monitoring_enabled and self.logger:
                            self.logger.info(f'All {self.game_count} games complete, exiting...')
                        print(f'All {self.game_count} games complete, exiting...')
                    else:
                        if self.monitoring_enabled and self.logger:
                            self.logger.info(f'Game {self.game_count - game_count_remaining}/{self.game_count} complete, starting next...')
                        print(f'Game {self.game_count - game_count_remaining}/{self.game_count} complete, starting next...')
                        self.game_start_timestamp = v7p3rUtilities.get_timestamp()
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
            if self.monitoring_enabled and self.logger:
                self.logger.error(f"[Error] Failed during engine cleanup: {e}")
            else:
                print(f"Error during engine cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup_engines()
        except:
            pass

    def _format_time_for_display(self, move_time: float) -> str:
        """
        Format move time for display with appropriate units.
        
        Args:
            move_time: Time in seconds (stored with high precision)
            
        Returns:
            Formatted time string with appropriate units (s or ms)
        """
        if move_time <= 0:
            return "0.000s"
        
        # If time is less than 0.1 seconds (100ms), display in milliseconds
        if move_time < 0.1:
            time_ms = move_time * 1000
            if time_ms < 1.0:
                # For very fast moves, show microseconds with higher precision
                return f"{time_ms:.3f}ms"
            else:
                # For sub-100ms moves, show milliseconds with 1 decimal place
                return f"{time_ms:.1f}ms"
        else:
            # For moves 100ms and above, display in seconds
            if move_time < 1.0:
                # Sub-second but >= 100ms: show 3 decimal places
                return f"{move_time:.3f}s"
            elif move_time < 10.0:
                # 1-10 seconds: show 2 decimal places
                return f"{move_time:.2f}s"
            else:
                # 10+ seconds: show 1 decimal place
                return f"{move_time:.1f}s"

    def display_move_made(self, move: chess.Move, move_time: float = 0.0):
        """Display a move with proper formatting and evaluation information."""
        # Get the player who just made the move (before the move, it was the previous player's turn)
        move_player = not self.board.turn  # Opposite of current turn since move was already made
        player_name = "White" if move_player == chess.WHITE else "Black"
        engine_name = self.white_player if move_player == chess.WHITE else self.black_player
        
        # Get evaluation from White's perspective (white_score - black_score)
        try:
            eval_score = self.engine.scoring_calculator.evaluate_position(self.board)
        except:
            eval_score = 0.0
        
        # Convert move time for appropriate display
        time_display = self._format_time_for_display(move_time)
        
        # Essential output (always shown)
        move_display = f"{player_name} ({engine_name}): {move}"
        # Always show time if we have timing data
        move_display += f" ({time_display})"
        if eval_score != 0.0:
            move_display += f" [Eval: {eval_score:+.2f}]"
        
        print(move_display)
        
        # Verbose output (only if verbose_output_enabled)
        if self.verbose_output_enabled:
            move_number = len(list(self.game.mainline_moves())) // 2 + 1
            print(f"  Move #{move_number} | Position: {self.board.fen()}")
            if eval_score > 0:
                print(f"  Position favors White by {abs(eval_score):.2f}")
            elif eval_score < 0:
                print(f"  Position favors Black by {abs(eval_score):.2f}")
            else:
                print(f"  Position is balanced")
        
        # Logging (if monitoring enabled) - use high precision for storage
        if self.monitoring_enabled and self.logger:
            self.logger.info(f"Move made: {player_name} ({engine_name}) played {move} with eval {eval_score:+.2f} in {move_time:.6f}s")
