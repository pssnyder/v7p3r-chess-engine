# v7p3r_play.py

import os
import sys
import pygame
import chess
import chess.pgn
import datetime
from typing import Optional
import time
from io import StringIO
from v7p3r_config import v7p3rConfig
from v7p3r_utilities import resource_path, get_timestamp
from v7p3r_chess_metrics import get_metrics_instance, GameMetric
from chess_core import ChessCore
from pgn_watcher import PGNWatcher

CONFIG_NAME = "default_config"

# Define the maximum frames per second for the game loop
MAX_FPS = 60

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import necessary modules first
from v7p3r import v7p3rEngine
from v7p3r_stockfish_handler import StockfishHandler

class v7p3rChess(ChessCore):
    def __init__(self, config_name: Optional[str] = None):
        """Initialize the chess game with configuration."""
        super().__init__()  # Initialize ChessCore
        
        # Initialize Pygame (even in headless mode, for internal timing)
        pygame.init()
        self.clock = pygame.time.Clock()

        # Load configuration first
        try:
            if config_name is None:
                self.config_manager = v7p3rConfig()
                self.config = self.config_manager.get_config()
                self.game_config = self.config_manager.get_game_config()
                self.engine_config = self.config_manager.get_engine_config()
                self.stockfish_config = self.config_manager.get_stockfish_config()
                self.puzzle_config = self.config_manager.get_puzzle_config()
            else:
                self.config_name = config_name
                self.config_manager = v7p3rConfig(config_name=config_name)
                self.config = self.config_manager.get_config()
                self.game_config = self.config_manager.get_game_config()
                self.engine_config = self.config_manager.get_engine_config()
                self.stockfish_config = self.config_manager.get_stockfish_config()
                self.puzzle_config = self.config_manager.get_puzzle_config()

            # Game configuration
            self.white_player = self.game_config.get('white_player', 'v7p3r')
            self.black_player = self.game_config.get('black_player', 'stockfish')
            self.starting_position = self.game_config.get('starting_position', 'default')
            
            # Engine configurations
            self.white_engine_config = self.engine_config if self.white_player == 'v7p3r' else None
            self.black_engine_config = self.engine_config if self.black_player == 'v7p3r' else None
                
            # Initialize game state
            self.board = chess.Board()
            self.headless = self.game_config.get('headless', True)
            self.metrics_enabled = self.game_config.get('record_metrics', True)
            self.current_player = chess.WHITE
            
            # Initialize components
            self.engine = v7p3rEngine(self.engine_config)
            self.rules_manager = self.engine.rules_manager
            self.stockfish = None if self.headless else StockfishHandler(self.stockfish_config)
            
            # Initialize game record
            self.game = chess.pgn.Game()
            self._setup_game_headers()
            
        except Exception as e:
            print(f"Error initializing game: {str(e)}")
            raise

    def _setup_game_headers(self):
        """Set up the PGN headers for the game"""
        self.game.headers["Event"] = "v7p3r Engine Chess Game"
        self.game.headers["Site"] = self.game_config.get('site', 'Local')
        self.game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        self.game.headers["Round"] = "#"
        self.game.headers["White"] = self.white_player
        self.game.headers["Black"] = self.black_player

    def new_game(self):
        """Reset the game state for a new game"""
        # Call parent new_game method
        super().new_game(self.starting_position)
        
        # Add engine-specific initialization
        self.last_engine_move = chess.Move.null()
        self.current_eval = 0.0
        self.move_start_time = 0
        self.move_end_time = 0
        self.move_duration = 0

        # Reset PGN headers with engine information
        super().set_headers(white_player=self.white_player, black_player=self.black_player, event="v7p3r Engine Chess Game")
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
            from v7p3r_chess_metrics import add_game_result
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
        except Exception as e:
            raise

    def handle_game_end(self):
        """Check if the game is over and handle end conditions."""
        if self.board.is_game_over():
            # Call parent method to handle basic game end logic
            super().handle_game_end()
            
            # Add engine-specific game end handling
            result = self.get_board_result()
            self.save_game_data()
            print(f"\nGame over: {result}")
            return True
        return False

    def record_evaluation(self):
        """Record evaluation score in PGN comments"""
        # Special handling for game-ending positions
        if self.board.is_checkmate():
            # Assign a huge negative score if white is checkmated, huge positive if black is checkmated
            score = -999999999 if self.board.turn == chess.WHITE else 999999999
                
            # Make sure the score format is consistent, especially for extreme values
            formatted_score = f"{score:+.2f}" if abs(score) < 10000 else f"{int(score):+d}"
            self.current_eval = score
            self.game_node.comment = f"Eval: {formatted_score}"
        else:
            # Use standard white perspective evaluation (white_score - black_score)
            score = self.engine.scoring_calculator.evaluate_position(self.board)
            
            self.current_eval = score
            self.game_node.comment = f"Eval: {score:+.2f}"
            
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
                from v7p3r_chess_metrics import get_metrics_instance
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
                except Exception as e:
                    raise
                  
            except Exception as e:
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
                
            # Save locally into pgn file
            pgn_filepath = f"games/eval_game_{timestamp}.pgn"
            with open(pgn_filepath, "w") as f:
                exporter = chess.pgn.FileExporter(f)
                self.game.accept(exporter)
                # Always append the result string at the end for compatibility
                if result != "*":
                    f.write(f"\n{result}\n")
            self.quick_save_pgn(f"games/{game_id}.pgn")

    # ===================================
    # ========= MOVE HANDLERS ===========
    
    def process_engine_move(self):
        """Process move for the engine"""
        engine_move = chess.Move.null()
        self.current_eval = 0.0
        # Explicitly convert board.turn to chess.Color
        self.current_player = chess.WHITE if self.board.turn else chess.BLACK
        self.move_start_time = time.time()  # Start timing the move
        self.move_end_time = 0
        self.move_duration = 0
        print(f"{self.white_player if self.current_player == chess.WHITE else self.black_player} is thinking...")
        
        # Send the move request to the appropriate engine or human interface
        if (self.white_player.lower() == 'human' and self.current_player == chess.WHITE) or (self.black_player.lower() == 'human' and self.current_player == chess.BLACK):
            # Handle human player input (not implemented here, placeholder)
            return
        else:
            # Determine current player engine
            current_engine_name = self.white_player.lower() if self.current_player == chess.WHITE else self.black_player.lower()

            # Keep track of failed engine attempts to prevent infinite loops
            if not hasattr(self, '_engine_failures'):
                self._engine_failures = {}
            
            # Check if this engine has already failed        
            if current_engine_name in self._engine_failures:
                return

            try:
                if current_engine_name == 'v7p3r':
                    engine_move = self.engine.search_engine.search(self.board, self.current_player)
                elif current_engine_name == 'stockfish':
                    if self.stockfish.process is None:
                        raise Exception("Stockfish process is not running")
                    engine_move = self.stockfish.search(self.board, self.current_player, self.stockfish_config)
                # ... other engine cases ...

                # Process the move if it's valid
                if isinstance(engine_move, chess.Move) and self.board.is_legal(engine_move):
                    self.move_end_time = time.time()
                    self.move_duration = self.move_end_time - self.move_start_time
                    self.push_move(engine_move)
                    self.last_engine_move = engine_move
                else:
                    raise Exception(f"Invalid move returned by engine: {engine_move}")

            except Exception as e:
                # Mark this engine as failed
                self._engine_failures[current_engine_name] = str(e)

    def push_move(self, move):
        """ Test and push a move to the board and game node """
        # Call parent method first for basic move validation and execution
        if not super().push_move(move):
            return False
            
        # Add engine-specific functionality after successful move
        try:
            # Calculate move time
            move_time = self.move_duration if hasattr(self, 'move_duration') and self.move_duration > 0 else 0.0
            
            # Display the move that was made
            self.display_move_made(move, move_time)
            
            # Record evaluation if using v7p3r engine
            if self.engine.name.lower() == 'v7p3r':
                self.record_evaluation()
            
            # The active_game.pgn is already updated in the parent chess_core.py push_move method
            
            return True
        except Exception as e:
            self.quick_save_pgn("games/game_error_dump.pgn")
            return False
    
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

    def _make_engine_move(self, board: chess.Board, color: chess.Color) -> Optional[chess.Move]:
        """Make a move using the v7p3r engine with enhanced evaluation"""
        try:
            # Get game phase for context
            game_phase = self.rules_manager.get_game_phase(board)
            
            # Record state before move
            prev_state = {
                'fen': board.fen(),
                'phase': game_phase,
                'move_number': board.fullmove_number,
                'color': 'White' if color == chess.WHITE else 'Black'
            }
            
            # Get engine move
            move = self.engine.get_move(board, color)
            
            # Verify move validity
            if not move or not board.is_legal(move):
                print(f"Warning: Engine returned invalid move: {move}")
                # Get emergency move using simple search
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    move = legal_moves[0]
                else:
                    return None
                    
            # Record metrics
            if self.metrics_enabled:
                metrics = {
                    'move': move.uci(),
                    'prev_state': prev_state,
                    'evaluation': self.engine.get_current_evaluation(),
                    'nodes_searched': self.engine.get_nodes_searched(),
                    'time_taken': self.engine.get_last_move_time(),
                    'game_phase': game_phase
                }
                self.record_metrics(metrics)
                
            return move
            
        except Exception as e:
            print(f"Error in engine move: {str(e)}")
            # Emergency fallback
            legal_moves = list(board.legal_moves)
            return legal_moves[0] if legal_moves else None

    def play_game(self):
        """Main game loop with enhanced move processing"""
        try:
            while not self.board.is_game_over():
                self.clock.tick(MAX_FPS)  # Control game loop timing
                
                # Handle events and display
                if not self.headless:
                    self._handle_events()
                    self._update_display()
                
                # Make moves based on current turn
                if self.board.turn == chess.WHITE:
                    if self.game_config['white_player'] == 'v7p3r':
                        move = self._make_engine_move(self.board, chess.WHITE)
                    else:
                        move = self._make_opponent_move(self.board, chess.WHITE)
                else:
                    if self.game_config['black_player'] == 'v7p3r':
                        move = self._make_engine_move(self.board, chess.BLACK)
                    else:
                        move = self._make_opponent_move(self.board, chess.BLACK)
                
                # Apply move if valid
                if move and self.board.is_legal(move):
                    self.board.push(move)
                    self._update_game_record(move)
                    
                    # Check for game end conditions
                    if self.board.is_checkmate():
                        print("Checkmate!")
                        break
                    elif self.board.is_stalemate():
                        print("Stalemate!")
                        break
                    elif self.board.can_claim_draw():
                        print("Draw position reached!")
                        break
                        
                else:
                    print("Invalid move received, ending game")
                    break
                    
            # Game over, record final state
            self._record_game_end()
            
        except Exception as e:
            print(f"Error in game loop: {str(e)}")
            self._record_game_end(error=str(e))

    # =============================================
    # ============ MAIN GAME LOOP =================

    def run(self, debug_mode: Optional[bool] = False):
        running = True
        game_count_remaining = self.game_count
        
        print(f"White: {self.white_player} vs Black: {self.black_player}")
        
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
                        print(f'All {self.game_count} games complete, exiting...')
                    else:
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
            # Cleanup Stockfish
            if hasattr(self, 'stockfish') and self.stockfish:
                self.stockfish.quit()
                
        except Exception as e:
            raise

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup_engines()
        except:
            pass

    def display_move_made(self, move: chess.Move, move_time: float = 0.0):
        """Display a move with proper formatting and evaluation information."""
        # Get the player who just made the move (before the move, it was the previous player's turn)
        move_player = not self.board.turn  # Opposite of current turn since move was already made
        player_name = "White" if move_player == chess.WHITE else "Black"
        engine_name = self.white_player if move_player == chess.WHITE else self.black_player
        
        # Get evaluation
        try:
            # Special handling for checkmate
            if self.board.is_checkmate():
                # Assign a huge negative score if white is checkmated, huge positive if black is checkmated
                eval_score = -999999999 if self.board.turn == chess.WHITE else 999999999
            else:
                # Use standard white perspective evaluation (white_score - black_score)
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
            # Format evaluation appropriately (integer for extreme values)
            if abs(eval_score) > 10000:
                move_display += f" [Eval: {int(eval_score):+d}]"
            else:
                move_display += f" [Eval: {eval_score:+.2f}]"
        
        print(move_display)
        
    def _handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._record_game_end()
                pygame.quit()
                sys.exit()

    def _update_display(self):
        """Update the game display"""
        if not self.headless:
            # Implementation for display update would go here
            pass

    def _make_opponent_move(self, board: chess.Board, color: chess.Color) -> Optional[chess.Move]:
        """Get move from opponent (e.g., Stockfish)"""
        try:
            if self.stockfish:
                move = self.stockfish.get_move(board)
                return move if move and board.is_legal(move) else None
        except Exception as e:
            print(f"Error getting opponent move: {str(e)}")
        return None

    def _update_game_record(self, move: chess.Move):
        """Update the game record with the new move"""
        node = self.game.add_variation(move)
        evaluation = self.engine.get_current_evaluation() if hasattr(self.engine, 'get_current_evaluation') else 0.0
        node.comment = f"Eval: {evaluation:.2f}"
        
        # Save PGN after each move
        with open("active_game.pgn", "w") as f:
            print(self.game, file=f, end="\n\n")

    def record_metrics(self, metrics: dict):
        """Record game metrics"""
        if self.metrics_enabled:
            game_metric = GameMetric(
                move=metrics['move'],
                evaluation=metrics['evaluation'],
                nodes_searched=metrics['nodes_searched'],
                time_taken=metrics['time_taken'],
                game_phase=metrics['game_phase']
            )
            get_metrics_instance().record_game_metric(game_metric)

    def _record_game_end(self, error: str = None):
        """Record the end of the game"""
        if error:
            self.game.headers["Result"] = "*"
            self.game.headers["Termination"] = f"Error: {error}"
        else:
            result = self.board.result()
            self.game.headers["Result"] = result
            
        # Save final PGN
        with open("active_game.pgn", "w") as f:
            print(self.game, file=f, end="\n\n")
            
if __name__ == "__main__":
    # Process command line arguments
    if CONFIG_NAME == "default_config":
        print("Using default configuration for v7p3rChess.")
    else:
        print(f"Using custom configuration: {CONFIG_NAME} for v7p3rChess.")

    game = v7p3rChess(config_name=CONFIG_NAME)

    # TODO Fix implementation so it doesn't crash or interfere with the game. Start the PGN watcher in a separate thread if it's not already running
    #pgn_watcher = PGNWatcher()
    #threading.Thread(target=pgn_watcher.run).start()
    
    game.run()
