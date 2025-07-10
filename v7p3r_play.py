# v7p3r_play.py

import os
import sys
import pygame
import chess
import chess.pgn
import datetime
import sqlite3
from typing import Optional
import time
from io import StringIO
from v7p3r_config import v7p3rConfig
from v7p3r_utilities import resource_path, get_timestamp
from v7p3r_chess_metrics import get_metrics_instance, GameMetric, MoveMetric
from chess_core import ChessCore
from pgn_watcher import PGNWatcher
from v7p3r import v7p3rEngine
from v7p3r_stockfish_handler import StockfishHandler

CONFIG_NAME = "default_config"

# Define the maximum frames per second for the game loop
MAX_FPS = 60

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class v7p3rChess:
    def __init__(self, config_name: Optional[str] = None):
        """Initialize the chess game with configuration."""
        # Initialize Pygame (even in headless mode, for internal timing)
        pygame.init()
        self.clock = pygame.time.Clock()
        
        # Initialize chess core component for basic chess functionality
        self.chess_core = ChessCore()

        try:
            # Load configuration first
            if config_name is None:
                self.config_manager = v7p3rConfig()
            else:
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
            
            # Game state tracking
            self.headless = self.game_config.get('headless', True)
            self.metrics_enabled = self.game_config.get('record_metrics', True)
            self.current_player = chess.WHITE
            self.game_count = self.game_config.get('game_count', 1)
            self.engines = {}
            self.current_game_id = None
            self.game_start_timestamp = get_timestamp()
            self.game_start_time = time.time()
            self.current_eval = 0.0
            self.last_engine_move = chess.Move.null()
            self.move_duration = 0
            
            # Initialize components
            self.engine = v7p3rEngine(self.engine_config)
            self.rules_manager = self.engine.rules_manager
            self.stockfish = StockfishHandler(self.stockfish_config)
            
            # Set up initial game state
            self._setup_game()
            
        except Exception as e:
            print(f"Error initializing game: {str(e)}")
            raise

    def _setup_game(self):
        """Set up a new game"""
        # Initialize chess core with starting position
        self.chess_core.new_game(self.starting_position)
        
        # Set up game headers
        self.chess_core.set_headers(
            white_player=self.white_player,
            black_player=self.black_player,
            event="v7p3r Engine Chess Game"
        )
        
        # Initialize game ID and save initial state
        self.current_game_id = f"eval_game_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.chess_core.quick_save_pgn("active_game.pgn")

    def push_move(self, move):
        """Push a move and update game state"""
        try:
            # Calculate move time
            move_time = self.move_duration if hasattr(self, 'move_duration') and self.move_duration > 0 else 0.0
            
            # Display the move that was made
            self.display_move_made(move, move_time)
            
            # Get evaluation before making the move
            score = self.engine.scoring_calculator.evaluate_position(self.chess_core.board)
            formatted_score = f"{score:+.2f}"
            self.current_eval = score
            
            # Push the move using chess_core
            if self.chess_core.push_move(move):
                # Add evaluation comment
                if self.chess_core.game_node.parent:
                    self.chess_core.game_node.comment = f"Eval: {formatted_score}"
                
                # Save updated game state
                self.chess_core.quick_save_pgn("active_game.pgn")
                return True
            return False
            
        except Exception as e:
            print(f"Error in push_move: {str(e)}")
            self.chess_core.quick_save_pgn("games/game_error_dump.pgn")
            return False

    def new_game(self):
        """Reset the game state for a new game"""
        
        # Add engine-specific initialization
        self.last_engine_move = chess.Move.null()
        self.current_eval = 0.0
        self.move_start_time = 0
        self.move_end_time = 0
        self.move_duration = 0

        # Reset PGN headers with engine information
        self.chess_core.set_headers(white_player=self.white_player, black_player=self.black_player, event="v7p3r Engine Chess Game")
        self.chess_core.quick_save_pgn("active_game.pgn")
        
        # Initialize metrics for new game
        self.current_game_id = f"eval_game_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Record game start in metrics (use a safer approach for async)
        try:
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

    def record_evaluation(self):
        """Record evaluation score in PGN comments"""
        # Special handling for game-ending positions
        if self.chess_core.board.is_checkmate():
            # Assign a huge negative score if white is checkmated, huge positive if black is checkmated
            score = -999999999 if self.chess_core.board.turn == chess.WHITE else 999999999
            formatted_score = f"{score:+.2f}" if abs(score) < 10000 else f"{int(score):+d}"
        else:
            # Use standard white perspective evaluation
            score = self.engine.scoring_calculator.evaluate_position(self.chess_core.board)
            formatted_score = f"{score:+.2f}"
        
        self.current_eval = score
        if self.chess_core.game_node:
            self.chess_core.game_node.comment = f"Eval: {formatted_score}"

    def save_game_data(self):
        """Save the game data to local files and database."""
        games_dir = "games"
        os.makedirs(games_dir, exist_ok=True)
        
        timestamp = self.game_start_timestamp
        game_id = f"eval_game_{timestamp}"

        # Get game result from chess_core
        result = self.chess_core.get_board_result()
        
        # Save PGN file
        pgn_filepath = f"games/{game_id}.pgn"
        self.chess_core.quick_save_pgn(pgn_filepath)
        
        # Update metrics if enabled
        if self.metrics_enabled and self.current_game_id:
            try:
                # Calculate final game metrics
                game_duration = time.time() - self.game_start_time
                total_moves = len(list(self.chess_core.game.mainline_moves()))
                
                # Get metrics instance
                metrics = get_metrics_instance()
                
                # Map result to database format
                result_map = {
                    "1-0": "win" if self.white_player == "v7p3r" else "loss",
                    "0-1": "win" if self.black_player == "v7p3r" else "loss",
                    "1/2-1/2": "draw",
                    "*": "incomplete"
                }
                game_result = result_map.get(result, "incomplete")
                
                # Update database with final result
                with sqlite3.connect(metrics.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE games
                        SET result = ?, game_length = ?, end_time = ?
                        WHERE game_id = ?
                    """, (game_result, total_moves, datetime.datetime.now().isoformat(), self.current_game_id))
                    conn.commit()
                    
            except Exception as e:
                print(f"Error saving game metrics: {str(e)}")

    def display_move_made(self, move: chess.Move, move_time: float):
        """Display information about a move that was made."""
        try:
            # Format move time
            time_str = f"{move_time:.3f}s" if move_time > 0 else "0.000s"
            
            # Get player name
            current_player = "White" if self.chess_core.board.turn else "Black"
            player_name = self.white_player if current_player == "White" else self.black_player
            
            # Print move info
            print(f"{current_player} ({player_name}): {move.uci()} ({time_str}) [Eval: {self.current_eval:+.2f}]")
            
        except Exception as e:
            print(f"Error displaying move: {str(e)}")

    def process_engine_move(self):
        """Process move for the engine"""
        try:
            engine_move = chess.Move.null()
            self.current_eval = 0.0
            
            # Get current player
            self.current_player = self.chess_core.board.turn
            
            # Start timing
            self.move_start_time = time.time()
            self.move_duration = 0
            
            print(f"{self.white_player if self.current_player == chess.WHITE else self.black_player} is thinking...")
            
            # Get the move from appropriate engine
            if self.current_player == chess.WHITE and self.white_player == 'v7p3r':
                engine_move = self.engine.search_engine.search(self.chess_core.board, self.current_player)
            elif self.current_player == chess.BLACK and self.black_player == 'v7p3r':
                engine_move = self.engine.search_engine.search(self.chess_core.board, self.current_player)
            elif self.current_player == chess.WHITE and self.white_player == 'stockfish':
                engine_move = self.stockfish.search(self.chess_core.board, self.current_player, self.stockfish_config)
            elif self.current_player == chess.BLACK and self.black_player == 'stockfish':
                engine_move = self.stockfish.search(self.chess_core.board, self.current_player, self.stockfish_config)
            
            # Process the move
            if isinstance(engine_move, chess.Move) and self.chess_core.board.is_legal(engine_move):
                self.move_end_time = time.time()
                self.move_duration = self.move_end_time - self.move_start_time
                self.push_move(engine_move)
                self.last_engine_move = engine_move
                return True
                
        except Exception as e:
            print(f"Error processing engine move: {str(e)}")
        return False

    def handle_game_end(self):
        """Check if the game is over and handle end conditions."""
        if self.chess_core.board.is_game_over():
            result = self.chess_core.get_board_result()
            print(f"\nGame over: {result}")
            self.save_game_data()
            return True
        return False

    def run(self):
        """Main game loop"""
        try:
            while not self.chess_core.board.is_game_over():
                # Process moves
                if not self.process_engine_move():
                    break
                    
                # Handle game end
                if self.handle_game_end():
                    break
                    
                # Maintain game loop timing
                self.clock.tick(MAX_FPS)
                
        except KeyboardInterrupt:
            print("\nGame interrupted by user")
            self.chess_core.quick_save_pgn("games/interrupted_game.pgn")
            
        except Exception as e:
            print(f"Error in game loop: {str(e)}")
            self.chess_core.quick_save_pgn("games/error_game.pgn")

if __name__ == "__main__":
    game = v7p3rChess()
    game.run()
