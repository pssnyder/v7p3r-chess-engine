# v7p3r_game.py

"""Main Game Controller for V7P3R Chess Engine
Handles game flow, UI rendering with pygame, and coordinates between engines.
"""

import pygame
import chess
import chess.pgn
import time
import sys
import os
from datetime import datetime
from v7p3r_config import V7P3RConfig
from v7p3r_engine import V7P3REngine
from v7p3r_stockfish import StockfishHandler
from metrics import ChessMetrics

# Constants
WIDTH, HEIGHT = 640, 640
DIMENSION = 8
SQ_SIZE = WIDTH // DIMENSION
MAX_FPS = 15
IMAGES = {}

class ChessGame:
    def __init__(self, config_file="config.json", headless=False):
        # Initialize pygame only if not headless
        self.headless = headless
        if not headless:
            pygame.init()
        else:
            # Initialize pygame with dummy driver for headless mode
            import os
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            pygame.init()
            pygame.display.set_mode((1, 1))  # Minimal display
        
        # Load configuration
        self.config = V7P3RConfig(config_file)
        game_config = self.config.get_game_config()
        
        # Game settings
        self.game_count = game_config.get('game_count', 1)
        self.white_player = game_config.get('white_player', 'v7p3r')
        self.black_player = game_config.get('black_player', 'stockfish')
        
        # Initialize engines
        self.v7p3r_engine = V7P3REngine(config_file)
        self.stockfish_handler = StockfishHandler(self.config)
        
        # Initialize metrics
        self.metrics = ChessMetrics()
        
        # Game state
        self.board = chess.Board()
        self.game_id = None
        self.move_number = 1
        self.game_start_time = None
        self.games_played = 0
        
        # UI state (only if not headless)
        if not headless:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("V7P3R Chess Engine")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None
        self.selected_square = None
        self.display_needs_update = True
        
        # Load images
        self.load_images()
    
    def load_images(self):
        """Load chess piece images"""
        pieces = ['wp', 'wN', 'wb', 'wr', 'wq', 'wk',
                 'bp', 'bN', 'bb', 'br', 'bq', 'bk']
        for piece in pieces:
            try:
                IMAGES[piece] = pygame.transform.scale(
                    pygame.image.load(f"images/{piece}.png"),
                    (SQ_SIZE, SQ_SIZE)
                )
            except pygame.error as e:
                print(f"Could not load image {piece}.png: {e}")
                # Create a placeholder colored square
                surf = pygame.Surface((SQ_SIZE, SQ_SIZE))
                color = (255, 255, 255) if piece.startswith('w') else (0, 0, 0)
                surf.fill(color)
                IMAGES[piece] = surf
    
    def run_games(self):
        """Run the configured number of games"""
        try:
            for game_num in range(self.game_count):
                print(f"\n=== Starting Game {game_num + 1}/{self.game_count} ===")
                print(f"White: {self.white_player}, Black: {self.black_player}")
                
                self.run_single_game()
                self.games_played += 1
                
                if game_num < self.game_count - 1:
                    self.reset_for_new_game()
                    time.sleep(1)  # Brief pause between games
        
        except KeyboardInterrupt:
            print("\nGames interrupted by user")
        except Exception as e:
            print(f"Error during games: {e}")
        finally:
            self.cleanup()
    
    def run_single_game(self):
        """Run a single chess game"""
        self.reset_for_new_game()
        
        # Record game start
        white_config = self.config.get_engine_config() if self.white_player == 'v7p3r' else None
        black_config = self.config.get_engine_config() if self.black_player == 'v7p3r' else None
        self.game_id = self.metrics.record_game_start(
            self.white_player, self.black_player, white_config, black_config
        )
        
        self.game_start_time = time.time()
        running = True
        
        while running and not self.board.is_game_over():
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break
            
            # Determine current player
            current_player = self.white_player if self.board.turn == chess.WHITE else self.black_player
            
            print(f"\nMove {self.move_number} - {current_player} to play")
            
            # Get move from appropriate engine
            move_start_time = time.time()
            move = self.get_engine_move(current_player)
            move_time = time.time() - move_start_time
            
            if move is None:
                print(f"No move available for {current_player}")
                break
            
            print(f"Selected move: {move} (took {move_time:.2f}s)")
            
            # Make the move
            if move in self.board.legal_moves:
                # Record move analysis
                evaluation = self.v7p3r_engine.get_evaluation(self.board) if current_player == 'v7p3r' else None
                player_color = 'white' if self.board.turn == chess.WHITE else 'black'
                
                self.metrics.record_move(
                    self.game_id, self.move_number, player_color, move.uci(),
                    evaluation_score=evaluation, search_time=move_time
                )
                
                self.board.push(move)
                
                if self.board.turn == chess.WHITE:
                    self.move_number += 1
                
                # Update display
                self.update_display()
                self.write_pgn()
                
            else:
                print(f"Illegal move attempted: {move}")
                break
        
        # Game finished
        self.finish_game()
    
    def get_engine_move(self, player):
        """Get move from the specified engine/player"""
        if player == 'v7p3r':
            return self.v7p3r_engine.find_move(self.board, time_limit=30.0)
        elif player == 'stockfish':
            return self.stockfish_handler.get_move(self.board)
        else:
            # Random move for unknown players
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                import random
                return random.choice(legal_moves)
            return None
    
    def finish_game(self):
        """Handle game completion"""
        game_duration = time.time() - self.game_start_time if self.game_start_time else 0
        
        # Determine result
        result = self.get_game_result()
        
        print(f"\n=== Game Finished ===")
        print(f"Result: {result}")
        print(f"Moves: {self.move_number - 1}")
        print(f"Duration: {game_duration:.1f}s")
        
        # Record game end
        pgn_string = self.get_pgn_string()
        self.metrics.record_game_end(
            self.game_id, result, self.move_number - 1, game_duration, pgn_string
        )
        
        # Update engine performance
        winning_engine = self.get_winning_engine(result)
        if winning_engine:
            opponent = self.black_player if winning_engine == self.white_player else self.white_player
            self.metrics.update_engine_performance(winning_engine, opponent, result)
        
        # Save final PGN
        self.save_game_pgn()
        
        # Display final position briefly
        self.update_display()
        time.sleep(2)
    
    def get_game_result(self):
        """Get the game result string"""
        if self.board.is_checkmate():
            return "0-1" if self.board.turn == chess.WHITE else "1-0"
        elif self.board.is_stalemate():
            return "1/2-1/2"
        elif self.board.is_insufficient_material():
            return "1/2-1/2"
        elif self.board.is_seventyfive_moves():
            return "1/2-1/2"
        elif self.board.is_fivefold_repetition():
            return "1/2-1/2"
        else:
            return "1/2-1/2"  # Unfinished game
    
    def get_winning_engine(self, result):
        """Determine which engine won"""
        if result == "1-0":
            return self.white_player
        elif result == "0-1":
            return self.black_player
        else:
            return None  # Draw or unfinished
    
    def update_display(self):
        """Update the chess board display"""
        if self.headless:
            return
            
        self.draw_board()
        self.draw_pieces()
        pygame.display.flip()
        self.clock.tick(MAX_FPS)
    
    def draw_board(self):
        """Draw the chess board"""
        if self.headless or self.screen is None:
            return
            
        colors = [pygame.Color("#f0d9b5"), pygame.Color("#b58863")]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                color = colors[(r + c) % 2]
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE)
                )
    
    def draw_pieces(self):
        """Draw pieces on the board"""
        if self.headless or self.screen is None:
            return
            
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                square = chess.square(c, 7-r)  # Convert to chess coordinates
                piece = self.board.piece_at(square)
                
                if piece:
                    piece_key = self._piece_image_key(piece)
                    if piece_key in IMAGES:
                        self.screen.blit(IMAGES[piece_key], (c*SQ_SIZE, r*SQ_SIZE))
    
    def _piece_image_key(self, piece):
        """Convert chess piece to image key"""
        color = 'w' if piece.color == chess.WHITE else 'b'
        symbol = piece.symbol().upper()
        return f"{color}N" if symbol == 'N' else f"{color}{symbol.lower()}"
    
    def write_pgn(self):
        """Write current game to active_game.pgn"""
        try:
            game = chess.pgn.Game.from_board(self.board)
            game.headers["White"] = self.white_player
            game.headers["Black"] = self.black_player
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["Event"] = "V7P3R Engine Test"
            
            with open("active_game.pgn", "w") as f:
                print(game, file=f)
        except Exception as e:
            print(f"Error writing PGN: {e}")
    
    def get_pgn_string(self):
        """Get PGN string of current game"""
        try:
            game = chess.pgn.Game.from_board(self.board)
            game.headers["White"] = self.white_player
            game.headers["Black"] = self.black_player
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["Result"] = self.get_game_result()
            return str(game)
        except:
            return ""
    
    def save_game_pgn(self):
        """Save completed game to games directory"""
        try:
            os.makedirs("pgn_game_records", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pgn_game_records/game_{timestamp}_{self.white_player}_vs_{self.black_player}.pgn"
            
            with open(filename, "w") as f:
                f.write(self.get_pgn_string())
                
        except Exception as e:
            print(f"Error saving game PGN: {e}")
    
    def reset_for_new_game(self):
        """Reset state for a new game"""
        self.board = chess.Board()
        self.move_number = 1
        self.game_start_time = None
        self.v7p3r_engine.reset_game()
    
    def cleanup(self):
        """Clean up resources"""
        if self.stockfish_handler:
            self.stockfish_handler.quit()
        pygame.quit()
        
        # Print final statistics
        self.print_final_stats()
    
    def print_final_stats(self):
        """Print final statistics"""
        print(f"\n=== Final Statistics ===")
        print(f"Games played: {self.games_played}")
        
        v7p3r_stats = self.metrics.get_engine_stats('v7p3r')
        print(f"V7P3R Results: {v7p3r_stats['wins']}W {v7p3r_stats['losses']}L {v7p3r_stats['draws']}D")
        print(f"V7P3R Win Rate: {v7p3r_stats['win_rate']:.1f}%")
        
        # Move time stats
        time_stats = self.metrics.get_move_time_stats('v7p3r')
        if time_stats:
            print(f"V7P3R Avg Move Time: {time_stats['avg_time']:.2f}s")
            print(f"V7P3R Max Move Time: {time_stats['max_time']:.2f}s")
