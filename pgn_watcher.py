# pgn_watcher.py
""" PGN Watcher for Active Games 
This script monitors the active_game.pgn in the logging directory for changes and updates the chess board display accordingly.
This is an asyncronous standalone renderer intended to provide visual feedback for games in progress without impacting engine performance.
Due to the asyncronous nature, automated games may not display all moves in real time. """

import os
import time
import pygame
import chess
import chess.pgn
import sys
import logging
import datetime

# Define constants locally instead of importing from chess_game
WIDTH, HEIGHT = 640, 640
DIMENSION = 8
SQ_SIZE = WIDTH // DIMENSION
MAX_FPS = 15
IMAGES = {}


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base = getattr(sys, '_MEIPASS', None)
    if base:
        return os.path.join(base, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
# =====================================
# ========== LOGGING SETUP ============
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create logging directory relative to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
log_dir = os.path.join(project_root, 'logging')
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# Setup individual logger for this file
timestamp = get_timestamp()
#log_filename = f"pgn_watcher_{timestamp}.log"
log_filename = f"pgn_watcher.log"  # Use a single log file for simplicity
log_file_path = os.path.join(log_dir, log_filename)

#pgn_watcher_logger = logging.getLogger(f"pgn_watcher_{timestamp}")
pgn_watcher_logger = logging.getLogger("pgn_watcher")
pgn_watcher_logger.setLevel(logging.DEBUG)

if not pgn_watcher_logger.handlers:
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
    pgn_watcher_logger.addHandler(file_handler)
    pgn_watcher_logger.propagate = False

class StandaloneChessRenderer:
    """Simplified standalone renderer for chess positions"""
    
    def __init__(self):
        self.board = chess.Board()
        self.watch_mode = True
        self.screen: pygame.Surface | None = None
        self.selected_square = None
        self.last_ai_move = None
        self.flip_board = False
        self.display_needs_update = True
        self.screen_ready = False
        
    def load_images(self):
        """Load chess piece images"""
        pieces = ['wp', 'wN', 'wb', 'wr', 'wq', 'wk',
                 'bp', 'bN', 'bb', 'br', 'bq', 'bk']
        for piece in pieces:
            try:
                IMAGES[piece] = pygame.transform.scale(
                    pygame.image.load(resource_path(f"images/{piece}.png")),
                    (SQ_SIZE, SQ_SIZE)
                )
            except pygame.error:
                print(f"Could not load image for {piece}")
                
    def draw_board(self):
        """Draw the chess board"""
        if not self.watch_mode or self.screen is None:
            return
        colors = [pygame.Color("#a8a9a8"), pygame.Color("#d8d9d8")]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                # Calculate chess square coordinates
                if self.flip_board:
                    file = 7 - c
                    rank = r
                else:
                    file = c
                    rank = 7 - r

                # Determine color based on chess square
                color = colors[(file + rank) % 2]
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE)
                )
                
    def _piece_image_key(self, piece):
        """Convert chess piece to image key"""
        color = 'w' if piece.color == chess.WHITE else 'b'
        symbol = piece.symbol().upper()
        return f"{color}N" if symbol == 'N' else f"{color}{symbol.lower()}"
    
    def draw_pieces(self):
        """Draw pieces on the board"""
        if not self.watch_mode or self.screen is None:
            return
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                # Calculate chess square based on perspective
                if self.flip_board:
                    file = 7 - c
                    rank = r
                else:
                    file = c
                    rank = 7 - r

                square = chess.square(file, rank)
                piece = self.board.piece_at(square)

                if piece:
                    # Calculate screen position
                    screen_x = c * SQ_SIZE
                    screen_y = r * SQ_SIZE

                    piece_key = self._piece_image_key(piece)
                    if piece_key in IMAGES:
                        self.screen.blit(IMAGES[piece_key], (screen_x, screen_y))
    
    def highlight_last_move(self):
        """Highlight the last move on the board"""
        if not self.watch_mode or self.screen is None or not self.board.move_stack:
            return
        
        last_move = self.board.move_stack[-1]
        for square in [last_move.from_square, last_move.to_square]:
            screen_x, screen_y = self.chess_to_screen(square)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)
            s.fill(pygame.Color('yellow'))
            self.screen.blit(s, (screen_x, screen_y))
    
    def chess_to_screen(self, square):
        """Convert chess board square to screen coordinates"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        if self.flip_board:
            screen_file = 7 - file
            screen_rank = rank
        else:
            screen_file = file
            screen_rank = 7 - rank

        return (screen_file * SQ_SIZE, screen_rank * SQ_SIZE)
    
    def mark_display_dirty(self):
        """Mark the display as needing an update"""
        self.display_needs_update = True
    
    def update_display(self):
        """Update the display"""
        if not self.watch_mode or self.screen is None:
            return
            
        if self.display_needs_update:
            self.draw_board()
            self.draw_pieces()
            
            if self.board.move_stack:
                self.highlight_last_move()
                
            pygame.display.flip()
            self.display_needs_update = False
            self.screen_ready = True


class PGNWatcher:
    def __init__(self, pgn_path="active_game.pgn"):
        self.pgn_path = pgn_path
        self.last_mtime = 0
        pygame.init()
        
        # Use standalone renderer instead of ChessGame
        self.game = StandaloneChessRenderer()
        self.game.watch_mode = True
        self.game.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("PGN Watcher")
        self.game.load_images()
        self.clock = pygame.time.Clock()

    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
            # poll file modification
            try:
                mtime = os.path.getmtime(self.pgn_path)
                if mtime != self.last_mtime:
                    self.last_mtime = mtime
                    self._reload_pgn()
            except FileNotFoundError:
                pass
                
            # redraw if needed
            if self.game.screen:
                self.game.update_display()
            self.clock.tick(10)
        pygame.quit()

    def _reload_pgn(self):
        try:
            # read the PGN and replay mainline
            with open(self.pgn_path, "r") as f:
                game = chess.pgn.read_game(f)
            
            if game:
                board = game.board()
                for mv in game.mainline_moves():
                    board.push(mv)
                
                # update renderer state
                self.game.board = board
                self.game.selected_square = None
                self.game.mark_display_dirty()
                
                # Extract player names and evaluation
                white = game.headers.get('White', 'Unknown')
                black = game.headers.get('Black', 'Unknown')
                
                # Extract evaluation if present
                eval_str = ""
                for key in ("Eval", "Evaluation"):
                    if key in game.headers:
                        eval_str = f" (Eval: {game.headers[key]})"
                        break
                # Print formatted info
                print(f"Loaded game: {white} vs {black}")
                print(f"Current position: {board.fen()}{eval_str}")
        except Exception as e:
            print(f"Error reloading PGN: {e}")


if __name__ == "__main__":
    # Create logging directory if it doesn't exist
    os.makedirs("logging", exist_ok=True)
    
    # Default path to watch
    pgn_path = "active_game.pgn"
    
    # Allow command-line override
    if len(sys.argv) > 1:
        pgn_path = sys.argv[1]
    
    print(f"Watching PGN file: {pgn_path}")
    PGNWatcher(pgn_path).run()
