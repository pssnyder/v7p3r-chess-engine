# chess_piece_utils.py - Utilities for chess pieces and image loading

import os
import pygame
import chess
import numpy as np

class ChessPieceImages:
    """
    Helper class to load and manage chess piece images for the GUI
    """
    
    def __init__(self, square_size):
        """
        Initialize with the square size to properly scale images
        """
        self.square_size = square_size
        self.piece_images = {}
        
        # Try to load piece images from various common paths
        self._try_load_images()
    
    def _try_load_images(self):
        """
        Try to load chess piece images from various paths
        Falls back to Unicode if no images found
        """
        # Possible paths for piece images
        image_paths = [
            # Images directory
            os.path.join(os.getcwd(), "images"),
        ]
        
        # Try each path
        for path in image_paths:
            if os.path.exists(path):
                self._load_images_from_path(path)
                return
        
        # If no images found, use Unicode fallback
        self._create_unicode_pieces()
    
    def _load_images_from_path(self, path):
        """
        Load chess piece images from a path
        Expects filenames like: "white_king.png", "black_pawn.png", etc.
        """
        # Mapping of piece types to image filenames
        piece_names = {
            chess.KING: "K", 
            chess.QUEEN: "Q", 
            chess.ROOK: "R", 
            chess.BISHOP: "B", 
            chess.KNIGHT: "N", 
            chess.PAWN: "p"
        }
        
        # Load images for both colors
        colors = {chess.WHITE: "w", chess.BLACK: "b"}
        
        for piece_type, piece_name in piece_names.items():
            for color, color_name in colors.items():
                # Try different common naming conventions
                filenames = [
                    f"{color_name}{piece_name}.png",
                    f"{color_name[0]}{piece_name[0]}.png",
                    f"{piece_name}{color_name}.png",
                    f"{piece_name[0]}{color_name[0]}.png"
                ]
                
                # Try each possible filename
                for filename in filenames:
                    filepath = os.path.join(path, filename)
                    if os.path.exists(filepath):
                        image = pygame.image.load(filepath)
                        image = pygame.transform.scale(image, (self.square_size, self.square_size))
                        
                        # Create symbol (e.g., 'K' for white king)
                        if color == chess.WHITE:
                            symbol = piece_names[piece_type][0].upper()
                        else:
                            symbol = piece_names[piece_type][0].lower()
                        
                        self.piece_images[symbol] = image
                        break
    
    def _create_unicode_pieces(self):
        """
        Create piece images using Unicode characters as fallback
        """
        # Unicode chess symbols
        unicode_pieces = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
        }
        
        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        
        # Font for pieces
        piece_font = pygame.font.Font(None, self.square_size - 10)
        
        # Create text surfaces for each piece
        for symbol, unicode_char in unicode_pieces.items():
            color = WHITE if symbol.isupper() else BLACK
            text_surface = piece_font.render(unicode_char, True, color)
            self.piece_images[symbol] = text_surface
    
    def get_image(self, piece):
        """
        Get image for a chess.Piece object
        
        Args:
            piece: A chess.Piece object
        
        Returns:
            A pygame Surface containing the piece image
        """
        if piece is None:
            return None
        
        symbol = piece.symbol()
        if symbol in self.piece_images:
            return self.piece_images[symbol]
        return None

class ChessHighlighter:
    """
    Utility class to create highlight effects for chess squares
    """
    
    def __init__(self, square_size):
        """Initialize with square size"""
        self.square_size = square_size
        
        # Colors with alpha for highlighting
        self.SELECTED_COLOR = (255, 255, 0, 128)      # Yellow with transparency
        self.LEGAL_MOVE_COLOR = (0, 255, 0, 100)      # Green with transparency
        self.CHECK_COLOR = (255, 0, 0, 128)           # Red with transparency
        self.LAST_MOVE_COLOR = (0, 0, 255, 80)        # Blue with transparency
    
    def get_highlight_surface(self, highlight_type):
        """
        Get a surface for highlighting squares
        
        Args:
            highlight_type: String indicating highlight type
                ("selected", "legal_move", "check", "last_move")
        
        Returns:
            A pygame Surface with transparency
        """
        surface = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        
        if highlight_type == "selected":
            surface.fill(self.SELECTED_COLOR)
        elif highlight_type == "legal_move":
            surface.fill(self.LEGAL_MOVE_COLOR)
        elif highlight_type == "check":
            surface.fill(self.CHECK_COLOR)
        elif highlight_type == "last_move":
            surface.fill(self.LAST_MOVE_COLOR)
        
        return surface
    
    def draw_move_indicator(self, surface, x, y, radius=8):
        """
        Draw a circle indicator for legal moves
        
        Args:
            surface: Pygame surface to draw on
            x, y: Center coordinates
            radius: Circle radius
        """
        pygame.draw.circle(surface, (0, 200, 0), (x, y), radius)


class PieceAnimation:
    """
    Class for handling smooth chess piece animations
    """
    
    def __init__(self, fps=60, anim_speed=10):
        """
        Initialize animation settings
        
        Args:
            fps: Frames per second
            anim_speed: Animation speed (higher = faster)
        """
        self.fps = fps
        self.anim_speed = anim_speed
        self.current_animation = None
    
    def animate_move(self, from_coords, to_coords):
        """
        Set up a new animation
        
        Args:
            from_coords: (x, y) start position
            to_coords: (x, y) end position
        """
        self.current_animation = {
            'from': from_coords,
            'to': to_coords,
            'current': from_coords,
            'progress': 0.0
        }
    
    def update(self):
        """
        Update animation progress
        
        Returns:
            Current position (x, y) or None if animation is complete
        """
        if not self.current_animation:
            return None
        
        # Update progress
        self.current_animation['progress'] += self.anim_speed / self.fps
        
        # Clamp progress between 0 and 1
        progress = min(1.0, self.current_animation['progress'])
        
        # Calculate current position using linear interpolation
        start_x, start_y = self.current_animation['from']
        end_x, end_y = self.current_animation['to']
        
        current_x = start_x + (end_x - start_x) * progress
        current_y = start_y + (end_y - start_y) * progress
        
        self.current_animation['current'] = (current_x, current_y)
        
        # If animation is complete, return None
        if progress >= 1.0:
            result = self.current_animation['current']
            self.current_animation = None
            return None
        
        return self.current_animation['current']

# Example usage in your chess GUI:
'''
# In ChessGUI.__init__
self.piece_images = ChessPieceImages(self.square_size).piece_images
self.highlighter = ChessHighlighter(self.square_size)
self.animator = PieceAnimation()

# In draw_highlights
highlight = self.highlighter.get_highlight_surface("selected")
self.screen.blit(highlight, (x, y))

# For move animation
def make_move_with_animation(self, move):
    # Get start and end coordinates
    from_x, from_y = self.square_to_coord(move.from_square)
    to_x, to_y = self.square_to_coord(move.to_square)
    
    # Start animation
    self.animator.animate_move((from_x, from_y), (to_x, to_y))
    
    # Update animation in game loop
    while True:
        pos = self.animator.update()
        if pos is None:
            break
            
        # Draw the piece at the current position
        # ...
        
        pygame.display.flip()
        self.clock.tick(60)
    
    # Finally make the actual move
    self.board.push(move)
'''