# Chess Engine Testing Interface - CORRECTED VERSION
# This version properly initializes the EvaluationEngine with a board parameter

import pygame
import chess
import os
import sys
from evaluation_engine import EvaluationEngine  # Import your evaluation engine

class ChessGUI:
    def __init__(self):
        """Initialize the chess GUI with proper engine initialization"""
        
        # Initialize pygame
        pygame.init()
        self.screen_width = 800
        self.screen_height = 800
        self.square_size = self.screen_width // 8
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Chess Engine Testing Interface")
        
        # Colors
        self.light_color = (216, 217, 216)
        self.dark_color = (168, 169, 168)
        self.highlight_color = (255, 255, 0, 128)
        self.legal_move_color = (0, 255, 0, 128)
        self.check_color = (255, 0, 0, 128)
        
        # Fonts
        self.font = pygame.font.Font(None, 24)
        
        # Initialize chess board - THIS IS CRITICAL!
        self.board = chess.Board()
        
        # Now properly initialize the evaluation engine with the board
        # FIXED: Pass the board parameter to EvaluationEngine
        self.engine = EvaluationEngine(self.board)
        
        # Game state
        self.selected_square = None
        self.legal_moves = []
        self.game_over = False
        self.white_to_move = True
        
        # Load piece images
        self.piece_images = {}
        self.load_piece_images()
        
        print("Chess Engine Testing Interface")
        print("==============================")
        print("Looking for piece images in 'images' directory...")
        print("Expected naming: wQ.png, bK.png, wp.png, etc.")
        print("Successfully initialized EvaluationEngine with board!")
        
    def load_piece_images(self):
        """Load chess piece images from the images directory"""
        # Mapping from chess piece types to your naming convention
        piece_mapping = {
            chess.PAWN: 'p',
            chess.ROOK: 'R', 
            chess.KNIGHT: 'N',
            chess.BISHOP: 'B',
            chess.QUEEN: 'Q',
            chess.KING: 'K'
        }
        
        color_mapping = {
            chess.WHITE: 'w',
            chess.BLACK: 'b'
        }
        
        loaded_count = 0
        total_pieces = 12
        
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in piece_mapping:
                color_prefix = color_mapping[color]
                piece_suffix = piece_mapping[piece_type]
                filename = f"{color_prefix}{piece_suffix}.png"
                filepath = os.path.join("images", filename)
                
                try:
                    # Load and scale the image
                    image = pygame.image.load(filepath)
                    scaled_image = pygame.transform.scale(
                        image, 
                        (self.square_size - 10, self.square_size - 10)
                    )
                    
                    # Store in dictionary using chess piece object as key
                    piece = chess.Piece(piece_type, color)
                    self.piece_images[piece] = scaled_image
                    loaded_count += 1
                    print(f"✓ Loaded {filename}")
                    
                except (pygame.error, FileNotFoundError) as e:
                    print(f"✗ Failed to load {filename}: {e}")
                    
        print(f"Loaded {loaded_count}/{total_pieces} piece images")
        
        # If images failed to load, we'll use Unicode fallback
        if loaded_count == 0:
            print("No images loaded, will use Unicode symbols as fallback")
            
    def get_square_from_mouse(self, mouse_pos):
        """Convert mouse position to chess square"""
        x, y = mouse_pos
        file = x // self.square_size
        rank = 7 - (y // self.square_size)  # Flip rank for chess coordinate system
        
        if 0 <= file <= 7 and 0 <= rank <= 7:
            return chess.square(file, rank)
        return None
        
    def square_to_pixel(self, square):
        """Convert chess square to pixel coordinates"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        x = file * self.square_size
        y = (7 - rank) * self.square_size  # Flip rank for display
        return x, y
        
    def draw_board(self):
        """Draw the chess board"""
        for rank in range(8):
            for file in range(8):
                color = self.light_color if (rank + file) % 2 == 0 else self.dark_color
                rect = pygame.Rect(
                    file * self.square_size,
                    rank * self.square_size,
                    self.square_size,
                    self.square_size
                )
                pygame.draw.rect(self.screen, color, rect)
                
    def draw_highlights(self):
        """Draw square highlights"""
        # Highlight selected square
        if self.selected_square is not None:
            x, y = self.square_to_pixel(self.selected_square)
            highlight_surface = pygame.Surface((self.square_size, self.square_size))
            highlight_surface.set_alpha(128)
            highlight_surface.fill(self.highlight_color[:3])
            self.screen.blit(highlight_surface, (x, y))
            
        # Highlight legal moves
        for move in self.legal_moves:
            x, y = self.square_to_pixel(move.to_square)
            circle_center = (x + self.square_size // 2, y + self.square_size // 2)
            pygame.draw.circle(self.screen, self.legal_move_color[:3], circle_center, 15)
            
        # Highlight check
        if self.board.is_check():
            king_square = self.board.king(self.board.turn)
            if king_square:
                x, y = self.square_to_pixel(king_square)
                check_surface = pygame.Surface((self.square_size, self.square_size))
                check_surface.set_alpha(128)
                check_surface.fill(self.check_color[:3])
                self.screen.blit(check_surface, (x, y))
                
    def draw_pieces(self):
        """Draw chess pieces on the board"""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                x, y = self.square_to_pixel(square)
                
                if piece in self.piece_images:
                    # Use loaded image
                    image_rect = self.piece_images[piece].get_rect()
                    image_rect.center = (x + self.square_size // 2, y + self.square_size // 2)
                    self.screen.blit(self.piece_images[piece], image_rect)
                else:
                    # Use Unicode fallback
                    self.draw_unicode_piece(piece, x, y)
                    
    def draw_unicode_piece(self, piece, x, y):
        """Draw piece using Unicode symbols as fallback"""
        unicode_pieces = {
            (chess.PAWN, chess.WHITE): '♙',
            (chess.ROOK, chess.WHITE): '♖',
            (chess.KNIGHT, chess.WHITE): '♘',
            (chess.BISHOP, chess.WHITE): '♗',
            (chess.QUEEN, chess.WHITE): '♕',
            (chess.KING, chess.WHITE): '♔',
            (chess.PAWN, chess.BLACK): '♟',
            (chess.ROOK, chess.BLACK): '♜',
            (chess.KNIGHT, chess.BLACK): '♞',
            (chess.BISHOP, chess.BLACK): '♝',
            (chess.QUEEN, chess.BLACK): '♛',
            (chess.KING, chess.BLACK): '♚',
        }
        
        symbol = unicode_pieces.get((piece.piece_type, piece.color), '?')
        font = pygame.font.Font(None, 48)
        text = font.render(symbol, True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (x + self.square_size // 2, y + self.square_size // 2)
        self.screen.blit(text, text_rect)
        
    def draw_info(self):
        """Draw game information"""
        # Current player
        turn_text = "White to move" if self.board.turn == chess.WHITE else "Black to move"
        turn_surface = self.font.render(turn_text, True, (0, 0, 0))
        self.screen.blit(turn_surface, (10, self.screen_height - 60))
        
        # Evaluation score (if engine is available)
        try:
            score = self.engine.evaluate_position()
            eval_text = f"Evaluation: {score:.2f}"
            eval_surface = self.font.render(eval_text, True, (0, 0, 0))
            self.screen.blit(eval_surface, (10, self.screen_height - 40))
        except Exception as e:
            error_text = f"Engine error: {str(e)[:30]}..."
            error_surface = self.font.render(error_text, True, (255, 100, 100))
            self.screen.blit(error_surface, (10, self.screen_height - 40))
        
        # Game status
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            status_text = f"Checkmate! {winner} wins!"
        elif self.board.is_stalemate():
            status_text = "Stalemate! Draw!"
        elif self.board.is_check():
            status_text = "Check!"
        else:
            status_text = ""
            
        if status_text:
            status_surface = self.font.render(status_text, True, (255, 255, 0))
            self.screen.blit(status_surface, (10, self.screen_height - 20))
            
        # Instructions
        instructions = [
            "Left click to select pieces and move",
            "Engine plays as Black, you play as White"
        ]
        for i, instruction in enumerate(instructions):
            text_surface = pygame.font.Font(None, 20).render(instruction, True, (200, 200, 200))
            self.screen.blit(text_surface, (10, 10 + i * 20))
        
    def handle_click(self, square):
        """Handle mouse click on a square"""
        if self.game_over:
            return
            
        piece = self.board.piece_at(square)
        
        # If clicking on a piece of the current player, select it
        if piece and piece.color == self.board.turn:
            self.selected_square = square
            self.legal_moves = [move for move in self.board.legal_moves 
                              if move.from_square == square]
        
        # If a square is selected and we're clicking on a legal move destination
        elif self.selected_square is not None:
            move = None
            for legal_move in self.legal_moves:
                if legal_move.to_square == square:
                    move = legal_move
                    break
                    
            if move:
                # Make the move
                self.board.push(move)
                self.selected_square = None
                self.legal_moves = []
                
                # Update engine board reference (important!)
                self.engine.board = self.board
                
                # Check for game over
                if self.board.is_game_over():
                    self.game_over = True
                    return
                
                # If it's now the engine's turn (Black), make engine move
                if self.board.turn == chess.BLACK:
                    self.make_engine_move()
        else:
            # Clear selection
            self.selected_square = None
            self.legal_moves = []
            
    def make_engine_move(self):
        """Make a move using the evaluation engine"""
        try:
            # Update engine's board reference
            self.engine.board = self.board
            
            # Get best move from engine
            best_move = None
            best_score = float('-inf')
            
            for move in self.board.legal_moves:
                score = self.engine.evaluate_move(move)
                if score > best_score:
                    best_score = score
                    best_move = move
                    
            if best_move:
                print(f"Engine plays: {self.board.san(best_move)} (score: {best_score:.2f})")
                self.board.push(best_move)
                
                # Check for game over
                if self.board.is_game_over():
                    self.game_over = True
                    
        except Exception as e:
            print(f"Engine error: {e}")
            # Fallback to random move
            import random
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                random_move = random.choice(legal_moves)
                print(f"Engine fallback random move: {self.board.san(random_move)}")
                self.board.push(random_move)
                
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        square = self.get_square_from_mouse(event.pos)
                        if square is not None:
                            self.handle_click(square)
                            
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # Reset game
                        self.board = chess.Board()
                        self.engine.board = self.board
                        self.selected_square = None
                        self.legal_moves = []
                        self.game_over = False
                        print("Game reset")
                        
            # Clear screen
            self.screen.fill((50, 50, 50))
            
            # Draw everything
            self.draw_board()
            self.draw_highlights()
            self.draw_pieces()
            self.draw_info()
            
            # Update display
            pygame.display.flip()
            clock.tick(60)
            
        pygame.quit()

def main():
    """Main function to start the chess GUI"""
    try:
        game = ChessGUI()
        game.run()
    except Exception as e:
        print(f"Error initializing chess GUI: {e}")
        print("\nMake sure you have:")
        print("1. pygame installed: pip install pygame")
        print("2. python-chess installed: pip install python-chess")
        print("3. evaluation_engine.py in the same directory")
        print("4. images directory with piece images (optional)")

if __name__ == "__main__":
    main()