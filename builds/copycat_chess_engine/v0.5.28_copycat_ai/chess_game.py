import sys
import os
from chess_core import ChessAI, ChessDataset
import chess
import chess.pgn
import torch
import numpy as np
import pygame
import random

# Pygame constants
WIDTH, HEIGHT = 640, 640
DIMENSION = 8
SQ_SIZE = WIDTH // DIMENSION
MAX_FPS = 15
IMAGES = {}

# Resource path config for distro
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class ChessGame:
    def __init__(self, model_path, username="User"):
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('v7p3r Chess AI')
        self.clock = pygame.time.Clock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize chess components
        self.board = chess.Board()
        self.selected_square = None
        self.player_clicks = []
        self.load_images()

         # Load move vocabulary
        import pickle
        with open("move_vocab.pkl", "rb") as f:
            self.move_to_index = pickle.load(f)
        
        # Initialize model
        self.model = ChessAI(num_classes=len(self.move_to_index)).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=False)
        )
        self.model.eval()

        # Game recording, headers dynamically set later
        self.game = chess.pgn.Game()
        self.username = username
        self.ai_color = None  # Will be set in run()
        self.human_color = None
        self.game_node = self.game
        
        # Turn management
        self.ai_turn = False
        self.last_ai_move = None  # Track AI's last move

    def load_images(self):
        pieces = ['wp', 'wN', 'wb', 'wr', 'wq', 'wk', 
                 'bp', 'bN', 'bb', 'br', 'bq', 'bk']
        for piece in pieces:
            IMAGES[piece] = pygame.transform.scale(
                pygame.image.load(resource_path(f"images/{piece}.png")), 
                (SQ_SIZE, SQ_SIZE)
            )

    def board_to_tensor(self, board):
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                tensor[channel][7 - square//8][square%8] = 1
        return torch.tensor(tensor, device=self.device).unsqueeze(0)

    def ai_move(self):
        with torch.no_grad():
            tensor = self.board_to_tensor(self.board)
            output = self.model(tensor)
            legal_moves = [m.uci() for m in self.board.legal_moves]
            
            # Filter moves to only those in vocabulary
            valid_legal_moves = [m for m in legal_moves if m in self.move_to_index]
            
            if not valid_legal_moves:
                # Fallback to random legal move if no known moves
                print("No known moves, using random legal move")
                fallback_move = np.random.choice(legal_moves)
                self.board.push_uci(fallback_move)
                return fallback_move
                
            legal_indices = [self.move_to_index[m] for m in valid_legal_moves]
            
            # Get probabilities only for valid moves
            valid_probs = output[0, legal_indices]
            best_idx = torch.argmax(valid_probs)
            best_move = valid_legal_moves[best_idx.item()]
            move_obj = chess.Move.from_uci(best_move)
            self.board.push(move_obj)
            self.last_ai_move = move_obj.to_square  # Store destination square
            return best_move

    def draw_board(self):
        # Set board colors to dark theme (white: #d8d9d8, black: #a8a9a8)
        colors = [pygame.Color("#d8d9d8"), pygame.Color("#a8a9a8")]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                # Flip row/col if needed
                draw_r = 7 - r if self.flip_board else r
                draw_c = 7 - c if self.flip_board else c
                color = colors[(draw_r + draw_c) % 2]
                pygame.draw.rect(self.screen, color,
                                pygame.Rect(draw_c*SQ_SIZE, draw_r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def draw_pieces(self):
        # Highlight AI's last move
        if self.last_ai_move:
            file = chess.square_file(self.last_ai_move)
            rank = chess.square_rank(self.last_ai_move)
            if self.flip_board:
                file = 7 - file
                rank = 7 - rank
            s = pygame.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)
            s.fill(pygame.Color('yellow'))
            self.screen.blit(s, (file*SQ_SIZE, (7-rank)*SQ_SIZE))
            
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                board_r = 7 - r if self.flip_board else r
                board_c = 7 - c if self.flip_board else c
                square = chess.square(board_c, 7 - board_r)
                piece = self.board.piece_at(square)
                if piece:
                    color = 'w' if piece.color == chess.WHITE else 'b'
                    piece_type = piece.symbol().upper()
                    img_key = f"{color}{piece_type.lower()}" if piece_type != 'N' else f"{color}N"
                    self.screen.blit(IMAGES[img_key],
                                    pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def handle_mouse_click(self, pos):
        col = pos[0] // SQ_SIZE
        row = pos[1] // SQ_SIZE
        if self.flip_board:
            col = 7 - col
            row = 7 - row
        
        # Convert pygame coordinates to chess board coordinates
        square = chess.square(col, 7 - row)  # Flip row for chess board
        
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
        else:
            move = chess.Move(self.selected_square, square)
            
            # Check for pawn promotion
            if (self.board.piece_at(self.selected_square).piece_type == chess.PAWN and 
                ((chess.square_rank(square) == 7 and self.board.turn == chess.WHITE) or 
                (chess.square_rank(square) == 0 and self.board.turn == chess.BLACK))):
                move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
            
            if move in self.board.legal_moves:
                self.board.push(move)
                self.game_node = self.game_node.add_variation(move)
                self.ai_turn = True
            self.selected_square = None


    def run(self):
        #user_color = input("Play as (w)hite or (b)lack? ").lower()
        user_color = random.choice(['w','b'])
        self.flip_board = (user_color == 'b')
        
        # Set PGN headers based on color choice
        if user_color == 'b':
            self.ai_color = chess.WHITE
            self.human_color = chess.BLACK
            self.game.headers["White"] = "v7p3r_chess_ai"
            self.game.headers["Black"] = self.username
        else:
            self.ai_color = chess.BLACK
            self.human_color = chess.WHITE
            self.game.headers["White"] = self.username
            self.game.headers["Black"] = "v7p3r_chess_ai"
        
        if user_color == 'b':
            self.ai_turn = True  # AI plays first if user is black

        running = True
        while running:
            self.draw_board()
            self.draw_pieces()
            
            # Highlight selected square (fixed for flipped board)
            if self.selected_square is not None:
                file = chess.square_file(self.selected_square)
                rank = chess.square_rank(self.selected_square)
                if self.flip_board:
                    file = 7 - file
                    rank = 7 - rank
                s = pygame.Surface((SQ_SIZE, SQ_SIZE))
                s.set_alpha(100)
                s.fill(pygame.Color('blue'))
                self.screen.blit(s, (file*SQ_SIZE, (7-rank)*SQ_SIZE))
            
            pygame.display.flip()
            
            # Handle AI turn
            if self.ai_turn and not self.board.is_game_over():
                ai_move = self.ai_move()
                print(f"AI plays: {ai_move}")
                self.ai_turn = False
                pygame.display.flip()
                pygame.time.wait(500)  # 0.5 second pause
                self.last_ai_move = None
                continue  # Skip event handling during AI turn

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.ai_turn:
                    self.handle_mouse_click(pygame.mouse.get_pos())

            self.clock.tick(MAX_FPS)

            if self.board.is_game_over():
                print(f"\nGame over: {self.board.result()}")
                self.save_pgn()
                running = False

            if self.ai_turn and not self.board.is_game_over():
                ai_move = self.ai_move()
                print(f"AI plays: {ai_move}")
                self.ai_turn = False
                pygame.display.flip()  # Update display
                pygame.time.wait(500)  # 0.5 second pause
                self.last_ai_move = None  # Clear highlight
                
        pygame.quit()


    def save_pgn(self, filename="ai_game.pgn"):
        with open(filename, "w") as f:
            exporter = chess.pgn.FileExporter(f)
            self.game.accept(exporter)

if __name__ == "__main__":
    game = ChessGame("v7p3r_chess_ai_model.pth")
    game.run()
