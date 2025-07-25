import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ChessDataset(Dataset):
    def __init__(self, pgn_path, username):
        self.positions = []
        self.moves = []
        
        pgn = open(pgn_path)
        while True:
            game = chess.pgn.read_game(pgn)
            if not game:
                break
            
            if game.headers["White"] == username or game.headers["Black"] == username:
                board = game.board()
                for move in game.mainline_moves():
                    if (board.turn == chess.WHITE and game.headers["White"] == username) or \
                       (board.turn == chess.BLACK and game.headers["Black"] == username):
                        self.positions.append(self.board_to_tensor(board))
                        self.moves.append(move.uci())
                    board.push(move)

    def board_to_tensor(self, board):
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                tensor[channel][7 - square//8][square%8] = 1
        return tensor

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.moves[idx]

class ChessAI(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Convolutional layers for spatial pattern recognition
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # Fully connected layers for move prediction
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Value head for position evaluation
        self.value_fc1 = nn.Linear(64 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Initialize piece-square tables
        self.initialize_piece_tables()
        
        # Genetic parameters (will be evolved)
        self.genetic_params = {
            'material_weight': 1.0,
            'position_weight': 0.5,
            'search_depth': 2
        }
    
    def forward(self, x):
        # CNN forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x_flat = x.view(x.size(0), -1)
        
        # Policy head (move prediction)
        policy = F.relu(self.fc1(x_flat))
        policy = F.relu(self.fc2(policy))
        policy = self.fc3(policy)  # Raw logits
        
        # Value head (position evaluation)
        value = F.relu(self.value_fc1(x_flat))
        value = torch.tanh(self.value_fc2(value))  # Value between -1 and 1
        
        return policy, value

    def initialize_piece_tables(self):
        """Initialize piece-square tables for positional evaluation"""
        
        # Pawn position table (incentivizes center control and advancement)
        self.pawn_table = np.array([
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ])
        
        # Knight position table (prefers center, avoids edges)
        self.knight_table = np.array([
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ])
        
        # Bishop piece-square table (encourages diagonals and center)
        self.bishop_table = np.array([
            [-20, -10, -10, -10, -10, -10, -10, -20],
            [-10,   0,   0,   0,   0,   0,   0, -10],
            [-10,   0,   5,  10,  10,   5,   0, -10],
            [-10,   5,   5,  10,  10,   5,   5, -10],
            [-10,   0,  10,  10,  10,  10,   0, -10],
            [-10,  10,  10,  10,  10,  10,  10, -10],
            [-10,   5,   0,   0,   0,   0,   5, -10],
            [-20, -10, -10, -10, -10, -10, -10, -20]
        ]).flatten()
        
        # Rook piece-square table (encourages 7th rank and open files)
        self.rook_table = np.array([
            [  0,   0,   0,   0,   0,   0,   0,   0],
            [  5,  10,  10,  10,  10,  10,  10,   5],
            [ -5,   0,   0,   0,   0,   0,   0,  -5],
            [ -5,   0,   0,   0,   0,   0,   0,  -5],
            [ -5,   0,   0,   0,   0,   0,   0,  -5],
            [ -5,   0,   0,   0,   0,   0,   0,  -5],
            [ -5,   0,   0,   0,   0,   0,   0,  -5],
            [  0,   0,   0,   5,   5,   0,   0,   0]
        ]).flatten()
        
        # Queen piece-square table (combines bishop and rook patterns)
        self.queen_table = np.array([
            [-20, -10, -10,  -5,  -5, -10, -10, -20],
            [-10,   0,   0,   0,   0,   0,   0, -10],
            [-10,   0,   5,   5,   5,   5,   0, -10],
            [ -5,   0,   5,   5,   5,   5,   0,  -5],
            [  0,   0,   5,   5,   5,   5,   0,  -5],
            [-10,   5,   5,   5,   5,   5,   0, -10],
            [-10,   0,   5,   0,   0,   0,   0, -10],
            [-20, -10, -10,  -5,  -5, -10, -10, -20]
        ]).flatten()
        
        # King middle game table (encourages castling and safety)
        self.king_mg_table = np.array([
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-20, -30, -30, -40, -40, -30, -30, -20],
            [-10, -20, -20, -20, -20, -20, -20, -10],
            [ 20,  20,   0,   0,   0,   0,  20,  20],
            [ 20,  30,  10,   0,   0,  10,  30,  20]
        ]).flatten()
        
        # King endgame table (encourages activity)
        self.king_eg_table = np.array([
            [-50, -40, -30, -20, -20, -30, -40, -50],
            [-30, -20, -10,   0,   0, -10, -20, -30],
            [-30, -10,  20,  30,  30,  20, -10, -30],
            [-30, -10,  30,  40,  40,  30, -10, -30],
            [-30, -10,  30,  40,  40,  30, -10, -30],
            [-30, -10,  20,  30,  30,  20, -10, -30],
            [-30, -30,   0,   0,   0,   0, -30, -30],
            [-50, -30, -33, -30, -30, -30, -30, -50]
        ]).flatten()


    def evaluate_position(self, board):
        """Evaluate a chess position using material and piece-square tables"""
        if board.is_checkmate():
            # High value for checkmate
            return 10000 if board.turn == chess.WHITE else -10000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0  # Draw
        
        # Count material and evaluate positions
        material_score = 0
        position_score = 0
        
        # Standard piece values
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Evaluate all pieces on the board
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Material score
                value = piece_values[piece.piece_type]
                material_score += value if piece.color == chess.WHITE else -value
                
                # Position score based on piece-square tables
                position_value = self._get_position_value(piece, square)
                position_score += position_value if piece.color == chess.WHITE else -position_value
        
        # Weight the scores using genetic parameters
        total_score = (
            self.genetic_params['material_weight'] * material_score + 
            self.genetic_params['position_weight'] * position_score
        )
        
        return total_score if board.turn == chess.WHITE else -total_score

    def _is_endgame(self, board):
        """Simple endgame detection based on remaining material"""
        white_pieces = 0
        black_pieces = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type not in [chess.KING, chess.PAWN]:
                if piece.color == chess.WHITE:
                    white_pieces += 1
                else:
                    black_pieces += 1
        
        return white_pieces <= 3 or black_pieces <= 3

    def _get_position_value(self, piece, square):
        """Get position value for a piece on a given square using piece-square tables"""
        square_index = square
        
        if piece.piece_type == chess.PAWN:
            return self.pawn_table[square_index]
        elif piece.piece_type == chess.KNIGHT:
            return self.knight_table[square_index]
        elif piece.piece_type == chess.BISHOP:
            return self.bishop_table[square_index]
        elif piece.piece_type == chess.ROOK:
            return self.rook_table[square_index]
        elif piece.piece_type == chess.QUEEN:
            return self.queen_table[square_index]
        elif piece.piece_type == chess.KING:
            # Adaptive king evaluation based on game phase
            if hasattr(self, '_current_board') and self._is_endgame(self._current_board):
                return self.king_eg_table[square_index]
            else:
                return self.king_mg_table[square_index]
        
        return 0
    
    def select_move(self, board):
        """Select best move using minimax with alpha-beta pruning"""
        depth = self.genetic_params['search_depth']
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0
        
        # Use minimax to find best move
        best_move = None
        best_score = float('-inf')
        
        for move in legal_moves:
            # Make move
            board.push(move)
            
            # Evaluate position using minimax
            score = -self._minimax(board, depth-1, float('-inf'), float('inf'), False)
            
            # Undo move
            board.pop()
            
            # Update best move if better score
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move, best_score

    def _minimax(self, board, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with alpha-beta pruning"""
        # Terminal node
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board)
        
        # Maximizing player
        if maximizing_player:
            value = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                value = max(value, self._minimax(board, depth-1, alpha, beta, False))
                board.pop()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # Beta cutoff
            return value
        
        # Minimizing player
        else:
            value = float('inf')
            for move in board.legal_moves:
                board.push(move)
                value = min(value, self._minimax(board, depth-1, alpha, beta, True))
                board.pop()
                beta = min(beta, value)
                if beta <= alpha:
                    break  # Alpha cutoff
            return value

    def select_fallback(self, board):
        """Fallback move selection using simple evaluation"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Evaluate each move with a simple 1-ply search
        best_move = None
        best_score = float('-inf')
        
        for move in legal_moves:
            board.push(move)
            score = -self.evaluate_position(board)
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
