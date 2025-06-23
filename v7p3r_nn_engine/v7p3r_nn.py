# v7p3r_nn_engine/v7p3r_nn.py
# V7P3R Chess Engine Neural Network Module
# TODO: refactor this code pulled from another project to fit the goals set out in the enhancement issue #82 [V7P3R AI Models] Phase 1: Neural Network Engine: Build and train v7p3r_nn.py for experience-based learning
import os
import chess
import chess.pgn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ChessDataset(Dataset):
    def __init__(self, pgn_path):
        self.positions = []
        self.moves = []

        # Load all PGN files in directory
        pgn_files = [os.path.join(pgn_path, f) for f in os.listdir(pgn_path) if f.endswith('.pgn')]
        
        for pgn_path in pgn_files:
            with open(pgn_path) as pgn:
                while True:
                    game = chess.pgn.read_game(pgn)
                    if not game:
                        break
                    
                    board = game.board()
                    for move in game.mainline_moves():
                        # Store all positions and moves regardless of player
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
        self.conv1 = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256*8*8, num_classes)  # Dynamic output size
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 256*8*8)
        return self.fc(x)
