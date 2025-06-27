# v7p3r_rl_engine/v7p3r_rl.py
# v7p3r Chess Engine Reinforcement Learning (Policy Gradient) Engine
"""
This engine implements a policy-gradient-based RL chess agent for the v7p3r project.
It starts from a config (as output by v7p3r_ga or hand-tuned), uses the scoring function
for reward/penalty, and learns to improve its policy through self-play.

- PyTorch-based, modular, and compatible with the v7p3r pipeline.
- Stores learned policy in model weights, but can output a versioned config for reference.
- Designed to be selectable as 'v7p3r_rl' in chess_game.py and simulation manager.

Author: v7p3r Project
Date: 2025-06-25
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import numpy as np
from torch.distributions import Categorical
from v7p3r_engine.v7p3r_score import v7p3rScore
from v7p3r_engine.v7p3r_pst import v7p3rPST

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

class V7P3RRLAgent:
    def __init__(self, config_path="config/v7p3r_rl_config.yaml", v7p3r_config_path="config/v7p3r_config.yaml"):
        self.config = self._load_config(config_path)
        self.v7p3r_config = self._load_config(v7p3r_config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load RL hyperparameters from config, with defaults
        self.input_dim = self.config.get('input_dim', 773)
        self.hidden_dim = self.config.get('hidden_dim', 128)
        self.output_dim = self.config.get('output_dim', 4672)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.gamma = self.config.get('gamma', 0.99)
        self.max_moves = self.config.get('max_moves', 200)
        self.policy_net = PolicyNetwork(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.saved_log_probs = []
        self.rewards = []
        # Setup for scoring calculation
        self.engine_config = self.config.get('v7p3r', self.config)  # fallback to config if not nested
        self.scorer = v7p3rScore(self.engine_config, self.v7p3r_config)

    def _load_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def board_to_tensor(self, board):
        # 12x8x8 binary planes for pieces + 13th plane for turn
        tensor = np.zeros((13, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                tensor[channel][7 - square//8][square%8] = 1
        tensor[12, :, :] = 1 if board.turn == chess.WHITE else 0
        return torch.tensor(tensor, dtype=torch.float32).flatten()

    def select_action(self, board):
        state = self.board_to_tensor(board).to(self.device)
        legal_moves = list(board.legal_moves)
        move_indices = [self.move_to_index(m) for m in legal_moves]
        logits = self.policy_net(state)
        logits = logits[move_indices]
        m = Categorical(logits)
        action_idx = m.sample()
        self.saved_log_probs.append(m.log_prob(action_idx))
        return legal_moves[int(action_idx.item())]

    def move_to_index(self, move):
        # Map move to a unique index (e.g., UCI string to integer)
        # For simplicity, use chess.Move.uci() and a hash table
        # In production, use a fixed move ordering
        return hash(move.uci()) % 4672

    def index_to_move(self, board, idx):
        # Map index back to a legal move
        legal_moves = list(board.legal_moves)
        for m in legal_moves:
            if self.move_to_index(m) == idx:
                return m
        return legal_moves[0]  # fallback

    def store_reward(self, reward):
        self.rewards.append(reward)

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.saved_log_probs = []
        self.rewards = []

    def play_game(self, max_moves=None):
        board = chess.Board()
        max_moves = max_moves if max_moves is not None else self.max_moves
        while not board.is_game_over() and board.fullmove_number < max_moves:
            move = self.select_action(board)
            board.push(move)
            # Use the scoring function for the current player
            reward = self.scorer.calculate_score(board, board.turn)
            self.store_reward(reward)
        self.finish_episode()
        return board.result()

    def save_policy(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_policy(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))

    def export_config(self, path):
        # Export current config and policy weights for reference
        export = {
            'config': self.config,
            'policy_state_dict': self.policy_net.state_dict()
        }
        torch.save(export, path)

# Entry point for integration with chess_game.py
class v7p3rRLEngine:
    def __init__(self, config_path="config/v7p3r_rl_config.yaml"):
        self.agent = V7P3RRLAgent(config_path)

    def reset(self, board=None):
        pass  # For compatibility

    def evaluate_position_from_perspective(self, board, perspective):
        # For compatibility with chess_game.py
        return self.agent.scorer.calculate_score(board, perspective)

    def search(self, board, player_color, engine_config=None):
        return self.agent.select_action(board)

    def train_self_play(self, episodes=100):
        for _ in range(episodes):
            self.agent.play_game()

    def save(self, path):
        self.agent.save_policy(path)

    def load(self, path):
        self.agent.load_policy(path)

    def export_config(self, path):
        self.agent.export_config(path)
