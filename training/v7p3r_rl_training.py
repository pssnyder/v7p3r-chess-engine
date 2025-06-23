# training/v7p3r_rl_training.py
# Training launcher for the v7p3r chess engine's reinforcement learning module.

import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import numpy as np
import yaml
import pickle
import os
from v7p3r_rl_engine.v7p3r_rl import ChessDataset, ReinforcementLearningAlgorithm, ChessAI

torch.backends.cudnn.benchmark = True  # Enable CuDNN auto-tuner

# Load configuration
with open("../config/v7p3r_ga_config.yaml") as f:
    config = yaml.safe_load(f)

# Enable CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")