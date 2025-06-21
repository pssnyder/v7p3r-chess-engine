# training/v7p3r_rl_training.py
# Training launcher for the V7P3R chess engine's reinforcement learning module.

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