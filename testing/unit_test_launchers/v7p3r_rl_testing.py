#!/usr/bin/env python3
"""
Unit tests for v7p3r_rl.py - v7p3r Chess Engine Reinforcement Learning Engine

This module contains unit tests for the v7p3rRLEngine and V7P3RRLAgent classes,
including policy network, action selection, reward storage, and integration with the scoring function.

Author: v7p3r Testing Suite
Date: 2025-06-25
"""

import sys
import os
import unittest
import tempfile
import shutil
import chess
import torch
import numpy as np
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from v7p3r_rl_engine.v7p3r_rl import (
    PolicyNetwork,
    V7P3RRLAgent,
    v7p3rRLEngine
)

class TestPolicyNetwork(unittest.TestCase):
    def test_forward_output_shape(self):
        net = PolicyNetwork(input_dim=773, hidden_dim=32, output_dim=100)
        x = torch.randn(773)
        out = net(x)
        self.assertEqual(out.shape, (100,))
        self.assertAlmostEqual(out.sum().item(), 1.0, places=5)

class TestV7P3RRLAgent(unittest.TestCase):
    @patch('v7p3r_rl_engine.v7p3r_rl.v7p3rScoringCalculation')
    @patch('v7p3r_rl_engine.v7p3r_rl.PieceSquareTables')
    def setUp(self, MockPST, MockScorer):
        # Patch config loading to avoid file IO
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, mode='w')
        self.temp_config.write('v7p3r:\n  ruleset: default_evaluation\n')
        self.temp_config.close()
        self.agent = V7P3RRLAgent(config_path=self.temp_config.name, v7p3r_config_path=self.temp_config.name)
        self.agent.scorer = Mock()
        self.agent.scorer.calculate_score.return_value = 1.0

    def tearDown(self):
        os.unlink(self.temp_config.name)

    def test_board_to_tensor_shape(self):
        board = chess.Board()
        tensor = self.agent.board_to_tensor(board)
        self.assertEqual(tensor.shape, (13*8*8,))

    def test_select_action_returns_legal_move(self):
        board = chess.Board()
        move = self.agent.select_action(board)
        self.assertIn(move, list(board.legal_moves))

    def test_store_and_finish_episode(self):
        self.agent.saved_log_probs = [torch.tensor(0.0), torch.tensor(0.0)]
        self.agent.rewards = [1.0, 1.0]
        self.agent.optimizer = Mock()
        self.agent.optimizer.zero_grad = Mock()
        self.agent.optimizer.step = Mock()
        self.agent.finish_episode()
        self.assertEqual(self.agent.saved_log_probs, [])
        self.assertEqual(self.agent.rewards, [])

    def test_play_game_runs(self):
        result = self.agent.play_game(max_moves=5)
        self.assertIn(result, ["1-0", "0-1", "1/2-1/2", "*"])

class TestV7p3rRLEngine(unittest.TestCase):
    @patch('v7p3r_rl_engine.v7p3r_rl.V7P3RRLAgent')
    def test_search_and_eval(self, MockAgent):
        mock_agent = MockAgent.return_value
        mock_agent.select_action.return_value = chess.Move.from_uci("e2e4")
        mock_agent.scorer.calculate_score.return_value = 1.0
        engine = v7p3rRLEngine(config_path="dummy.yaml")
        board = chess.Board()
        move = engine.search(board, chess.WHITE)
        self.assertEqual(move, chess.Move.from_uci("e2e4"))
        score = engine.evaluate_position_from_perspective(board, chess.WHITE)
        self.assertEqual(score, 1.0)

if __name__ == "__main__":
    unittest.main()
