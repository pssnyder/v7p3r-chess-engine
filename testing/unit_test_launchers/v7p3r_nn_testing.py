#!/usr/bin/env python3
"""
Unit tests for v7p3r_nn.py - v7p3r Chess Engine Neural Network Module

This module contains comprehensive unit tests for the v7p3rNeuralNetwork class,
its components (ChessNN, MoveLibrary, ChessPositionDataset), and their interactions.

Author: v7p3r Testing Suite
Date: 2025-06-24
"""

import sys
import os
import unittest
import tempfile
import shutil
import yaml
import chess
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from v7p3r_nn_engine.v7p3r_nn import (
    ChessPositionDataset,
    ChessNN,
    MoveLibrary,
    v7p3rNeuralNetwork
)

class TestChessPositionDataset(unittest.TestCase):
    """Tests for the ChessPositionDataset class."""

    def test_fen_to_features(self):
        """Test the conversion of a FEN string to a feature tensor."""
        dataset = ChessPositionDataset([], [])
        fen = chess.STARTING_FEN
        features = dataset._fen_to_features(fen)
        
        self.assertIsInstance(features, torch.Tensor)
        # 12 piece layers + 1 turn layer = 13
        self.assertEqual(features.shape, (13, 8, 8))
        
        # Check a known feature: e.g., a white pawn at e2
        # White pawn = index 0, e2 = row 1, col 4
        self.assertEqual(features[0, 1, 4], 1.0)
        # White to move layer should be all 1s
        self.assertTrue(torch.all(features[12] == 1.0))

class TestChessNN(unittest.TestCase):
    """Tests for the ChessNN model."""

    def test_forward_pass(self):
        """Test a forward pass through the network."""
        model = ChessNN()
        # Create a dummy input tensor (batch of 1)
        dummy_input = torch.randn(1, 13, 8, 8)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (1, 1))
        # Check if output is within Tanh range [-1, 1] scaled by 10
        self.assertTrue(-10.0 <= output.item() <= 10.0)

class TestMoveLibrary(unittest.TestCase):
    """Tests for the MoveLibrary class using an in-memory database."""

    def setUp(self):
        """Set up an in-memory SQLite database for each test."""
        self.move_lib = MoveLibrary(db_path=":memory:")

    def test_add_and_get_position(self):
        """Test adding and retrieving a position evaluation."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.move_lib.add_position(fen, 0.5, source="test")
        evaluation = self.move_lib.get_position_evaluation(fen)
        self.assertEqual(evaluation, 0.5)

    def test_add_and_get_best_move(self):
        """Test adding and retrieving a best move."""
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        self.move_lib.add_best_move(fen, "e7e5", 0.2, source="test", confidence=0.9)
        best_move_info = self.move_lib.get_best_move(fen)
        self.assertIsNotNone(best_move_info)
        if best_move_info:
            self.assertEqual(best_move_info['move'], "e7e5")
            self.assertEqual(best_move_info['evaluation'], 0.2)

    def tearDown(self):
        """Close the database connection."""
        self.move_lib.close()

class TestV7P3RNeuralNetwork(unittest.TestCase):
    """Tests for the main v7p3rNeuralNetwork class."""

    def setUp(self):
        """Set up a temporary directory and mock config."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_config = {
            'training': {
                'model': {'hidden_layers': [32, 16], 'dropout_rate': 0.1},
                'storage': {'enabled': True, 'model_path': self.temp_dir}
            },
            'move_library': {'db_path': os.path.join(self.temp_dir, 'test.db')}
        }

    @patch('v7p3r_nn_engine.v7p3r_nn.yaml.safe_load')
    @patch('v7p3r_nn_engine.v7p3r_nn.os.path.exists')
    def test_initialization_with_config(self, mock_exists, mock_safe_load):
        """Test engine initialization with a mocked config file."""
        mock_exists.return_value = True
        mock_safe_load.return_value = self.mock_config
        
        engine = v7p3rNeuralNetwork(config_path="dummy/path.yaml")
        
        self.assertIsNotNone(engine.model)
        self.assertIsNotNone(engine.move_library)
        self.assertEqual(engine.config, self.mock_config)
        engine.close()

    @patch('v7p3r_nn_engine.v7p3r_nn.v7p3rNeuralNetwork._load_config')
    @patch('v7p3r_nn_engine.v7p3r_nn.ChessNN')
    @patch('v7p3r_nn_engine.v7p3r_nn.MoveLibrary')
    def test_evaluate_position(self, MockMoveLibrary, MockChessNN, mock_load_config):
        """Test position evaluation logic, mocking the NN and DB."""
        mock_load_config.return_value = self.mock_config
        
        # Mock MoveLibrary instance and its methods
        mock_move_lib_instance = MockMoveLibrary.return_value
        mock_move_lib_instance.get_position_evaluation.return_value = None # Not in library
        
        # Mock ChessNN model and its output
        mock_model_instance = MockChessNN.return_value
        mock_model_instance.return_value = torch.tensor([[0.75]]) # Mocked NN output

        engine = v7p3rNeuralNetwork()
        engine.model = mock_model_instance
        engine.move_library = mock_move_lib_instance

        fen = chess.STARTING_FEN
        evaluation = engine.evaluate_position(fen)

        # Scaled by 10 in the forward pass
        self.assertAlmostEqual(evaluation, 7.5)
        # Verify it was added to the library
        mock_move_lib_instance.add_position.assert_called_with(fen, 7.5, source="nn")
        engine.close()

    @patch('v7p3r_nn_engine.v7p3r_nn.v7p3rNeuralNetwork._collect_training_data')
    @patch('v7p3r_nn_engine.v7p3r_nn.DataLoader')
    @patch('v7p3r_nn_engine.v7p3r_nn.optim.Adam')
    @patch('v7p3r_nn_engine.v7p3r_nn.v7p3rNeuralNetwork._save_model')
    def test_training_loop(self, mock_save, mock_adam, mock_dataloader, mock_collect_data):
        """Test the overall training process orchestration."""
        with patch('v7p3r_nn_engine.v7p3r_nn.v7p3rNeuralNetwork._load_config') as mock_load_config:
            mock_load_config.return_value = self.mock_config
            engine = v7p3rNeuralNetwork()

        # Mock data collection to return some dummy data
        mock_collect_data.return_value = (["fen1", "fen2"], [0.1, -0.2])
        # Mock DataLoader to be iterable
        mock_dataloader.return_value = [(
            torch.randn(2, 13, 8, 8), 
            torch.tensor([0.1, -0.2]))
        ]

        engine.train(pgn_files=["dummy.pgn"], epochs=1)

        mock_collect_data.assert_called_once()
        mock_dataloader.assert_called_once()
        mock_adam.assert_called_once()
        self.assertTrue(engine.model.train.called)
        mock_save.assert_called_once()
        engine.close()

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)

if __name__ == "__main__":
    unittest.main()