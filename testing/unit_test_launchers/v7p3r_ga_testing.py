#!/usr/bin/env python3
"""
Unit tests for v7p3r_ga.py - v7p3r Chess Engine Genetic Algorithm Module

This module contains unit tests for the GeneticAlgorithm class, testing
initialization, population management, fitness evaluation, and evolution processes.

Author: v7p3r Testing Suite
Date: 2025-06-24
"""

import sys
import os
import unittest
import chess
import chess.pgn
from unittest.mock import Mock, patch, MagicMock, call

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from v7p3r_ga_engine.v7p3r_ga import GeneticAlgorithm, ChessAI

class TestGeneticAlgorithm(unittest.TestCase):
    """Test suite for the GeneticAlgorithm class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.ga = GeneticAlgorithm(population_size=10, mutation_rate=0.1, elite_count=2)
        
        # Mock the ChessAI model template
        self.mock_model_template = Mock(spec=ChessAI)
        self.mock_model_template.genetic_params = {
            'material_weight': 1.0,
            'position_weight': 0.5,
            'search_depth': 2
        }
        # Add select_move to the spec of the mock
        self.mock_model_template.select_move = Mock(return_value=(chess.Move.from_uci("a1a2"), 0.1))

    def test_initialization(self):
        """Test the initialization of the GeneticAlgorithm class."""
        self.assertEqual(self.ga.population_size, 10)
        self.assertEqual(self.ga.mutation_rate, 0.1)
        self.assertEqual(self.ga.elite_count, 2)
        self.assertEqual(self.ga.population, [])

    @patch('v7p3r_ga_engine.v7p3r_ga.copy.deepcopy')
    @patch('v7p3r_ga_engine.v7p3r_ga.random.uniform')
    @patch('v7p3r_ga_engine.v7p3r_ga.random.randint')
    def test_initialize_population(self, mock_randint, mock_uniform, mock_deepcopy):
        """Test the population initialization process."""
        # Configure mocks
        # Create a new mock for each deepcopy call to avoid modifying the original
        mock_model_copies = []
        for i in range(10):
            mock_copy = Mock(spec=ChessAI)
            mock_copy.genetic_params = {}
            mock_model_copies.append(mock_copy)
        
        mock_deepcopy.side_effect = mock_model_copies
        mock_uniform.side_effect = [1.1, 0.6] * 10 # material_weight, position_weight
        mock_randint.side_effect = [3] * 10 # search_depth

        # Call the method
        self.ga.initialize_population(self.mock_model_template)

        # Assertions
        self.assertEqual(len(self.ga.population), 10)
        self.assertEqual(mock_deepcopy.call_count, 10)
        # Check if genetic params were randomized for each individual
        for model in self.ga.population:
            self.assertIsNotNone(model.genetic_params)
            self.assertIn('material_weight', model.genetic_params)
            self.assertIn('position_weight', model.genetic_params)
            self.assertIn('search_depth', model.genetic_params)

    def create_mock_game(self, white, black, result, moves):
        """Helper to create a mock PGN game."""
        game = chess.pgn.Game()
        game.headers["White"] = white
        game.headers["Black"] = black
        game.headers["Result"] = result
        
        node = game
        for move_uci in moves:
            node = node.add_variation(chess.Move.from_uci(move_uci))
        return game

    def test_evaluate_fitness(self):
        """Test the fitness evaluation logic."""
        mock_model = Mock(spec=ChessAI)
        
        # Mock games
        game1 = self.create_mock_game("v7p3r", "opponent", "1-0", ["e2e4", "e7e5"]) # Win
        game2 = self.create_mock_game("opponent", "v7p3r", "0-1", ["d2d4", "d7d5"]) # Loss for white, but win for black v7p3r
        game3 = self.create_mock_game("v7p3r", "opponent", "1/2-1/2", ["c2c4", "c7c5"]) # Draw
        games = [game1, game2, game3]

        # Mock model's move prediction
        def select_move_side_effect(board):
            # If the board is in the starting position, predict the correct first move for game1
            if board.fen() == chess.STARTING_FEN:
                return chess.Move.from_uci("e2e4"), 0.9
            # Otherwise, for any other position, return a non-matching move to test the logic
            return chess.Move.from_uci("a1a2"), 0.1
        
        mock_model.select_move.side_effect = select_move_side_effect

        # Calculate fitness
        fitness = self.ga.evaluate_fitness(mock_model, games)

        # Expected fitness:
        # Game 1 (Win): 10 points + 2 points (correct move prediction) = 12
        # Game 2 (Win for black): 10 points
        # Game 3 (Draw): 2 points
        # Total = 24
        self.assertEqual(fitness, 24)

    @patch('v7p3r_ga_engine.v7p3r_ga.GeneticAlgorithm._mutate')
    @patch('v7p3r_ga_engine.v7p3r_ga.GeneticAlgorithm._crossover')
    @patch('v7p3r_ga_engine.v7p3r_ga.GeneticAlgorithm._tournament_selection')
    @patch('v7p3r_ga_engine.v7p3r_ga.GeneticAlgorithm.evaluate_fitness')
    def test_evolve_population(self, mock_evaluate, mock_selection, mock_crossover, mock_mutate):
        """Test the main population evolution process."""
        # Setup mocks
        initial_population = [Mock(spec=ChessAI) for _ in range(10)]
        self.ga.population = initial_population
        
        mock_evaluate.side_effect = list(range(10, 0, -1)) # Fitness scores 10 down to 1
        
        # The top 2 are elite
        elite_models = initial_population[:2]
        
        # Mock selection to return parents for crossover
        # This will be called 16 times for 8 children
        mock_selection.side_effect = [initial_population[i % 8 + 2] for i in range(16)]

        # Mock crossover to return new children
        mock_crossover.side_effect = [Mock(spec=ChessAI) for _ in range(8)]
        mock_mutate.side_effect = lambda x: x # Identity function for mutation

        # Run evolution
        self.ga.evolve_population(games=[])

        # Assertions
        self.assertEqual(len(self.ga.population), 10) # Population size is maintained
        self.assertEqual(mock_evaluate.call_count, 10) # Fitness evaluated for each individual
        
        # Check that elites are carried over
        for elite in elite_models:
            self.assertIn(elite, self.ga.population)
            
        self.assertEqual(mock_crossover.call_count, 8) 
        self.assertEqual(mock_mutate.call_count, 8)

if __name__ == "__main__":
    unittest.main()