# testing/test_launchers/game_simulation_manager_testing.py
# Test script for the game simulation manager functionality of the v7p3r chess engine.

import os
import sys
import yaml
import unittest
from unittest.mock import patch, MagicMock

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engine_utilities import game_simulation_manager

class TestGameSimulationManager(unittest.TestCase):
    def setUp(self):
        # Use a minimal stub config for testing
        self.config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/simulation_config.yaml'))
        assert os.path.exists(self.config_path), f"Config file not found: {self.config_path}"
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

    @patch('engine_utilities.game_simulation_manager.ChessGame')
    def test_simulation_manager_runs_all_games(self, mock_ChessGame):
        # Patch ChessGame.run so it doesn't actually play games
        mock_game_instance = MagicMock()
        mock_ChessGame.return_value = mock_game_instance
        # Patch metrics_store.close to avoid side effects
        mock_game_instance.metrics_store.close = MagicMock()
        # Run the simulation scenarios
        game_simulation_manager.run_simulation_scenarios()
        # Count how many games should be scheduled
        expected_games = sum(sim.get('games_to_run', 1) for sim in self.config.get('simulations', []))
        # Each scenario instantiates ChessGame once and calls run() once
        self.assertEqual(mock_ChessGame.call_count, len(self.config.get('simulations', [])))
        self.assertEqual(mock_game_instance.run.call_count, len(self.config.get('simulations', [])))
        print(f"ChessGame instantiated {mock_ChessGame.call_count} times (expected {len(self.config.get('simulations', []))})")
        print(f"ChessGame.run called {mock_game_instance.run.call_count} times (expected {len(self.config.get('simulations', []))})")

if __name__ == "__main__":
    unittest.main()