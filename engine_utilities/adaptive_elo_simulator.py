# engine_utilities/adaptive_elo_simulator.py
"""
Adaptive ELO Simulator for V7P3R Chess Engine

This module provides functionality to adaptively adjust Stockfish ELO based on 
game results, helping to find the true strength of the V7P3R engine configuration.
"""

import os
import uuid
import yaml
import json
import math
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy
from chess_game import ChessGame
from engine_utilities.engine_db_manager import EngineDBClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveEloSimulator:
    """
    A simulation manager that adaptively adjusts Stockfish ELO ratings based on 
    game results to find the approximate ELO strength of the V7P3R engine.
    """
    
    def __init__(self, 
                 initial_elo: int = 1500, 
                 min_elo: int = 800, 
                 max_elo: int = 3200,
                 adjustment_factor: float = 1.0,
                 convergence_threshold: float = 0.05,
                 min_games_for_convergence: int = 20,
                 max_games: int = 100,
                 v7p3r_config: Optional[Dict] = None,
                 game_config: Optional[Dict] = None,
                 use_central_storage: bool = False):
        """
        Initialize the adaptive ELO simulator.
        
        Args:
            initial_elo: Starting ELO for Stockfish
            min_elo: Minimum ELO to test
            max_elo: Maximum ELO to test
            adjustment_factor: Controls how aggressively ELO changes (higher = more aggressive)
            convergence_threshold: When win rate stabilizes within this percentage, consider converged
            min_games_for_convergence: Minimum number of games before checking for convergence
            max_games: Maximum number of games to play
            v7p3r_config: Configuration overrides for V7P3R engine
            game_config: Configuration overrides for chess game
            use_central_storage: Whether to use central database storage
        """
        self.current_elo = initial_elo
        self.min_elo = min_elo
        self.max_elo = max_elo
        self.adjustment_factor = adjustment_factor
        self.convergence_threshold = convergence_threshold
        self.min_games_for_convergence = min_games_for_convergence
        self.max_games = max_games
        
        # Load base configurations
        with open("config/chess_game.yaml") as f:
            self.base_game_config = yaml.safe_load(f)
        with open("config/v7p3r.yaml") as f:
            self.base_v7p3r_config = yaml.safe_load(f)
        with open("config/stockfish_handler.yaml") as f:
            self.base_stockfish_config = yaml.safe_load(f)
            
        # Apply overrides
        self.v7p3r_config = self._deep_merge(v7p3r_config or {}, deepcopy(self.base_v7p3r_config))
        self.game_config = self._deep_merge(game_config or {}, deepcopy(self.base_game_config))
        
        # Initialize stockfish config with the initial ELO
        self.stockfish_config = deepcopy(self.base_stockfish_config)
        if 'stockfish_config' not in self.stockfish_config:
            self.stockfish_config['stockfish_config'] = {}
        self.stockfish_config['stockfish_config']['elo_rating'] = self.current_elo
        self.stockfish_config['stockfish_config']['uci_limit_strength'] = True
        
        # Initialize game history and stats
        self.games_played = 0
        self.games_won = 0
        self.games_lost = 0
        self.games_drawn = 0
        self.game_history = []
        self.elo_history = []
        self.win_rate_history = []
        
        # Create a unique simulation ID
        self.simulation_id = f"elo_finder_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set up central storage if requested
        self.use_central_storage = use_central_storage
        self.db_client = None
        if use_central_storage:
            try:
                self.db_client = EngineDBClient()
                logger.info(f"DB client initialized for ELO finder simulation {self.simulation_id}")
            except Exception as e:
                logger.error(f"Failed to initialize DB client: {e}")
    
    def _deep_merge(self, source, destination):
        """Deep merge two dictionaries."""
        for key, value in source.items():
            if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                self._deep_merge(value, destination[key])
            else:
                destination[key] = value
        return destination
    
    def _calculate_elo_adjustment(self, result: str) -> int:
        """
        Calculate the ELO adjustment based on the game result.
        Uses a logarithmic scale to make larger adjustments early on and smaller ones as it converges.
        
        Args:
            result: Game result string ('1-0' for V7P3R win, '0-1' for loss, '1/2-1/2' for draw)
            
        Returns:
            The ELO adjustment value (positive means increase, negative means decrease)
        """
        base_adjustment = 100 * self.adjustment_factor
        
        # Logarithmic decay of adjustment size as more games are played
        decay_factor = max(0.2, 1.0 / math.log2(2 + self.games_played / 2))
        
        if result == '1-0':  # V7P3R won as white
            # Increase stockfish ELO, but more aggressively at first and less later
            return int(base_adjustment * decay_factor)
        elif result == '0-1':  # V7P3R lost as white
            # Decrease stockfish ELO
            return int(-base_adjustment * decay_factor)
        else:  # Draw
            # Small increase - draws against higher-rated opponents suggest strength
            return int(base_adjustment * 0.2 * decay_factor)
    
    def _update_elo(self, result: str) -> None:
        """
        Update the current ELO based on the game result.
        
        Args:
            result: Game result string ('1-0' for V7P3R win, '0-1' for loss, '1/2-1/2' for draw)
        """
        adjustment = self._calculate_elo_adjustment(result)
        new_elo = max(self.min_elo, min(self.max_elo, self.current_elo + adjustment))
        
        logger.info(f"Game result: {result}. Adjusting ELO from {self.current_elo} to {new_elo} (change: {adjustment})")
        self.current_elo = new_elo
        
        # Update stockfish config with new ELO
        self.stockfish_config['stockfish_config']['elo_rating'] = self.current_elo
    
    def _check_convergence(self) -> bool:
        """
        Check if the ELO has converged to a stable value.
        
        Returns:
            True if converged, False otherwise
        """
        if self.games_played < self.min_games_for_convergence:
            return False
        
        # Check the recent win rate stability
        window_size = min(10, self.games_played // 2)
        if window_size < 5:
            return False
            
        # Calculate win rates for two consecutive windows
        recent_results = self.game_history[-window_size*2:]
        if len(recent_results) < window_size*2:
            return False
            
        first_window = recent_results[:window_size]
        second_window = recent_results[window_size:window_size*2]
        
        first_win_rate = sum(1 for r in first_window if r == '1-0') / window_size
        second_win_rate = sum(1 for r in second_window if r == '1-0') / window_size
        
        # Check if ELO is also stable
        recent_elos = self.elo_history[-window_size:]
        elo_std_dev = 0
        if len(recent_elos) >= 5:
            mean_elo = sum(recent_elos) / len(recent_elos)
            elo_std_dev = math.sqrt(sum((elo - mean_elo) ** 2 for elo in recent_elos) / len(recent_elos))
        
        # Consider converged if win rate difference is small and ELO is stable
        win_rate_diff = abs(first_win_rate - second_win_rate)
        logger.info(f"Convergence check: Win rate diff={win_rate_diff:.3f}, ELO std dev={elo_std_dev:.1f}")
        
        return (win_rate_diff < self.convergence_threshold and 
                elo_std_dev < 50 and 
                self.games_played >= self.min_games_for_convergence)
    
    def _create_data_collector(self, game_id: str):
        """Create a data collector function for the chess game."""
        def data_collector(data_type, data):
            """Collect data from the chess game and send to central storage if enabled."""
            # Always save locally first
            if data_type == 'game_result':
                # Write PGN to file
                if 'game_pgn' in data:
                    os.makedirs('games', exist_ok=True)
                    with open(f"games/{game_id}.pgn", 'w') as f:
                        f.write(data['game_pgn'])
                
                # Write metadata to YAML
                metadata = {k: v for k, v in data.items() if k != 'game_pgn'}
                metadata['current_elo'] = self.current_elo
                metadata['games_played'] = self.games_played
                metadata['win_rate'] = self.games_won / self.games_played if self.games_played > 0 else 0
                
                with open(f"games/{game_id}.yaml", 'w') as f:
                    yaml.dump(metadata, f)
                    
            elif data_type == 'move_metric':
                # Append to move metrics file
                os.makedirs('games', exist_ok=True)
                with open(f"games/{game_id}_moves.json", 'a') as f:
                    f.write(json.dumps(data) + '\n')
            
            # Send to central storage if enabled
            if self.db_client:
                try:
                    if data_type == 'game_result':
                        # Add ELO information to the data
                        data['stockfish_elo'] = self.current_elo
                        data['elo_adjustment'] = self._calculate_elo_adjustment(data.get('result', ''))
                        data['simulation_type'] = 'elo_finder'
                        self.db_client.send_game_data(data)
                    elif data_type == 'move_metric':
                        self.db_client.send_move_data(data)
                except Exception as e:
                    logger.error(f"Failed to send {data_type} data: {e}")
        
        return data_collector
    
    def run_simulation(self) -> Dict:
        """
        Run the adaptive ELO simulation.
        
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Starting ELO Finder simulation {self.simulation_id} with initial ELO {self.current_elo}")
        
        # Save simulation metadata
        simulation_metadata = {
            'id': self.simulation_id,
            'type': 'elo_finder',
            'timestamp': datetime.datetime.now().isoformat(),
            'initial_elo': self.current_elo,
            'min_elo': self.min_elo,
            'max_elo': self.max_elo,
            'adjustment_factor': self.adjustment_factor,
            'convergence_threshold': self.convergence_threshold,
            'min_games_for_convergence': self.min_games_for_convergence,
            'max_games': self.max_games,
            'v7p3r_config': self.v7p3r_config,
            'game_config': self.game_config,
        }
        
        os.makedirs('games', exist_ok=True)
        with open(f"games/{self.simulation_id}_metadata.yaml", 'w') as f:
            yaml.dump(simulation_metadata, f)
        
        # Send metadata to central storage if enabled
        if self.use_central_storage and self.db_client:
            try:
                self.db_client.send_raw_simulation(simulation_metadata)
            except Exception as e:
                logger.error(f"Failed to send simulation metadata: {e}")
        
        converged = False
        
        # Run games until convergence or max games reached
        while self.games_played < self.max_games and not converged:
            # Create unique game ID
            game_id = f"{self.simulation_id}_{self.games_played + 1}"
            
            # Log current state
            logger.info(f"Starting game {self.games_played + 1}/{self.max_games} with Stockfish ELO: {self.current_elo}")
            
            # Set up the game
            game_config = deepcopy(self.game_config)
            game_config['game_id'] = game_id
            
            # Make sure V7P3R plays white (for consistency in the adaptive algorithm)
            game_config['white_ai_config'] = {'engine': 'v7p3r', 'ai_type': 'deepsearch'}
            game_config['black_ai_config'] = {'engine': 'stockfish'}
            
            # Create and run the game
            game = ChessGame(
                game_config=game_config,
                v7p3r_config=deepcopy(self.v7p3r_config),
                stockfish_config=deepcopy(self.stockfish_config),
                data_collector=self._create_data_collector(game_id)
            )
            
            # Run the game and get the result
            result = game.run()
            
            # Update statistics
            self.games_played += 1
            if result and result.get('result') == '1-0':
                self.games_won += 1
            elif result and result.get('result') == '0-1':
                self.games_lost += 1
            else:
                self.games_drawn += 1
            
            # Record history
            if result and 'result' in result:
                # Record history
                self.game_history.append(result['result'])
                self.elo_history.append(self.current_elo)
                win_rate = self.games_won / self.games_played if self.games_played > 0 else 0
                self.win_rate_history.append(win_rate)
                
                # Update ELO for next game
                self._update_elo(result['result'])
            else:
                logger.error("Game result is None or missing 'result' key.")
            
            # Check for convergence
            converged = self._check_convergence()
            if converged:
                logger.info(f"ELO has converged after {self.games_played} games. Final ELO: {self.current_elo}")
        
        # Calculate final statistics
        win_rate = self.games_won / self.games_played if self.games_played > 0 else 0
        draw_rate = self.games_drawn / self.games_played if self.games_played > 0 else 0
        loss_rate = self.games_lost / self.games_played if self.games_played > 0 else 0
        
        # Create final results
        final_results = {
            'id': self.simulation_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'games_played': self.games_played,
            'initial_elo': simulation_metadata['initial_elo'],
            'final_elo': self.current_elo,
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'loss_rate': loss_rate,
            'win_count': self.games_won,
            'draw_count': self.games_drawn,
            'loss_count': self.games_lost,
            'converged': converged,
            'elo_history': self.elo_history,
            'win_rate_history': self.win_rate_history,
            'game_history': self.game_history,
        }
        
        # Calculate estimated V7P3R ELO
        # A simple estimate: if win rate is 50%, V7P3R ELO ≈ Stockfish ELO
        # Adjust based on win percentage - each 10% above/below 50% ~ +/-100 ELO points
        v7p3r_estimated_elo = self.current_elo + ((win_rate - 0.5) * 1000)
        final_results['v7p3r_estimated_elo'] = int(v7p3r_estimated_elo)
        
        # Save final results
        with open(f"games/{self.simulation_id}_results.yaml", 'w') as f:
            yaml.dump(final_results, f)
            
        # Send final results to central storage if enabled
        if self.use_central_storage and self.db_client:
            try:
                self.db_client.send_raw_simulation({
                    'id': f"{self.simulation_id}_final",
                    'type': 'elo_finder_results',
                    'data': final_results
                })
                
                # Flush any buffered data
                self.db_client.flush_offline_buffer()
            except Exception as e:
                logger.error(f"Failed to send final simulation results: {e}")
                if self.db_client:
                    self.db_client.save_offline_buffer()
        
        logger.info(f"ELO Finder simulation completed. V7P3R estimated ELO: {final_results['v7p3r_estimated_elo']}")
        return final_results

if __name__ == "__main__":
    # Example of running an ELO finder simulation directly
    simulator = AdaptiveEloSimulator(
        initial_elo=1500,
        min_elo=800,
        max_elo=3200,
        adjustment_factor=1.0,
        convergence_threshold=0.05,
        min_games_for_convergence=20,
        max_games=100
    )
    results = simulator.run_simulation()
    print(f"V7P3R estimated ELO: {results['v7p3r_estimated_elo']}")
