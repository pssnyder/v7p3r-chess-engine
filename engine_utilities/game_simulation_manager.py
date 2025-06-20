# /engine_utilities/game_simulation_manager.py

import asyncio
import yaml
import logging
import os
import multiprocessing
from chess_game import ChessGame
from copy import deepcopy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_game_instance(game_config_override, v7p3r_config_override, stockfish_config_override, sim_name):
    """Wrapper function to run a single game instance in a separate process."""
    try:
        logger.info(f"Starting game simulation process for: {sim_name}")
        # Load base configurations
        with open("chess_game.yaml") as f:
            base_game_config = yaml.safe_load(f)
        with open("v7p3r.yaml") as f:
            base_v7p3r_config = yaml.safe_load(f)
        with open("engine_utilities/stockfish_handler.yaml") as f:
            base_stockfish_config = yaml.safe_load(f)

        # Deep merge the overrides
        def deep_merge(source, destination):
            for key, value in source.items():
                if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                    deep_merge(value, destination[key])
                else:
                    destination[key] = value
            return destination

        final_game_config = deep_merge(game_config_override, base_game_config)
        final_v7p3r_config = deep_merge(v7p3r_config_override, base_v7p3r_config)
        final_stockfish_config = deep_merge(stockfish_config_override, base_stockfish_config)

        game = ChessGame(
            game_config=final_game_config,
            v7p3r_config=final_v7p3r_config,
            stockfish_config=final_stockfish_config
        )
        game.run()
        logger.info(f"Game simulation finished for: {sim_name}")
    except Exception as e:
        logger.error(f"Error running game simulation for {sim_name}: {e}", exc_info=True)

class GameSimulationManager:
    def __init__(self, simulation_config_path="simulation_config.yaml"):
        self.simulation_config_path = simulation_config_path
        try:
            with open(self.simulation_config_path) as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Simulation config file not found at {self.simulation_config_path}")
            self.config = {'simulations': []}
        
        max_sim = self.config.get('max_concurrent_simulations', os.cpu_count() or 1)
        if isinstance(max_sim, list):
            self.max_concurrent_simulations = os.cpu_count() or 1
        else:
            self.max_concurrent_simulations = int(max_sim)

    def run_simulations(self):
        """
        Launches and manages game simulations based on the configuration.
        """
        logger.info("Starting game simulation manager.")
        
        # Use spawn context for better cross-platform compatibility
        ctx = multiprocessing.get_context('spawn')
        pool = ctx.Pool(processes=self.max_concurrent_simulations)
        
        tasks = []
        for sim_details in self.config.get('simulations', []):
            name = sim_details.get('name', 'Unnamed Simulation')
            num_games = sim_details.get('games_to_run', 1)
            
            logger.info(f"Queueing {num_games} games for simulation: {name}")

            for i in range(num_games):
                game_config_override = sim_details.get('chess_game', {})
                v7p3r_config_override = sim_details.get('v7p3r', {})
                stockfish_config_override = sim_details.get('stockfish_handler', {})
                
                # The configs need to be deepcopied for each process
                task_args = (
                    deepcopy(game_config_override),
                    deepcopy(v7p3r_config_override),
                    deepcopy(stockfish_config_override),
                    f"{name} - Game {i+1}/{num_games}"
                )
                tasks.append(task_args)

        if tasks:
            pool.starmap(run_game_instance, tasks)

        pool.close()
        pool.join()
        
        logger.info("All game simulations completed.")

if __name__ == "__main__":
    # This allows the script to be run as the main entry point for simulations
    manager = GameSimulationManager()
    manager.run_simulations()
