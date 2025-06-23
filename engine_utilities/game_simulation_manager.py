# /engine_utilities/game_simulation_manager.py

import sys
import os

# Adjust the module search path if the script is executed directly
if __name__ == "__main__" and __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

import asyncio
import yaml
import logging
import os
import uuid
import datetime
import json
import multiprocessing
from functools import partial
from chess_game import ChessGame
from copy import deepcopy
from engine_utilities.engine_db_manager import EngineDBClient
from engine_utilities.adaptive_elo_finder import AdaptiveEloSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_game_instance(game_config_override, v7p3r_config_override, stockfish_handler_override, sim_name, simulation_id, use_central_storage=False):
    """Wrapper function to run a single game instance in a separate process."""
    try:
        logger.info(f"Starting game simulation process for: {sim_name}")
        
        # Create a unique game ID
        game_id = f"{simulation_id}_{uuid.uuid4()}"
          # Load base configurations
        with open("config/chess_game_config.yaml") as f:
            base_game_config = yaml.safe_load(f)
        with open("config/v7p3r_config.yaml") as f:
            base_v7p3r_config = yaml.safe_load(f)
        with open("config/stockfish_config.yaml") as f:
            base_stockfish_handler = yaml.safe_load(f)        # Deep merge the overrides
        def deep_merge(source, destination):
            for key, value in source.items():
                if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                    deep_merge(value, destination[key])
                else:
                    destination[key] = value
            return destination
            
        final_game_config = deep_merge(game_config_override, base_game_config)
        final_v7p3r_config = deep_merge(v7p3r_config_override, base_v7p3r_config)
        final_stockfish_handler = deep_merge(stockfish_handler_override, base_stockfish_handler)
        
        # Ensure AI vs AI is set to true for simulations
        if 'game_config' in final_game_config:
            final_game_config['game_config']['ai_vs_ai'] = True
            logger.info("Setting AI vs AI mode to True for simulation")
        else:
            # If game_config isn't in the structure, add it
            final_game_config['game_config'] = {'ai_vs_ai': True}
            logger.info("Created game_config section with AI vs AI mode")
        # Remove human_color if present (not needed for AI vs AI)
        if 'human_color' in final_game_config['game_config']:
            del final_game_config['game_config']['human_color']
            logger.info("Removed human_color from game_config for AI vs AI simulation")
        
        # Include game_id in configuration
        if 'game_id' not in final_game_config:
            final_game_config['game_id'] = game_id

        # Initialize the DB client if central storage is enabled
        db_client = None
        if use_central_storage:
            try:
                db_client = EngineDBClient()
                logger.info(f"DB client initialized for game {game_id}")
            except Exception as e:
                logger.error(f"Failed to initialize DB client: {e}")
        
        # Create a data collector function to pass to the chess game
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
                with open(f"games/{game_id}.yaml", 'w') as f:
                    yaml.dump(metadata, f)
                    
            elif data_type == 'move_metric':
                # Append to move metrics file
                os.makedirs('games', exist_ok=True)
                with open(f"games/{game_id}_moves.json", 'a') as f:
                    f.write(json.dumps(data) + '\n')
            
            # Send to central storage if enabled
            if db_client:
                try:
                    if data_type == 'game_result':
                        db_client.send_game_data(data)
                    elif data_type == 'move_metric':
                        db_client.send_move_data(data)
                except Exception as e:
                    logger.error(f"Failed to send {data_type} data: {e}")
        
        # Merge all configurations into a single dictionary
        combined_config = {
            'game_config': final_game_config,
            'v7p3r_config': final_v7p3r_config,
            'stockfish_handler': final_stockfish_handler,
        }
        if use_central_storage:
            combined_config['data_collector'] = data_collector

        # Pass the combined configuration to ChessGame
        game = ChessGame(config=combined_config)
        result = game.run()
        
        # Upload raw game data if central storage is enabled
        if use_central_storage and db_client:
            try:
                # Create a complete game record with all configs and results
                raw_game_data = {
                    'id': game_id,
                    'simulation_id': simulation_id,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'name': sim_name,
                    'result': result,
                    'configs': {
                        'game': final_game_config,
                        'v7p3r': final_v7p3r_config,
                        'stockfish': final_stockfish_handler
                    }
                }
                
                # Add PGN if available
                pgn_path = f"games/{game_id}.pgn"
                if os.path.exists(pgn_path):
                    with open(pgn_path, 'r') as f:
                        raw_game_data['pgn'] = f.read()
                
                # Send raw data
                db_client.send_raw_simulation(raw_game_data)
                
                # Flush any buffered data
                db_client.flush_offline_buffer()
            except Exception as e:
                logger.error(f"Failed to upload raw game data: {e}")
                # Save offline buffer for later recovery
                db_client.save_offline_buffer()
        
        logger.info(f"Game simulation finished for: {sim_name}")
        return result
    except Exception as e:
        logger.error(f"Error running game simulation for {sim_name}: {e}", exc_info=True)
        return {"error": str(e)}

class GameSimulationManager:
    def __init__(self, simulation_config_path="config/simulation_config.yaml"):
        self.simulation_config_path = simulation_config_path
        self.db_client = None
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
        self.use_central_storage = bool(self.config.get('use_central_storage', False))
        if self.use_central_storage:
            logger.info("Central data storage is enabled - simulation results will be sent to configured server")
            try:
                self.db_client = EngineDBClient()
                logger.info("EngineDBClient initialized for simulation manager")
            except Exception as e:
                logger.error(f"Failed to initialize EngineDBClient: {e}")
        else:
            logger.info("Central data storage is disabled - simulation results will be stored locally only")
        self.templates = self.config.get('simulation_templates', [])
    
    def get_template_by_id(self, template_id):
        """Get a simulation template by its ID."""
        for template in self.templates:
            if template.get('id') == template_id:
                return template
        return None
        
    def run_simulations(self):
        """
        Launches and manages game simulations based on the configuration.
        """
        logger.info("Starting game simulation manager.")
        
        # Generate a unique simulation ID
        simulation_id = f"sim_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Simulation ID: {simulation_id}")
        
        # Save simulation metadata
        simulation_metadata = {
            'id': simulation_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'config': self.config,
        }
        os.makedirs('games', exist_ok=True)
        with open(f"games/{simulation_id}_metadata.yaml", 'w') as f:
            yaml.dump(simulation_metadata, f)
        
        # Process adaptive ELO simulations first (these run sequentially)
        adaptive_results = []
        for sim_details in self.config.get('simulations', []):
            # Check if this is a template-based simulation
            if 'template' in sim_details:
                template_id = sim_details['template']
                template = self.get_template_by_id(template_id)
                
                if template and template.get('type') == 'adaptive_elo':
                    logger.info(f"Running adaptive ELO simulation from template: {template_id}")
                    
                    # Merge template config with simulation overrides
                    template_config = template.get('config', {})
                    sim_config = {**template_config, **sim_details.get('config', {})}
                    
                    # Set up the simulator
                    simulator = AdaptiveEloSimulator(
                        initial_elo=sim_config.get('initial_elo', 1500),
                        min_elo=sim_config.get('min_elo', 800),
                        max_elo=sim_config.get('max_elo', 3200),
                        adjustment_factor=sim_config.get('adjustment_factor', 1.0),
                        convergence_threshold=sim_config.get('convergence_threshold', 0.05),
                        min_games_for_convergence=sim_config.get('min_games_for_convergence', 20),
                        max_games=sim_config.get('max_games', 100),
                        v7p3r_config=sim_details.get('v7p3r', {}),
                        game_config=sim_details.get('chess_game', {}),
                        use_central_storage=self.use_central_storage
                    )
                    
                    # Run the simulation
                    result = simulator.run_simulation()
                    adaptive_results.append(result)
                    continue
              # Use spawn context for better cross-platform compatibility
        ctx = multiprocessing.get_context('spawn')
        pool = ctx.Pool(processes=self.max_concurrent_simulations)
        
        # Set the global pool for termination handling
        if __name__ == "__main__":
            global global_pool
            global_pool = pool
        
        tasks = []
        for sim_details in self.config.get('simulations', []):
            # Skip template-based adaptive simulations (already processed)
            if 'template' in sim_details:
                template_id = sim_details['template']
                template = self.get_template_by_id(template_id)
                if template and template.get('type') == 'adaptive_elo':
                    continue
                
            name = sim_details.get('name', 'Unnamed Simulation')
            num_games = sim_details.get('games_to_run', 1)
            
            logger.info(f"Queueing {num_games} games for simulation: {name}")

            for i in range(num_games):
                game_config_override = sim_details.get('chess_game', {})
                v7p3r_config_override = sim_details.get('v7p3r', {})
                stockfish_handler_override = sim_details.get('stockfish_handler', {})
                
                # The configs need to be deepcopied for each process
                task_args = (
                    deepcopy(game_config_override),
                    deepcopy(v7p3r_config_override),
                    deepcopy(stockfish_handler_override),
                    f"{name} - Game {i+1}/{num_games}",
                    simulation_id,
                    self.use_central_storage
                )
                tasks.append(task_args)

        # Create a client to upload the overall simulation metadata if central storage is enabled
        if self.use_central_storage and self.db_client:
            try:
                self.db_client.send_raw_simulation(simulation_metadata)
                logger.info("Simulation metadata sent to central storage")
            except Exception as e:
                logger.error(f"Failed to send simulation metadata: {e}")

        # Run the simulations
        results = []
        try:
            if tasks:
                results = pool.starmap(run_game_instance, tasks)
        except KeyboardInterrupt:
            logger.warning("Simulation interrupted by user. Terminating pool...")
            pool.terminate()
            results = []
        finally:
            pool.close()
            pool.join()
            # Flush and close DB client if used
            if self.use_central_storage and self.db_client:
                try:
                    self.db_client.flush_offline_buffer()
                    logger.info("Flushed DB client offline buffer on shutdown.")
                except Exception as e:
                    logger.error(f"Failed to flush DB client: {e}")

        # Save overall results
        simulation_results = {
            'id': simulation_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'results': results,
            'completed': len([r for r in results if r and 'error' not in r]),
            'failed': len([r for r in results if not r or 'error' in r]),
            'total': len(tasks)
        }
        with open(f"games/{simulation_id}_results.yaml", 'w') as f:
            yaml.dump(simulation_results, f)
        # Send final results to central storage if enabled
        if self.use_central_storage and self.db_client:
            try:
                self.db_client.send_raw_simulation({
                    'id': f"{simulation_id}_final",
                    'type': 'simulation_results',
                    'data': simulation_results
                })
                logger.info("Final simulation results sent to central storage")
            except Exception as e:
                logger.error(f"Failed to send final simulation results: {e}")
        logger.info(f"All game simulations completed. {simulation_results['completed']} succeeded, {simulation_results['failed']} failed.")
        return simulation_results

if __name__ == "__main__":
    # This allows the script to be run as the main entry point for simulations
    import signal
    import sys
    
    # Global pool variable for termination
    global_pool = None
    
    # Handle Ctrl+C more gracefully
    def signal_handler(sig, frame):
        print("\nReceived Ctrl+C. Terminating simulations gracefully...")
        # Force terminate any running pool
        if global_pool is not None:
            global_pool.terminate()
            global_pool.join()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check if config exists in the root directory or try to copy it
    if not os.path.exists("simulation_config.yaml") and os.path.exists("config/simulation_config.yaml"):
        try:
            import shutil
            shutil.copy2("config/simulation_config.yaml", "simulation_config.yaml")
            logger.info("Copied simulation_config.yaml from config directory to root")
        except Exception as e:
            logger.warning(f"Could not copy simulation config: {e}")
    
    # Create a simple simulation config if none exists
    if not os.path.exists("simulation_config.yaml"):
        logger.warning("No simulation config found. Creating a minimal config for testing.")
        with open("simulation_config.yaml", "w") as f:
            f.write("""# simulation_config.yaml
# Main configuration for the Game Simulation Manager

# Maximum number of simulations to run in parallel.
max_concurrent_simulations: 2

# Enable central data storage for simulation results
use_central_storage: false

simulations:
  # Simple test simulation
  - name: "Quick Test - V7P3R vs. Stockfish (Skill 1)"
    games_to_run: 2
    chess_game:
      white_engine_config:
        engine: 'v7p3r'
        engine_type: 'deepsearch'
      black_engine_config:
        engine: 'stockfish'
      game_config:
        ai_vs_ai: true
        human_color: 'white'
    v7p3r:
      v7p3r:
        ruleset: 'default_evaluation'
        depth: 3
    stockfish_handler:
      stockfish:
        skill_level: 1
""")
            logger.info("Created minimal simulation configuration for testing")
    
    # Run the simulations
    manager = GameSimulationManager()
    manager.run_simulations()
