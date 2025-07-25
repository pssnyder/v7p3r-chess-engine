"""
Game Simulation Manager for v7p3r Chess Engine
Automates running multiple simulation scenarios as described in config/simulation_config.yaml.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from copy import deepcopy
from chess_game import ChessGame

# Helper to load a YAML config file
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def deep_update(base, overrides):
    """Recursively update dict base with overrides."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

class ChessGameConfig:
    def __init__(self, game_config, v7p3r_config, stockfish_config):
        self.fen_position = None
        self.data_collector = None
        self.game_config = game_config
        self.v7p3r_config = v7p3r_config
        self.stockfish_handler = stockfish_config


def run_simulation_scenarios():
    # Load simulation config
    sim_cfg = load_yaml('config/simulation_config.yaml')
    scenarios = sim_cfg.get('simulations', [])

    # Load base configs
    base_game_cfg = load_yaml('config/chess_game_config.yaml')
    base_v7p3r_cfg = load_yaml('config/v7p3r_config.yaml')
    base_stockfish_cfg = load_yaml('config/stockfish_config.yaml')

    for idx, scenario in enumerate(scenarios):
        print(f"\n=== Running Simulation {idx}: {scenario.get('name', 'Unnamed')} ===")
        games_to_run = scenario.get('games_to_run', 1)

        # Prepare configs for this scenario
        # Start with deep copies of base configs
        game_cfg = deepcopy(base_game_cfg)
        v7p3r_cfg = deepcopy(base_v7p3r_cfg)
        stockfish_cfg = deepcopy(base_stockfish_cfg)

        # Apply scenario chess_game overrides
        if 'chess_game' in scenario:
            deep_update(game_cfg, scenario['chess_game'])
        # Apply scenario v7p3r/stockfish overrides
        if 'v7p3r' in scenario:
            for engine_key, overrides in scenario['v7p3r'].items():
                if engine_key in v7p3r_cfg:
                    deep_update(v7p3r_cfg[engine_key], overrides)
                else:
                    v7p3r_cfg[engine_key] = deepcopy(overrides)
        if 'stockfish_handler' in scenario:
            deep_update(stockfish_cfg.get('stockfish_handler', {}), scenario['stockfish_handler'])

        # Set the number of games for this scenario
        if 'ai_vs_ai' in game_cfg.get('game_config', {}):
            game_cfg['game_config']['ai_game_count'] = games_to_run
        else:
            game_cfg.setdefault('game_config', {})['ai_game_count'] = games_to_run
        game_cfg['game_config']['ai_vs_ai'] = True

        # Build config object and run games
        config = ChessGameConfig(game_cfg, v7p3r_cfg, stockfish_cfg)
        game = ChessGame(config)
        game.run()
        game.metrics_store.close()
        print(f"Simulation {idx} complete.\n")

if __name__ == "__main__":
    run_simulation_scenarios()
