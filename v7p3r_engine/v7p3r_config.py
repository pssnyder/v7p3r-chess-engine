import json
import os
from typing import Optional

class v7p3rConfig:
    """
    Configuration class for the v7p3r chess engine.
    This class generates the configuration settings for the engine, including game settings, engine settings, and Stockfish settings.
    """
    def __init__(self, config_path: Optional[str] = None):
        if config_path is not None and os.path.isfile(config_path):
            self.config_path = config_path
        else:
            self.config_path = os.path.join(os.path.dirname(__file__), 'saved_configs', 'default_config.json')
        self.config = {}
        self.game_config = {}
        self.engine_config = {}
        self.stockfish_config = {}
        self.puzzle_config = {}
        self.rulesets = {}
        self.ruleset_name = "default_ruleset"
        self.ruleset = {}

        # Load the default configuration
        self._load_config()

    def _load_config(self):
        """
        Load the default configuration from a JSON file.
        The default configuration is stored in 'saved_configs/default_config.json'.
        """
        try:
            # Open the default configuration file and load its contents
            with open(self.config_path, 'r') as config_path:
                self.config = json.load(config_path)
                if isinstance(self.config, dict):
                    # Load module-specific configurations
                    self._load_module_configs()
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from configuration file: {e}")

    def _load_module_configs(self):
        """
        Load module-specific configurations from the main configuration.
        :param config: The main configuration dictionary.
        """
        if hasattr(self.config, 'game_config'):
            self.game_config = self.config.get('game_config', {})
        if hasattr(self.config, 'engine_config'):
            self.engine_config = self.config.get('engine_config', {})
        if hasattr(self.config, 'stockfish_config'):
            self.stockfish_config = self.config.get('stockfish_config', {})
        if hasattr(self.config, 'puzzle_config'):
            self.puzzle_config = self.config.get('puzzle_config', {})
        if hasattr(self.config, 'logging_config'):
            self.logging_config = self.config.get('logging_config', {})
        if hasattr(self.config, 'metrics_config'):
            self.metrics_config = self.config.get('metrics_config', {})
        if hasattr(self.engine_config, 'ruleset'):
            self.ruleset_name = self.engine_config.get('ruleset', "default_ruleset")
            self.ruleset = self._load_ruleset()

    def _load_ruleset(self):
        """
        Load the default ruleset configuration.
        The default ruleset is stored in 'saved_configs/rulesets/default_ruleset.json'.
        """
        ruleset_path = os.path.join(os.path.dirname(__file__), 'saved_configs', 'rulesets', f"custom_rulesets.json")
        try:
            with open(ruleset_path, 'r') as file:
                self.rulesets = json.load(file)
                self.ruleset = self.rulesets.get(self.ruleset_name, {})
        except FileNotFoundError:
            raise FileNotFoundError(f"Ruleset file not found: {ruleset_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from ruleset file: {e}")

    
    def get_config(self):
        return self.config
    
    def get_game_config(self):
        return self.game_config

    def get_engine_config(self):
        return self.engine_config

    def get_stockfish_config(self):
        return self.stockfish_config
    
    def get_puzzle_config(self):
        return self.puzzle_config
    
    def get_logging_config(self):
        return self.logging_config
    
    def get_metrics_config(self):
        return self.metrics_config
    
    def get_ruleset(self):
        return self.ruleset