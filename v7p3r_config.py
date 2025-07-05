import json
import os
from typing import Optional, Dict, Any
import copy

class v7p3rConfig:
    """
    Configuration class for the v7p3r chess engine.
    This class generates the configuration settings for the engine, including game settings, engine settings, and Stockfish settings.
    Supports centralized configuration management with override capabilities for all modules.
    """
    def __init__(self, config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None):
        if config_path is not None and os.path.isfile(config_path):
            self.config_path = config_path
        else:
            self.config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default_config.json')
        
        # Initialize all configuration sections
        self.config = {}
        self.game_config = {}
        self.engine_config = {}
        self.stockfish_config = {}
        self.puzzle_config = {}
        self.logging_config = {}
        self.metrics_config = {}
        self.v7p3r_nn_config = {}
        self.v7p3r_ga_config = {}
        self.v7p3r_rl_config = {}
        self.rulesets = {}
        self.ruleset_name = "default_ruleset"
        self.ruleset = {}

        # Store overrides for later application
        self.overrides = overrides or {}

        # Load the default configuration
        self._load_config()

    def _load_config(self):
        """
        Load the default configuration from a JSON file.
        The default configuration is stored in 'configs/default_config.json'.
        """
        try:
            # Open the default configuration file and load its contents
            with open(self.config_path, 'r') as config_file:
                self.config = json.load(config_file)
                if isinstance(self.config, dict):
                    # Load module-specific configurations
                    self._load_module_configs()
                    # Apply any overrides
                    self._apply_overrides()
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from configuration file: {e}")

    def _load_module_configs(self):
        """
        Load module-specific configurations from the main configuration.
        """
        # Load module-specific configurations directly from the config dictionary
        self.game_config = self.config.get('game_config', {})
        self.engine_config = self.config.get('engine_config', {})
        self.stockfish_config = self.config.get('stockfish_config', {})
        self.puzzle_config = self.config.get('puzzle_config', {})
        self.logging_config = self.config.get('logging_config', {})
        self.metrics_config = self.config.get('metrics_config', {})
        self.v7p3r_nn_config = self.config.get('v7p3r_nn_config', {})
        self.v7p3r_ga_config = self.config.get('v7p3r_ga_config', {})
        self.v7p3r_rl_config = self.config.get('v7p3r_rl_config', {})
        
        # Load ruleset if specified
        if isinstance(self.engine_config, dict) and 'ruleset' in self.engine_config:
            self.ruleset_name = self.engine_config.get('ruleset', "default_ruleset")
            self._load_ruleset()

    def _apply_overrides(self):
        """
        Apply configuration overrides if provided.
        Overrides are applied recursively to nested dictionaries.
        """
        if not self.overrides:
            return

        def deep_update(target: dict, override: dict):
            """Recursively update target dict with override dict"""
            for key, value in override.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_update(target[key], value)
                else:
                    target[key] = value

        # Apply overrides to main config
        deep_update(self.config, self.overrides)
        
        # Reload module configs to reflect overrides
        self._load_module_configs()

    def _load_ruleset(self):
        """
        Load the ruleset configuration.
        The rulesets are stored in 'configs/rulesets/custom_rulesets.json'.
        """
        ruleset_path = os.path.join(os.path.dirname(__file__), 'configs', 'rulesets', 'custom_rulesets.json')
        try:
            with open(ruleset_path, 'r') as file:
                self.rulesets = json.load(file)
                self.ruleset = self.rulesets.get(self.ruleset_name, {})
        except FileNotFoundError:
            # If no custom rulesets file exists, create default structure
            self.rulesets = {"default_ruleset": {}}
            self.ruleset = {}
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from ruleset file: {e}")

    def get_config(self):
        """Get the complete configuration dictionary"""
        return copy.deepcopy(self.config)
    
    def get_game_config(self):
        """Get game-specific configuration"""
        return copy.deepcopy(self.game_config)

    def get_engine_config(self):
        """Get engine-specific configuration"""
        return copy.deepcopy(self.engine_config)

    def get_stockfish_config(self):
        """Get Stockfish-specific configuration"""
        return copy.deepcopy(self.stockfish_config)
    
    def get_puzzle_config(self):
        """Get puzzle-specific configuration"""
        return copy.deepcopy(self.puzzle_config)
    
    def get_logging_config(self):
        """Get logging-specific configuration"""
        return copy.deepcopy(self.logging_config)
    
    def get_metrics_config(self):
        """Get metrics-specific configuration"""
        return copy.deepcopy(self.metrics_config)
    
    def get_v7p3r_nn_config(self):
        """Get Neural Network engine configuration"""
        return copy.deepcopy(self.v7p3r_nn_config)
    
    def get_v7p3r_ga_config(self):
        """Get Genetic Algorithm engine configuration"""
        return copy.deepcopy(self.v7p3r_ga_config)
    
    def get_v7p3r_rl_config(self):
        """Get Reinforcement Learning engine configuration"""
        return copy.deepcopy(self.v7p3r_rl_config)
    
    def get_ruleset(self):
        """Get the current ruleset configuration"""
        return copy.deepcopy(self.ruleset)
    
    def set_override(self, key_path: str, value: Any):
        """
        Set a configuration override using dot notation.
        Example: set_override('engine_config.depth', 10)
        """
        keys = key_path.split('.')
        target = self.overrides
        
        # Navigate to the target location
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        # Set the value
        target[keys[-1]] = value
        
        # Reapply all overrides
        self._apply_overrides()

    def get_config_for_module(self, module_name: str):
        """
        Get configuration for a specific module.
        Supported modules: 'nn', 'ga', 'rl', 'engine', 'stockfish', 'puzzle', 'game', 'logging', 'metrics'
        """
        module_getters = {
            'nn': self.get_v7p3r_nn_config,
            'ga': self.get_v7p3r_ga_config,
            'rl': self.get_v7p3r_rl_config,
            'engine': self.get_engine_config,
            'stockfish': self.get_stockfish_config,
            'puzzle': self.get_puzzle_config,
            'game': self.get_game_config,
            'logging': self.get_logging_config,
            'metrics': self.get_metrics_config
        }
        
        if module_name not in module_getters:
            raise ValueError(f"Unknown module: {module_name}. Supported modules: {list(module_getters.keys())}")
        
        return module_getters[module_name]()
    
    def create_runtime_config(self, base_config: Optional[Dict[str, Any]] = None, **overrides) -> 'v7p3rConfig':
        """
        Create a new configuration instance with runtime overrides.
        
        Args:
            base_config: Optional base configuration dictionary
            **overrides: Configuration overrides using dot notation keys
        
        Returns:
            New v7p3rConfig instance with overrides applied
        
        Example:
            config = manager.create_runtime_config(
                engine_config__depth=10,
                stockfish_config__elo_rating=1500
            )
        """
        # Convert double underscore notation to nested dict
        override_dict = {}
        for key, value in overrides.items():
            keys = key.split('__')
            target = override_dict
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = value
        
        # Create new config instance
        if base_config:
            # Save current config temporarily
            temp_config = self.config.copy()
            self.config = base_config
            new_instance = v7p3rConfig(self.config_path, override_dict)
            self.config = temp_config
        else:
            new_instance = v7p3rConfig(self.config_path, override_dict)
        
        return new_instance
    
    def apply_engine_specific_overrides(self, engine_name: str, engine_config: Dict[str, Any]) -> 'v7p3rConfig':
        """
        Apply engine-specific configuration overrides.
        
        Args:
            engine_name: Name of the engine ('v7p3r', 'v7p3r_nn', 'v7p3r_ga', 'v7p3r_rl')
            engine_config: Engine-specific configuration overrides
        
        Returns:
            New v7p3rConfig instance with engine overrides applied
        """
        override_dict = {}
        
        if engine_name == 'v7p3r':
            override_dict['engine_config'] = engine_config
        elif engine_name == 'v7p3r_nn':
            override_dict['v7p3r_nn_config'] = engine_config
        elif engine_name == 'v7p3r_ga':
            override_dict['v7p3r_ga_config'] = engine_config
        elif engine_name == 'v7p3r_rl':
            override_dict['v7p3r_rl_config'] = engine_config
        else:
            raise ValueError(f"Unknown engine: {engine_name}")
        
        return v7p3rConfig(self.config_path, override_dict)
    
    def get_override_summary(self) -> Dict[str, Any]:
        """Get a summary of all currently applied overrides."""
        return copy.deepcopy(self.overrides)
    
    def clear_overrides(self):
        """Clear all configuration overrides and reload from file."""
        self.overrides = {}
        self._load_config()