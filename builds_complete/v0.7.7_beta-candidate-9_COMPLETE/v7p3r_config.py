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
        Load configuration with layered override system.
        
        1. First loads default_config.json as the base layer (REQUIRED)
        2. Then overlays custom config values (if custom config specified)
        3. Custom values override defaults, everything else inherits
        4. Fails fast if default_config.json cannot be loaded
        """
        # STEP 1: Load default configuration as base layer (REQUIRED)
        default_config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default_config.json')
        
        try:
            with open(default_config_path, 'r') as config_file:
                self.config = json.load(config_file)
                
                if not isinstance(self.config, dict):
                    raise ValueError("default_config.json does not contain a valid configuration dictionary")
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"CRITICAL ERROR: default_config.json not found at {default_config_path}. Engine cannot start without default configuration.")
        except json.JSONDecodeError as e:
            raise ValueError(f"CRITICAL ERROR: Invalid JSON in default_config.json: {e}. Engine cannot start.")
        
        # STEP 2: If custom config specified, overlay custom values
        if self.config_path != default_config_path:
            try:
                with open(self.config_path, 'r') as custom_file:
                    custom_config = json.load(custom_file)
                    
                    if isinstance(custom_config, dict):
                        # Recursively overlay custom values onto default base
                        self._deep_update(self.config, custom_config)
                    else:
                        print(f"WARNING: Custom config file {self.config_path} does not contain valid dictionary. Using defaults only.")
                        
            except FileNotFoundError:
                print(f"WARNING: Custom config file {self.config_path} not found. Using default_config.json only.")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in custom config file {self.config_path}: {e}")
        
        # Load module-specific configurations
        self._load_module_configs()
        
        # Apply any runtime overrides
        self._apply_overrides()
        
        # Validate that we have required configuration values
        if not self.config:
            raise ValueError("CRITICAL ERROR: No valid configuration loaded. Engine cannot start.")
    
    def _deep_update(self, target: dict, source: dict):
        """
        Recursively update target dict with source dict.
        Source values override target values, but preserve nested structure.
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

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

        # Apply overrides to main config using the new deep_update method
        self._deep_update(self.config, self.overrides)
        
        # Reload module configs to reflect overrides
        self._load_module_configs()

    def _load_ruleset(self):
        """
        Load the ruleset configuration with layered override system.
        
        1. First loads default_ruleset.json as the base layer
        2. Then overlays custom ruleset values (if specified and exists)
        3. Custom values override defaults, everything else inherits
        4. Fails fast if default_ruleset.json cannot be loaded
        """
        # STEP 1: Load default ruleset as base layer (REQUIRED)
        default_ruleset_path = os.path.join(os.path.dirname(__file__), 'configs', 'rulesets', 'default_ruleset.json')
        
        try:
            with open(default_ruleset_path, 'r') as file:
                default_data = json.load(file)
                self.ruleset = default_data.get('default_ruleset', {})
                
                if not self.ruleset:
                    raise ValueError("default_ruleset.json exists but does not contain 'default_ruleset' key")
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"CRITICAL ERROR: default_ruleset.json not found at {default_ruleset_path}. Engine cannot start without default configuration.")
        except json.JSONDecodeError as e:
            raise ValueError(f"CRITICAL ERROR: Invalid JSON in default_ruleset.json: {e}. Engine cannot start.")
        
        # STEP 2: If custom ruleset specified, overlay custom values
        if self.ruleset_name != 'default_ruleset':
            # Try individual custom ruleset file first
            custom_individual_path = os.path.join(os.path.dirname(__file__), 'configs', 'rulesets', f'{self.ruleset_name}.json')
            custom_found = False
            
            try:
                with open(custom_individual_path, 'r') as file:
                    custom_data = json.load(file)
                    custom_ruleset = custom_data.get(self.ruleset_name, {})
                    if custom_ruleset:
                        # Overlay custom values onto default base
                        self.ruleset.update(custom_ruleset)
                        custom_found = True
            except FileNotFoundError:
                pass  # Try custom_rulesets.json next
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in custom ruleset file {custom_individual_path}: {e}")
            
            # If not found in individual file, try custom_rulesets.json
            if not custom_found:
                custom_rulesets_path = os.path.join(os.path.dirname(__file__), 'configs', 'rulesets', 'custom_rulesets.json')
                try:
                    with open(custom_rulesets_path, 'r') as file:
                        custom_data = json.load(file)
                        custom_ruleset = custom_data.get(self.ruleset_name, {})
                        if custom_ruleset:
                            # Overlay custom values onto default base
                            self.ruleset.update(custom_ruleset)
                            custom_found = True
                except FileNotFoundError:
                    pass  # Custom ruleset not found, use defaults
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in custom_rulesets.json: {e}")
            
            # Log if custom ruleset was requested but not found
            if not custom_found:
                print(f"WARNING: Custom ruleset '{self.ruleset_name}' not found. Using default_ruleset values only.")
        
        # Validate that we have required ruleset values
        if not self.ruleset:
            raise ValueError("CRITICAL ERROR: No valid ruleset configuration loaded. Engine cannot start.")

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
