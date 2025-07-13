"""V7P3R Chess Engine Configuration Module

This module handles all configuration management for the V7P3R chess engine.
It provides a centralized configuration system with override capabilities,
layered configuration loading, and type-safe access to all settings.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, cast, Union
import copy
import os

from v7p3r_paths import paths
from v7p3r_config_types import (
    V7P3RConfig, GameConfig, EngineConfig, StockfishConfig,
    PuzzleConfig, MetricsConfig, NeuralNetworkConfig as NNConfig,
    GeneticAlgorithmConfig as GAConfig,
    ReinforcementLearningConfig as RLConfig
)

class v7p3rConfig:
    """
    Configuration class for the v7p3r chess engine.
    This class generates the configuration settings for the engine, including game settings, 
    engine settings, and Stockfish settings. Supports centralized configuration management 
    with override capabilities for all modules.
    """
    def __init__(self, config_path: Optional[Union[str, Path]] = None, overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to config file (Path or string, without .json)
            overrides: Optional dictionary of configuration overrides
        """
        # Determine config path
        config_name = 'default_config'
        if config_path is not None:
            if isinstance(config_path, Path):
                config_name = config_path.stem
            else:
                config_name = str(config_path)
        self.config_path = paths.get_config_file(config_name)

        # Initialize class attributes with type hints
        self.config_path: Path
        self.config: Dict[str, Any] = {}
        self.game_config: GameConfig = cast(GameConfig, {})
        self.engine_config: EngineConfig = cast(EngineConfig, {})
        self.stockfish_config: StockfishConfig = cast(StockfishConfig, {})
        self.puzzle_config: PuzzleConfig = cast(PuzzleConfig, {})
        self.metrics_config: MetricsConfig = cast(MetricsConfig, {})

        # Optional configs
        self.nn_config: Optional[NNConfig] = None
        self.ga_config: Optional[GAConfig] = None
        self.rl_config: Optional[RLConfig] = None
        self.v7p3r_nn_config: Dict[str, Any] = {}
        self.v7p3r_ga_config: Dict[str, Any] = {}
        self.v7p3r_rl_config: Dict[str, Any] = {}
        self.rulesets: Dict[str, Any] = {}
        self.ruleset_name: str = "default_ruleset"
        self.ruleset: Dict[str, Any] = {}

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
            
            # Warn the user if custom ruleset was requested but not found
            if not custom_found:
                print(f"WARNING: Custom ruleset '{self.ruleset_name}' not found. Using default_ruleset values only.")
        
        # Validate that we have required ruleset values
        if not self.ruleset:
            raise ValueError("CRITICAL ERROR: No valid ruleset configuration loaded. Engine cannot start.")

    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary"""
        return copy.deepcopy(self.config)
    
    def get_game_config(self) -> GameConfig:
        """Get game-specific configuration"""
        return copy.deepcopy(self.game_config)

    def get_engine_config(self) -> EngineConfig:
        """Get engine-specific configuration"""
        return copy.deepcopy(self.engine_config)

    def get_stockfish_config(self) -> StockfishConfig:
        """Get Stockfish-specific configuration"""
        return copy.deepcopy(self.stockfish_config)
    
    def get_puzzle_config(self) -> PuzzleConfig:
        """Get puzzle-specific configuration"""
        return copy.deepcopy(self.puzzle_config)
    
    def get_metrics_config(self) -> MetricsConfig:
        """Get metrics-specific configuration"""
        return copy.deepcopy(self.metrics_config)
    
    def get_v7p3r_nn_config(self) -> Dict[str, Any]:
        """Get Neural Network engine configuration"""
        return copy.deepcopy(self.v7p3r_nn_config)
    
    def get_v7p3r_ga_config(self) -> Dict[str, Any]:
        """Get Genetic Algorithm engine configuration"""
        return copy.deepcopy(self.v7p3r_ga_config)
    
    def get_v7p3r_rl_config(self) -> Dict[str, Any]:
        """Get Reinforcement Learning engine configuration"""
        return copy.deepcopy(self.v7p3r_rl_config)
    
    def get_ruleset(self) -> Dict[str, Any]:
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
        Supported modules: 'nn', 'ga', 'rl', 'engine', 'stockfish', 'puzzle', 'game', 'metrics'
        """
        module_getters = {
            'nn': self.get_v7p3r_nn_config,
            'ga': self.get_v7p3r_ga_config,
            'rl': self.get_v7p3r_rl_config,
            'engine': self.get_engine_config,
            'stockfish': self.get_stockfish_config,
            'puzzle': self.get_puzzle_config,
            'game': self.get_game_config,
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
            new_instance = v7p3rConfig(str(self.config_path), override_dict)
            self.config = temp_config
        else:
            new_instance = v7p3rConfig(str(self.config_path), override_dict)
        
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
        
        return v7p3rConfig(str(self.config_path), override_dict)
    
    def get_override_summary(self) -> Dict[str, Any]:
        """Get a summary of all currently applied overrides."""
        return copy.deepcopy(self.overrides)
    
    def clear_overrides(self) -> None:
        """Clear all configuration overrides and reload from file."""
        self.overrides = {}
        self._load_config()

    def _get_default_engine_config(self) -> EngineConfig:
        """Get default engine configuration"""
        return {
            'depth': 20,
            'max_depth': 30,  # Maximum search depth
            'multi_threaded': True,
            'use_transpositions': True,
            'use_tablebases': True,
            'use_ponder': True,
            'use_null_move': True,
            'null_move_threshold': 3,
            'use_late_move_reduction': True,
            'late_move_reduction_factor': 3,
            'use_quiescence': False,
            'use_game_phase': True,
            'use_mvv_lva': True,  # Enable/disable MVV-LVA calculations
            'mvv_lva_settings': {
                'use_safety_checks': True,  # Consider piece safety in MVV-LVA
                'use_position_context': True,  # Consider positional factors
                'safety_margin': 200,  # Base safety margin for captures
                'position_bonus': 50,  # Bonus for positionally sound captures
            },
            'use_dtm': False,
            'dtm_depth': 10,
            'dtm_window': 5,
            'use_aspiration': True,
            'aspiration_window': 10,
            'use_futility_pruning': True,
            'futility_margin': 100,
            'use_extended_pawn_structure': True,
            'pawn_structure_weight': 10,
            'use_king_safety': True,
            'king_safety_weight': 20,
            'use_piece_activity': True,
            'piece_activity_weight': 15,
            'use_space_control': True,
            'space_control_weight': 10,
            'use_mobility': True,
            'mobility_weight': 10,
            'use_material_balance': True,
            'material_balance_weight': 10,
            'use_attack_defense': True,
            'attack_defense_weight': 10,
            'use_blockade': True,
            'blockade_weight': 10,
            'use_open_files': True,
            'open_files_weight': 10,
            'use_semi_open_files': True,
            'semi_open_files_weight': 5,
            'use_passed_pawns': True,
            'passed_pawns_weight': 10,
            'use_isolated_pawns': True,
            'isolated_pawns_weight': 10,
            'use_backwards_pawns': True,
            'backwards_pawns_weight': 10,
            'use_doubled_pawns': True,
            'doubled_pawns_weight': 10,
            'use_weak_squares': True,
            'weak_squares_weight': 10,
            'use_strong_squares': True,
            'strong_squares_weight': 10,
            'use_color_complexity': True,
            'color_complexity_weight': 10,
            'use_piece_swarm': True,
            'piece_swarm_weight': 10,
            'use_knight_outposts': True,
            'knight_outposts_weight': 10,
            'use_bishop_pair': True,
            'bishop_pair_weight': 10,
            'use_rook_lift': True,
            'rook_lift_weight': 10,
            'use_queen_activity': True,
            'queen_activity_weight': 10,
            'use_king_activity': True,
            'king_activity_weight': 10,
            'use_castle_safety': True,
            'castle_safety_weight': 10,
            'use_risk_management': True,
            'risk_management_weight': 10,
            'use_material_count': True,
            'material_count_weight': 10,
            'use_piece_value': True,
            'piece_value_weight': 10,
            'use_king_distance': True,
            'king_distance_weight': 10,
            'use_queen_distance': True,
            'queen_distance_weight': 10,
            'use_rook_distance': True,
            'rook_distance_weight': 10,
            'use_bishop_distance': True,
            'bishop_distance_weight': 10,
            'use_knight_distance': True,
            'knight_distance_weight': 10,
            'use_pawn_distance': True,
            'pawn_distance_weight': 10,
            'use_king_pawn_distance': True,
            'king_pawn_distance_weight': 10,
            'use_queen_pawn_distance': True,
            'queen_pawn_distance_weight': 10,
            'use_rook_pawn_distance': True,
            'rook_pawn_distance_weight': 10,
            'use_bishop_pawn_distance': True,
            'bishop_pawn_distance_weight': 10,
            'use_knight_pawn_distance': True,
            'knight_pawn_distance_weight': 10,
            'use_passed_pawn_distance': True,
            'passed_pawn_distance_weight': 10,
            'use_isolated_pawn_distance': True,
            'isolated_pawn_distance_weight': 10,
            'use_backwards_pawn_distance': True,
            'backwards_pawn_distance_weight': 10,
            'use_doubled_pawn_distance': True,
            'doubled_pawn_distance_weight': 10,
            'use_weak_square_distance': True,
            'weak_square_distance_weight': 10,
            'use_strong_square_distance': True,
            'strong_square_distance_weight': 10,
            'use_color_complexity_distance': True,
            'color_complexity_distance_weight': 10,
            'use_piece_swarm_distance': True,
            'piece_swarm_distance_weight': 10,
            'use_knight_outpost_distance': True,
            'knight_outpost_distance_weight': 10,
            'use_bishop_pair_distance': True,
            'bishop_pair_distance_weight': 10,
            'use_rook_lift_distance': True,
            'rook_lift_distance_weight': 10,
            'use_queen_activity_distance': True,
            'queen_activity_distance_weight': 10,
            'use_king_activity_distance': True,
            'king_activity_distance_weight': 10,
            'use_castle_safety_distance': True,
            'castle_safety_distance_weight': 10,
            'use_risk_management_distance': True,
            'risk_management_distance_weight': 10,
            'use_material_count_distance': True,
            'material_count_distance_weight': 10,
            'use_piece_value_distance': True,
            'piece_value_distance_weight': 10,
            'use_king_distance_weight': True,
            'king_distance_weight_weight': 10,
            'use_queen_distance_weight': True,
            'queen_distance_weight_weight': 10,
            'use_rook_distance_weight': True,
            'rook_distance_weight_weight': 10,
            'use_bishop_distance_weight': True,
            'bishop_distance_weight_weight': 10,
            'use_knight_distance_weight': True,
            'knight_distance_weight_weight': 10,
            'use_pawn_distance_weight': True,
            'pawn_distance_weight_weight': 10,
            'use_king_pawn_distance_weight': True,
            'king_pawn_distance_weight_weight': 10,
            'use_queen_pawn_distance_weight': True,
            'queen_pawn_distance_weight_weight': 10,
            'use_rook_pawn_distance_weight': True,
            'rook_pawn_distance_weight_weight': 10,
            'use_bishop_pawn_distance_weight': True,
            'bishop_pawn_distance_weight_weight': 10,
            'use_knight_pawn_distance_weight': True,
            'knight_pawn_distance_weight_weight': 10,
            'use_passed_pawn_distance_weight': True,
            'passed_pawn_distance_weight_weight': 10,
            'use_isolated_pawn_distance_weight': True,
            'isolated_pawn_distance_weight_weight': 10,
            'use_backwards_pawn_distance_weight': True,
            'backwards_pawn_distance_weight_weight': 10,
            'use_doubled_pawn_distance_weight': True,
            'doubled_pawn_distance_weight_weight': 10,
            'use_weak_square_distance_weight': True,
            'weak_square_distance_weight_weight': 10,
            'use_strong_square_distance_weight': True,
            'strong_square_distance_weight_weight': 10,
            'use_color_complexity_distance_weight': True,
            'color_complexity_distance_weight_weight': 10,
            'use_piece_swarm_distance_weight': True,
            'piece_swarm_distance_weight_weight': 10,
            'use_knight_outpost_distance_weight': True,
            'knight_outpost_distance_weight_weight': 10,
            'use_bishop_pair_distance_weight': True,
            'bishop_pair_distance_weight_weight': 10,
            'use_rook_lift_distance_weight': True,
            'rook_lift_distance_weight_weight': 10,
            'use_queen_activity_distance_weight': True,
            'queen_activity_distance_weight_weight': 10,
            'use_king_activity_distance_weight': True,
            'king_activity_distance_weight_weight': 10,
            'use_castle_safety_distance_weight': True,
            'castle_safety_distance_weight_weight': 10,
            'use_risk_management_distance_weight': True,
            'risk_management_distance_weight_weight': 10,
            'use_material_count_distance_weight': True,
            'material_count_distance_weight_weight': 10,
            'use_piece_value_distance_weight': True,
            'piece_value_distance_weight_weight': 10,
        }

    def _get_default_game_config(self) -> GameConfig:
        """Get default game configuration"""
        return {
            "game_count": 1,
            "starting_position": "default",
            "white_player": "v7p3r",
            "black_player": "stockfish"
        }

    def _get_default_puzzle_config(self) -> PuzzleConfig:
        """Get default puzzle configuration"""
        return {
            "puzzle_database": {
                "db_path": "puzzle_data.db",
                "selection": {
                    "min_rating": 800,
                    "max_rating": 3000,
                    "batch_size": 25,
                    "themes": ["mate"],
                    "strict_theme_matching": False
                },
                "adaptive_elo": {
                    "enabled": False,
                    "starting_elo": 1200,
                    "increment": 100,
                    "decrement": 50,
                    "success_threshold": 0.8
                },
                "maintenance": {
                    "auto_vacuum": True,
                    "max_attempts_per_puzzle": 10
                }
            },
            "puzzle_solver": {
                "engine": {
                    "depth": 3,
                    "time_limit": 5000
                },
                "tracking": {
                    "record_attempts": True,
                    "save_positions": True
                },
                "integration": {
                    "update_transposition_table": False,
                    "stockfish_verification": False
                },
                "display": {
                    "show_solution": True
                }
            }
        }

    def _get_default_metrics_config(self) -> MetricsConfig:
        """Get default metrics configuration"""
        return {
            "metrics_to_track": [
                "evaluation",
                "depth",
                "nodes_searched",
                "time_taken"
            ],
            "include_engines": ["v7p3r"],
            "exclude_engine_ids": [],
            "group_by": "engine_id",
            "respect_exclusion_flags": True,
            "default_grouping": "engine_id",
            "show_engine_version": True,
            "show_engine_config_hash": True
        }

    def load_config(self, config_path: str) -> None:
        """Load a specific configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = json.loads(f.read())
                if 'engine_config' in config:
                    self.engine_config = config['engine_config']
                if 'game_config' in config:
                    self.game_config = config['game_config']
                if 'puzzle_config' in config:
                    self.puzzle_config = config['puzzle_config']
                if 'metrics_config' in config:
                    self.metrics_config = config['metrics_config']
        except Exception as e:
            print(f"Error loading config from {config_path}: {str(e)}")
            # Fall back to defaults if loading fails
            self.engine_config = cast(EngineConfig, self._get_default_engine_config())
            self.game_config = cast(GameConfig, self._get_default_game_config())
            self.puzzle_config = cast(PuzzleConfig, self._get_default_puzzle_config())
            self.metrics_config = cast(MetricsConfig, self._get_default_metrics_config())