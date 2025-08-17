# v7p3r_config.py

"""Configuration Handler for V7P3R Chess Engine
Loads and manages engine configuration settings from JSON files.
"""

import json
import os

class V7P3RConfig:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_file} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file {self.config_file}")
    
    def get_game_config(self):
        """Get game configuration settings"""
        return self.config.get("game_config", {})
    
    def get_engine_config(self):
        """Get v7p3r engine configuration settings"""
        return self.config.get("engine_config", {})
    
    def get_stockfish_config(self):
        """Get Stockfish configuration settings"""
        return self.config.get("stockfish_config", {})
    
    def get_puzzle_config(self):
        """Get puzzle configuration settings"""
        return self.config.get("puzzle_config", {})
    
    def get_setting(self, section, key, default=None):
        """Get a specific setting from a section"""
        return self.config.get(section, {}).get(key, default)
    
    def is_enabled(self, section, key):
        """Check if a feature is enabled"""
        return self.get_setting(section, key, False)
