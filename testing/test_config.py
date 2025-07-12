"""Test suite for v7p3r configuration management.
Tests configuration loading, validation, and access patterns."""

import os
import sys
import json
import unittest
from typing import Dict, Any

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_config import v7p3rConfig

class TestV7P3RConfig(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = v7p3rConfig()
        
    def test_init_configuration(self):
        """Test configuration initialization."""
        self.assertIsNotNone(self.config_manager)
        engine_config = self.config_manager.get_engine_config()
        self.assertIsInstance(engine_config, dict)
        
    def test_default_values(self):
        """Test default configuration values."""
        config = self.config_manager.get_engine_config()
        # Test critical default values
        self.assertIn('depth', config)
        self.assertIn('max_depth', config)
        self.assertIn('use_quiescence', config)
        self.assertIn('search_time_limit', config)
        
    def test_config_override(self):
        """Test configuration override functionality."""
        original_config = self.config_manager.get_engine_config()
        test_override = {'depth': 5, 'max_depth': 7}
        
        # Apply override
        self.config_manager.override_config(test_override)
        new_config = self.config_manager.get_engine_config()
        
        # Verify override worked
        self.assertEqual(new_config['depth'], 5)
        self.assertEqual(new_config['max_depth'], 7)
        
    def test_config_validation(self):
        """Test configuration validation."""
        invalid_config = {'depth': 'invalid', 'max_depth': -1}
        with self.assertRaises(ValueError):
            self.config_manager.override_config(invalid_config)
            
    def test_config_persistence(self):
        """Test configuration persistence."""
        test_config = {
            'depth': 4,
            'max_depth': 6,
            'use_quiescence': True
        }
        
        # Save configuration
        config_path = os.path.join(parent_dir, 'configs', 'test_config.json')
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
            
        # Load configuration
        loaded_config = self.config_manager.load_config(config_path)
        self.assertEqual(loaded_config['depth'], 4)
        self.assertEqual(loaded_config['max_depth'], 6)
        
        # Clean up
        os.remove(config_path)

if __name__ == '__main__':
    unittest.main()
