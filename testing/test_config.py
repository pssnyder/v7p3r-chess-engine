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
        
    def test_config_loading(self):
        """Test configuration loading functionality."""
        # Create a test config file
        test_config = {
            'engine': {
                'depth': 4,
                'max_depth': 6,
                'use_quiescence': True
            }
        }
        
        config_path = os.path.join(parent_dir, 'configs', 'test_config.json')
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
            
        # Load configuration
        self.config_manager.load_config(config_path)
        engine_config = self.config_manager.get_engine_config()
        
        # Clean up
        os.remove(config_path)
        
        # Verify config was loaded
        self.assertIsNotNone(engine_config)
        
    def test_get_default_config(self):
        """Test getting default configuration."""
        default_config = self.config_manager.get_engine_config()
        self.assertIsNotNone(default_config)
        
    def test_config_validation(self):
        """Test configuration structure validation."""
        engine_config = self.config_manager.get_engine_config()
        
        # Required fields should be present
        required_fields = [
            'depth',
            'max_depth',
            'use_quiescence',
            'quiescence_depth',
            'search_time_limit'
        ]
        
        for field in required_fields:
            self.assertIn(field, engine_config)

if __name__ == '__main__':
    unittest.main()
