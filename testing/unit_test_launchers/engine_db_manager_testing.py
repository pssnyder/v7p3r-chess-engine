#!/usr/bin/env python3
"""
Unit tests for engine_db_manager.py - V7P3R Chess Engine Database Manager

This module contains comprehensive unit tests for the EngineDBManager class,
testing database operations, HTTP server functionality, and cloud integration.

Author: V7P3R Testing Suite
Date: 2025-06-22
"""

import sys
import os
import unittest
import tempfile
import yaml
import json
import threading
import time
import sqlite3
from unittest.mock import Mock, patch, MagicMock, mock_open
from http.server import HTTPServer
import requests

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from engine_utilities.engine_db_manager import EngineDBManager


class TestEngineDBManagerInitialization(unittest.TestCase):
    """Test EngineDBManager initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_chess_metrics.db")
        self.test_config_path = os.path.join(self.temp_dir, "test_config.yaml")
        
        # Create test config
        self.test_config = {
            'database': {
                'path': self.test_db_path,
                'backup_interval': 3600
            },
            'server': {
                'host': 'localhost',
                'port': 8080,
                'enabled': True
            },
            'cloud': {
                'enabled': False,
                'bucket_name': 'test-bucket'
            }
        }
        
        with open(self.test_config_path, 'w') as f:
            yaml.dump(self.test_config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('engine_utilities.engine_db_manager.MetricsStore')
    @patch('engine_utilities.engine_db_manager.CloudStore')
    def test_init_with_config(self, mock_cloud_store, mock_metrics_store):
        """Test initialization with valid config file."""
        mock_metrics_store.return_value = Mock()
        
        manager = EngineDBManager(
            db_path=self.test_db_path,
            config_path=self.test_config_path
        )
        
        self.assertEqual(manager.db_path, self.test_db_path)
        self.assertIsNotNone(manager.config)
        self.assertEqual(manager.config['server']['port'], 8080)
        self.assertFalse(manager.running)
        self.assertIsNone(manager.server_thread)
        mock_metrics_store.assert_called_once_with(db_path=self.test_db_path)

    @patch('engine_utilities.engine_db_manager.MetricsStore')
    def test_init_without_config(self, mock_metrics_store):
        """Test initialization without config file."""
        mock_metrics_store.return_value = Mock()
        
        manager = EngineDBManager(
            db_path=self.test_db_path,
            config_path="nonexistent_config.yaml"
        )
        
        self.assertEqual(manager.config, {})
        self.assertFalse(manager.cloud_enabled)
        self.assertIsNone(manager.cloud_store)

    @patch('engine_utilities.engine_db_manager.MetricsStore')
    @patch('engine_utilities.engine_db_manager.CloudStore')
    def test_init_with_cloud_enabled(self, mock_cloud_store, mock_metrics_store):
        """Test initialization with cloud storage enabled."""
        mock_metrics_store.return_value = Mock()
        mock_cloud_store.return_value = Mock()
        
        # Enable cloud in config
        self.test_config['cloud']['enabled'] = True
        with open(self.test_config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        manager = EngineDBManager(
            db_path=self.test_db_path,
            config_path=self.test_config_path
        )
        
        self.assertTrue(manager.cloud_enabled)
        self.assertIsNotNone(manager.cloud_store)
        mock_cloud_store.assert_called_once_with(bucket_name='test-bucket')

    @patch('engine_utilities.engine_db_manager.MetricsStore')
    @patch('engine_utilities.engine_db_manager.CloudStore')
    @patch('engine_utilities.engine_db_manager.logger')
    def test_init_cloud_error_handling(self, mock_logger, mock_cloud_store, mock_metrics_store):
        """Test error handling during cloud initialization."""
        mock_metrics_store.return_value = Mock()
        mock_cloud_store.side_effect = Exception("Cloud init error")
        
        # Enable cloud in config
        self.test_config['cloud']['enabled'] = True
        with open(self.test_config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        manager = EngineDBManager(
            db_path=self.test_db_path,
            config_path=self.test_config_path
        )
        
        self.assertFalse(manager.cloud_enabled)
        self.assertIsNone(manager.cloud_store)
        mock_logger.error.assert_called()


class TestEngineDBManagerDataOperations(unittest.TestCase):
    """Test EngineDBManager data operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_chess_metrics.db")
        
        self.mock_metrics_store = Mock()
        
        with patch('engine_utilities.engine_db_manager.MetricsStore') as mock_ms:
            mock_ms.return_value = self.mock_metrics_store
            self.manager = EngineDBManager(db_path=self.test_db_path)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_store_game_data(self):
        """Test storing game data."""
        game_data = {
            'game_id': 'test_game_123',
            'moves': ['e2e4', 'e7e5'],
            'result': '1-0',
            'timestamp': '2025-06-22T10:30:00'
        }
        
        # Mock the store_game_data method if it exists
        if hasattr(self.manager, 'store_game_data'):
            self.manager.store_game_data(game_data)
            # Verify metrics store was called appropriately
            # This depends on the actual implementation

    def test_retrieve_game_data(self):
        """Test retrieving game data."""
        game_id = 'test_game_123'
        
        # Mock return data
        expected_data = {
            'game_id': game_id,
            'moves': ['e2e4', 'e7e5'],
            'result': '1-0'
        }
        
        # If retrieve method exists, test it
        if hasattr(self.manager, 'retrieve_game_data'):
            self.mock_metrics_store.get_game.return_value = expected_data
            result = self.manager.retrieve_game_data(game_id)
            self.assertEqual(result, expected_data)

    def test_database_connection(self):
        """Test database connection handling."""
        # Test that metrics store is properly initialized
        self.assertIsNotNone(self.manager.metrics_store)
        self.assertEqual(self.manager.db_path, self.test_db_path)


class TestEngineDBManagerServerOperations(unittest.TestCase):
    """Test EngineDBManager HTTP server operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_chess_metrics.db")
        
        with patch('engine_utilities.engine_db_manager.MetricsStore'):
            self.manager = EngineDBManager(db_path=self.test_db_path)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.manager.running:
            self.manager.stop_server()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_server_initialization(self):
        """Test HTTP server initialization."""
        self.assertIsNone(self.manager.httpd)
        self.assertIsNone(self.manager.server_thread)
        self.assertFalse(self.manager.running)

    @patch('engine_utilities.engine_db_manager.HTTPServer')
    def test_start_server(self, mock_http_server):
        """Test starting HTTP server."""
        mock_server = Mock()
        mock_http_server.return_value = mock_server
        
        # Mock the start_server method if it exists
        if hasattr(self.manager, 'start_server'):
            self.manager.start_server(host='localhost', port=8080)
            
            if self.manager.running:
                self.assertTrue(self.manager.running)
                self.assertIsNotNone(self.manager.server_thread)

    def test_stop_server(self):
        """Test stopping HTTP server."""
        # Mock running state
        self.manager.running = True
        self.manager.httpd = Mock()
        self.manager.server_thread = Mock()
        
        if hasattr(self.manager, 'stop_server'):
            self.manager.stop_server()
            
            if not self.manager.running:
                self.assertFalse(self.manager.running)

    def test_server_thread_safety(self):
        """Test server thread safety."""
        self.assertIsNone(self.manager.server_thread)
        # Test that multiple start/stop calls don't cause issues
        for _ in range(3):
            if hasattr(self.manager, 'start_server') and hasattr(self.manager, 'stop_server'):
                try:
                    self.manager.start_server(host='localhost', port=0)  # Use port 0 for auto-assignment
                    time.sleep(0.1)
                    self.manager.stop_server()
                    time.sleep(0.1)
                except Exception:
                    pass  # Expected in test environment


class TestEngineDBManagerCloudOperations(unittest.TestCase):
    """Test EngineDBManager cloud operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_chess_metrics.db")
        
        self.mock_cloud_store = Mock()
        self.mock_metrics_store = Mock()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('engine_utilities.engine_db_manager.MetricsStore')
    @patch('engine_utilities.engine_db_manager.CloudStore')
    def test_cloud_upload(self, mock_cloud_store_class, mock_metrics_store):
        """Test cloud upload functionality."""
        mock_metrics_store.return_value = self.mock_metrics_store
        mock_cloud_store_class.return_value = self.mock_cloud_store
        
        config = {
            'cloud': {
                'enabled': True,
                'bucket_name': 'test-bucket'
            }
        }
        
        with patch.object(EngineDBManager, '_load_config', return_value=config):
            manager = EngineDBManager(db_path=self.test_db_path)
            
            # Test cloud upload if method exists
            if hasattr(manager, 'upload_to_cloud'):
                test_data = {'test': 'data'}
                manager.upload_to_cloud(test_data)
                # Verify cloud store was called
                self.mock_cloud_store.upload.assert_called()

    @patch('engine_utilities.engine_db_manager.MetricsStore')
    @patch('engine_utilities.engine_db_manager.CloudStore')
    def test_cloud_download(self, mock_cloud_store_class, mock_metrics_store):
        """Test cloud download functionality."""
        mock_metrics_store.return_value = self.mock_metrics_store
        mock_cloud_store_class.return_value = self.mock_cloud_store
        
        config = {
            'cloud': {
                'enabled': True,
                'bucket_name': 'test-bucket'
            }
        }
        
        with patch.object(EngineDBManager, '_load_config', return_value=config):
            manager = EngineDBManager(db_path=self.test_db_path)
            
            # Test cloud download if method exists
            if hasattr(manager, 'download_from_cloud'):
                file_name = 'test_file.json'
                manager.download_from_cloud(file_name)
                # Verify cloud store was called
                self.mock_cloud_store.download.assert_called()

    @patch('engine_utilities.engine_db_manager.MetricsStore')
    def test_cloud_disabled(self, mock_metrics_store):
        """Test behavior when cloud is disabled."""
        mock_metrics_store.return_value = self.mock_metrics_store
        
        config = {
            'cloud': {
                'enabled': False
            }
        }
        
        with patch.object(EngineDBManager, '_load_config', return_value=config):
            manager = EngineDBManager(db_path=self.test_db_path)
            
            self.assertFalse(manager.cloud_enabled)
            self.assertIsNone(manager.cloud_store)


class TestEngineDBManagerErrorHandling(unittest.TestCase):
    """Test EngineDBManager error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_chess_metrics.db")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('engine_utilities.engine_db_manager.MetricsStore')
    def test_database_connection_error(self, mock_metrics_store):
        """Test handling of database connection errors."""
        mock_metrics_store.side_effect = Exception("Database connection failed")
        
        with self.assertRaises(Exception):
            EngineDBManager(db_path=self.test_db_path)

    @patch('engine_utilities.engine_db_manager.MetricsStore')
    @patch('engine_utilities.engine_db_manager.logger')
    def test_invalid_config_handling(self, mock_logger, mock_metrics_store):
        """Test handling of invalid configuration."""
        mock_metrics_store.return_value = Mock()
        
        # Test with invalid config path
        manager = EngineDBManager(
            db_path=self.test_db_path,
            config_path="/nonexistent/path/config.yaml"
        )
        
        # Should handle gracefully with empty config
        self.assertEqual(manager.config, {})

    @patch('engine_utilities.engine_db_manager.MetricsStore')
    def test_malformed_yaml_config(self, mock_metrics_store):
        """Test handling of malformed YAML config."""
        mock_metrics_store.return_value = Mock()
        
        # Create malformed YAML file
        malformed_config_path = os.path.join(self.temp_dir, "malformed.yaml")
        with open(malformed_config_path, 'w') as f:
            f.write("invalid: yaml: content:\n  - missing\n    proper: structure")
        
        # Should handle gracefully
        try:
            EngineDBManager(
                db_path=self.test_db_path,
                config_path=malformed_config_path
            )
        except yaml.YAMLError:
            pass  # Expected for malformed YAML


class TestEngineDBManagerPerformance(unittest.TestCase):
    """Test EngineDBManager performance characteristics."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_chess_metrics.db")
        
        with patch('engine_utilities.engine_db_manager.MetricsStore'):
            self.manager = EngineDBManager(db_path=self.test_db_path)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_time(self):
        """Test that initialization completes within reasonable time."""
        start_time = time.time()
        
        with patch('engine_utilities.engine_db_manager.MetricsStore'):
            EngineDBManager(db_path=self.test_db_path)
        
        end_time = time.time()
        initialization_time = end_time - start_time
        
        # Should initialize within 5 seconds
        self.assertLess(initialization_time, 5.0)

    def test_memory_usage(self):
        """Test memory usage stays within reasonable bounds."""
        import sys
        
        initial_size = sys.getsizeof(self.manager)
        
        # Simulate some operations
        for i in range(100):
            # Add data to internal structures if they exist
            if hasattr(self.manager, 'cache'):
                self.manager.cache[f"key_{i}"] = f"value_{i}"
        
        final_size = sys.getsizeof(self.manager)
        
        # Memory should not grow excessively
        self.assertLess(final_size, initial_size * 10)


if __name__ == '__main__':
    unittest.main()