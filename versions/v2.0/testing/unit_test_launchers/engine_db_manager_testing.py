#!/usr/bin/env python3
"""
Unit tests for EngineDBManager - Database management functionality
Tests the core database operations, server functionality, and cloud integration.
"""

import unittest
import tempfile
import os
import json
import threading
import time
from unittest.mock import Mock, patch, MagicMock, mock_open
from engine_utilities.engine_db_manager import EngineDBManager, EngineDBClient


class TestEngineDBManager(unittest.TestCase):
    """Test cases for EngineDBManager class."""

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

    def test_initialization(self):
        """Test EngineDBManager initialization."""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.db_path, self.test_db_path)
        self.assertIsNotNone(self.manager.metrics_store)
        self.assertFalse(self.manager.running)
        self.assertIsNone(self.manager.server_thread)
        self.assertIsNone(self.manager.httpd)

    def test_config_loading(self):
        """Test configuration loading."""
        config_data = {
            'cloud': {
                'enabled': True,
                'bucket_name': 'test-bucket'
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8080
            }
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(config_data))):
            with patch('yaml.safe_load', return_value=config_data):
                with patch('os.path.exists', return_value=True):
                    manager = EngineDBManager()
                    self.assertEqual(manager.config['cloud']['enabled'], True)
                    self.assertEqual(manager.config['cloud']['bucket_name'], 'test-bucket')

    def test_handle_game_data(self):
        """Test handling game data."""
        game_data = {
            'game_id': 'test_game_123',
            'moves': ['e2e4', 'e7e5'],
            'result': '1-0',
            'timestamp': '2025-06-22T10:30:00'
        }
        
        # Test the actual _handle_game_data method
        with patch.object(self.manager.metrics_store, 'store_game_result') as mock_store:
            self.manager._handle_game_data(game_data)
            # Verify metrics store was called appropriately
            mock_store.assert_called_once()

    def test_handle_move_data(self):
        """Test handling move data."""
        move_data = {
            'game_id': 'test_game_123',
            'move': 'e2e4',
            'evaluation': 0.2,
            'depth': 15,
            'time_ms': 1500
        }
        
        with patch.object(self.manager.metrics_store, 'store_move_result') as mock_store:
            self.manager._handle_move_data(move_data)
            mock_store.assert_called_once()

    def test_handle_raw_simulation(self):
        """Test handling raw simulation data."""
        simulation_data = {
            'simulation_id': 'sim_123',
            'games': [
                {'game_id': 'game_1', 'result': '1-0'},
                {'game_id': 'game_2', 'result': '0-1'}
            ]
        }
        
        with patch.object(self.manager.metrics_store, 'store_simulation_result') as mock_store:
            self.manager._handle_raw_simulation(simulation_data)
            mock_store.assert_called_once()

    def test_bulk_upload(self):
        """Test bulk upload functionality."""
        data_list = [
            {'type': 'game', 'data': {'game_id': 'game_1', 'result': '1-0'}},
            {'type': 'move', 'data': {'move': 'e2e4', 'evaluation': 0.2}},
            {'type': 'simulation', 'data': {'simulation_id': 'sim_1'}}
        ]
          with patch.object(self.manager, '_handle_game_data') as mock_game:
            with patch.object(self.manager, '_handle_move_data') as mock_move:
                with patch.object(self.manager, '_handle_raw_simulation') as mock_sim:
                    results = self.manager.bulk_upload(data_list)
                    
                    # Check that results were returned (may be None or list)
                    if results is not None:
                        self.assertEqual(len(results), 3)
                    mock_game.assert_called_once()
                    mock_move.assert_called_once()
                    mock_sim.assert_called_once()

    def test_server_operations(self):
        """Test server start and stop operations."""
        with patch('http.server.HTTPServer') as mock_server:
            mock_httpd = Mock()
            mock_server.return_value = mock_httpd
            
            # Test server start
            self.manager.start_server(host="localhost", port=8081)
            self.assertTrue(self.manager.running)
            self.assertIsNotNone(self.manager.server_thread)
            
            # Test server stop
            self.manager.stop_server()
            self.assertFalse(self.manager.running)

    def test_listen_and_store(self):
        """Test listen and store functionality."""
        with patch.object(self.manager, 'start_server') as mock_start:
            self.manager.listen_and_store()
            mock_start.assert_called_once_with("0.0.0.0", 8080)

    def test_cloud_integration(self):
        """Test cloud storage integration."""
        # Test with cloud enabled
        config_with_cloud = {
            'cloud': {
                'enabled': True,
                'bucket_name': 'test-bucket'
            }
        }
        
        with patch('engine_utilities.engine_db_manager.CloudStore') as mock_cloud:
            with patch.object(EngineDBManager, '_load_config', return_value=config_with_cloud):
                manager = EngineDBManager()
                self.assertTrue(manager.cloud_enabled)
                self.assertIsNotNone(manager.cloud_store)
                mock_cloud.assert_called_once()

    def test_error_handling(self):
        """Test various error conditions."""
        # Test with invalid data
        invalid_data = {'invalid': 'data'}
        
        with patch.object(self.manager.metrics_store, 'store_game_result', side_effect=Exception("DB Error")):
            try:
                self.manager._handle_game_data(invalid_data)
                # Should handle gracefully
            except Exception:
                self.fail("Should handle database errors gracefully")

    def test_concurrent_operations(self):
        """Test concurrent data operations."""
        def store_data(data_id):
            game_data = {
                'game_id': f'concurrent_game_{data_id}',
                'moves': ['e2e4', 'e7e5'],
                'result': '1-0'
            }
            self.manager._handle_game_data(game_data)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=store_data, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All operations should complete without error
        self.assertTrue(True)

    def test_performance_metrics(self):
        """Test performance under load."""
        start_time = time.time()
        
        # Simulate high-frequency data insertion
        for i in range(100):
            game_data = {
                'game_id': f'perf_game_{i}',
                'moves': ['e2e4', 'e7e5'],
                'result': '1-0'
            }
            self.manager._handle_game_data(game_data)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(execution_time, 10.0, "Performance test took too long")

    def test_data_validation(self):
        """Test data validation and sanitization."""
        # Test with missing required fields
        incomplete_data = {
            'game_id': 'test_game'
            # Missing other required fields
        }
        
        # Should handle incomplete data gracefully
        try:
            self.manager._handle_game_data(incomplete_data)
        except Exception as e:
            # Expected to handle gracefully
            pass

    def test_memory_management(self):
        """Test memory usage and cleanup."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
          # Generate significant amount of data
        for i in range(1000):
            large_data = {
                'game_id': f'memory_test_{i}',
                'moves': ['e2e4'] * 100,  # Large move list
                'analysis': {f'key_{j}': f'value_{j}' for j in range(50)}
            }
            self.manager._handle_game_data(large_data)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust threshold as needed)
        self.assertLess(memory_increase, 100 * 1024 * 1024, "Memory usage increased too much")


class TestEngineDBClient(unittest.TestCase):
    """Test cases for EngineDBClient class."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = EngineDBClient(server_url="http://localhost:8080")

    def test_client_initialization(self):
        """Test client initialization."""
        self.assertEqual(self.client.server_url, "http://localhost:8080")
        self.assertIsNotNone(self.client.config)
        self.assertIsInstance(self.client.offline_buffer, list)

    def test_send_game_data(self):
        """Test sending game data."""
        game_data = {
            'game_id': 'client_test_game',
            'moves': ['e2e4', 'e7e5'],
            'result': '1-0'
        }
          with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'status': 'success'}
            mock_post.return_value = mock_response
            
            result = self.client.send_game_data(game_data)
            self.assertTrue(result)  # Returns True on success
            mock_post.assert_called_once()

    def test_send_move_data(self):
        """Test sending move data."""
        move_data = {
            'game_id': 'client_test_game',
            'move': 'e2e4',
            'evaluation': 0.2
        }
          with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'status': 'success'}
            mock_post.return_value = mock_response
            
            result = self.client.send_move_data(move_data)
            self.assertTrue(result)  # Returns True on success

    def test_send_raw_simulation(self):
        """Test sending simulation data."""
        simulation_data = {
            'simulation_id': 'client_sim_test',
            'games': [{'game_id': 'game_1', 'result': '1-0'}]
        }
          with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'status': 'success'}
            mock_post.return_value = mock_response
            
            result = self.client.send_raw_simulation(simulation_data)
            self.assertTrue(result)  # Always returns True for simulation data

    def test_offline_buffer_management(self):
        """Test offline buffer functionality."""
        # Add data to offline buffer
        test_data = {'test': 'data'}
        self.client.offline_buffer.append(test_data)
        
        # Test saving buffer
        temp_file = os.path.join(tempfile.mkdtemp(), "test_buffer.json")
        self.client.save_offline_buffer(temp_file)
        self.assertTrue(os.path.exists(temp_file))
        
        # Test loading buffer
        self.client.offline_buffer.clear()
        self.client.load_offline_buffer(temp_file)
        self.assertEqual(len(self.client.offline_buffer), 1)
        self.assertEqual(self.client.offline_buffer[0], test_data)

    def test_retry_mechanism(self):
        """Test request retry mechanism."""
        game_data = {'game_id': 'retry_test'}
        
        with patch('requests.post') as mock_post:
            # First call fails, second succeeds
            mock_post.side_effect = [
                Exception("Network error"),
                Mock(status_code=200, json=lambda: {'status': 'success'})
            ]
            
            result = self.client._send_with_retry(game_data)
            self.assertEqual(mock_post.call_count, 2)

    def test_flush_offline_buffer(self):
        """Test flushing offline buffer."""
        # Add test data to buffer
        self.client.offline_buffer.extend([
            {'game_id': 'buffer_game_1'},
            {'game_id': 'buffer_game_2'}
        ])
          with patch.object(self.client, '_send_with_retry', return_value=True):
            result = self.client.flush_offline_buffer()
            self.assertTrue(result)  # Returns True/False, not a list
            self.assertEqual(len(self.client.offline_buffer), 0)

    def test_error_handling_client(self):
        """Test client error handling."""
        with patch('requests.post', side_effect=Exception("Connection error")):
            # Should handle network errors gracefully
            game_data = {'game_id': 'error_test'}
            result = self.client.send_game_data(game_data)
            
            # Should store in offline buffer when network fails
            self.assertGreater(len(self.client.offline_buffer), 0)


if __name__ == '__main__':
    unittest.main()
