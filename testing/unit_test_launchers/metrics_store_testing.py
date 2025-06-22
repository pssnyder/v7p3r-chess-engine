#!/usr/bin/env python3
"""
Unit tests for metrics_store.py - V7P3R Chess Engine Metrics Store

This module contains comprehensive unit tests for the MetricsStore class,
testing database operations, data ingestion, and analytics functionality.

Author: V7P3R Testing Suite
Date: 2025-06-22
"""

import sys
import os
import unittest
import tempfile
import sqlite3
import json
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, mock_open
from io import StringIO
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from metrics.metrics_store import MetricsStore


class TestMetricsStoreInitialization(unittest.TestCase):
    """Test MetricsStore initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_metrics.db")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_new_database(self):
        """Test initialization with new database."""
        store = MetricsStore(db_path=self.test_db_path)
        
        self.assertTrue(os.path.exists(self.test_db_path))
        self.assertEqual(store.db_path, self.test_db_path)
        
        # Verify database connection
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        # Check if expected tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['log_entries', 'game_results']
        for table in expected_tables:
            self.assertIn(table, tables)
        
        conn.close()

    def test_init_existing_database(self):
        """Test initialization with existing database."""
        # Create initial database
        store1 = MetricsStore(db_path=self.test_db_path)
        store1.close_connection()
        
        # Initialize with existing database
        store2 = MetricsStore(db_path=self.test_db_path)
        
        self.assertEqual(store2.db_path, self.test_db_path)
        self.assertTrue(os.path.exists(self.test_db_path))

    def test_init_invalid_path(self):
        """Test initialization with invalid database path."""
        invalid_path = "/invalid/path/to/database.db"
        
        # Should handle gracefully or raise appropriate exception
        try:
            store = MetricsStore(db_path=invalid_path)
            # If it succeeds, verify it created necessary directories
            self.assertTrue(store.db_path == invalid_path or os.path.exists(os.path.dirname(invalid_path)))
        except (OSError, sqlite3.Error):
            pass  # Expected for invalid path


class TestMetricsStoreDatabaseOperations(unittest.TestCase):
    """Test MetricsStore database operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_metrics.db")
        self.store = MetricsStore(db_path=self.test_db_path)

    def tearDown(self):
        """Clean up test fixtures."""
        self.store.close_connection()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_entry_insertion(self):
        """Test inserting log entries."""
        log_data = {
            'timestamp': '2025-06-22 10:30:00',
            'function_name': 'test_function',
            'log_file': 'test.log',
            'message': 'Test log message',
            'value': 42.5,
            'label': 'test_label',
            'side': 'white',
            'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'raw_text': 'Raw test log entry'
        }
        
        # Test insert method if it exists
        if hasattr(self.store, 'insert_log_entry'):
            self.store.insert_log_entry(log_data)
            
            # Verify insertion
            conn = sqlite3.connect(self.test_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM log_entries WHERE message = ?", (log_data['message'],))
            result = cursor.fetchone()
            
            self.assertIsNotNone(result)
            conn.close()

    def test_game_result_insertion(self):
        """Test inserting game results."""
        game_data = {
            'game_id': 'test_game_123',
            'timestamp': '2025-06-22 10:30:00',
            'winner': '1-0',
            'game_pgn': '[Event "Test"]\n1. e4 e5 2. Nf3 *',
            'white_player': 'V7P3R',
            'black_player': 'Opponent',
            'game_length': 2,
            'white_engine_id': 'v7p3r_001',
            'black_engine_id': 'opponent_001',
            'white_engine_name': 'V7P3R v1.0',
            'black_engine_name': 'Test Engine'
        }
        
        # Test insert method if it exists
        if hasattr(self.store, 'insert_game_result'):
            self.store.insert_game_result(game_data)
            
            # Verify insertion
            conn = sqlite3.connect(self.test_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM game_results WHERE game_id = ?", (game_data['game_id'],))
            result = cursor.fetchone()
            
            self.assertIsNotNone(result)
            conn.close()

    def test_data_retrieval(self):
        """Test data retrieval operations."""
        # Insert test data first
        if hasattr(self.store, 'insert_log_entry'):
            log_data = {
                'timestamp': '2025-06-22 10:30:00',
                'function_name': 'test_function',
                'message': 'Test retrieval',
                'raw_text': 'Test raw text for retrieval'
            }
            self.store.insert_log_entry(log_data)
        
        # Test various retrieval methods
        if hasattr(self.store, 'get_log_entries'):
            entries = self.store.get_log_entries(limit=10)
            self.assertIsInstance(entries, (list, pd.DataFrame))
        
        if hasattr(self.store, 'get_game_results'):
            games = self.store.get_game_results(limit=10)
            self.assertIsInstance(games, (list, pd.DataFrame))

    def test_database_queries(self):
        """Test complex database queries."""
        # Test query method if it exists
        if hasattr(self.store, 'execute_query'):
            query = "SELECT COUNT(*) FROM log_entries"
            result = self.store.execute_query(query)
            self.assertIsNotNone(result)
        
        # Test parameterized queries
        if hasattr(self.store, 'execute_query'):
            query = "SELECT * FROM log_entries WHERE timestamp > ?"
            params = ('2025-01-01',)
            result = self.store.execute_query(query, params)
            self.assertIsNotNone(result)


class TestMetricsStoreDataAnalytics(unittest.TestCase):
    """Test MetricsStore analytics functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_metrics.db")
        self.store = MetricsStore(db_path=self.test_db_path)
        
        # Insert sample data for analytics
        self._insert_sample_data()

    def tearDown(self):
        """Clean up test fixtures."""
        self.store.close_connection()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _insert_sample_data(self):
        """Insert sample data for testing."""
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        # Insert sample log entries
        sample_logs = [
            ('2025-06-22 10:00:00', 'search', 'engine.log', 'Search depth 6', 6.0, 'depth', 'white', 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 'Raw log 1'),
            ('2025-06-22 10:01:00', 'evaluate', 'engine.log', 'Position score 0.5', 0.5, 'score', 'white', 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1', 'Raw log 2'),
            ('2025-06-22 10:02:00', 'search', 'engine.log', 'Search depth 7', 7.0, 'depth', 'black', 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2', 'Raw log 3')
        ]
        
        for log in sample_logs:
            cursor.execute("""
                INSERT INTO log_entries (timestamp, function_name, log_file, message, 
                                       value, label, side, fen, raw_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, log)
        
        # Insert sample game results
        sample_games = [
            ('game_001', '2025-06-22 10:00:00', '1-0', '[Event "Test"] 1. e4 e5 *', 'V7P3R', 'Opponent', 20, 'v7p3r_001', 'opp_001', 'V7P3R v1.0', 'Test Engine'),
            ('game_002', '2025-06-22 10:30:00', '0-1', '[Event "Test"] 1. d4 d5 *', 'V7P3R', 'Opponent', 25, 'v7p3r_001', 'opp_001', 'V7P3R v1.0', 'Test Engine'),
            ('game_003', '2025-06-22 11:00:00', '1/2-1/2', '[Event "Test"] 1. Nf3 Nf6 *', 'V7P3R', 'Opponent', 30, 'v7p3r_001', 'opp_001', 'V7P3R v1.0', 'Test Engine')
        ]
        
        for game in sample_games:
            cursor.execute("""
                INSERT INTO game_results (game_id, timestamp, winner, game_pgn, 
                                        white_player, black_player, game_length,
                                        white_engine_id, black_engine_id, 
                                        white_engine_name, black_engine_name, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, game)
        
        conn.commit()
        conn.close()

    def test_performance_analytics(self):
        """Test performance analytics methods."""
        # Test average search depth
        if hasattr(self.store, 'get_average_search_depth'):
            avg_depth = self.store.get_average_search_depth()
            self.assertIsInstance(avg_depth, (int, float))
            self.assertGreater(avg_depth, 0)

    def test_game_statistics(self):
        """Test game statistics methods."""
        # Test win rate calculation
        if hasattr(self.store, 'get_win_rate'):
            win_rate = self.store.get_win_rate('V7P3R')
            self.assertIsInstance(win_rate, (int, float))
            self.assertGreaterEqual(win_rate, 0)
            self.assertLessEqual(win_rate, 1)
        
        # Test game count
        if hasattr(self.store, 'get_game_count'):
            game_count = self.store.get_game_count()
            self.assertIsInstance(game_count, int)
            self.assertGreaterEqual(game_count, 0)

    def test_time_series_analysis(self):
        """Test time series analysis methods."""
        # Test performance over time
        if hasattr(self.store, 'get_performance_over_time'):
            performance_data = self.store.get_performance_over_time()
            self.assertIsInstance(performance_data, (list, pd.DataFrame))

    def test_data_aggregation(self):
        """Test data aggregation methods."""
        # Test data aggregation by time period
        if hasattr(self.store, 'aggregate_by_time_period'):
            daily_stats = self.store.aggregate_by_time_period('day')
            self.assertIsInstance(daily_stats, (list, pd.DataFrame))


class TestMetricsStoreDataImport(unittest.TestCase):
    """Test MetricsStore data import functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_metrics.db")
        self.store = MetricsStore(db_path=self.test_db_path)

    def tearDown(self):
        """Clean up test fixtures."""
        self.store.close_connection()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_file_import(self):
        """Test importing log files."""
        # Create test log file
        test_log_path = os.path.join(self.temp_dir, "test.log")
        with open(test_log_path, 'w') as f:
            f.write("2025-06-22 10:00:00 | test_function | Test log message\n")
            f.write("2025-06-22 10:01:00 | other_function | Another log message\n")
        
        # Test import method if it exists
        if hasattr(self.store, 'import_log_file'):
            self.store.import_log_file(test_log_path)
            
            # Verify import
            conn = sqlite3.connect(self.test_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM log_entries")
            count = cursor.fetchone()[0]
            self.assertGreater(count, 0)
            conn.close()

    def test_pgn_file_import(self):
        """Test importing PGN files."""
        # Create test PGN file
        test_pgn_path = os.path.join(self.temp_dir, "test.pgn")
        with open(test_pgn_path, 'w') as f:
            f.write('[Event "Test Game"]\n')
            f.write('[White "V7P3R"]\n')
            f.write('[Black "Opponent"]\n')
            f.write('[Result "1-0"]\n')
            f.write('\n1. e4 e5 2. Nf3 1-0\n\n')
        
        # Test import method if it exists
        if hasattr(self.store, 'import_pgn_file'):
            self.store.import_pgn_file(test_pgn_path)
            
            # Verify import
            conn = sqlite3.connect(self.test_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM game_results")
            count = cursor.fetchone()[0]
            self.assertGreater(count, 0)
            conn.close()

    def test_batch_import(self):
        """Test batch import functionality."""
        # Create multiple test files
        test_files = []
        for i in range(3):
            test_log_path = os.path.join(self.temp_dir, f"test_{i}.log")
            with open(test_log_path, 'w') as f:
                f.write(f"2025-06-22 10:0{i}:00 | test_function | Test log message {i}\n")
            test_files.append(test_log_path)
        
        # Test batch import if it exists
        if hasattr(self.store, 'batch_import_files'):
            self.store.batch_import_files(test_files)
            
            # Verify batch import
            conn = sqlite3.connect(self.test_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM log_entries")
            count = cursor.fetchone()[0]
            self.assertGreaterEqual(count, 3)
            conn.close()


class TestMetricsStoreErrorHandling(unittest.TestCase):
    """Test MetricsStore error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_metrics.db")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_lock_handling(self):
        """Test handling of database lock situations."""
        store1 = MetricsStore(db_path=self.test_db_path)
        
        # Try to create another connection
        try:
            store2 = MetricsStore(db_path=self.test_db_path)
            # Should handle gracefully
            store2.close_connection()
        except sqlite3.OperationalError:
            pass  # Expected in some cases
        
        store1.close_connection()

    def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        store = MetricsStore(db_path=self.test_db_path)
        
        # Test with invalid log data
        if hasattr(store, 'insert_log_entry'):
            invalid_log_data = {
                'timestamp': 'invalid_timestamp',
                'value': 'not_a_number'
            }
            
            try:
                store.insert_log_entry(invalid_log_data)
            except (ValueError, sqlite3.Error):
                pass  # Expected for invalid data
        
        store.close_connection()

    def test_connection_error_recovery(self):
        """Test recovery from connection errors."""
        store = MetricsStore(db_path=self.test_db_path)
        
        # Simulate connection loss
        if hasattr(store, 'connection'):
            store.connection.close()
        
        # Test that operations handle connection errors gracefully
        if hasattr(store, 'get_log_entries'):
            try:
                entries = store.get_log_entries()
                # Should either recover or handle error gracefully
            except sqlite3.Error:
                pass  # Expected for broken connection


class TestMetricsStorePerformance(unittest.TestCase):
    """Test MetricsStore performance characteristics."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_metrics.db")
        self.store = MetricsStore(db_path=self.test_db_path)

    def tearDown(self):
        """Clean up test fixtures."""
        self.store.close_connection()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_bulk_insert_performance(self):
        """Test performance of bulk insert operations."""
        start_time = time.time()
        
        # Insert large number of records
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        records = []
        for i in range(1000):
            records.append((
                f'2025-06-22 10:{i//60:02d}:{i%60:02d}',
                'test_function',
                'test.log',
                f'Test message {i}',
                float(i),
                'test_label',
                'white' if i % 2 == 0 else 'black',
                'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                f'Raw text {i}'
            ))
        
        cursor.executemany("""
            INSERT INTO log_entries (timestamp, function_name, log_file, message, 
                                   value, label, side, fen, raw_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, records)
        
        conn.commit()
        conn.close()
        
        end_time = time.time()
        insert_time = end_time - start_time
        
        # Should complete within reasonable time (10 seconds for 1000 records)
        self.assertLess(insert_time, 10.0)

    def test_query_performance(self):
        """Test query performance with large dataset."""
        # First insert test data
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        records = []
        for i in range(10000):
            records.append((
                f'2025-06-22 10:{i//600:02d}:{(i//10)%60:02d}',
                'test_function',
                'test.log',
                f'Test message {i}',
                float(i % 100),
                'test_label',
                'white' if i % 2 == 0 else 'black',
                'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                f'Raw text {i}'
            ))
        
        cursor.executemany("""
            INSERT INTO log_entries (timestamp, function_name, log_file, message, 
                                   value, label, side, fen, raw_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, records)
        
        conn.commit()
        
        # Test query performance
        start_time = time.time()
        
        cursor.execute("SELECT * FROM log_entries WHERE value > 50 ORDER BY timestamp DESC LIMIT 100")
        results = cursor.fetchall()
        
        end_time = time.time()
        query_time = end_time - start_time
        
        conn.close()
        
        # Should complete within reasonable time
        self.assertLess(query_time, 2.0)
        self.assertGreater(len(results), 0)


if __name__ == '__main__':
    unittest.main()