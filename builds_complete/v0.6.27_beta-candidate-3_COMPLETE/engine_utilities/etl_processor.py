"""
ETL module for transforming raw chess engine data into structured analytics data.

This module implements a robust ETL process that:
1. Extracts raw game and move data from cloud storage or local files
2. Validates and cleans the data
3. Transforms it into an analytics-optimized schema
4. Loads it into a reporting database

The ETL process is designed to be idempotent, allowing for reprocessing of data
when schemas or business logic changes.

FEATURES:
- Extract from local SQLite database and Google Cloud Firestore
- Data validation and cleaning
- Transformation into a normalized, analytics-optimized schema
- Batch processing with parallelization
- Idempotent operation with job tracking
- Comprehensive job metrics collection
- Support for versioning and backfilling/reprocessing
- Google Cloud Scheduler integration for robust scheduling
- Resource usage monitoring and reporting

DEPRECATED:
The original metrics system is still available but considered deprecated.
This module provides a more robust analytics-focused alternative.
"""

import os
import json
import yaml
import time
import uuid
import logging
import datetime
import sqlite3
import threading
import pandas as pd
import traceback
import argparse
import sys
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logging/etl_process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("chess_etl")

@dataclass
class ETLJobMetrics:
    """Metrics tracked for each ETL job run."""
    job_id: str
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    records_processed: int = 0
    records_failed: int = 0
    records_skipped: int = 0
    bytes_processed: int = 0
    extraction_time_seconds: float = 0
    transform_time_seconds: float = 0
    load_time_seconds: float = 0
    validation_time_seconds: float = 0
    cpu_usage_percent: float = 0
    memory_usage_mb: float = 0
    data_version: str = "1.0.0"
    source_type: str = "both"  # local, cloud, or both
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
    
    def add_error(self, error_type: str, message: str, source_id: Optional[str] = None):
        """Add an error to the metrics."""
        self.errors.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "type": error_type,
            "message": message,
            "source_id": source_id,
            "stack_trace": traceback.format_exc()
        })
    
    def add_warning(self, warning_type: str, message: str, source_id: Optional[str] = None):
        """Add a warning to the metrics."""
        self.warnings.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "type": warning_type,
            "message": message,
            "source_id": source_id
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "job_id": self.job_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "records_processed": self.records_processed,
            "records_failed": self.records_failed,
            "records_skipped": self.records_skipped,
            "bytes_processed": self.bytes_processed,
            "extraction_time_seconds": self.extraction_time_seconds,
            "transform_time_seconds": self.transform_time_seconds,
            "load_time_seconds": self.load_time_seconds,
            "validation_time_seconds": self.validation_time_seconds,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "data_version": self.data_version,
            "source_type": self.source_type,
            "errors": self.errors,            "warnings": self.warnings,
            "success_rate": (
                (self.records_processed / (self.records_processed + self.records_failed))
                if (self.records_processed + self.records_failed) > 0 
                else 0
            ),
            "total_time_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time else None
            )
        }


class ChessAnalyticsETL:
    """
    ETL processor for chess game data.
    Extracts from local chess_metrics.db only. No cloud or file support.
    """
    
    def __init__(self, config_path="config/etl_config.yaml"):
        """
        Initialize the ETL processor.
        
        Args:
            config_path: Path to the ETL configuration file.
        """
        self.config = self._load_config(config_path)
        self.job_metrics = ETLJobMetrics(
            job_id=str(uuid.uuid4()),
            start_time=datetime.datetime.now()
        )
        # Only local DB
        self.metrics_db_path = self.config.get('metrics_db', {}).get('path', 'metrics/chess_metrics.db')
        self.db_path = self.config.get('reporting_db', {}).get('path', 'metrics/chess_analytics.db')
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.db_lock = threading.RLock()
        self._init_reporting_db()
        self.processed_game_ids = set()
        self.processed_move_ids = set()
        self.batch_size = self.config.get('processing', {}).get('batch_size', 100)
        self.max_workers = self.config.get('processing', {}).get('max_workers', 4)
        logger.info(f"ChessAnalyticsETL initialized with job ID: {self.job_metrics.job_id}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}
    
    def _init_reporting_db(self):
        """Initialize the reporting database schema."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            
            # Store ETL job metrics
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS etl_job_metrics (
                job_id TEXT PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                records_processed INTEGER,
                records_failed INTEGER,
                records_skipped INTEGER,
                bytes_processed INTEGER,
                extraction_time_seconds REAL,
                transform_time_seconds REAL,
                load_time_seconds REAL,
                success_rate REAL,
                total_time_seconds REAL,
                errors TEXT
            )
            ''')
            
            # Game Analytics table - designed for fast queries
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_analytics (
                id TEXT PRIMARY KEY,
                game_id TEXT,
                timestamp TEXT,
                processing_date TEXT,
                schema_version TEXT,
                white_engine TEXT,
                black_engine TEXT,
                white_engine_type TEXT,
                black_engine_type TEXT,
                white_engine_version TEXT,
                black_engine_version TEXT,
                result TEXT,
                termination_reason TEXT,
                game_length INTEGER,
                avg_move_time_white REAL,
                avg_move_time_black REAL,
                avg_eval_white REAL,
                avg_eval_black REAL,
                avg_depth_white REAL,
                avg_depth_black REAL,
                avg_nodes_white REAL,
                avg_nodes_black REAL,
                max_eval_white REAL,
                max_eval_black REAL,
                min_eval_white REAL,
                min_eval_black REAL,
                opening_name TEXT,
                opening_moves TEXT,
                game_phase_stats TEXT,
                material_balance_stats TEXT,
                move_accuracy_white REAL,
                move_accuracy_black REAL,
                blunder_count_white INTEGER,
                blunder_count_black INTEGER,
                capture_count_white INTEGER,
                capture_count_black INTEGER,
                check_count_white INTEGER,
                check_count_black INTEGER,
                source_type TEXT,
                source_system TEXT,
                tags TEXT
            )
            ''')
            
            # Create indexes for common queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ga_white_engine ON game_analytics(white_engine)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ga_black_engine ON game_analytics(black_engine)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ga_result ON game_analytics(result)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ga_timestamp ON game_analytics(timestamp)')
            
            # Move Analytics table - detailed move statistics
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS move_analytics (
                id TEXT PRIMARY KEY,
                game_id TEXT,
                move_number INTEGER,
                processing_date TEXT,
                schema_version TEXT,
                half_move INTEGER,
                player_color TEXT,
                move_uci TEXT,
                move_san TEXT,
                time_taken REAL,
                eval_before REAL,
                eval_after REAL,
                eval_change REAL,
                position_type TEXT,
                material_balance_before INTEGER,
                material_balance_after INTEGER,
                nodes_searched INTEGER,
                depth INTEGER,
                is_capture INTEGER,
                is_check INTEGER,
                is_checkmate INTEGER,
                is_castle INTEGER,
                is_promotion INTEGER,
                move_accuracy REAL,
                is_blunder INTEGER,
                piece_moved TEXT,
                piece_captured TEXT,
                FOREIGN KEY (game_id) REFERENCES game_analytics(game_id)
            )
            ''')
            
            # Create indexes for move analytics
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ma_game_id ON move_analytics(game_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ma_player_color ON move_analytics(player_color)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ma_is_blunder ON move_analytics(is_blunder)')
            
            # Engine Performance table - aggregated statistics by engine
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS engine_performance (
                id TEXT PRIMARY KEY,
                engine_name TEXT,
                engine_version TEXT,
                engine_type TEXT,
                processing_date TEXT,
                schema_version TEXT,
                games_played INTEGER,
                games_won INTEGER,
                games_lost INTEGER,
                games_drawn INTEGER,
                win_rate REAL,
                avg_game_length REAL,
                avg_move_time REAL,
                avg_eval REAL,
                avg_depth REAL,
                avg_nodes REAL,
                avg_move_accuracy REAL,
                blunder_rate REAL,
                date_from TEXT,
                date_to TEXT,
                opponent_engines TEXT
            )
            ''')
            
            # Create indexes for engine performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ep_engine_name ON engine_performance(engine_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ep_engine_version ON engine_performance(engine_version)')
            
            # Data Processing Log table - tracks which data has been processed
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_processing_log (
                id TEXT PRIMARY KEY,
                source_id TEXT,
                source_type TEXT,
                processing_date TEXT,
                status TEXT,
                version TEXT,
                error_message TEXT
            )
            ''')
            
            # Create index for data processing log
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dpl_source_id ON data_processing_log(source_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dpl_status ON data_processing_log(status)')
            
            conn.commit()
            logger.info("Reporting database schema initialized")
    
    def _get_db_connection(self):
        """Get a connection to the reporting database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def extract_raw_game_data(self, limit=None, start_date=None, end_date=None) -> list:
        """
        Extract raw game and move data from local chess_metrics.db only.
        
        Args:
            limit: Maximum number of games to extract (for testing/debugging)
            start_date: Only extract games after this date (format: YYYYMMDD)
            end_date: Only extract games before this date (format: YYYYMMDD)
            
        Returns:
            List of raw game data dictionaries
        """
        extraction_start = time.time()
        raw_games = []
        
        # Determine which games have already been processed
        self._load_processed_game_ids()
        
        try:
            conn = sqlite3.connect(self.metrics_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build query for game_results
            query = "SELECT * FROM game_results"
            params = []
            if start_date:
                query += " WHERE timestamp >= ?"
                params.append(start_date)
            if end_date:
                if 'WHERE' in query:
                    query += " AND timestamp <= ?"
                else:
                    query += " WHERE timestamp <= ?"
                params.append(end_date)
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            cursor.execute(query, params)
            games = cursor.fetchall()
            for game_row in games:
                game_id = game_row['game_id']
                
                if game_id in self.processed_game_ids:
                    self.job_metrics.records_skipped += 1
                    continue
                
                # Get moves for this game
                move_cursor = conn.cursor()
                move_cursor.execute("SELECT * FROM move_metrics WHERE game_id = ? ORDER BY move_number ASC", (game_id,))
                moves = move_cursor.fetchall()
                
                # Build game dict
                game_data = dict(game_row)
                game_data['moves'] = [dict(m) for m in moves]
                raw_games.append(game_data)
                
                # Update metrics
                self.job_metrics.records_processed += 1
                self.job_metrics.bytes_processed += len(json.dumps(game_data))
            
            conn.close()
        except Exception as e:
            error_msg = f"Error extracting from chess_metrics.db: {e}"
            logger.error(error_msg)
            self.job_metrics.add_error("extraction", error_msg)
            self.job_metrics.records_failed += 1
        
        self.job_metrics.extraction_time_seconds = time.time() - extraction_start
        logger.info(f"Extracted {len(raw_games)} raw game records in {self.job_metrics.extraction_time_seconds:.2f} seconds")
        return raw_games
    
    def _load_processed_game_ids(self):
        """Load IDs of games that have already been processed."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT source_id FROM data_processing_log WHERE status = 'success'")
            rows = cursor.fetchall()
            self.processed_game_ids = {row['source_id'] for row in rows}
        logger.info(f"Loaded {len(self.processed_game_ids)} already processed game IDs")
    
    def validate_game_data(self, game_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate raw game data to ensure it has required fields and correct formats.
        
        Args:
            game_data: Raw game data dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        required_fields = ['id', 'timestamp']
        for field in required_fields:
            if field not in game_data:
                return False, f"Missing required field: {field}"
        
        # Check timestamp format
        try:
            # Try to parse the timestamp
            timestamp = game_data.get('timestamp')
            if isinstance(timestamp, str):
                datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return False, f"Invalid timestamp format: {game_data.get('timestamp')}"
        
        # Validate configs if present
        configs = game_data.get('configs', {})
        if configs:
            if not isinstance(configs, dict):
                return False, "Configs must be a dictionary"
        
        # Validate PGN if present
        pgn = game_data.get('pgn')
        if pgn and not isinstance(pgn, str):
            return False, "PGN must be a string"
        
        return True, None
    
    def transform_game_data(self, raw_games: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transform raw game data into analytics-ready format.
        
        Args:
            raw_games: List of raw game data dictionaries
            
        Returns:
            Tuple of (game_analytics, move_analytics, engine_performance)
        """
        transform_start = time.time()
        
        game_analytics = []
        move_analytics = []
        engine_performance_data = {}  # Will be aggregated
        
        # Process games in parallel for better performance
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_game = {
                executor.submit(self._transform_single_game, game): game
                for game in raw_games
            }
            
            # Process results as they complete
            for future in as_completed(future_to_game):
                game = future_to_game[future]
                try:
                    game_result, move_results = future.result()
                    if game_result:
                        game_analytics.append(game_result)
                        move_analytics.extend(move_results)
                          # Update engine performance data
                        self._update_engine_performance(engine_performance_data, game_result)
                except Exception as e:
                    error_msg = f"Error transforming game {game.get('id', 'unknown')}: {e}"
                    logger.error(error_msg)
                    self.job_metrics.add_error("transformation", error_msg, game.get('id', 'unknown'))
                    self.job_metrics.records_failed += 1
        
        # Convert engine performance data to list
        engine_performance = self._finalize_engine_performance(engine_performance_data)
        
        self.job_metrics.transform_time_seconds = time.time() - transform_start
        logger.info(
            f"Transformed {len(game_analytics)} games and {len(move_analytics)} moves "
            f"in {self.job_metrics.transform_time_seconds:.2f} seconds"
        )
        
        return game_analytics, move_analytics, engine_performance
    
    def _transform_single_game(self, game: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transform a single game into analytics format.
        
        Args:
            game: Raw game data dictionary
            
        Returns:
            Tuple of (game_analytics_dict, list_of_move_analytics_dicts)
        """
        # Validate the game data
        is_valid, error = self.validate_game_data(game)
        if not is_valid:
            logger.warning(f"Invalid game data: {error}")
            return None, []
        
        # Extract game data
        game_id = game.get('id')
        timestamp = game.get('timestamp')
        result = game.get('result', {}).get('winner') if isinstance(game.get('result'), dict) else game.get('result')
        
        # Extract config data
        configs = game.get('configs', {})
        game_config = configs.get('game', {})
        white_config = game_config.get('white_engine_config', {})
        black_config = game_config.get('black_engine_config', {})
        
        # Extract PGN data
        pgn = game.get('pgn', '')
        
        # Extract move data (if available)
        move_data = self._extract_moves_from_pgn(pgn) if pgn else []
        
        # Calculate game statistics
        game_stats = self._calculate_game_statistics(move_data, result)
        
        # Create game analytics record
        game_analytics = {
            'id': str(uuid.uuid4()),
            'game_id': game_id,
            'timestamp': timestamp,
            'processing_date': datetime.datetime.now().isoformat(),
            'schema_version': '1.0',
            'white_engine': white_config.get('engine', 'unknown'),
            'black_engine': black_config.get('engine', 'unknown'),
            'white_engine_type': white_config.get('engine_type', 'unknown'),
            'black_engine_type': black_config.get('engine_type', 'unknown'),
            'white_engine_version': white_config.get('engine_version', '1.0'),
            'black_engine_version': black_config.get('engine_version', '1.0'),
            'result': result or 'unknown',
            'termination_reason': game.get('termination', 'unknown'),
            'game_length': len(move_data) // 2 if move_data else 0,
            'avg_move_time_white': game_stats.get('avg_move_time_white', 0),
            'avg_move_time_black': game_stats.get('avg_move_time_black', 0),
            'avg_eval_white': game_stats.get('avg_eval_white', 0),
            'avg_eval_black': game_stats.get('avg_eval_black', 0),
            'avg_depth_white': game_stats.get('avg_depth_white', 0),
            'avg_depth_black': game_stats.get('avg_depth_black', 0),
            'avg_nodes_white': game_stats.get('avg_nodes_white', 0),
            'avg_nodes_black': game_stats.get('avg_nodes_black', 0),
            'max_eval_white': game_stats.get('max_eval_white', 0),
            'max_eval_black': game_stats.get('max_eval_black', 0),
            'min_eval_white': game_stats.get('min_eval_white', 0),
            'min_eval_black': game_stats.get('min_eval_black', 0),
            'opening_name': game_stats.get('opening_name', 'unknown'),
            'opening_moves': game_stats.get('opening_moves', ''),
            'game_phase_stats': json.dumps(game_stats.get('game_phase_stats', {})),
            'material_balance_stats': json.dumps(game_stats.get('material_balance_stats', {})),
            'move_accuracy_white': game_stats.get('move_accuracy_white', 0),
            'move_accuracy_black': game_stats.get('move_accuracy_black', 0),
            'blunder_count_white': game_stats.get('blunder_count_white', 0),
            'blunder_count_black': game_stats.get('blunder_count_black', 0),
            'capture_count_white': game_stats.get('capture_count_white', 0),
            'capture_count_black': game_stats.get('capture_count_black', 0),
            'check_count_white': game_stats.get('check_count_white', 0),
            'check_count_black': game_stats.get('check_count_black', 0),
            'source_type': 'local',
            'source_system': game.get('source_system', 'unknown'),
            'tags': json.dumps(game.get('tags', {}))
        }
        
        # Transform move data into analytics format
        move_analytics = []
        for move in move_data:
            move_analytics.append({
                'id': str(uuid.uuid4()),
                'game_id': game_id,
                'move_number': move.get('move_number', 0),
                'processing_date': datetime.datetime.now().isoformat(),
                'schema_version': '1.0',
                'half_move': move.get('half_move', 0),
                'player_color': move.get('player_color', 'unknown'),
                'move_uci': move.get('move_uci', ''),
                'move_san': move.get('move_san', ''),
                'time_taken': move.get('time_taken', 0),
                'eval_before': move.get('eval_before', 0),
                'eval_after': move.get('eval_after', 0),
                'eval_change': move.get('eval_after', 0) - move.get('eval_before', 0),
                'position_type': move.get('position_type', 'unknown'),
                'material_balance_before': move.get('material_balance_before', 0),
                'material_balance_after': move.get('material_balance_after', 0),
                'nodes_searched': move.get('nodes_searched', 0),
                'depth': move.get('depth', 0),
                'is_capture': 1 if move.get('is_capture', False) else 0,
                'is_check': 1 if move.get('is_check', False) else 0,
                'is_checkmate': 1 if move.get('is_checkmate', False) else 0,
                'is_castle': 1 if move.get('is_castle', False) else 0,
                'is_promotion': 1 if move.get('is_promotion', False) else 0,
                'move_accuracy': move.get('move_accuracy', 0),
                'is_blunder': 1 if move.get('is_blunder', False) else 0,
                'piece_moved': move.get('piece_moved', ''),
                'piece_captured': move.get('piece_captured', '')
            })
        
        return game_analytics, move_analytics
    
    def _extract_moves_from_pgn(self, pgn_text: str) -> List[Dict[str, Any]]:
        """
        Extract move data from PGN text.
        
        Args:
            pgn_text: PGN text string
            
        Returns:
            List of move dictionaries
        """
        moves = []
        
        # Very basic PGN parser - in a real implementation, use a proper chess library
        # This is a placeholder for demonstration
        
        move_pattern = r'(\d+)\.\s+([^\s]+)(?:\s+\{([^}]+)\})?(?:\s+([^\s]+)(?:\s+\{([^}]+)\})?)?'
        matches = re.findall(move_pattern, pgn_text)
        
        half_move = 0
        for match in matches:
            move_num = int(match[0])
            white_move = match[1]
            white_comment = match[2] if len(match) > 2 else ''
            black_move = match[3] if len(match) > 3 else ''
            black_comment = match[4] if len(match) > 4 else ''
            
            # Extract evaluation from comment
            def extract_eval(comment):
                eval_match = re.search(r'Eval:\s*([-+]?\d+\.\d+)', comment)
                return float(eval_match.group(1)) if eval_match else 0
            
            # Process white move
            if white_move:
                half_move += 1
                white_eval = extract_eval(white_comment)
                
                moves.append({
                    'move_number': move_num,
                    'half_move': half_move,
                    'player_color': 'white',
                    'move_san': white_move,
                    'eval_after': white_eval,
                    'time_taken': 0,  # Can't extract from standard PGN
                    'nodes_searched': 0,  # Can't extract from standard PGN
                    'depth': 0,  # Can't extract from standard PGN
                    'is_capture': '+' in white_move or 'x' in white_move,
                    'is_check': '+' in white_move,
                    'is_checkmate': '#' in white_move,
                    'is_castle': white_move in ['O-O', 'O-O-O'],
                    'is_promotion': '=' in white_move,
                    'move_accuracy': 0,  # Need more context to calculate
                    'is_blunder': False  # Need more context to calculate
                })
            
            # Process black move
            if black_move:
                half_move += 1
                black_eval = extract_eval(black_comment)
                
                moves.append({
                    'move_number': move_num,
                    'half_move': half_move,
                    'player_color': 'black',
                    'move_san': black_move,
                    'eval_after': black_eval,
                    'time_taken': 0,  # Can't extract from standard PGN
                    'nodes_searched': 0,  # Can't extract from standard PGN
                    'depth': 0,  # Can't extract from standard PGN
                    'is_capture': '+' in black_move or 'x' in black_move,
                    'is_check': '+' in black_move,
                    'is_checkmate': '#' in black_move,
                    'is_castle': black_move in ['O-O', 'O-O-O'],
                    'is_promotion': '=' in black_move,
                    'move_accuracy': 0,  # Need more context to calculate
                    'is_blunder': False  # Need more context to calculate
                })
        
        # Set eval_before based on the next move's eval_after
        for i in range(len(moves) - 1):
            moves[i]['eval_before'] = moves[i + 1]['eval_after']
        
        return moves
    def _calculate_game_statistics(self, move_data: List[Dict[str, Any]], result: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate various statistics from move data.
        
        Args:
            move_data: List of move dictionaries
            result: Game result string, or None if not available
            
        Returns:
            Dictionary of game statistics
        """
        stats = {
            'avg_move_time_white': 0,
            'avg_move_time_black': 0,
            'avg_eval_white': 0,
            'avg_eval_black': 0,
            'avg_depth_white': 0,
            'avg_depth_black': 0,
            'avg_nodes_white': 0,
            'avg_nodes_black': 0,
            'max_eval_white': float('-inf'),
            'max_eval_black': float('-inf'),
            'min_eval_white': float('inf'),
            'min_eval_black': float('inf'),
            'opening_name': 'unknown',
            'opening_moves': '',
            'game_phase_stats': {'opening': 0, 'middlegame': 0, 'endgame': 0},
            'material_balance_stats': {},
            'move_accuracy_white': 0,
            'move_accuracy_black': 0,
            'blunder_count_white': 0,
            'blunder_count_black': 0,
            'capture_count_white': 0,
            'capture_count_black': 0,
            'check_count_white': 0,
            'check_count_black': 0
        }
        
        if not move_data:
            return stats
        
        # Separate white and black moves
        white_moves = [m for m in move_data if m.get('player_color') == 'white']
        black_moves = [m for m in move_data if m.get('player_color') == 'black']
        
        # Calculate averages
        if white_moves:
            stats['avg_move_time_white'] = sum(m.get('time_taken', 0) for m in white_moves) / len(white_moves)
            stats['avg_eval_white'] = sum(m.get('eval_after', 0) for m in white_moves) / len(white_moves)
            stats['avg_depth_white'] = sum(m.get('depth', 0) for m in white_moves) / len(white_moves)
            stats['avg_nodes_white'] = sum(m.get('nodes_searched', 0) for m in white_moves) / len(white_moves)
            stats['max_eval_white'] = max(m.get('eval_after', float('-inf')) for m in white_moves)
            stats['min_eval_white'] = min(m.get('eval_after', float('inf')) for m in white_moves)
            stats['blunder_count_white'] = sum(1 for m in white_moves if m.get('is_blunder', False))
            stats['capture_count_white'] = sum(1 for m in white_moves if m.get('is_capture', False))
            stats['check_count_white'] = sum(1 for m in white_moves if m.get('is_check', False))
        
        if black_moves:
            stats['avg_move_time_black'] = sum(m.get('time_taken', 0) for m in black_moves) / len(black_moves)
            stats['avg_eval_black'] = sum(m.get('eval_after', 0) for m in black_moves) / len(black_moves)
            stats['avg_depth_black'] = sum(m.get('depth', 0) for m in black_moves) / len(black_moves)
            stats['avg_nodes_black'] = sum(m.get('nodes_searched', 0) for m in black_moves) / len(black_moves)
            stats['max_eval_black'] = max(m.get('eval_after', float('-inf')) for m in black_moves)
            stats['min_eval_black'] = min(m.get('eval_after', float('inf')) for m in black_moves)
            stats['blunder_count_black'] = sum(1 for m in black_moves if m.get('is_blunder', False))
            stats['capture_count_black'] = sum(1 for m in black_moves if m.get('is_capture', False))
            stats['check_count_black'] = sum(1 for m in black_moves if m.get('is_check', False))
        
        # Determine game phases (rough estimate)
        total_moves = len(move_data)
        if total_moves <= 10:
            stats['game_phase_stats']['opening'] = total_moves
        elif total_moves <= 30:
            stats['game_phase_stats']['opening'] = 10
            stats['game_phase_stats']['middlegame'] = total_moves - 10
        else:
            stats['game_phase_stats']['opening'] = 10
            stats['game_phase_stats']['middlegame'] = 20
            stats['game_phase_stats']['endgame'] = total_moves - 30
        
        # Extract opening moves (first 10 or fewer)
        opening_moves = [m.get('move_san', '') for m in move_data[:min(10, len(move_data))]]
        stats['opening_moves'] = ' '.join(opening_moves)
        
        return stats
    
    def _update_engine_performance(self, performance_data: Dict[str, Dict[str, Any]], game_result: Dict[str, Any]):
        """
        Update engine performance data with a new game result.
        
        Args:
            performance_data: Dictionary of engine performance data to update
            game_result: Game analytics data
        """
        # Update white engine data
        white_engine = game_result.get('white_engine')
        white_version = game_result.get('white_engine_version')
        white_engine_type = game_result.get('white_engine_type')
        white_key = f"{white_engine}_{white_version}_{white_engine_type}"
        
        if white_key not in performance_data:
            performance_data[white_key] = {
                'engine_name': white_engine,
                'engine_version': white_version,
                'engine_type': white_engine_type,
                'games_played': 0,
                'games_won': 0,
                'games_lost': 0,
                'games_drawn': 0,
                'total_move_time': 0,
                'total_eval': 0,
                'total_depth': 0,
                'total_nodes': 0,
                'total_accuracy': 0,
                'total_blunders': 0,
                'earliest_date': game_result.get('timestamp'),
                'latest_date': game_result.get('timestamp'),
                'opponents': set()
            }
        
        # Update black engine data
        black_engine = game_result.get('black_engine')
        black_version = game_result.get('black_engine_version')
        black_engine_type = game_result.get('black_engine_type')
        black_key = f"{black_engine}_{black_version}_{black_engine_type}"
        
        if black_key not in performance_data:
            performance_data[black_key] = {
                'engine_name': black_engine,
                'engine_version': black_version,
                'engine_type': black_engine_type,
                'games_played': 0,
                'games_won': 0,
                'games_lost': 0,
                'games_drawn': 0,
                'total_move_time': 0,
                'total_eval': 0,
                'total_depth': 0,
                'total_nodes': 0,
                'total_accuracy': 0,
                'total_blunders': 0,
                'earliest_date': game_result.get('timestamp'),
                'latest_date': game_result.get('timestamp'),
                'opponents': set()
            }
        
        # Update game counts based on result
        result = game_result.get('result')
        
        # Update white engine stats
        performance_data[white_key]['games_played'] += 1
        performance_data[white_key]['total_move_time'] += game_result.get('avg_move_time_white', 0) * game_result.get('game_length', 0)
        performance_data[white_key]['total_eval'] += game_result.get('avg_eval_white', 0)
        performance_data[white_key]['total_depth'] += game_result.get('avg_depth_white', 0)
        performance_data[white_key]['total_nodes'] += game_result.get('avg_nodes_white', 0)
        performance_data[white_key]['total_accuracy'] += game_result.get('move_accuracy_white', 0)
        performance_data[white_key]['total_blunders'] += game_result.get('blunder_count_white', 0)
        performance_data[white_key]['opponents'].add(black_engine)
        
        # Update latest date if newer
        if game_result.get('timestamp') > performance_data[white_key]['latest_date']:
            performance_data[white_key]['latest_date'] = game_result.get('timestamp')
        
        # Update earliest date if older
        if game_result.get('timestamp') < performance_data[white_key]['earliest_date']:
            performance_data[white_key]['earliest_date'] = game_result.get('timestamp')
        
        # Update black engine stats
        performance_data[black_key]['games_played'] += 1
        performance_data[black_key]['total_move_time'] += game_result.get('avg_move_time_black', 0) * game_result.get('game_length', 0)
        performance_data[black_key]['total_eval'] += game_result.get('avg_eval_black', 0)
        performance_data[black_key]['total_depth'] += game_result.get('avg_depth_black', 0)
        performance_data[black_key]['total_nodes'] += game_result.get('avg_nodes_black', 0)
        performance_data[black_key]['total_accuracy'] += game_result.get('move_accuracy_black', 0)
        performance_data[black_key]['total_blunders'] += game_result.get('blunder_count_black', 0)
        performance_data[black_key]['opponents'].add(white_engine)
        
        # Update latest date if newer
        if game_result.get('timestamp') > performance_data[black_key]['latest_date']:
            performance_data[black_key]['latest_date'] = game_result.get('timestamp')
        
        # Update earliest date if older
        if game_result.get('timestamp') < performance_data[black_key]['earliest_date']:
            performance_data[black_key]['earliest_date'] = game_result.get('timestamp')
        
        # Update win/loss/draw counts
        if result == '1-0':
            performance_data[white_key]['games_won'] += 1
            performance_data[black_key]['games_lost'] += 1
        elif result == '0-1':
            performance_data[white_key]['games_lost'] += 1
            performance_data[black_key]['games_won'] += 1
        elif result == '1/2-1/2':
            performance_data[white_key]['games_drawn'] += 1
            performance_data[black_key]['games_drawn'] += 1
    
    def _finalize_engine_performance(self, performance_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Finalize engine performance data by calculating averages and formatting for storage.
        
        Args:
            performance_data: Dictionary of engine performance data
            
        Returns:
            List of engine performance dictionaries
        """
        result = []
        
        for key, data in performance_data.items():
            games_played = data['games_played']
            if games_played == 0:
                continue
            
            # Calculate averages
            avg_move_time = data['total_move_time'] / games_played
            avg_eval = data['total_eval'] / games_played
            avg_depth = data['total_depth'] / games_played
            avg_nodes = data['total_nodes'] / games_played
            avg_accuracy = data['total_accuracy'] / games_played
            blunder_rate = data['total_blunders'] / games_played
            win_rate = data['games_won'] / games_played
            
            # Format for storage
            result.append({
                'id': str(uuid.uuid4()),
                'engine_name': data['engine_name'],
                'engine_version': data['engine_version'],
                'engine_type': data['engine_type'],
                'processing_date': datetime.datetime.now().isoformat(),
                'schema_version': '1.0',
                'games_played': games_played,
                'games_won': data['games_won'],
                'games_lost': data['games_lost'],
                'games_drawn': data['games_drawn'],
                'win_rate': win_rate,
                'avg_game_length': 0,  # Would need additional data to calculate
                'avg_move_time': avg_move_time,
                'avg_eval': avg_eval,
                'avg_depth': avg_depth,
                'avg_nodes': avg_nodes,
                'avg_move_accuracy': avg_accuracy,
                'blunder_rate': blunder_rate,
                'date_from': data['earliest_date'],
                'date_to': data['latest_date'],
                'opponent_engines': json.dumps(list(data['opponents']))
            })
        
        return result
    
    def load_analytics_data(self, game_analytics: List[Dict[str, Any]], move_analytics: List[Dict[str, Any]], engine_performance: List[Dict[str, Any]]) -> bool:
        """
        Load transformed data into the reporting database.
        
        Args:
            game_analytics: List of game analytics dictionaries
            move_analytics: List of move analytics dictionaries
            engine_performance: List of engine performance dictionaries
            
        Returns:
            True if successful
        """
        load_start = time.time()
        
        # Use a batch approach to avoid overloading the database
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # Process game analytics in batches
                for i in range(0, len(game_analytics), self.batch_size):
                    batch = game_analytics[i:i+self.batch_size]
                    self._load_game_analytics_batch(cursor, batch)
                    
                    # Log progress
                    logger.info(f"Loaded game analytics batch {i//self.batch_size + 1}/{(len(game_analytics) + self.batch_size - 1)//self.batch_size}")
                
                # Process move analytics in batches
                for i in range(0, len(move_analytics), self.batch_size):
                    batch = move_analytics[i:i+self.batch_size]
                    self._load_move_analytics_batch(cursor, batch)
                    
                    # Log progress
                    logger.info(f"Loaded move analytics batch {i//self.batch_size + 1}/{(len(move_analytics) + self.batch_size - 1)//self.batch_size}")
                
                # Process engine performance in batches
                for i in range(0, len(engine_performance), self.batch_size):
                    batch = engine_performance[i:i+self.batch_size]
                    self._load_engine_performance_batch(cursor, batch)
                    
                    # Log progress
                    logger.info(f"Loaded engine performance batch {i//self.batch_size + 1}/{(len(engine_performance) + self.batch_size - 1)//self.batch_size}")
                
                conn.commit()
            except Exception as e:                
                conn.rollback()
                error_msg = f"Error loading analytics data: {e}"
                logger.error(error_msg)
                self.job_metrics.add_error("load", error_msg)
                return False
        
        self.job_metrics.load_time_seconds = time.time() - load_start
        logger.info(f"Loaded all analytics data in {self.job_metrics.load_time_seconds:.2f} seconds")
        
        return True
    
    def _load_game_analytics_batch(self, cursor, batch: List[Dict[str, Any]]):
        """Load a batch of game analytics records."""
        for game in batch:
            # Insert into game_analytics table
            placeholders = ', '.join(['?'] * len(game))
            columns = ', '.join(game.keys())
            values = tuple(game.values())
            
            cursor.execute(f"""
            INSERT INTO game_analytics ({columns})
            VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET
                processing_date = excluded.processing_date,
                schema_version = excluded.schema_version
            """, values)
            
            # Update the data processing log
            cursor.execute("""
            INSERT INTO data_processing_log (id, source_id, source_type, processing_date, status, version)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                processing_date = excluded.processing_date,
                status = excluded.status,
                version = excluded.version
            """, (
                str(uuid.uuid4()),
                game['game_id'],
                'game',
                datetime.datetime.now().isoformat(),
                'success',
                '1.0'
            ))
    
    def _load_move_analytics_batch(self, cursor, batch: List[Dict[str, Any]]):
        """Load a batch of move analytics records."""
        for move in batch:
            # Insert into move_analytics table
            placeholders = ', '.join(['?'] * len(move))
            columns = ', '.join(move.keys())
            values = tuple(move.values())
            
            cursor.execute(f"""
            INSERT INTO move_analytics ({columns})
            VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET
                processing_date = excluded.processing_date,
                schema_version = excluded.schema_version
            """, values)
    
    def _load_engine_performance_batch(self, cursor, batch: List[Dict[str, Any]]):
        """Load a batch of engine performance records."""
        for perf in batch:
            # Insert into engine_performance table
            placeholders = ', '.join(['?'] * len(perf))
            columns = ', '.join(perf.keys())
            values = tuple(perf.values())
            
            cursor.execute(f"""
            INSERT INTO engine_performance ({columns})
            VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET
                processing_date = excluded.processing_date,
                schema_version = excluded.schema_version,
                games_played = excluded.games_played,
                games_won = excluded.games_won,
                games_lost = excluded.games_lost,
                games_drawn = excluded.games_drawn,
                win_rate = excluded.win_rate,
                avg_game_length = excluded.avg_game_length,
                avg_move_time = excluded.avg_move_time,
                avg_eval = excluded.avg_eval,
                avg_depth = excluded.avg_depth,
                avg_nodes = excluded.avg_nodes,
                avg_move_accuracy = excluded.avg_move_accuracy,
                blunder_rate = excluded.blunder_rate,
                date_to = excluded.date_to,
                opponent_engines = excluded.opponent_engines
            """, values)
    
    def save_job_metrics(self):
        """Save ETL job metrics to the database."""
        # Complete job metrics
        self.job_metrics.end_time = datetime.datetime.now()
        
        # Save to database
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            metrics_dict = self.job_metrics.to_dict()
            metrics_dict['errors'] = json.dumps(metrics_dict['errors'])
            
            placeholders = ', '.join(['?'] * len(metrics_dict))
            columns = ', '.join(metrics_dict.keys())
            values = tuple(metrics_dict.values())
            
            cursor.execute(f"""
            INSERT INTO etl_job_metrics ({columns})
            VALUES ({placeholders})
            ON CONFLICT(job_id) DO UPDATE SET
                end_time = excluded.end_time,
                records_processed = excluded.records_processed,
                records_failed = excluded.records_failed,
                records_skipped = excluded.records_skipped,
                bytes_processed = excluded.bytes_processed,
                extraction_time_seconds = excluded.extraction_time_seconds,
                transform_time_seconds = excluded.transform_time_seconds,
                load_time_seconds = excluded.load_time_seconds,
                success_rate = excluded.success_rate,
                total_time_seconds = excluded.total_time_seconds,
                errors = excluded.errors
            """, values)
            
            conn.commit()
        
        logger.info(f"Job metrics saved: {metrics_dict['records_processed']} processed, {metrics_dict['records_failed']} failed, {metrics_dict['records_skipped']} skipped")
    
    def run_etl_job(self, limit=None, start_date=None, end_date=None):
        """
        Run a complete ETL job.
        
        Args:
            limit: Maximum number of records to process (for testing)
            start_date: Only process records after this date (format: YYYYMMDD)
            end_date: Only process records before this date (format: YYYYMMDD)
            
        Returns:
            ETLJobMetrics instance with job results
        """
        logger.info(f"Starting ETL job with ID {self.job_metrics.job_id}")
        
        try:
            # Extract raw data
            raw_games = self.extract_raw_game_data(limit, start_date, end_date)
            
            if not raw_games:
                logger.info("No new data to process")
                self.job_metrics.end_time = datetime.datetime.now()
                self.save_job_metrics()
                return self.job_metrics
            
            # Transform data
            game_analytics, move_analytics, engine_performance = self.transform_game_data(raw_games)
            
            # Load data
            success = self.load_analytics_data(game_analytics, move_analytics, engine_performance)
            
            if not success:
                logger.error("ETL job failed during data loading")
            
            # Save job metrics
            self.save_job_metrics()
            
            logger.info(f"ETL job completed: {self.job_metrics.records_processed} records processed, {self.job_metrics.records_failed} failed")
            
            return self.job_metrics
        except Exception as e:            
            error_msg = f"ETL job failed: {e}"
            logger.error(error_msg)
            self.job_metrics.add_error("job", error_msg)
            self.job_metrics.end_time = datetime.datetime.now()
            self.save_job_metrics()
            return self.job_metrics
    
    def schedule_etl_job(self, frequency: str = "hourly"):
        pass  # Remove GCP scheduler logic


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ETL process for chess analytics')
    parser.add_argument('--limit', type=int, help='Limit the number of records to process')
    parser.add_argument('--start-date', help='Process records from this date (format: YYYYMMDD)')
    parser.add_argument('--end-date', help='Process records until this date (format: YYYYMMDD)')
    parser.add_argument('--config', default='config/etl_config.yaml', help='Path to ETL config file')
    parser.add_argument('--schedule', action='store_true', help='Schedule the ETL job with Google Cloud Scheduler')
    parser.add_argument('--frequency', default='hourly', help='Frequency for scheduled job (e.g., hourly, daily, weekly)')
    
    args = parser.parse_args()
    
    etl = ChessAnalyticsETL(config_path=args.config)
    
    if args.schedule:
        etl.schedule_etl_job(frequency=args.frequency)
    else:
        metrics = etl.run_etl_job(limit=args.limit, start_date=args.start_date, end_date=args.end_date)
        
        print(f"ETL job completed in {metrics.to_dict()['total_time_seconds']:.2f} seconds")
        print(f"Processed: {metrics.records_processed}, Failed: {metrics.records_failed}, Skipped: {metrics.records_skipped}")
    args = parser.parse_args()
    
    etl = ChessAnalyticsETL(config_path=args.config)
    
    if args.schedule:
        etl.schedule_etl_job(frequency=args.frequency)
    else:
        metrics = etl.run_etl_job(limit=args.limit, start_date=args.start_date, end_date=args.end_date)
        
        print(f"ETL job completed in {metrics.to_dict()['total_time_seconds']:.2f} seconds")
        print(f"Processed: {metrics.records_processed}, Failed: {metrics.records_failed}, Skipped: {metrics.records_skipped}")
