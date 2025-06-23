# metrics_store.py

import os
import sqlite3
import json
import glob
import threading
import hashlib
from typing import Optional
import chess.pgn
import io
import pandas as pd
import re
import time
from datetime import datetime
import yaml

"""
Database Schema Documentation:

1. log_entries Table:
   - Stores log data from the chess engine.
   - Columns:
     - id: Primary key.
     - timestamp: Time of the log entry.
     - function_name: Name of the function generating the log.
     - log_file: Source log file.
     - message: Log message.
     - value: Numeric value associated with the log.
     - label: Label for the log entry.
     - side: Side (white/black) associated with the log.
     - fen: FEN string representing the board state.
     - raw_text: Raw log text (unique).
     - created_at: Timestamp of entry creation.

2. game_results Table:
   - Stores results of chess games.
   - Columns:
     - id: Primary key.
     - game_id: Unique identifier for the game.
     - timestamp: Time of the game.
     - winner: Winner of the game (1-0, 0-1, 1/2-1/2).
     - game_pgn: PGN of the game.
     - white_player: Name of the white player.
     - black_player: Name of the black player.
     - game_length: Number of moves in the game.
     - created_at: Timestamp of entry creation.
     - white_engine_id: Engine ID for white (added for direct access).
     - black_engine_id: Engine ID for black (added for direct access).
     - white_engine_name: Engine name for white (added for direct access).
     - black_engine_name: Engine name for black (added for direct access).
     - white_engine_version: Engine version for white (added for direct access).
     - black_engine_version: Engine version for black (added for direct access).
     - exclude_white_from_metrics: Exclusion flag for white's metrics (added for direct access).
     - exclude_black_from_metrics: Exclusion flag for black's metrics (added for direct access).


3. config_settings Table:
   - Stores AI configuration data.
   - Columns:
     - id: Primary key.
     - config_id: Unique identifier for the configuration.
     - timestamp: Time of the configuration.
     - game_id: Associated game ID.
     - config_data: JSON representation of the configuration.
     - engine_id: Engine ID (added for direct access).
     - engine_name: Engine name (added for direct access).
     - engine_version: Engine version (added for direct access).
     - created_at: Timestamp of entry creation.

4. metrics Table:
   - Stores computed metrics.
   - Columns:
     - id: Primary key.
     - metric_name: Name of the metric.
     - metric_value: Value of the metric.
     - side: Side (white/black) associated with the metric.
     - function_name: Function generating the metric.
     - timestamp: Time of the metric.
     - game_id: Associated game ID.
     - config_id: Associated configuration ID.
     - created_at: Timestamp of entry creation.

5. move_metrics Table (NEW):
   - Stores detailed metrics for each individual move.
   - Columns:
     - id: Primary key.
     - game_id: Foreign key to game_results.
     - move_number: The fullmove number (1, 2, 3...).
     - player_color: Color of the player making the move ('w' or 'b').
     - move_uci: The move in UCI format (e.g., 'e2e4').
     - fen_before: FEN string of the board *before* the move.
     - evaluation: The evaluation score after the move from the perspective of the player whose turn it *became*.
     - engine_type: The AI type used for this move (e.g., 'deepsearch', 'negamax').
     - depth: The search depth reached for this move.
     - nodes_searched: Number of nodes explored for this move.
     - time_taken: Time in seconds taken to find the move.
     - pv_line: Principal Variation (best line of play found).
     - created_at: Timestamp of entry creation.
"""

class MetricsStore:
    """
    A persistent storage solution for chess engine metrics.
    Uses SQLite to store parsed logs, game results, and configuration data.
    """
    
    def __init__(self, db_path="metrics/chess_metrics.db"):
        """Initialize the metrics store with the given database path."""
        # Ensure metrics directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.lock = threading.RLock()  # Use a reentrant lock for thread safety
        self.local = threading.local()  # Thread-local storage for connections
        
        # Create tables if they don't exist
        self._initialize_database()
        
        # Set up background collection thread
        self.collection_active = False
        self.collection_thread = None
    
    def _get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self.local, 'connection') or self.local.connection is None:
            self.local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        return self.local.connection
    
    def _initialize_database(self):
        connection = self._get_connection()
        with connection:
            cursor = connection.cursor()
            # Drop old tables if needed (for full rebuild)
            cursor.execute('DROP TABLE IF EXISTS move_metrics')
            cursor.execute('DROP TABLE IF EXISTS game_results')
            cursor.execute('DROP TABLE IF EXISTS config_settings')
            # Create new game_results table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                timestamp TEXT,
                winner TEXT,
                game_pgn TEXT,
                white_player TEXT,
                black_player TEXT,
                game_length INTEGER,
                created_at TEXT,
                white_engine_id TEXT,
                black_engine_id TEXT,
                white_engine_name TEXT,
                black_engine_name TEXT,
                white_engine_version TEXT,
                black_engine_version TEXT,
                exclude_white_from_metrics INTEGER,
                exclude_black_from_metrics INTEGER
            )''')
            # Create new move_metrics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS move_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                move_number INTEGER,
                player_color TEXT,
                move_uci TEXT,
                fen_before TEXT,
                evaluation REAL,
                engine_type TEXT,
                depth INTEGER,
                nodes_searched INTEGER,
                time_taken REAL,
                pv_line TEXT,
                created_at TEXT,
                engine_id TEXT,
                engine_name TEXT,
                engine_version TEXT,
                exclude_from_metrics INTEGER
            )''')            # Create new config_settings table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS config_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_id TEXT,
                timestamp TEXT,
                game_id TEXT,
                config_data TEXT,
                engine_id TEXT,
                engine_name TEXT,
                engine_version TEXT,
                white_engine_type TEXT,
                black_engine_type TEXT,
                white_depth INTEGER,
                black_depth INTEGER,
                created_at TEXT
            )''')
            connection.commit()            # Add missing columns to game_results table for direct config access
            try:
                cursor.execute('ALTER TABLE game_results ADD COLUMN white_engine_type TEXT')
                cursor.execute('ALTER TABLE game_results ADD COLUMN black_engine_type TEXT')
                cursor.execute('ALTER TABLE game_results ADD COLUMN white_depth INTEGER')
                cursor.execute('ALTER TABLE game_results ADD COLUMN black_depth INTEGER')
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    raise

            # Add missing columns to config_settings table
            try:
                cursor.execute('ALTER TABLE config_settings ADD COLUMN white_engine_type TEXT')
                cursor.execute('ALTER TABLE config_settings ADD COLUMN black_engine_type TEXT')
                cursor.execute('ALTER TABLE config_settings ADD COLUMN white_depth INTEGER')
                cursor.execute('ALTER TABLE config_settings ADD COLUMN black_depth INTEGER')
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    raise
    
    def start_collection(self, interval=60):
        """Start periodic data collection with the specified interval in seconds."""
        if self.collection_active:
            return  # Already running
            
        self.collection_active = True
        
        def collection_worker():
            while self.collection_active:
                try:
                    self.collect_all_data()
                except Exception as e:
                    print(f"Error during data collection: {e}")
                # Sleep for the specified interval
                for _ in range(interval):
                    if not self.collection_active:
                        break
                    time.sleep(1)
        
        self.collection_thread = threading.Thread(target=collection_worker, daemon=True)
        self.collection_thread.start()
    
    def stop_collection(self):
        """Stop the periodic data collection."""
        self.collection_active = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
    
    def collect_all_data(self):
        """Collect all available data from logs, games, and config files."""
        # The order here is important: configs, then games, then logs, then compute metrics
        # because game_results needs config data, and move_metrics/logs need game_ids.
        self.collect_config_data()
        self.collect_game_data()
        self.collect_log_data() # Re-enable when log parsing is implemented
        self.compute_metrics()
    
    def collect_log_data(self, log_dir="logging"):
        """Parse and store log data from the logging directory."""
        log_files = glob.glob(os.path.join(log_dir, "*.log"))
        for log_file in log_files:
            self._process_log_file(log_file)
    
    def _execute_with_retry(self, cursor, query, params=(), max_retries=5):
        """Execute a query with retry logic for handling locked database."""
        retries = 0
        while retries < max_retries:
            try:
                return cursor.execute(query, params)
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() or "database is locked" in str(e):
                    retries += 1
                    time.sleep(0.1)
                else:
                    raise
        raise sqlite3.OperationalError("Max retries reached for query execution.")
    
    def _process_log_file(self, log_file):
        """Process a single log file."""
        # Placeholder for actual implementation.
        # This should read the log file, parse lines using _parse_log_line,
        # and insert into the log_entries table.
        pass

    def _parse_log_line(self, line, log_file):
        """Parse a single line from a log file."""
        # Placeholder
        return None
    
    def collect_game_data(self, games_dir="games"):
        """Parse and store game data from PGN files, respecting exclude_from_metrics flag."""
        pgn_files = glob.glob(os.path.join(games_dir, "eval_game_*.pgn"))
        for pgn_file in pgn_files:
            self._process_pgn_file(pgn_file)
    
    def _process_pgn_file(self, pgn_file):
        """Process a single PGN file and store its data in the database."""
        try:
            game_id = os.path.basename(pgn_file)
            connection = self._get_connection()
            with connection:
                cursor = connection.cursor()
                self._execute_with_retry(cursor, "SELECT COUNT(*) FROM game_results WHERE game_id = ?", (game_id,))
                if cursor.fetchone()[0] > 0:
                    return # Already processed

            with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
                pgn_text = f.read()
                pgn_io = io.StringIO(pgn_text)
                game = chess.pgn.read_game(pgn_io)

                if game is None:
                    return

                headers = game.headers
                winner = headers.get("Result", "*")
                white_player = headers.get("White", "Unknown")
                black_player = headers.get("Black", "Unknown")
                  # Retrieve AI config data that should have been collected by collect_config_data
                # We need to link by game_id which is based on timestamp
                config_id = game_id.replace('.pgn', '.yaml')
                
                # Try to get config data, handle missing columns gracefully
                try:
                    cursor.execute("SELECT white_engine_type, black_engine_type, white_depth, black_depth FROM config_settings WHERE config_id = ?", (config_id,))
                    config_row = cursor.fetchone()
                except sqlite3.OperationalError:
                    # Columns don't exist, use default values
                    config_row = None

                if config_row:
                    white_engine_type = config_row[0] if config_row[0] else "unknown"
                    black_engine_type = config_row[1] if config_row[1] else "unknown"
                    white_depth = config_row[2] if config_row[2] else 0
                    black_depth = config_row[3] if config_row[3] else 0
                else:
                    # Extract engine info from PGN headers if config not available
                    white_player_header = white_player
                    black_player_header = black_player
                    
                    # Parse engine type from headers like "AI: viper via simple_search"
                    white_engine_type = "unknown"
                    black_engine_type = "unknown"
                    white_depth = 0
                    black_depth = 0
                    
                    if "via" in white_player_header:
                        parts = white_player_header.split("via")
                        if len(parts) > 1:
                            white_engine_type = parts[1].strip().rstrip(")")
                    
                    if "via" in black_player_header:
                        parts = black_player_header.split("via")
                        if len(parts) > 1:
                            black_engine_type = parts[1].strip().rstrip(")")

                game_length = 0
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    game_length += 1

                timestamp = None
                match = re.search(r'eval_game_(\d{8}_\d{6})\.pgn', game_id)
                if match:
                    timestamp = match.group(1)

                with connection:
                    cursor = connection.cursor()
                    self._execute_with_retry(cursor, '''
                    INSERT OR IGNORE INTO game_results
                    (game_id, timestamp, winner, game_pgn, white_player, black_player, game_length, white_engine_type, black_engine_type, white_depth, black_depth)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        game_id,
                        timestamp,
                        winner,
                        pgn_text,
                        white_player,
                        black_player,
                        game_length,
                        white_engine_type,
                        black_engine_type,
                        white_depth,
                        black_depth
                    ))
                    connection.commit()

        except Exception as e:
            print(f"Error processing PGN file {pgn_file}: {e}")

    def collect_config_data(self, games_dir="games"):
        """Parse and store configuration data from YAML files."""
        yaml_files = glob.glob(os.path.join(games_dir, "eval_game_*.yaml"))
        for yaml_file in yaml_files:
            self._process_config_file(yaml_file)
    
    def _process_config_file(self, yaml_file):
        """Process a single config file and store its data in the database."""
        try:
            # Create a config ID from the filename
            config_id = os.path.basename(yaml_file)
            
            # Check if this config is already in the database
            connection = self._get_connection()
            with connection:
                cursor = connection.cursor()
                self._execute_with_retry(cursor, "SELECT COUNT(*) FROM config_settings WHERE config_id = ?", (config_id,))
                if cursor.fetchone()[0] > 0:
                    return  # Already processed
              # Parse the YAML file
            with open(yaml_file, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # Extract key configuration details
            white_engine_config = config_data.get('white_engine_config', {})
            black_engine_config = config_data.get('black_engine_config', {})
            
            engine_id = hashlib.md5(white_engine_config.get('engine', 'unknown').encode()).hexdigest()
            engine_name = white_engine_config.get('engine', 'unknown')
            engine_version = white_engine_config.get('version', 'unknown')
            
            # Extract engine types and depths
            white_engine_type = white_engine_config.get('engine_type', 'unknown')
            black_engine_type = black_engine_config.get('engine_type', 'unknown')
            white_depth = white_engine_config.get('depth', 0)
            black_depth = black_engine_config.get('depth', 0)
            
            # Match with corresponding game
            game_id = config_id.replace('.yaml', '.pgn')
            
            # Extract timestamp from filename
            timestamp = None
            match = re.search(r'eval_game_(\d{8}_\d{6})\.yaml', config_id)
            if match:
                timestamp = match.group(1)
            
            # Store config data
            with connection:
                cursor = connection.cursor()
                self._execute_with_retry(cursor, '''
                INSERT OR IGNORE INTO config_settings
                (config_id, timestamp, game_id, config_data, engine_id, engine_name, engine_version,
                 white_engine_type, black_engine_type, white_depth, black_depth)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    config_id,
                    timestamp,
                    game_id,
                    json.dumps(config_data),
                    engine_id,
                    engine_name,
                    engine_version,
                    white_engine_type,
                    black_engine_type,
                    white_depth,
                    black_depth
                ))
                connection.commit()
        
        except Exception as e:
            print(f"Error processing config file {yaml_file}: {e}")
    
    def compute_metrics(self):
        """Compute derived metrics from the raw data and store them."""
        connection = self._get_connection()
        with connection:
            cursor = connection.cursor()
            
            # 1. Compute average metrics by function and side
            self._execute_with_retry(cursor, '''
            SELECT function_name, label, side, AVG(value) as avg_value, COUNT(*) as count,
                   MIN(value) as min_value, MAX(value) as max_value, MAX(timestamp) as latest_timestamp
            FROM log_entries
            WHERE value IS NOT NULL AND side IS NOT NULL
            GROUP BY function_name, label, side
            ''')
            
            metrics_to_store = []
            for row in cursor.fetchall():
                function_name, label, side, avg_value, count, min_value, max_value, latest_timestamp = row
                
                # Collect metrics for batch insert
                metrics_to_store.append((f"avg_{label}", avg_value, side, function_name, latest_timestamp))
                metrics_to_store.append((f"min_{label}", min_value, side, function_name, latest_timestamp))
                metrics_to_store.append((f"max_{label}", max_value, side, function_name, latest_timestamp))
                metrics_to_store.append((f"count_{label}", count, side, function_name, latest_timestamp))
            
            # Batch insert all metrics
            for metric in metrics_to_store:
                metric_name, metric_value, side, function_name, timestamp = metric
            
                def _store_metric(metric_name, metric_value, side, function_name, timestamp):
                    """Store a computed metric in the database."""
                    connection = self._get_connection()
                    with connection:
                        cursor = connection.cursor()
                        try:
                            self._execute_with_retry(cursor, '''
                            INSERT OR IGNORE INTO metrics
                            (metric_name, metric_value, side, function_name, timestamp)
                            VALUES (?, ?, ?, ?, ?)
                            ''', (
                                metric_name,                                metric_value,
                                side,
                                function_name,
                                timestamp
                            ))
                            connection.commit()
                        except sqlite3.Error as e:
                            print(f"Error storing metric {metric_name}: {e}")
                _store_metric(metric_name, metric_value, side, function_name, timestamp)

    def add_game_result(self, game_id: str, timestamp: str, winner: str, game_pgn: str,
                        white_player: str, black_player: str, game_length: int,
                        white_engine_config: dict, black_engine_config: dict):
        """
        Inserts a single game's result and associated AI configurations into the game_results table.
        The columns white_engine_name and black_engine_name are set from
        white_engine_config['engine'] and black_engine_config['engine'] respectively.
        These values are the unique engine identifiers as defined in chess_game_config.yaml.
        """
        connection = self._get_connection()
        with connection:
            cursor = connection.cursor()
            try:
                # Extract AI types and depths from the provided configs
                white_engine_id = hashlib.md5(white_engine_config.get('engine', 'unknown').encode()).hexdigest()
                black_engine_id = hashlib.md5(black_engine_config.get('engine', 'unknown').encode()).hexdigest()
                white_engine_name = white_engine_config.get('engine', 'unknown')
                black_engine_name = black_engine_config.get('engine', 'unknown')
                white_engine_version = white_engine_config.get('version', 'unknown')
                black_engine_version = black_engine_config.get('version', 'unknown')
                exclude_white_from_metrics = int(white_engine_config.get('exclude_from_metrics', False))
                exclude_black_from_metrics = int(black_engine_config.get('exclude_from_metrics', False))
                created_at = datetime.now().isoformat()

                self._execute_with_retry(cursor, '''
                INSERT OR IGNORE INTO game_results
                (game_id, timestamp, winner, game_pgn, white_player, black_player, game_length,
                 created_at, white_engine_id, black_engine_id, white_engine_name, black_engine_name, 
                 white_engine_version, black_engine_version, exclude_white_from_metrics, exclude_black_from_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    game_id, timestamp, winner, game_pgn, white_player, black_player, game_length,
                    created_at, white_engine_id, black_engine_id, white_engine_name, black_engine_name, 
                    white_engine_version, black_engine_version, exclude_white_from_metrics, exclude_black_from_metrics
                ))
                connection.commit()
            except sqlite3.Error as e:
                print(f"Error adding game result for game {game_id}: {e}")


    def add_move_metric(self, game_id: str, move_number: int, player_color: str,
                        move_uci: str, fen_before: str, evaluation: float,
                        engine_type: str, depth: int, nodes_searched: int,
                        time_taken: float, pv_line: str):
        """
        Inserts a single move's detailed metrics into the move_metrics table.
        The engine_type field here refers to the search algorithm used for the move (e.g., 'deepsearch'),
        not the engine name. The engine name for the move is determined by the parent game's
        white_engine_name or black_engine_name in game_results, as set from the config's 'engine' key.
        """
        # Normalize player_color
        if player_color.lower() in ('w', 'white'):
            player_color_db = 'white'
        elif player_color.lower() in ('b', 'black'):
            player_color_db = 'black'
        else:
            player_color_db = player_color  # fallback, but should not happen

        connection = self._get_connection()
        with connection:
            cursor = connection.cursor()
            try:
                self._execute_with_retry(cursor, '''
                INSERT OR IGNORE INTO move_metrics
                (game_id, move_number, player_color, move_uci, fen_before,
                 evaluation, engine_type, depth, nodes_searched, time_taken, pv_line)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    game_id, move_number, player_color_db, move_uci, fen_before,
                    evaluation, engine_type, depth, nodes_searched, time_taken, pv_line
                ))
                connection.commit()
            except sqlite3.Error as e:
                print(f"Error adding move metric for game {game_id}, move {move_number}: {e}")

    def get_game_statistics(self):
        """
        Retrieve game statistics such as total games, wins, losses, and draws.
        """
        connection = self._get_connection()
        with connection:
            cursor = connection.cursor()
            
            # Retrieve game counts
            cursor.execute("SELECT COUNT(*) FROM game_results")
            total_games = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM game_results WHERE winner = '1-0'")
            white_wins = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM game_results WHERE winner = '0-1'")
            black_wins = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM game_results WHERE winner = '1/2-1/2'")
            draws = cursor.fetchone()[0]

            return {
                "total_games": total_games,
                "white_wins": white_wins,
                "black_wins": black_wins,
                "draws": draws,
            }
            
    def get_all_game_results_df(self):
        """
        Retrieves all game results from the database as a Pandas DataFrame.
        """
        connection = self._get_connection()
        with connection:
            df = pd.read_sql_query("SELECT * FROM game_results", connection)
        return df

    def get_distinct_move_metric_names(self):
        """
        Retrieves distinct column names from move_metrics that are suitable for plotting.
        Excludes primary keys, foreign keys, and text fields.
        """
        connection = self._get_connection()
        with connection:
            cursor = connection.cursor()
            # Query PRAGMA table_info to get column details
            cursor.execute("PRAGMA table_info(move_metrics)")
            columns_info = cursor.fetchall()
            
            plot_eligible_types = ['REAL', 'INTEGER'] # Numeric types
            exclude_names = ['id', 'game_id', 'move_number', 'player_color', 'move_uci', 'fen_before', 'engine_type', 'pv_line', 'created_at']
            
            metric_names = []
            for col in columns_info:
                col_name = col[1] # Column name is at index 1
                col_type = col[2].upper() # Column type is at index 2
                
                if col_name not in exclude_names and col_type in plot_eligible_types:
                    metric_names.append(col_name)
        return sorted(metric_names)

    def get_filtered_move_metrics(self, white_engine_names: Optional[list] = None, black_engine_names: Optional[list] = None, metric_name: Optional[str] = None, white_search_algorithms: Optional[list] = None, black_search_algorithms: Optional[list] = None):
        """
        Retrieves move metrics filtered by engine name (from config's 'engine' key, stored as white_engine_name/black_engine_name in game_results).
        Optionally, can filter by search algorithm (engine_type in move_metrics) for deep-dive analysis.
        Example: To get all moves by the 'v7p3r_nn' engine, set white_engine_names=['v7p3r_nn'] or black_engine_names=['v7p3r_nn'].
        """
        connection = self._get_connection()
        
        # Build the WHERE clause dynamically
        where_clauses = []
        params = []

        if white_engine_names:
            placeholders = ','.join(['?'] * len(white_engine_names))
            where_clauses.append(f"gr.white_engine_name IN ({placeholders})")
            params.extend(white_engine_names)
        if black_engine_names:
            placeholders = ','.join(['?'] * len(black_engine_names))
            where_clauses.append(f"gr.black_engine_name IN ({placeholders})")
            params.extend(black_engine_names)
        # Optional: secondary filter by search algorithm (engine_type in move_metrics)
        if white_search_algorithms:
            placeholders = ','.join(['?'] * len(white_search_algorithms))
            where_clauses.append(f"(mm.player_color = 'white' AND mm.engine_type IN ({placeholders}))")
            params.extend(white_search_algorithms)
        if black_search_algorithms:
            placeholders = ','.join(['?'] * len(black_search_algorithms))
            where_clauses.append(f"(mm.player_color = 'black' AND mm.engine_type IN ({placeholders}))")
            params.extend(black_search_algorithms)

        select_columns = "mm.game_id, mm.move_number, mm.player_color, mm.move_uci, mm.fen_before, mm.created_at, mm.evaluation, mm.nodes_searched, mm.time_taken, mm.depth, mm.pv_line"
        if metric_name:
            if metric_name in self.get_distinct_move_metric_names():
                select_columns += f", mm.{metric_name}"
            else:
                print(f"Warning: Attempted to query invalid or non-numeric metric_name: {metric_name}")
                return []
        query = f"""
        SELECT {select_columns}, gr.white_engine_name, gr.black_engine_name, gr.white_depth, gr.black_depth
        FROM move_metrics mm
        JOIN game_results gr ON mm.game_id = gr.game_id
        """
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        query += " ORDER BY mm.created_at"
        results = []
        try:
            with connection:
                cursor = connection.cursor()
                cursor.execute(query, params)
                cols = [description[0] for description in cursor.description]
                for row in cursor.fetchall():
                    results.append(dict(zip(cols, row)))
        except sqlite3.Error as e:
            print(f"Error querying filtered move metrics: {e}")
            return []
        return results


    def get_side_performance_metrics(self, side):
        """
        Retrieve performance metrics for a specific side ('w' for white, 'b' for black).
        Accepts both 'w'/'b' and 'white'/'black' for compatibility.
        """
        # Normalize side
        if side.lower() in ('w', 'white'):
            side_db = 'white'
        elif side.lower() in ('b', 'black'):
            side_db = 'black'
        else:
            raise ValueError("Invalid side. Use 'w'/'white' for white or 'b'/'black' for black.")

        connection = self._get_connection()
        metrics = []
        with connection:
            cursor = connection.cursor()
            cursor.execute('''
            SELECT AVG(evaluation) AS avg_eval, COUNT(evaluation) AS count_eval,
                   AVG(nodes_searched) AS avg_nodes, COUNT(nodes_searched) AS count_nodes,
                   AVG(time_taken) AS avg_time, COUNT(time_taken) AS count_time
            FROM move_metrics
            WHERE player_color = ?
            ''', (side_db,))
            row = cursor.fetchone()
            if row:
                avg_eval, count_eval, avg_nodes, count_nodes, avg_time, count_time = row
                if avg_eval is not None:
                    metrics.append({'label': 'Average Evaluation', 'avg_value': avg_eval, 'count': count_eval})
                if avg_nodes is not None:
                    metrics.append({'label': 'Average Nodes Searched', 'avg_value': avg_nodes, 'count': count_nodes})
                if avg_time is not None:
                    metrics.append({'label': 'Average Time Taken (s)', 'avg_value': avg_time, 'count': count_time})
        return metrics
    
    def get_errors_and_warnings(self, limit=100):
        """
        Retrieve recent error and warning log entries from the log_entries table.
        Returns a list of dicts with timestamp, function_name, message, and log_file.
        """
        connection = self._get_connection()
        results = []
        with connection:
            cursor = connection.cursor()
            self._execute_with_retry(
                cursor,
                '''
                SELECT timestamp, function_name, message, log_file, raw_text
                FROM log_entries
                WHERE message LIKE '%error%' OR message LIKE '%warning%' OR message LIKE '%exception%'
                ORDER BY id DESC
                LIMIT ?
                ''',
                (limit,)
            )
            for row in cursor.fetchall():
                results.append({
                    "timestamp": row[0],
                    "function_name": row[1],
                    "message": row[2],
                    "log_file": row[3],
                    "raw": row[4] # Include raw text for full message
                })
        return results
    
    def get_metrics_trend_data(self, metric_name, side=None, limit=100):
        """Retrieve trend data for a specific metric from the 'metrics' table."""
        # This is for the aggregated 'metrics' table, not 'move_metrics'
        connection = self._get_connection()
        query = '''
        SELECT timestamp, metric_value
        FROM metrics
        WHERE metric_name = ?
        '''
        params = [metric_name]

        if side:
            query += ' AND side = ?'
            params.append(side)

        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)

        with connection:
            cursor = connection.cursor()
            self._execute_with_retry(cursor, query, params)
            return [
                {"timestamp": row[0], "metric_value": row[1]}
                for row in cursor.fetchall()
            ]

    def close(self):
        """Close the database connection."""
        if hasattr(self.local, 'connection') and self.local.connection:
            self.local.connection.close()
            self.local.connection = None

    def _get_engine_config(self, color):
        """Retrieve AI configuration for the given color. (Placeholder for real config loading)"""
        return {}

    def _hash_engine_config(self, config_dict):
        # Create a unique hash for an engine config dict
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()

    def get_engine_id_from_config(self, config_dict):
        # Helper to get a unique engine_id for a config dict
        return self._hash_engine_config(config_dict)

    def rebuild_metrics_from_files(self, games_dir="games"):
        """
        Rebuild the metrics database from all available PGN, YAML, and log files.
        This will parse each game's config, PGN, and log, and repopulate the metrics tables.
        """
        connection = self._get_connection()
        with connection:
            cursor = connection.cursor()
            # Clear all tables
            cursor.execute('DELETE FROM move_metrics')
            cursor.execute('DELETE FROM game_results')
            cursor.execute('DELETE FROM config_settings')
            connection.commit()

        # Find all games (assume .pgn, .yaml, .log triplets)
        pgn_files = glob.glob(os.path.join(games_dir, '*.pgn'))
        for pgn_file in pgn_files:
            base = os.path.splitext(pgn_file)[0]
            yaml_file = base + '.yaml'
            log_file = base + '.log'
            # Parse YAML config
            if not os.path.exists(yaml_file):
                continue
            with open(yaml_file, 'r') as f:
                config_data = yaml.safe_load(f)
            # Extract engine configs and exclusion flags
            white_cfg = config_data.get('white_engine_config', {})
            black_cfg = config_data.get('black_engine_config', {})
            white_engine = white_cfg.get('engine', 'unknown')
            black_engine = black_cfg.get('engine', 'unknown')
            white_exclude = bool(white_cfg.get('exclude_from_metrics', False))
            black_exclude = bool(black_cfg.get('exclude_from_metrics', False))
            # Use config hash as engine_id
            white_engine_id = self._hash_engine_config(white_cfg)
            black_engine_id = self._hash_engine_config(black_cfg)
            white_engine_version = white_cfg.get('version', '')
            black_engine_version = black_cfg.get('version', '')
            # Store config_settings
            for color, cfg, eid, ename, ever in [
                ('white', white_cfg, white_engine_id, white_engine, white_engine_version),
                ('black', black_cfg, black_engine_id, black_engine, black_engine_version)
            ]:
                cursor.execute('''INSERT INTO config_settings (config_id, timestamp, game_id, config_data, engine_id, engine_name, engine_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
                    f"{base}_{color}", datetime.now().isoformat(), base, json.dumps(cfg), eid, ename, ever, datetime.now().isoformat()
                ))
            # Parse PGN for game result
            with open(pgn_file, 'r') as f:
                pgn_text = f.read()
            # Extract winner, moves, etc. (simplified)
            winner = None
            for line in pgn_text.splitlines():
                if line.startswith('[Result '):
                    winner = line.split('"')[1]
                    break
            # Insert game_results
            cursor.execute('''INSERT INTO game_results (game_id, timestamp, winner, game_pgn, white_player, black_player, game_length, created_at, white_engine_id, black_engine_id, white_engine_name, black_engine_name, white_engine_version, black_engine_version, exclude_white_from_metrics, exclude_black_from_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                base, datetime.now().isoformat(), winner, pgn_text, white_engine, black_engine, 0, datetime.now().isoformat(),
                white_engine_id, black_engine_id, white_engine, black_engine, white_engine_version, black_engine_version, int(white_exclude), int(black_exclude)
            ))
            # --- Begin move_metrics ingestion ---
            pgn_io = io.StringIO(pgn_text)
            game = chess.pgn.read_game(pgn_io)
            if game is not None:
                board = game.board()
                move_number = 1
                for node in game.mainline():
                    move = node.move
                    fen_before = board.fen()
                    move_uci = move.uci()
                    player_color = 'white' if board.turn else 'black'
                    # Try to extract additional info from comments or logs (if available)
                    # For now, just basic info
                    cursor.execute('''INSERT INTO move_metrics (game_id, move_number, player_color, move_uci, fen_before, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)''', (
                        base,
                        move_number,
                        player_color,
                        move_uci,
                        fen_before,
                        datetime.now().isoformat()
                    ))
                    board.push(move)
                    if not board.turn:
                        move_number += 1
            # --- End move_metrics ingestion ---
            connection.commit()
    
    def migrate_player_color_normalization(self):
        """
        Update all move_metrics.player_color values to 'white' or 'black' if they are 'w' or 'b'.
        """
        connection = self._get_connection()
        with connection:
            cursor = connection.cursor()
            # Update 'w' to 'white'
            cursor.execute("UPDATE move_metrics SET player_color = 'white' WHERE player_color = 'w'")
            # Update 'b' to 'black'
            cursor.execute("UPDATE move_metrics SET player_color = 'black' WHERE player_color = 'b'")
            connection.commit()
            # Check for any non-normalized values
            cursor.execute("SELECT DISTINCT player_color FROM move_metrics")
            all_colors = [row[0] for row in cursor.fetchall()]
            for color in all_colors:
                if color not in ('white', 'black'):
                    print(f"Warning: Found non-normalized player_color value in DB: {color}")

    def reingest_all_games(self, games_dir="games"):
        """
        Rebuild the metrics database from all available PGN and YAML files.
        """
        print("Re-ingesting all games and configs from disk...")
        self.rebuild_metrics_from_files(games_dir=games_dir)
        self.migrate_player_color_normalization()
        print("Re-ingestion and normalization complete.")

    def test_player_color_normalization(self):
        """
        Test that all move_metrics.player_color values are normalized.
        """
        connection = self._get_connection()
        with connection:
            cursor = connection.cursor()
            cursor.execute("SELECT DISTINCT player_color FROM move_metrics")
            all_colors = [row[0] for row in cursor.fetchall()]
            print(f"Unique player_color values in move_metrics: {all_colors}")
            assert set(all_colors).issubset({'white', 'black'}), "Non-normalized player_color values found!"

# Test code (if `if __name__ == "__main__":` block exists, add this inside it)
if __name__ == "__main__":
    # Test `MetricsStore` enhancements
    print("\n--- MetricsStore Dashboard Integration Test ---")
    
    # Initialize a new MetricsStore (or existing one)
    db_test_path = "metrics/test_dashboard_metrics.db"
    if os.path.exists(db_test_path):
        os.remove(db_test_path) # Start fresh for testing
    store = MetricsStore(db_path=db_test_path)
    
    # Simulate a game result with full AI configs
    test_game_id_1 = f"eval_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}_1.pgn"
    test_timestamp_1 = datetime.now().strftime('%Y%m%d_%H%M%S')
    white_config_1 = {'engine_type': 'deepsearch', 'depth': 5, 'engine': 'Viper', 'exclude_from_metrics': False}
    black_config_1 = {'engine_type': 'negamax', 'depth': 4, 'engine': 'Viper', 'exclude_from_metrics': False}
    
    store.add_game_result(
        game_id=test_game_id_1, timestamp=test_timestamp_1, winner='1-0', game_pgn="...",
        white_player='Viper Deep', black_player='Viper Negamax', game_length=42,
        white_engine_config=white_config_1, black_engine_config=black_config_1
    )

    # Simulate moves for game 1
    store.add_move_metric(game_id=test_game_id_1, move_number=1, player_color='w', move_uci='e2e4', fen_before='...', evaluation=0.2, engine_type='deepsearch', depth=3, nodes_searched=1500, time_taken=0.15, pv_line='e2e4 c7c5 g1f3')
    store.add_move_metric(game_id=test_game_id_1, move_number=1, player_color='b', move_uci='c7c5', fen_before='...', evaluation=-0.1, engine_type='negamax', depth=2, nodes_searched=800, time_taken=0.08, pv_line='c7c5 g1f3 d7d6')
    store.add_move_metric(game_id=test_game_id_1, move_number=2, player_color='w', move_uci='g1f3', fen_before='...', evaluation=0.3, engine_type='deepsearch', depth=4, nodes_searched=2500, time_taken=0.25, pv_line='g1f3 d7d6 d2d4')

    time.sleep(1) # Simulate time passing for next game
    test_game_id_2 = f"eval_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}_2.pgn"
    test_timestamp_2 = datetime.now().strftime('%Y%m%d_%H%M%S')
    white_config_2 = {'engine_type': 'stockfish', 'depth': 0, 'engine': 'Stockfish', 'exclude_from_metrics': False}
    black_config_2 = {'engine_type': 'deepsearch', 'depth': 3, 'engine': 'Viper', 'exclude_from_metrics': False}

    store.add_game_result(
        game_id=test_game_id_2, timestamp=test_timestamp_2, winner='0-1', game_pgn="...",
        white_player='Stockfish', black_player='Viper Deep', game_length=30,
        white_engine_config=white_config_2, black_engine_config=black_config_2
    )
    # Simulate moves for game 2
    store.add_move_metric(game_id=test_game_id_2, move_number=1, player_color='w', move_uci='d2d4', fen_before='...', evaluation=0.1, engine_type='stockfish', depth=18, nodes_searched=100000, time_taken=0.5, pv_line='d2d4 g8f6 c2c4')
    store.add_move_metric(game_id=test_game_id_2, move_number=1, player_color='b', move_uci='g8f6', fen_before='...', evaluation=-0.3, engine_type='deepsearch', depth=3, nodes_searched=2000, time_taken=0.2, pv_line='g8f6 c2c4 e7e6')


    print("\nTesting get_distinct_move_metric_names:")
    metric_names = store.get_distinct_move_metric_names()
    print(f"Distinct move metric names: {metric_names}")
    assert 'evaluation' in metric_names and 'nodes_searched' in metric_names, "Expected metric names not found"

    print("\nTesting get_filtered_move_metrics (White: deepsearch, Black: negamax, Metric: evaluation):")
    filtered_moves = store.get_filtered_move_metrics(
        white_engine_names=['Viper'],
        black_engine_names=['Viper'],
        white_search_algorithms=['deepsearch'],
        black_search_algorithms=['negamax'],
        metric_name='evaluation'
    )
    print(f"Filtered moves count: {len(filtered_moves)}")
    for move in filtered_moves:
        print(f"  Game: {move['game_id']}, Move: {move['move_uci']}, Eval: {move['evaluation']:.3f}")
    assert len(filtered_moves) == 3, "Expected 3 moves for deepsearch vs negamax"

    print("\nTesting get_filtered_move_metrics (White: Stockfish, Black: deepsearch, Metric: nodes_searched):")
    filtered_moves_sf = store.get_filtered_move_metrics(
        white_engine_names=['Stockfish'],
        black_engine_names=['Viper'],
        metric_name='nodes_searched'
    )
    print(f"Filtered moves count: {len(filtered_moves_sf)}")
    for move in filtered_moves_sf:
        print(f"  Game: {move['game_id']}, Move: {move['move_uci']}, Nodes: {move['nodes_searched']}")
    assert len(filtered_moves_sf) == 2, "Expected 2 moves for Stockfish vs deepsearch"


    print("\nTesting get_all_game_results_df:")
    df_all_games = store.get_all_game_results_df()
    print(df_all_games.head())
    assert not df_all_games.empty, "DataFrame of all game results should not be empty"

    store.close()
    print("--- MetricsStore Dashboard Integration Tests Complete ---")
    # --- Additional migration and test for legacy data ---
    print("\n--- Running migration and normalization for legacy data ---")
    store = MetricsStore()
    store.reingest_all_games(games_dir="games")
    store.test_player_color_normalization()
    print("--- Migration and normalization test complete ---")