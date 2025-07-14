# metrics.py

"""Chess Engine Metrics System
A simple metrics database implementation for game results and move score and move decision logic records.
Serves to be light weight and flexible as new metrics data is needed for engine performance analysis.
Builds compiled reports of to date metrics on the engine for analysis and enhancements.
"""

import sqlite3
import time
import json
from datetime import datetime

class ChessMetrics:
    def __init__(self, db_path="engine_metrics.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the metrics database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Game results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS game_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    white_player TEXT NOT NULL,
                    black_player TEXT NOT NULL,
                    result TEXT NOT NULL,
                    total_moves INTEGER,
                    game_duration REAL,
                    white_engine_config TEXT,
                    black_engine_config TEXT,
                    pgn TEXT
                )
            ''')
            
            # Move analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS move_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER,
                    move_number INTEGER,
                    player_color TEXT,
                    move_uci TEXT,
                    evaluation_score REAL,
                    search_depth INTEGER,
                    nodes_searched INTEGER,
                    search_time REAL,
                    book_move BOOLEAN,
                    evaluation_details TEXT,
                    FOREIGN KEY (game_id) REFERENCES game_results (id)
                )
            ''')
            
            # Engine performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS engine_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    engine_name TEXT NOT NULL,
                    opponent_name TEXT NOT NULL,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    draws INTEGER DEFAULT 0,
                    total_games INTEGER DEFAULT 0,
                    avg_move_time REAL,
                    avg_nodes_per_second REAL
                )
            ''')
            
            conn.commit()
    
    def record_game_start(self, white_player, black_player, white_config=None, black_config=None):
        """Record the start of a new game"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO game_results 
                (timestamp, white_player, black_player, result, total_moves, game_duration, white_engine_config, black_engine_config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, white_player, black_player, "In Progress", 0, 0.0,
                json.dumps(white_config) if white_config else None,
                json.dumps(black_config) if black_config else None
            ))
            
            return cursor.lastrowid
    
    def record_move(self, game_id, move_number, player_color, move_uci, 
                   evaluation_score=None, search_depth=None, nodes_searched=None, 
                   search_time=None, book_move=False, evaluation_details=None):
        """Record a move and its analysis"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO move_analysis 
                (game_id, move_number, player_color, move_uci, evaluation_score, 
                 search_depth, nodes_searched, search_time, book_move, evaluation_details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                game_id, move_number, player_color, move_uci, evaluation_score,
                search_depth, nodes_searched, search_time, book_move,
                json.dumps(evaluation_details) if evaluation_details else None
            ))
    
    def record_game_end(self, game_id, result, total_moves, game_duration, pgn=None):
        """Record the end of a game"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE game_results 
                SET result = ?, total_moves = ?, game_duration = ?, pgn = ?
                WHERE id = ?
            ''', (result, total_moves, game_duration, pgn, game_id))
    
    def update_engine_performance(self, engine_name, opponent_name, result):
        """Update engine performance statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if record exists
            cursor.execute('''
                SELECT id, wins, losses, draws, total_games 
                FROM engine_performance 
                WHERE engine_name = ? AND opponent_name = ?
            ''', (engine_name, opponent_name))
            
            row = cursor.fetchone()
            
            if row:
                # Update existing record
                record_id, wins, losses, draws, total_games = row
                
                if result == "1-0" and engine_name == "white":
                    wins += 1
                elif result == "0-1" and engine_name == "black":
                    wins += 1
                elif result == "1-0" and engine_name == "black":
                    losses += 1
                elif result == "0-1" and engine_name == "white":
                    losses += 1
                elif result == "1/2-1/2":
                    draws += 1
                
                total_games += 1
                
                cursor.execute('''
                    UPDATE engine_performance 
                    SET wins = ?, losses = ?, draws = ?, total_games = ?, timestamp = ?
                    WHERE id = ?
                ''', (wins, losses, draws, total_games, datetime.now().isoformat(), record_id))
                
            else:
                # Create new record
                wins = 1 if ((result == "1-0" and engine_name == "white") or 
                           (result == "0-1" and engine_name == "black")) else 0
                losses = 1 if ((result == "1-0" and engine_name == "black") or 
                             (result == "0-1" and engine_name == "white")) else 0
                draws = 1 if result == "1/2-1/2" else 0
                
                cursor.execute('''
                    INSERT INTO engine_performance 
                    (timestamp, engine_name, opponent_name, wins, losses, draws, total_games)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (datetime.now().isoformat(), engine_name, opponent_name, wins, losses, draws, 1))
    
    def get_engine_stats(self, engine_name, opponent_name=None):
        """Get engine performance statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if opponent_name:
                cursor.execute('''
                    SELECT wins, losses, draws, total_games, avg_move_time, avg_nodes_per_second
                    FROM engine_performance 
                    WHERE engine_name = ? AND opponent_name = ?
                ''', (engine_name, opponent_name))
            else:
                cursor.execute('''
                    SELECT SUM(wins), SUM(losses), SUM(draws), SUM(total_games)
                    FROM engine_performance 
                    WHERE engine_name = ?
                ''', (engine_name,))
            
            row = cursor.fetchone()
            if row:
                wins, losses, draws, total = row[:4]
                wins = wins or 0
                losses = losses or 0
                draws = draws or 0
                total = total or 0
                
                win_rate = (wins / total * 100) if total > 0 else 0
                
                return {
                    'wins': wins,
                    'losses': losses,
                    'draws': draws,
                    'total_games': total,
                    'win_rate': win_rate
                }
            
            return {'wins': 0, 'losses': 0, 'draws': 0, 'total_games': 0, 'win_rate': 0}
    
    def get_recent_games(self, limit=10, engine_name=None):
        """Get recent game results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if engine_name:
                cursor.execute('''
                    SELECT timestamp, white_player, black_player, result, total_moves, game_duration
                    FROM game_results 
                    WHERE white_player = ? OR black_player = ?
                    ORDER BY timestamp DESC LIMIT ?
                ''', (engine_name, engine_name, limit))
            else:
                cursor.execute('''
                    SELECT timestamp, white_player, black_player, result, total_moves, game_duration
                    FROM game_results 
                    ORDER BY timestamp DESC LIMIT ?
                ''', (limit,))
            
            return cursor.fetchall()
    
    def get_move_time_stats(self, engine_name):
        """Get move time statistics for an engine"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT AVG(search_time), MAX(search_time), MIN(search_time), COUNT(*)
                FROM move_analysis ma
                JOIN game_results gr ON ma.game_id = gr.id
                WHERE (gr.white_player = ? AND ma.player_color = 'white') 
                   OR (gr.black_player = ? AND ma.player_color = 'black')
                AND ma.search_time IS NOT NULL
            ''', (engine_name, engine_name))
            
            row = cursor.fetchone()
            if row and row[0]:
                return {
                    'avg_time': row[0],
                    'max_time': row[1],
                    'min_time': row[2],
                    'total_moves': row[3]
                }
            
            return None