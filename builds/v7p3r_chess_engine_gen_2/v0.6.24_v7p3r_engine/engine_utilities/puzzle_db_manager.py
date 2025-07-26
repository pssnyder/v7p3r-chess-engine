import os
import sqlite3
import csv
import json
from datetime import datetime

class PuzzleDBManager:
    def __init__(self, db_path="puzzles/puzzle_data.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        c = self.conn.cursor()
        # Puzzle table
        c.execute('''CREATE TABLE IF NOT EXISTS puzzles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            puzzle_id TEXT UNIQUE,
            fen TEXT,
            moves TEXT,
            rating INTEGER,
            rating_deviation INTEGER,
            popularity INTEGER,
            nb_plays INTEGER,
            themes TEXT,
            game_url TEXT,
            opening_tags TEXT
        )''')
        # Attempt table
        c.execute('''CREATE TABLE IF NOT EXISTS puzzle_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            puzzle_id TEXT,
            engine_config TEXT,
            attempt_moves TEXT,
            correct INTEGER,
            timestamp DATETIME,
            attempts_count INTEGER,
            result_elo INTEGER
        )''')
        # Transposition table
        c.execute('''CREATE TABLE IF NOT EXISTS transpositions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fen TEXT UNIQUE,
            best_move TEXT,
            score REAL,
            source TEXT,
            last_updated DATETIME
        )''')
        # Anti-transposition table
        c.execute('''CREATE TABLE IF NOT EXISTS anti_transpositions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fen TEXT UNIQUE,
            bad_move TEXT,
            score REAL,
            source TEXT,
            last_updated DATETIME
        )''')
        self.conn.commit()

    def ingest_puzzles_from_csv(self, csv_path):
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    self.conn.execute('''INSERT OR IGNORE INTO puzzles (
                        puzzle_id, fen, moves, rating, rating_deviation, popularity, nb_plays, themes, game_url, opening_tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                        row['PuzzleId'], row['FEN'], row['Moves'], int(row['Rating']), int(row['RatingDeviation']),
                        int(row['Popularity']), int(row['NbPlays']), row['Themes'], row['GameUrl'], row['OpeningTags']
                    ))
                except Exception as e:
                    print(f"Error ingesting puzzle {row.get('PuzzleId')}: {e}")
            self.conn.commit()

    def get_puzzles_by_elo(self, min_elo, max_elo, limit=50):
        c = self.conn.cursor()
        c.execute('''SELECT * FROM puzzles WHERE rating BETWEEN ? AND ? ORDER BY rating LIMIT ?''', (min_elo, max_elo, limit))
        return c.fetchall()

    def record_attempt(self, puzzle_id, engine_config, attempt_moves, correct, attempts_count, result_elo):
        self.conn.execute('''INSERT INTO puzzle_attempts (
            puzzle_id, engine_config, attempt_moves, correct, timestamp, attempts_count, result_elo
        ) VALUES (?, ?, ?, ?, ?, ?, ?)''', (
            puzzle_id, json.dumps(engine_config), ','.join(attempt_moves), int(correct), datetime.now(), attempts_count, result_elo
        ))
        self.conn.commit()

    def add_transposition(self, fen, best_move, score, source):
        self.conn.execute('''INSERT OR REPLACE INTO transpositions (
            fen, best_move, score, source, last_updated
        ) VALUES (?, ?, ?, ?, ?)''', (
            fen, best_move, score, source, datetime.now()
        ))
        self.conn.commit()

    def add_anti_transposition(self, fen, bad_move, score, source):
        self.conn.execute('''INSERT OR REPLACE INTO anti_transpositions (
            fen, bad_move, score, source, last_updated
        ) VALUES (?, ?, ?, ?, ?)''', (
            fen, bad_move, score, source, datetime.now()
        ))
        self.conn.commit()

    def close(self):
        self.conn.close()
