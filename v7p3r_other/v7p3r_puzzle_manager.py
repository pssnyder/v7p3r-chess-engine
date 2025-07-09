#!/usr/bin/env python3
"""
Puzzle Database Manager for the v7p3r Chess Engine

This module provides functionality to:
1. Import and manage Lichess puzzle data in a SQLite database
2. Query puzzles based on various criteria (ELO, themes, etc.)
3. Generate test sets for engine evaluation
4. Provide a console-based UI for interactive operations

The database uses a normalized schema with only a puzzles table.
"""

import os
import sqlite3
import csv
import json
from datetime import datetime
from v7p3r_config import v7p3rConfig

# Define ELO difficulty levels
ELO_DIFFICULTY_LEVELS = {
    "Beginner": (800, 1200),
    "Intermediate": (1201, 1600),
    "Advanced": (1601, 2000),
    "Expert": (2001, 2400),
    "Master": (2401, 3000)
}


class PuzzleDBManager:
    """
    Manages a database of chess puzzles for the v7p3r engine, providing
    import, query, and test set generation functionality.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the puzzle database manager.
        
        Args:
            config_manager (v7p3rConfig): Configuration manager instance. If None, 
                                        creates a new one with default config.
        """
        # Load configuration
        if config_manager is None:
            self.config_manager = v7p3rConfig()
        else:
            self.config_manager = config_manager
        self.config = self.config_manager.get_puzzle_config()
        
        # Set up paths
        db_dir = os.path.dirname(self.config['puzzle_database']['db_path'])
        os.makedirs(db_dir, exist_ok=True)
        
        # Connect to database
        self.conn = None
        try:
            self.conn = sqlite3.connect(self.config['puzzle_database']['db_path'])
            self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise
        
        # Create tables if they don't exist
        self._create_tables()
        
    def _create_tables(self):
        """Create the database tables if they don't exist."""
        if not self.conn:
            return
            
        cursor = self.conn.cursor()
        
        # Create puzzles table with ELO difficulty category
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS puzzles (
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
            opening_tags TEXT,
            difficulty_level TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create indexes for faster querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_puzzles_rating ON puzzles(rating)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_puzzles_themes ON puzzles(themes)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_puzzles_difficulty ON puzzles(difficulty_level)')
        
        self.conn.commit()
    
    def ingest_puzzles_from_csv(self, csv_path):
        """
        Import puzzles from a Lichess puzzle CSV file.
        
        Args:
            csv_path (str): Path to the CSV file.
            
        Returns:
            int: Number of puzzles imported.
        """
        if not self.conn:
            return 0
            
        imported_count = 0
        skipped_count = 0
        
        try:
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    try:
                        # Calculate difficulty level based on rating
                        difficulty_level = self._get_difficulty_level(int(row['Rating']))
                        
                        # Insert the puzzle
                        self.conn.execute('''
                        INSERT OR IGNORE INTO puzzles (
                            puzzle_id, fen, moves, rating, rating_deviation, 
                            popularity, nb_plays, themes, game_url, opening_tags,
                            difficulty_level
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            row['PuzzleId'], row['FEN'], row['Moves'], 
                            int(row['Rating']), int(row['RatingDeviation']),
                            int(row['Popularity']), int(row['NbPlays']), 
                            row['Themes'], row['GameUrl'], row['OpeningTags'],
                            difficulty_level
                        ))
                        imported_count += 1
                        
                        # Record progress periodically
                        if imported_count % 1000 == 0:
                            self.conn.commit()
                            
                    except Exception as e:
                        skipped_count += 1
                
                self.conn.commit()
                
            return imported_count
            
        except Exception as e:
            return 0

    def _get_difficulty_level(self, rating):
        """
        Determine the difficulty level based on the puzzle rating.
        
        Args:
            rating (int): The puzzle's ELO rating.
            
        Returns:
            str: The difficulty level name.
        """
        for level, (min_elo, max_elo) in ELO_DIFFICULTY_LEVELS.items():
            if min_elo <= rating <= max_elo:
                return level
        return "Unknown"
    
    def get_puzzles(self, filters=None, limit=50, offset=0):
        """
        Get puzzles based on filters.
        
        Args:
            filters (dict): Dictionary of filters to apply.
            limit (int): Maximum number of puzzles to return.
            offset (int): Offset for pagination.
            
        Returns:
            list: List of puzzle dictionaries.
        """
        if not self.conn:
            return []
            
        if filters is None:
            filters = {}
            
        query_parts = ["SELECT * FROM puzzles WHERE 1=1"]
        query_params = []
        
        # Apply filters
        if 'min_rating' in filters:
            query_parts.append("AND rating >= ?")
            query_params.append(filters['min_rating'])
            
        if 'max_rating' in filters:
            query_parts.append("AND rating <= ?")
            query_params.append(filters['max_rating'])
            
        if 'difficulty_level' in filters:
            query_parts.append("AND difficulty_level = ?")
            query_params.append(filters['difficulty_level'])
            
        if 'themes' in filters and filters['themes']:
            themes_list = filters['themes']
            if isinstance(themes_list, str):
                themes_list = [themes_list]
                
            for theme in themes_list:
                query_parts.append("AND themes LIKE ?")
                query_params.append(f"%{theme}%")
                
        if 'min_popularity' in filters:
            query_parts.append("AND popularity >= ?")
            query_params.append(filters['min_popularity'])
            
        if 'min_plays' in filters:
            query_parts.append("AND nb_plays >= ?")
            query_params.append(filters['min_plays'])
            
        if 'opening_tags' in filters and filters['opening_tags']:
            query_parts.append("AND opening_tags LIKE ?")
            query_params.append(f"%{filters['opening_tags']}%")
            
        # Add order by clause
        order_by = filters.get('order_by', 'rating')
        order_dir = filters.get('order_dir', 'ASC')
        query_parts.append(f"ORDER BY {order_by} {order_dir}")
        
        # Add limit and offset
        query_parts.append("LIMIT ? OFFSET ?")
        query_params.extend([limit, offset])
        
        # Execute the query
        cursor = self.conn.cursor()
        cursor.execute(" ".join(query_parts), query_params)
        
        # Convert rows to dictionaries
        puzzles = [dict(row) for row in cursor.fetchall()]
        return puzzles
    
    def get_puzzle_by_id(self, puzzle_id):
        """
        Get a puzzle by its ID.
        
        Args:
            puzzle_id (str): The puzzle ID.
            
        Returns:
            dict: The puzzle data or None if not found.
        """
        if not self.conn:
            return None
            
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM puzzles WHERE puzzle_id = ?", (puzzle_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_available_themes(self):
        """
        Get a list of all available themes in the database.
        
        Returns:
            list: Sorted list of unique themes.
        """
        if not self.conn:
            return []
            
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT themes FROM puzzles WHERE themes IS NOT NULL AND themes != ''")
        
        # Extract and flatten themes
        all_themes = set()
        for row in cursor.fetchall():
            themes = row['themes'].split()
            all_themes.update(themes)
            
        return sorted(list(all_themes))
    
    def get_database_stats(self):
        """
        Get statistics about the puzzle database.
        
        Returns:
            dict: Database statistics.
        """
        if not self.conn:
            return {
                "total_puzzles": 0,
                "difficulty_distribution": {},
                "rating_stats": {"min": 0, "max": 0, "avg": 0},
                "top_themes": {}
            }
            
        cursor = self.conn.cursor()
        
        # Total puzzles
        cursor.execute("SELECT COUNT(*) as count FROM puzzles")
        total_puzzles = cursor.fetchone()['count']
        
        # Puzzles by difficulty level
        difficulty_stats = {}
        cursor.execute("SELECT difficulty_level, COUNT(*) as count FROM puzzles GROUP BY difficulty_level")
        for row in cursor.fetchall():
            difficulty_stats[row['difficulty_level']] = row['count']
            
        # Rating distribution
        cursor.execute("SELECT MIN(rating) as min, MAX(rating) as max, AVG(rating) as avg FROM puzzles")
        rating_stats = dict(cursor.fetchone())
        
        # Theme distribution (top 10)
        theme_counts = {}
        cursor.execute("SELECT themes FROM puzzles WHERE themes IS NOT NULL AND themes != ''")
        for row in cursor.fetchall():
            for theme in row['themes'].split():
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
                
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_puzzles": total_puzzles,
            "difficulty_distribution": difficulty_stats,
            "rating_stats": rating_stats,
            "top_themes": dict(top_themes)
        }
    
    def generate_test_set(self, filters=None, size=50, output_path=None):
        """
        Generate a test set of puzzles based on filters and save to a file.
        
        Args:
            filters (dict): Dictionary of filters to apply.
            size (int): Number of puzzles in the test set.
            output_path (str): Path to save the test set. If None, uses default location.
            
        Returns:
            str: Path to the generated test set file.
        """
        if not self.conn:
            return None
            
        # Get puzzles based on filters
        puzzles = self.get_puzzles(filters, limit=size)
        
        if not puzzles:
            return None
            
        # Create test set data
        test_set = {
            "metadata": {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "size": len(puzzles),
                "filters": filters
            },
            "puzzles": puzzles
        }
        
        # Determine output path
        if output_path is None:
            output_dir = self.config.get('test_set_output', {}).get('directory', 
                                       'training_data/fen_data_puzzle_lists/')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create a descriptive filename based on filters
            filter_desc = ""
            if filters:
                if 'difficulty_level' in filters:
                    filter_desc += f"_{filters['difficulty_level']}"
                elif 'min_rating' in filters and 'max_rating' in filters:
                    filter_desc += f"_{filters['min_rating']}-{filters['max_rating']}"
                
                if 'themes' in filters and filters['themes']:
                    if isinstance(filters['themes'], list):
                        filter_desc += f"_{'-'.join(filters['themes'][:2])}"
                    else:
                        filter_desc += f"_{filters['themes']}"
            
            output_path = os.path.join(output_dir, f"puzzle_test_set{filter_desc}_{timestamp}.json")
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(test_set, f, indent=2, default=str)
            
        return output_path
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            try:
                self.conn.close()
            except Exception as e:
                raise
            finally:
                self.conn = None

    def __del__(self):
        """Destructor to ensure the database connection is closed."""
        self.close()

    def get_random_fens(self, count=50, min_elo=None, max_elo=None, themes=None):
        """
        Fetch a random set of FENs from the database, optionally filtered by ELO and themes.
        Returns a list of FEN strings.
        """
        filters = {}
        if min_elo is not None:
            filters['min_rating'] = min_elo
        if max_elo is not None:
            filters['max_rating'] = max_elo
        if themes:
            filters['themes'] = themes
        # Use ORDER BY RANDOM() for random selection
        query_parts = ["SELECT fen FROM puzzles WHERE 1=1"]
        query_params = []
        if 'min_rating' in filters:
            query_parts.append("AND rating >= ?")
            query_params.append(filters['min_rating'])
        if 'max_rating' in filters:
            query_parts.append("AND rating <= ?")
            query_params.append(filters['max_rating'])
        if 'themes' in filters and filters['themes']:
            themes_list = filters['themes']
            if isinstance(themes_list, str):
                themes_list = [themes_list]
            for theme in themes_list:
                query_parts.append("AND themes LIKE ?")
                query_params.append(f"%{theme}%")
        query_parts.append("ORDER BY RANDOM() LIMIT ?")
        query_params.append(count)
        if not self.conn:
            return []
        if not self.conn:
            return []
        cursor = self.conn.cursor()
        cursor.execute(" ".join(query_parts), query_params)
        return [row['fen'] for row in cursor.fetchall()]


class ConsoleUI:
    """Basic console UI for the Puzzle Database Manager."""
    
    def __init__(self, db_manager):
        """
        Initialize the console UI.
        
        Args:
            db_manager (PuzzleDBManager): The puzzle database manager instance.
        """
        self.db_manager = db_manager
    
    def start(self):
        """Start the console UI."""
        while True:
            self._display_menu()
            choice = input("Enter your choice: ")
            
            if choice == '1':
                self._import_puzzles()
            elif choice == '2':
                self._query_puzzles()
            elif choice == '3':
                self._generate_test_set()
            elif choice == '4':
                self._display_db_stats()
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")
                
            print("\nPress Enter to continue...")
            input()
    
    def _display_menu(self):
        """Display the main menu."""
        print("\n" + "=" * 50)
        print("V7P3R CHESS ENGINE - PUZZLE DATABASE MANAGER")
        print("=" * 50)
        print("1. Import puzzles from CSV")
        print("2. Query puzzles")
        print("3. Generate test set")
        print("4. Display database statistics")
        print("5. Exit")
        print("=" * 50)
    
    def _import_puzzles(self):
        """Import puzzles from a CSV file."""
        print("\n--- IMPORT PUZZLES ---")
        
        # Get CSV path
        default_path = "training_data/csv_data_puzzles/lichess_db_puzzle.csv"
        path = input(f"Enter the CSV file path [{default_path}]: ")
        path = path.strip() or default_path
        
        if not os.path.exists(path):
            print(f"Error: File does not exist: {path}")
            return
            
        # Confirm import
        print(f"Importing puzzles from {path}...")
        if input("This may take a while. Continue? (y/n): ").lower() != 'y':
            print("Import cancelled.")
            return
            
        # Import puzzles
        count = self.db_manager.ingest_puzzles_from_csv(path)
        print(f"Successfully imported {count} puzzles.")
    
    def _query_puzzles(self):
        """Query puzzles with filters."""
        print("\n--- QUERY PUZZLES ---")
        
        # Build filters
        filters = {}
        
        # Difficulty level
        print("\nDifficulty levels:")
        for i, level in enumerate(ELO_DIFFICULTY_LEVELS.keys(), 1):
            min_elo, max_elo = ELO_DIFFICULTY_LEVELS[level]
            print(f"{i}. {level} ({min_elo}-{max_elo})")
        print("0. Custom ELO range")
        
        difficulty_choice = input("Select difficulty level [0]: ")
        difficulty_choice = difficulty_choice.strip() or '0'
        
        if difficulty_choice == '0':
            min_rating = input("Enter minimum rating [800]: ")
            filters['min_rating'] = int(min_rating) if min_rating.strip() else 800
            
            max_rating = input("Enter maximum rating [3000]: ")
            filters['max_rating'] = int(max_rating) if max_rating.strip() else 3000
        else:
            try:
                level_idx = int(difficulty_choice) - 1
                level_name = list(ELO_DIFFICULTY_LEVELS.keys())[level_idx]
                filters['difficulty_level'] = level_name
            except (ValueError, IndexError):
                print("Invalid choice, using default range")
                filters['min_rating'] = 800
                filters['max_rating'] = 3000
        
        # Themes
        print("\nCommon themes: mate, fork, pin, discovery, sacrifice, hanging, tactical, endgame")
        themes_input = input("Enter themes (comma-separated) or press Enter to skip: ")
        if themes_input.strip():
            filters['themes'] = [theme.strip() for theme in themes_input.split(',')]
        
        # Minimum popularity
        pop_input = input("Enter minimum popularity [0]: ")
        if pop_input.strip():
            filters['min_popularity'] = int(pop_input)
        
        # Limit
        limit_input = input("Enter maximum number of puzzles to retrieve [50]: ")
        limit = int(limit_input) if limit_input.strip() else 50
        
        # Query puzzles
        puzzles = self.db_manager.get_puzzles(filters, limit=limit)
        
        if not puzzles:
            print("No puzzles found matching the criteria.")
            return
            
        print(f"\nFound {len(puzzles)} puzzles:")
        print("-" * 80)
        print(f"{'ID':<10} {'Rating':<7} {'Difficulty':<12} {'Themes':<40} {'Popularity':<10}")
        print("-" * 80)
        
        for puzzle in puzzles[:20]:  # Show first 20
            themes_display = puzzle['themes'][:37] + '...' if len(puzzle['themes']) > 40 else puzzle['themes']
            print(f"{puzzle['puzzle_id']:<10} {puzzle['rating']:<7} {puzzle['difficulty_level']:<12} {themes_display:<40} {puzzle['popularity']:<10}")
            
        if len(puzzles) > 20:
            print(f"... and {len(puzzles) - 20} more puzzles")
    
    def _generate_test_set(self):
        """Generate a test set based on filters."""
        print("\n--- GENERATE TEST SET ---")
        
        # Build filters (similar to query_puzzles)
        filters = {}
        
        # Difficulty level
        print("\nDifficulty levels:")
        for i, level in enumerate(ELO_DIFFICULTY_LEVELS.keys(), 1):
            min_elo, max_elo = ELO_DIFFICULTY_LEVELS[level]
            print(f"{i}. {level} ({min_elo}-{max_elo})")
        print("0. Custom ELO range")
        
        difficulty_choice = input("Select difficulty level [0]: ")
        difficulty_choice = difficulty_choice.strip() or '0'
        
        if difficulty_choice == '0':
            min_rating = input("Enter minimum rating [800]: ")
            filters['min_rating'] = int(min_rating) if min_rating.strip() else 800
            
            max_rating = input("Enter maximum rating [3000]: ")
            filters['max_rating'] = int(max_rating) if max_rating.strip() else 3000
        else:
            try:
                level_idx = int(difficulty_choice) - 1
                level_name = list(ELO_DIFFICULTY_LEVELS.keys())[level_idx]
                filters['difficulty_level'] = level_name
            except (ValueError, IndexError):
                print("Invalid choice, using default range")
                filters['min_rating'] = 800
                filters['max_rating'] = 3000
        
        # Themes
        print("\nCommon themes: mate, fork, pin, discovery, sacrifice, hanging, tactical, endgame")
        themes_input = input("Enter themes (comma-separated) or press Enter to skip: ")
        if themes_input.strip():
            filters['themes'] = [theme.strip() for theme in themes_input.split(',')]
        
        # Minimum popularity
        pop_input = input("Enter minimum popularity [0]: ")
        if pop_input.strip():
            filters['min_popularity'] = int(pop_input)
        
        # Test set size
        size_input = input("Enter test set size [50]: ")
        size = int(size_input) if size_input.strip() else 50
        
        # Generate test set
        output_path = self.db_manager.generate_test_set(filters, size)
        
        if output_path:
            print(f"Test set generated successfully: {output_path}")
        else:
            print("Failed to generate test set.")
    
    def _display_db_stats(self):
        """Display database statistics."""
        print("\n--- DATABASE STATISTICS ---")
        
        stats = self.db_manager.get_database_stats()
        
        print(f"Total puzzles: {stats['total_puzzles']}")
        
        print("\nPuzzles by difficulty level:")
        for level, count in stats['difficulty_distribution'].items():
            print(f"  {level}: {count}")
            
        print("\nRating statistics:")
        print(f"  Minimum: {stats['rating_stats']['min']}")
        print(f"  Maximum: {stats['rating_stats']['max']}")
        print(f"  Average: {stats['rating_stats']['avg']:.2f}")
        
        print("\nTop 10 themes:")
        for theme, count in stats['top_themes'].items():
            print(f"  {theme}: {count}")




def main():
    import argparse
    parser = argparse.ArgumentParser(description="V7P3R Chess Engine Puzzle Database Manager")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--csv", help="Path to CSV file to import")
    parser.add_argument("--generate-test", action="store_true", help="Generate a test set")
    parser.add_argument("--no-ui", action="store_true", help="Run without UI (for batch operations)")
    args = parser.parse_args()

    db_manager = PuzzleDBManager(args.config)

    # Import CSV if specified
    if args.csv:
        print(f"Importing puzzles from {args.csv}...")
        count = db_manager.ingest_puzzles_from_csv(args.csv)
        print(f"Successfully imported {count} puzzles.")
        db_manager.close()
        return 0

    # Generate test set if specified
    if args.generate_test:
        print("Generating test set...")
        output_path = db_manager.generate_test_set()
        print(f"Test set generated at {output_path}")
        db_manager.close()
        return 0

    # Always use the simple console UI if not in batch mode
    if not args.no_ui:
        ui = ConsoleUI(db_manager)
        try:
            ui.start()
        finally:
            db_manager.close()
    else:
        print("No operation specified. Use --csv to import, --generate-test to create a test set, or run without --no-ui for interactive mode.")
        db_manager.close()
        return 0

if __name__ == "__main__":
    main()
