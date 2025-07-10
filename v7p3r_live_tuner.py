# v7p3r_live_tuner.py
import os
import sys
from typing import Optional
import sqlite3
import chess
from v7p3r import v7p3rEngine
from v7p3r_config import v7p3rConfig

CONFIG_NAME = 'mvv_lva_test_config'  # Configuration file for MVV-LVA testing

OBSERVATION_MODE = False  # Set to True to enable observation features, such as pausing after each position

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base = getattr(sys, '_MEIPASS', None)
    if base:
        return os.path.join(base, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class v7p3rTuner:
    """v7p3rTuner
    This class is responsible for tuning the v7p3r engine using starting positions from a database.
    """
    def __init__(self, config_name: Optional[str] = 'default_config'):
        # Load Configuration
        self.config_manager = v7p3rConfig(config_name=config_name)
        
        # Get configs
        self.engine_config = self.config_manager.get_engine_config()
        self.puzzle_config = self.config_manager.get_puzzle_config()
        
        # Initialize the engine with the test config
        self.engine = v7p3rEngine(self.engine_config)

        # Initialize the board and other attributes
        self.board = chess.Board()
        self.current_position = 0
        
        # Set up position config from puzzle config
        self.position_config = self.puzzle_config.get('position_config', {})
        self.max_rating = self.position_config.get("max_rating", 2000)
        self.themes = self.position_config.get("themes", ["mate"])
        self.max_moves = self.position_config.get("max_moves", 10)
        self.position_limit = self.position_config.get("position_limit", 25)
        self.query_type = self.position_config.get("query_type", "loose")

    def get_starting_positions(self):
        """
        Fetches starting positions from the database meeting the criteria from config
        If criteria is passed, it overrides the config settings
        """

        # Open read only db connection to puzzle_data.db (located at project root)
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                              self.puzzle_config.get('puzzle_database', {}).get('db_path', 'puzzle_data.db')))
                                              
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        # Prepare query for multiple themes, supporting strict (AND) and loose (OR) logic
        themes = self.themes
        query_type = self.query_type
        if query_type not in ['strict', 'loose']:
            raise ValueError("query_type must be 'strict' or 'loose'")
        if not isinstance(themes, list):
            themes = [themes]
        
        # Build theme filtering clause correctly
        if themes:
            theme_clauses = ["themes LIKE ?" for _ in themes]
            joiner = " AND " if query_type == 'strict' else " OR "
            where_themes = f"({joiner.join(theme_clauses)})"
            theme_params = [f"%{theme}%" for theme in themes]
        else:
            where_themes = "1=1"  # Always true if no themes specified
            theme_params = []
        
        # Build the complete query
        query = f"""
            SELECT fen, moves 
            FROM puzzles 
            WHERE rating <= ? 
            AND (LENGTH(moves) - LENGTH(REPLACE(moves, ' ', '')) + 1) <= ? 
            AND {where_themes}
            ORDER BY rating ASC
            LIMIT ?
        """
        
        # Prepare parameters in correct order
        rating_limit = self.max_rating
        max_moves = self.max_moves
        limit = self.position_limit
        params = [rating_limit, max_moves] + theme_params + [limit]
        print(f"Executing query with params: {params}")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            if results:
                # Extract FEN and moves from results
                starting_positions = []
                for i, row in enumerate(results):
                    fen = row[0]  # FEN is in the first column
                    moves = row[1]  # Moves are in the second column, a space-separated string of UCI moves
                    if moves:
                        moves_list = moves.strip().split(' ')
                        # Filter out empty strings
                        moves_list = [move for move in moves_list if move.strip()]
                    else:
                        moves_list = []
                    
                    starting_positions.append({fen: moves_list})
                
                print(f"Found {len(starting_positions)} starting positions matching criteria")
                print(f"Criteria: max_rating={self.max_rating}, max_moves={self.max_moves}, themes={self.themes}, position_limit={self.position_limit}, query_type={self.query_type}")
            else:
                print(f"No starting positions found matching criteria: {self.puzzle_config}")
                starting_positions = []
        return starting_positions
    
    def solve_position(self, position_object, position_count, current_position=0) -> bool:
        current_position_fen = next(iter(position_object.keys()))
        solution_moves = position_object.get(current_position_fen, [])
        current_position += 1
        self.current_position = current_position

        # Set up the self.board with the starting position
        self.board = chess.Board(current_position_fen)
        self.white_to_move = self.board.turn  # True if it's White's turn, False if it's Black
        total_moves = len(solution_moves)
        current_move_number = 0
        played_moves = []

        print(f"\n--- Starting FEN: {current_position_fen}")
        print(f"Solution move sequence: {solution_moves}")

        for solution_move in solution_moves:
            current_move_number += 1
            if current_move_number % 2 == 1: 
                # Odd moves are meant to be played automatically
                try:
                    # Validate the move before pushing
                    move = chess.Move.from_uci(solution_move)
                    if move in self.board.legal_moves:
                        self.board.push_uci(solution_move)
                        played_moves.append(solution_move)
                        print(f"Last played move {current_move_number}/{total_moves}: {solution_move} | FEN: {self.board.fen()}")
                    else:
                        print(f"ERROR: Solution move {solution_move} is not legal in position {self.board.fen()}")
                        print(f"Legal moves: {[move.uci() for move in self.board.legal_moves]}")
                        break
                except ValueError as e:
                    print(f"ERROR: Invalid move format '{solution_move}': {e}")
                    break
                except Exception as e:
                    print(f"ERROR: Failed to execute move '{solution_move}': {e}")
                    break
                continue
            else:
                # If it's an even move, we need to let the engine play
                print(f"Engine is thinking... (engine should play {solution_move})")
                
                # Find the engine's move
                engine_guess = self.engine.search_engine.search(self.board, self.board.turn)

                # Validate engine move
                if engine_guess is None or str(engine_guess) == "0000":
                    break
                if engine_guess not in self.board.legal_moves:
                    break

                # Record engine move
                played_moves.append(engine_guess.uci())

                # Push engine move
                self.board.push(engine_guess)
                print(f"Engine played move {current_move_number}/{total_moves}: {engine_guess.uci()} | FEN: {self.board.fen()}")
                
                if self.board.is_game_over():
                    break

        # After all moves, compare engine's sequence to solution
        print(f"Engine move sequence: {played_moves}")
        print(f"Solution move sequence: {solution_moves}")
        if played_moves == solution_moves:
            print("Engine solved the position correctly! WIN recorded.")
            return True
        else:
            print("Engine did not solve the position. LOSS recorded.")
            return False

if __name__ == "__main__":
    try:
        
        tuner = v7p3rTuner(CONFIG_NAME)
        # Get starting positions using config settings
        positions = tuner.get_starting_positions()
        positions_solved = 0
        current_position_solved = False
        # Test each position
        for i, position in enumerate(positions, 1):
            current_position_solved = False
            current_position_solved = tuner.solve_position(position, len(positions), i)

            if current_position_solved:
                positions_solved += 1

            if OBSERVATION_MODE:
                # Pause and wait for the user to review and continue
                input("Press Enter to continue to the next position... (CTRL+C to exit)")
        print(f"\nTotal positions solved: {positions_solved}/{len(positions)}")
    except Exception as e:
        print("Exception occurred:", e)