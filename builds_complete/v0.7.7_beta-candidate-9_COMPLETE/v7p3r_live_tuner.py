# v7p3r_live_tuner.py
import os
import sys
import datetime
import sqlite3
import chess
from v7p3r import v7p3rEngine
from v7p3r_config import v7p3rConfig
from v7p3r_debug import v7p3rLogger

# Setup centralized logging for this module
v7p3r_live_tuner_logger = v7p3rLogger.setup_logger("v7p3r_live_tuner")

OBSERVATION_MODE = False  # Set to True to enable observation features, such as pausing after each position
position_config = {
    "rating": 2000, # Max rating for starting positions
    "themes": ["mateIn1", "mateIn2","mateIn3"], # Themes to filter by, e.g., 'mate', 'tactics'
    "max_moves": 10, # Max moves in the position
    "limit": 25, # Limit the number of starting positions returned
    "query_type": "loose" # 'strict' uses 'AND' logic, 'loose' uses 'OR' logic, when querying themes
}

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base = getattr(sys, '_MEIPASS', None)
    if base:
        return os.path.join(base, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
# =====================================
# ========== LOGGING SETUP ============
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create logging directory relative to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
log_dir = os.path.join(project_root, 'logging')
class v7p3rTuner:
    """v7p3rTuner
    This class is responsible for tuning the v7p3r engine using starting positions from a database.
    """
    def __init__(self):
        # Initialize logger
        self.logger = v7p3r_live_tuner_logger
        
        # Load Configuration
        self.config_manager = v7p3rConfig()
        self.engine_config = self.config_manager.get_engine_config()
        self.monitoring_enabled = self.engine_config.get("monitoring_enabled", True)
        self.verbose_output_enabled = self.engine_config.get("verbose_output", True)
        
        # Initialize the engine with the loaded config
        self.engine = v7p3rEngine(self.engine_config)

        # Initialize the board and other attributes
        self.board = chess.Board()  # Start with an empty board
        self.current_position = 0

    def get_starting_positions(self, criteria: dict):
        """
        Fetches starting positions from the database meeting the criteria
        """
        # Open read only db connection to puzzle_data.db (located at project root /puzzles/puzzle_data.db)
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'puzzles', 'puzzle_data.db'))
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        # Prepare query for multiple themes, supporting strict (AND) and loose (OR) logic
        themes = criteria.get('themes', ['mate'])
        query_type = criteria.get('query_type', 'strict')
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
        rating_limit = criteria.get('rating', 3000)
        max_moves = criteria.get('max_moves', 10)
        limit = criteria.get('limit', 25)
        params = [rating_limit, max_moves] + theme_params + [limit]
        
        # Debug output
        if self.monitoring_enabled and self.logger:
            self.logger.info(f"Database query: {query}")
            self.logger.info(f"Query parameters: {params}")
            self.logger.info(f"Criteria: rating <= {rating_limit}, max_moves <= {max_moves}, themes: {themes} ({query_type}), limit: {limit}")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"Database returned {len(results)} raw results")
            
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
                    
                    if self.monitoring_enabled and self.logger and i < 3:
                        self.logger.info(f"Sample position {i+1}: FEN={fen[:50]}..., moves={moves_list}")
                    
                    starting_positions.append({fen: moves_list})
                
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Successfully processed {len(starting_positions)} starting positions matching criteria: {criteria}")
                print(f"Found {len(starting_positions)} starting positions matching criteria")
            else:
                if self.monitoring_enabled and self.logger:
                    self.logger.warning(f"No starting positions found matching criteria: {criteria}")
                print(f"No starting positions found matching criteria: {criteria}")
                starting_positions = []
        return starting_positions
    
    def solve_position(self, position_object, position_count, current_position=0):
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

        if self.monitoring_enabled and self.logger:
            self.logger.info(f"Starting position {current_position}/{position_count}: FEN={current_position_fen} | Total moves: {total_moves} | Solution moves: {solution_moves}")
        print(f"\n--- Starting FEN: {current_position_fen}")
        print(f"Solution move sequence: {solution_moves}")

        for solution_move in solution_moves:
            current_move_number += 1
            if current_move_number % 2 == 1: 
                # Odd moves are meant to be played automatically
                self.board.push_uci(solution_move)
                played_moves.append(solution_move)
                print(f"Last played move {current_move_number}/{total_moves}: {solution_move} | FEN: {self.board.fen()}")
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Last played move {current_move_number}/{total_moves}: {solution_move} | FEN: {self.board.fen()}")
                continue
            else:
                # If it's an even move, we need to let the engine play
                print(f"Engine is thinking... (engine should play {solution_move})")
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Sending position to engine: {self.board.fen()} (engine should play {solution_move})")

                # Find the engine's move
                engine_guess = self.engine.search_engine.search(self.board, self.board.turn)

                # Validate engine move
                if engine_guess is None or str(engine_guess) == "0000":
                    if self.monitoring_enabled and self.logger:
                        self.logger.error(f"[Error] Engine could not find a valid move for position {current_position}/{position_count} at move {current_move_number}.")
                    break
                if engine_guess not in self.board.legal_moves:
                    if self.monitoring_enabled and self.logger:
                        self.logger.error(f"[Error] Engine guess {engine_guess} is not a legal move in this position. Skipping.")
                    break

                # Record engine move
                played_moves.append(engine_guess.uci())

                # Push engine move
                self.board.push(engine_guess)
                print(f"Engine played move {current_move_number}/{total_moves}: {engine_guess.uci()} | FEN: {self.board.fen()}")
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Engine played move {current_move_number}/{total_moves}: {engine_guess.uci()} | FEN: {self.board.fen()}")
                
                if self.board.is_game_over():
                    print(f"Game over: {self.board.result()} Reason: {self.board.outcome()}")
                    if self.monitoring_enabled and self.logger:
                        self.logger.info(f"Game over: {self.board.result()} Reason: {self.board.outcome()}")
                    break

        # After all moves, compare engine's sequence to solution
        print(f"Engine move sequence: {played_moves}")
        print(f"Solution move sequence: {solution_moves}")
        if played_moves == solution_moves:
            print("Engine solved the position correctly! WIN recorded.")
            if self.monitoring_enabled and self.logger:
                self.logger.info("Engine solved the position correctly! WIN recorded.")
        else:
            print("Engine did not solve the position. LOSS recorded.")
            if self.monitoring_enabled and self.logger:
                self.logger.info("Engine did not solve the position. LOSS recorded.")

def main(position_config):
    # Initialize the tuner
    tuner = v7p3rTuner()
    starting_positions = tuner.get_starting_positions(position_config)
    position_count = len(starting_positions)
    current_position = 0
    if position_count == 0:
        print("No starting positions found matching the criteria.")
        return
    for position_object in starting_positions:
        current_position += 1
        tuner.solve_position(position_object, position_count, current_position)

        if OBSERVATION_MODE:
            # Pause and wait for the user to review and continue
            input("Press Enter to continue to the next position... (CTRL+C to exit)")

if __name__ == "__main__":
    try:
        main(position_config)
    except Exception as e:
        print("Exception occurred:", e)
        import traceback
        traceback.print_exc()
