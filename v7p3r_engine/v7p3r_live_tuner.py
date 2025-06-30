# v7p3r_live_tuner.py
import os
import sys
import logging
import datetime
import sqlite3
import chess
from v7p3r_engine.v7p3r import v7p3rEngine

DEBUG_MODE = False  # Set to True to enable debugging features, such as pausing after each position
position_config = {
    "rating": 3000, # Max rating for starting positions
    "themes": ["mate", "endgame"], # Themes to filter by, e.g., 'mate', 'tactics'
    "max_moves": 10, # Max moves in the position
    "limit": 25, # Limit the number of starting positions returned
    "query_type": "strict" # 'strict' uses 'AND' logic, 'loose' uses 'OR' logic, when querying themes
}
config = {
    "engine_config": {
        "name": "v7p3r",                     # Name of the engine, used for identification and logging
        "version": "1.0.0",                  # Version of the engine, used for identification and logging
        "ruleset": "default_evaluation",    # Name of the evaluation rule set to use, see below for available options
        "search_algorithm": "lookahead",       # Move search type for White (see search_algorithms for options)
        "depth": 3,                          # Depth of search for AI, 1 for random, 2 for simple search, 3+ for more complex searches
        "max_depth": 4,                      # Max depth of search for AI, 1 for random, 2 for simple search, 3+ for more complex searches
        "use_game_phase": False,             # Use game phase evaluation
        "monitoring_enabled": True,          # Enable or disable monitoring features
        "verbose_output": True,              # Enable or disable verbose output for debugging
        "logger": "v7p3r_tuning_logger",     # Logger name for the engine, used for logging engine-specific events
    }
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
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def get_log_file_path():
    # Optional timestamp for log file name
    timestamp = get_timestamp()
    log_dir = "logging"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"v7p3r_evaluation_engine.log")
v7p3r_engine_logger = logging.getLogger("v7p3r_evaluation_engine")
v7p3r_engine_logger.setLevel(logging.DEBUG)
_init_status = globals().get("_init_status", {})
if not _init_status.get("initialized", False):
    log_file_path = get_log_file_path()
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10*1024*1024,
        backupCount=3,
        delay=True
    )
    formatter = logging.Formatter(
        '%(asctime)s | %(funcName)-15s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    v7p3r_engine_logger.addHandler(file_handler)
    v7p3r_engine_logger.propagate = False
    _init_status["initialized"] = True
    # Store the log file path for later use (e.g., to match with PGN/config)
    _init_status["log_file_path"] = log_file_path

class v7p3rTuner:
    """v7p3rTuner
    This class is responsible for tuning the v7p3r engine using starting positions from a database.
    """
    def __init__(self, config):
        # Set up logging
        logging.basicConfig(
            filename='logging/v7p3r_live_tuner.log',
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s'
        )
        self.logger = logging.getLogger("v7p3r_live_tuner")
        self.current_position = 0
        self.engine_config = config.get("engine_config", {})
        self.monitoring_enabled = self.engine_config.get("monitoring_enabled", False)
        self.verbose_output_enabled = self.engine_config.get("verbose_output", False)
        
        # Initialize the engine with the loaded config
        self.engine = v7p3rEngine(self.engine_config)

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
        if themes:
            theme_clauses = ["themes LIKE ?" for _ in themes]
            joiner = " AND " if query_type == 'strict' else " OR "
            where_themes = f"({' '.join([joiner.join(theme_clauses)])})"
            theme_params = [f"%{theme}%" for theme in themes]
        else:
            where_themes = "(1)"  # Always true if no themes specified
            theme_params = []
        query = f"""
            SELECT fen, moves 
            FROM puzzles 
            WHERE rating < ? 
            AND LENGTH(moves) < ? 
            AND {where_themes}
            LIMIT ?
        """
        params = [criteria.get('rating', 3000), criteria.get('max_moves', 2)] + theme_params + [criteria.get('limit', 10)]
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            if results:
                # Extract FEN and moves from results
                starting_positions = []
                for row in results:
                    fen = row[0]  # FEN is in the first column
                    moves = row[1]  # Moves are in the second column, a space-separated string of UCI moves
                    if moves:
                        moves = moves.split(' ')
                    starting_positions.append({fen: moves})
                logging.info(f"Found {len(starting_positions)} starting positions matching criteria: {criteria}")
            else:
                logging.warning(f"No starting positions found matching criteria: {criteria}")
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
        engine_moves = []

        print(f"\n--- Starting FEN: {current_position_fen}")
        print(f"Solution move sequence: {solution_moves}")

        for solution_move in solution_moves:
            current_move_number += 1
            print(f"Processing position {current_position}/{position_count}, move {current_move_number}/{total_moves}: (engine should play {solution_move})")
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"Processing position {current_position}/{position_count}, move {current_move_number}/{total_moves}: (engine should play {solution_move})")
            engine_guess = self.engine.search_engine.search(self.board, self.board.turn)

            # Validate engine move
            if engine_guess is None or str(engine_guess) == "0000":
                print(f"Engine could not find a valid move for position {current_position}/{position_count} at move {current_move_number}.")
                if self.logger and self.monitoring_enabled:
                    self.logger.error(f"[Error] Engine could not find a valid move for position {current_position}/{position_count} at move {current_move_number}.")
                break
            if engine_guess not in self.board.legal_moves:
                print(f"Engine guess {engine_guess} is not a legal move in this position. Skipping.")
                if self.logger and self.monitoring_enabled:
                    self.logger.error(f"[Error] Engine guess {engine_guess} is not a legal move in this position. Skipping.")
                break

            # Record engine move
            engine_moves.append(engine_guess.uci())

            # Push engine move
            self.board.push(engine_guess)
            print(f"Engine played move {current_move_number}/{total_moves}: {engine_guess.uci()}")
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"Engine played move {current_move_number}/{total_moves}: {engine_guess.uci()}")
            # Print board state for debugging
            print(self.board)
            if self.board.is_game_over():
                print(f"Game over: {self.board.result()} Reason: {self.board.outcome()}")
                if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                    self.logger.info(f"Game over: {self.board.result()} Reason: {self.board.outcome()}")
                break

        # After all moves, compare engine's sequence to solution
        print(f"Engine move sequence: {engine_moves}")
        print(f"Solution move sequence: {solution_moves}")
        if engine_moves == solution_moves:
            print("Engine solved the position correctly! WIN recorded.")
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info("Engine solved the position correctly! WIN recorded.")
        else:
            print("Engine did not solve the position. LOSS recorded.")
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info("Engine did not solve the position. LOSS recorded.")

def main(config, position_config):
    # Initialize the tuner
    tuner = v7p3rTuner(config)
    starting_positions = tuner.get_starting_positions(position_config)
    position_count = len(starting_positions)
    current_position = 0
    if position_count == 0:
        print("No starting positions found matching the criteria.")
        return
    for position_object in starting_positions:
        tuner.solve_position(position_object, position_count, current_position)

        if DEBUG_MODE:
            # Pause and wait for the user to review and continue
            input("Press Enter to continue to the next position... (CTRL+C to exit)")

if __name__ == "__main__":
    try:
        main(config, position_config)
    except Exception as e:
        print("Exception occurred:", e)
        import traceback
        traceback.print_exc()