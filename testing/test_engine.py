from v7p3r import v7p3rEngine
from v7p3r_config import v7p3rConfig
import chess
import os

# Create engine with centipawn_test_config
config = v7p3rConfig('configs/centipawn_test_config.json')
engine_config = config.get_engine_config()
print(f'Config loaded, search_algorithm = {engine_config.get("search_algorithm", "not found")}')

# Create engine with this config
engine = v7p3rEngine(engine_config=engine_config)
print(f'Engine created, search_algorithm = {engine.search_algorithm}')
print(f'Search engine search_algorithm = {engine.search_engine.search_algorithm}')

# Create a new board
board = chess.Board()

# Make a simple search (just to log the search algorithm being used)
print("About to search...")
move = engine.search_engine.search(board, chess.WHITE)
print(f"Engine returned move: {move}")

# Check the search log to confirm algorithm used
log_file_path = os.path.join("logging", "v7p3r_search.log")
if os.path.exists(log_file_path):
    with open(log_file_path, 'r') as log_file:
        first_line = log_file.readline().strip()
        print(f"First line of search log: {first_line}")
else:
    print("Log file not found")
