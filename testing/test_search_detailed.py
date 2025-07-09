import os
import time

# Clear the search log first
log_file_path = os.path.join("logging", "v7p3r_search.log")
if os.path.exists(log_file_path):
    with open(log_file_path, 'w') as log_file:
        log_file.write("")  # Clear the file

# Now run the actual test
from v7p3r import v7p3rEngine
from v7p3r_config import v7p3rConfig
import chess

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

# Wait a moment for the log to be written
time.sleep(1)

# Check the search log to confirm algorithm used
if os.path.exists(log_file_path):
    with open(log_file_path, 'r') as log_file:
        lines = log_file.readlines()
        print(f"Total lines in log: {len(lines)}")
        if lines:
            first_line = lines[0].strip()
            print(f"First line of search log: {first_line}")
            
            # Check all lines for mentions of simple search
            simple_search_lines = [line for line in lines if "_simple_search" in line]
            print(f"Found {len(simple_search_lines)} lines containing '_simple_search'")
            if simple_search_lines:
                print("First few simple search lines:")
                for line in simple_search_lines[:3]:
                    print(f"  {line.strip()}")
                    
            # Check all lines for mentions of minimax search
            minimax_lines = [line for line in lines if "_minimax_search" in line]
            print(f"Found {len(minimax_lines)} lines containing '_minimax_search'")
            if minimax_lines:
                print("First few minimax lines:")
                for line in minimax_lines[:3]:
                    print(f"  {line.strip()}")
else:
    print("Log file not found")
