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

# Test with different configs to verify the fix works across configurations
configs_to_test = [
    ('centipawn_test_config', 'minimax'),
    ('stability_test_config', 'minimax'),
    ('test_config', 'simple')
]

for config_name, expected_algorithm in configs_to_test:
    print(f"\n===== Testing {config_name} =====")
    
    # Clear the log for each test
    if os.path.exists(log_file_path):
        with open(log_file_path, 'w') as log_file:
            log_file.write("")
    
    # Create engine with the current config
    config = v7p3rConfig(f'configs/{config_name}.json')
    engine_config = config.get_engine_config()
    engine = v7p3rEngine(engine_config=engine_config)
    
    print(f'Config loaded, search_algorithm = {engine_config.get("search_algorithm", "not found")}')
    print(f'Engine created, search_algorithm = {engine.search_algorithm}')
    print(f'Search engine search_algorithm = {engine.search_engine.search_algorithm}')
    
    # Create a new board
    board = chess.Board()
    
    # Make a simple search (just to log the search algorithm being used)
    print(f"Making a search with {config_name}...")
    move = engine.search_engine.search(board, chess.WHITE)
    print(f"Engine returned move: {move}")
    
    # Wait a moment for the log to be written
    time.sleep(1)
    
    # Check the search log to confirm algorithm used
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            first_line = log_file.readline().strip()
            print(f"Search log begins with: {first_line}")
            
            # Check if the expected algorithm is in the log
            if expected_algorithm in first_line:
                print(f"✓ SUCCESS: Found expected algorithm '{expected_algorithm}' in log")
            else:
                print(f"✗ FAILURE: Did not find expected algorithm '{expected_algorithm}' in log")
    else:
        print("Log file not found")
