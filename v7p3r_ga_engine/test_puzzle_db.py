#!/usr/bin/env python3
"""
Test GA script with minimal configuration to ensure it's working.
"""

import os
import sys

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puzzles.puzzle_db_manager import PuzzleDBManager

def test_puzzle_db():
    """Test the puzzle database connection and data retrieval."""
    print("=== Testing Puzzle Database ===")
    
    # Test with config file
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'puzzle_config.yaml'))
    print(f"Using config: {config_path}")
    
    db = PuzzleDBManager(config_path)
    
    # Test get_puzzles
    puzzles = db.get_puzzles(limit=3)
    print(f"Found {len(puzzles)} puzzles")
    if puzzles:
        print(f"First puzzle FEN: {puzzles[0]['fen']}")
    
    # Test get_random_fens
    fens = db.get_random_fens(count=5)
    print(f"Found {len(fens)} random FENs")
    if fens:
        print(f"First random FEN: {fens[0]}")
    
    return len(fens) > 0

if __name__ == "__main__":
    success = test_puzzle_db()
    print(f"Test successful: {success}")
