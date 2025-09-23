#!/usr/bin/env python3
"""
Simple test to identify the 'pop from empty list' error
"""

import os
import sys
import chess

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine

def simple_test():
    print("Simple V11.5 Test")
    print("==================")
    
    try:
        # Initialize engine
        engine = V7P3REngine()
        print("✅ Engine initialized successfully")
        
        # Test basic position
        board = chess.Board()
        print(f"✅ Board created: {board.fen()}")
        
        # Try a simple search
        print("Attempting search...")
        move, score, search_info = engine.search(board, time_limit=1.0, depth=2)
        print(f"✅ Search completed: {move} (score: {score})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()