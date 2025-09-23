#!/usr/bin/env python3
"""
Debug version with detailed error tracking
"""

import os
import sys
import chess

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine

def debug_test():
    print("Debug V11.5 Test")
    print("=================")
    
    try:
        # Initialize engine
        engine = V7P3REngine()
        print("✅ Engine initialized successfully")
        
        # Test basic position
        board = chess.Board()
        print(f"✅ Board created: {board.fen()}")
        
        # Override the killer moves class to add debugging
        original_pop = list.pop
        
        def debug_pop(self, *args, **kwargs):
            if len(self) == 0:
                print(f"❌ Attempting to pop from empty list: {self}")
                import traceback
                traceback.print_stack()
                raise IndexError("pop from empty list")
            return original_pop(self, *args, **kwargs)
        
        list.pop = debug_pop
        
        # Try a simple search
        print("Attempting search...")
        move, score, search_info = engine.search(board, time_limit=1.0, depth=2)
        print(f"✅ Search completed: {move} (score: {score})")
        
        # Restore original pop
        list.pop = original_pop
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_test()