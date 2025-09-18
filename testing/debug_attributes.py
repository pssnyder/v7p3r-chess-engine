#!/usr/bin/env python3
"""
Quick debug to check engine object attributes
"""
import sys
import os
import chess

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine

def debug_engine_attributes():
    """Check what attributes the engine has"""
    
    engine = V7P3REngine()
    
    print("=== ENGINE ATTRIBUTES ===")
    for attr in dir(engine):
        if not attr.startswith('_'):
            value = getattr(engine, attr)
            print(f"{attr}: {type(value)}")
            if hasattr(value, '__dict__'):
                print(f"  - Methods: {[m for m in dir(value) if not m.startswith('_')]}")
    
    # Test simple evaluation
    board = chess.Board("8/8/8/8/3k4/8/8/3K4 w - - 0 1")
    print(f"\nSimple evaluation: {engine.evaluate(board)}")

if __name__ == "__main__":
    debug_engine_attributes()