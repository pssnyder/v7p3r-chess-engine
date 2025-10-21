#!/usr/bin/env python3
"""
Quick test script for VPR v3.0 engine functionality
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, 'src')

try:
    from vpr import VPREngine
    import chess
    
    print("=== VPR Pure Potential Engine v3.0 Test ===")
    
    # Create engine
    engine = VPREngine()
    print("✓ Engine created successfully")
    
    # Create starting position
    board = chess.Board()
    print(f"✓ Starting position: {board.fen()}")
    
    # Test a quick search
    print("Testing search (3 seconds)...")
    best_move = engine.search(board, time_limit=3.0)
    print(f"✓ Best move found: {best_move}")
    
    # Test engine info
    info = engine.get_engine_info()
    print(f"✓ Engine info: {info}")
    
    print("\n=== VPR v3.0 is working correctly! ===")
    print("Ready for Arena Chess GUI deployment")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()