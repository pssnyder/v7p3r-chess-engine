#!/usr/bin/env python3
"""
Quick test of v11 integration fixes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3REngine

def test_integration_fixes():
    """Quick test of strategic/tactical integration fixes"""
    print("TESTING V7P3R v11 INTEGRATION FIXES")
    print("=" * 50)
    
    engine = V7P3REngine()
    
    print("1. Strategic Database Integration:")
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    try:
        strategic_bonus = engine.strategic_database.get_strategic_evaluation_bonus(board)
        print(f"   Strategic bonus: {strategic_bonus:.3f} ✅")
    except Exception as e:
        print(f"   Strategic bonus: ERROR - {e} ❌")
    
    print("\n2. Posture Assessment:")
    try:
        posture = engine.posture_assessor.assess_position_posture(board)
        print(f"   Posture: {posture} ✅")
    except Exception as e:
        print(f"   Posture: ERROR - {e} ❌")
    
    print("\n3. Lightweight Defense:")
    try:
        defense_score = engine.lightweight_defense.quick_defensive_assessment(board)
        print(f"   Defense score: {defense_score:.3f} ✅")
    except Exception as e:
        print(f"   Defense score: ERROR - {e} ❌")
    
    print("\n4. Middlegame Depth Test:")
    board = chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    
    print("   Testing depth 6 achievement...")
    engine.default_depth = 6
    start_time = time.time()
    
    try:
        move = engine.search(board, time_limit=25.0)
        elapsed = time.time() - start_time
        
        if elapsed <= 25.0:
            print(f"   Depth 6: {elapsed:6.2f}s ✅ ACHIEVED")
        else:
            print(f"   Depth 6: {elapsed:6.2f}s ❌ TIMEOUT")
            
    except Exception as e:
        print(f"   Depth 6: ERROR - {e} ❌")

if __name__ == "__main__":
    test_integration_fixes()