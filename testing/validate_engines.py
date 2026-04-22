#!/usr/bin/env python3
"""
Quick UCI Engine Validation

WHY THIS EXISTS: Verify engines respond to UCI commands before running full tournament

WHAT IT DOES:
- Tests UCI communication with both engines
- Verifies they can find legal moves
- Checks for basic functionality
"""

import chess
import chess.engine
import sys
from pathlib import Path


def test_engine_uci(engine_cmd: list, name: str) -> bool:
    """Test basic UCI communication with engine"""
    print(f"\nTesting {name}...")
    print(f"  Command: {' '.join(engine_cmd)}")
    
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_cmd) as engine:
            # Test 1: UCI identity
            print(f"  ✓ UCI communication working")
            print(f"    Name: {engine.id.get('name', 'Unknown')}")
            print(f"    Author: {engine.id.get('author', 'Unknown')}")
            
            # Test 2: Can find a move
            board = chess.Board()
            result = engine.play(board, chess.engine.Limit(time=0.5))
            print(f"  ✓ Move generation working (found {result.move})")
            
            # Test 3: Can evaluate position
            info = engine.analyse(board, chess.engine.Limit(time=0.1))
            score = info.get("score")
            print(f"  ✓ Position evaluation working (score: {score})")
            
            return True
            
    except chess.engine.EngineTerminatedError:
        print(f"  ✗ Engine terminated unexpectedly")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("=" * 80)
    print("UCI ENGINE VALIDATION")
    print("=" * 80)
    
    # Test v19.0
    v19_cmd = [sys.executable, "src/v7p3r_uci.py"]
    v19_ok = test_engine_uci(v19_cmd, "V7P3R v19.0")
    
    # Test v18.4
    v18_4_cmd = [sys.executable, r"e:\Programming Stuff\Chess Engines\Tournament Engines\V7P3R\V7P3R_v18.4\src\v7p3r_uci.py"]
    v18_4_ok = test_engine_uci(v18_4_cmd, "V7P3R v18.4")
    
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    if v19_ok and v18_4_ok:
        print("✓ Both engines passed validation")
        print("\nReady to run tournament:")
        print("  python testing/test_v19_vs_v18_4.py")
        return 0
    else:
        print("✗ Engine validation failed")
        if not v19_ok:
            print("  - v19.0 engine not working")
        if not v18_4_ok:
            print("  - v18.4 engine not working")
        return 1


if __name__ == "__main__":
    sys.exit(main())
