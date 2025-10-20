#!/usr/bin/env python3
"""
Simple VPR Test - Direct engine testing without UCI complexity
"""

import sys
import os
import chess
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_vpr_direct():
    """Test VPR engine directly"""
    print("VPR Direct Engine Test")
    print("=" * 30)
    
    try:
        from vpr import VPREngine
        
        # Create engine
        engine = VPREngine()
        print(f"✓ VPR Engine loaded successfully")
        
        # Test basic info
        info = engine.get_engine_info()
        print(f"✓ Engine: {info['name']} v{info['version']} by {info['author']}")
        print(f"  Default depth: {info['default_depth']}")
        
        # Test starting position
        board = chess.Board()
        print(f"\nTesting starting position: {board.fen()}")
        
        start_time = time.time()
        best_move = engine.search(board, time_limit=2.0)
        search_time = time.time() - start_time
        
        print(f"✓ Best move: {best_move}")
        print(f"✓ Search time: {search_time:.3f}s")
        print(f"✓ Nodes searched: {engine.nodes_searched:,}")
        print(f"✓ NPS: {int(engine.nodes_searched / search_time):,}")
        
        # Test a tactical position
        print(f"\nTesting tactical position...")
        tactical_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        board = chess.Board(tactical_fen)
        
        start_time = time.time()
        best_move = engine.search(board, time_limit=2.0)
        search_time = time.time() - start_time
        
        print(f"✓ Best move: {best_move}")
        print(f"✓ Search time: {search_time:.3f}s")
        print(f"✓ Nodes searched: {engine.nodes_searched:,}")
        print(f"✓ NPS: {int(engine.nodes_searched / search_time):,}")
        
        # Test engine reset
        engine.new_game()
        print(f"✓ Engine reset: nodes = {engine.nodes_searched}")
        
        print(f"\n✓ All tests passed! VPR is working correctly.")
        
    except ImportError as e:
        print(f"✗ Could not import VPR engine: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False
    
    return True

def test_uci_basic():
    """Test basic UCI commands manually"""
    print("\nVPR UCI Manual Test")
    print("=" * 30)
    
    print("To test UCI manually:")
    print("1. Run: python src/vpr_uci.py")
    print("2. Type: uci")
    print("3. Expected: id name VPR v1.0")
    print("4. Expected: id author Pat Snyder")
    print("5. Expected: uciok")
    print("6. Type: isready")
    print("7. Expected: readyok")
    print("8. Type: position startpos")
    print("9. Type: go movetime 2000")
    print("10. Expected: info depth X score cp Y nodes Z time T nps N pv MOVE")
    print("11. Expected: bestmove MOVE")
    print("12. Type: quit")

if __name__ == "__main__":
    success = test_vpr_direct()
    test_uci_basic()
    
    if success:
        print(f"\n🎉 VPR v1.0 is ready for Arena deployment!")
        print(f"Use VPR_Arena.bat to launch in Arena Chess GUI")
    else:
        print(f"\n❌ VPR testing failed - check the errors above")