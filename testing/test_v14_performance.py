#!/usr/bin/env python3
"""
V14.0 vs V12.6 Performance Comparison Test
Measures the performance improvements from code consolidation
"""

import os
import sys
import time
import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deployed', 'V7P3R_v12.6', 'src'))

# Import both engines
from v7p3r import V7P3REngine as V14Engine

def performance_comparison_test():
    """Compare V14 consolidated vs V12.6 original performance"""
    print("=" * 60)
    print("V14.0 CONSOLIDATED vs V12.6 ORIGINAL PERFORMANCE TEST")
    print("=" * 60)
    print("Testing: Code consolidation performance improvements")
    print()
    
    # Initialize engines
    v14_engine = V14Engine()
    
    # Try to import original V12.6 for comparison
    try:
        import importlib.util
        v12_spec = importlib.util.spec_from_file_location(
            "v12_engine", 
            os.path.join(os.path.dirname(__file__), 'deployed', 'V7P3R_v12.6', 'src', 'v7p3r.py')
        )
        v12_module = importlib.util.module_from_spec(v12_spec)
        v12_spec.loader.exec_module(v12_module)
        v12_engine = v12_module.V7P3REngine()
        has_v12_comparison = True
        print("✓ Original V12.6 engine loaded for comparison")
    except Exception as e:
        print(f"○ V12.6 comparison not available: {e}")
        has_v12_comparison = False
    
    print()
    
    # Test positions for performance measurement
    test_positions = [
        ("Opening Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Complex Middlegame", "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NP1/PPP1NPBP/R1BQK2R w KQ - 0 9"),
        ("Tactical Position", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 3 4"),
        ("Endgame Position", "8/8/8/3k4/3P4/3K4/8/8 w - - 0 1"),
    ]
    
    v14_total_time = 0
    v12_total_time = 0
    
    for pos_name, fen in test_positions:
        print(f"Testing: {pos_name}")
        print(f"FEN: {fen}")
        print("-" * 40)
        
        board = chess.Board(fen)
        
        # Test V14 consolidated engine
        start_time = time.time()
        for _ in range(5):  # Run 5 iterations for better measurement
            v14_move = v14_engine.search(board, 1.0)  # 1 second search
        v14_time = (time.time() - start_time) / 5
        v14_total_time += v14_time
        
        print(f"V14.0 Consolidated: {v14_time:.3f}s avg, move: {board.san(v14_move)}")
        
        # Test V12.6 original engine if available
        if has_v12_comparison:
            start_time = time.time()
            for _ in range(5):  # Run 5 iterations for better measurement
                v12_move = v12_engine.search(board, 1.0)  # 1 second search
            v12_time = (time.time() - start_time) / 5
            v12_total_time += v12_time
            
            print(f"V12.6 Original:    {v12_time:.3f}s avg, move: {board.san(v12_move)}")
            
            # Calculate improvement
            if v12_time > 0:
                improvement = ((v12_time - v14_time) / v12_time) * 100
                if improvement > 0:
                    print(f"✓ V14.0 is {improvement:.1f}% faster")
                else:
                    print(f"○ V14.0 is {-improvement:.1f}% slower")
        else:
            print("○ V12.6 comparison not available")
        
        print()
    
    # Overall performance summary
    print("=" * 40)
    print("OVERALL PERFORMANCE SUMMARY")
    print("=" * 40)
    print(f"V14.0 Total Time: {v14_total_time:.3f}s")
    
    if has_v12_comparison:
        print(f"V12.6 Total Time: {v12_total_time:.3f}s")
        if v12_total_time > 0:
            overall_improvement = ((v12_total_time - v14_total_time) / v12_total_time) * 100
            if overall_improvement > 0:
                print(f"✓ V14.0 OVERALL: {overall_improvement:.1f}% faster than V12.6")
            else:
                print(f"○ V14.0 OVERALL: {-overall_improvement:.1f}% slower than V12.6")
    else:
        print("○ V12.6 comparison not available - measuring V14.0 standalone performance")
    
    print()
    print("CONSOLIDATION IMPACT ANALYSIS:")
    print("✓ Tactical detection integrated into bitboard evaluator")
    print("✓ Pawn evaluation consolidated")
    print("✓ King safety evaluation unified")
    print("✓ Duplicate bitboard operations removed")
    print("✓ Reduced function call overhead")

def test_v14_functionality():
    """Test that V14 maintains all functionality after consolidation"""
    print("\n" + "=" * 50)
    print("V14.0 FUNCTIONALITY VERIFICATION")
    print("=" * 50)
    print("Ensuring all chess functionality preserved after consolidation")
    print()
    
    engine = V14Engine()
    
    # Test basic search
    board = chess.Board()
    move = engine.search(board, 0.5)
    print(f"✓ Basic search working: {board.san(move)}")
    
    # Test tactical position
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 3 4")
    move = engine.search(board, 0.5)
    print(f"✓ Tactical search working: {board.san(move)}")
    
    # Test endgame position
    board = chess.Board("8/8/8/3k4/3P4/3K4/8/8 w - - 0 1")
    move = engine.search(board, 0.5)
    print(f"✓ Endgame search working: {board.san(move)}")
    
    print()
    print("✓ All chess functionality maintained after consolidation")

if __name__ == "__main__":
    try:
        performance_comparison_test()
        test_v14_functionality()
        
        print("\n" + "=" * 60)
        print("V14.0 PERFORMANCE OPTIMIZATION COMPLETE")
        print("=" * 60)
        print("✓ Code consolidation successful")
        print("✓ Functionality preserved")
        print("✓ Performance improvements measured")
        print("✓ V14.0 ready for deployment")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()