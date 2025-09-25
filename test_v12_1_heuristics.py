#!/usr/bin/env python3
"""
Test V7P3R v12.1 Heuristic Improvements
=====================================

Tests the new v12.1 heuristics:
1. Increased castling bonus (+40/+30)
2. Opening center control bonus
3. Development penalties for knights/bishops on starting squares
4. Stricter draw prevention logic

Results are logged to validate that the improvements work as expected.
"""

import sys
import os
import chess
import time
import json
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from v7p3r import V7P3REngine
    from v7p3r_bitboard_evaluator import V7P3RBitboardEvaluator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the v7p3r-chess-engine directory")
    sys.exit(1)

def test_castling_bonus():
    """Test that castling receives proper bonus in v12.1"""
    print("Testing Castling Bonus...")
    
    # Standard piece values
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    evaluator = V7P3RBitboardEvaluator(piece_values)
    
    # Position before castling
    board_before = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    score_before = evaluator.evaluate_bitboard(board_before, chess.WHITE)
    
    # Position after white castles kingside
    board_after = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4")
    score_after = evaluator.evaluate_bitboard(board_after, chess.WHITE)
    
    castling_bonus = score_after - score_before
    print(f"  Castling bonus detected: {castling_bonus}")
    print(f"  Expected bonus: ~40 (should be positive and significant)")
    
    return castling_bonus > 30  # Should be at least 30

def test_center_control():
    """Test opening center control bonus for minor pieces"""
    print("Testing Center Control Bonus...")
    
    # Standard piece values
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    evaluator = V7P3RBitboardEvaluator(piece_values)
    
    # Position with knights on rim
    board_rim = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    score_rim = evaluator.evaluate_bitboard(board_rim, chess.WHITE)
    
    # Position with knights towards center (Italian game setup)
    board_center = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    score_center = evaluator.evaluate_bitboard(board_center, chess.WHITE)
    
    center_bonus = score_center - score_rim
    print(f"  Center control improvement: {center_bonus}")
    print(f"  Expected: positive (developed pieces should score better)")
    
    return center_bonus > 0

def test_development_penalties():
    """Test penalties for undeveloped pieces"""
    print("Testing Development Penalties...")
    
    # Standard piece values
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    evaluator = V7P3RBitboardEvaluator(piece_values)
    
    # Starting position (all pieces undeveloped)
    board_start = chess.Board()
    score_start = evaluator.evaluate_bitboard(board_start, chess.WHITE)
    
    # After developing some pieces
    board_dev = chess.Board("rnbqkb1r/pppppppp/5n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 2")
    score_dev = evaluator.evaluate_bitboard(board_dev, chess.WHITE)
    
    development_bonus = score_dev - score_start
    print(f"  Development improvement: {development_bonus}")
    print(f"  Expected: positive (developed position should score better)")
    
    return development_bonus > 0

def test_draw_prevention():
    """Test stricter draw prevention logic"""
    print("Testing Draw Prevention...")
    
    # Standard piece values
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    evaluator = V7P3RBitboardEvaluator(piece_values)
    
    # Create a position approaching fifty-move rule
    board = chess.Board("8/8/8/8/8/8/k7/K7 w - - 35 100")  # 35 halfmoves without capture/pawn move
    score_near_draw = evaluator.evaluate_bitboard(board, chess.WHITE)
    
    # Same position with normal halfmove clock
    board_normal = chess.Board("8/8/8/8/8/8/k7/K7 w - - 5 100")
    score_normal = evaluator.evaluate_bitboard(board_normal, chess.WHITE)
    
    draw_penalty = score_normal - score_near_draw
    print(f"  Draw penalty applied: {draw_penalty}")
    print(f"  Expected: positive (high halfmove clock should be penalized)")
    
    return draw_penalty > 0

def test_basic_functionality():
    """Test that basic engine functionality still works"""
    print("Testing Basic Engine Functionality...")
    
    try:
        engine = V7P3REngine()
        board = chess.Board()
        
        # Test that engine can make a move using search method
        result = engine.search(board, time_limit=1.0)
        
        # Handle if result is directly a move or a result object
        if isinstance(result, chess.Move):
            move = result
            if move and move in board.legal_moves:
                print(f"  Engine suggests: {move} (valid)")
                return True
        elif result and hasattr(result, 'best_move'):
            move = result.best_move
            if move and move in board.legal_moves:
                print(f"  Engine suggests: {move} (valid)")
                return True
        
        print(f"  Engine failed to suggest valid move")
        return False
            
    except Exception as e:
        print(f"  Engine error: {e}")
        return False

def run_all_tests():
    """Run all heuristic tests and generate report"""
    print("V7P3R v12.1 Heuristic Test Suite")
    print("=================================\n")
    
    test_results = {}
    test_start = time.time()
    
    # Run individual tests
    tests = [
        ("castling_bonus", test_castling_bonus),
        ("center_control", test_center_control),
        ("development_penalties", test_development_penalties),
        ("draw_prevention", test_draw_prevention),
        ("basic_functionality", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            test_results[test_name] = {
                "passed": result,
                "duration": round(duration, 3)
            }
            
            status = "PASS" if result else "FAIL"
            print(f"  Status: {status} ({duration:.3f}s)\n")
            
            if result:
                passed += 1
                
        except Exception as e:
            test_results[test_name] = {
                "passed": False,
                "error": str(e),
                "duration": 0
            }
            print(f"  Status: ERROR - {e}\n")
    
    # Generate summary
    total_time = time.time() - test_start
    
    print("Test Summary")
    print("============")
    print(f"Passed: {passed}/{total}")
    print(f"Total time: {total_time:.3f}s")
    
    if passed == total:
        print("🎉 All v12.1 heuristics working correctly!")
    else:
        print("⚠️  Some tests failed - review results above")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"v12_1_heuristics_test_{timestamp}.json"
    
    final_results = {
        "version": "v12.1",
        "timestamp": timestamp,
        "summary": {
            "passed": passed,
            "total": total,
            "duration": round(total_time, 3)
        },
        "tests": test_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)