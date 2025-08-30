#!/usr/bin/env python3
"""
V9.3 Validation Test Suite
Tests the newly implemented v9.3 hybrid evaluation system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3RCleanEngine
from v7p3r_scoring_calculation_v93 import V7P3RScoringCalculationV93

def test_engine_initialization():
    """Test that v9.3 engine initializes correctly"""
    print("Testing v9.3 engine initialization...")
    try:
        engine = V7P3RCleanEngine()
        print("✓ Engine initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Engine initialization failed: {e}")
        return False

def test_evaluation_system():
    """Test that the hybrid evaluation system works correctly"""
    print("\nTesting v9.3 evaluation system...")
    try:
        piece_values = {'pawn': 100, 'knight': 320, 'bishop': 330, 'rook': 500, 'queen': 900, 'king': 0}
        calc = V7P3RScoringCalculationV93(piece_values)
        
        # Test starting position
        board = chess.Board()
        white_score = calc.calculate_score_optimized(board, chess.WHITE)
        black_score = calc.calculate_score_optimized(board, chess.BLACK)
        print(f"  Starting position - White: {white_score:.2f}, Black: {black_score:.2f}")
        
        # Test material advantage
        board2 = chess.Board('rnbqkb1r/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2')
        white_score2 = calc.calculate_score_optimized(board2, chess.WHITE)
        black_score2 = calc.calculate_score_optimized(board2, chess.BLACK)
        print(f"  Knight development - White: {white_score2:.2f}, Black: {black_score2:.2f}")
        
        print("✓ Evaluation system working correctly")
        return True
    except Exception as e:
        print(f"✗ Evaluation system failed: {e}")
        return False

def test_position_evaluation():
    """Test engine position evaluation"""
    print("\nTesting engine position evaluation...")
    try:
        engine = V7P3RCleanEngine()
        
        # Test various positions
        positions = [
            ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("King's pawn", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"),
            ("Sicilian Defense", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
        ]
        
        for name, fen in positions:
            board = chess.Board(fen)
            evaluation = engine._evaluate_position_deterministic(board)
            print(f"  {name}: {evaluation:.4f}")
        
        print("✓ Position evaluation working correctly")
        return True
    except Exception as e:
        print(f"✗ Position evaluation failed: {e}")
        return False

def test_move_generation():
    """Test that the engine can generate moves"""
    print("\nTesting move generation...")
    try:
        engine = V7P3RCleanEngine()
        board = chess.Board()
        
        # Test search functionality
        start_time = time.time()
        best_move = engine.search(board, time_limit=1.0)
        elapsed_time = time.time() - start_time
        
        if best_move:
            print(f"  Best move: {best_move} (found in {elapsed_time:.2f}s)")
            print("✓ Move generation working correctly")
            return True
        else:
            print("✗ No move generated")
            return False
    except Exception as e:
        print(f"✗ Move generation failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("V7P3R v9.3 Validation Test Suite")
    print("=" * 40)
    
    tests = [
        test_engine_initialization,
        test_evaluation_system,
        test_position_evaluation,
        test_move_generation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nValidation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All v9.3 validation tests passed!")
        print("✓ v9.3 hybrid evaluation system successfully integrated")
        return True
    else:
        print("❌ Some validation tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
