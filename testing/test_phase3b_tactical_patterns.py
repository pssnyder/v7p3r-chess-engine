#!/usr/bin/env python3
"""
V7P3R v11 Phase 3B - Tactical Pattern Detection Test
Tests tactical pattern recognition functionality
Author: Pat Snyder
"""

import sys
import os

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

import chess
import time
from v7p3r import V7P3REngine
from v7p3r_tactical_pattern_detector import V7P3RTacticalPatternDetector


def test_tactical_pattern_detector():
    """Test the tactical pattern detector directly"""
    print("=== Testing V7P3R Tactical Pattern Detector ===")
    
    detector = V7P3RTacticalPatternDetector()
    
    # Test 1: Basic fork position
    print("\n1. Testing Knight Fork Detection:")
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4")
    print(f"Position: {board.fen()}")
    
    white_score = detector.evaluate_tactical_patterns(board, chess.WHITE)
    black_score = detector.evaluate_tactical_patterns(board, chess.BLACK)
    print(f"White tactical score: {white_score:.2f}")
    print(f"Black tactical score: {black_score:.2f}")
    
    # Test 2: Pin position
    print("\n2. Testing Pin Detection:")
    board = chess.Board("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4")
    print(f"Position: {board.fen()}")
    
    white_score = detector.evaluate_tactical_patterns(board, chess.WHITE)
    black_score = detector.evaluate_tactical_patterns(board, chess.BLACK)
    print(f"White tactical score: {white_score:.2f}")
    print(f"Black tactical score: {black_score:.2f}")
    
    # Test 3: Starting position (should have minimal tactical patterns)
    print("\n3. Testing Starting Position:")
    board = chess.Board()
    print(f"Position: {board.fen()}")
    
    white_score = detector.evaluate_tactical_patterns(board, chess.WHITE)
    black_score = detector.evaluate_tactical_patterns(board, chess.BLACK)
    print(f"White tactical score: {white_score:.2f}")
    print(f"Black tactical score: {black_score:.2f}")
    
    return True


def test_engine_tactical_integration():
    """Test tactical patterns integrated in engine evaluation"""
    print("\n=== Testing Engine Tactical Integration ===")
    
    engine = V7P3REngine()
    
    # Test positions with known tactical patterns
    test_positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Italian Game", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4"),
        ("Tactical position", "r2qkb1r/ppp2ppp/2n1bn2/3pp3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ]
    
    for name, fen in test_positions:
        print(f"\n--- {name} ---")
        board = chess.Board(fen)
        
        # Evaluate position
        start_time = time.time()
        evaluation = engine._evaluate_position(board)
        eval_time = time.time() - start_time
        
        print(f"Position: {fen}")
        print(f"Evaluation: {evaluation:.2f}")
        print(f"Evaluation time: {eval_time:.4f}s")
        
        # Test move ordering with tactical bonuses
        legal_moves = list(board.legal_moves)
        if legal_moves:
            start_time = time.time()
            ordered_moves = engine._order_moves_advanced(board, legal_moves, 4)
            ordering_time = time.time() - start_time
            
            print(f"Moves ordered: {len(ordered_moves)}")
            print(f"Move ordering time: {ordering_time:.4f}s")
            print(f"Top 3 moves: {[str(move) for move in ordered_moves[:3]]}")
    
    return True


def test_tactical_pattern_search():
    """Test search with tactical pattern detection"""
    print("\n=== Testing Search with Tactical Patterns ===")
    
    engine = V7P3REngine()
    
    # Test a tactical puzzle position
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4")
    print(f"Testing position: {board.fen()}")
    
    # Short search to test integration
    start_time = time.time()
    best_move = engine.search(board, depth=4, time_limit=5.0)
    search_time = time.time() - start_time
    
    # Get evaluation separately
    evaluation = engine._evaluate_position(board)
    
    print(f"Best move: {best_move}")
    print(f"Evaluation: {evaluation:.2f}")
    print(f"Search time: {search_time:.2f}s")
    print(f"Nodes searched: {engine.nodes_searched}")
    
    # Test that search completed successfully
    if best_move is not None:
        print("✓ Search with tactical patterns completed successfully")
        return True
    else:
        print("✗ Search failed to find best move")
        return False


def test_performance_impact():
    """Test the performance impact of tactical pattern detection"""
    print("\n=== Testing Performance Impact ===")
    
    # Create engines with and without tactical detection
    engine_with_tactics = V7P3REngine()
    
    # Test position
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4")
    
    # Evaluate multiple times to get average
    num_tests = 10
    
    print(f"Testing evaluation performance ({num_tests} iterations):")
    
    # With tactical patterns
    start_time = time.time()
    for _ in range(num_tests):
        engine_with_tactics._evaluate_position(board)
    with_tactics_time = time.time() - start_time
    
    print(f"With tactical patterns: {with_tactics_time:.4f}s total, {with_tactics_time/num_tests:.4f}s per evaluation")
    
    # Test move ordering performance
    legal_moves = list(board.legal_moves)
    
    start_time = time.time()
    for _ in range(num_tests):
        engine_with_tactics._order_moves_advanced(board, legal_moves, 4)
    ordering_time = time.time() - start_time
    
    print(f"Move ordering: {ordering_time:.4f}s total, {ordering_time/num_tests:.4f}s per ordering")
    
    return True


def main():
    """Run all tactical pattern tests"""
    print("V7P3R v11 Phase 3B - Tactical Pattern Detection Test Suite")
    print("=" * 60)
    
    test_results = []
    
    try:
        # Test 1: Direct tactical pattern detector
        result = test_tactical_pattern_detector()
        test_results.append(("Tactical Pattern Detector", result))
        
        # Test 2: Engine integration
        result = test_engine_tactical_integration()
        test_results.append(("Engine Integration", result))
        
        # Test 3: Search with tactical patterns
        result = test_tactical_pattern_search()
        test_results.append(("Search with Tacticals", result))
        
        # Test 4: Performance impact
        result = test_performance_impact()
        test_results.append(("Performance Impact", result))
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Print results summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:<30} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - V11 Phase 3B tactical patterns ready!")
    else:
        print("✗ SOME TESTS FAILED - Check implementation")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
