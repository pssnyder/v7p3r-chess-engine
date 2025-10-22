#!/usr/bin/env python3
"""
V7P3R v13.0 Tactical Enhancement Test Suite
Tests the new tactical detection and dynamic evaluation systems
"""

import sys
import os
import time
import chess

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_tactical_detector():
    """Test the tactical detection system"""
    print("=== Testing V13.0 Tactical Detection System ===")
    
    try:
        from v7p3r_tactical_detector import V7P3RTacticalDetector
        
        detector = V7P3RTacticalDetector()
        print("✅ Tactical detector initialized successfully")
        
        # Test 1: Simple pin position
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
        patterns = detector.detect_all_tactical_patterns(board, True)
        print(f"✅ Pin detection test: Found {len(patterns)} patterns")
        
        for pattern in patterns[:3]:  # Show top 3
            print(f"   - {pattern.pattern_type}: {pattern.tactical_value:.1f}cp (forcing: {pattern.forcing_level})")
        
        # Test 2: Fork position
        board = chess.Board("rnbqkb1r/ppp2ppp/3p1n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 0 1")
        patterns = detector.detect_forks(board, True)
        print(f"✅ Fork detection test: Found {len(patterns)} forks")
        
        # Test 3: Performance test
        start_time = time.time()
        for _ in range(100):
            patterns = detector.detect_all_tactical_patterns(board, True)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000
        print(f"✅ Performance test: {avg_time:.2f}ms per position")
        
        # Show profiling stats
        stats = detector.get_profiling_stats()
        print(f"✅ Pattern frequency: {stats}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Tactical detector import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Tactical detector test failed: {e}")
        return False

def test_dynamic_evaluator():
    """Test the dynamic evaluation system"""
    print("\n=== Testing V13.0 Dynamic Evaluation System ===")
    
    try:
        from v7p3r_tactical_detector import V7P3RTacticalDetector
        from v7p3r_dynamic_evaluator import V7P3RDynamicEvaluator
        
        detector = V7P3RTacticalDetector()
        evaluator = V7P3RDynamicEvaluator(detector)
        print("✅ Dynamic evaluator initialized successfully")
        
        # Test 1: Starting position
        board = chess.Board()
        value = evaluator.evaluate_dynamic_position_value(board, True)
        print(f"✅ Starting position evaluation: {value:.1f}cp")
        
        # Test 2: Complex middlegame position
        board = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQR1K1 w - - 0 1")
        value = evaluator.evaluate_dynamic_position_value(board, True)
        print(f"✅ Complex position evaluation: {value:.1f}cp")
        
        # Test 3: Performance test
        start_time = time.time()
        for _ in range(50):
            value = evaluator.evaluate_dynamic_position_value(board, True)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 50 * 1000
        print(f"✅ Performance test: {avg_time:.2f}ms per evaluation")
        
        # Show profiling stats
        stats = evaluator.get_profiling_stats()
        print(f"✅ Evaluation stats: {stats}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Dynamic evaluator import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Dynamic evaluator test failed: {e}")
        return False

def test_v13_engine_integration():
    """Test V13 engine integration"""
    print("\n=== Testing V13.0 Engine Integration ===")
    
    try:
        from v7p3r import V7P3REngine
        
        engine = V7P3REngine()
        print("✅ V13.0 engine initialized successfully")
        
        # Check V13 features are enabled
        print(f"   - Tactical Detection: {engine.ENABLE_TACTICAL_DETECTION}")
        print(f"   - Dynamic Evaluation: {engine.ENABLE_DYNAMIC_EVALUATION}")
        print(f"   - Tal Complexity: {engine.ENABLE_TAL_COMPLEXITY_BONUS}")
        
        # Test basic search functionality
        board = chess.Board()
        start_time = time.time()
        move, score = engine.search(board, depth=4)
        end_time = time.time()
        
        search_time = (end_time - start_time) * 1000
        print(f"✅ Search test: {move} (score: {score:.1f}cp, time: {search_time:.0f}ms)")
        
        # Check NPS (nodes per second)
        if hasattr(engine, 'nodes_searched'):
            nps = engine.nodes_searched / max(end_time - start_time, 0.001)
            print(f"✅ Performance: {nps:.0f} NPS")
        
        return True
        
    except ImportError as e:
        print(f"❌ V13 engine import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ V13 engine test failed: {e}")
        return False

def test_tactical_positions():
    """Test engine on known tactical positions"""
    print("\n=== Testing Tactical Position Recognition ===")
    
    try:
        from v7p3r import V7P3REngine
        
        engine = V7P3REngine()
        
        # Test positions with known tactical themes
        tactical_positions = [
            ("Pin: 8/8/8/3k4/8/3K1Q2/8/r7 w - - 0 1", "Queen pins rook to king"),
            ("Fork: rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1", "Knight fork potential"),
            ("Skewer: 8/8/8/8/8/3k4/8/R3K3 w - - 0 1", "Rook skewer king and piece"),
        ]
        
        for fen, description in tactical_positions:
            print(f"\n📋 Testing: {description}")
            board = chess.Board(fen)
            
            move, score = engine.search(board, depth=5)
            print(f"   Best move: {move} (score: {score:.1f}cp)")
            
            # Check if tactical patterns were detected
            if engine.tactical_detector:
                patterns = engine.tactical_detector.detect_all_tactical_patterns(board, board.turn)
                print(f"   Patterns detected: {len(patterns)}")
                for pattern in patterns[:2]:
                    print(f"      - {pattern.pattern_type}: {pattern.tactical_value:.1f}cp")
        
        return True
        
    except Exception as e:
        print(f"❌ Tactical position test failed: {e}")
        return False

def main():
    """Run all V13.0 tests"""
    print("V7P3R v13.0 Tactical Enhancement Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    # Run all tests
    if test_tactical_detector():
        tests_passed += 1
    
    if test_dynamic_evaluator():
        tests_passed += 1
    
    if test_v13_engine_integration():
        tests_passed += 1
    
    if test_tactical_positions():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All V13.0 tests passed! Tactical enhancement ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check implementations.")
        return 1

if __name__ == "__main__":
    sys.exit(main())