#!/usr/bin/env python3
"""
Quick test to validate V10.9 tactical pattern detector integration
Tests basic functionality without performance impact assessment
"""

import sys
import time
import chess
sys.path.append('src')

from v7p3r import V7P3REngine
from v7p3r_tactical_pattern_detector import TimeControlAdaptiveTacticalDetector

def test_tactical_detector_standalone():
    """Test the tactical detector in isolation"""
    print("=== Testing Tactical Pattern Detector (Standalone) ===")
    
    detector = TimeControlAdaptiveTacticalDetector()
    board = chess.Board()
    
    # Test basic functionality
    time_remaining = 600000  # 10 minutes
    moves_played = 20
    
    start_time = time.time()
    patterns, score = detector.detect_tactical_patterns(board, time_remaining, moves_played)
    detection_time = (time.time() - start_time) * 1000
    
    print(f"Time remaining: {time_remaining}ms, Moves played: {moves_played}")
    print(f"Patterns found: {len(patterns)}")
    print(f"Tactical score: {score}")
    print(f"Detection time: {detection_time:.2f}ms")
    print(f"Time control detected: {detector.current_time_control}")
    print(f"Tactical budget: {detector._get_tactical_budget():.2f}ms")
    
    # Test time pressure scenario
    print("\n--- Testing Time Pressure Scenario ---")
    time_remaining = 5000  # 5 seconds
    moves_played = 35
    
    start_time = time.time()
    patterns, score = detector.detect_tactical_patterns(board, time_remaining, moves_played)
    detection_time = (time.time() - start_time) * 1000
    
    print(f"Time remaining: {time_remaining}ms, Moves played: {moves_played}")
    print(f"Patterns found: {len(patterns)}")
    print(f"Tactical score: {score}")
    print(f"Detection time: {detection_time:.2f}ms")
    print(f"Time control detected: {detector.current_time_control}")
    print(f"Tactical budget: {detector._get_tactical_budget():.2f}ms")
    
    # Show statistics
    print(f"\nDetection stats: {detector.get_detection_stats()}")

def test_engine_integration():
    """Test the tactical detector integration with V7P3R engine"""
    print("\n=== Testing Engine Integration ===")
    
    engine = V7P3REngine()
    board = chess.Board()
    
    # Update time control info
    engine.update_time_control_info(600000)  # 10 minutes
    
    # Test evaluation with tactical patterns
    start_time = time.time()
    evaluation = engine._evaluate_position(board)
    eval_time = (time.time() - start_time) * 1000
    
    print(f"Position evaluation: {evaluation}")
    print(f"Evaluation time: {eval_time:.2f}ms")
    print(f"Engine time remaining: {engine.current_time_remaining_ms}ms")
    print(f"Engine moves played: {engine.current_moves_played}")
    
    # Test tactical detector stats
    stats = engine.tactical_pattern_detector.get_detection_stats()
    print(f"Tactical detector stats: {stats}")

def test_different_time_controls():
    """Test tactical detector across different time control formats"""
    print("\n=== Testing Different Time Control Formats ===")
    
    detector = TimeControlAdaptiveTacticalDetector()
    board = chess.Board()
    
    test_scenarios = [
        (60000, 15, "1-minute bullet"),
        (180000, 25, "3-minute blitz"),
        (300000, 20, "5-minute rapid"),
        (600000, 15, "10-minute standard"),
        (1800000, 10, "30-minute long")
    ]
    
    for time_ms, moves, description in test_scenarios:
        start_time = time.time()
        patterns, score = detector.detect_tactical_patterns(board, time_ms, moves)
        detection_time = (time.time() - start_time) * 1000
        
        print(f"{description}: {detector.current_time_control}, "
              f"Budget: {detector._get_tactical_budget():.2f}ms, "
              f"Actual: {detection_time:.2f}ms, "
              f"Patterns: {len(patterns)}")

if __name__ == "__main__":
    print("V7P3R v10.9 Tactical Pattern Integration Test")
    print("=" * 50)
    
    try:
        test_tactical_detector_standalone()
        test_engine_integration() 
        test_different_time_controls()
        
        print("\n✅ All tests completed successfully!")
        print("✅ Tactical pattern detector integration is working!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)