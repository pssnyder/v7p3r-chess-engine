#!/usr/bin/env python3
"""
V7P3R v10.9 Performance Validation Test
Compares tactical pattern integration performance vs expected v10.8 baseline
Tests across multiple time control formats and tactical scenarios
"""

import sys
import time
import chess
import chess.pgn
import io
from statistics import mean, stdev
sys.path.append('src')

from v7p3r import V7P3REngine

# Test positions with tactical opportunities
TACTICAL_TEST_POSITIONS = [
    # Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    
    # Tactical puzzle position with hanging pieces
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
    
    # Fork opportunity  
    "rnbqkb1r/ppp2ppp/5n2/3pp3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 0 1",
    
    # Pin opportunity
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
    
    # Mid-game tactical position
    "r2qk2r/ppp2ppp/2n1bn2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w kq - 0 1",
    
    # Endgame with tactics
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
]

def benchmark_position_evaluation(engine, positions, time_control_ms, iterations=10):
    """Benchmark position evaluation across multiple positions and iterations"""
    times = []
    evaluations = []
    
    # Set up time control
    engine.update_time_control_info(time_control_ms)
    
    for pos in positions:
        board = chess.Board(pos)
        engine.current_moves_played = 25  # Mid-game simulation
        
        for _ in range(iterations):
            start_time = time.time()
            evaluation = engine._evaluate_position(board)
            eval_time = (time.time() - start_time) * 1000
            
            times.append(eval_time)
            evaluations.append(evaluation)
    
    return {
        'avg_time_ms': mean(times),
        'max_time_ms': max(times),
        'min_time_ms': min(times),
        'std_dev_ms': stdev(times) if len(times) > 1 else 0,
        'total_evaluations': len(times),
        'evaluations': evaluations,
        'times': times
    }

def benchmark_search_performance(engine, positions, time_limit_seconds, time_control_ms):
    """Benchmark search performance (move selection)"""
    search_times = []
    nodes_searched = []
    
    # Set up time control
    engine.update_time_control_info(time_control_ms)
    
    for pos in positions:
        board = chess.Board(pos)
        
        start_time = time.time()
        move = engine.search(board, time_limit=time_limit_seconds)
        search_time = (time.time() - start_time) * 1000
        
        search_times.append(search_time)
        nodes_searched.append(engine.nodes_searched)
        
        # Calculate NPS (nodes per second)
        nps = engine.nodes_searched / (search_time / 1000) if search_time > 0 else 0
        
        print(f"Position: {pos[:30]}... -> Move: {move}, "
              f"Time: {search_time:.1f}ms, Nodes: {engine.nodes_searched}, NPS: {nps:.0f}")
    
    return {
        'avg_search_time_ms': mean(search_times),
        'max_search_time_ms': max(search_times),
        'total_nodes': sum(nodes_searched),
        'avg_nodes': mean(nodes_searched),
        'avg_nps': mean([nodes / (time_ms / 1000) for nodes, time_ms in zip(nodes_searched, search_times) if time_ms > 0])
    }

def test_tactical_detector_performance():
    """Test tactical detector performance in isolation"""
    print("=== Tactical Detector Performance Test ===")
    
    from v7p3r_tactical_pattern_detector import TimeControlAdaptiveTacticalDetector
    detector = TimeControlAdaptiveTacticalDetector()
    
    time_controls = [
        (60000, "1-minute bullet"),
        (180000, "3-minute blitz"), 
        (300000, "5-minute rapid"),
        (600000, "10-minute standard"),
        (1800000, "30-minute long")
    ]
    
    for time_ms, description in time_controls:
        position_times = []
        
        for pos in TACTICAL_TEST_POSITIONS:
            board = chess.Board(pos)
            
            start_time = time.time()
            patterns, score = detector.detect_tactical_patterns(board, time_ms, 25)
            detection_time = (time.time() - start_time) * 1000
            
            position_times.append(detection_time)
        
        avg_time = mean(position_times)
        max_time = max(position_times)
        budget = detector._get_tactical_budget()
        
        print(f"{description}: Budget={budget:.2f}ms, Avg={avg_time:.2f}ms, Max={max_time:.2f}ms")
        
        # Check if within budget
        if max_time <= budget * 1.5:  # Allow 50% buffer
            print(f"  ✅ Within budget")
        else:
            print(f"  ⚠️  Exceeds budget by {max_time - budget:.2f}ms")

def run_performance_validation():
    """Run comprehensive performance validation"""
    print("V7P3R v10.9 Performance Validation")
    print("=" * 50)
    
    # Test tactical detector performance first
    test_tactical_detector_performance()
    
    print("\n=== Engine Integration Performance ===")
    
    engine = V7P3REngine()
    
    # Test evaluation performance across time controls
    time_controls = [
        (60000, "1-minute bullet"),
        (300000, "5-minute rapid"),
        (600000, "10-minute standard"),
        (1800000, "30-minute long")
    ]
    
    print("\n--- Position Evaluation Performance ---")
    for time_ms, description in time_controls:
        results = benchmark_position_evaluation(engine, TACTICAL_TEST_POSITIONS, time_ms, iterations=5)
        
        print(f"{description}: Avg={results['avg_time_ms']:.2f}ms, "
              f"Max={results['max_time_ms']:.2f}ms, "
              f"StdDev={results['std_dev_ms']:.2f}ms")
        
        # Performance expectation: evaluation should be under 20ms average
        if results['avg_time_ms'] <= 20.0:
            print(f"  ✅ Performance acceptable")
        else:
            print(f"  ⚠️  Performance concern: {results['avg_time_ms']:.2f}ms > 20ms")
    
    print("\n--- Search Performance (1-second searches) ---")
    for time_ms, description in time_controls[:3]:  # Only test faster time controls for search
        print(f"\n{description}:")
        results = benchmark_search_performance(engine, TACTICAL_TEST_POSITIONS[:3], 1.0, time_ms)
        
        print(f"  Avg search time: {results['avg_search_time_ms']:.1f}ms")
        print(f"  Avg nodes: {results['avg_nodes']:.0f}")
        print(f"  Avg NPS: {results['avg_nps']:.0f}")
        
        # Performance expectation: should maintain reasonable NPS (>1000)
        if results['avg_nps'] >= 1000:
            print(f"  ✅ NPS acceptable")
        else:
            print(f"  ⚠️  Low NPS: {results['avg_nps']:.0f}")
    
    # Test tactical detector statistics
    print(f"\n--- Tactical Detector Statistics ---")
    stats = engine.tactical_pattern_detector.get_detection_stats()
    print(f"Total calls: {stats['total_calls']}")
    print(f"Patterns found: {stats['patterns_found']}")
    print(f"Budget exceeded: {stats['time_budget_exceeded']}")
    print(f"Emergency fallbacks: {stats['emergency_fallbacks']}")
    print(f"Avg detection time: {stats['avg_detection_time_ms']:.2f}ms")
    
    if stats['time_budget_exceeded'] == 0:
        print("✅ No time budget exceeded")
    else:
        print(f"⚠️  Time budget exceeded {stats['time_budget_exceeded']} times")

if __name__ == "__main__":
    try:
        run_performance_validation()
        print("\n✅ Performance validation completed!")
        
    except Exception as e:
        print(f"\n❌ Performance validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)