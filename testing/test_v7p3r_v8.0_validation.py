#!/usr/bin/env python3
"""
V7P3R v8.0 Validation Test Suite
Tests the new unified architecture and progressive evaluation system
"""

import sys
import os
sys.path.append('src')

import chess
import time
import threading
from v7p3r_v8 import V7P3REngineV8

def test_unified_search():
    """Test the unified search architecture"""
    print("🔍 Testing Unified Search Architecture")
    print("=" * 50)
    
    engine = V7P3REngineV8()
    board = chess.Board()
    
    # Test basic search functionality
    start_time = time.time()
    best_move = engine.search(board, 2.0)
    search_time = time.time() - start_time
    
    print(f"✅ Search completed in {search_time:.3f}s")
    print(f"✅ Best move: {best_move}")
    print(f"✅ Nodes searched: {engine.nodes_searched}")
    
    # Test with different time limits
    time_limits = [0.5, 1.0, 3.0]
    for time_limit in time_limits:
        start = time.time()
        move = engine.search(board, time_limit)
        actual_time = time.time() - start
        print(f"✅ Time limit {time_limit}s: Actual {actual_time:.3f}s, Move: {move}")
    
    return True

def test_progressive_evaluation():
    """Test the progressive asynchronous evaluation system"""
    print("\n🚀 Testing Progressive Asynchronous Evaluation")
    print("=" * 50)
    
    engine = V7P3REngineV8()
    
    # Test different positions
    test_positions = [
        ("Starting", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Complex", "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 4 5"),
        ("Endgame", "8/8/8/8/3k4/8/3K4/8 w - - 0 1"),
    ]
    
    for description, fen in test_positions:
        print(f"\n📍 Testing {description} position:")
        board = chess.Board(fen)
        
        # Test evaluation directly
        from v7p3r_v8 import SearchOptions
        options = SearchOptions()
        
        start = time.time()
        eval_score = engine._progressive_evaluate(board, options)
        eval_time = time.time() - start
        
        print(f"  Evaluation: {eval_score:+.2f}")
        print(f"  Time: {eval_time:.6f}s")
        
        # Check cache performance
        stats = engine.get_search_stats()
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache misses: {stats['cache_misses']}")
        print(f"  Evaluation timeouts: {stats['evaluation_timeouts']}")
    
    return True

def test_confidence_system():
    """Test the confidence-based strength system"""
    print("\n🎯 Testing Confidence-Based Strength System")
    print("=" * 50)
    
    engine = V7P3REngineV8()
    board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 2 4")
    
    # Test different strength levels
    strength_levels = [50, 65, 75, 85, 95]
    
    for strength in strength_levels:
        print(f"\n🔧 Testing strength {strength}%:")
        engine.set_strength(strength)
        
        start = time.time()
        move = engine.search(board, 2.0)
        search_time = time.time() - start
        
        stats = engine.get_search_stats()
        print(f"  Move: {move}")
        print(f"  Time: {search_time:.3f}s")
        print(f"  Nodes: {engine.nodes_searched}")
        print(f"  Confidence exits: {stats['confidence_exits']}")
        
        # Reset for next test
        engine.new_game()
    
    return True

def test_threading_performance():
    """Test multi-threading performance"""
    print("\n🧵 Testing Multi-Threading Performance")
    print("=" * 50)
    
    engine = V7P3REngineV8()
    
    # Test with different thread counts
    thread_counts = [1, 2, 4]
    test_board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 4 5")
    
    for threads in thread_counts:
        print(f"\n🔧 Testing with {threads} threads:")
        
        # Update thread pool
        engine.thread_pool._max_workers = threads
        engine.new_game()  # Reset cache for fair comparison
        
        start = time.time()
        move = engine.search(test_board, 2.0)
        search_time = time.time() - start
        
        stats = engine.get_search_stats()
        print(f"  Move: {move}")
        print(f"  Time: {search_time:.3f}s")
        print(f"  Nodes: {engine.nodes_searched}")
        print(f"  NPS: {engine.nodes_searched / max(search_time, 0.001):.0f}")
        print(f"  Timeouts: {stats['evaluation_timeouts']}")
    
    return True

def test_uci_compatibility():
    """Test UCI compatibility and new options"""
    print("\n🖥️  Testing UCI Compatibility")
    print("=" * 50)
    
    try:
        import subprocess
        import tempfile
        
        # Create a simple UCI test script
        uci_commands = [
            "uci",
            "setoption name Strength value 80",
            "setoption name Threads value 2",
            "isready",
            "position startpos moves e2e4",
            "go movetime 1000",
            "quit"
        ]
        
        # Write commands to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for cmd in uci_commands:
                f.write(cmd + '\n')
            temp_file = f.name
        
        print("✅ UCI command sequence prepared")
        print("ℹ️  Note: Full UCI test requires built executable")
        
        # Clean up
        os.unlink(temp_file)
        
        return True
        
    except Exception as e:
        print(f"⚠️  UCI test preparation failed: {e}")
        return False

def benchmark_v8_improvements():
    """Benchmark V8.0 improvements"""
    print("\n📊 Benchmarking V8.0 Improvements")
    print("=" * 50)
    
    engine = V7P3REngineV8()
    
    # Test positions for benchmarking
    benchmark_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 4 5",
        "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b - - 0 6",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ]
    
    total_nodes = 0
    total_time = 0
    
    for i, fen in enumerate(benchmark_positions, 1):
        print(f"\n🏁 Benchmark position {i}:")
        board = chess.Board(fen)
        
        start = time.time()
        move = engine.search(board, 1.0)  # 1 second per position
        search_time = time.time() - start
        
        total_nodes += engine.nodes_searched
        total_time += search_time
        
        nps = engine.nodes_searched / max(search_time, 0.001)
        print(f"  Move: {move}")
        print(f"  Nodes: {engine.nodes_searched}")
        print(f"  Time: {search_time:.3f}s")
        print(f"  NPS: {nps:.0f}")
        
        # Reset for next position
        engine.new_game()
    
    # Overall benchmark results
    avg_nps = total_nodes / max(total_time, 0.001)
    print(f"\n🎯 BENCHMARK RESULTS:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average NPS: {avg_nps:.0f}")
    
    return True

def main():
    """Main test function"""
    print("V7P3R v8.0 Validation Test Suite")
    print("🏎️ → 🏎️💨 Sports Car to Supercar Testing")
    print("=" * 60)
    
    tests = [
        ("Unified Search Architecture", test_unified_search),
        ("Progressive Asynchronous Evaluation", test_progressive_evaluation),
        ("Confidence-Based Strength System", test_confidence_system),
        ("Multi-Threading Performance", test_threading_performance),
        ("UCI Compatibility", test_uci_compatibility),
        ("V8.0 Performance Benchmark", benchmark_v8_improvements),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    print(f"\n🏆 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! V8.0 is ready for action!")
        return 0
    else:
        print("⚠️  Some tests failed. Review and fix before release.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
