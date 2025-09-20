#!/usr/bin/env python3
"""
V7P3R v11.1 Simplified Engine Test
Quick validation of the emergency performance patches
"""

import chess
import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from v7p3r_v11_1_simplified import V7P3REngineSimple
    print("✅ Successfully imported V7P3R v11.1 simplified engine")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_basic_functionality():
    """Test basic engine functionality"""
    print("\n🔧 Testing Basic Functionality")
    print("=" * 40)
    
    engine = V7P3REngineSimple()
    
    # Test 1: Engine creation
    print("✅ Engine created successfully")
    
    # Test 2: Start position evaluation
    board = chess.Board()
    try:
        start_time = time.time()
        move = engine.search(board, time_limit=2.0)
        search_time = time.time() - start_time
        
        print(f"✅ Start position search: {move} in {search_time:.3f}s")
        print(f"   Nodes searched: {engine.search_stats['nodes_searched']}")
        
        # Calculate NPS
        if search_time > 0:
            nps = engine.search_stats['nodes_searched'] / search_time
            print(f"   NPS: {nps:.0f}")
            
            if nps > 5000:
                print("✅ Performance target achieved (>5000 NPS)")
            else:
                print("⚠️  Performance below target")
        
    except Exception as e:
        print(f"❌ Start position search failed: {e}")
        return False
    
    # Test 3: Tactical position
    tactical_fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"
    board = chess.Board(tactical_fen)
    
    try:
        start_time = time.time()
        move = engine.search(board, time_limit=1.5)
        search_time = time.time() - start_time
        
        print(f"✅ Tactical position search: {move} in {search_time:.3f}s")
        print(f"   Nodes searched: {engine.search_stats['nodes_searched']}")
        
    except Exception as e:
        print(f"❌ Tactical position search failed: {e}")
        return False
    
    return True


def test_evaluation_system():
    """Test the bitboard evaluation system"""
    print("\n⚡ Testing Bitboard Evaluation")
    print("=" * 40)
    
    engine = V7P3REngineSimple()
    
    # Test positions
    positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Queen up", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"),
        ("Material advantage", "rnbqkb1r/pppppppp/5n2/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 2 2")
    ]
    
    for name, fen in positions:
        board = chess.Board(fen)
        try:
            eval_score = engine._evaluate_position(board)
            print(f"✅ {name}: {eval_score:.2f}")
        except Exception as e:
            print(f"❌ {name} evaluation failed: {e}")
            return False
    
    return True


def performance_benchmark():
    """Quick performance benchmark"""
    print("\n🚀 Performance Benchmark")
    print("=" * 40)
    
    engine = V7P3REngineSimple()
    board = chess.Board()
    
    # Warm up
    engine.search(board, time_limit=1.0)
    
    # Benchmark
    total_nodes = 0
    total_time = 0
    trials = 3
    
    for i in range(trials):
        engine.new_game()  # Reset stats
        start_time = time.time()
        engine.search(board, time_limit=2.0)
        trial_time = time.time() - start_time
        trial_nodes = engine.search_stats['nodes_searched']
        
        total_nodes += trial_nodes
        total_time += trial_time
        
        print(f"Trial {i+1}: {trial_nodes} nodes in {trial_time:.3f}s = {trial_nodes/trial_time:.0f} NPS")
    
    avg_nps = total_nodes / total_time
    print(f"\n📊 Average Performance: {avg_nps:.0f} NPS")
    
    if avg_nps > 10000:
        print("🏆 EXCELLENT: Performance exceeds 10,000 NPS")
    elif avg_nps > 5000:
        print("✅ GOOD: Performance exceeds 5,000 NPS")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Performance below 5,000 NPS")
    
    return avg_nps


def main():
    """Run all tests"""
    print("🎯 V7P3R v11.1 SIMPLIFIED ENGINE TEST")
    print("=" * 50)
    
    success = True
    
    # Run tests
    success &= test_basic_functionality()
    success &= test_evaluation_system()
    
    # Performance benchmark
    avg_nps = performance_benchmark()
    
    # Final results
    print(f"\n{'='*50}")
    if success and avg_nps > 5000:
        print("🎉 ALL TESTS PASSED! v11.1 simplified engine is ready.")
        print("📈 Performance recovery successful - ready for incremental tuning.")
    elif success:
        print("✅ Functionality tests passed, but performance needs optimization.")
        print("🔧 Consider further simplification or optimization.")
    else:
        print("❌ Tests failed - engine needs debugging before use.")
    
    return success


if __name__ == "__main__":
    main()