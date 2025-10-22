#!/usr/bin/env python3
"""
V7P3R v13.0 Tactical Pattern Profiling
Tests which tactical patterns fire most frequently to optimize performance
"""

import sys
import os
import time
import chess
import random

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r_tactical_detector import V7P3RTacticalDetector
from v7p3r_dynamic_evaluator import V7P3RDynamicEvaluator

def generate_test_positions(count=100):
    """Generate diverse test positions"""
    positions = []
    
    # Starting position
    positions.append(chess.Board())
    
    # Generate random positions by playing random moves
    for _ in range(count - 1):
        board = chess.Board()
        moves_count = random.randint(5, 25)
        
        for _ in range(moves_count):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            board.push(move)
            
        positions.append(board)
    
    return positions

def profile_tactical_patterns():
    """Profile which tactical patterns fire most frequently"""
    print("=== V13.0 Tactical Pattern Profiling ===")
    
    detector = V7P3RTacticalDetector()
    evaluator = V7P3RDynamicEvaluator(detector)
    
    # Generate test positions
    print("Generating test positions...")
    positions = generate_test_positions(50)  # Smaller set for faster testing
    print(f"Generated {len(positions)} test positions")
    
    # Profile tactical detection
    total_patterns = 0
    total_time = 0
    
    start_time = time.time()
    
    for i, board in enumerate(positions):
        if i % 10 == 0:
            print(f"Processing position {i+1}/{len(positions)}...")
            
        # Test both sides
        for color in [True, False]:
            pos_start = time.time()
            patterns = detector.detect_all_tactical_patterns(board, color)
            pos_time = time.time() - pos_start
            
            total_patterns += len(patterns)
            total_time += pos_time
    
    end_time = time.time()
    
    # Results
    total_test_time = end_time - start_time
    avg_time_per_position = total_test_time / (len(positions) * 2) * 1000
    avg_patterns_per_position = total_patterns / (len(positions) * 2)
    
    print(f"\n📊 Tactical Detection Results:")
    print(f"   Total patterns found: {total_patterns}")
    print(f"   Average per position: {avg_patterns_per_position:.1f}")
    print(f"   Average time per position: {avg_time_per_position:.2f}ms")
    
    # Pattern frequency breakdown
    stats = detector.get_profiling_stats()
    print(f"\n📈 Pattern Frequency Breakdown:")
    total_fired = sum(stats.values())
    for pattern_type, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / max(total_fired, 1)) * 100
        print(f"   {pattern_type:20s}: {count:4d} ({percentage:4.1f}%)")
    
    return stats

def profile_dynamic_evaluation():
    """Profile dynamic evaluation performance"""
    print("\n=== V13.0 Dynamic Evaluation Profiling ===")
    
    detector = V7P3RTacticalDetector()
    evaluator = V7P3RDynamicEvaluator(detector)
    
    # Generate fewer positions for evaluation testing (it's more expensive)
    positions = generate_test_positions(20)
    
    total_time = 0
    
    start_time = time.time()
    
    for i, board in enumerate(positions):
        print(f"Evaluating position {i+1}/{len(positions)}...")
        
        # Test both sides
        for color in [True, False]:
            pos_start = time.time()
            value = evaluator.evaluate_dynamic_position_value(board, color)
            pos_time = time.time() - pos_start
            total_time += pos_time
    
    end_time = time.time()
    
    total_test_time = end_time - start_time
    avg_time_per_evaluation = total_test_time / (len(positions) * 2) * 1000
    
    print(f"\n📊 Dynamic Evaluation Results:")
    print(f"   Average time per evaluation: {avg_time_per_evaluation:.2f}ms")
    
    # Get evaluation stats
    stats = evaluator.get_profiling_stats()
    print(f"   Total evaluations: {stats['evaluations']}")
    print(f"   Dynamic adjustments: {stats['dynamic_adjustments']}")
    print(f"   Adjustment rate: {stats['adjustment_rate']:.1%}")
    
    return stats

def profile_integration_performance():
    """Profile full V13 integration performance vs baseline"""
    print("\n=== V13.0 Integration Performance Test ===")
    
    # Import main engine
    from v7p3r import V7P3REngine
    
    # Test positions
    test_positions = [
        chess.Board(),  # Starting position
        chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"),  # Italian
        chess.Board("rnbqkb1r/ppp2ppp/3p1n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 0 1"),  # French
    ]
    
    engine = V7P3REngine()
    
    total_search_time = 0
    total_nodes = 0
    
    for i, board in enumerate(test_positions):
        print(f"Testing search on position {i+1}/{len(test_positions)}...")
        
        start_time = time.time()
        move = engine.search(board, depth=4)
        end_time = time.time()
        
        search_time = end_time - start_time
        total_search_time += search_time
        
        if hasattr(engine, 'nodes_searched'):
            total_nodes += engine.nodes_searched
            
        print(f"   Move: {move}, Time: {search_time*1000:.0f}ms")
    
    avg_search_time = total_search_time / len(test_positions)
    nps = total_nodes / max(total_search_time, 0.001)
    
    print(f"\n📊 Search Performance:")
    print(f"   Average search time: {avg_search_time*1000:.0f}ms")
    print(f"   Total nodes searched: {total_nodes:,}")
    print(f"   Nodes per second: {nps:,.0f} NPS")
    
    # Show feature status
    print(f"\n🎯 V13.0 Feature Status:")
    print(f"   Tactical Detection: {'✅' if engine.ENABLE_TACTICAL_DETECTION else '❌'}")
    print(f"   Dynamic Evaluation: {'✅' if engine.ENABLE_DYNAMIC_EVALUATION else '❌'}")
    print(f"   Tal Complexity: {'✅' if engine.ENABLE_TAL_COMPLEXITY_BONUS else '❌'}")
    
    return nps

def main():
    """Run V13.0 profiling tests"""
    print("V7P3R v13.0 Tactical Enhancement Profiling")
    print("=" * 50)
    
    try:
        # Test tactical pattern detection frequency
        tactical_stats = profile_tactical_patterns()
        
        # Test dynamic evaluation performance  
        evaluation_stats = profile_dynamic_evaluation()
        
        # Test full integration performance
        nps = profile_integration_performance()
        
        print("\n" + "=" * 50)
        print("🎉 V13.0 Profiling Complete!")
        print(f"\nKey Findings:")
        print(f"   Most frequent pattern: {max(tactical_stats.items(), key=lambda x: x[1])[0]}")
        print(f"   Evaluation adjustment rate: {evaluation_stats['adjustment_rate']:.1%}")
        print(f"   Search performance: {nps:,.0f} NPS")
        
        # Recommendations based on profiling
        print(f"\n💡 Optimization Recommendations:")
        
        # Check if any patterns never fire
        zero_patterns = [pattern for pattern, count in tactical_stats.items() if count == 0]
        if zero_patterns:
            print(f"   Consider removing unused patterns: {', '.join(zero_patterns)}")
        
        # Check performance impact
        if nps < 15000:
            print(f"   Performance impact detected - consider optimization")
        elif nps > 20000:
            print(f"   Excellent performance maintained")
        
        return 0
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())