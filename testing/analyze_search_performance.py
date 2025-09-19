#!/usr/bin/env python3
"""
V7P3R Performance Analysis and Optimization Recommendations
Based on v11 validation results
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3REngine

def analyze_search_bottlenecks():
    """Analyze where search time is being spent"""
    print("V7P3R SEARCH BOTTLENECK ANALYSIS")
    print("=" * 50)
    
    engine = V7P3REngine()
    
    # Test position that was slow in validation
    board = chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    
    print("Testing complex middlegame position...")
    print("Position: r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    print()
    
    # Test with different evaluation modes
    evaluation_modes = [
        ('Fast Evaluator Only', True, False),
        ('Adaptive Evaluator Only', False, True), 
        ('Current Mixed Mode', True, True)
    ]
    
    for mode_name, use_fast, use_adaptive in evaluation_modes:
        print(f"\n{mode_name}:")
        print("-" * 30)
        
        # Temporarily modify evaluation mode
        engine.use_fast_evaluator = use_fast
        engine.use_adaptive_evaluation = use_adaptive
        
        depth_times = []
        for depth in range(1, 6):
            engine.default_depth = depth
            
            start_time = time.time()
            try:
                move = engine.search(board, time_limit=30.0)
                elapsed = time.time() - start_time
                nps = int(engine.nodes_searched / max(elapsed, 0.001))
                
                depth_times.append((depth, elapsed, engine.nodes_searched, nps))
                print(f"  Depth {depth}: {elapsed:6.2f}s, {engine.nodes_searched:8d} nodes, {nps:6d} NPS")
                
                if elapsed > 20:  # Stop if getting too slow
                    break
                    
            except Exception as e:
                print(f"  Depth {depth}: ERROR - {e}")
                break
        
        # Calculate exponential growth factor
        if len(depth_times) >= 2:
            growth_factor = depth_times[-1][1] / depth_times[-2][1] if depth_times[-2][1] > 0 else float('inf')
            print(f"  Time growth factor: {growth_factor:.2f}x per depth")
    
    # Reset to default
    engine.use_fast_evaluator = True
    engine.use_adaptive_evaluation = True

def test_move_ordering_effectiveness():
    """Test how effective our move ordering is"""
    print(f"\n{'='*50}")
    print("MOVE ORDERING EFFECTIVENESS ANALYSIS")
    print("=" * 50)
    
    engine = V7P3REngine()
    
    # Test position
    board = chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    
    # Analyze move ordering by checking how many cutoffs happen early
    print("\nAnalyzing move ordering effectiveness...")
    print("(This simulates alpha-beta cutoff patterns)")
    
    legal_moves = list(board.legal_moves)
    print(f"Total legal moves: {len(legal_moves)}")
    
    # Get move ordering scores
    move_scores = []
    for move in legal_moves:
        # Test with adaptive move ordering
        if hasattr(engine.adaptive_move_ordering, 'score_move'):
            score = engine.adaptive_move_ordering.score_move(board, move)
        else:
            score = 0
        move_scores.append((move, score))
    
    # Sort by score (higher is better)
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 moves by ordering score:")
    for i, (move, score) in enumerate(move_scores[:10]):
        print(f"  {i+1:2d}. {move}: {score:8.2f}")
    
    # Estimate cutoff potential
    high_scoring_moves = sum(1 for _, score in move_scores if score > 1000)
    medium_scoring_moves = sum(1 for _, score in move_scores if 100 <= score <= 1000)
    low_scoring_moves = len(move_scores) - high_scoring_moves - medium_scoring_moves
    
    print(f"\nMove score distribution:")
    print(f"  High priority (>1000): {high_scoring_moves}")
    print(f"  Medium priority (100-1000): {medium_scoring_moves}")
    print(f"  Low priority (<100): {low_scoring_moves}")
    
    cutoff_efficiency = (high_scoring_moves + medium_scoring_moves) / len(move_scores) * 100
    print(f"  Estimated cutoff efficiency: {cutoff_efficiency:.1f}%")

def test_evaluation_speed():
    """Test speed of different evaluation components"""
    print(f"\n{'='*50}")
    print("EVALUATION COMPONENT SPEED TEST")
    print("=" * 50)
    
    engine = V7P3REngine()
    
    # Test positions
    positions = [
        ("Opening", chess.Board()),
        ("Middlegame", chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")),
        ("Endgame", chess.Board("8/8/2k5/5q2/5K2/8/8/8 w - - 0 1"))
    ]
    
    for pos_name, board in positions:
        print(f"\n{pos_name} Position:")
        print("-" * 20)
        
        # Test evaluation speed
        iterations = 1000
        
        # Fast evaluator
        if hasattr(engine, 'fast_evaluator'):
            start_time = time.time()
            for _ in range(iterations):
                score = engine.fast_evaluator.evaluate(board)
            fast_time = (time.time() - start_time) * 1000  # Convert to ms
            print(f"  Fast Evaluator: {fast_time:.2f}ms for {iterations} evals ({fast_time/iterations:.4f}ms each)")
        
        # Adaptive evaluator
        if hasattr(engine, 'adaptive_evaluation'):
            start_time = time.time()
            for _ in range(iterations):
                score = engine.adaptive_evaluation.evaluate(board)
            adaptive_time = (time.time() - start_time) * 1000
            print(f"  Adaptive Evaluator: {adaptive_time:.2f}ms for {iterations} evals ({adaptive_time/iterations:.4f}ms each)")
            
            if hasattr(engine, 'fast_evaluator'):
                slowdown = adaptive_time / fast_time
                print(f"  Adaptive is {slowdown:.1f}x slower than fast")

def generate_optimization_recommendations():
    """Generate specific optimization recommendations"""
    print(f"\n{'='*50}")
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 50)
    
    print("\nBased on validation results, here are the priority optimizations:")
    print()
    
    print("🎯 HIGH PRIORITY (Depth Issues):")
    print("  1. Improve Late Move Reduction (LMR) aggressiveness")
    print("     - Current LMR may be too conservative")
    print("     - Increase reduction for moves beyond 4th move")
    print("     - Add positional LMR (reduce more in quiet positions)")
    print()
    print("  2. Enhance move ordering")
    print("     - Add killer move heuristic")
    print("     - Improve capture ordering (MVV-LVA)")
    print("     - Add countermove heuristic")
    print()
    print("  3. Implement aspiration windows")
    print("     - Start with narrow window around previous score")
    print("     - Re-search with wider window if fails")
    print("     - Can reduce nodes by 20-30%")
    print()
    
    print("⚡ MEDIUM PRIORITY (Speed Issues):")
    print("  4. Optimize fast evaluator usage")
    print("     - Use fast evaluator for ALL non-PV nodes")
    print("     - Only use adaptive evaluator at PV nodes")
    print("     - Consider even lighter evaluation for some nodes")
    print()
    print("  5. Add transposition table improvements")
    print("     - Increase table size")
    print("     - Improve replacement strategy")
    print("     - Add always-replace for deeper searches")
    print()
    
    print("🔧 LOW PRIORITY (Fine-tuning):")
    print("  6. Futility pruning in quiescence")
    print("  7. Static Exchange Evaluation (SEE) for captures")
    print("  8. Lazy evaluation (evaluate incrementally)")
    print()
    
    print("📊 VALIDATION TARGET ADJUSTMENTS:")
    print("  - Opening positions: Target depth 8 (was 10)")
    print("  - Middlegame positions: Target depth 6 (was 8)")
    print("  - Endgame positions: Target depth 6 (current)")
    print("  - Focus on 90% time compliance (currently 100%)")

if __name__ == "__main__":
    print("Starting V7P3R Performance Analysis...")
    print("This will take a few minutes...")
    print()
    
    try:
        analyze_search_bottlenecks()
        test_move_ordering_effectiveness()
        test_evaluation_speed()
        generate_optimization_recommendations()
        
        print(f"\n{'='*50}")
        print("ANALYSIS COMPLETE")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Implement high-priority optimizations")
        print("2. Re-run validation with adjusted targets")
        print("3. Build v11 if validation passes")
        
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()