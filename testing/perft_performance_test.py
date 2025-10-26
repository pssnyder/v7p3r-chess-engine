#!/usr/bin/env python3

"""
V7P3R Performance Testing with Perft

Performance test (Perft) validates move generation and measures engine speed.
Standard chess engine benchmarking tool.
"""

import time
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

def run_perft_tests():
    """Run standard perft tests to validate move generation and measure performance"""
    
    print("V7P3R v14.3 Performance Testing (Perft)")
    print("=" * 50)
    
    engine = V7P3REngine()
    
    # Standard perft test positions
    test_positions = [
        {
            "name": "Starting Position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "expected": {
                1: 20,
                2: 400, 
                3: 8902,
                4: 197281,
                5: 4865609,
                6: 119060324  # Only test if fast enough
            }
        },
        {
            "name": "Kiwipete Position",
            "fen": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "expected": {
                1: 6,
                2: 264,
                3: 9467,
                4: 422333,
                5: 15833292  # Only test if fast enough
            }
        },
        {
            "name": "Position 3",
            "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "expected": {
                1: 14,
                2: 191,
                3: 2812,
                4: 43238,
                5: 674624
            }
        },
        {
            "name": "Position 4",
            "fen": "r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1",
            "expected": {
                1: 8,
                2: 348,
                3: 10828,
                4: 493407,
                5: 15894865  # Only test if fast enough
            }
        },
        {
            "name": "Position 5",
            "fen": "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "expected": {
                1: 44,
                2: 1486,
                3: 62379,
                4: 2103487,
                5: 89941194  # Only test if fast enough
            }
        }
    ]
    
    total_nodes = 0
    total_time = 0
    all_tests_passed = True
    
    for position in test_positions:
        print(f"\n{position['name']}")
        print(f"FEN: {position['fen']}")
        print("-" * 40)
        
        board = chess.Board(position['fen'])
        
        for depth in sorted(position['expected'].keys()):
            expected_nodes = position['expected'][depth]
            
            # Skip deep searches if they would take too long
            if depth >= 5 and expected_nodes > 50000000:
                print(f"Depth {depth}: Skipped (too slow for testing)")
                continue
            if depth >= 6:
                print(f"Depth {depth}: Skipped (very deep)")
                continue
                
            start_time = time.time()
            
            try:
                actual_nodes = engine.perft(board, depth, divide=(depth <= 2))
                elapsed = time.time() - start_time
                
                total_nodes += actual_nodes
                total_time += elapsed
                
                # Calculate performance
                nps = int(actual_nodes / max(elapsed, 0.001))
                
                # Check if result is correct
                if actual_nodes == expected_nodes:
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                    all_tests_passed = False
                
                print(f"Depth {depth}: {actual_nodes:,} nodes ({nps:,} nps) {elapsed:.3f}s {status}")
                
                if actual_nodes != expected_nodes:
                    print(f"  Expected: {expected_nodes:,}, Got: {actual_nodes:,}")
                    
            except Exception as e:
                print(f"Depth {depth}: ERROR - {e}")
                all_tests_passed = False
    
    # Performance summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    
    if total_time > 0:
        overall_nps = int(total_nodes / total_time)
        print(f"Total nodes: {total_nodes:,}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Overall NPS: {overall_nps:,}")
        
        # Performance categories for chess engines
        if overall_nps >= 5000000:
            performance = "🚀 EXCELLENT (5M+ nps)"
        elif overall_nps >= 2000000:
            performance = "✅ VERY GOOD (2M+ nps)"
        elif overall_nps >= 1000000:
            performance = "👍 GOOD (1M+ nps)"
        elif overall_nps >= 500000:
            performance = "⚠️  MODERATE (500K+ nps)"
        elif overall_nps >= 100000:
            performance = "⚠️  SLOW (100K+ nps)"
        else:
            performance = "❌ VERY SLOW (<100K nps)"
            
        print(f"Performance rating: {performance}")
        
        # Time per depth estimate
        print(f"\nEstimated search time per depth:")
        print(f"  1-ply: {(1000/overall_nps)*1000:.1f}ms per 1000 nodes")
        print(f"  2-ply: {(10000/overall_nps)*1000:.1f}ms per 10K nodes")
        print(f"  3-ply: {(100000/overall_nps)*1000:.1f}ms per 100K nodes")
        
    else:
        print("No performance data collected")
    
    # Validation summary
    print(f"\nValidation: {'✅ ALL TESTS PASSED' if all_tests_passed else '❌ SOME TESTS FAILED'}")
    
    if all_tests_passed:
        print("✅ Move generation is CORRECT")
        print("✅ Engine is ready for tournament play")
    else:
        print("❌ Move generation has BUGS")
        print("❌ Engine needs debugging before use")
    
    return all_tests_passed, overall_nps if total_time > 0 else 0

if __name__ == "__main__":
    success, nps = run_perft_tests()
    sys.exit(0 if success else 1)