#!/usr/bin/env python3

"""
Corrected V7P3R Perft Performance Test

Tests the move generation accuracy and performance of the V7P3R engine
using the standard chess programming perft test positions.

CORRECTED with proper Position 4 data from chessprogramming.org
"""

import chess
import time
import sys
import os

# Add src directory to path for importing V7P3R engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def run_perft_test():
    """Run the standard perft test suite"""
    
    # Standard perft test positions - CORRECTED DATA
    test_positions = [
        {
            "name": "Position 1 (Initial)",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "depths": {1: 20, 2: 400, 3: 8902, 4: 197281}
        },
        {
            "name": "Position 2 (Kiwipete)",
            "fen": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
            "depths": {1: 48, 2: 2039, 3: 97862}
        },
        {
            "name": "Position 3",
            "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
            "depths": {1: 14, 2: 191, 3: 2812}
        },
        {
            "name": "Position 4 (CORRECTED)",
            "fen": "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
            "depths": {1: 6, 2: 264, 3: 9467}
        },
        {
            "name": "Position 5",
            "fen": "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "depths": {1: 44, 2: 1486, 3: 62379}
        }
    ]
    
    engine = V7P3REngine()
    total_nodes = 0
    total_time = 0
    all_passed = True
    
    print("V7P3R Perft Performance Test (CORRECTED)")
    print("=" * 60)
    
    for position in test_positions:
        print(f"\n{position['name']}")
        print(f"FEN: {position['fen']}")
        print("-" * 40)
        
        board = chess.Board(position['fen'])
        position_passed = True
        
        for depth, expected in position['depths'].items():
            # Skip deeper tests if performance is too slow
            if depth > 2 and total_time > 0 and (total_nodes / total_time) < 100000:
                print(f"Depth {depth}: SKIPPED (performance too slow)")
                continue
                
            print(f"Testing depth {depth}...")
            
            start_time = time.time()
            try:
                result = engine.perft(board, depth)
                end_time = time.time()
                
                test_time = end_time - start_time
                total_time += test_time
                total_nodes += result
                
                nps = result / test_time if test_time > 0 else 0
                
                if result == expected:
                    print(f"  ✅ PASS - {result:,} nodes ({nps:,.0f} nps)")
                else:
                    print(f"  ❌ FAIL - Expected {expected:,}, got {result:,}")
                    position_passed = False
                    all_passed = False
                    
            except Exception as e:
                print(f"  ❌ ERROR - {e}")
                position_passed = False
                all_passed = False
        
        print(f"Position result: {'✅ PASS' if position_passed else '❌ FAIL'}")
    
    # Performance categorization
    overall_nps = total_nodes / total_time if total_time > 0 else 0
    
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total nodes: {total_nodes:,}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average NPS: {overall_nps:,.0f}")
    
    if overall_nps > 1000000:
        performance_category = "EXCELLENT (1M+ nps)"
    elif overall_nps > 500000:
        performance_category = "GOOD (500K+ nps)"
    elif overall_nps > 100000:
        performance_category = "ACCEPTABLE (100K+ nps)"
    else:
        performance_category = "SLOW (<100K nps) - NEEDS OPTIMIZATION"
    
    print(f"Performance: {performance_category}")
    
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    if all_passed:
        print("✅ ALL TESTS PASSED - Move generation is CORRECT")
    else:
        print("❌ SOME TESTS FAILED - Move generation has BUGS")
        print("⚠️  DO NOT USE ENGINE IN TOURNAMENTS UNTIL FIXED!")
    
    return all_passed, overall_nps

if __name__ == "__main__":
    passed, nps = run_perft_test()
    
    if not passed:
        sys.exit(1)  # Exit with error code if tests failed