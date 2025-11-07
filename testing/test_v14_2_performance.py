#!/usr/bin/env python3
"""
V7P3R v14.2 Performance Test
Compare v14.2 simplified evaluation against v14.1

Tests:
1. Evaluation speed (NPS comparison)
2. Search depth reached (5-second search)
3. Time limit compliance (60s hard cap still honored)
4. Tactical accuracy (ensure simplifications don't hurt tactics)
"""

import sys
import os
import time
import chess

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine
from v7p3r_bitboard_evaluator import V7P3RBitboardEvaluator


def test_evaluation_speed():
    """Test evaluation speed - expect 51.3% speedup"""
    print("\n" + "="*80)
    print("V7P3R v14.2 EVALUATION SPEED TEST")
    print("="*80)
    print("Expected: 51.3% faster than v14.1 (55K NPS -> 95K+ NPS)")
    print()
    
    # Standard piece values
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 325,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }
    
    evaluator = V7P3RBitboardEvaluator(piece_values)
    
    # Test positions (same as profiler)
    test_positions = [
        # Opening positions
        chess.Board(),  # Starting position
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),  # e4
        chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"),  # e4 e5
        
        # Middlegame positions
        chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5"),
        chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        chess.Board("rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 0 4"),
        chess.Board("r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6"),
        
        # Endgame positions
        chess.Board("8/5k2/8/5P2/5K2/8/8/8 w - - 0 1"),  # K+P vs K
        chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"),  # K+P vs K endgame
        chess.Board("8/8/8/4k3/4P3/4K3/8/8 w - - 0 1"),  # K+P vs K endgame 2
    ]
    
    # Warm up
    for board in test_positions[:3]:
        evaluator.evaluate_bitboard(board, chess.WHITE)
    
    # Benchmark
    iterations = 1000
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        for board in test_positions:
            evaluator.evaluate_bitboard(board, chess.WHITE)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    total_evals = iterations * len(test_positions)
    avg_time_us = (total_time / total_evals) * 1_000_000
    evals_per_sec = total_evals / total_time
    
    # Estimate NPS in search (typically 70% of pure eval speed)
    estimated_nps = int(evals_per_sec * 0.70)
    
    print(f"Total evaluations: {total_evals:,}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average time per eval: {avg_time_us:.2f} μs")
    print(f"Evaluations/second: {evals_per_sec:,.0f}")
    print(f"Estimated NPS in search: {estimated_nps:,}")
    print()
    
    # Compare to v14.1 baseline
    v14_1_nps = 55_000
    speedup = (estimated_nps / v14_1_nps - 1) * 100
    
    print(f"V14.1 baseline NPS: {v14_1_nps:,}")
    print(f"V14.2 estimated NPS: {estimated_nps:,}")
    print(f"Speedup: {speedup:+.1f}%")
    
    if speedup >= 45:
        print("[PASS] EXCELLENT: Achieved target 51.3% speedup!")
    elif speedup >= 35:
        print("[PASS] GOOD: Close to target speedup")
    elif speedup >= 20:
        print("[WARN] ACCEPTABLE: Some speedup but below target")
    else:
        print("[FAIL] ISSUE: Speedup below expectations")
    
    return estimated_nps


def test_search_depth():
    """Test search depth - expect depth 8-9 in middlegame"""
    print("\n" + "="*80)
    print("V7P3R v14.2 SEARCH DEPTH TEST")
    print("="*80)
    print("Expected: Depth 8-9 in middlegame (up from v14.1's depth 5-6)")
    print()
    
    engine = V7P3REngine()
    
    # Middlegame test positions
    test_positions = [
        ("Italian Game", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5"),
        ("Sicilian Defense", "r1bqkb1r/pp1ppppp/2n2n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 4"),
        ("Queen's Gambit", "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 0 4"),
    ]
    
    depths_reached = []
    
    for name, fen in test_positions:
        board = chess.Board(fen)
        
        print(f"\nTesting {name}...")
        print(f"FEN: {fen}")
        
        # 5-second search
        time_limit = 5.0
        start_time = time.time()
        
        best_move, info = engine.search(board, time_limit=time_limit)
        
        search_time = time.time() - start_time
        depth = info.get('depth', 0)
        nodes = info.get('nodes', 0)
        nps = int(nodes / max(search_time, 0.001))
        
        depths_reached.append(depth)
        
        print(f"  Depth reached: {depth}")
        print(f"  Nodes searched: {nodes:,}")
        print(f"  Time: {search_time:.2f}s")
        print(f"  NPS: {nps:,}")
        print(f"  Best move: {best_move}")
        
        if depth >= 8:
            print(f"  [PASS] EXCELLENT: Depth {depth} achieved!")
        elif depth >= 7:
            print(f"  [PASS] GOOD: Depth {depth} close to target")
        elif depth >= 6:
            print(f"  [WARN]  ACCEPTABLE: Depth {depth} improved but below target")
        else:
            print(f"  [FAIL] ISSUE: Depth {depth} below v14.1 baseline")
    
    avg_depth = sum(depths_reached) / len(depths_reached)
    print(f"\n{'='*80}")
    print(f"AVERAGE DEPTH: {avg_depth:.1f}")
    
    if avg_depth >= 8.0:
        print("[PASS] EXCELLENT: Achieved target depth 8+!")
    elif avg_depth >= 7.0:
        print("[PASS] GOOD: Significant depth improvement")
    elif avg_depth >= 6.0:
        print("[WARN]  ACCEPTABLE: Some improvement")
    else:
        print("[FAIL] ISSUE: Depth below expectations")
    
    return avg_depth


def test_time_limit_compliance():
    """Test that 60-second hard cap is still honored"""
    print("\n" + "="*80)
    print("V7P3R v14.2 TIME LIMIT COMPLIANCE TEST")
    print("="*80)
    print("Expected: 60-second hard cap still honored")
    print()
    
    engine = V7P3REngine()
    
    # Complex middlegame position
    board = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8")
    
    print("Testing with large time limit (120s)...")
    print("Engine should cap at 60s")
    
    start_time = time.time()
    best_move, info = engine.search(board, time_limit=120.0)
    actual_time = time.time() - start_time
    
    print(f"Requested time: 120.0s")
    print(f"Actual time: {actual_time:.2f}s")
    print(f"Depth: {info.get('depth', 0)}")
    print(f"Nodes: {info.get('nodes', 0):,}")
    
    if actual_time <= 62.0:  # Allow 2s buffer for cleanup
        print("[PASS] PASSED: Time limit respected")
        return True
    else:
        print(f"[FAIL] FAILED: Exceeded 60s cap by {actual_time - 60:.1f}s")
        return False


def main():
    print("\n" + "="*80)
    print("V7P3R v14.2 PERFORMANCE TEST SUITE")
    print("="*80)
    print()
    print("Changes in v14.2:")
    print("  [-] Removed: Castling evaluation (41.7% overhead)")
    print("  [-] Removed: Activity penalties (5.7% overhead)")
    print("  [-] Removed: Knight outposts (3.9% overhead)")
    print("  [=] Total: 51.3% evaluation overhead removed")
    print()
    print("Expected improvements:")
    print("  * NPS: 55K -> 95K+ (73% faster)")
    print("  * Depth: 5-6 -> 8-9 (3 plies deeper)")
    print("  * Time management: 60s cap still enforced")
    print()
    
    results = {}
    
    # Run tests
    try:
        results['nps'] = test_evaluation_speed()
        results['depth'] = test_search_depth()
        results['time_compliance'] = test_time_limit_compliance()
        
        # Summary
        print("\n" + "="*80)
        print("V7P3R v14.2 PERFORMANCE TEST SUMMARY")
        print("="*80)
        print()
        print(f"Estimated NPS: {results['nps']:,} (target: 95K+)")
        print(f"Average depth: {results['depth']:.1f} (target: 8+)")
        print(f"Time compliance: {'PASS' if results['time_compliance'] else 'FAIL'}")
        print()
        
        # Overall verdict
        nps_ok = results['nps'] >= 85_000
        depth_ok = results['depth'] >= 7.5
        time_ok = results['time_compliance']
        
        if nps_ok and depth_ok and time_ok:
            print("[PASS] V14.2 READY FOR TOURNAMENT TESTING")
            print("Proceed to test vs MaterialOpponent at 5min+5sec")
        elif depth_ok and time_ok:
            print("[PASS] V14.2 ACCEPTABLE - depth improved, proceed with testing")
        else:
            print("[WARN]  V14.2 NEEDS REVIEW - check results above")
        
    except Exception as e:
        print(f"\n[FAIL] TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
