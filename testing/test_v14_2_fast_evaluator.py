#!/usr/bin/env python3
"""
V14.2 Fast Evaluator Speed & Depth Test
Compare fast evaluator vs bitboard evaluator performance
"""

import sys
import time
import chess
sys.path.insert(0, 's:/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/src')

from v7p3r import V7P3REngine

def test_evaluator_speed(engine, board, iterations=1000):
    """Test raw evaluation speed"""
    start = time.time()
    for _ in range(iterations):
        engine._evaluate_position(board)
    elapsed = time.time() - start
    per_eval = (elapsed / iterations) * 1000  # milliseconds
    return elapsed, per_eval

def test_search_depth(engine, board, time_limit=5.0):
    """Test search depth achieved in given time"""
    depths_reached = []
    
    # Run 5 searches from different positions
    test_fens = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "rnbqkb1r/pp2pppp/2p2n2/3p4/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 5",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5",
        "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq d6 0 4",
    ]
    
    for fen in test_fens:
        test_board = chess.Board(fen)
        engine.nodes_searched = 0
        start_time = time.time()
        
        # Perform search
        move = engine.search(test_board, time_limit=time_limit)
        
        elapsed = time.time() - start_time
        nodes = engine.nodes_searched
        nps = int(nodes / elapsed) if elapsed > 0 else 0
        
        # Estimate depth from nodes (rough approximation)
        # Typical branching factor ~35, so depth ≈ log_35(nodes)
        import math
        estimated_depth = math.log(nodes, 35) if nodes > 1 else 1
        depths_reached.append(estimated_depth)
        
        print(f"  Position {len(depths_reached)}: {nodes:,} nodes, {elapsed:.2f}s, {nps:,} nps, est. depth {estimated_depth:.1f}")
    
    avg_depth = sum(depths_reached) / len(depths_reached)
    min_depth = min(depths_reached)
    max_depth = max(depths_reached)
    
    return avg_depth, min_depth, max_depth

def main():
    print("=" * 80)
    print("V14.2 FAST EVALUATOR PERFORMANCE TEST")
    print("=" * 80)
    print()
    
    # Test position
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("e7e5"))
    
    # TEST 1: Fast Evaluator
    print("TEST 1: Fast Evaluator (V16.1 Speed)")
    print("-" * 80)
    engine_fast = V7P3REngine(use_fast_evaluator=True)
    
    print("Evaluation Speed Test (1000 iterations)...")
    elapsed, per_eval = test_evaluator_speed(engine_fast, board, 1000)
    print(f"  Total: {elapsed:.3f}s")
    print(f"  Per evaluation: {per_eval:.4f}ms")
    print(f"  Evaluations/second: {int(1000/per_eval):,}")
    print()
    
    print("Search Depth Test (5 seconds per position)...")
    avg_depth, min_depth, max_depth = test_search_depth(engine_fast, board, 5.0)
    print(f"  Average depth: {avg_depth:.1f}")
    print(f"  Depth range: {min_depth:.1f} - {max_depth:.1f}")
    print()
    
    # TEST 2: Bitboard Evaluator (for comparison)
    print("TEST 2: Bitboard Evaluator (V14.1 Comprehensive)")
    print("-" * 80)
    engine_bitboard = V7P3REngine(use_fast_evaluator=False)
    
    print("Evaluation Speed Test (1000 iterations)...")
    elapsed, per_eval = test_evaluator_speed(engine_bitboard, board, 1000)
    print(f"  Total: {elapsed:.3f}s")
    print(f"  Per evaluation: {per_eval:.4f}ms")
    print(f"  Evaluations/second: {int(1000/per_eval):,}")
    print()
    
    print("Search Depth Test (5 seconds per position)...")
    avg_depth, min_depth, max_depth = test_search_depth(engine_bitboard, board, 5.0)
    print(f"  Average depth: {avg_depth:.1f}")
    print(f"  Depth range: {min_depth:.1f} - {max_depth:.1f}")
    print()
    
    # SUMMARY
    print("=" * 80)
    print("SUMMARY & ANALYSIS")
    print("=" * 80)
    print()
    print("Goal: Achieve consistent depth 6-8 (like V16.1)")
    print()
    print("Fast Evaluator:")
    print(f"  ✓ Speed: ~{test_evaluator_speed(engine_fast, board, 100)[1]:.3f}ms per evaluation")
    print(f"  ✓ Depth: {avg_depth:.1f} average")
    print()
    print("Bitboard Evaluator:")
    print(f"  • Speed: ~{test_evaluator_speed(engine_bitboard, board, 100)[1]:.3f}ms per evaluation")
    print(f"  • Depth: {avg_depth:.1f} average")
    print()
    
    # Determine success
    engine_fast_depth = test_search_depth(engine_fast, board, 5.0)[0]
    if engine_fast_depth >= 6.0:
        print("🟢 SUCCESS: Fast evaluator achieves target depth 6+!")
        print("   → Ready for tournament testing")
    elif engine_fast_depth >= 5.0:
        print("🟡 PROMISING: Fast evaluator reaches depth 5+")
        print("   → May need move filtering enhancement (Phase 2)")
    else:
        print("🔴 NEEDS WORK: Fast evaluator below depth 5")
        print("   → Review search efficiency and move ordering")

if __name__ == "__main__":
    main()
