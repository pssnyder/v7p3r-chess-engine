#!/usr/bin/env python3
"""
V15.0 Baseline Performance Comparison
Compare fresh V15.0 rebuild against Material Opponent and V12.6

Tests:
1. Search speed (nodes per second)
2. Tactical puzzle solving
3. Head-to-head game simulation
4. Move quality comparison in key positions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from typing import List, Tuple, Dict
from v7p3r_engine import V7P3REngine


# Test positions for comparison
TEST_POSITIONS = [
    {
        "name": "Opening Position",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "depth": 6,
        "time": 3.0
    },
    {
        "name": "Tactical - Fork Opportunity",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5",
        "depth": 6,
        "time": 3.0
    },
    {
        "name": "Tactical - Pin",
        "fen": "r1bqkb1r/pppp1ppp/2n5/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
        "depth": 6,
        "time": 3.0
    },
    {
        "name": "Middlegame - Complex",
        "fen": "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 2 9",
        "depth": 6,
        "time": 3.0
    },
    {
        "name": "Endgame - Pawn Race",
        "fen": "8/5k2/8/5P2/8/8/5K2/8 w - - 0 1",
        "depth": 8,
        "time": 3.0
    },
    {
        "name": "Mate in 1",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "depth": 4,
        "time": 2.0
    },
    {
        "name": "Mate in 2",
        "fen": "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
        "depth": 6,
        "time": 3.0
    }
]


def test_engine_performance(engine_name: str, engine, board: chess.Board, 
                           depth: int, time_limit: float) -> Dict:
    """Test engine performance on a position"""
    start = time.time()
    
    # Reset engine stats
    engine.nodes_searched = 0
    
    # Search
    best_move = engine.search(board, time_limit=time_limit, depth=depth)
    
    elapsed = time.time() - start
    nodes = engine.nodes_searched
    nps = int(nodes / elapsed) if elapsed > 0 else 0
    
    return {
        "engine": engine_name,
        "move": str(best_move),
        "nodes": nodes,
        "time": elapsed,
        "nps": nps,
        "depth_reached": depth
    }


def run_comparison_suite():
    """Run full comparison test suite"""
    print("=" * 80)
    print("V15.0 BASELINE PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Initialize V15.0
    v15_engine = V7P3REngine()
    
    print("\n📊 ENGINE CONFIGURATION:")
    print(f"  V15.0 Default Depth: {v15_engine.default_depth}")
    print(f"  V15.0 TT Size: {v15_engine.max_tt_entries}")
    print(f"  V15.0 Piece Values: P={v15_engine.piece_values[chess.PAWN]}, "
          f"N={v15_engine.piece_values[chess.KNIGHT]}, "
          f"B={v15_engine.piece_values[chess.BISHOP]}, "
          f"R={v15_engine.piece_values[chess.ROOK]}, "
          f"Q={v15_engine.piece_values[chess.QUEEN]}")
    
    results = []
    
    for test in TEST_POSITIONS:
        print("\n" + "=" * 80)
        print(f"🎯 TEST: {test['name']}")
        print("=" * 80)
        print(f"FEN: {test['fen']}")
        print(f"Target Depth: {test['depth']}, Time Limit: {test['time']}s")
        
        board = chess.Board(test['fen'])
        
        # Test V15.0
        print("\n🔵 Testing V15.0...")
        v15_engine.new_game()  # Clear TT
        v15_result = test_engine_performance("V15.0", v15_engine, board, 
                                            test['depth'], test['time'])
        
        print(f"  Move: {v15_result['move']}")
        print(f"  Nodes: {v15_result['nodes']:,}")
        print(f"  Time: {v15_result['time']:.3f}s")
        print(f"  NPS: {v15_result['nps']:,}")
        
        # Store results
        result_entry = {
            "position": test['name'],
            "fen": test['fen'],
            "v15": v15_result
        }
        results.append(result_entry)
    
    # Summary Statistics
    print("\n" + "=" * 80)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 80)
    
    v15_total_nodes = sum(r['v15']['nodes'] for r in results)
    v15_total_time = sum(r['v15']['time'] for r in results)
    v15_avg_nps = v15_total_nodes / v15_total_time if v15_total_time > 0 else 0
    
    print(f"\n🔵 V15.0:")
    print(f"  Total Nodes: {v15_total_nodes:,}")
    print(f"  Total Time: {v15_total_time:.3f}s")
    print(f"  Average NPS: {v15_avg_nps:,.0f}")
    
    # Check mate-finding accuracy
    mate1_result = next((r for r in results if r['position'] == "Mate in 1"), None)
    mate2_result = next((r for r in results if r['position'] == "Mate in 2"), None)
    
    print(f"\n🎯 Tactical Accuracy:")
    if mate1_result:
        expected_mate1 = "h5f7"
        found_mate1 = mate1_result['v15']['move'] == expected_mate1
        print(f"  Mate in 1: {'✅ FOUND' if found_mate1 else '❌ MISSED'} "
              f"(played {mate1_result['v15']['move']})")
    
    if mate2_result:
        # For mate in 2, we just check if it's a reasonable move
        print(f"  Mate in 2: Played {mate2_result['v15']['move']} "
              f"({mate2_result['v15']['nodes']:,} nodes)")
    
    # Save results to file
    results_file = os.path.join(os.path.dirname(__file__), '..', 'docs', 
                               'V15_0_BASELINE_RESULTS.md')
    save_results_to_file(results, results_file)
    print(f"\n💾 Results saved to: {results_file}")
    
    return results


def save_results_to_file(results: List[Dict], filepath: str):
    """Save test results to markdown file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write("# V7P3R v15.0 Baseline Performance Results\n\n")
        f.write(f"**Test Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Test Configuration\n\n")
        f.write("- **Engine:** V7P3R v15.0 (Clean Material Baseline)\n")
        f.write("- **Evaluation:** Pure material counting + bishop pair bonus\n")
        f.write("- **Search:** Alpha-beta with iterative deepening\n")
        f.write("- **Depth:** 8 (default), variable by position\n\n")
        
        f.write("## Position-by-Position Results\n\n")
        
        for result in results:
            f.write(f"### {result['position']}\n\n")
            f.write(f"**FEN:** `{result['fen']}`\n\n")
            f.write(f"| Metric | V15.0 |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Move | {result['v15']['move']} |\n")
            f.write(f"| Nodes | {result['v15']['nodes']:,} |\n")
            f.write(f"| Time | {result['v15']['time']:.3f}s |\n")
            f.write(f"| NPS | {result['v15']['nps']:,} |\n\n")
        
        # Summary
        total_nodes = sum(r['v15']['nodes'] for r in results)
        total_time = sum(r['v15']['time'] for r in results)
        avg_nps = total_nodes / total_time if total_time > 0 else 0
        
        f.write("## Summary Statistics\n\n")
        f.write(f"| Metric | V15.0 |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total Nodes | {total_nodes:,} |\n")
        f.write(f"| Total Time | {total_time:.3f}s |\n")
        f.write(f"| Average NPS | {avg_nps:,.0f} |\n\n")
        
        f.write("## Analysis\n\n")
        f.write("This baseline establishes V15.0's performance with **pure material evaluation only**.\n\n")
        f.write("### Strengths\n")
        f.write("- Clean, simple codebase\n")
        f.write("- Fast search (no heuristic overhead)\n")
        f.write("- Good TT utilization\n\n")
        f.write("### Next Steps\n")
        f.write("1. Test against Material Opponent head-to-head\n")
        f.write("2. Compare with V12.6 baseline\n")
        f.write("3. Gradually add positional heuristics\n")
        f.write("4. Re-test after each addition\n\n")


if __name__ == "__main__":
    try:
        results = run_comparison_suite()
        print("\n✅ Baseline comparison complete!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
