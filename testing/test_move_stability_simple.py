"""
Simple Move Stability Test - Direct Move Comparison Across Depths

Tests if the engine changes its best move as search depth increases.
Uses the actual search() function and compares returned moves.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine
import time

def test_position_stability(fen, description, target_depths=[1, 2, 3, 4, 5, 6]):
    """
    Test move stability by running search at increasing depths
    
    Returns:
        dict with stability analysis
    """
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"FEN: {fen}")
    print(f"{'='*80}\n")
    
    board = chess.Board(fen)
    results = []
    move_changes = []
    
    for depth in target_depths:
        # Create fresh engine for each depth (avoid TT contamination)
        engine = V7P3REngine()
        engine.default_depth = depth
        
        start_time = time.time()
        best_move = engine.search(board.copy(), time_limit=30.0)
        elapsed = time.time() - start_time
        
        # Get search statistics
        nodes = engine.nodes_searched
        seldepth = engine.seldepth
        nps = int(nodes / elapsed) if elapsed > 0 else 0
        
        result = {
            'depth': depth,
            'move': best_move.uci() if best_move else 'none',
            'nodes': nodes,
            'seldepth': seldepth,
            'time_ms': int(elapsed * 1000),
            'nps': nps
        }
        
        results.append(result)
        
        # Print result
        print(f"Depth {depth:2d} | seldepth {seldepth:2d} | move {result['move']:6s} | "
              f"nodes {nodes:8,d} | time {result['time_ms']:5d}ms | nps {nps:7,d}")
        
        # Check for move change
        if len(results) > 1:
            prev_move = results[-2]['move']
            curr_move = result['move']
            
            if prev_move != curr_move:
                change = {
                    'from_depth': depth - 1,
                    'to_depth': depth,
                    'old_move': prev_move,
                    'new_move': curr_move,
                    'seldepth_change': seldepth - results[-2]['seldepth']
                }
                move_changes.append(change)
                print(f"  WARNING: MOVE CHANGED: {prev_move} -> {curr_move} "
                      f"(seldepth change: {change['seldepth_change']:+d})")
    
    # Analysis
    total_depth_increases = len(results) - 1
    stability_pct = 100 * (1 - (len(move_changes) / total_depth_increases)) if total_depth_increases > 0 else 100
    
    print(f"\n{'-'*80}")
    print(f"STABILITY ANALYSIS:")
    print(f"{'-'*80}")
    print(f"Move Changes: {len(move_changes)}/{total_depth_increases} depth transitions")
    print(f"Stability: {stability_pct:.1f}%")
    
    if len(move_changes) == 0:
        print("✅ STABLE: No move changes detected")
    elif len(move_changes) <= total_depth_increases * 0.25:
        print("🟡 MODERATE: Some move instability")
    else:
        print("🔴 UNSTABLE: Frequent move changes")
    
    # Analyze seldepth progression
    seldepths = [r['seldepth'] for r in results]
    depths = [r['depth'] for r in results]
    avg_overhead = sum(s - d for s, d in zip(seldepths, depths)) / len(depths)
    
    print(f"\nQuiescence Overhead:")
    print(f"  Depths:    {' '.join(f'{d:2d}' for d in depths)}")
    print(f"  Seldepths: {' '.join(f'{s:2d}' for s in seldepths)}")
    print(f"  Average overhead: {avg_overhead:.1f} plies")
    
    if avg_overhead > 4:
        print("  ⚠️  High quiescence overhead (>4 plies)")
    
    # Check if quiescence correlates with move changes
    if move_changes:
        quiescence_correlated = sum(1 for c in move_changes if abs(c['seldepth_change']) >= 2)
        if quiescence_correlated > 0:
            print(f"\n⚠️  Quiescence Correlation: {quiescence_correlated}/{len(move_changes)} move changes")
            print(f"   have seldepth jumps ≥2, suggesting quiescence influence")
    
    return {
        'description': description,
        'fen': fen,
        'results': results,
        'move_changes': move_changes,
        'stability_pct': stability_pct,
        'avg_quiescence_overhead': avg_overhead
    }

def main():
    """Run move stability tests"""
    
    print("\n" + "="*80)
    print("V7P3R MOVE STABILITY DIAGNOSTIC")
    print("Testing if quiescence causes the engine to change moves at deeper depths")
    print("="*80)
    
    test_results = []
    
    # Test 1: Tactical position
    test_results.append(test_position_stability(
        fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        description="Tactical - Italian Game",
        target_depths=[1, 2, 3, 4, 5, 6, 7]
    ))
    
    # Test 2: Discovered attack (100% puzzle success theme)
    test_results.append(test_position_stability(
        fen="r1bq1rk1/ppp2ppp/2n5/3np3/1b1P4/2NB1N2/PPP2PPP/R1BQK2R w KQ - 0 8",
        description="Discovered Attack Theme",
        target_depths=[1, 2, 3, 4, 5, 6, 7]
    ))
    
    # Test 3: Zugzwang (60.5% puzzle success - weakness)
    test_results.append(test_position_stability(
        fen="8/8/p7/1p6/1P6/P7/8/k1K5 w - - 0 1",
        description="Zugzwang - Pawn Endgame",
        target_depths=[1, 2, 3, 4, 5, 6, 7, 8]
    ))
    
    # Test 4: Complex middlegame
    test_results.append(test_position_stability(
        fen="r2qkb1r/ppp2ppp/2n5/3pP3/3Pn1b1/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 7",
        description="Complex Middlegame - French Defense",
        target_depths=[1, 2, 3, 4, 5, 6]
    ))
    
    # Test 5: Quiet positional
    test_results.append(test_position_stability(
        fen="rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",
        description="Quiet Positional - QGD",
        target_depths=[1, 2, 3, 4, 5, 6]
    ))
    
    # Test 6: Forced mate
    test_results.append(test_position_stability(
        fen="6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
        description="Forced Mate in 2",
        target_depths=[1, 2, 3, 4, 5, 6]
    ))
    
    # Summary Report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    total_positions = len(test_results)
    total_changes = sum(len(r['move_changes']) for r in test_results)
    avg_stability = sum(r['stability_pct'] for r in test_results) / total_positions
    avg_quiescence = sum(r['avg_quiescence_overhead'] for r in test_results) / total_positions
    
    print(f"\nPositions Tested: {total_positions}")
    print(f"Total Move Changes: {total_changes}")
    print(f"Average Stability: {avg_stability:.1f}%")
    print(f"Average Quiescence Overhead: {avg_quiescence:.1f} plies")
    
    print(f"\nPer-Position Results:")
    for result in test_results:
        status = "✅" if result['stability_pct'] >= 75 else ("🟡" if result['stability_pct'] >= 50 else "🔴")
        print(f"  {status} {result['description']:40s} {result['stability_pct']:5.1f}% stable, "
              f"{len(result['move_changes'])} changes, {result['avg_quiescence_overhead']:.1f} ply Q-overhead")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print(f"{'='*80}")
    
    if avg_stability < 60:
        print("\n⚠️  LOW STABILITY DETECTED")
        print("   The engine is changing moves frequently as depth increases.")
        print("   This suggests quiescence or evaluation inconsistency.")
    elif avg_stability < 85:
        print("\n🟡 MODERATE STABILITY")
        print("   Some positions show move instability.")
        print("   Review positions with high change rates.")
    else:
        print("\n✅ EXCELLENT STABILITY")
        print("   The engine maintains consistent move selection across depths.")
    
    if avg_quiescence > 4:
        print("\n⚠️  HIGH QUIESCENCE OVERHEAD")
        print(f"   Average of {avg_quiescence:.1f} plies beyond regular search depth.")
        print("   Consider reducing MAX_QUIESCENCE_DEPTH from 10 to 4-6.")
    
    # Check correlation
    positions_with_q_correlation = sum(1 for r in test_results 
                                       if any(abs(c['seldepth_change']) >= 2 for c in r['move_changes']))
    if positions_with_q_correlation > 0:
        print(f"\n⚠️  QUIESCENCE CORRELATION")
        print(f"   {positions_with_q_correlation}/{total_positions} positions show move changes")
        print("   correlated with seldepth increases.")
        print("   This suggests quiescence is influencing move selection.")

if __name__ == "__main__":
    main()
