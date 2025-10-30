#!/usr/bin/env python3
"""
Measure V14.6 Performance Improvements
Compare NPS (nodes per second) across game phases
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3REngine

def measure_nps(engine, board, depth=4, time_limit=10.0):
    """Measure NPS for a given position"""
    start_time = time.time()
    
    # Run search (returns just best_move)
    best_move = engine.search(board, time_limit, depth)
    
    elapsed = time.time() - start_time
    nodes = engine.nodes_searched
    nps = nodes / elapsed if elapsed > 0 else 0
    
    # Get score from last search (if available)
    score = engine.best_score if hasattr(engine, 'best_score') else 0
    
    return {
        'move': best_move,
        'score': score,
        'nodes': nodes,
        'time': elapsed,
        'nps': nps,
        'depth_reached': depth
    }

def test_phase_performance():
    """Test performance across different game phases"""
    
    engine = V7P3REngine()
    
    positions = [
        {
            'name': 'Opening (Move 1)',
            'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'expected_phase': 'opening',
            'target_nps': 3000,  # Target: +20% from V14.5 (~2500)
        },
        {
            'name': 'Opening (Move 5)',
            'fen': 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 5',
            'expected_phase': 'opening',
            'target_nps': 3000,
        },
        {
            'name': 'Middlegame (Tactical)',
            'fen': 'r1bq1rk1/ppp2ppp/2n5/3p4/2PP4/2N2N2/PP3PPP/R1BQ1RK1 w - - 0 10',
            'expected_phase': 'middlegame',
            'target_nps': 2400,  # Target: +8% from V14.5 (~2200)
        },
        {
            'name': 'Middlegame (Complex)',
            'fen': '1r3rk1/p4ppp/2p5/q1n1p3/2B1P3/2N2Q1P/PP3PP1/3R1RK1 w - - 0 20',
            'expected_phase': 'middlegame',
            'target_nps': 2400,
        },
        {
            'name': 'Early Endgame (Rook)',
            'fen': '8/5pk1/6p1/8/8/6P1/5PK1/3R4 w - - 0 40',
            'expected_phase': 'early_endgame',
            'target_nps': 3500,  # Target: +25% from V14.5 (~2800)
        },
        {
            'name': 'Late Endgame (K+P)',
            'fen': '8/8/4k3/8/3K4/8/3P4/8 w - - 0 50',
            'expected_phase': 'late_endgame',
            'target_nps': 4500,  # Target: +35% from V14.5 (~3300)
        },
    ]
    
    print("=" * 80)
    print("V14.6 Performance Measurement - NPS per Phase")
    print("=" * 80)
    print()
    
    results = []
    
    for pos_info in positions:
        board = chess.Board(pos_info['fen'])
        
        print(f"Testing: {pos_info['name']}")
        print(f"Expected phase: {pos_info['expected_phase']}")
        
        # Warm-up run (not counted)
        engine.nodes_searched = 0
        _ = engine.search(board, depth=3, time_limit=1.0)
        
        # Actual measurement
        engine.nodes_searched = 0
        result = measure_nps(engine, board, depth=4, time_limit=5.0)
        
        nps_achievement = (result['nps'] / pos_info['target_nps']) * 100
        
        print(f"  Nodes: {result['nodes']:,}")
        print(f"  Time: {result['time']:.2f}s")
        print(f"  NPS: {result['nps']:.0f} (target: {pos_info['target_nps']}, {nps_achievement:.0f}%)")
        print(f"  Move: {result['move']}, Score: {result['score']:.1f}")
        
        if result['nps'] >= pos_info['target_nps']:
            print(f"  OK Target achieved!")
        else:
            deficit = pos_info['target_nps'] - result['nps']
            print(f"  WARN {deficit:.0f} NPS below target")
        
        print()
        
        results.append({
            **pos_info,
            **result
        })
    
    # Summary
    print("=" * 80)
    print("Summary by Phase")
    print("=" * 80)
    print()
    
    phases = ['opening', 'middlegame', 'early_endgame', 'late_endgame']
    for phase in phases:
        phase_results = [r for r in results if r['expected_phase'] == phase]
        if not phase_results:
            continue
        
        avg_nps = sum(r['nps'] for r in phase_results) / len(phase_results)
        avg_target = sum(r['target_nps'] for r in phase_results) / len(phase_results)
        achievement = (avg_nps / avg_target) * 100
        
        print(f"{phase.upper()}")
        print(f"  Average NPS: {avg_nps:.0f}")
        print(f"  Target NPS: {avg_target:.0f}")
        print(f"  Achievement: {achievement:.0f}%")
        
        if achievement >= 100:
            print(f"  OK Performance target met!")
        else:
            print(f"  WARN {100-achievement:.0f}% below target")
        print()
    
    # Overall
    total_avg_nps = sum(r['nps'] for r in results) / len(results)
    total_avg_target = sum(r['target_nps'] for r in results) / len(results)
    overall_achievement = (total_avg_nps / total_avg_target) * 100
    
    print("OVERALL")
    print(f"  Average NPS: {total_avg_nps:.0f}")
    print(f"  Target NPS: {total_avg_target:.0f}")
    print(f"  Achievement: {overall_achievement:.0f}%")
    print()
    
    if overall_achievement >= 90:
        print("SUCCESS V14.6 performance goals achieved!")
        return True
    else:
        print("WARN Performance below target - may need further optimization")
        return False

if __name__ == "__main__":
    print("Testing V14.6 Phase-Based Performance\n")
    success = test_phase_performance()
    sys.exit(0 if success else 1)
