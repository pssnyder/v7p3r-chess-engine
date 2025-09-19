#!/usr/bin/env python3
"""
Quick V7P3R v11 Search Depth Validation
Test search depth performance after dynamic move selector integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3REngine

def test_search_depth_performance():
    """Test search depth achievement with time constraints"""
    print("V7P3R v11 SEARCH DEPTH VALIDATION")
    print("(After Dynamic Move Selector Integration)")
    print("=" * 50)
    
    engine = V7P3REngine()
    
    # Test scenarios with realistic time limits
    test_scenarios = [
        {
            'name': 'Opening Position (30min game)',
            'board': chess.Board(),
            'time_limit': 45.0,
            'target_depth': 6  # Realistic target
        },
        {
            'name': 'Middlegame Position (10min game)', 
            'board': chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
            'time_limit': 15.0,
            'target_depth': 6  # Ambitious target
        },
        {
            'name': 'Tactical Position (5min game)',
            'board': chess.Board("r2qkb1r/ppp2ppp/2n1bn2/3pp3/3PP3/3B1N2/PPP2PPP/RNBQK2R w KQkq - 4 6"),
            'time_limit': 10.0,
            'target_depth': 6
        },
        {
            'name': 'Endgame Position (5min game)',
            'board': chess.Board("8/8/2k5/5q2/5K2/8/8/8 w - - 0 1"),
            'time_limit': 8.0,
            'target_depth': 7  # Endgames should be deeper
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}")
        print(f"Time limit: {scenario['time_limit']}s, Target depth: {scenario['target_depth']}")
        print("-" * 40)
        
        max_depth_achieved = 0
        last_good_time = 0
        depth_times = []
        
        for test_depth in range(1, scenario['target_depth'] + 3):
            engine.default_depth = test_depth
            
            start_time = time.time()
            try:
                move = engine.search(scenario['board'], time_limit=scenario['time_limit'])
                elapsed = time.time() - start_time
                nps = int(engine.nodes_searched / max(elapsed, 0.001))
                
                depth_times.append((test_depth, elapsed, engine.nodes_searched, nps))
                
                print(f"  Depth {test_depth}: {elapsed:6.2f}s, {engine.nodes_searched:8,} nodes, {nps:6,} NPS")
                
                if elapsed <= scenario['time_limit']:
                    max_depth_achieved = test_depth
                    last_good_time = elapsed
                else:
                    print(f"  -> Exceeded time limit at depth {test_depth}")
                    break
                    
            except Exception as e:
                print(f"  Depth {test_depth}: ERROR - {e}")
                break
        
        # Calculate efficiency metrics
        goal_met = max_depth_achieved >= scenario['target_depth']
        efficiency = (max_depth_achieved / scenario['target_depth']) * 100 if scenario['target_depth'] > 0 else 0
        
        # Calculate growth factor
        growth_factor = None
        if len(depth_times) >= 2:
            last_time = depth_times[-1][1]
            prev_time = depth_times[-2][1] 
            growth_factor = last_time / prev_time if prev_time > 0 else float('inf')
        
        results.append({
            'scenario': scenario['name'],
            'target_depth': scenario['target_depth'],
            'achieved_depth': max_depth_achieved,
            'goal_met': goal_met,
            'efficiency': efficiency,
            'time_used': last_good_time,
            'growth_factor': growth_factor,
            'depth_times': depth_times
        })
        
        status = "✅ PASS" if goal_met else "❌ FAIL"
        print(f"  Result: {status} - Achieved depth {max_depth_achieved}/{scenario['target_depth']} ({efficiency:.1f}%)")
        if growth_factor:
            print(f"  Growth factor: {growth_factor:.2f}x per depth")
    
    # Generate summary
    print(f"\n{'='*50}")
    print("SEARCH DEPTH VALIDATION SUMMARY")
    print("=" * 50)
    
    goals_met = sum(1 for r in results if r['goal_met'])
    total_scenarios = len(results)
    
    print(f"\nDepth Goals: {goals_met}/{total_scenarios} scenarios passed")
    
    for result in results:
        status = "✅" if result['goal_met'] else "❌"
        growth = f", {result['growth_factor']:.2f}x growth" if result['growth_factor'] else ""
        print(f"  {status} {result['scenario']}: {result['achieved_depth']}/{result['target_depth']} depth ({result['efficiency']:.1f}%{growth})")
    
    # Performance analysis
    print(f"\nPERFORMANCE ANALYSIS:")
    avg_growth = sum(r['growth_factor'] for r in results if r['growth_factor']) / len([r for r in results if r['growth_factor']])
    print(f"  Average growth factor: {avg_growth:.2f}x per depth")
    
    if avg_growth <= 3.0:
        print(f"  ✅ Growth factor excellent (<= 3.0x)")
    elif avg_growth <= 4.0:
        print(f"  ✅ Growth factor good (<= 4.0x)")
    else:
        print(f"  ⚠️  Growth factor needs improvement (> 4.0x)")
    
    validation_passed = goals_met >= total_scenarios * 0.75  # 75% pass rate
    
    if validation_passed:
        print(f"\n🎉 V7P3R v11 SEARCH DEPTH VALIDATION PASSED!")
        print(f"  ✅ Dynamic move selector working")
        print(f"  ✅ Depth targets achievable")
        print(f"  ✅ Performance optimization successful")
    else:
        print(f"\n⚠️  V7P3R v11 needs more optimization")
        print(f"  🔧 Review failing scenarios")
    
    return validation_passed

if __name__ == "__main__":
    passed = test_search_depth_performance()
    print(f"\n{'Ready for v11 acceptance testing!' if passed else 'Needs more work...'}")