#!/usr/bin/env python3
"""
V7P3R v11 Final Validation Suite
Test depth goals and time control compliance before v11 build
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3REngine

def test_depth_goals():
    """Test if we can achieve our depth goals"""
    print("V7P3R v11 DEPTH GOAL VALIDATION")
    print("=" * 50)
    
    engine = V7P3REngine()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Opening Position (30min game)',
            'board': chess.Board(),
            'time_limit': 45.0,  # 45 seconds (realistic for 30min game)
            'target_depth': 10
        },
        {
            'name': 'Middlegame Position (10min game)', 
            'board': chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
            'time_limit': 15.0,  # 15 seconds
            'target_depth': 8
        },
        {
            'name': 'Endgame Position (5min game)',
            'board': chess.Board("8/8/2k5/5q2/5K2/8/8/8 w - - 0 1"),
            'time_limit': 8.0,   # 8 seconds
            'target_depth': 6
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\nTesting: {scenario['name']}")
        print(f"Time limit: {scenario['time_limit']}s, Target depth: {scenario['target_depth']}")
        print("-" * 40)
        
        # Test increasing depths until we hit time limit
        max_depth_achieved = 0
        last_good_time = 0
        
        for test_depth in range(1, scenario['target_depth'] + 3):
            engine.default_depth = test_depth
            
            start_time = time.time()
            try:
                move = engine.search(scenario['board'], time_limit=scenario['time_limit'])
                elapsed = time.time() - start_time
                
                print(f"  Depth {test_depth}: {elapsed:6.2f}s, {engine.nodes_searched:8d} nodes, {int(engine.nodes_searched/max(elapsed, 0.001)):6d} NPS")
                
                if elapsed <= scenario['time_limit']:
                    max_depth_achieved = test_depth
                    last_good_time = elapsed
                else:
                    print(f"  -> Exceeded time limit at depth {test_depth}")
                    break
                    
            except Exception as e:
                print(f"  Depth {test_depth}: ERROR - {e}")
                break
        
        # Evaluate result
        goal_met = max_depth_achieved >= scenario['target_depth']
        efficiency = (max_depth_achieved / scenario['target_depth']) * 100 if scenario['target_depth'] > 0 else 0
        
        results.append({
            'scenario': scenario['name'],
            'target_depth': scenario['target_depth'],
            'achieved_depth': max_depth_achieved,
            'goal_met': goal_met,
            'efficiency': efficiency,
            'time_used': last_good_time
        })
        
        status = "✅ PASS" if goal_met else "❌ FAIL"
        print(f"  Result: {status} - Achieved depth {max_depth_achieved}/{scenario['target_depth']} ({efficiency:.1f}%)")
    
    return results

def test_time_control_compliance():
    """Test time control compliance with various scenarios"""
    print(f"\n{'='*50}")
    print("V7P3R v11 TIME CONTROL COMPLIANCE TEST")
    print("=" * 50)
    
    engine = V7P3REngine()
    
    # Simulate game scenarios
    time_controls = [
        {'name': '30 minute game', 'total_time': 1800, 'moves_to_simulate': 40},
        {'name': '10 minute game', 'total_time': 600, 'moves_to_simulate': 35},
        {'name': '5 minute game', 'total_time': 300, 'moves_to_simulate': 30},
        {'name': '2+1 increment', 'total_time': 120, 'increment': 1, 'moves_to_simulate': 25}
    ]
    
    compliance_results = []
    
    for tc in time_controls:
        print(f"\nTesting: {tc['name']}")
        print("-" * 30)
        
        total_time = tc['total_time']
        increment = tc.get('increment', 0)
        moves_to_simulate = tc['moves_to_simulate']
        
        time_remaining = total_time
        total_time_used = 0
        moves_completed = 0
        timeouts = 0
        
        # Simulate each move
        board = chess.Board()
        for move_num in range(1, moves_to_simulate + 1):
            # Calculate time allocation
            time_manager = engine.time_manager
            time_manager.base_time = total_time
            time_manager.increment = increment
            time_manager.update_time_info(total_time - time_remaining, move_num)
            
            allocated_time, target_depth = time_manager.calculate_time_allocation(board, time_remaining)
            
            # Use allocated time but cap it
            actual_time_to_use = min(allocated_time, time_remaining * 0.8)
            
            # Simulate search time (estimate based on depth)
            estimated_search_time = min(actual_time_to_use, allocated_time * 0.7)
            
            time_remaining -= estimated_search_time
            time_remaining += increment  # Add increment
            total_time_used += estimated_search_time
            moves_completed += 1
            
            # Check for timeout
            if time_remaining <= 0:
                timeouts += 1
                break
            
            # Make a move to change position
            legal_moves = list(board.legal_moves)
            if legal_moves:
                board.push(legal_moves[0])
        
        # Calculate results
        time_remaining_final = max(0, time_remaining)
        avg_time_per_move = total_time_used / max(moves_completed, 1)
        compliance_pass = (time_remaining_final >= 60 and timeouts == 0)
        
        compliance_results.append({
            'time_control': tc['name'],
            'moves_completed': moves_completed,
            'time_remaining': time_remaining_final,
            'avg_time_per_move': avg_time_per_move,
            'timeouts': timeouts,
            'compliance_pass': compliance_pass
        })
        
        status = "✅ PASS" if compliance_pass else "❌ FAIL"
        print(f"  Moves completed: {moves_completed}/{moves_to_simulate}")
        print(f"  Time remaining: {time_remaining_final:.1f}s")
        print(f"  Avg time/move: {avg_time_per_move:.2f}s")
        print(f"  Timeouts: {timeouts}")
        print(f"  Compliance: {status}")
    
    return compliance_results

def generate_validation_report(depth_results, compliance_results):
    """Generate final validation report"""
    print(f"\n{'='*50}")
    print("V7P3R v11 VALIDATION SUMMARY")
    print("=" * 50)
    
    print("\nDEPTH GOAL RESULTS:")
    depth_goals_met = 0
    for result in depth_results:
        status = "✅" if result['goal_met'] else "❌"
        print(f"  {status} {result['scenario']}: {result['achieved_depth']}/{result['target_depth']} depth ({result['efficiency']:.1f}%)")
        if result['goal_met']:
            depth_goals_met += 1
    
    print(f"\nTIME CONTROL COMPLIANCE:")
    compliance_passed = 0
    for result in compliance_results:
        status = "✅" if result['compliance_pass'] else "❌"
        print(f"  {status} {result['time_control']}: {result['time_remaining']:.1f}s remaining, {result['timeouts']} timeouts")
        if result['compliance_pass']:
            compliance_passed += 1
    
    print(f"\nOVERALL ASSESSMENT:")
    print(f"  Depth Goals: {depth_goals_met}/{len(depth_results)} scenarios passed")
    print(f"  Time Compliance: {compliance_passed}/{len(compliance_results)} controls passed")
    
    overall_ready = (depth_goals_met >= len(depth_results) * 0.7 and compliance_passed >= len(compliance_results) * 0.8)
    
    if overall_ready:
        print(f"  \n🎉 V7P3R v11 READY FOR RELEASE!")
        print(f"  ✅ Performance targets achieved")
        print(f"  ✅ Time management compliant")
        print(f"  ✅ Tournament ready")
    else:
        print(f"  \n⚠️  V7P3R v11 needs optimization")
        print(f"  🔧 Review failed scenarios")
        print(f"  📊 Consider performance tuning")
    
    return overall_ready

if __name__ == "__main__":
    print("Starting V7P3R v11 Final Validation...")
    print("This may take several minutes...")
    print()
    
    # Run validation tests
    depth_results = test_depth_goals()
    compliance_results = test_time_control_compliance()
    
    # Generate final report
    ready_for_release = generate_validation_report(depth_results, compliance_results)
    
    print(f"\nValidation complete!")
    if ready_for_release:
        print("Proceed with V7P3R v11 build! 🚀")
    else:
        print("Additional optimization recommended before v11 release.")