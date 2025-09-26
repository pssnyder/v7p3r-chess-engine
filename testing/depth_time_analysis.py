#!/usr/bin/env python3
"""
V7P3R Depth vs Time Analysis
Simple test to measure how long each depth takes and identify bottlenecks
"""

import sys
import time
import chess
import os

sys.path.append('src')
from v7p3r import V7P3REngine

def test_depth_performance():
    """Test how long each depth takes with current engine"""
    
    print("V7P3R v12.0 - Depth vs Time Analysis")
    print("=" * 50)
    
    engine = V7P3REngine()
    test_board = chess.Board()  # Starting position
    
    print("Testing search depths 1-8 from starting position")
    print("Each test uses a 30-second time limit to avoid timeout interference")
    print()
    print("Depth | Time (s) | Nodes    | NPS     | Target")
    print("------|----------|----------|---------|--------")
    
    results = []
    
    for depth in range(1, 9):
        # Reset engine state
        engine = V7P3REngine()
        engine.default_depth = depth
        
        # Time the search
        start_time = time.perf_counter()
        
        try:
            move = engine.search(test_board, time_limit=30.0)
            elapsed = time.perf_counter() - start_time
            
            nps = int(engine.nodes_searched / max(elapsed, 0.001))
            
            # Performance targets based on typical expectations
            if depth <= 4:
                target = "< 1s"
            elif depth <= 6:
                target = "< 5s" 
            elif depth <= 8:
                target = "< 15s"
            else:
                target = "< 30s"
            
            print(f"  {depth:2d}  | {elapsed:8.2f} | {engine.nodes_searched:8,d} | {nps:7,d} | {target}")
            
            results.append({
                'depth': depth,
                'time': elapsed,
                'nodes': engine.nodes_searched,
                'nps': nps,
                'move': str(move)
            })
            
        except Exception as e:
            print(f"  {depth:2d}  | ERROR: {e}")
    
    print()
    print("Analysis:")
    print("-" * 30)
    
    if len(results) >= 4:
        # Check depth 4 performance (should be < 1 second)
        depth4 = next((r for r in results if r['depth'] == 4), None)
        if depth4:
            if depth4['time'] > 5.0:
                print("🚨 CRITICAL: Depth 4 takes > 5 seconds - major optimization needed")
            elif depth4['time'] > 2.0:
                print("⚠️  WARNING: Depth 4 takes > 2 seconds - optimization recommended")
            else:
                print("✅ Depth 4 performance acceptable")
    
    # Check NPS consistency
    if results:
        avg_nps = sum(r['nps'] for r in results) / len(results)
        print(f"Average NPS: {avg_nps:,.0f}")
        
        if avg_nps < 1000:
            print("🚨 CRITICAL: NPS < 1,000 - severe performance bottleneck")
        elif avg_nps < 5000:
            print("🚨 CRITICAL: NPS < 5,000 - major performance issues")
        elif avg_nps < 20000:
            print("⚠️  WARNING: NPS < 20,000 - needs optimization")
        elif avg_nps < 50000:
            print("✅ Moderate performance - room for improvement")
        else:
            print("✅ Good performance")
    
    # Time management recommendations
    print("\nTime Management Recommendations:")
    print("-" * 35)
    
    if len(results) >= 6:
        depth6 = next((r for r in results if r['depth'] == 6), None)
        if depth6:
            if depth6['time'] < 2.0:
                print("• Time allocation too conservative - can afford deeper search")
            elif depth6['time'] > 10.0:
                print("• Time allocation too aggressive - risk of timeout")
            else:
                print("• Time allocation seems reasonable for depth 6")
    
    return results

def analyze_uci_time_management():
    """Analyze current UCI time management calculations"""
    
    print("\n" + "=" * 50)
    print("UCI TIME MANAGEMENT ANALYSIS")
    print("=" * 50)
    
    # Simulate different time controls
    scenarios = [
        {"name": "1+1 Blitz", "wtime": 60000, "increment": 1000, "moves": 10},
        {"name": "10+5 Blitz", "wtime": 600000, "increment": 5000, "moves": 20},
        {"name": "30+1 Rapid", "wtime": 1800000, "increment": 1000, "moves": 30},
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']} - Remaining time: {scenario['wtime']/1000:.0f}s")
        print("-" * 40)
        
        remaining_time = scenario['wtime'] / 1000.0
        moves_played = scenario['moves']
        
        # Current time management logic from v7p3r_uci.py
        if moves_played < 20:
            time_factor = 25.0  # Early game: use 1/25th
        elif moves_played < 40:
            time_factor = 30.0  # Mid game: use 1/30th  
        else:
            time_factor = 40.0  # End game: use 1/40th
        
        time_limit = min(remaining_time / time_factor, 10.0)
        
        print(f"Move {moves_played}: {time_limit:.2f}s allocated ({time_factor:.0f}x factor)")
        print(f"Percentage of remaining time: {(time_limit/remaining_time)*100:.1f}%")
        
        # Calculate moves until time runs out at this rate
        moves_left = remaining_time / time_limit
        print(f"Estimated moves possible: {moves_left:.0f}")

def main():
    """Run complete depth and time analysis"""
    
    # Test current performance
    results = test_depth_performance()
    
    # Analyze time management
    analyze_uci_time_management()
    
    # Summary recommendations
    print("\n" + "=" * 50)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 50)
    
    if results:
        # Find bottleneck depth
        slow_depths = [r for r in results if r['time'] > 5.0 and r['depth'] <= 6]
        if slow_depths:
            print("🎯 PRIMARY ISSUE: Search too slow for tournament play")
            print("   - Focus on evaluation optimization")
            print("   - Reduce expensive king safety calculations")
            print("   - Minimize FEN generation calls")
        
        fast_depths = [r for r in results if r['time'] < 1.0 and r['depth'] >= 4]
        if fast_depths and not slow_depths:
            print("🎯 PRIMARY ISSUE: Time management too conservative")
            print("   - Can afford to search deeper")
            print("   - Reduce time factors in UCI interface")
    
    print("\n📋 Next Steps for V7P3R v12.2:")
    print("1. Run profiler on evaluation functions")
    print("2. Compare with v10.8 time management settings")
    print("3. Optimize king safety evaluation")
    print("4. Reduce FEN string generation")
    print("5. Test revised time factors")

if __name__ == "__main__":
    main()