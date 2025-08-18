#!/usr/bin/env python3
"""
Test V7P3R v4.3 Engine Time Controls - Same test as v4.1/v4.2
Test actual time control compliance using wtime/btime like in tournaments.
"""

import subprocess
import time
import os

def test_v4_3_time_controls():
    """Test v4.3 engine with actual time control scenarios like tournaments use."""
    print("Testing V7P3R v4.3 Engine Time Controls")
    print("Same test as performed on v4.1 and v4.2")
    print("=" * 60)
    
    exe_path = "dist/V7P3R_v4.3.exe"
    
    if not os.path.exists(exe_path):
        print(f"❌ Engine not found: {exe_path}")
        return
    
    # Tournament-style time control tests
    test_scenarios = [
        {
            "name": "Blitz 3+2 (180 seconds, 2 sec increment)",
            "commands": [
                "uci",
                "isready", 
                "position startpos",
                "go wtime 180000 btime 180000 winc 2000 binc 2000",
                "quit"
            ]
        },
        {
            "name": "Rapid 10+5 (600 seconds, 5 sec increment)", 
            "commands": [
                "uci",
                "isready",
                "position startpos", 
                "go wtime 600000 btime 600000 winc 5000 binc 5000",
                "quit"
            ]
        },
        {
            "name": "Bullet 1+1 (60 seconds, 1 sec increment)",
            "commands": [
                "uci",
                "isready",
                "position startpos",
                "go wtime 60000 btime 60000 winc 1000 binc 1000", 
                "quit"
            ]
        },
        {
            "name": "Low time pressure (10 seconds left)",
            "commands": [
                "uci",
                "isready",
                "position startpos moves e2e4 e7e5",
                "go wtime 10000 btime 60000 winc 0 binc 0",
                "quit"
            ]
        },
        {
            "name": "Critical time (5 seconds left)",
            "commands": [
                "uci", 
                "isready",
                "position startpos moves e2e4 e7e5 g1f3",
                "go wtime 5000 btime 60000 winc 0 binc 0",
                "quit"
            ]
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        # Show the time control being tested
        go_command = [cmd for cmd in scenario['commands'] if cmd.startswith('go')][0]
        print(f"Time control: {go_command}")
        
        command_input = "\n".join(scenario['commands']) + "\n"
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [exe_path],
                input=command_input,
                text=True,
                capture_output=True,
                timeout=30
            )
            
            elapsed_time = time.time() - start_time
            
            # Parse output
            best_move = "No move found"
            for line in result.stdout.split('\n'):
                if line.startswith('bestmove'):
                    best_move = line.strip()
                    break
            
            print(f"  Response time: {elapsed_time:.2f} seconds")
            print(f"  Move: {best_move}")
            
            # Determine if this is reasonable for the time control
            expected_time = None
            if "Bullet" in scenario['name']:
                expected_time = 3.0  # Should be very fast for bullet
            elif "Blitz" in scenario['name']:
                expected_time = 5.0  # Should be reasonable for blitz
            elif "10 seconds left" in scenario['name']:
                expected_time = 2.0  # Should be quick when low on time
            elif "5 seconds left" in scenario['name']:
                expected_time = 1.5  # Should be very quick in time pressure
            else:
                expected_time = 8.0  # Rapid can take a bit longer
            
            status = "✅" if elapsed_time <= expected_time else "⚠️"
            if elapsed_time > expected_time:
                print(f"  {status} WARNING: Took {elapsed_time:.2f}s, expected ≤{expected_time:.1f}s")
            else:
                print(f"  {status} Good timing for this time control")
            
            results.append({
                'scenario': scenario['name'],
                'time': elapsed_time,
                'expected': expected_time,
                'move': best_move,
                'within_limit': elapsed_time <= expected_time
            })
            
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            print(f"  ❌ TIMEOUT after {elapsed_time:.2f} seconds")
            results.append({
                'scenario': scenario['name'],
                'time': elapsed_time,
                'expected': 30.0,
                'move': "TIMEOUT",
                'within_limit': False
            })
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"  ❌ ERROR: {e}")
            results.append({
                'scenario': scenario['name'],
                'time': elapsed_time,
                'expected': 30.0,
                'move': "ERROR",
                'within_limit': False
            })
    
    # Summary
    print(f"\n{'='*70}")
    print("V7P3R v4.3 TIME CONTROL COMPLIANCE SUMMARY")
    print(f"{'='*70}")
    
    within_limit = [r for r in results if r['within_limit']]
    avg_time = sum(r['time'] for r in results) / len(results)
    
    print(f"Average response time: {avg_time:.2f} seconds")
    print(f"Time control compliance: {len(within_limit)}/{len(results)} ({len(within_limit)/len(results)*100:.1f}%)")
    
    print(f"\nDetailed Results:")
    for r in results:
        status = "✅" if r['within_limit'] else "❌"
        print(f"  {status} {r['scenario']:<35} {r['time']:>6.2f}s (limit: {r['expected']:>4.1f}s)")
    
    if len(within_limit) == len(results):
        print(f"\n🎉 EXCELLENT: All time controls respected!")
    elif len(within_limit) >= len(results) * 0.8:
        print(f"\n✅ GOOD: Most time controls respected")
    else:
        print(f"\n⚠️ CONCERN: Engine may still be too slow for tournament play")
        print(f"   This explains the 10-12 second delays you're seeing in tournaments")

if __name__ == "__main__":
    test_v4_3_time_controls()
