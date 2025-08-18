#!/usr/bin/env python3
"""
Direct V7P3R v4.3 Executable Time Test
Test the actual executable with UCI commands to measure real tournament-like performance.
"""

import subprocess
import time
import os

def test_executable_timing(exe_path):
    """Test the executable directly with UCI commands and measure response times."""
    print(f"Testing V7P3R v4.3 Executable Timing")
    print(f"Executable: {exe_path}")
    print("=" * 60)
    
    if not os.path.exists(exe_path):
        print(f"❌ Executable not found: {exe_path}")
        return
    
    # Test scenarios with different movetime settings
    test_scenarios = [
        ("1 second movetime", ["uci", "isready", "position startpos", "go movetime 1000", "quit"]),
        ("2 second movetime", ["uci", "isready", "position startpos", "go movetime 2000", "quit"]),
        ("3 second movetime", ["uci", "isready", "position startpos", "go movetime 3000", "quit"]),
        ("5 second movetime", ["uci", "isready", "position startpos", "go movetime 5000", "quit"]),
        ("Quick middle game", ["uci", "isready", "position startpos moves e2e4 e7e5 g1f3 b8c6", "go movetime 2000", "quit"]),
    ]
    
    results = []
    
    for scenario_name, uci_commands in test_scenarios:
        print(f"\n--- {scenario_name} ---")
        
        # Prepare UCI command sequence
        command_input = "\n".join(uci_commands) + "\n"
        print(f"UCI Commands: {' -> '.join(uci_commands[:-1])}")
        
        # Measure actual wall clock time
        start_time = time.time()
        
        try:
            # Run the executable with UCI commands
            result = subprocess.run(
                [exe_path],
                input=command_input,
                text=True,
                capture_output=True,
                timeout=20  # 20 second timeout
            )
            
            wall_time = time.time() - start_time
            
            # Parse the output
            lines = result.stdout.strip().split('\n')
            best_move = "No move found"
            info_lines = []
            
            for line in lines:
                if line.startswith('bestmove'):
                    best_move = line
                elif line.startswith('info') and ('depth' in line or 'time' in line or 'nodes' in line):
                    info_lines.append(line)
            
            print(f"  Wall clock time: {wall_time:.2f} seconds")
            print(f"  Result: {best_move}")
            
            # Show engine thinking info
            if info_lines:
                print(f"  Engine info (last 2 lines):")
                for line in info_lines[-2:]:
                    print(f"    {line}")
            
            # Check if stderr has any time-related debug info
            if result.stderr:
                stderr_lines = result.stderr.strip().split('\n')
                time_debug = [line for line in stderr_lines if 'time' in line.lower() or 'debug' in line.lower()]
                if time_debug:
                    print(f"  Debug info:")
                    for line in time_debug[-2:]:
                        print(f"    {line}")
            
            results.append({
                'scenario': scenario_name,
                'wall_time': wall_time,
                'move': best_move,
                'success': True
            })
            
        except subprocess.TimeoutExpired:
            wall_time = time.time() - start_time
            print(f"  ❌ TIMEOUT after {wall_time:.2f} seconds")
            results.append({
                'scenario': scenario_name,
                'wall_time': wall_time,
                'move': "TIMEOUT",
                'success': False
            })
        except Exception as e:
            wall_time = time.time() - start_time
            print(f"  ❌ ERROR: {e}")
            results.append({
                'scenario': scenario_name,
                'wall_time': wall_time,
                'move': "ERROR",
                'success': False
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("V7P3R v4.3 DIRECT EXECUTABLE TIMING SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = [r for r in results if r['success']]
    if successful_tests:
        avg_time = sum(r['wall_time'] for r in successful_tests) / len(successful_tests)
        print(f"Average response time: {avg_time:.2f} seconds")
        print(f"Success rate: {len(successful_tests)}/{len(results)} ({len(successful_tests)/len(results)*100:.1f}%)")
        
        print(f"\nDetailed timing results:")
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"  {status} {result['scenario']:<25} {result['wall_time']:.2f}s - {result['move']}")
        
        # Analysis
        over_limit = [r for r in successful_tests if r['wall_time'] > 6.0]
        if over_limit:
            print(f"\n⚠️  WARNING: {len(over_limit)} tests took longer than expected")
            print("The engine may still have timing issues in tournament conditions")
        else:
            print(f"\n✅ All tests completed within reasonable time limits")
    else:
        print("❌ No successful tests completed")

if __name__ == "__main__":
    exe_path = "dist/V7P3R_v4.3.exe"
    test_executable_timing(exe_path)
