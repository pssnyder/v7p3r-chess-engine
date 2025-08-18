#!/usr/bin/env python3
"""
Test V7P3R v4.3 Engine with User's Specific Time Controls
Tests: 2/1, 5/5, 10min, 60s bullet
"""

import subprocess
import time
import sys
import os

def test_engine_time_control(executable_path, time_control_cmd, expected_max_time, description):
    """Test engine with specific time control"""
    print(f"\n--- {description} ---")
    print(f"Time control: {time_control_cmd}")
    
    try:
        # Start engine process
        process = subprocess.Popen(
            [executable_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Initialize UCI protocol
        process.stdin.write("uci\n")
        process.stdin.flush()
        
        # Wait for uciok
        while True:
            line = process.stdout.readline()
            if "uciok" in line:
                break
            if not line:
                break
        
        # Send position and go command
        process.stdin.write("position startpos\n")
        process.stdin.flush()
        
        # Measure time for move
        start_time = time.time()
        process.stdin.write(f"{time_control_cmd}\n")
        process.stdin.flush()
        
        # Wait for bestmove
        move_found = False
        best_move = None
        while True:
            line = process.stdout.readline()
            if line.startswith("bestmove"):
                end_time = time.time()
                move_found = True
                best_move = line.strip()
                break
            if not line:
                break
        
        # Clean up
        try:
            process.stdin.write("quit\n")
            process.stdin.flush()
        except:
            pass
        process.terminate()
        
        if move_found:
            response_time = end_time - start_time
            print(f"  Response time: {response_time:.2f} seconds")
            print(f"  Move: {best_move}")
            
            if response_time <= expected_max_time:
                print(f"  ✅ Good timing for this time control")
                return True, response_time
            else:
                print(f"  ⚠️ WARNING: Took {response_time:.2f}s, expected ≤{expected_max_time:.1f}s")
                return False, response_time
        else:
            print(f"  ❌ ERROR: No move received")
            return False, 999.0
            
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False, 999.0

def main():
    # Find the executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    executable_path = os.path.join(repo_root, "dist", "V7P3R_v4.3.exe")
    
    if not os.path.exists(executable_path):
        print(f"ERROR: Executable not found at {executable_path}")
        return
    
    print("Testing V7P3R v4.3 Engine with Your Custom Time Controls")
    print("2/1, 5/5, 10min, 60s bullet")
    print("=" * 65)
    
    # Test cases for your specific time controls
    test_cases = [
        # time_control_cmd, max_expected_time, description
        ("go wtime 120000 btime 120000 winc 1000 binc 1000", 3.0, "2+1 Blitz (2 min + 1 sec increment)"),
        ("go wtime 300000 btime 300000 winc 5000 binc 5000", 4.0, "5+5 Blitz (5 min + 5 sec increment)"),
        ("go wtime 600000 btime 600000 winc 0 binc 0", 6.0, "10+0 Rapid (10 minutes)"),
        ("go wtime 60000 btime 60000 winc 0 binc 0", 2.5, "60s Bullet (1 minute)"),
        
        # Additional edge cases
        ("go wtime 30000 btime 60000 winc 0 binc 0", 1.5, "Low time (30 seconds left)"),
        ("go wtime 15000 btime 60000 winc 0 binc 0", 1.0, "Very low time (15 seconds left)"),
    ]
    
    results = []
    total_time = 0
    passed = 0
    
    for time_control, max_time, description in test_cases:
        success, response_time = test_engine_time_control(
            executable_path, time_control, max_time, description
        )
        results.append((description, success, response_time, max_time))
        total_time += response_time
        if success:
            passed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("V7P3R v4.3 CUSTOM TIME CONTROL COMPLIANCE SUMMARY")
    print("=" * 70)
    print(f"Average response time: {total_time / len(results):.2f} seconds")
    print(f"Time control compliance: {passed}/{len(results)} ({100.0 * passed / len(results):.1f}%)")
    print()
    print("Detailed Results:")
    
    for description, success, response_time, max_time in results:
        status = "✅" if success else "❌"
        print(f"  {status} {description:<35} {response_time:.2f}s (limit: {max_time:4.1f}s)")
    
    if passed == len(results):
        print("\n🎉 EXCELLENT: Engine meets all your time control requirements!")
    elif passed >= len(results) * 0.8:
        print("\n✅ GOOD: Engine meets most of your time control requirements")
    elif passed >= len(results) * 0.5:
        print("\n⚠️ ACCEPTABLE: Engine meets some time control requirements")
    else:
        print("\n❌ CONCERN: Engine may be too slow for your time controls")

if __name__ == "__main__":
    main()
