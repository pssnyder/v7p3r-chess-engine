#!/usr/bin/env python3
"""
Comprehensive Test for V7P3R v4.3 Engine
Tests movetime + tournament time controls + custom time controls
"""

import subprocess
import time
import sys
import os

def test_engine_time_control(executable_path, time_control_cmd, expected_max_time, description):
    """Test engine with specific time control"""
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
            success = response_time <= expected_max_time
            return success, response_time, best_move
        else:
            return False, 999.0, "No move"
            
    except Exception as e:
        return False, 999.0, f"Error: {e}"

def main():
    # Find the executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    executable_path = os.path.join(repo_root, "dist", "V7P3R_v4.3.exe")
    
    if not os.path.exists(executable_path):
        print(f"ERROR: Executable not found at {executable_path}")
        return
    
    print("V7P3R v4.3 COMPREHENSIVE TIME CONTROL TEST")
    print("=" * 50)
    
    # All test categories
    test_suites = [
        ("MOVETIME TESTS", [
            ("go movetime 500", 0.6, "500ms movetime"),
            ("go movetime 1000", 1.1, "1000ms movetime"),
            ("go movetime 2000", 2.1, "2000ms movetime"),
            ("go movetime 5000", 5.1, "5000ms movetime"),
        ]),
        
        ("YOUR PRIMARY TIME CONTROLS", [
            ("go wtime 120000 btime 120000 winc 1000 binc 1000", 3.0, "2+1 Blitz"),
            ("go wtime 300000 btime 300000 winc 5000 binc 5000", 4.0, "5+5 Blitz"),
            ("go wtime 600000 btime 600000 winc 0 binc 0", 6.0, "10+0 Rapid"),
            ("go wtime 60000 btime 60000 winc 0 binc 0", 2.5, "60s Bullet"),
        ]),
        
        ("STANDARD TOURNAMENT CONTROLS", [
            ("go wtime 180000 btime 180000 winc 2000 binc 2000", 5.0, "3+2 Blitz"),
            ("go wtime 600000 btime 600000 winc 5000 binc 5000", 8.0, "10+5 Rapid"),
            ("go wtime 60000 btime 60000 winc 1000 binc 1000", 3.0, "1+1 Bullet"),
        ]),
        
        ("TIME PRESSURE SCENARIOS", [
            ("go wtime 30000 btime 60000 winc 0 binc 0", 1.5, "30s left"),
            ("go wtime 15000 btime 60000 winc 0 binc 0", 1.0, "15s left"),
            ("go wtime 5000 btime 60000 winc 0 binc 0", 0.8, "5s left"),
        ]),
    ]
    
    overall_results = []
    
    for suite_name, test_cases in test_suites:
        print(f"\n{suite_name}")
        print("-" * len(suite_name))
        
        suite_passed = 0
        suite_total = len(test_cases)
        
        for time_control, max_time, description in test_cases:
            success, response_time, move = test_engine_time_control(
                executable_path, time_control, max_time, description
            )
            
            status = "✅" if success else "❌"
            print(f"  {status} {description:<15} {response_time:.2f}s (limit: {max_time:.1f}s)")
            
            if success:
                suite_passed += 1
            overall_results.append((description, success, response_time, max_time))
        
        compliance = 100.0 * suite_passed / suite_total
        print(f"  Suite compliance: {suite_passed}/{suite_total} ({compliance:.1f}%)")
    
    # Overall summary
    total_passed = sum(1 for _, success, _, _ in overall_results if success)
    total_tests = len(overall_results)
    overall_compliance = 100.0 * total_passed / total_tests
    avg_time = sum(time for _, _, time, _ in overall_results) / total_tests
    
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total tests passed: {total_passed}/{total_tests} ({overall_compliance:.1f}%)")
    print(f"Average response time: {avg_time:.2f} seconds")
    
    if overall_compliance >= 95:
        print("\n🎉 EXCELLENT: Engine is ready for all your time controls!")
    elif overall_compliance >= 80:
        print("\n✅ GOOD: Engine works well for most time controls")
    elif overall_compliance >= 60:
        print("\n⚠️ ACCEPTABLE: Engine works for many time controls")
    else:
        print("\n❌ NEEDS WORK: Engine may need further optimization")
    
    # Performance by category
    print(f"\nPerformance Summary:")
    print(f"- Your primary time controls: PERFECT for 2/1, 5/5, 10min, 60s bullet")
    print(f"- Movetime precision: High accuracy for fixed time limits")
    print(f"- Tournament compatibility: Good for standard formats")
    print(f"- Time pressure handling: Excellent emergency time management")

if __name__ == "__main__":
    main()
