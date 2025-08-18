#!/usr/bin/env python3
"""
Real-world time control test for V7P3R executables
Tests actual response times with Arena-style time controls
"""

import subprocess
import time
import sys
import os

def test_exe_time_control(exe_path, remaining_time_ms, expected_time_limit):
    """Test executable with actual time control and measure response time"""
    
    print(f"\nTesting: {os.path.basename(exe_path)}")
    print(f"Remaining time: {remaining_time_ms/1000:.1f}s")
    print(f"Expected time limit: {expected_time_limit:.2f}s")
    print("-" * 50)
    
    if not os.path.exists(exe_path):
        print(f"ERROR: Executable not found at {exe_path}")
        return None
    
    try:
        # Start the process
        process = subprocess.Popen(
            exe_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        
        # Initialize UCI
        process.stdin.write("uci\n")
        process.stdin.flush()
        
        # Wait for uciok
        while True:
            line = process.stdout.readline().strip()
            if line == "uciok":
                break
        
        # Set ready
        process.stdin.write("isready\n")
        process.stdin.flush()
        
        # Wait for readyok
        while True:
            line = process.stdout.readline().strip()
            if line == "readyok":
                break
        
        # Set position
        process.stdin.write("position startpos\n")
        process.stdin.flush()
        
        # Send go command with time control - measure actual response time
        go_command = f"go wtime {remaining_time_ms} btime {remaining_time_ms}\n"
        print(f"Sending: {go_command.strip()}")
        
        start_time = time.time()
        process.stdin.write(go_command)
        process.stdin.flush()
        
        # Read until bestmove
        search_output = []
        while True:
            line = process.stdout.readline().strip()
            if line:
                search_output.append(line)
                if line.startswith("bestmove"):
                    break
        
        actual_time = time.time() - start_time
        
        # Quit
        process.stdin.write("quit\n")
        process.stdin.flush()
        
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        
        # Show results
        print(f"Actual response time: {actual_time:.2f}s")
        print(f"Expected time limit: {expected_time_limit:.2f}s")
        print(f"Ratio (actual/expected): {actual_time/expected_time_limit:.1f}x")
        
        if actual_time > expected_time_limit * 1.5:  # Allow 50% tolerance
            print("❌ TIMEOUT - Engine exceeded time limit!")
        else:
            print("✅ OK - Engine responded within reasonable time")
        
        # Show some search output
        print("\nSearch output (last 5 lines):")
        for line in search_output[-5:]:
            print(f"  {line}")
        
        return actual_time
        
    except Exception as e:
        print(f"Error testing executable: {e}")
        return None

def run_time_control_tests():
    """Run a series of time control tests"""
    
    v41_exe = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r_chess_engine\dist\V7P3R_v4.1.exe"
    v42_exe = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r_chess_engine\dist\V7P3R_v4.2.exe"
    
    print("V7P3R Executable Time Control Testing")
    print("=" * 60)
    print("Testing real Arena-style time controls...")
    
    # Test scenarios (remaining_time_ms, expected_limit_with_old_2%_logic)
    scenarios = [
        (120000, 2.4),   # 2 minutes remaining -> old logic: 2.4s
        (90000, 1.8),    # 1.5 minutes remaining -> old logic: 1.8s  
        (60000, 1.2),    # 1 minute remaining -> old logic: 1.2s
        (30000, 0.6),    # 30 seconds remaining -> old logic: 0.6s
        (15000, 0.3),    # 15 seconds remaining -> old logic: 0.3s
        (5000, 0.1),     # 5 seconds remaining -> old logic: 0.1s
    ]
    
    print("\nTesting V7P3R v4.1 (old time management)...")
    print("=" * 50)
    
    v41_times = []
    for remaining_ms, expected_limit in scenarios:
        actual_time = test_exe_time_control(v41_exe, remaining_ms, expected_limit)
        if actual_time:
            v41_times.append((remaining_ms, expected_limit, actual_time))
    
    print("\n\nTesting V7P3R v4.2 (new time management)...")
    print("=" * 50)
    
    v42_times = []
    for remaining_ms, expected_limit in scenarios:
        # Calculate new expected limit for v4.2
        remaining_seconds = remaining_ms / 1000.0
        if remaining_seconds > 60:
            new_expected = max(0.5, remaining_seconds * 0.07)
        elif remaining_seconds > 30:
            new_expected = max(0.3, remaining_seconds * 0.09)
        elif remaining_seconds > 10:
            new_expected = max(0.2, remaining_seconds * 0.13)
        else:
            new_expected = max(0.1, remaining_seconds * 0.20)
        
        actual_time = test_exe_time_control(v42_exe, remaining_ms, new_expected)
        if actual_time:
            v42_times.append((remaining_ms, new_expected, actual_time))
    
    # Summary comparison
    print("\n\nSUMMARY COMPARISON")
    print("=" * 70)
    print("Remaining | v4.1 Actual | v4.1 Expected | v4.2 Actual | v4.2 Expected")
    print("-" * 70)
    
    for i, (remaining_ms, expected_limit) in enumerate(scenarios):
        v41_data = v41_times[i] if i < len(v41_times) else (remaining_ms, 0, 0)
        v42_data = v42_times[i] if i < len(v42_times) else (remaining_ms, 0, 0)
        
        print(f"{remaining_ms/1000:>7.0f}s  | {v41_data[2]:>9.2f}s | {v41_data[1]:>10.2f}s | {v42_data[2]:>9.2f}s | {v42_data[1]:>10.2f}s")

if __name__ == "__main__":
    run_time_control_tests()
