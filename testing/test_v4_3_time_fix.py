#!/usr/bin/env python3
"""
Quick time control test for V7P3R v4.3
Test if the critical time management fix is working
"""

import subprocess
import time
import sys
import os

def test_v43_time_control():
    """Test v4.3 with a specific time control"""
    
    exe_path = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r_chess_engine\dist\V7P3R_v4.3.exe"
    
    print("V7P3R v4.3 Time Control Fix Test")
    print("=" * 40)
    print(f"Testing: {os.path.basename(exe_path)}")
    
    if not os.path.exists(exe_path):
        print(f"ERROR: Executable not found at {exe_path}")
        return False
    
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
        while True:
            line = process.stdout.readline().strip()
            if line == "uciok":
                break
        
        process.stdin.write("isready\n")
        process.stdin.flush()
        while True:
            line = process.stdout.readline().strip()
            if line == "readyok":
                break
        
        # Set position
        process.stdin.write("position startpos\n")
        process.stdin.flush()
        
        # Test with 30 seconds - should take about 2.7s (9% of 30s)
        print("\nTest 1: 30 seconds remaining")
        print("Expected time: ~2.7s (new aggressive 9% allocation)")
        
        start_time = time.time()
        process.stdin.write("go wtime 30000 btime 30000\n")
        process.stdin.flush()
        
        while True:
            line = process.stdout.readline().strip()
            if line.startswith("bestmove"):
                break
            elif "[DEBUG]" in line:
                print(f"  {line}")
        
        actual_time = time.time() - start_time
        print(f"Actual response time: {actual_time:.2f}s")
        
        if actual_time <= 4.0:  # Allow some tolerance
            print("✅ SUCCESS - Time management is working!")
        else:
            print("❌ FAIL - Still taking too long")
        
        # Test with 5 seconds - should take about 1.0s (20% of 5s)
        print("\nTest 2: 5 seconds remaining")
        print("Expected time: ~1.0s (critical time 20% allocation)")
        
        start_time = time.time()
        process.stdin.write("go wtime 5000 btime 5000\n")
        process.stdin.flush()
        
        while True:
            line = process.stdout.readline().strip()
            if line.startswith("bestmove"):
                break
        
        actual_time = time.time() - start_time
        print(f"Actual response time: {actual_time:.2f}s")
        
        if actual_time <= 2.0:  # Allow some tolerance
            print("✅ SUCCESS - Critical time management working!")
        else:
            print("❌ FAIL - Still taking too long in critical time")
        
        # Quit
        process.stdin.write("quit\n")
        process.stdin.flush()
        process.wait(timeout=2)
        
        return True
        
    except Exception as e:
        print(f"Error testing executable: {e}")
        return False

if __name__ == "__main__":
    test_v43_time_control()
