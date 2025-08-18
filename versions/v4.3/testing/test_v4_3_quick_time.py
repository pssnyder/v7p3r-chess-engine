#!/usr/bin/env python3
"""
Simple test to understand V7P3R v4.3 time usage patterns
"""

import subprocess
import time
import sys
import os

def quick_time_test():
    """Quick test to see timing patterns"""
    
    exe_path = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r_chess_engine\dist\V7P3R_v4.3.exe"
    
    print("V7P3R v4.3 Quick Time Analysis")
    print("=" * 40)
    
    if not os.path.exists(exe_path):
        print(f"ERROR: v4.3 executable not found")
        return
    
    try:
        # Start process
        process = subprocess.Popen(
            exe_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        
        # Quick init
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
        
        process.stdin.write("position startpos\n")
        process.stdin.flush()
        
        # Test different time allocations to see pattern
        time_tests = [10000, 5000, 3000, 2000, 1000]  # milliseconds
        
        for ms in time_tests:
            expected_time = ms / 1000.0
            
            print(f"\nTesting with {ms}ms remaining time:")
            print(f"Expected engine time allocation: ~{expected_time * 0.16:.2f}s (16%)")
            
            start_time = time.time()
            process.stdin.write(f"go wtime {ms} btime {ms}\n")
            process.stdin.flush()
            
            while True:
                line = process.stdout.readline().strip()
                if line.startswith("bestmove"):
                    break
            
            actual_time = time.time() - start_time
            print(f"Actual response time: {actual_time:.2f}s")
            
            ratio = actual_time / (expected_time * 0.16) if expected_time > 0 else 0
            print(f"Ratio (actual/expected): {ratio:.1f}x")
            
            if actual_time < expected_time * 0.16 * 2:  # Within 2x tolerance
                print("✅ REASONABLE")
            else:
                print("❌ TOO SLOW")
        
        # Test very short time
        print(f"\nEmergency time test (500ms):")
        start_time = time.time()
        process.stdin.write("go wtime 500 btime 500\n")
        process.stdin.flush()
        
        while True:
            line = process.stdout.readline().strip()
            if line.startswith("bestmove"):
                break
        
        actual_time = time.time() - start_time
        print(f"Actual response time: {actual_time:.2f}s")
        
        if actual_time < 0.5:
            print("✅ GOOD - Under emergency threshold")
        else:
            print("❌ CRITICAL - Too slow for emergency time")
        
        process.stdin.write("quit\n")
        process.stdin.flush()
        process.wait(timeout=2)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    quick_time_test()
