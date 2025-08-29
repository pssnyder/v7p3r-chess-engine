#!/usr/bin/env python3
"""
Simple UCI test for mate reporting
"""

import subprocess
import sys
import time

def test_mate_reporting():
    """Test that the engine reports mates correctly via UCI"""
    
    # Test the back rank mate position
    engine_path = "dist/V7P3R_v7.2.exe"
    
    try:
        process = subprocess.Popen(
            [engine_path], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # UCI commands to test mate detection
        commands = [
            "uci",
            "isready", 
            "ucinewgame",
            "position fen 6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",  # Back rank mate
            "go depth 4",
            "quit"
        ]
        
        for cmd in commands:
            print(f"> {cmd}")
            process.stdin.write(cmd + "\n")
            process.stdin.flush()
            time.sleep(0.1)
        
        # Close stdin and wait for output
        process.stdin.close()
        
        # Read all output
        output, error = process.communicate(timeout=10)
        
        print("=== ENGINE OUTPUT ===")
        print(output)
        
        if error:
            print("=== ENGINE ERRORS ===")
            print(error)
            
        # Check if mate was detected
        if "mate " in output:
            print("✅ Mate detection working!")
        else:
            print("❌ Mate not detected in output")
            
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_mate_reporting()
