#!/usr/bin/env python3
"""
Quick test to verify V7P3R with mate scoring fix works in UCI mode
"""

import subprocess
import time

def test_uci_mate_fix():
    """Test the UCI interface with the mate scoring fix"""
    print("Testing V7P3R UCI interface with mate scoring fix...")
    
    # Start the engine
    try:
        process = subprocess.Popen(
            ['python', 'src/v7p3r_uci.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine'
        )
        
        # Send UCI commands
        commands = [
            "uci",
            "isready", 
            "position startpos moves e2e4 e7e5 f1c4 b8c6 d1h5 g8f6",  # Setting up Scholar's mate threat
            "go depth 4"
        ]
        
        for cmd in commands:
            print(f">>> {cmd}")
            process.stdin.write(cmd + '\n')
            process.stdin.flush()
            time.sleep(0.5)
            
        # Read output for a few seconds
        time.sleep(3)
        process.stdin.write("quit\n")
        process.stdin.flush()
        
        # Get output
        stdout, stderr = process.communicate(timeout=5)
        
        print("Engine output:")
        print(stdout)
        
        if stderr:
            print("Errors:")
            print(stderr)
            
        # Check for mate scores in output
        if "mate" in stdout and not ("M500" in stdout or "M384" in stdout):
            print("\n✅ SUCCESS: Engine is reporting realistic mate scores!")
        elif "M500" in stdout or "M384" in stdout:
            print("\n❌ FAILED: Still seeing inflated mate scores")
        else:
            print("\n⚠️  UNCERTAIN: No mate scores detected in output")
            
    except Exception as e:
        print(f"Error testing UCI: {e}")
        

if __name__ == "__main__":
    test_uci_mate_fix()
