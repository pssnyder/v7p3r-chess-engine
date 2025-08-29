#!/usr/bin/env python3
"""
Test UCI delay functionality 
"""

import subprocess
import time

def test_uci_delay():
    """Test that the engine includes the delay for better visibility"""
    
    engine_path = "dist/V7P3R_v7.2.exe"
    
    try:
        process = subprocess.Popen(
            [engine_path], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0  # Unbuffered
        )
        
        # Test timing of responses
        commands = [
            "uci",
            "isready", 
            "position startpos",
            "go depth 3"
        ]
        
        for cmd in commands:
            print(f"> {cmd}")
            start_time = time.time()
            process.stdin.write(cmd + "\n")
            process.stdin.flush()
            
            if cmd.startswith("go"):
                # For go commands, wait for bestmove and measure timing
                output = ""
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    output += line
                    print(line.strip())
                    if "bestmove" in line:
                        end_time = time.time()
                        print(f"Time taken: {end_time - start_time:.2f} seconds")
                        break
            else:
                # For other commands, just wait for response
                time.sleep(0.1)
        
        # Clean shutdown
        process.stdin.write("quit\n")
        process.stdin.close()
        process.wait(timeout=5)
        
        print("✅ UCI delay test completed")
        
    except Exception as e:
        print(f"Test failed: {e}")
        if process:
            process.terminate()


if __name__ == "__main__":
    test_uci_delay()
