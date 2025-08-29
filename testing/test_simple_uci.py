#!/usr/bin/env python3
"""
Simple UCI test for V7P3R v7.2
"""

import subprocess
import time

def simple_uci_test():
    """Simple UCI test to verify basic functionality"""
    print("Simple UCI test for V7P3R v7.2...")
    
    try:
        process = subprocess.Popen(
            ['python', 'src/v7p3r_uci.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine'
        )
        
        # Simple test commands
        commands = [
            "uci",
            "isready",
            "position startpos",
            "go depth 4",
            "quit"
        ]
        
        for cmd in commands:
            print(f">>> {cmd}")
            process.stdin.write(cmd + '\n')
            process.stdin.flush()
            time.sleep(0.2)
        
        # Get output with shorter timeout
        stdout, stderr = process.communicate(timeout=15)
        
        print("\nEngine output:")
        print(stdout)
        
        if stderr:
            print("Errors:")
            print(stderr)
            
        # Check if it completed successfully
        if "bestmove" in stdout:
            print("\n✅ SUCCESS: Engine responded with bestmove")
        else:
            print("\n❌ FAILED: No bestmove found")
            
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT: Engine took too long")
        process.kill()
    except Exception as e:
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    simple_uci_test()
