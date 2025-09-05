#!/usr/bin/env python3
"""Quick UCI test for V7P3R engine."""

import subprocess
import sys
import time

def test_v7p3r_uci():
    """Test V7P3R UCI communication."""
    print("Testing V7P3R UCI Communication...")
    print("=" * 50)
    
    try:
        # Start the engine using the UCI interface
        proc = subprocess.Popen(
            [sys.executable, 'src/v7p3r_uci.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        def send_command(cmd):
            print(f"→ {cmd}")
            proc.stdin.write(f"{cmd}\n")
            proc.stdin.flush()
            time.sleep(0.5)
            
        def read_output():
            output_lines = []
            while True:
                try:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        print(f"← {line}")
                        output_lines.append(line)
                        if line == "uciok" or line == "readyok":
                            break
                except:
                    break
            return output_lines
        
        # Test UCI initialization
        print("\n1. Testing UCI initialization:")
        send_command("uci")
        uci_output = read_output()
        
        print("\n2. Testing isready:")
        send_command("isready")
        ready_output = read_output()
        
        print("\n3. Testing a simple position and search:")
        send_command("ucinewgame")
        time.sleep(0.2)
        send_command("position startpos")
        time.sleep(0.2)
        send_command("go depth 3")
        
        # Read search output
        search_output = []
        start_time = time.time()
        while time.time() - start_time < 10:  # 10 second timeout
            try:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    print(f"← {line}")
                    search_output.append(line)
                    if line.startswith("bestmove"):
                        break
            except:
                break
        
        print("\n4. Quitting engine:")
        send_command("quit")
        
        proc.wait(timeout=2)
        
        print("\n" + "=" * 50)
        print("UCI Test Complete!")
        
        # Summary
        print(f"\nSummary:")
        print(f"- UCI initialization: {'✓' if any('uciok' in line for line in uci_output) else '✗'}")
        print(f"- Ready response: {'✓' if any('readyok' in line for line in ready_output) else '✗'}")
        print(f"- Search output: {'✓' if any('bestmove' in line for line in search_output) else '✗'}")
        
    except Exception as e:
        print(f"Error during UCI test: {e}")
        if 'proc' in locals():
            proc.terminate()

if __name__ == "__main__":
    test_v7p3r_uci()
