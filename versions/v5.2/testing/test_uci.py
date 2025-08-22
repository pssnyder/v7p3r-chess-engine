#!/usr/bin/env python3
"""
Quick UCI test script
"""

import subprocess
import time

def test_uci():
    """Test basic UCI communication"""
    print("Testing UCI interface...")
    
    # Start the engine
    engine = subprocess.Popen(
        ["python", "v7p3r_uci.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="."
    )
    
    def send_command(cmd):
        print(f">>> {cmd}")
        engine.stdin.write(cmd + "\n")
        engine.stdin.flush()
        time.sleep(0.1)
    
    def read_output(timeout=1.0):
        start_time = time.time()
        output = []
        while time.time() - start_time < timeout:
            if engine.poll() is not None:
                break
            try:
                line = engine.stdout.readline()
                if line:
                    output.append(line.strip())
                    print(f"<<< {line.strip()}")
                else:
                    time.sleep(0.01)
            except:
                break
        return output
    
    try:
        # Test basic UCI commands
        send_command("uci")
        output = read_output()
        
        send_command("isready")
        output = read_output()
        
        send_command("ucinewgame")
        
        send_command("position startpos")
        
        send_command("go movetime 1000")
        output = read_output(3.0)  # Give more time for move calculation
        
        # Quit
        send_command("quit")
        
        print("UCI test completed successfully!")
        
    except Exception as e:
        print(f"UCI test failed: {e}")
    finally:
        engine.terminate()
        engine.wait()

if __name__ == "__main__":
    test_uci()
