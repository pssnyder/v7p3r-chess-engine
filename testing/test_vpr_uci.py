#!/usr/bin/env python3
"""
Quick VPR UCI Test - Verify UCI protocol works correctly
"""

import subprocess
import time
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_uci_protocol():
    """Test basic UCI commands with VPR engine"""
    print("Testing VPR UCI Protocol")
    print("=" * 40)
    
    # Start VPR engine
    engine_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'vpr_uci.py')
    process = subprocess.Popen(
        ['python', engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    def send_command(cmd):
        """Send command and get response"""
        print(f"→ {cmd}")
        process.stdin.write(cmd + '\n')
        process.stdin.flush()
        
        # Read response lines
        responses = []
        start_time = time.time()
        
        while time.time() - start_time < 2.0:  # 2 second timeout
            if process.poll() is not None:
                break
            
            try:
                process.stdout.settimeout(0.1)
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    print(f"← {line}")
                    responses.append(line)
                    
                    # Stop reading after certain responses
                    if line in ['uciok', 'readyok'] or line.startswith('bestmove'):
                        break
                else:
                    time.sleep(0.01)
            except:
                time.sleep(0.01)
                
        return responses
    
    try:
        # Test basic UCI commands
        print("\n1. Testing UCI identification...")
        send_command('uci')
        
        print("\n2. Testing ready status...")
        send_command('isready')
        
        print("\n3. Testing new game...")
        send_command('ucinewgame')
        
        print("\n4. Testing position setup...")
        send_command('position startpos')
        
        print("\n5. Testing quick search...")
        responses = send_command('go movetime 1000')
        
        # Look for bestmove in responses
        best_move_found = any(r.startswith('bestmove') for r in responses)
        if best_move_found:
            print("✓ VPR UCI interface working correctly!")
        else:
            print("✗ No bestmove response received")
            
    except Exception as e:
        print(f"Error during UCI test: {e}")
    
    finally:
        # Clean shutdown
        try:
            process.stdin.write('quit\n')
            process.stdin.flush()
            process.wait(timeout=2)
        except:
            process.terminate()
    
    print("\nUCI Test Complete")

if __name__ == "__main__":
    test_uci_protocol()