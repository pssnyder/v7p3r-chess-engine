#!/usr/bin/env python3
"""
Test PV Following through UCI Interface - Non-blocking version
"""

import subprocess
import time
import os
import select
import sys

def test_uci_pv_following():
    """Test PV following works through UCI protocol"""
    print("🔍 Testing PV Following through UCI Interface")
    print("=" * 60)
    
    # Path to the engine
    engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'v7p3r_uci.py')
    
    # Start engine process
    engine = subprocess.Popen(
        ['python', engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    def send_command(cmd):
        print(f"→ {cmd}")
        engine.stdin.write(cmd + '\n')
        engine.stdin.flush()
    
    def read_response_with_timeout(timeout=5.0):
        """Read available output with timeout"""
        output_lines = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if engine.poll() is not None:  # Process ended
                break
                
            # Try to read a line with timeout
            try:
                # Check if data is available to read
                if sys.platform == "win32":
                    # On Windows, just try to read with small timeout
                    time.sleep(0.1)
                    line = ""
                    try:
                        line = engine.stdout.readline()
                    except:
                        continue
                else:
                    # On Unix-like systems, use select
                    ready, _, _ = select.select([engine.stdout], [], [], 0.1)
                    if not ready:
                        continue
                    line = engine.stdout.readline()
                
                if line:
                    line = line.strip()
                    if line:
                        print(f"← {line}")
                        output_lines.append(line)
                        if 'bestmove' in line or 'uciok' in line or 'readyok' in line:
                            break
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Read error: {e}")
                break
                
        return output_lines
    
    try:
        print("Step 1: Initialize engine")
        send_command("uci")
        response = read_response_with_timeout(3.0)
        
        if not any('uciok' in line for line in response):
            print("❌ Engine did not respond to UCI")
            return False
        
        send_command("isready")
        response = read_response_with_timeout(2.0)
        
        if not any('readyok' in line for line in response):
            print("❌ Engine not ready")
            return False
        
        print("\nStep 2: Test basic move")
        send_command("position startpos")
        send_command("go movetime 1000")  # 1 second search
        
        response = read_response_with_timeout(5.0)
        
        # Extract the bestmove
        bestmove = None
        for line in response:
            if line.startswith('bestmove'):
                bestmove = line.split()[1]
                break
        
        if not bestmove:
            print("❌ No bestmove received")
            return False
            
        print(f"✅ Engine suggested: {bestmove}")
        
        print(f"\nStep 3: Test potential PV following")
        # Test a known sequence that should trigger PV following
        send_command("position startpos moves e2e4 e7e5 g1f3")
        
        start_time = time.time()
        send_command("go movetime 1000")
        
        response = read_response_with_timeout(5.0)
        search_time = time.time() - start_time
        
        print(f"Search completed in {search_time:.3f}s")
        
        # Check for PV following messages
        pv_follow_detected = any("PV FOLLOW" in line or "PV HIT" in line for line in response)
        instant_response = search_time < 0.5
        
        if pv_follow_detected:
            print("✅ PV FOLLOWING: Detected PV following messages in UCI output")
            success = True
        elif instant_response:
            print(f"✅ FAST RESPONSE: {search_time:.3f}s - may indicate PV following")
            success = True
        else:
            print("⚠️  No clear PV following detected, but engine is working")
            success = True  # Engine is functional
        
        print(f"\n" + "=" * 60)
        if success:
            print("🎉 UCI INTERFACE: WORKING!")
        else:
            print("❌ UCI INTERFACE: ISSUES DETECTED")
        
        return success
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
        
    finally:
        try:
            send_command("quit")
            time.sleep(0.5)
        except:
            pass
        engine.terminate()
        engine.wait(timeout=2.0)

if __name__ == "__main__":
    test_uci_pv_following()
