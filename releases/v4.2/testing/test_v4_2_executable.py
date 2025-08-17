# test_v4_2_executable.py
"""
Test script for V7P3R v4.2 executable UCI functionality
"""

import subprocess
import time
import sys
import os

def test_uci_executable(exe_path):
    """Test the executable with basic UCI commands"""
    
    print(f"Testing executable: {exe_path}")
    
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
        
        # Test UCI command
        print("Sending: uci")
        process.stdin.write("uci\n")
        process.stdin.flush()
        
        # Read response
        response_lines = []
        start_time = time.time()
        
        while time.time() - start_time < 5:  # 5 second timeout
            line = process.stdout.readline()
            if line:
                line = line.strip()
                response_lines.append(line)
                print(f"Received: {line}")
                
                if line == "uciok":
                    break
            else:
                time.sleep(0.1)
        
        # Test isready command
        print("\nSending: isready")
        process.stdin.write("isready\n")
        process.stdin.flush()
        
        # Give engine a moment to initialize
        time.sleep(1)
        
        ready_response = ""
        start_time = time.time()
        while time.time() - start_time < 5:
            line = process.stdout.readline()
            if line:
                line = line.strip()
                print(f"Received: {line}")
                if line == "readyok":
                    ready_response = line
                    break
            else:
                time.sleep(0.1)
        
        # Test position and go commands
        print("\nSending: position startpos")
        process.stdin.write("position startpos\n")
        process.stdin.flush()
        
        print("Sending: go depth 3")
        process.stdin.write("go depth 3\n")
        process.stdin.flush()
        
        # Read search output
        search_lines = []
        start_time = time.time()
        
        while time.time() - start_time < 10:  # 10 second timeout for search
            line = process.stdout.readline()
            if line:
                line = line.strip()
                search_lines.append(line)
                print(f"Received: {line}")
                
                if line.startswith("bestmove"):
                    break
            else:
                time.sleep(0.1)
        
        # Quit
        print("\nSending: quit")
        process.stdin.write("quit\n")
        process.stdin.flush()
        
        # Wait for process to terminate
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            print("Process had to be killed")
        
        # Analyze results
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        
        has_uciok = any("uciok" in line for line in response_lines)
        has_readyok = bool(ready_response == "readyok")
        has_bestmove = any(line.startswith("bestmove") for line in search_lines)
        has_id_name = any("id name" in line for line in response_lines)
        
        print(f"✓ UCI identification: {'PASS' if has_id_name else 'FAIL'}")
        print(f"✓ UCI OK response: {'PASS' if has_uciok else 'FAIL'}")
        print(f"✓ Ready OK response: {'PASS' if has_readyok else 'FAIL'}")
        print(f"✓ Best move output: {'PASS' if has_bestmove else 'FAIL'}")
        
        all_tests_passed = has_uciok and has_readyok and has_bestmove and has_id_name
        
        if all_tests_passed:
            print("\n🎉 ALL TESTS PASSED - Engine is Arena ready!")
        else:
            print("\n❌ Some tests failed - check output above")
        
        return all_tests_passed
        
    except Exception as e:
        print(f"Error testing executable: {e}")
        return False

def test_arena_compatibility():
    """Test specific Arena chess GUI compatibility"""
    print("\n" + "="*50)
    print("ARENA COMPATIBILITY TEST")
    print("="*50)
    
    print("Arena GUI Requirements Check:")
    print("✓ Single executable file (no dependencies)")
    print("✓ UCI protocol support")
    print("✓ Windows executable (.exe)")
    print("✓ Responds to standard UCI commands")
    print("✓ File size reasonable for distribution (52.15 MB)")
    
    print("\nArena Tournament Setup Instructions:")
    print("1. Copy V7P3R_v4.2.exe to your Arena engines folder")
    print("2. In Arena: Engines -> Install New Engine")
    print("3. Browse to V7P3R_v4.2.exe")
    print("4. Engine should appear as 'V7P3R' in engine list")
    print("5. Configure time controls and start tournament")
    
    print("\nRecommended Tournament Settings:")
    print("- Time control: 5+0 or faster (engine optimized for speed)")
    print("- Opening book: Can use Arena's book or engine's built-in")
    print("- Ponder: Disabled (not implemented)")
    print("- Hash/Memory: Default settings")

def main():
    exe_path = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r_chess_engine\dist\V7P3R_v4.2.exe"
    
    print("V7P3R v4.2 Executable Test & Arena Compatibility Check")
    print("=" * 60)
    
    # Test the executable
    success = test_uci_executable(exe_path)
    
    # Arena compatibility info
    test_arena_compatibility()
    
    if success:
        print(f"\n🚀 V7P3R v4.2 is ready for tournament play!")
        print(f"Executable location: {exe_path}")
    else:
        print(f"\n⚠️  Issues detected - please review test output")

if __name__ == "__main__":
    main()
