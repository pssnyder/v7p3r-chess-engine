#!/usr/bin/env python3
"""
Quick test to verify V7P3R v13.0 is Arena-ready
"""

import subprocess
import sys
import os

def test_uci_interface():
    """Test that UCI interface works cleanly"""
    print("🧪 Testing V7P3R v13.0 UCI Interface...")
    
    # Path to our engine
    engine_path = os.path.join("src", "v7p3r_uci.py")
    python_path = r"C:\Users\patss\AppData\Local\Programs\Python\Python313\python.exe"
    
    # UCI commands to test
    uci_commands = [
        "uci",
        "isready", 
        "position startpos",
        "go depth 2",
        "quit"
    ]
    
    try:
        # Run the engine with our test commands
        process = subprocess.Popen(
            [python_path, engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send commands and get output
        input_text = "\n".join(uci_commands)
        stdout, stderr = process.communicate(input=input_text, timeout=30)
        
        print("✅ Engine Output:")
        print(stdout)
        
        if stderr.strip():
            print("⚠️ Stderr Output (should be empty for Arena):")
            print(stderr)
        else:
            print("✅ No stderr output (good for Arena)")
            
        # Check for required UCI responses
        lines = stdout.split('\n')
        has_uciok = any('uciok' in line for line in lines)
        has_readyok = any('readyok' in line for line in lines)
        has_bestmove = any('bestmove' in line for line in lines)
        has_warnings = any('⚠️' in line for line in lines)
        
        print(f"\n📊 UCI Compliance Check:")
        print(f"✅ Has 'uciok': {has_uciok}")
        print(f"✅ Has 'readyok': {has_readyok}")
        print(f"✅ Has 'bestmove': {has_bestmove}")
        print(f"❌ Has warnings: {has_warnings}")
        
        if has_uciok and has_readyok and has_bestmove and not has_warnings:
            print("\n🎉 V7P3R v13.0 is Arena-ready!")
            print("You can now use V7P3R_Test_Engine.bat in Arena Chess GUI")
            return True
        else:
            print("\n🔧 Still needs work for Arena compatibility")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Engine timed out")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ Error testing engine: {e}")
        return False

if __name__ == "__main__":
    success = test_uci_interface()
    sys.exit(0 if success else 1)