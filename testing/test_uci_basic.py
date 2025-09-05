#!/usr/bin/env python3
"""
Simple UCI Test - Just verify engine responds correctly
"""

import os
import sys
import subprocess
import time

def test_uci_basic():
    """Basic UCI test without hanging"""
    print("🔍 Testing Basic UCI Functionality")
    print("=" * 50)
    
    # Path to the engine
    engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'v7p3r_uci.py')
    
    # Simple test using echo to pipe commands
    print("Testing UCI protocol...")
    
    try:
        # Test 1: Basic UCI initialization
        result = subprocess.run(
            ['python', engine_path],
            input="uci\nisready\nquit\n",
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if "uciok" in result.stdout and "readyok" in result.stdout:
            print("✅ UCI initialization: SUCCESS")
        else:
            print("❌ UCI initialization: FAILED")
            print(f"Output: {result.stdout}")
            return False
        
        # Test 2: Simple position and search
        print("Testing basic search...")
        result = subprocess.run(
            ['python', engine_path],
            input="uci\nisready\nposition startpos\ngo movetime 1000\nquit\n",
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if "bestmove" in result.stdout:
            print("✅ Basic search: SUCCESS")
            
            # Check for PV following setup messages
            if "PV FOLLOW setup" in result.stdout:
                print("✅ PV following: INITIALIZED")
            else:
                print("⚠️  PV following: Not seen in basic test")
        else:
            print("❌ Basic search: FAILED")
            print(f"Output: {result.stdout}")
            return False
        
        print(f"\n" + "=" * 50)
        print("🎉 UCI BASIC TEST: SUCCESS!")
        print("Engine is responding correctly to UCI commands")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Test timeout - engine may be hanging")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    test_uci_basic()
