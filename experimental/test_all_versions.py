#!/usr/bin/env python3
"""
Test all VPR experimental versions for Arena deployment readiness
"""

import subprocess
import sys
import os
import time

def test_uci_engine(version_dir, version_name):
    """Test UCI communication with an engine"""
    print(f"\n=== Testing {version_name} ===")
    
    # Change to the engine directory
    engine_dir = f"s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/experimental/{version_dir}"
    python_exe = "C:/Users/patss/AppData/Local/Programs/Python/Python313/python.exe"
    uci_script = f"{engine_dir}/src/vpr_uci.py"
    
    try:
        # Test UCI command
        process = subprocess.Popen(
            [python_exe, uci_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            cwd=engine_dir
        )
        
        # Send UCI command
        stdout, stderr = process.communicate(input="uci\nquit\n", timeout=10)
        
        if "uciok" in stdout:
            print(f"✓ {version_name} UCI communication working")
            print(f"  Engine ID found in response")
            return True
        else:
            print(f"❌ {version_name} UCI failed")
            print(f"  stdout: {stdout[:200]}")
            print(f"  stderr: {stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {version_name} timed out")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ {version_name} error: {e}")
        return False

def main():
    print("=== VPR Arena Deployment Test Suite ===")
    print("Testing all experimental versions for Arena Chess GUI compatibility")
    
    # Test all versions
    versions = [
        ("VPR_v1.0", "VPR v1.0 (TAL-BOT Legacy)"),
        ("VPR_v2.0", "VPR v2.0 (TAL-BOT Enhanced)"), 
        ("VPR_v3.0", "VPR v3.0 (Pure Potential)"),
        ("VPR_v4.0", "VPR v4.0 (Bitboard Breakthrough)"),
        ("VPR_v5.0", "VPR v5.0 (Calculated Outcomes)"),
        ("VPR_v6.0", "VPR v6.0 (Chaos Capitalization)")
    ]
    
    results = []
    for version_dir, version_name in versions:
        success = test_uci_engine(version_dir, version_name)
        results.append((version_name, success))
    
    print("\n=== Summary ===")
    working_count = 0
    for name, success in results:
        status = "✓ Ready" if success else "❌ Failed"
        print(f"  {name}: {status}")
        if success:
            working_count += 1
    
    print(f"\n{working_count}/3 experimental versions ready for Arena deployment")
    
    if working_count > 0:
        print("\n🚀 Arena Integration Instructions:")
        print("1. Open Arena Chess GUI")
        print("2. Go to Engines > Install New Engine")
        print("3. Browse to experimental directory")
        print("4. Select the appropriate .bat file (VPR_v1.0.bat, VPR_v2.0.bat, or VPR_v3.0.bat)")
        print("5. Engine should appear in Arena engine list")
        print("6. Start a game and test!")

if __name__ == "__main__":
    main()