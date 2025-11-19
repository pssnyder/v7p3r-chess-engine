#!/usr/bin/env python3
"""
Quick UCI test for V15.3 to ensure book integration works
"""

import subprocess
import sys
import os

def test_uci():
    """Test UCI interface with opening book"""
    
    print("=" * 60)
    print("V15.3 UCI Integration Test")
    print("=" * 60)
    
    uci_script = os.path.join("src", "v7p3r_uci.py")
    
    # Commands to send
    commands = [
        "uci",
        "isready",
        "position startpos",
        "go movetime 100",
        "position startpos moves e2e4 c7c6",
        "go movetime 100",
        "quit"
    ]
    
    print("\nSending UCI commands:")
    for cmd in commands:
        print(f"  > {cmd}")
    
    print("\nEngine output:")
    print("-" * 60)
    
    # Run the engine
    process = subprocess.Popen(
        [sys.executable, uci_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send commands
    input_text = "\n".join(commands)
    stdout, stderr = process.communicate(input=input_text, timeout=10)
    
    print(stdout)
    if stderr:
        print("Errors:")
        print(stderr)
    
    print("-" * 60)
    
    # Check output
    checks = {
        "id name V7P3R v15.3": False,
        "option name OwnBook": False,
        "option name BookFile": False,
        "option name BookDepth": False,
        "option name BookVariety": False,
        "info string Book move": False,
    }
    
    for line in stdout.split("\n"):
        for check in checks:
            if check in line:
                checks[check] = True
    
    print("\nValidation:")
    all_passed = True
    for check, passed in checks.items():
        status = "✓" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All UCI checks passed!")
        return True
    else:
        print("\n❌ Some UCI checks failed")
        return False


if __name__ == "__main__":
    success = test_uci()
    sys.exit(0 if success else 1)
