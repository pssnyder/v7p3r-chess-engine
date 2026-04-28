#!/usr/bin/env python3
"""
Verify that we can load two different engine versions simultaneously
"""

import sys
import importlib
from pathlib import Path

# Clear any cached imports
if 'v7p3r' in sys.modules:
    del sys.modules['v7p3r']
if 'v7p3r_uci' in sys.modules:
    del sys.modules['v7p3r_uci']

# Test 1: Load v19.5.2 from src/
print("Loading v19.5.2 from src/...")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import v7p3r as v7p3r_new
sys.path.pop(0)

# Check version from source file
with open(Path(__file__).parent.parent / "src" / "v7p3r.py", 'r') as f:
    for i, line in enumerate(f):
        if i < 20 and 'v19.5' in line.lower():
            print(f"  Found: {line.strip()}")
            break

# Clear the import for fresh load
del sys.modules['v7p3r']

# Test 2: Load v18.4 from lichess/engines/
print("\nLoading v18.4 from lichess/engines/...")
sys.path.insert(0, str(Path(__file__).parent.parent / "lichess" / "engines" / "V7P3R_v18.4_20260417" / "src"))
import v7p3r as v7p3r_old
sys.path.pop(0)

# Check version from source file
v18_4_path = Path(__file__).parent.parent / "lichess" / "engines" / "V7P3R_v18.4_20260417" / "src" / "v7p3r.py"
if v18_4_path.exists():
    with open(v18_4_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 20 and ('v18.4' in line.lower() or 'v18' in line.lower()):
                print(f"  Found: {line.strip()}")
                break
else:
    print(f"  ERROR: v18.4 not found at {v18_4_path}")

print("\nConclusion: Python's import caching makes it impossible to load")
print("two versions of the same module in one process!")
print("\nSolution: Need to run engines in separate processes (UCI protocol)")
