#!/usr/bin/env python3
"""
Simple test to see actual time being used
"""
import subprocess
import sys

engine = subprocess.Popen(
    [sys.executable, "src/v7p3r_uci.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Initialize
engine.stdin.write("uci\n")
engine.stdin.flush()
while True:
    line = engine.stdout.readline()
    print(line.strip())
    if "uciok" in line:
        break

engine.stdin.write("isready\n")
engine.stdin.flush()
while True:
    line = engine.stdout.readline()
    print(line.strip())
    if "readyok" in line:
        break

# Test move
print("\n=== TESTING OPENING MOVE ===")
engine.stdin.write("ucinewgame\n")
engine.stdin.write("position startpos\n")
engine.stdin.write("go wtime 300000 btime 300000 winc 5000 binc 5000\n")
engine.stdin.flush()

while True:
    line = engine.stdout.readline()
    print(line.strip())
    if "bestmove" in line:
        break

engine.stdin.write("quit\n")
engine.stdin.flush()
engine.wait()
