#!/usr/bin/env python3
"""Final Arena-style test of the clean exe"""
import subprocess
import sys
import time

print("=== FINAL ARENA-STYLE TEST ===")

exe_path = r"dist\V7P3R_v4.1.exe"

# Simulate Arena tournament interaction
print("\nSimulating Arena tournament commands...")
start_time = time.time()

proc = subprocess.Popen([exe_path], 
                       stdin=subprocess.PIPE, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE,
                       text=True)

# Standard Arena sequence
commands = """uci
setoption name Hash value 128
setoption name Threads value 1
isready
position startpos
go movetime 2000
position startpos moves e2e4
go movetime 2000
quit
"""

proc.stdin.write(commands)
proc.stdin.close()
out, err = proc.communicate(timeout=20)
elapsed = time.time() - start_time

print(f"Total execution time: {elapsed:.2f}s")
print(f"Return code: {proc.returncode}")

# Check output quality
lines = out.strip().split('\n')
uci_responses = []
moves = []

for line in lines:
    if line.strip():
        if any(keyword in line for keyword in ['uciok', 'readyok', 'bestmove']):
            uci_responses.append(line.strip())
        if line.startswith('bestmove'):
            moves.append(line.strip())

print(f"\n=== RESULTS ===")
print(f"UCI responses: {len(uci_responses)}")
print(f"Moves generated: {len(moves)}")
print(f"Has errors: {'Yes' if err.strip() else 'No'}")

print(f"\nUCI Responses:")
for resp in uci_responses:
    print(f"  {resp}")

print(f"\nMoves:")
for move in moves:
    print(f"  {move}")

# Arena compatibility check
has_uciok = any('uciok' in line for line in lines)
has_readyok = any('readyok' in line for line in lines)
has_moves = len(moves) >= 2
no_errors = not err.strip()
reasonable_time = elapsed < 15

print(f"\n=== ARENA COMPATIBILITY ===")
print(f"✓ UCI Protocol: {'PASS' if has_uciok else 'FAIL'}")
print(f"✓ Ready Status: {'PASS' if has_readyok else 'FAIL'}")
print(f"✓ Move Generation: {'PASS' if has_moves else 'FAIL'}")
print(f"✓ Clean Output: {'PASS' if no_errors else 'FAIL'}")
print(f"✓ Response Time: {'PASS' if reasonable_time else 'FAIL'}")

overall = has_uciok and has_readyok and has_moves and no_errors and reasonable_time
print(f"\n🏆 OVERALL: {'ARENA READY' if overall else 'NEEDS WORK'}")

if not overall:
    print(f"\nDEBUG OUTPUT:")
    print(f"STDOUT: {repr(out)}")
    print(f"STDERR: {repr(err)}")
