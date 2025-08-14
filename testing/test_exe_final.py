#!/usr/bin/env python3
"""Test the built exe file step by step"""
import subprocess
import sys
import time

print("Testing BUILT EXE step by step...")

exe_path = r"dist\V7P3R_v4.1.exe"

# Test 1: Just uci command
print("\n=== Test 1: uci command (EXE) ===")
proc = subprocess.Popen([exe_path], 
                       stdin=subprocess.PIPE, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE,
                       text=True)

proc.stdin.write("uci\n")
proc.stdin.close()
out, err = proc.communicate(timeout=5)
print("STDOUT:", repr(out))
print("STDERR:", repr(err))
print("Return code:", proc.returncode)

# Test 2: uci + isready
print("\n=== Test 2: uci + isready (EXE) ===")
proc = subprocess.Popen([exe_path], 
                       stdin=subprocess.PIPE, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE,
                       text=True)

proc.stdin.write("uci\n")
proc.stdin.write("isready\n")
proc.stdin.write("quit\n")
proc.stdin.close()
out, err = proc.communicate(timeout=7)
print("STDOUT:", repr(out))
print("STDERR:", repr(err))
print("Return code:", proc.returncode)

# Test 3: Full sequence with timing
print("\n=== Test 3: Full sequence with timing (EXE) ===")
start_time = time.time()
proc = subprocess.Popen([exe_path], 
                       stdin=subprocess.PIPE, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE,
                       text=True)

commands = "uci\nisready\nposition startpos\ngo movetime 1000\nquit\n"
proc.stdin.write(commands)
proc.stdin.close()
out, err = proc.communicate(timeout=15)
elapsed = time.time() - start_time

print("STDOUT:", repr(out))
print("STDERR:", repr(err))
print("Return code:", proc.returncode)
print(f"Total time: {elapsed:.2f}s")

# Check for key responses
has_uciok = "uciok" in out
has_readyok = "readyok" in out
has_bestmove = "bestmove" in out

print(f"\nValidation:")
print(f"- Has uciok: {has_uciok}")
print(f"- Has readyok: {has_readyok}")
print(f"- Has bestmove: {has_bestmove}")
print(f"- Arena compatible: {has_uciok and has_readyok and has_bestmove}")
