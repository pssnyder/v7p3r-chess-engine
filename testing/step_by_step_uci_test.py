#!/usr/bin/env python3
"""Test UCI communication step by step"""
import subprocess
import sys
import time

print("Testing UCI module step by step...")

# Test 1: Just uci command
print("\n=== Test 1: uci command ===")
proc = subprocess.Popen([sys.executable, "src/v7p3r_uci.py"], 
                       stdin=subprocess.PIPE, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE,
                       text=True)

proc.stdin.write("uci\n")
proc.stdin.close()
out, err = proc.communicate(timeout=3)
print("STDOUT:", repr(out))
print("STDERR:", repr(err))
print("Return code:", proc.returncode)

# Test 2: uci + isready
print("\n=== Test 2: uci + isready ===")
proc = subprocess.Popen([sys.executable, "src/v7p3r_uci.py"], 
                       stdin=subprocess.PIPE, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE,
                       text=True)

proc.stdin.write("uci\n")
proc.stdin.write("isready\n")
proc.stdin.write("quit\n")
proc.stdin.close()
out, err = proc.communicate(timeout=5)
print("STDOUT:", repr(out))
print("STDERR:", repr(err))
print("Return code:", proc.returncode)

# Test 3: Full sequence
print("\n=== Test 3: Full sequence ===")
proc = subprocess.Popen([sys.executable, "src/v7p3r_uci.py"], 
                       stdin=subprocess.PIPE, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE,
                       text=True)

commands = "uci\nisready\nposition startpos\ngo movetime 1000\nquit\n"
proc.stdin.write(commands)
proc.stdin.close()
out, err = proc.communicate(timeout=10)
print("STDOUT:", repr(out))
print("STDERR:", repr(err))
print("Return code:", proc.returncode)
