#!/usr/bin/env python3
"""
Simple UCI command sender - manually type UCI commands and see responses
This helps debug why the engine isn't responding properly
"""

import subprocess
import sys
import threading
import time

# Start the UCI engine from src
proc = subprocess.Popen([sys.executable, "src/v7p3r_uci.py"], 
                       stdin=subprocess.PIPE, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.STDOUT,
                       text=True,
                       bufsize=1)

def read_output():
    """Read and print engine output in a separate thread"""
    while True:
        try:
            line = proc.stdout.readline()
            if not line:
                break
            print(f"ENGINE: {line.strip()}")
        except:
            break

# Start output reader thread
output_thread = threading.Thread(target=read_output, daemon=True)
output_thread.start()

print("UCI Manual Test - Type commands and see responses")
print("Common commands: uci, isready, position startpos, go movetime 1000, quit")
print()

try:
    while True:
        cmd = input("UCI> ").strip()
        if cmd == "quit":
            break
        if cmd:
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()
            time.sleep(0.1)  # Give engine time to respond
finally:
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait(timeout=2)
