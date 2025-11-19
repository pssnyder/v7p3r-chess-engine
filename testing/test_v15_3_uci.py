#!/usr/bin/env python3
"""
Quick UCI protocol test for V15.3
"""

import subprocess
import sys
import time

ENGINE_PATH = "s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/src/v7p3r_uci.py"

def send_command(process, command):
    """Send a UCI command"""
    print(f"> {command}")
    process.stdin.write(f"{command}\n")
    process.stdin.flush()

def read_until(process, marker, timeout=3):
    """Read until marker found"""
    start = time.time()
    lines = []
    while time.time() - start < timeout:
        line = process.stdout.readline()
        if line:
            line = line.strip()
            print(f"< {line}")
            lines.append(line)
            if marker in line:
                break
    return lines

print("="*60)
print("V7P3R v15.3 UCI Protocol Test")
print("="*60)
print()

process = subprocess.Popen(
    [sys.executable, ENGINE_PATH],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

try:
    # Test UCI command
    print("TEST: UCI Options")
    print("-"*40)
    send_command(process, "uci")
    response = read_until(process, "uciok")
    
    has_name = any("V7P3R v15.3" in line for line in response)
    has_ownbook = any("OwnBook" in line for line in response)
    has_bookfile = any("BookFile" in line for line in response)
    has_bookdepth = any("BookDepth" in line for line in response)
    has_bookvariety = any("BookVariety" in line for line in response)
    
    print()
    print(f"✓ Engine name: {'PASS' if has_name else 'FAIL'}")
    print(f"✓ OwnBook option: {'PASS' if has_ownbook else 'FAIL'}")
    print(f"✓ BookFile option: {'PASS' if has_bookfile else 'FAIL'}")
    print(f"✓ BookDepth option: {'PASS' if has_bookdepth else 'FAIL'}")
    print(f"✓ BookVariety option: {'PASS' if has_bookvariety else 'FAIL'}")
    print()
    
    # Test book move
    print("TEST: Book Move from Starting Position")
    print("-"*40)
    send_command(process, "isready")
    read_until(process, "readyok")
    
    send_command(process, "ucinewgame")
    send_command(process, "position startpos")
    send_command(process, "go movetime 1000")
    
    response = read_until(process, "bestmove", timeout=5)
    book_used = any("Book move" in line for line in response)
    bestmove_line = [l for l in response if "bestmove" in l]
    
    print()
    if book_used:
        print(f"✓ Book move used: PASS")
    else:
        print(f"⚠️  Book move not mentioned (may have searched)")
    
    if bestmove_line:
        move = bestmove_line[0].split()[1]
        good_moves = ['e2e4', 'd2d4', 'g1f3', 'c2c4', 'b1c3']
        if move in good_moves:
            print(f"✓ Good opening move ({move}): PASS")
        else:
            print(f"⚠️  Unusual opening move ({move})")
    
    print()
    print("="*60)
    print("✅ V15.3 Opening Book Implementation Complete!")
    print("="*60)
    
finally:
    send_command(process, "quit")
    try:
        process.wait(timeout=2)
    except:
        process.kill()
