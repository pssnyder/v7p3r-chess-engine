#!/usr/bin/env python3
"""
Quick test to verify V15.2 in engine-tester responds correctly to UCI commands
and doesn't show perspective issues in move selection.
"""

import subprocess
import sys
import time

ENGINE_PATH = "s:/Maker Stuff/Programming/Chess Engines/Chess Engine Playground/engine-tester/engines/V7P3R/V7P3R_v15.2/V7P3R_v15.2.bat"

def send_command(process, command):
    """Send a UCI command to the engine"""
    print(f"> {command}")
    process.stdin.write(f"{command}\n")
    process.stdin.flush()
    time.sleep(0.1)

def read_until(process, marker, timeout=5):
    """Read engine output until a marker is found"""
    start_time = time.time()
    lines = []
    while time.time() - start_time < timeout:
        line = process.stdout.readline()
        if line:
            line = line.strip()
            print(f"< {line}")
            lines.append(line)
            if marker in line:
                break
    return lines

def cleanup_process(process):
    """Safely cleanup engine process"""
    try:
        send_command(process, "quit")
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
    except:
        pass

def test_engine_basic():
    """Test basic UCI communication"""
    print("="*60)
    print("TEST 1: Basic UCI Communication")
    print("="*60)
    
    process = subprocess.Popen(
        ENGINE_PATH,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    try:
        # UCI handshake
        send_command(process, "uci")
        response = read_until(process, "uciok", timeout=3)
        
        if any("uciok" in line for line in response):
            print("✓ UCI handshake successful")
        else:
            print("✗ UCI handshake failed")
            return False
        
        # Check engine identity
        name_line = [line for line in response if "id name" in line]
        if name_line:
            print(f"✓ Engine identified: {name_line[0]}")
        
        return True
        
    finally:
        cleanup_process(process)

def test_opening_move_white():
    """Test V15.2 as White - should make sensible opening move"""
    print("\n" + "="*60)
    print("TEST 2: Opening Move as White")
    print("="*60)
    
    process = subprocess.Popen(
        ENGINE_PATH,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    try:
        send_command(process, "uci")
        read_until(process, "uciok")
        
        send_command(process, "isready")
        read_until(process, "readyok")
        
        send_command(process, "ucinewgame")
        send_command(process, "position startpos")
        send_command(process, "go movetime 6000")
        
        response = read_until(process, "bestmove", timeout=10)
        
        bestmove_line = [line for line in response if "bestmove" in line]
        if bestmove_line:
            move = bestmove_line[0].split()[1]
            print(f"\n✓ White opening move: {move}")
            
            # Check if it's a reasonable move
            good_moves = ['e2e4', 'd2d4', 'g1f3', 'b1c3', 'e2e3', 'd2d3', 'c2c4']
            bad_moves = ['h2h3', 'h2h4', 'a2a3', 'a2a4', 'g2g3', 'g2g4']
            
            if move in good_moves:
                print(f"✓ GOOD: Central/developing move")
                return True, move
            elif move in bad_moves:
                print(f"⚠️  WARNING: Edge/passive move (possible issue)")
                return False, move
            else:
                print(f"⚠️  Unusual but possibly OK")
                return True, move
        else:
            print("✗ No bestmove received")
            return False, None
            
    finally:
        cleanup_process(process)

def test_opening_response_black():
    """Test V15.2 as Black responding to 1.e4"""
    print("\n" + "="*60)
    print("TEST 3: Opening Response as Black (after 1.e4)")
    print("="*60)
    
    process = subprocess.Popen(
        ENGINE_PATH,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    try:
        send_command(process, "uci")
        read_until(process, "uciok")
        
        send_command(process, "isready")
        read_until(process, "readyok")
        
        send_command(process, "ucinewgame")
        send_command(process, "position startpos moves e2e4")
        send_command(process, "go movetime 6000")
        
        response = read_until(process, "bestmove", timeout=10)
        
        bestmove_line = [line for line in response if "bestmove" in line]
        if bestmove_line:
            move = bestmove_line[0].split()[1]
            print(f"\n✓ Black response to 1.e4: {move}")
            
            # Check if it's a reasonable response
            good_moves = ['e7e5', 'c7c5', 'e7e6', 'c7c6', 'g8f6', 'b8c6', 'd7d5', 'd7d6']
            bad_moves = ['h7h6', 'h7h5', 'a7a6', 'a7a5', 'g7g6', 'g7g5']
            
            if move in good_moves:
                print(f"✓ GOOD: Principled response")
                return True, move
            elif move in bad_moves:
                print(f"⚠️  WARNING: Passive/anti-positional (possible issue)")
                return False, move
            else:
                print(f"⚠️  Unusual but possibly OK")
                return True, move
        else:
            print("✗ No bestmove received")
            return False, None
            
    finally:
        cleanup_process(process)

def main():
    print("V7P3R v15.2 - Perspective Bug Validation (Engine-Tester)")
    print("Testing deployed V15.2 from engine-tester directory")
    print()
    
    # Test 1: Basic communication
    if not test_engine_basic():
        print("\n❌ FAILED: Basic UCI communication failed")
        return 1
    
    # Test 2: White opening
    white_ok, white_move = test_opening_move_white()
    
    # Test 3: Black opening
    black_ok, black_move = test_opening_response_black()
    
    # Analysis
    print("\n" + "="*60)
    print("PERSPECTIVE BUG ANALYSIS")
    print("="*60)
    
    print(f"\nWhite opening: {white_move} - {'✓ Good' if white_ok else '⚠️  Questionable'}")
    print(f"Black opening: {black_move} - {'✓ Good' if black_ok else '⚠️  Questionable'}")
    
    if white_ok and black_ok:
        print("\n✅ PERSPECTIVE BUG APPEARS FIXED")
        print("   V15.2 makes sensible moves as both White and Black")
        print("   Ready for Arena tournament testing")
        return 0
    elif not white_ok and not black_ok:
        print("\n⚠️  BOTH COLORS SHOW ISSUES")
        print("   This is consistent behavior (not perspective-dependent)")
        print("   May indicate PST or move ordering issue, but NOT perspective bug")
        return 0
    else:
        print("\n❌ POSSIBLE PERSPECTIVE ISSUE")
        print("   One color plays well, the other doesn't")
        print("   This mirrors the V15.1 alternating pattern")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
