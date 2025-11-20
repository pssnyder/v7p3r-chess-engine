#!/usr/bin/env python3
"""
Quick test to verify V7P3R v16.3 PV display works correctly
"""

import sys
import subprocess
import time

def test_pv_display():
    """Test that PV display shows multiple moves, not just one"""
    
    print("=" * 60)
    print("V7P3R v16.3 PV Display Test")
    print("=" * 60)
    print()
    
    # Start UCI engine
    print("Starting v16.3 UCI interface...")
    engine = subprocess.Popen(
        ['python', 'v7p3r_uci_v163.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    def send_command(cmd):
        print(f">>> {cmd}")
        engine.stdin.write(cmd + '\n')
        engine.stdin.flush()
    
    def read_until(keyword, timeout=5):
        """Read output until keyword found or timeout"""
        start = time.time()
        lines = []
        while time.time() - start < timeout:
            line = engine.stdout.readline().strip()
            if line:
                print(f"<<< {line}")
                lines.append(line)
                if keyword in line:
                    return lines
        return lines
    
    try:
        # Initialize
        send_command('uci')
        read_until('uciok')
        
        send_command('isready')
        read_until('readyok')
        
        # Test 1: Starting position
        print("\n" + "=" * 60)
        print("TEST 1: Starting Position (should show multiple moves in PV)")
        print("=" * 60)
        send_command('position startpos')
        send_command('go depth 5')
        
        output = read_until('bestmove', timeout=30)
        
        # Check for PV lines
        pv_lines = [line for line in output if 'info depth' in line and 'pv' in line]
        
        if not pv_lines:
            print("\n❌ FAIL: No PV lines found in output!")
            return False
        
        print(f"\n✓ Found {len(pv_lines)} PV lines")
        
        # Check last PV line has multiple moves
        last_pv = pv_lines[-1]
        print(f"\nLast PV line: {last_pv}")
        
        if 'pv ' in last_pv:
            pv_part = last_pv.split('pv ')[1]
            pv_moves = pv_part.split()
            print(f"PV moves: {pv_moves}")
            print(f"PV length: {len(pv_moves)} moves")
            
            if len(pv_moves) >= 3:
                print(f"\n✅ SUCCESS: PV shows {len(pv_moves)} moves (expected 3+)")
                print(f"   First 3 moves: {' '.join(pv_moves[:3])}")
                return True
            else:
                print(f"\n❌ FAIL: PV only shows {len(pv_moves)} move(s) (expected 3+)")
                return False
        else:
            print("\n❌ FAIL: No 'pv' found in output line")
            return False
        
    finally:
        # Cleanup
        send_command('quit')
        engine.wait(timeout=2)
        engine.terminate()

if __name__ == '__main__':
    import os
    os.chdir('s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/src')
    
    success = test_pv_display()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ V16.3 PV DISPLAY TEST PASSED")
        print("PV display is working correctly!")
    else:
        print("❌ V16.3 PV DISPLAY TEST FAILED")
        print("Check output above for details")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
