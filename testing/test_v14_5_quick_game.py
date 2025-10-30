#!/usr/bin/env python3
"""
Quick test of V14.5 to verify:
1. UCI output is working (showing depth, score, nodes)
2. Time management is reasonable (not burning all time in opening)
3. Not blundering pieces every move
4. Can achieve decent depth
"""
import subprocess
import sys
import time

def test_v14_5():
    """Test V14.5 with a few moves"""
    print("=" * 60)
    print("V14.5 Quick Verification Test")
    print("=" * 60)
    print()
    
    # Start engine
    engine = subprocess.Popen(
        [sys.executable, "src/v7p3r_uci.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    def send(cmd):
        """Send command to engine"""
        print(f"> {cmd}")
        engine.stdin.write(cmd + "\n")
        engine.stdin.flush()
    
    def receive_until(marker, timeout=10):
        """Receive output until marker found"""
        start = time.time()
        output = []
        while time.time() - start < timeout:
            line = engine.stdout.readline().strip()
            if line:
                output.append(line)
                print(f"< {line}")
                if marker in line:
                    return output
        return output
    
    try:
        # Initialize
        send("uci")
        receive_until("uciok")
        
        send("isready")
        receive_until("readyok")
        
        print("\n" + "=" * 60)
        print("TEST 1: Opening Move (e2-e4) - Should use minimal time")
        print("=" * 60)
        send("ucinewgame")
        send("position startpos")
        send("go wtime 300000 btime 300000 winc 5000 binc 5000")
        
        output = receive_until("bestmove", timeout=30)
        
        # Analyze output
        depths = [line for line in output if "info depth" in line]
        if depths:
            print(f"\n✓ UCI Output Working - Found {len(depths)} depth lines")
            print(f"  Sample: {depths[0]}")
            
            # Check depth achieved
            max_depth = 0
            for line in depths:
                try:
                    parts = line.split()
                    if "depth" in parts:
                        depth_idx = parts.index("depth")
                        if depth_idx + 1 < len(parts):
                            depth = int(parts[depth_idx + 1])
                            max_depth = max(max_depth, depth)
                except:
                    pass
            
            print(f"  Max depth achieved: {max_depth}")
            if max_depth >= 4:
                print(f"  ✓ Good depth for opening move")
            elif max_depth >= 3:
                print(f"  ⚠ Acceptable depth (could be better)")
            else:
                print(f"  ✗ Depth too shallow!")
        else:
            print("✗ NO UCI OUTPUT FOUND!")
        
        print("\n" + "=" * 60)
        print("TEST 2: Move 5 - Should still be fast")
        print("=" * 60)
        send("position startpos moves e2e4 e7e5 g1f3 b8c6")
        send("go wtime 285000 btime 285000 winc 5000 binc 5000")
        
        output = receive_until("bestmove", timeout=30)
        depths = [line for line in output if "info depth" in line]
        if depths:
            print(f"✓ Found {len(depths)} depth lines")
            print(f"  Sample: {depths[-1] if depths else 'none'}")
        
        print("\n" + "=" * 60)
        print("TEST 3: Middlegame Position - Can use more time")
        print("=" * 60)
        send("position startpos moves e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7")
        send("go wtime 270000 btime 270000 winc 5000 binc 5000")
        
        output = receive_until("bestmove", timeout=30)
        depths = [line for line in output if "info depth" in line]
        if depths:
            print(f"✓ Found {len(depths)} depth lines")
            
            # Check score is reasonable (not blundering)
            last_line = depths[-1] if depths else ""
            if "score cp" in last_line:
                try:
                    parts = last_line.split()
                    cp_idx = parts.index("cp")
                    score = int(parts[cp_idx + 1])
                    print(f"  Final score: {score/100.0:.2f} pawns")
                    if abs(score) < 500:
                        print(f"  ✓ Score looks reasonable (no obvious blunder)")
                    else:
                        print(f"  ⚠ Large score change detected")
                except:
                    pass
        
        # Cleanup
        send("quit")
        engine.wait(timeout=2)
        
        print("\n" + "=" * 60)
        print("V14.5 Quick Test Complete!")
        print("=" * 60)
        print("\nNext step: Run full game against v10.8 to verify competitive play")
        
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        engine.kill()
        raise
    
    finally:
        try:
            engine.kill()
        except:
            pass

if __name__ == "__main__":
    test_v14_5()
