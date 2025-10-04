#!/usr/bin/env python3
"""
Quick UCI Perft Test for V7P3R Executable
Tests the actual compiled engine executable performance
"""

import subprocess
import time
import sys

def test_uci_perft(engine_path, depth=4):
    """Test perft via UCI interface"""
    print(f"Testing UCI Perft on {engine_path}")
    print(f"Depth: {depth}")
    print("=" * 50)
    
    # Standard perft positions
    positions = [
        ("startpos", "Starting Position"),
        ("fen r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", "Kiwipete"),
    ]
    
    for pos_cmd, name in positions:
        print(f"\n🔍 Testing: {name}")
        
        # Prepare UCI commands
        commands = [
            "uci",
            f"position {pos_cmd}",
            f"go perft {depth}",
            "quit"
        ]
        
        # Run the engine
        start_time = time.time()
        try:
            result = subprocess.run(
                [engine_path],
                input="\n".join(commands),
                text=True,
                capture_output=True,
                timeout=30  # 30 second timeout
            )
            end_time = time.time()
            
            if result.returncode == 0:
                # Parse perft output
                lines = result.stdout.split('\n')
                nodes = None
                for line in lines:
                    if line.startswith('Nodes searched:'):
                        nodes = int(line.split(':')[1].strip())
                    elif 'bestmove' in line:
                        break
                
                time_taken = end_time - start_time
                if nodes:
                    nps = nodes / time_taken if time_taken > 0 else 0
                    print(f"  ✅ Nodes: {nodes:,}")
                    print(f"  ⏱️  Time: {time_taken:.3f}s")
                    print(f"  🚀 NPS: {nps:,.0f}")
                else:
                    print(f"  ⚠️  Could not parse nodes from output")
                    print(f"  📄 Output: {result.stdout}")
            else:
                print(f"  ❌ Engine failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout after 30 seconds")
        except Exception as e:
            print(f"  💥 Error: {e}")

if __name__ == "__main__":
    engine_path = sys.argv[1] if len(sys.argv) > 1 else "./dist/V7P3R_v12.5_FIXED.exe"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    
    test_uci_perft(engine_path, depth)