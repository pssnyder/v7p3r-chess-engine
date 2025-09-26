#!/usr/bin/env python3
"""
V7P3R v12.2 UCI Debug Test
==========================
Diagnose why v12.2 is timing out in tournament validation
"""

import subprocess
import time
import os

def test_engine_uci_basic(engine_path):
    """Test basic UCI communication"""
    print(f"🔍 Testing UCI communication: {os.path.basename(engine_path)}")
    
    try:
        process = subprocess.Popen(
            engine_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(engine_path)
        )
        
        # Send basic UCI command
        print("  Sending: uci")
        process.stdin.write("uci\n")
        process.stdin.flush()
        
        # Wait for response
        start_time = time.time()
        output_lines = []
        
        while True:
            if time.time() - start_time > 10.0:
                print("  ❌ TIMEOUT after 10 seconds")
                process.kill()
                return False
                
            # Check if process has output
            try:
                process.stdout.settimeout(0.1)
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    output_lines.append(line)
                    print(f"  ← {line}")
                    
                    if line == "uciok":
                        print("  ✅ UCI communication successful")
                        break
                else:
                    time.sleep(0.1)
            except:
                time.sleep(0.1)
        
        # Test position setting
        print("  Sending: position startpos")
        process.stdin.write("position startpos\n")
        process.stdin.flush()
        time.sleep(0.5)
        
        print("  Sending: go movetime 1000")
        process.stdin.write("go movetime 1000\n")
        process.stdin.flush()
        
        # Wait for bestmove
        start_time = time.time()
        while True:
            if time.time() - start_time > 5.0:
                print("  ❌ TIMEOUT waiting for bestmove")
                process.kill()
                return False
                
            try:
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    print(f"  ← {line}")
                    
                    if line.startswith("bestmove"):
                        print("  ✅ Move generation successful")
                        break
                else:
                    time.sleep(0.1)
            except:
                time.sleep(0.1)
        
        # Clean shutdown
        print("  Sending: quit")
        process.stdin.write("quit\n")
        process.stdin.flush()
        process.wait(timeout=2.0)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False

def test_engine_direct_vs_subprocess():
    """Compare direct Python testing vs subprocess UCI"""
    print(f"\n{'='*60}")
    print("DIRECT PYTHON TEST vs UCI SUBPROCESS COMPARISON")
    print(f"{'='*60}")
    
    # Test the engine directly using our earlier test
    print("\n🐍 Direct Python Test (like test_v12_2_tournament_ready.py):")
    
    try:
        # Import and run direct test
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
        
        from v7p3r import V7P3REngine
        import chess
        
        # Test directly
        engine = V7P3REngine()
        board = chess.Board()
        
        start_time = time.time()
        best_move, info = engine.get_best_move(board, time_limit=2.0)
        elapsed = time.time() - start_time
        
        print(f"  ✅ Direct test successful")
        print(f"  Move: {best_move}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Depth: {info.get('depth', 'Unknown')}")
        print(f"  Nodes: {info.get('nodes', 'Unknown')}")
        
    except Exception as e:
        print(f"  ❌ Direct test failed: {e}")
    
    # Test via UCI subprocess
    print(f"\n🔧 UCI Subprocess Test:")
    engine_path = r"s:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester\engines\V7P3R\V7P3R_v12.2.exe"
    test_engine_uci_basic(engine_path)

def main():
    print("🔧 V7P3R v12.2 UCI Debug Test")
    print("🎯 Investigating timeout issues in tournament validation")
    
    # Test all three engines
    engines = {
        "V12.2": r"s:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester\engines\V7P3R\V7P3R_v12.2.exe",
        "V12.0": r"s:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester\engines\V7P3R\V7P3R_v12.0.exe",
        "V10.8": r"s:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester\engines\V7P3R\V7P3R_v10.8.exe"
    }
    
    for name, path in engines.items():
        print(f"\n{'='*40}")
        success = test_engine_uci_basic(path)
        print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
    
    # Special focus on v12.2
    print(f"\n{'='*60}")
    print("V12.2 DETAILED ANALYSIS")
    test_engine_direct_vs_subprocess()

if __name__ == "__main__":
    main()