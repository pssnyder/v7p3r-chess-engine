#!/usr/bin/env python3
"""
Simple V12.6 vs V12.2 Direct Testing
Direct command-line testing without complex subprocess handling
"""

import os
import subprocess
import time
from pathlib import Path


def test_engine_basic(engine_path, version_name):
    """Test basic engine functionality"""
    print(f"\n=== Testing {version_name} ===")
    
    # Test UCI command
    print(f"Testing UCI response...")
    result = subprocess.run(
        [str(engine_path)],
        input="uci\nquit\n",
        text=True,
        capture_output=True,
        timeout=5
    )
    
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if line.startswith("id name"):
                print(f"✅ Engine ID: {line}")
            elif line == "uciok":
                print(f"✅ UCI protocol: OK")
    else:
        print(f"❌ Engine failed to start: {result.stderr}")
        return False
        
    # Test simple position analysis
    print(f"Testing position analysis...")
    result = subprocess.run(
        [str(engine_path)],
        input="uci\nposition startpos\ngo movetime 1000\nquit\n",
        text=True,
        capture_output=True,
        timeout=10
    )
    
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        bestmove = None
        depth_reached = 0
        nodes_searched = 0
        
        for line in lines:
            if line.startswith("bestmove"):
                bestmove = line.split()[1] if len(line.split()) > 1 else "none"
            elif "depth" in line and "score" in line:
                parts = line.split()
                try:
                    depth_idx = parts.index("depth")
                    depth_reached = max(depth_reached, int(parts[depth_idx + 1]))
                except:
                    pass
                try:
                    nodes_idx = parts.index("nodes")
                    nodes_searched = max(nodes_searched, int(parts[nodes_idx + 1]))
                except:
                    pass
        
        print(f"✅ Best move: {bestmove}")
        print(f"✅ Max depth reached: {depth_reached}")
        print(f"✅ Nodes searched: {nodes_searched}")
        
        return {
            "bestmove": bestmove,
            "depth": depth_reached,
            "nodes": nodes_searched
        }
    else:
        print(f"❌ Analysis failed: {result.stderr}")
        return None


def test_position_comparison(engine1_path, engine2_path, name1, name2, fen, position_name):
    """Compare both engines on a specific position"""
    print(f"\n=== Position Test: {position_name} ===")
    print(f"FEN: {fen}")
    
    results = {}
    
    for engine_path, name in [(engine1_path, name1), (engine2_path, name2)]:
        print(f"\nTesting {name}...")
        
        if fen == "startpos":
            position_cmd = "position startpos"
        else:
            position_cmd = f"position fen {fen}"
            
        result = subprocess.run(
            [str(engine_path)],
            input=f"uci\n{position_cmd}\ngo movetime 2000\nquit\n",
            text=True,
            capture_output=True,
            timeout=15
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            bestmove = "none"
            best_eval = "N/A"
            depth = 0
            nodes = 0
            
            for line in lines:
                if line.startswith("bestmove"):
                    bestmove = line.split()[1] if len(line.split()) > 1 else "none"
                elif "score cp" in line:
                    parts = line.split()
                    try:
                        score_idx = parts.index("score")
                        if score_idx + 2 < len(parts) and parts[score_idx + 1] == "cp":
                            best_eval = int(parts[score_idx + 2])
                    except:
                        pass
                elif "depth" in line:
                    parts = line.split()
                    try:
                        depth_idx = parts.index("depth")
                        depth = max(depth, int(parts[depth_idx + 1]))
                    except:
                        pass
                elif "nodes" in line:
                    parts = line.split()
                    try:
                        nodes_idx = parts.index("nodes")
                        nodes = max(nodes, int(parts[nodes_idx + 1]))
                    except:
                        pass
            
            results[name] = {
                "move": bestmove,
                "eval": best_eval, 
                "depth": depth,
                "nodes": nodes
            }
            
            print(f"Move: {bestmove}, Eval: {best_eval}, Depth: {depth}, Nodes: {nodes}")
        else:
            print(f"❌ {name} failed on position")
            results[name] = None
    
    # Compare results
    if name1 in results and name2 in results and results[name1] and results[name2]:
        r1, r2 = results[name1], results[name2]
        
        print(f"\n📊 Comparison:")
        if r1['move'] == r2['move']:
            print(f"✅ Both chose same move: {r1['move']}")
        else:
            print(f"⚠️  Different moves: {name1}={r1['move']}, {name2}={r2['move']}")
            
        if r1['eval'] != "N/A" and r2['eval'] != "N/A":
            eval_diff = abs(r1['eval'] - r2['eval'])
            print(f"📈 Evaluation difference: {eval_diff} centipawns")
            
        nodes_ratio = r2['nodes'] / max(r1['nodes'], 1)
        print(f"🔍 {name2} searched {nodes_ratio:.2f}x as many nodes as {name1}")
    
    return results


def main():
    base_path = Path("s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine")
    v12_2_path = base_path / "dist" / "V7P3R_v12.2.exe"
    v12_6_path = base_path / "dist" / "V7P3R_v12.6.exe"
    
    print("V7P3R V12.6 vs V12.2 DIRECT TESTING")
    print("="*40)
    
    # Verify engines exist
    if not v12_2_path.exists():
        print(f"❌ V12.2 not found: {v12_2_path}")
        return
    if not v12_6_path.exists():
        print(f"❌ V12.6 not found: {v12_6_path}")
        return
        
    print(f"✅ V12.2: {v12_2_path}")
    print(f"✅ V12.6: {v12_6_path}")
    
    # Test basic functionality
    v12_2_basic = test_engine_basic(v12_2_path, "V12.2")
    v12_6_basic = test_engine_basic(v12_6_path, "V12.6")
    
    if not v12_2_basic or not v12_6_basic:
        print("❌ Basic tests failed, stopping")
        return
    
    # Compare on test positions
    test_positions = [
        ("startpos", "Starting Position"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "After 1.e4"),
        ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3", "Italian Game Setup"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4", "Four Knights Opening")
    ]
    
    all_results = {}
    
    for fen, name in test_positions:
        results = test_position_comparison(v12_2_path, v12_6_path, "V12.2", "V12.6", fen, name)
        all_results[name] = results
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    # Basic performance comparison
    print(f"\nBasic Performance:")
    print(f"V12.2 - Move: {v12_2_basic['bestmove']}, Depth: {v12_2_basic['depth']}, Nodes: {v12_2_basic['nodes']}")
    print(f"V12.6 - Move: {v12_6_basic['bestmove']}, Depth: {v12_6_basic['depth']}, Nodes: {v12_6_basic['nodes']}")
    
    # Count agreement vs disagreement
    agreements = 0
    disagreements = 0
    
    for pos_name, results in all_results.items():
        if "V12.2" in results and "V12.6" in results and results["V12.2"] and results["V12.6"]:
            if results["V12.2"]["move"] == results["V12.6"]["move"]:
                agreements += 1
            else:
                disagreements += 1
                print(f"❗ Disagreement on {pos_name}: V12.2={results['V12.2']['move']}, V12.6={results['V12.6']['move']}")
    
    total_positions = agreements + disagreements
    if total_positions > 0:
        agreement_rate = agreements / total_positions * 100
        print(f"\n📊 Move Agreement: {agreements}/{total_positions} ({agreement_rate:.1f}%)")
        
        if agreement_rate >= 75:
            print("✅ High agreement - V12.6 appears consistent with V12.2")
        elif agreement_rate >= 50:
            print("⚠️  Moderate agreement - some differences detected")
        else:
            print("❌ Low agreement - significant differences detected")
    
    print(f"\n🏁 Testing complete!")


if __name__ == "__main__":
    main()