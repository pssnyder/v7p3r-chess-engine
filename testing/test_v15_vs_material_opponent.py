#!/usr/bin/env python3
"""
V15.0 vs Material Opponent - Direct Comparison
Tests identical positions with both engines to compare performance
"""

import sys
import os
import subprocess
import time
import chess
from typing import Dict, Optional

# Add V15.0 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from v7p3r_engine import V7P3REngine


def send_uci_command(process, command: str, wait_for: str = None, timeout: float = 10.0) -> str:
    """Send UCI command and get response"""
    process.stdin.write(f"{command}\n")
    process.stdin.flush()
    
    if wait_for:
        start = time.time()
        output = []
        while time.time() - start < timeout:
            line = process.stdout.readline().strip()
            output.append(line)
            if wait_for in line:
                break
        return "\n".join(output)
    return ""


def get_material_opponent_move(fen: str, time_ms: int = 3000) -> Dict:
    """Get move from Material Opponent engine"""
    material_path = r"s:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester\engines\Opponents\beginner\MaterialOpponent_v1.0\material_opponent.py"
    
    try:
        # Start Material Opponent
        process = subprocess.Popen(
            ['python', material_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Initialize UCI
        send_uci_command(process, "uci", "uciok")
        send_uci_command(process, "isready", "readyok")
        send_uci_command(process, "ucinewgame")
        
        # Set position and search
        send_uci_command(process, f"position fen {fen}")
        
        start = time.time()
        output = send_uci_command(process, f"go movetime {time_ms}", "bestmove", timeout=time_ms/1000 + 2)
        elapsed = time.time() - start
        
        # Parse output
        nodes = 0
        nps = 0
        best_move = None
        depth_reached = 0
        
        for line in output.split('\n'):
            if 'nodes' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'nodes' and i + 1 < len(parts):
                        try:
                            nodes = int(parts[i + 1])
                        except:
                            pass
                    if part == 'nps' and i + 1 < len(parts):
                        try:
                            nps = int(parts[i + 1])
                        except:
                            pass
                    if part == 'depth' and i + 1 < len(parts):
                        try:
                            depth_reached = int(parts[i + 1])
                        except:
                            pass
            
            if 'bestmove' in line:
                parts = line.split()
                if len(parts) >= 2:
                    best_move = parts[1]
        
        process.terminate()
        
        return {
            "move": best_move,
            "nodes": nodes,
            "time": elapsed,
            "nps": nps,
            "depth": depth_reached
        }
        
    except Exception as e:
        print(f"Error running Material Opponent: {e}")
        return None


def run_comparison():
    """Run head-to-head comparison"""
    print("=" * 80)
    print("V15.0 vs MATERIAL OPPONENT - HEAD-TO-HEAD COMPARISON")
    print("=" * 80)
    
    v15 = V7P3REngine()
    
    # Test positions
    tests = [
        {
            "name": "Opening Position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "time_ms": 3000
        },
        {
            "name": "Tactical Fork",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5",
            "time_ms": 3000
        },
        {
            "name": "Mate in 1",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "time_ms": 2000
        },
        {
            "name": "Complex Middlegame",
            "fen": "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 2 9",
            "time_ms": 3000
        }
    ]
    
    results = []
    
    for test in tests:
        print(f"\n{'=' * 80}")
        print(f"🎯 {test['name']}")
        print(f"{'=' * 80}")
        print(f"FEN: {test['fen']}")
        print(f"Time: {test['time_ms']}ms")
        
        board = chess.Board(test['fen'])
        
        # Test V15.0
        print(f"\n🔵 V15.0 Thinking...")
        v15.new_game()
        start = time.time()
        v15_move = v15.search(board, time_limit=test['time_ms']/1000)
        v15_time = time.time() - start
        v15_nodes = v15.nodes_searched
        v15_nps = int(v15_nodes / v15_time) if v15_time > 0 else 0
        
        print(f"  Move: {v15_move}")
        print(f"  Nodes: {v15_nodes:,}")
        print(f"  Time: {v15_time:.3f}s")
        print(f"  NPS: {v15_nps:,}")
        
        # Test Material Opponent
        print(f"\n🟡 Material Opponent Thinking...")
        mat_result = get_material_opponent_move(test['fen'], test['time_ms'])
        
        if mat_result:
            print(f"  Move: {mat_result['move']}")
            print(f"  Nodes: {mat_result['nodes']:,}")
            print(f"  Time: {mat_result['time']:.3f}s")
            print(f"  NPS: {mat_result['nps']:,}")
            
            # Compare
            print(f"\n📊 Comparison:")
            print(f"  Same Move: {'✅ YES' if str(v15_move) == mat_result['move'] else '❌ NO'}")
            
            if mat_result['nodes'] > 0:
                node_ratio = v15_nodes / mat_result['nodes']
                print(f"  V15 Node Ratio: {node_ratio:.2f}x")
            
            if mat_result['nps'] > 0:
                nps_ratio = v15_nps / mat_result['nps']
                print(f"  V15 Speed Ratio: {nps_ratio:.2f}x")
        else:
            print("  ❌ Material Opponent failed to respond")
        
        results.append({
            "test": test['name'],
            "v15_move": str(v15_move),
            "v15_nodes": v15_nodes,
            "v15_nps": v15_nps,
            "mat_move": mat_result['move'] if mat_result else None,
            "mat_nodes": mat_result['nodes'] if mat_result else 0,
            "mat_nps": mat_result['nps'] if mat_result else 0
        })
    
    # Summary
    print(f"\n{'=' * 80}")
    print("📈 OVERALL COMPARISON")
    print(f"{'=' * 80}")
    
    v15_total_nodes = sum(r['v15_nodes'] for r in results)
    mat_total_nodes = sum(r['mat_nodes'] for r in results if r['mat_nodes'])
    
    v15_avg_nps = sum(r['v15_nps'] for r in results) / len(results)
    mat_avg_nps = sum(r['mat_nps'] for r in results if r['mat_nps']) / len([r for r in results if r['mat_nps']])
    
    print(f"\n🔵 V15.0:")
    print(f"  Total Nodes: {v15_total_nodes:,}")
    print(f"  Average NPS: {v15_avg_nps:,.0f}")
    
    print(f"\n🟡 Material Opponent:")
    print(f"  Total Nodes: {mat_total_nodes:,}")
    print(f"  Average NPS: {mat_avg_nps:,.0f}")
    
    if mat_total_nodes > 0:
        print(f"\n📊 V15.0 vs Material Opponent:")
        print(f"  Node Efficiency: {v15_total_nodes / mat_total_nodes:.2f}x")
        print(f"  Speed Ratio: {v15_avg_nps / mat_avg_nps:.2f}x")
    
    # Agreement
    same_moves = sum(1 for r in results if r['v15_move'] == r['mat_move'])
    print(f"\n🤝 Move Agreement: {same_moves}/{len(results)} positions")
    
    return results


if __name__ == "__main__":
    try:
        results = run_comparison()
        print("\n✅ Comparison complete!")
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
