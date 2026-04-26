#!/usr/bin/env python3
"""
Simple Direct Comparison: v7p3r Phase 1 vs MaterialOpponent

No profiling overhead - just run both engines and compare UCI output.
Goal: Understand why v7p3r reaches lower depths despite comparable/better NPS.
"""

import sys
import os
import chess
import subprocess
import time
import re

def parse_uci_output(output):
    """Parse UCI output to extract depth, nodes, time, NPS from final depth"""
    lines = output.strip().split('\n')
    final_depth = None
    final_nodes = None
    final_time_ms = None
    final_nps = None
    final_pv = None
    
    for line in lines:
        if line.startswith('info depth'):
            # Extract depth, nodes, time, nps
            depth_match = re.search(r'depth (\d+)', line)
            nodes_match = re.search(r'nodes (\d+)', line)
            time_match = re.search(r'time (\d+)', line)
            nps_match = re.search(r'nps (\d+)', line)
            pv_match = re.search(r'pv (.+)$', line)
            
            if depth_match:
                depth = int(depth_match.group(1))
                if final_depth is None or depth > final_depth:
                    final_depth = depth
                    final_nodes = int(nodes_match.group(1)) if nodes_match else None
                    final_time_ms = int(time_match.group(1)) if time_match else None
                    final_nps = int(nps_match.group(1)) if nps_match else None
                    final_pv = pv_match.group(1) if pv_match else None
    
    return {
        'final_depth': final_depth,
        'nodes': final_nodes,
        'time_ms': final_time_ms,
        'nps': final_nps,
        'pv': final_pv
    }

def test_phase1(fen, search_time=10.0):
    """Test v7p3r Phase 1"""
    print(f"=" * 80)
    print(f"TESTING v7p3r Phase 1 ({search_time}s search)")
    print(f"=" * 80)
    
    code = f"""
import sys
sys.path.insert(0, r'E:\\Programming Stuff\\Chess Engines\\V7P3R Chess Engine\\v7p3r-chess-engine\\src')
from v7p3r import V7P3REngine
import chess

board = chess.Board('{fen}')
engine = V7P3REngine()
move = engine.search(board, time_limit={search_time})
print(f'\\nBest move: {{move}}')
"""
    
    start = time.time()
    result = subprocess.run(
        ['python', '-c', code],
        capture_output=True,
        text=True,
        cwd=r'E:\Programming Stuff\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine'
    )
    elapsed = time.time() - start
    
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    
    stats = parse_uci_output(result.stdout)
    stats['actual_time'] = elapsed
    
    return stats

def test_materialopponent(fen, search_time=10.0):
    """Test MaterialOpponent"""
    print(f"\n" + "=" * 80)
    print(f"TESTING MaterialOpponent ({search_time}s search)")
    print(f"=" * 80)
    
    code = f"""
import sys
sys.path.insert(0, r'E:\\Programming Stuff\\Chess Engines\\Opponent Chess Engines\\opponent-chess-engines\\src\\MaterialOpponent')
from material_opponent import MaterialOpponent
import chess

board = chess.Board('{fen}')
engine = MaterialOpponent(max_depth=10)
engine.board = board
engine.time_limit = {search_time}

# Run search with fixed time limit
import time
start = time.time()
move = engine.get_best_move(time_left=0)  # 0 = use engine.time_limit
elapsed = time.time() - start

print(f'\\nBest move: {{move}}')
print(f'Actual search time: {{elapsed:.2f}}s')
"""
    
    start = time.time()
    result = subprocess.run(
        ['python', '-c', code],
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start
    
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    
    stats = parse_uci_output(result.stdout)
    stats['actual_time'] = elapsed
    
    return stats

def main():
    print("""
================================================================================
DIRECT COMPARISON: v7p3r Phase 1 vs MaterialOpponent
================================================================================
No profiling overhead - just run both engines and compare raw performance.

Goal: Understand depth gap (Phase 1: 5-6 vs MaterialOpponent: 10)
Test: 10 second search on standard middlegame position
================================================================================
""")
    
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
    search_time = 10.0
    
    print(f"Position: {fen}")
    print(f"Search time: {search_time}s\n")
    
    # Test Phase 1
    phase1_stats = test_phase1(fen, search_time)
    
    # Test MaterialOpponent
    material_stats = test_materialopponent(fen, search_time)
    
    # Compare
    print(f"\n" + "=" * 80)
    print(f"COMPARISON")
    print(f"=" * 80)
    print(f"\n{'Metric':<25} {'Phase 1':<20} {'MaterialOpp':<20} {'Difference'}")
    print(f"─" * 85)
    
    if phase1_stats['final_depth'] and material_stats['final_depth']:
        depth_diff = material_stats['final_depth'] - phase1_stats['final_depth']
        print(f"{'Final Depth':<25} {phase1_stats['final_depth']:<20} {material_stats['final_depth']:<20} {depth_diff:+d}")
    
    if phase1_stats['nodes'] and material_stats['nodes']:
        nodes_diff = material_stats['nodes'] - phase1_stats['nodes']
        nodes_pct = (nodes_diff / phase1_stats['nodes']) * 100
        print(f"{'Nodes Searched':<25} {phase1_stats['nodes']:<20,} {material_stats['nodes']:<20,} {nodes_diff:+,} ({nodes_pct:+.1f}%)")
    
    if phase1_stats['nps'] and material_stats['nps']:
        nps_diff = material_stats['nps'] - phase1_stats['nps']
        nps_pct = (nps_diff / phase1_stats['nps']) * 100
        print(f"{'NPS (from UCI)':<25} {phase1_stats['nps']:<20,} {material_stats['nps']:<20,} {nps_diff:+,} ({nps_pct:+.1f}%)")
    
    if phase1_stats['nodes'] and phase1_stats['actual_time']:
        phase1_true_nps = phase1_stats['nodes'] / phase1_stats['actual_time']
        material_true_nps = material_stats['nodes'] / material_stats['actual_time'] if material_stats['actual_time'] > 0 else 0
        nps_diff = material_true_nps - phase1_true_nps
        nps_pct = (nps_diff / phase1_true_nps) * 100
        print(f"{'NPS (actual)':<25} {phase1_true_nps:<20,.0f} {material_true_nps:<20,.0f} {nps_diff:+,.0f} ({nps_pct:+.1f}%)")
    
    print(f"{'Actual Time':<25} {phase1_stats['actual_time']:<20.2f}s {material_stats['actual_time']:<20.2f}s")
    
    # Key insight
    print(f"\n" + "=" * 80)
    print(f"KEY INSIGHTS")
    print(f"=" * 80)
    
    if phase1_stats['nodes'] and material_stats['nodes']:
        nodes_ratio = phase1_stats['nodes'] / material_stats['nodes']
        print(f"\nPhase 1 searched {nodes_ratio:.1f}x as many nodes as MaterialOpponent")
        
        if nodes_ratio > 2:
            print(f"🔴 BRANCHING FACTOR ISSUE: Phase 1 is exploring WAY more nodes")
            print(f"   This suggests:")
            print(f"   - Weaker move ordering (bad moves searched deeply)")
            print(f"   - Ineffective pruning (not cutting off bad branches)")
            print(f"   - TT not helping enough (re-searching same positions)")
        elif nodes_ratio > 1.5:
            print(f"⚠️  Phase 1 explores 50%+ more nodes - room for optimization")
        else:
            print(f"✓ Node count is reasonable")
    
    if phase1_stats['final_depth'] and material_stats['final_depth']:
        depth_gap = material_stats['final_depth'] - phase1_stats['final_depth']
        if depth_gap >= 4:
            print(f"\n🔴 DEPTH GAP: MaterialOpponent reaches depth {material_stats['final_depth']} vs Phase 1's {phase1_stats['final_depth']}")
            print(f"   With comparable/better NPS, this suggests excessive node exploration")

if __name__ == "__main__":
    main()
