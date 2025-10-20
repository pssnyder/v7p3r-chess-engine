#!/usr/bin/env python3
"""
VPR vs V7P3R Performance Comparison Test

Compares the barebones VPR engine against the full V7P3R engine
to measure depth reached, nodes searched, and time spent.
"""

import sys
import os
import time
import chess

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vpr import VPREngine
from v7p3r import V7P3REngine


def test_position(fen: str, description: str, time_limit: float = 3.0):
    """
    Test both engines on a single position
    
    Args:
        fen: Position to test
        description: Human-readable description
        time_limit: Time limit in seconds
    """
    print(f"\n{'='*70}")
    print(f"Position: {description}")
    print(f"FEN: {fen}")
    print(f"Time limit: {time_limit}s")
    print(f"{'='*70}")
    
    board = chess.Board(fen)
    
    # Test VPR engine
    print("\n[VPR Engine - Barebones]")
    vpr = VPREngine()
    start = time.time()
    vpr_move = vpr.search(board, time_limit=time_limit)
    vpr_time = time.time() - start
    vpr_nodes = vpr.nodes_searched
    vpr_nps = int(vpr_nodes / vpr_time) if vpr_time > 0 else 0
    
    print(f"Best move: {vpr_move}")
    print(f"Nodes: {vpr_nodes:,}")
    print(f"Time: {vpr_time:.3f}s")
    print(f"NPS: {vpr_nps:,}")
    
    # Test V7P3R engine
    print("\n[V7P3R Engine - Full Featured]")
    board = chess.Board(fen)  # Reset board
    v7p3r = V7P3REngine()
    start = time.time()
    v7p3r_move = v7p3r.search(board, time_limit=time_limit)
    v7p3r_time = time.time() - start
    v7p3r_nodes = v7p3r.nodes_searched
    v7p3r_nps = int(v7p3r_nodes / v7p3r_time) if v7p3r_time > 0 else 0
    
    print(f"Best move: {v7p3r_move}")
    print(f"Nodes: {v7p3r_nodes:,}")
    print(f"Time: {v7p3r_time:.3f}s")
    print(f"NPS: {v7p3r_nps:,}")
    
    # Comparison
    print("\n[Comparison]")
    node_ratio = vpr_nodes / v7p3r_nodes if v7p3r_nodes > 0 else 0
    nps_ratio = vpr_nps / v7p3r_nps if v7p3r_nps > 0 else 0
    
    print(f"VPR searched {node_ratio:.2f}x nodes vs V7P3R")
    print(f"VPR achieved {nps_ratio:.2f}x NPS vs V7P3R")
    print(f"Same move: {'✓' if vpr_move == v7p3r_move else '✗'}")
    
    return {
        'description': description,
        'vpr_move': vpr_move,
        'vpr_nodes': vpr_nodes,
        'vpr_time': vpr_time,
        'vpr_nps': vpr_nps,
        'v7p3r_move': v7p3r_move,
        'v7p3r_nodes': v7p3r_nodes,
        'v7p3r_time': v7p3r_time,
        'v7p3r_nps': v7p3r_nps,
        'node_ratio': node_ratio,
        'nps_ratio': nps_ratio,
        'same_move': vpr_move == v7p3r_move
    }


def main():
    """Run comprehensive performance comparison"""
    print("VPR vs V7P3R Performance Comparison")
    print("====================================")
    
    # Test positions
    test_positions = [
        # Starting position
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        
        # Middle game position
        ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5", "Italian Game middlegame"),
        
        # Tactical position
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Center tension"),
        
        # Endgame position
        ("8/5k2/8/3K4/8/8/5P2/8 w - - 0 1", "King and pawn endgame"),
        
        # Complex middlegame
        ("r2qkb1r/pp2pppp/2n2n2/3p1b2/3P4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 7", "Complex middlegame")
    ]
    
    results = []
    time_limit = 3.0
    
    for fen, description in test_positions:
        try:
            result = test_position(fen, description, time_limit)
            results.append(result)
        except Exception as e:
            print(f"\nError testing {description}: {e}")
            continue
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    if results:
        avg_node_ratio = sum(r['node_ratio'] for r in results) / len(results)
        avg_nps_ratio = sum(r['nps_ratio'] for r in results) / len(results)
        same_moves = sum(1 for r in results if r['same_move'])
        
        print(f"\nTotal positions tested: {len(results)}")
        print(f"Average node ratio (VPR/V7P3R): {avg_node_ratio:.2f}x")
        print(f"Average NPS ratio (VPR/V7P3R): {avg_nps_ratio:.2f}x")
        print(f"Same move chosen: {same_moves}/{len(results)} ({same_moves/len(results)*100:.1f}%)")
        
        # Detailed table
        print(f"\n{'Position':<30} {'VPR Nodes':>12} {'V7P3R Nodes':>12} {'Ratio':>8} {'NPS Ratio':>10}")
        print("-" * 80)
        for r in results:
            print(f"{r['description'][:30]:<30} {r['vpr_nodes']:>12,} {r['v7p3r_nodes']:>12,} "
                  f"{r['node_ratio']:>8.2f}x {r['nps_ratio']:>9.2f}x")
        
        print("\n" + "="*70)
        print("CONCLUSION:")
        if avg_node_ratio > 1.5:
            print(f"✓ VPR searches significantly MORE nodes ({avg_node_ratio:.2f}x)")
            print("  This suggests lower overhead per node (goal achieved)")
        elif avg_node_ratio > 1.1:
            print(f"✓ VPR searches moderately more nodes ({avg_node_ratio:.2f}x)")
        else:
            print(f"✗ VPR does not search significantly more nodes ({avg_node_ratio:.2f}x)")
        
        if avg_nps_ratio > 1.2:
            print(f"✓ VPR achieves significantly higher NPS ({avg_nps_ratio:.2f}x)")
        elif avg_nps_ratio > 1.0:
            print(f"≈ VPR achieves slightly higher NPS ({avg_nps_ratio:.2f}x)")
        else:
            print(f"✗ VPR does not achieve higher NPS ({avg_nps_ratio:.2f}x)")
        
        print("="*70)


if __name__ == "__main__":
    main()
