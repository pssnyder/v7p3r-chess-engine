"""
Test search depth comparison between v17.3 and v17.1.1
Checks if SEE-based quiescence is throttling search depth
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chess
from src.v7p3r import V7P3REngine
import time

def test_depth_progression():
    """Test how deep each version searches in same time"""
    
    # Standard opening position after 1.e4 e5 2.Nf3 Nc6 3.Bc4
    test_position = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    
    print("\n" + "="*60)
    print("V7P3R Depth Progression Test")
    print("="*60)
    print(f"Position: {test_position.fen()}")
    print()
    
    # Test v17.3 (current)
    print("Testing v17.3 (SEE-based quiescence)...")
    print("-" * 60)
    engine_v17_3 = V7P3REngine()
    
    v17_3_results = []
    for depth in range(1, 8):
        start = time.time()
        move, eval_score, nodes = engine_v17_3.search(test_position, depth=depth)
        elapsed = time.time() - start
        
        # Get seldepth from last search
        seldepth = engine_v17_3.max_seldepth
        nps = int(nodes / elapsed) if elapsed > 0 else 0
        
        v17_3_results.append({
            'depth': depth,
            'seldepth': seldepth,
            'nodes': nodes,
            'time': elapsed,
            'nps': nps,
            'move': move.uci() if move else 'none'
        })
        
        print(f"Depth {depth}: seldepth={seldepth}, nodes={nodes:,}, time={elapsed:.2f}s, nps={nps:,}, move={move.uci() if move else 'none'}")
    
    print("\n" + "="*60)
    print("v17.3 Summary:")
    print(f"  Max seldepth reached: {max(r['seldepth'] for r in v17_3_results)}")
    print(f"  Total nodes: {sum(r['nodes'] for r in v17_3_results):,}")
    print(f"  Avg NPS: {int(sum(r['nps'] for r in v17_3_results) / len(v17_3_results)):,}")
    
    # Calculate quiescence overhead
    total_seldepth_overhead = sum(r['seldepth'] - r['depth'] for r in v17_3_results)
    avg_overhead = total_seldepth_overhead / len(v17_3_results)
    print(f"  Avg quiescence overhead: {avg_overhead:.1f} plies")
    print("="*60)
    
    return v17_3_results

def test_quiescence_behavior():
    """Test quiescence search behavior directly"""
    
    print("\n" + "="*60)
    print("Quiescence Behavior Analysis")
    print("="*60)
    
    engine = V7P3REngine()
    
    # Tactical position with captures
    tactical_pos = chess.Board("r1bqk2r/ppp2ppp/2n5/2bpp3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 6")
    
    print(f"\nTactical position: {tactical_pos.fen()}")
    print("\nTesting quiescence on d1-h5 fork threat...")
    
    # Test a few depths
    for depth in [3, 5, 7]:
        engine.max_seldepth = 0  # Reset
        move, eval_score, nodes = engine.search(tactical_pos, depth=depth)
        
        print(f"\nDepth {depth}:")
        print(f"  Best move: {move.uci() if move else 'none'}")
        print(f"  Seldepth: {engine.max_seldepth}")
        print(f"  Nodes: {nodes:,}")
        print(f"  Quiescence extension: {engine.max_seldepth - depth} plies")
        
        # Check if quiescence is even being called
        if engine.max_seldepth == depth:
            print(f"  ⚠️ WARNING: No quiescence extension! Seldepth = depth")
    
    print("="*60)

if __name__ == "__main__":
    print("\nRunning depth comparison test...")
    v17_3_results = test_depth_progression()
    
    print("\n")
    test_quiescence_behavior()
    
    print("\n✅ Test complete")
