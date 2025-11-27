"""
Test search depth comparison between v17.3 and v17.1
Using Tournament Engines src versions for consistency
"""
import sys
import os
import chess
import time

def test_version(version_path, version_name):
    """Test a specific version's depth progression"""
    
    # Add version to path
    sys.path.insert(0, version_path)
    
    # Import fresh copy
    if 'v7p3r' in sys.modules:
        del sys.modules['v7p3r']
    from v7p3r import V7P3REngine
    
    # Standard opening position after 1.e4 e5 2.Nf3 Nc6 3.Bc4
    test_position = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    
    print(f"\nTesting {version_name}...")
    print("-" * 60)
    
    engine = V7P3REngine()
    
    results = []
    for depth in range(1, 8):
        start = time.time()
        move, eval_score, nodes = engine.search(test_position, depth=depth)
        elapsed = time.time() - start
        
        seldepth = engine.max_seldepth
        nps = int(nodes / elapsed) if elapsed > 0 else 0
        
        results.append({
            'depth': depth,
            'seldepth': seldepth,
            'nodes': nodes,
            'time': elapsed,
            'nps': nps,
            'move': move.uci() if move else 'none',
            'overhead': seldepth - depth
        })
        
        print(f"Depth {depth}: seldepth={seldepth:2d} (+{seldepth-depth}), nodes={nodes:>8,}, time={elapsed:5.2f}s, nps={nps:>8,}")
    
    print()
    print(f"{version_name} Summary:")
    print(f"  Max seldepth: {max(r['seldepth'] for r in results)}")
    print(f"  Total nodes: {sum(r['nodes'] for r in results):,}")
    print(f"  Avg NPS: {int(sum(r['nps'] for r in results) / len(results)):,}")
    
    total_overhead = sum(r['overhead'] for r in results)
    avg_overhead = total_overhead / len(results)
    print(f"  Avg quiescence overhead: {avg_overhead:.1f} plies")
    print(f"  Max quiescence extension: {max(r['overhead'] for r in results)} plies")
    
    # Remove from path
    sys.path.remove(version_path)
    
    return results

if __name__ == "__main__":
    print("\n" + "="*60)
    print("V7P3R Depth Progression Comparison")
    print("="*60)
    print("Position: Italian Game (r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R)")
    print("="*60)
    
    # Test v17.1
    v17_1_path = "s:/Programming/Chess Engines/Tournament Engines/V7P3R/V7P3R_v17.1/src"
    v17_1_results = test_version(v17_1_path, "v17.1")
    
    # Test v17.3
    v17_3_path = "s:/Programming/Chess Engines/Tournament Engines/V7P3R/V7P3R_v17.3/src"
    v17_3_results = test_version(v17_3_path, "v17.3")
    
    # Comparison
    print("="*60)
    print("COMPARISON:")
    print("="*60)
    print(f"{'Depth':<8} {'v17.1':<20} {'v17.3':<20} {'Difference':<15}")
    print("-" * 60)
    
    for i in range(len(v17_1_results)):
        v171 = v17_1_results[i]
        v173 = v17_3_results[i]
        depth = v171['depth']
        
        v171_sd = v171['seldepth']
        v173_sd = v173['seldepth']
        sd_diff = v173_sd - v171_sd
        
        v171_nodes = v171['nodes']
        v173_nodes = v173['nodes']
        nodes_pct = ((v173_nodes - v171_nodes) / v171_nodes * 100) if v171_nodes > 0 else 0
        
        print(f"D={depth:<5} seldepth={v171_sd:2d} (+{v171['overhead']})   seldepth={v173_sd:2d} (+{v173['overhead']})   {sd_diff:+3d} plies")
        print(f"        nodes={v171_nodes:>8,}      nodes={v173_nodes:>8,}      {nodes_pct:+6.1f}%")
        print()
    
    # Summary comparison
    v171_avg_overhead = sum(r['overhead'] for r in v17_1_results) / len(v17_1_results)
    v173_avg_overhead = sum(r['overhead'] for r in v17_3_results) / len(v17_3_results)
    
    v171_total_nodes = sum(r['nodes'] for r in v17_1_results)
    v173_total_nodes = sum(r['nodes'] for r in v17_3_results)
    
    print("="*60)
    print("OVERALL COMPARISON:")
    print(f"  v17.1 avg quiescence overhead: {v171_avg_overhead:.1f} plies")
    print(f"  v17.3 avg quiescence overhead: {v173_avg_overhead:.1f} plies")
    print(f"  Change: {v173_avg_overhead - v171_avg_overhead:+.1f} plies ({(v173_avg_overhead - v171_avg_overhead) / v171_avg_overhead * 100:+.1f}%)")
    print()
    print(f"  v17.1 total nodes: {v171_total_nodes:,}")
    print(f"  v17.3 total nodes: {v173_total_nodes:,}")
    print(f"  Change: {(v173_total_nodes - v171_total_nodes) / v171_total_nodes * 100:+.1f}%")
    print("="*60)
    
    # Diagnosis
    if v173_avg_overhead < v171_avg_overhead * 0.5:
        print("\n⚠️ WARNING: v17.3 quiescence overhead is SIGNIFICANTLY LOWER")
        print("   SEE filtering may be too aggressive, throttling tactical search")
    elif v173_avg_overhead < v171_avg_overhead * 0.7:
        print("\n⚠️ CAUTION: v17.3 quiescence overhead is notably lower")
        print("   This may impact tactical depth in complex positions")
    else:
        print("\n✅ Quiescence overhead appears reasonable")
    
    print("\n✅ Test complete")
