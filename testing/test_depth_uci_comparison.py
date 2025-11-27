"""
UCI-based depth comparison between v17.3 and v17.1
Tests seldepth progression across different depths
"""
import subprocess
import time
import re

def run_uci_search(bat_path, fen, depth):
    """Run UCI search and extract info"""
    commands = [
        "uci",
        "isready",
        f"position fen {fen}",
        f"go depth {depth}",
        "quit"
    ]
    
    cmd_input = "\n".join(commands) + "\n"
    
    try:
        result = subprocess.run(
            [bat_path],
            input=cmd_input,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout
        
        # Extract final info line before bestmove
        info_lines = [l for l in output.split('\n') if l.startswith('info')]
        if not info_lines:
            return None
        
        last_info = info_lines[-1]
        
        # Parse info line
        info = {}
        if 'seldepth' in last_info:
            match = re.search(r'seldepth (\d+)', last_info)
            info['seldepth'] = int(match.group(1)) if match else 0
        else:
            info['seldepth'] = 0
            
        if 'nodes' in last_info:
            match = re.search(r'nodes (\d+)', last_info)
            info['nodes'] = int(match.group(1)) if match else 0
        else:
            info['nodes'] = 0
            
        if 'nps' in last_info:
            match = re.search(r'nps (\d+)', last_info)
            info['nps'] = int(match.group(1)) if match else 0
        else:
            info['nps'] = 0
        
        return info
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Italian Game position
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    
    v17_1_bat = "s:/Programming/Chess Engines/Tournament Engines/V7P3R/V7P3R_v17.1/V7P3R_v17.1.bat"
    v17_3_bat = "s:/Programming/Chess Engines/Tournament Engines/V7P3R/V7P3R_v17.3/V7P3R_v17.3.bat"
    
    print("\n" + "="*60)
    print("V7P3R Depth Progression Comparison (UCI)")
    print("="*60)
    print(f"Position: Italian Game")
    print(f"FEN: {fen}")
    print("="*60)
    
    # Test v17.1
    print("\nTesting v17.1...")
    print("-" * 60)
    v17_1_results = []
    for depth in range(1, 8):
        print(f"Depth {depth}...", end=" ", flush=True)
        info = run_uci_search(v17_1_bat, fen, depth)
        if info:
            v17_1_results.append({'depth': depth, **info})
            overhead = info['seldepth'] - depth
            print(f"seldepth={info['seldepth']:2d} (+{overhead}), nodes={info['nodes']:>8,}, nps={info['nps']:>8,}")
        else:
            print("FAILED")
    
    # Test v17.3
    print("\nTesting v17.3...")
    print("-" * 60)
    v17_3_results = []
    for depth in range(1, 8):
        print(f"Depth {depth}...", end=" ", flush=True)
        info = run_uci_search(v17_3_bat, fen, depth)
        if info:
            v17_3_results.append({'depth': depth, **info})
            overhead = info['seldepth'] - depth
            print(f"seldepth={info['seldepth']:2d} (+{overhead}), nodes={info['nodes']:>8,}, nps={info['nps']:>8,}")
        else:
            print("FAILED")
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON:")
    print("="*60)
    print(f"{'Depth':<8} {'v17.1':<25} {'v17.3':<25} {'Difference':<15}")
    print("-" * 60)
    
    for i in range(min(len(v17_1_results), len(v17_3_results))):
        v171 = v17_1_results[i]
        v173 = v17_3_results[i]
        depth = v171['depth']
        
        v171_overhead = v171['seldepth'] - depth
        v173_overhead = v173['seldepth'] - depth
        sd_diff = v173['seldepth'] - v171['seldepth']
        
        v171_nodes = v171['nodes']
        v173_nodes = v173['nodes']
        nodes_pct = ((v173_nodes - v171_nodes) / v171_nodes * 100) if v171_nodes > 0 else 0
        
        print(f"D={depth:<5} seldepth={v171['seldepth']:2d} (+{v171_overhead})     seldepth={v173['seldepth']:2d} (+{v173_overhead})     {sd_diff:+3d} plies")
        print(f"        nodes={v171_nodes:>10,}    nodes={v173_nodes:>10,}    {nodes_pct:+6.1f}%")
        print()
    
    # Summary
    if v17_1_results and v17_3_results:
        v171_avg_overhead = sum(r['seldepth'] - r['depth'] for r in v17_1_results) / len(v17_1_results)
        v173_avg_overhead = sum(r['seldepth'] - r['depth'] for r in v17_3_results) / len(v17_3_results)
        
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
