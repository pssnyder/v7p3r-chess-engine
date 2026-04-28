#!/usr/bin/env python3
"""
V19.5.6 Efficiency Analysis

While v19.5.6 passed validation (50% win rate, 0 timeouts), 
let's analyze search efficiency to identify optimization opportunities.

Key metrics:
- NPS (nodes per second)
- Depth per second
- Time per depth level
- Move ordering effectiveness
- TT hit rate
"""

import subprocess
import time
import chess
from pathlib import Path

V19_PATH = Path(__file__).parent.parent / "src" / "v7p3r_uci.py"
V18_PATH = Path(__file__).parent.parent / "lichess" / "engines" / "V7P3R_v18.4_20260417" / "src" / "v7p3r_uci.py"

def detailed_search_analysis(engine_path: Path, position_fen: str, time_limit: float = 10.0):
    """Detailed search analysis with UCI info output"""
    process = subprocess.Popen(
        ["python", str(engine_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Initialize
    process.stdin.write("uci\n")
    process.stdin.flush()
    
    engine_name = None
    while True:
        line = process.stdout.readline().strip()
        if not line:
            break
        if line == "uciok":
            break
        if line.startswith("id name"):
            engine_name = line.split("id name ")[1]
    
    process.stdin.write("isready\n")
    process.stdin.flush()
    while True:
        line = process.stdout.readline().strip()
        if line == "readyok":
            break
    
    # Search with detailed info
    process.stdin.write(f"position fen {position_fen}\n")
    process.stdin.write(f"go movetime {int(time_limit * 1000)}\n")
    process.stdin.flush()
    
    start = time.time()
    depth_stats = []
    max_depth = 0
    total_nodes = 0
    
    while True:
        line = process.stdout.readline().strip()
        if line.startswith("info depth"):
            parts = line.split()
            try:
                depth_idx = parts.index("depth")
                depth = int(parts[depth_idx + 1])
                
                time_idx = parts.index("time")
                time_ms = int(parts[time_idx + 1])
                
                # Extract nodes if available
                nodes = 0
                if "nodes" in parts:
                    nodes_idx = parts.index("nodes")
                    nodes = int(parts[nodes_idx + 1])
                
                # Extract score if available
                score_cp = None
                if "score" in parts and "cp" in parts:
                    score_idx = parts.index("score")
                    if score_idx + 2 < len(parts) and parts[score_idx + 1] == "cp":
                        score_cp = int(parts[score_idx + 2])
                
                if depth > max_depth:
                    max_depth = depth
                    depth_stats.append({
                        "depth": depth,
                        "time_ms": time_ms,
                        "nodes": nodes,
                        "score_cp": score_cp
                    })
                    total_nodes = nodes
            except (ValueError, IndexError):
                pass
        elif line.startswith("bestmove"):
            break
    
    elapsed = time.time() - start
    
    process.stdin.write("quit\n")
    process.stdin.flush()
    process.wait(timeout=2)
    
    return {
        "engine": engine_name,
        "max_depth": max_depth,
        "elapsed": elapsed,
        "depth_stats": depth_stats,
        "total_nodes": total_nodes,
        "nps": total_nodes / elapsed if elapsed > 0 else 0
    }

def compare_efficiency(v19_result, v18_result):
    """Compare search efficiency between versions"""
    print("\n" + "="*80)
    print("SEARCH EFFICIENCY COMPARISON")
    print("="*80)
    
    # Overall metrics
    print(f"\n{'Metric':<30} {'v19.5.6':<20} {'v18.4':<20} {'Ratio':<15}")
    print("-" * 80)
    
    print(f"{'Max Depth':<30} {v19_result['max_depth']:<20} {v18_result['max_depth']:<20} {v19_result['max_depth']/v18_result['max_depth']:.2f}x")
    print(f"{'Total Time (s)':<30} {v19_result['elapsed']:.2f}s{'':<14} {v18_result['elapsed']:.2f}s{'':<14} {v19_result['elapsed']/v18_result['elapsed']:.2f}x")
    print(f"{'Total Nodes':<30} {v19_result['total_nodes']:,}{' '*(20-len(f'{v19_result['total_nodes']:,}'))} {v18_result['total_nodes']:,}{' '*(20-len(f'{v18_result['total_nodes']:,}'))} {v19_result['total_nodes']/v18_result['total_nodes']:.2f}x")
    print(f"{'NPS (nodes/sec)':<30} {v19_result['nps']:,.0f}{' '*(20-len(f'{v19_result['nps']:,.0f}'))} {v18_result['nps']:,.0f}{' '*(20-len(f'{v18_result['nps']:,.0f}'))} {v19_result['nps']/v18_result['nps']:.2f}x")
    
    # Depth progression analysis
    print("\n" + "="*80)
    print("DEPTH PROGRESSION ANALYSIS")
    print("="*80)
    
    print(f"\n{'Depth':<10} {'v19.5.6 Time':<20} {'v18.4 Time':<20} {'Delta':<20}")
    print("-" * 80)
    
    v19_depths = {d['depth']: d for d in v19_result['depth_stats']}
    v18_depths = {d['depth']: d for d in v18_result['depth_stats']}
    
    all_depths = sorted(set(v19_depths.keys()) | set(v18_depths.keys()))
    
    for depth in all_depths:
        v19_time = v19_depths[depth]['time_ms'] / 1000 if depth in v19_depths else None
        v18_time = v18_depths[depth]['time_ms'] / 1000 if depth in v18_depths else None
        
        v19_str = f"{v19_time:.2f}s" if v19_time else "N/A"
        v18_str = f"{v18_time:.2f}s" if v18_time else "N/A"
        
        if v19_time and v18_time:
            delta = f"+{v19_time - v18_time:.2f}s slower" if v19_time > v18_time else f"{v19_time - v18_time:.2f}s faster"
        else:
            delta = "N/A"
        
        print(f"{depth:<10} {v19_str:<20} {v18_str:<20} {delta:<20}")
    
    # Efficiency per depth
    print("\n" + "="*80)
    print("NODES PER DEPTH")
    print("="*80)
    
    print(f"\n{'Depth':<10} {'v19.5.6 Nodes':<20} {'v18.4 Nodes':<20} {'Ratio':<20}")
    print("-" * 80)
    
    for depth in all_depths:
        v19_nodes = v19_depths[depth]['nodes'] if depth in v19_depths else None
        v18_nodes = v18_depths[depth]['nodes'] if depth in v18_depths else None
        
        v19_str = f"{v19_nodes:,}" if v19_nodes else "N/A"
        v18_str = f"{v18_nodes:,}" if v18_nodes else "N/A"
        
        if v19_nodes and v18_nodes and v18_nodes > 0:
            ratio = f"{v19_nodes / v18_nodes:.2f}x"
        else:
            ratio = "N/A"
        
        print(f"{depth:<10} {v19_str:<20} {v18_str:<20} {ratio:<20}")

def identify_bottlenecks(v19_result, v18_result):
    """Identify specific bottlenecks"""
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)
    
    issues = []
    
    # NPS comparison
    nps_ratio = v19_result['nps'] / v18_result['nps']
    if nps_ratio < 0.8:
        issues.append({
            "severity": "HIGH",
            "area": "Search Speed (NPS)",
            "impact": f"v19.5.6 is {(1-nps_ratio)*100:.1f}% slower per node",
            "possible_causes": [
                "Inefficient move ordering overhead",
                "Excessive TT probing/storing",
                "Expensive evaluation calls",
                "Deep recursive function calls"
            ]
        })
    
    # Node count comparison (search efficiency)
    if v19_result['total_nodes'] > v18_result['total_nodes'] * 1.5:
        issues.append({
            "severity": "MEDIUM",
            "area": "Search Efficiency",
            "impact": f"v19.5.6 searches {v19_result['total_nodes']/v18_result['total_nodes']:.1f}x more nodes",
            "possible_causes": [
                "Poor move ordering (more cutoffs missed)",
                "TT replacement strategy issues",
                "Ineffective pruning",
                "Suboptimal aspiration window sizes"
            ]
        })
    
    # Time per depth
    v19_depths = {d['depth']: d for d in v19_result['depth_stats']}
    v18_depths = {d['depth']: d for d in v18_result['depth_stats']}
    
    # Check depth 3 time (common depth)
    if 3 in v19_depths and 3 in v18_depths:
        v19_d3_time = v19_depths[3]['time_ms'] / 1000
        v18_d3_time = v18_depths[3]['time_ms'] / 1000
        if v19_d3_time > v18_d3_time * 2:
            issues.append({
                "severity": "HIGH",
                "area": "Depth 3 Performance",
                "impact": f"v19.5.6 takes {v19_d3_time:.2f}s vs v18.4's {v18_d3_time:.2f}s ({v19_d3_time/v18_d3_time:.1f}x slower)",
                "possible_causes": [
                    "First few depths have high overhead",
                    "Initial move ordering poor",
                    "TT warming inefficient"
                ]
            })
    
    if issues:
        print("\nIssues identified:\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. [{issue['severity']}] {issue['area']}")
            print(f"   Impact: {issue['impact']}")
            print(f"   Possible causes:")
            for cause in issue['possible_causes']:
                print(f"     - {cause}")
            print()
    else:
        print("\n✓ No major bottlenecks identified")
    
    return issues

if __name__ == "__main__":
    # Test positions
    test_positions = [
        {
            "name": "Complex Middlegame",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
        },
        {
            "name": "Tactical Position",
            "fen": "r1bqk2r/ppp2ppp/2n1pn2/3p4/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQkq - 0 1"
        }
    ]
    
    print("="*80)
    print("V19.5.6 EFFICIENCY ANALYSIS")
    print("="*80)
    print("\nGoal: Identify optimization opportunities before deployment\n")
    
    for position in test_positions:
        print("\n" + "="*80)
        print(f"TESTING: {position['name']}")
        print("="*80)
        print(f"FEN: {position['fen']}\n")
        
        print("Running v19.5.6...")
        v19_result = detailed_search_analysis(V19_PATH, position['fen'], time_limit=10.0)
        
        print("Running v18.4...")
        v18_result = detailed_search_analysis(V18_PATH, position['fen'], time_limit=10.0)
        
        compare_efficiency(v19_result, v18_result)
        issues = identify_bottlenecks(v19_result, v18_result)
        
    print("\n" + "="*80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*80)
    print("""
Based on efficiency analysis, consider:

1. **NPS Optimization** (if v19 < 0.8x v18):
   - Profile CPU hotspots with cProfile
   - Review evaluation function overhead
   - Check TT probe/store efficiency
   
2. **Search Efficiency** (if v19 searches >1.5x nodes):
   - Tune move ordering (history heuristic, killer moves)
   - Adjust TT replacement strategy
   - Review pruning thresholds
   
3. **Time Management** (already optimal):
   - v19.5.6 matches v18.4 approach ✓
   
4. **Playing Strength** (50% vs v18.4):
   - Consider evaluation tuning
   - Test against broader opponent pool
   - Analyze lost games for patterns

Next steps:
- If efficiency issues found: Profile and optimize before deployment
- If efficiency good: Consider extended validation (10+ games)
- If strength concerns: Analyze game quality vs just results
""")
