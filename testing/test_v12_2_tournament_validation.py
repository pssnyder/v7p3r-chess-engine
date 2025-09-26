#!/usr/bin/env python3
"""
V7P3R v12.2 Tournament Validation Test
=====================================
Comprehensive testing to validate v12.2 performance against baselines.
Compares v12.2 vs v12.0 vs v10.8 across multiple scenarios.

Target: Prove v12.2 is tournament-ready with improved performance
"""

import sys
import os
import time
import subprocess
import threading
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

# Import chess for position validation
import chess

def run_engine_analysis(engine_path, position_fen, time_limit=5.0):
    """
    Run engine analysis via UCI and capture results
    """
    try:
        # Start engine process
        process = subprocess.Popen(
            engine_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(engine_path)
        )
        
        # UCI communication
        commands = [
            "uci",
            "ucinewgame",
            f"position fen {position_fen}",
            f"go movetime {int(time_limit * 1000)}",
        ]
        
        input_text = "\n".join(commands) + "\nquit\n"
        
        # Run with timeout
        stdout, stderr = process.communicate(input=input_text, timeout=time_limit + 2.0)
        
        # Parse UCI output
        lines = stdout.split('\n')
        best_move = None
        depth = 0
        nodes = 0
        time_ms = 0
        nps = 0
        score_cp = 0
        
        for line in lines:
            if line.startswith('info'):
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'depth' and i+1 < len(parts):
                        try:
                            depth = int(parts[i+1])
                        except: pass
                    elif part == 'nodes' and i+1 < len(parts):
                        try:
                            nodes = int(parts[i+1])
                        except: pass
                    elif part == 'time' and i+1 < len(parts):
                        try:
                            time_ms = int(parts[i+1])
                        except: pass
                    elif part == 'nps' and i+1 < len(parts):
                        try:
                            nps = int(parts[i+1])
                        except: pass
                    elif part == 'score' and i+1 < len(parts) and parts[i+1] == 'cp' and i+2 < len(parts):
                        try:
                            score_cp = int(parts[i+2])
                        except: pass
            elif line.startswith('bestmove'):
                parts = line.split()
                if len(parts) > 1:
                    best_move = parts[1]
        
        return {
            'success': True,
            'best_move': best_move,
            'depth': depth,
            'nodes': nodes,
            'time_ms': time_ms,
            'nps': nps,
            'score_cp': score_cp,
            'time_used': time_ms / 1000.0 if time_ms > 0 else time_limit
        }
        
    except subprocess.TimeoutExpired:
        process.kill()
        return {
            'success': False,
            'error': 'Timeout',
            'time_used': time_limit
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'time_used': time_limit
        }

def test_position_set(engines, positions, time_limit=5.0):
    """
    Test multiple engines on a set of positions
    """
    results = {}
    
    for engine_name, engine_path in engines.items():
        print(f"\n🔍 Testing {engine_name}...")
        results[engine_name] = []
        
        for i, (name, fen) in enumerate(positions.items(), 1):
            print(f"  Position {i}/{len(positions)}: {name}")
            result = run_engine_analysis(engine_path, fen, time_limit)
            result['position'] = name
            result['fen'] = fen
            results[engine_name].append(result)
            
            if result['success']:
                print(f"    ✅ Depth {result['depth']}, {result['nodes']} nodes, {result['nps']} NPS")
            else:
                print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
    
    return results

def print_comparison_table(results, time_limit):
    """
    Print a detailed comparison table of results
    """
    print(f"\n{'='*80}")
    print(f"V7P3R TOURNAMENT VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Time Limit: {time_limit}s per position")
    
    # Get all engines and positions
    engines = list(results.keys())
    if not engines:
        return
        
    positions = [r['position'] for r in results[engines[0]]]
    
    # Print header
    print(f"\n{'Position':<20} | {'Metric':<12} |", end="")
    for engine in engines:
        print(f" {engine:<15} |", end="")
    print()
    print("-" * 80)
    
    # Print results for each position
    for i, position in enumerate(positions):
        # Get results for this position across all engines
        pos_results = {}
        for engine in engines:
            pos_results[engine] = results[engine][i]
        
        # Print multiple rows per position
        metrics = [
            ('Depth', 'depth'),
            ('Nodes', 'nodes'), 
            ('NPS', 'nps'),
            ('Time (s)', 'time_used'),
            ('Move', 'best_move')
        ]
        
        for j, (metric_name, metric_key) in enumerate(metrics):
            if j == 0:
                print(f"{position:<20} | {metric_name:<12} |", end="")
            else:
                print(f"{'':<20} | {metric_name:<12} |", end="")
                
            for engine in engines:
                result = pos_results[engine]
                if result['success']:
                    value = result.get(metric_key, 'N/A')
                    if metric_key == 'nodes':
                        value = f"{value:,}" if isinstance(value, int) else value
                    elif metric_key == 'nps':
                        value = f"{value:,}" if isinstance(value, int) else value
                    elif metric_key == 'time_used':
                        value = f"{value:.1f}" if isinstance(value, (int, float)) else value
                else:
                    value = "FAIL"
                print(f" {str(value):<15} |", end="")
            print()
        print("-" * 80)

def calculate_performance_summary(results):
    """
    Calculate overall performance metrics
    """
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    for engine_name, engine_results in results.items():
        successful = [r for r in engine_results if r['success']]
        total = len(engine_results)
        
        if successful:
            avg_depth = sum(r['depth'] for r in successful) / len(successful)
            avg_nodes = sum(r['nodes'] for r in successful) / len(successful)
            avg_nps = sum(r['nps'] for r in successful) / len(successful)
            avg_time = sum(r['time_used'] for r in successful) / len(successful)
            success_rate = len(successful) / total * 100
            
            print(f"\n🔧 {engine_name}:")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Average Depth: {avg_depth:.1f}")
            print(f"  Average Nodes: {avg_nodes:,.0f}")
            print(f"  Average NPS: {avg_nps:,.0f}")
            print(f"  Average Time: {avg_time:.1f}s")
        else:
            print(f"\n❌ {engine_name}: All tests failed")

def main():
    print("🚀 V7P3R v12.2 Tournament Validation Test")
    print("🎯 Goal: Validate v12.2 performance improvements")
    print("📊 Testing: v12.2 vs v12.0 vs v10.8")
    
    # Define engine paths
    engine_dir = r"s:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester\engines\V7P3R"
    engines = {
        "V12.2": os.path.join(engine_dir, "V7P3R_v12.2.exe"),
        "V12.0": os.path.join(engine_dir, "V7P3R_v12.0.exe"), 
        "V10.8": os.path.join(engine_dir, "V7P3R_v10.8.exe")
    }
    
    # Verify engines exist
    print(f"\n🔍 Verifying engines...")
    for name, path in engines.items():
        if os.path.exists(path):
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: NOT FOUND - {path}")
            return 1
    
    # Test positions covering various game phases
    test_positions = {
        "Opening": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Middlegame": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
        "Tactical": "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "Endgame": "8/8/8/3k4/3P4/3K4/8/8 w - - 0 50",
        "Complex": "r2qkb1r/pb2nppp/1p2pn2/3p4/2PP4/1P3NP1/PB2PPBP/RN1QK2R w KQkq d6 0 9"
    }
    
    # Run 5-second tests
    print(f"\n⏱️  Running 5-second analysis tests...")
    results_5s = test_position_set(engines, test_positions, 5.0)
    print_comparison_table(results_5s, 5.0)
    calculate_performance_summary(results_5s)
    
    # Run quick 2-second blitz tests
    print(f"\n⚡ Running 2-second blitz tests...")
    blitz_positions = {
        "Blitz Opening": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "Blitz Tactics": "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    }
    results_2s = test_position_set(engines, blitz_positions, 2.0)
    print_comparison_table(results_2s, 2.0)
    calculate_performance_summary(results_2s)
    
    # Final tournament readiness assessment
    print(f"\n{'='*60}")
    print("🏆 TOURNAMENT READINESS ASSESSMENT")
    print(f"{'='*60}")
    
    v12_2_results = results_5s.get("V12.2", [])
    v12_0_results = results_5s.get("V12.0", [])
    v10_8_results = results_5s.get("V10.8", [])
    
    if v12_2_results:
        v12_2_success = [r for r in v12_2_results if r['success']]
        if v12_2_success:
            avg_nps_12_2 = sum(r['nps'] for r in v12_2_success) / len(v12_2_success)
            avg_depth_12_2 = sum(r['depth'] for r in v12_2_success) / len(v12_2_success)
            
            print(f"\n🎯 V12.2 Performance:")
            print(f"  Average NPS: {avg_nps_12_2:,.0f}")
            print(f"  Average Depth (5s): {avg_depth_12_2:.1f}")
            
            # Compare with v12.0
            if v12_0_results:
                v12_0_success = [r for r in v12_0_results if r['success']]
                if v12_0_success:
                    avg_nps_12_0 = sum(r['nps'] for r in v12_0_success) / len(v12_0_success)
                    improvement = (avg_nps_12_2 / avg_nps_12_0) if avg_nps_12_0 > 0 else 1
                    print(f"  Improvement vs V12.0: {improvement:.1f}x faster")
            
            # Tournament readiness criteria
            print(f"\n📋 Tournament Readiness Checklist:")
            print(f"  ✅ NPS > 3,000: {'✅' if avg_nps_12_2 > 3000 else '❌'} ({avg_nps_12_2:,.0f})")
            print(f"  ✅ Depth 5+ in 5s: {'✅' if avg_depth_12_2 >= 5 else '❌'} ({avg_depth_12_2:.1f})")
            print(f"  ✅ No timeouts: {'✅' if len(v12_2_success) == len(v12_2_results) else '❌'}")
            
            if avg_nps_12_2 > 3000 and avg_depth_12_2 >= 5:
                print(f"\n🏆 RESULT: V12.2 is TOURNAMENT READY!")
                print(f"   Ready for competitive play against v10.8")
            else:
                print(f"\n⚠️  RESULT: V12.2 needs more optimization")
                print(f"   Consider further performance improvements")

if __name__ == "__main__":
    main()