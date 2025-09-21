#!/usr/bin/env python3
"""
V7P3R Quick Performance Profiler
Simple and reliable profiling to identify bottlenecks
"""

import sys
import time
import chess
import cProfile
import pstats
import io
from typing import Dict, List
import json

sys.path.append('src')
from v7p3r import V7P3REngine

def profile_engine_search(position_fen: str, depth: int = 5) -> Dict:
    """Profile a search and return function timing data"""
    
    print(f"🔍 Profiling position: {position_fen}")
    print(f"Search depth: {depth}")
    
    engine = V7P3REngine()
    board = chess.Board(position_fen)
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Profile the search
    profiler.enable()
    start_time = time.perf_counter()
    
    try:
        best_move = engine.search(board, time_limit=10.0, depth=depth)
        search_time = time.perf_counter() - start_time
        
        # Get the last evaluation score from the engine's internal state
        score = 0  # We'll get this from the last evaluation
        success = True if best_move else False
        
    except Exception as e:
        search_time = time.perf_counter() - start_time
        best_move = None
        score = 0
        success = False
        print(f"Search failed: {e}")
    
    profiler.disable()
    
    # Get stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(25)
    
    output = stream.getvalue()
    
    return {
        'success': success,
        'search_time': search_time,
        'nodes_searched': engine.nodes_searched,
        'nps': engine.nodes_searched / max(search_time, 0.001),
        'best_move': str(best_move) if best_move else "None",
        'score': score,
        'profile_output': output
    }

def analyze_profile_output(output: str, search_time: float) -> List[Dict]:
    """Extract function timing data from profile output"""
    
    lines = output.split('\n')
    functions = []
    
    # Find data section
    data_start = 0
    for i, line in enumerate(lines):
        if 'ncalls' in line and 'tottime' in line:
            data_start = i + 1
            break
    
    # Parse function data
    for line in lines[data_start:data_start + 20]:
        if line.strip() and not line.startswith('---'):
            parts = line.split()
            if len(parts) >= 6:
                try:
                    ncalls = parts[0]
                    tottime = float(parts[1])
                    cumtime = float(parts[3])
                    filename_func = ' '.join(parts[5:])
                    
                    # Clean function name
                    if '(' in filename_func:
                        func_name = filename_func.split('(')[1].split(')')[0]
                    else:
                        func_name = filename_func
                    
                    # Extract call count
                    call_count = int(ncalls.split('/')[0]) if '/' in ncalls else int(ncalls)
                    
                    if tottime > 0.001 or 'v7p3r' in func_name.lower():
                        functions.append({
                            'function': func_name,
                            'calls': call_count,
                            'total_time_ms': tottime * 1000,
                            'cumulative_time_ms': cumtime * 1000,
                            'percent': (cumtime / search_time) * 100
                        })
                        
                except (ValueError, IndexError):
                    continue
    
    return sorted(functions, key=lambda x: x['cumulative_time_ms'], reverse=True)

def run_quick_profiling_analysis():
    """Run quick profiling on key positions"""
    
    print("V7P3R Quick Performance Profiler")
    print("=" * 50)
    
    positions = [
        ("Starting", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Tactical", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
        ("Endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    ]
    
    all_results = {}
    
    for name, fen in positions:
        print(f"\n{'-' * 50}")
        print(f"Testing: {name}")
        print(f"{'-' * 50}")
        
        # Profile this position
        result = profile_engine_search(fen, depth=5)
        
        if result['success']:
            print(f"✅ Search completed successfully")
            print(f"Time: {result['search_time']:.3f}s")
            print(f"Nodes: {result['nodes_searched']:,}")
            print(f"NPS: {result['nps']:,.0f}")
            print(f"Best move: {result['best_move']}")
            
            # Analyze functions
            functions = analyze_profile_output(result['profile_output'], result['search_time'])
            
            print(f"\n🔥 Top 10 Functions by Time:")
            for i, func in enumerate(functions[:10]):
                print(f"  {i+1:2d}. {func['function'][:40]:40} {func['cumulative_time_ms']:6.1f}ms ({func['percent']:4.1f}%)")
            
            # Performance assessment
            nps = result['nps']
            if nps > 200000:
                print(f"\n✅ EXCELLENT performance ({nps:,.0f} NPS)")
            elif nps > 100000:
                print(f"\n✅ GOOD performance ({nps:,.0f} NPS)")
            elif nps > 50000:
                print(f"\n⚠️  MODERATE performance ({nps:,.0f} NPS)")
            else:
                print(f"\n🚨 POOR performance ({nps:,.0f} NPS)")
            
            all_results[name] = {
                'result': result,
                'functions': functions
            }
        else:
            print(f"❌ Search failed")
    
    # Overall analysis
    if all_results:
        print(f"\n{'=' * 50}")
        print("OVERALL ANALYSIS")
        print(f"{'=' * 50}")
        
        avg_nps = sum(r['result']['nps'] for r in all_results.values()) / len(all_results)
        print(f"Average NPS: {avg_nps:,.0f}")
        
        # Find most common slow functions
        all_functions = []
        for r in all_results.values():
            all_functions.extend(r['functions'])
        
        # Group by function name and sum times
        func_totals = {}
        for func in all_functions:
            name = func['function']
            if name not in func_totals:
                func_totals[name] = {'total_time': 0, 'appearances': 0}
            func_totals[name]['total_time'] += func['cumulative_time_ms']
            func_totals[name]['appearances'] += 1
        
        # Sort by total time across all tests
        sorted_funcs = sorted(func_totals.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        print(f"\n🎯 Most Time-Consuming Functions Across All Tests:")
        for i, (func_name, data) in enumerate(sorted_funcs[:8]):
            avg_time = data['total_time'] / data['appearances']
            print(f"  {i+1}. {func_name[:45]:45} {avg_time:6.1f}ms avg")
        
        print(f"\n💡 Optimization Recommendations:")
        
        if avg_nps < 100000:
            print("  🚨 CRITICAL: Performance needs major optimization")
            print("     - Focus on the top 3 slowest functions")
            print("     - Consider algorithmic improvements")
            print("     - Profile individual evaluation components")
        else:
            print("  ✅ Performance is good, focus on fine-tuning")
            print("     - Optimize functions using >10% of search time")
            print("     - Improve time management for different time controls")
        
        # Check for specific issues
        tactical_heavy = any('tactical' in func.lower() for func in func_totals.keys())
        eval_heavy = any('evaluat' in func.lower() for func in func_totals.keys())
        
        if tactical_heavy:
            print("     - Tactical pattern detection may be expensive")
        if eval_heavy:
            print("     - Position evaluation may need optimization")
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"quick_profiling_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {results_file}")

if __name__ == "__main__":
    run_quick_profiling_analysis()