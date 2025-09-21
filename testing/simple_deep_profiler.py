#!/usr/bin/env python3
"""
V7P3R Simple Deep Code Profiler - Built-in Tools Only
Advanced profiling using only Python standard library tools
"""

import sys
import time
import chess
import cProfile
import pstats
import io
import tracemalloc
from typing import Dict, List, Tuple, Any
from functools import wraps
from collections import defaultdict
import json
import threading

sys.path.append('src')
from v7p3r import V7P3REngine

class SimpleProfiler:
    """Simple but effective profiler using built-in Python tools"""
    
    def __init__(self):
        self.engine = V7P3REngine()
        
    def profile_search_execution(self, test_position: str, depth: int = 5) -> Dict:
        """Profile a search execution with detailed function analysis"""
        
        print(f"🔍 Profiling search execution (depth {depth})...")
        print(f"Position: {test_position}")
        
        board = chess.Board(test_position)
        
        # Create profiler
        profiler = cProfile.Profile()
        
        # Start memory tracking
        tracemalloc.start()
        memory_before = tracemalloc.get_traced_memory()[0]
        
        # Profile the search
        profiler.enable()
        start_time = time.perf_counter()
        best_move, score = self.engine.search(board, depth, time_limit=15.0)
        search_time = time.perf_counter() - start_time
        profiler.disable()
        
        # Get memory usage
        memory_after = tracemalloc.get_traced_memory()[0]
        memory_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        # Analyze profiling results
        stats = pstats.Stats(profiler)
        
        # Create a StringIO to capture the stats output
        stream = io.StringIO()
        stats.print_stats()
        stats_output = stream.getvalue()
        
        # Get function statistics by parsing the output
        function_stats = []
        lines = stats_output.split('\n')
        
        # Find the start of function data (after headers)
        data_start = 0
        for i, line in enumerate(lines):
            if 'ncalls' in line and 'tottime' in line:
                data_start = i + 1
                break
        
        # Parse function data
        for line in lines[data_start:data_start + 50]:  # Parse up to 50 functions
            if line.strip() and not line.startswith('---'):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        ncalls = parts[0]
                        tottime = float(parts[1])
                        cumtime = float(parts[3])
                        filename_func = ' '.join(parts[5:])
                        
                        # Extract just the function name and file
                        if '(' in filename_func:
                            func_info = filename_func.split('(')[1].split(')')[0]
                        else:
                            func_info = filename_func
                        
                        # Filter for our engine functions or important functions
                        if ('v7p3r' in func_info.lower() or 
                            'chess' in func_info.lower() or
                            tottime > 0.001):  # Functions taking > 1ms
                            
                            function_stats.append({
                                'function': func_info,
                                'call_count': ncalls,
                                'total_time_ms': tottime * 1000,
                                'cumulative_time_ms': cumtime * 1000,
                                'avg_time_us': (tottime / max(int(ncalls.split('/')[0]) if '/' in ncalls else int(ncalls), 1)) * 1000000,
                                'percent_of_total': (cumtime / search_time) * 100
                            })
                    except (ValueError, IndexError):
                        continue
        
        # Sort by cumulative time
        function_stats.sort(key=lambda x: x['cumulative_time_ms'], reverse=True)
        
        return {
            'search_time': search_time,
            'nodes_searched': self.engine.nodes_searched,
            'nps': self.engine.nodes_searched / max(search_time, 0.001),
            'best_move': str(best_move),
            'score': score,
            'memory_used_kb': (memory_after - memory_before) / 1024,
            'memory_peak_kb': memory_peak / 1024,
            'function_stats': function_stats[:30]  # Top 30 functions
        }
    
    def analyze_function_hotspots(self, stats: Dict) -> List[str]:
        """Analyze function statistics to identify optimization targets"""
        
        insights = []
        
        # Performance overview
        nps = stats['nps']
        search_time = stats['search_time']
        
        insights.append(f"📊 Performance Overview:")
        insights.append(f"   Search Time: {search_time:.3f}s")
        insights.append(f"   Nodes: {stats['nodes_searched']:,}")
        insights.append(f"   NPS: {nps:,.0f}")
        insights.append(f"   Memory Used: {stats['memory_used_kb']:.1f} KB")
        insights.append(f"   Memory Peak: {stats['memory_peak_kb']:.1f} KB")
        
        # Performance assessment
        if nps > 200000:
            insights.append("   ✅ EXCELLENT performance")
        elif nps > 100000:
            insights.append("   ✅ GOOD performance")
        elif nps > 50000:
            insights.append("   ⚠️  ACCEPTABLE performance")
        else:
            insights.append("   🚨 POOR performance - optimization needed")
        
        insights.append("")
        
        # Function analysis
        insights.append("🔥 Function Hotspots (Top 10):")
        
        for i, func in enumerate(stats['function_stats'][:10]):
            func_name = func['function'].split(':')[-1]
            
            insights.append(f"   {i+1:2d}. {func_name}")
            insights.append(f"       Time: {func['cumulative_time_ms']:.2f}ms ({func['percent_of_total']:.1f}%)")
            insights.append(f"       Calls: {func['call_count']:,}")
            insights.append(f"       Avg: {func['avg_time_us']:.1f}μs per call")
            
            # Add optimization hints
            if func['percent_of_total'] > 20:
                insights.append("       🚨 CRITICAL: Major optimization target")
            elif func['percent_of_total'] > 10:
                insights.append("       ⚠️  HIGH: Optimization recommended")
            elif func['call_count'] > 10000:
                insights.append("       📈 HIGH FREQUENCY: Called very often")
            elif func['avg_time_us'] > 1000:
                insights.append("       🐌 SLOW: High per-call cost")
            
            insights.append("")
        
        return insights
    
    def generate_optimization_recommendations(self, stats: Dict) -> List[str]:
        """Generate specific optimization recommendations"""
        
        recommendations = []
        
        # Analyze the top functions for specific recommendations
        top_functions = stats['function_stats'][:10]
        
        recommendations.append("💡 Optimization Recommendations:")
        recommendations.append("")
        
        # Check for common optimization targets
        evaluation_time = sum(f['cumulative_time_ms'] for f in top_functions 
                             if 'evaluat' in f['function'].lower())
        tactical_time = sum(f['cumulative_time_ms'] for f in top_functions 
                           if 'tactical' in f['function'].lower())
        search_time = sum(f['cumulative_time_ms'] for f in top_functions 
                         if 'search' in f['function'].lower())
        
        total_search_time_ms = stats['search_time'] * 1000
        
        if evaluation_time > total_search_time_ms * 0.3:
            recommendations.append("🔧 EVALUATION OPTIMIZATION:")
            recommendations.append("   - Cache evaluation results")
            recommendations.append("   - Simplify position evaluation")
            recommendations.append("   - Use incremental evaluation")
            recommendations.append("")
        
        if tactical_time > total_search_time_ms * 0.15:
            recommendations.append("🔧 TACTICAL PATTERN OPTIMIZATION:")
            recommendations.append("   - Reduce tactical pattern complexity")
            recommendations.append("   - Use time budgets for tactical analysis")
            recommendations.append("   - Cache tactical evaluations")
            recommendations.append("")
        
        if search_time > total_search_time_ms * 0.4:
            recommendations.append("🔧 SEARCH ALGORITHM OPTIMIZATION:")
            recommendations.append("   - Improve move ordering")
            recommendations.append("   - Better transposition table usage")
            recommendations.append("   - Optimize alpha-beta pruning")
            recommendations.append("")
        
        # Check for high-frequency functions
        high_freq_functions = [f for f in top_functions if f['call_count'] > 5000]
        if high_freq_functions:
            recommendations.append("🔧 HIGH-FREQUENCY FUNCTION OPTIMIZATION:")
            for func in high_freq_functions[:3]:
                func_name = func['function'].split(':')[-1]
                recommendations.append(f"   - Optimize {func_name} ({func['call_count']:,} calls)")
            recommendations.append("")
        
        # Memory optimization
        if stats['memory_peak_kb'] > 10000:  # > 10MB
            recommendations.append("🔧 MEMORY OPTIMIZATION:")
            recommendations.append("   - Reduce memory allocations")
            recommendations.append("   - Use object pooling")
            recommendations.append("   - Optimize data structures")
            recommendations.append("")
        
        # General recommendations
        recommendations.extend([
            "🔧 GENERAL OPTIMIZATIONS:",
            "   1. Profile before and after each optimization",
            "   2. Focus on functions using >5% of total time",
            "   3. Optimize hot paths first",
            "   4. Consider algorithmic improvements",
            "   5. Test with different positions and depths"
        ])
        
        return recommendations
    
    def run_comprehensive_profiling(self) -> Dict:
        """Run comprehensive profiling analysis"""
        
        print("V7P3R Simple Deep Code Profiler")
        print("=" * 50)
        
        # Test positions
        test_positions = [
            ("Complex Tactical", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
            ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("Endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
        ]
        
        all_results = {}
        
        for name, fen in test_positions:
            print(f"\n{'='*50}")
            print(f"Testing: {name}")
            print(f"{'='*50}")
            
            # Profile this position
            stats = self.profile_search_execution(fen, depth=5)
            
            # Analyze hotspots
            insights = self.analyze_function_hotspots(stats)
            
            # Generate recommendations
            recommendations = self.generate_optimization_recommendations(stats)
            
            # Display results
            for insight in insights:
                print(insight)
            
            print("\n" + "-" * 50)
            for rec in recommendations:
                print(rec)
            
            all_results[name] = {
                'stats': stats,
                'insights': insights,
                'recommendations': recommendations
            }
        
        # Save comprehensive results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"comprehensive_profiling_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n📁 Complete analysis saved to: {results_file}")
        
        return all_results

def main():
    """Run the profiling analysis"""
    profiler = SimpleProfiler()
    results = profiler.run_comprehensive_profiling()
    
    print(f"\n🎯 Analysis Complete!")
    print("Review the function hotspots and optimization recommendations above.")
    print("Focus on optimizing the functions that use the most cumulative time.")

if __name__ == "__main__":
    main()