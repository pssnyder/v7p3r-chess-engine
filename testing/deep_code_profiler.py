#!/usr/bin/env python3
"""
V7P3R Deep Code Profiler - Microsecond-Level Function Analysis
Advanced profiling system to identify performance bottlenecks at the function level

This tool provides:
1. Function call timing down to microseconds
2. Call frequency analysis
3. Hot path identification
4. Memory usage tracking
5. Cumulative time analysis
6. Call graph visualization data
"""

import sys
import time
import chess
import cProfile
import pstats
import io
import line_profiler
import memory_profiler
import tracemalloc
from typing import Dict, List, Tuple, Any
from functools import wraps
from collections import defaultdict, Counter
import json
import threading

sys.path.append('src')
from v7p3r import V7P3REngine

class FunctionTimer:
    """High-precision function timer using performance counters"""
    
    def __init__(self):
        self.function_stats = defaultdict(lambda: {
            'total_time': 0.0,
            'call_count': 0,
            'max_time': 0.0,
            'min_time': float('inf'),
            'avg_time': 0.0
        })
        self.call_stack = []
        self.active_calls = {}
        
    def start_call(self, func_name: str) -> int:
        """Start timing a function call"""
        call_id = id(threading.current_thread()) + len(self.call_stack)
        start_time = time.perf_counter()
        
        self.call_stack.append({
            'func_name': func_name,
            'start_time': start_time,
            'call_id': call_id
        })
        
        self.active_calls[call_id] = start_time
        return call_id
        
    def end_call(self, func_name: str, call_id: int):
        """End timing a function call"""
        end_time = time.perf_counter()
        
        if call_id in self.active_calls:
            start_time = self.active_calls.pop(call_id)
            duration = end_time - start_time
            
            stats = self.function_stats[func_name]
            stats['total_time'] += duration
            stats['call_count'] += 1
            stats['max_time'] = max(stats['max_time'], duration)
            stats['min_time'] = min(stats['min_time'], duration)
            stats['avg_time'] = stats['total_time'] / stats['call_count']
            
            # Remove from call stack
            if self.call_stack and self.call_stack[-1]['call_id'] == call_id:
                self.call_stack.pop()
    
    def get_stats(self) -> Dict:
        """Get formatted statistics"""
        return dict(self.function_stats)

# Global timer instance
function_timer = FunctionTimer()

def profile_function(func):
    """Decorator to profile individual functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__qualname__}"
        call_id = function_timer.start_call(func_name)
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            function_timer.end_call(func_name, call_id)
    
    return wrapper

class V7P3RDeepProfiler:
    """Comprehensive profiling system for V7P3R engine"""
    
    def __init__(self):
        self.engine = V7P3REngine()
        self.profiling_results = {}
        
    def patch_engine_methods(self):
        """Dynamically patch engine methods with profiling decorators"""
        
        # List of methods to profile
        methods_to_profile = [
            'search',
            '_search_with_alpha_beta',
            '_evaluate_position',
            '_detect_bitboard_tactics',
            '_order_moves',
            '_calculate_move_score',
            '_apply_killer_move_bonus',
            '_apply_history_bonus',
            '_apply_nudge_bonus',
            '_evaluate_draw_penalty',
            '_evaluate_enhanced_endgame_king',
            '_evaluate_move_classification',
            '_evaluate_king_restriction',
            '_apply_phase_aware_weighting'
        ]
        
        # Patch bitboard evaluator methods
        bitboard_methods = [
            'evaluate_position',
            'evaluate_material',
            'evaluate_piece_square_tables',
            'evaluate_pawn_structure',
            'evaluate_king_safety',
            'evaluate_piece_mobility'
        ]
        
        # Patch tactical pattern detector methods
        tactical_methods = [
            'evaluate_tactical_patterns'
        ]
        
        print("🔧 Patching engine methods for profiling...")
        
        # Patch main engine methods
        for method_name in methods_to_profile:
            if hasattr(self.engine, method_name):
                original_method = getattr(self.engine, method_name)
                patched_method = profile_function(original_method)
                setattr(self.engine, method_name, patched_method)
                
        # Patch evaluator methods
        if hasattr(self.engine, 'bitboard_evaluator'):
            for method_name in bitboard_methods:
                if hasattr(self.engine.bitboard_evaluator, method_name):
                    original_method = getattr(self.engine.bitboard_evaluator, method_name)
                    patched_method = profile_function(original_method)
                    setattr(self.engine.bitboard_evaluator, method_name, patched_method)
        
        # Patch tactical detector methods
        if hasattr(self.engine, 'tactical_pattern_detector'):
            for method_name in tactical_methods:
                if hasattr(self.engine.tactical_pattern_detector, method_name):
                    original_method = getattr(self.engine.tactical_pattern_detector, method_name)
                    patched_method = profile_function(original_method)
                    setattr(self.engine.tactical_pattern_detector, method_name, patched_method)
        
        print("✅ Patching complete!")
    
    def run_cprofile_analysis(self, test_position: str, depth: int = 5) -> Dict:
        """Run cProfile analysis for detailed function statistics"""
        
        print(f"📊 Running cProfile analysis (depth {depth})...")
        
        board = chess.Board(test_position)
        profiler = cProfile.Profile()
        
        # Profile the search
        profiler.enable()
        start_time = time.perf_counter()
        best_move, score = self.engine.search(board, depth, time_limit=10.0)
        search_time = time.perf_counter() - start_time
        profiler.disable()
        
        # Analyze results
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(50)  # Top 50 functions
        
        # Parse the most expensive functions
        stats_output = stream.getvalue()
        lines = stats_output.split('\n')
        
        # Extract function data
        function_data = []
        for line in lines[5:55]:  # Skip header, get top 50
            if line.strip() and 'function calls' not in line and '---' not in line:
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        ncalls = parts[0]
                        tottime = float(parts[1])
                        cumtime = float(parts[3])
                        filename_func = ' '.join(parts[5:])
                        
                        function_data.append({
                            'ncalls': ncalls,
                            'tottime': tottime,
                            'cumtime': cumtime,
                            'filename_func': filename_func
                        })
                    except (ValueError, IndexError):
                        continue
        
        return {
            'search_time': search_time,
            'nodes_searched': self.engine.nodes_searched,
            'nps': self.engine.nodes_searched / max(search_time, 0.001),
            'function_data': function_data,
            'raw_output': stats_output
        }
    
    def run_custom_timer_analysis(self, test_position: str, depth: int = 5) -> Dict:
        """Run analysis using our custom function timer"""
        
        print(f"⏱️  Running custom timer analysis (depth {depth})...")
        
        # Reset timer
        global function_timer
        function_timer = FunctionTimer()
        
        # Patch methods
        self.patch_engine_methods()
        
        board = chess.Board(test_position)
        
        # Run search with custom timing
        start_time = time.perf_counter()
        best_move, score = self.engine.search(board, depth, time_limit=10.0)
        search_time = time.perf_counter() - start_time
        
        # Get timing statistics
        stats = function_timer.get_stats()
        
        # Convert to sorted list by total time
        sorted_stats = []
        for func_name, data in stats.items():
            sorted_stats.append({
                'function': func_name,
                'total_time_ms': data['total_time'] * 1000,
                'call_count': data['call_count'],
                'avg_time_us': data['avg_time'] * 1000000,
                'max_time_us': data['max_time'] * 1000000,
                'min_time_us': data['min_time'] * 1000000 if data['min_time'] != float('inf') else 0,
                'percent_of_total': (data['total_time'] / search_time) * 100
            })
        
        sorted_stats.sort(key=lambda x: x['total_time_ms'], reverse=True)
        
        return {
            'search_time': search_time,
            'nodes_searched': self.engine.nodes_searched,
            'nps': self.engine.nodes_searched / max(search_time, 0.001),
            'function_stats': sorted_stats
        }
    
    def run_memory_analysis(self, test_position: str, depth: int = 5) -> Dict:
        """Run memory usage analysis"""
        
        print(f"🧠 Running memory analysis (depth {depth})...")
        
        board = chess.Board(test_position)
        
        # Start memory tracing
        tracemalloc.start()
        
        # Take snapshot before search
        snapshot_before = tracemalloc.take_snapshot()
        
        # Run search
        start_time = time.perf_counter()
        best_move, score = self.engine.search(board, depth, time_limit=10.0)
        search_time = time.perf_counter() - start_time
        
        # Take snapshot after search
        snapshot_after = tracemalloc.take_snapshot()
        
        # Analyze memory differences
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        
        memory_data = []
        for stat in top_stats[:20]:  # Top 20 memory allocators
            memory_data.append({
                'traceback': str(stat.traceback),
                'size_diff_kb': stat.size_diff / 1024,
                'count_diff': stat.count_diff
            })
        
        # Stop tracing
        tracemalloc.stop()
        
        return {
            'search_time': search_time,
            'memory_allocations': memory_data,
            'peak_memory_kb': tracemalloc.get_traced_memory()[1] / 1024 if tracemalloc.is_tracing() else 0
        }
    
    def analyze_hot_paths(self, cprofile_data: Dict) -> List[Dict]:
        """Analyze the hottest execution paths"""
        
        print("🔥 Analyzing hot execution paths...")
        
        hot_paths = []
        
        # Identify functions that take > 1% of total time
        for func_data in cprofile_data['function_data']:
            if func_data['cumtime'] > cprofile_data['search_time'] * 0.01:  # > 1% of total time
                hot_paths.append({
                    'function': func_data['filename_func'],
                    'cumulative_time_ms': func_data['cumtime'] * 1000,
                    'total_time_ms': func_data['tottime'] * 1000,
                    'calls': func_data['ncalls'],
                    'percent_of_search': (func_data['cumtime'] / cprofile_data['search_time']) * 100
                })
        
        # Sort by cumulative time
        hot_paths.sort(key=lambda x: x['cumulative_time_ms'], reverse=True)
        
        return hot_paths
    
    def generate_optimization_insights(self, cprofile_data: Dict, custom_timer_data: Dict, hot_paths: List[Dict]) -> List[str]:
        """Generate specific optimization recommendations based on profiling data"""
        
        insights = []
        
        # Overall performance assessment
        nps = cprofile_data['nps']
        if nps < 50000:
            insights.append(f"🚨 CRITICAL: NPS is {nps:,.0f} - major optimizations needed")
        elif nps < 100000:
            insights.append(f"⚠️  WARNING: NPS is {nps:,.0f} - optimizations recommended")
        else:
            insights.append(f"✅ GOOD: NPS is {nps:,.0f} - fine-tuning opportunities")
        
        # Analyze hot paths
        insights.append("\n🔥 Hot Path Analysis:")
        for i, path in enumerate(hot_paths[:5]):  # Top 5 hot paths
            insights.append(f"  {i+1}. {path['function']}: {path['percent_of_search']:.1f}% of search time")
            
            if path['percent_of_search'] > 20:
                insights.append(f"     🚨 CRITICAL: This function uses too much time - priority optimization target")
            elif path['percent_of_search'] > 10:
                insights.append(f"     ⚠️  HIGH: Consider optimizing this function")
        
        # Function-specific insights from custom timer
        insights.append("\n⏱️  Function-Level Insights:")
        for func_stat in custom_timer_data['function_stats'][:10]:  # Top 10 functions
            func_name = func_stat['function'].split('.')[-1]  # Get just the function name
            avg_time = func_stat['avg_time_us']
            call_count = func_stat['call_count']
            
            if avg_time > 1000:  # > 1ms per call
                insights.append(f"  {func_name}: {avg_time:.0f}μs avg ({call_count} calls) - SLOW FUNCTION")
            elif call_count > 10000:  # Called very frequently
                insights.append(f"  {func_name}: {call_count} calls - HIGH FREQUENCY FUNCTION")
        
        # Specific optimization recommendations
        insights.append("\n💡 Specific Optimization Recommendations:")
        
        # Check for tactical pattern overhead
        tactical_functions = [f for f in custom_timer_data['function_stats'] 
                            if 'tactical' in f['function'].lower()]
        if tactical_functions:
            total_tactical_time = sum(f['total_time_ms'] for f in tactical_functions)
            tactical_percent = (total_tactical_time / (cprofile_data['search_time'] * 1000)) * 100
            
            if tactical_percent > 15:
                insights.append(f"  🔧 Tactical patterns use {tactical_percent:.1f}% of search time - optimize or reduce complexity")
            else:
                insights.append(f"  ✅ Tactical patterns use {tactical_percent:.1f}% of search time - reasonable overhead")
        
        # Check for evaluation overhead
        eval_functions = [f for f in custom_timer_data['function_stats'] 
                         if 'evaluat' in f['function'].lower()]
        if eval_functions:
            total_eval_time = sum(f['total_time_ms'] for f in eval_functions)
            eval_percent = (total_eval_time / (cprofile_data['search_time'] * 1000)) * 100
            
            if eval_percent > 30:
                insights.append(f"  🔧 Position evaluation uses {eval_percent:.1f}% of search time - major optimization target")
            elif eval_percent > 20:
                insights.append(f"  ⚠️  Position evaluation uses {eval_percent:.1f}% of search time - optimization opportunity")
        
        # Generic optimization suggestions
        insights.extend([
            "\n🛠️  General Optimization Strategies:",
            "  1. Cache expensive function results",
            "  2. Reduce function call overhead in hot paths",
            "  3. Optimize data structures for better cache locality",
            "  4. Use bit operations where possible",
            "  5. Minimize memory allocations in search loops"
        ])
        
        return insights
    
    def run_comprehensive_analysis(self) -> Dict:
        """Run the complete profiling analysis suite"""
        
        print("V7P3R Deep Code Profiler - Comprehensive Analysis")
        print("=" * 60)
        
        # Test position with tactical complexity
        test_position = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
        depth = 5
        
        print(f"Test position: {test_position}")
        print(f"Search depth: {depth}")
        print(f"Analysis starting...")
        
        # Run different profiling methods
        cprofile_data = self.run_cprofile_analysis(test_position, depth)
        custom_timer_data = self.run_custom_timer_analysis(test_position, depth)
        memory_data = self.run_memory_analysis(test_position, depth)
        
        # Analyze hot paths
        hot_paths = self.analyze_hot_paths(cprofile_data)
        
        # Generate insights
        insights = self.generate_optimization_insights(cprofile_data, custom_timer_data, hot_paths)
        
        # Compile results
        results = {
            'timestamp': time.strftime("%Y%m%d_%H%M%S"),
            'test_position': test_position,
            'search_depth': depth,
            'cprofile_analysis': cprofile_data,
            'custom_timer_analysis': custom_timer_data,
            'memory_analysis': memory_data,
            'hot_paths': hot_paths,
            'optimization_insights': insights
        }
        
        # Display results
        print("\n" + "="*60)
        print("📊 ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\n🎯 Performance Summary:")
        print(f"  Search time: {cprofile_data['search_time']:.3f}s")
        print(f"  Nodes: {cprofile_data['nodes_searched']:,}")
        print(f"  NPS: {cprofile_data['nps']:,.0f}")
        
        print(f"\n📈 Top Time-Consuming Functions:")
        for i, func in enumerate(custom_timer_data['function_stats'][:5]):
            func_name = func['function'].split('.')[-1]
            print(f"  {i+1}. {func_name}: {func['total_time_ms']:.2f}ms ({func['call_count']} calls)")
        
        print(f"\n💡 Optimization Insights:")
        for insight in insights:
            print(insight)
        
        # Save detailed results
        results_file = f"deep_profiling_analysis_{results['timestamp']}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Complete analysis saved to: {results_file}")
        
        return results

def main():
    """Run the deep profiling analysis"""
    profiler = V7P3RDeepProfiler()
    results = profiler.run_comprehensive_analysis()
    
    print(f"\n🔬 Deep Code Analysis Complete!")
    print(f"Use the insights above to optimize the highest-impact functions")
    print(f"Re-run this analysis after optimizations to measure improvements")

if __name__ == "__main__":
    main()