#!/usr/bin/env python3
"""
V7P3R Function-Level Performance Analyzer
Direct profiling of engine functions to identify bottlenecks
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

class V7P3RFunctionProfiler:
    """Direct function-level profiler for V7P3R engine"""
    
    def __init__(self):
        self.engine = V7P3REngine()
        
    def profile_single_search(self, position_fen: str, depth: int = 5) -> Dict:
        """Profile a single search and return detailed function statistics"""
        
        print(f"🔍 Profiling search at depth {depth}")
        print(f"Position: {position_fen}")
        
        board = chess.Board(position_fen)
        
        # Create and configure profiler
        profiler = cProfile.Profile()
        
        # Run the search with profiling
        profiler.enable()
        start_time = time.perf_counter()
        
        # Call search with correct parameters
        best_move, score, _ = self.engine._search_with_alpha_beta(
            board, depth, float('-inf'), float('inf'), True, start_time, 10.0
        )
        
        search_time = time.perf_counter() - start_time
        profiler.disable()
        
        # Get statistics
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(30)  # Top 30 functions
        
        profile_output = stream.getvalue()
        
        # Parse the output to extract function data
        function_data = self._parse_profile_output(profile_output, search_time)
        
        return {
            'search_time': search_time,
            'nodes_searched': self.engine.nodes_searched,
            'nps': self.engine.nodes_searched / max(search_time, 0.001),
            'best_move': str(best_move) if best_move else "None",
            'score': score,
            'function_data': function_data,
            'raw_profile': profile_output
        }
    
    def _parse_profile_output(self, output: str, total_time: float) -> List[Dict]:
        """Parse cProfile output to extract function timing data"""
        
        lines = output.split('\n')
        function_data = []
        
        # Find the start of the data section
        data_start = 0
        for i, line in enumerate(lines):
            if 'ncalls' in line and 'tottime' in line:
                data_start = i + 1
                break
        
        # Parse each function line
        for line in lines[data_start:data_start + 25]:  # Top 25 functions
            if line.strip() and not line.startswith('---') and len(line.split()) >= 6:
                parts = line.split()
                try:
                    ncalls = parts[0]
                    tottime = float(parts[1])
                    cumtime = float(parts[3])
                    filename_func = ' '.join(parts[5:])
                    
                    # Clean up the function name
                    if '(' in filename_func and ')' in filename_func:
                        func_part = filename_func.split('(')[1].split(')')[0]
                    else:
                        func_part = filename_func
                    
                    # Extract call count (handle recursive calls like "100/50")
                    call_count = int(ncalls.split('/')[0]) if '/' in ncalls else int(ncalls)
                    
                    # Only include functions that took meaningful time or are engine functions
                    if (tottime > 0.001 or 'v7p3r' in func_part.lower() or 
                        'evaluate' in func_part.lower() or 'search' in func_part.lower()):
                        
                        function_data.append({
                            'function': func_part,
                            'call_count': call_count,
                            'total_time_ms': tottime * 1000,
                            'cumulative_time_ms': cumtime * 1000,
                            'avg_time_us': (tottime / max(call_count, 1)) * 1000000,
                            'percent_of_search': (cumtime / total_time) * 100
                        })
                        
                except (ValueError, IndexError):
                    continue
        
        # Sort by cumulative time
        function_data.sort(key=lambda x: x['cumulative_time_ms'], reverse=True)
        return function_data
    
    def analyze_performance_bottlenecks(self, profile_data: Dict) -> List[str]:
        """Analyze profiling data to identify performance bottlenecks"""
        
        analysis = []
        
        # Overall performance metrics
        nps = profile_data['nps']
        search_time = profile_data['search_time']
        nodes = profile_data['nodes_searched']
        
        analysis.append("📊 PERFORMANCE SUMMARY")
        analysis.append("=" * 40)
        analysis.append(f"Search Time: {search_time:.3f} seconds")
        analysis.append(f"Nodes Searched: {nodes:,}")
        analysis.append(f"Nodes per Second: {nps:,.0f}")
        analysis.append(f"Best Move: {profile_data['best_move']}")
        analysis.append(f"Score: {profile_data['score']:.2f}")
        analysis.append("")
        
        # Performance assessment
        if nps > 200000:
            analysis.append("✅ EXCELLENT: Performance is very good")
        elif nps > 100000:
            analysis.append("✅ GOOD: Performance is solid")
        elif nps > 50000:
            analysis.append("⚠️  MODERATE: Performance could be improved")
        else:
            analysis.append("🚨 POOR: Performance needs significant optimization")
        
        analysis.append("")
        
        # Function-level analysis
        analysis.append("🔥 TOP PERFORMANCE BOTTLENECKS")
        analysis.append("=" * 40)
        
        function_data = profile_data['function_data']
        
        for i, func in enumerate(function_data[:10]):  # Top 10 functions
            analysis.append(f"{i+1:2d}. {func['function']}")
            analysis.append(f"    Time: {func['cumulative_time_ms']:.2f}ms ({func['percent_of_search']:.1f}% of search)")
            analysis.append(f"    Calls: {func['call_count']:,}")
            analysis.append(f"    Avg: {func['avg_time_us']:.1f}μs per call")
            
            # Add specific insights
            if func['percent_of_search'] > 25:
                analysis.append("    🚨 CRITICAL: This function dominates search time!")
            elif func['percent_of_search'] > 15:
                analysis.append("    ⚠️  HIGH IMPACT: Major optimization candidate")
            elif func['percent_of_search'] > 8:
                analysis.append("    📈 SIGNIFICANT: Good optimization target")
            elif func['call_count'] > 10000:
                analysis.append("    🔄 HIGH FREQUENCY: Called very often")
            
            analysis.append("")
        
        return analysis
    
    def generate_optimization_recommendations(self, profile_data: Dict) -> List[str]:
        """Generate specific optimization recommendations"""
        
        recommendations = []
        function_data = profile_data['function_data']
        
        recommendations.append("💡 OPTIMIZATION RECOMMENDATIONS")
        recommendations.append("=" * 40)
        
        # Analyze function categories
        evaluation_funcs = [f for f in function_data if 'evaluat' in f['function'].lower()]
        search_funcs = [f for f in function_data if 'search' in f['function'].lower()]
        tactical_funcs = [f for f in function_data if 'tactical' in f['function'].lower()]
        move_funcs = [f for f in function_data if 'move' in f['function'].lower() or 'order' in f['function'].lower()]
        
        total_time = profile_data['search_time'] * 1000  # Convert to ms
        
        # Evaluation optimization
        if evaluation_funcs:
            eval_time = sum(f['cumulative_time_ms'] for f in evaluation_funcs)
            eval_percent = (eval_time / total_time) * 100
            
            recommendations.append(f"🔧 POSITION EVALUATION ({eval_percent:.1f}% of time):")
            if eval_percent > 30:
                recommendations.append("   🚨 CRITICAL: Evaluation is too slow!")
                recommendations.append("   • Cache evaluation results")
                recommendations.append("   • Simplify evaluation functions")
                recommendations.append("   • Use incremental evaluation")
            elif eval_percent > 20:
                recommendations.append("   ⚠️  Evaluation optimization recommended")
                recommendations.append("   • Profile individual evaluation components")
                recommendations.append("   • Optimize piece-square tables")
            else:
                recommendations.append("   ✅ Evaluation time is reasonable")
            recommendations.append("")
        
        # Tactical optimization
        if tactical_funcs:
            tactical_time = sum(f['cumulative_time_ms'] for f in tactical_funcs)
            tactical_percent = (tactical_time / total_time) * 100
            
            recommendations.append(f"🔧 TACTICAL PATTERNS ({tactical_percent:.1f}% of time):")
            if tactical_percent > 20:
                recommendations.append("   🚨 Tactical patterns are too expensive!")
                recommendations.append("   • Implement time budgets")
                recommendations.append("   • Reduce pattern complexity")
                recommendations.append("   • Cache tactical evaluations")
            elif tactical_percent > 10:
                recommendations.append("   ⚠️  Consider optimizing tactical detection")
                recommendations.append("   • Profile individual pattern detectors")
            else:
                recommendations.append("   ✅ Tactical overhead is acceptable")
            recommendations.append("")
        
        # Search optimization
        if search_funcs:
            search_time = sum(f['cumulative_time_ms'] for f in search_funcs)
            search_percent = (search_time / total_time) * 100
            
            recommendations.append(f"🔧 SEARCH ALGORITHM ({search_percent:.1f}% of time):")
            recommendations.append("   • Improve transposition table hit rate")
            recommendations.append("   • Optimize alpha-beta pruning")
            recommendations.append("   • Better move ordering")
            recommendations.append("")
        
        # Move ordering optimization
        if move_funcs:
            move_time = sum(f['cumulative_time_ms'] for f in move_funcs)
            move_percent = (move_time / total_time) * 100
            
            recommendations.append(f"🔧 MOVE ORDERING ({move_percent:.1f}% of time):")
            if move_percent > 15:
                recommendations.append("   ⚠️  Move ordering is expensive")
                recommendations.append("   • Cache move scores")
                recommendations.append("   • Simplify scoring functions")
            recommendations.append("")
        
        # High-frequency function optimization
        high_freq = [f for f in function_data if f['call_count'] > 5000]
        if high_freq:
            recommendations.append("🔧 HIGH-FREQUENCY FUNCTIONS:")
            for func in high_freq[:3]:
                recommendations.append(f"   • {func['function']}: {func['call_count']:,} calls")
            recommendations.append("   → Focus on micro-optimizations")
            recommendations.append("")
        
        # General recommendations
        recommendations.extend([
            "🔧 GENERAL OPTIMIZATION STRATEGIES:",
            "   1. Profile after each optimization to measure impact",
            "   2. Focus on functions using >5% of search time",
            "   3. Optimize algorithms before micro-optimizations",
            "   4. Use bit operations where possible",
            "   5. Minimize memory allocations in hot paths",
            "   6. Consider lookup tables for expensive calculations"
        ])
        
        return recommendations
    
    def run_multi_position_analysis(self) -> Dict:
        """Run profiling analysis on multiple test positions"""
        
        print("V7P3R Function-Level Performance Analyzer")
        print("=" * 60)
        
        test_positions = [
            ("Tactical Complex", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
            ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("Middlegame", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4")
        ]
        
        all_results = {}
        
        for name, fen in test_positions:
            print(f"\n{'=' * 60}")
            print(f"ANALYZING: {name}")
            print(f"{'=' * 60}")
            
            # Profile this position
            profile_data = self.profile_single_search(fen, depth=5)
            
            # Analyze bottlenecks
            analysis = self.analyze_performance_bottlenecks(profile_data)
            
            # Generate recommendations
            recommendations = self.generate_optimization_recommendations(profile_data)
            
            # Display results
            for line in analysis:
                print(line)
            
            print()
            for line in recommendations:
                print(line)
            
            all_results[name] = {
                'profile_data': profile_data,
                'analysis': analysis,
                'recommendations': recommendations
            }
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"function_profiling_analysis_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n📁 Complete analysis saved to: {results_file}")
        
        return all_results

def main():
    """Run the function-level profiling analysis"""
    profiler = V7P3RFunctionProfiler()
    results = profiler.run_multi_position_analysis()
    
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print("Review the bottlenecks and recommendations above.")
    print("Focus on optimizing functions that use >5% of search time.")

if __name__ == "__main__":
    main()