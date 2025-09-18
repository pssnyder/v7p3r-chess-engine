#!/usr/bin/env python3
"""
V7P3R v11 Phase 3A: Lightweight Defense Testing
Performance-focused testing for defensive analysis components
Author: Pat Snyder
"""

import time
import chess
import sys
import os
import json
from typing import Dict, List, Tuple

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r_lightweight_defense import V7P3RLightweightDefense, V7P3RTacticalEscape, Phase3PerformanceMonitor


class Phase3DefenseTester:
    """Test suite for Phase 3A lightweight defensive analysis"""
    
    def __init__(self):
        self.lightweight_defense = V7P3RLightweightDefense()
        self.tactical_escape = V7P3RTacticalEscape()
        self.performance_monitor = Phase3PerformanceMonitor(baseline_nps=2200)
        
        # Test positions with different defensive characteristics
        self.test_positions = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting Position", "balanced"),
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", "Complex Tactical", "tactical"),
            ("8/8/8/4k3/4K3/8/8/8 w - - 0 1", "King Endgame", "endgame"),
            ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Italian Game", "opening"),
            ("rnbqk1nr/pppp1ppp/4p3/2b5/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3", "Hanging Pieces", "tactical_escape"),
            ("2rq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP1QPPP/RNB2RK1 w - - 0 8", "Middlegame", "positional")
        ]
    
    def test_defensive_analysis_performance(self) -> Dict:
        """Test defensive analysis performance and accuracy"""
        print("=" * 70)
        print("TESTING LIGHTWEIGHT DEFENSIVE ANALYSIS")
        print("=" * 70)
        
        results = {
            'performance_tests': [],
            'accuracy_tests': [],
            'cache_efficiency': {},
            'component_stats': {}
        }
        
        # Test 1: Performance benchmarking
        print("Testing defensive analysis performance...")
        
        total_time = 0.0
        position_count = 0
        
        for fen, name, position_type in self.test_positions:
            board = chess.Board(fen)
            
            # Run multiple iterations for accurate timing
            iterations = 100
            start_time = time.time()
            
            for _ in range(iterations):
                defense_score = self.lightweight_defense.quick_defensive_assessment(board)
            
            elapsed_time = time.time() - start_time
            avg_time_ms = (elapsed_time / iterations) * 1000
            
            performance_test = {
                'position_name': name,
                'position_type': position_type,
                'avg_time_ms': avg_time_ms,
                'defense_score': defense_score,
                'meets_target': avg_time_ms < 2.0  # Target: <2ms
            }
            
            results['performance_tests'].append(performance_test)
            total_time += elapsed_time
            position_count += iterations
            
            status = "✅" if avg_time_ms < 2.0 else "⚠️"
            print(f"   {name}: {avg_time_ms:.3f}ms avg, score: {defense_score:.3f} {status}")
        
        overall_avg_ms = (total_time / position_count) * 1000
        print(f"\nOverall Performance: {overall_avg_ms:.3f}ms average")
        
        if overall_avg_ms < 2.0:
            print("✅ Performance target met (<2ms)")
        else:
            print("⚠️ Performance target missed (>2ms)")
        
        # Test 2: Tactical escape performance
        print(f"\nTesting tactical escape detection...")
        
        escape_times = []
        escape_results = []
        
        for fen, name, position_type in self.test_positions:
            board = chess.Board(fen)
            
            start_time = time.time()
            escape_bonus = self.tactical_escape.detect_escape_opportunities(board, board.turn)
            escape_time = (time.time() - start_time) * 1000
            
            escape_times.append(escape_time)
            escape_test = {
                'position_name': name,
                'escape_bonus': escape_bonus,
                'time_ms': escape_time,
                'meets_target': escape_time < 3.0  # Target: <3ms
            }
            
            escape_results.append(escape_test)
            
            status = "✅" if escape_time < 3.0 else "⚠️"
            print(f"   {name}: {escape_time:.3f}ms, bonus: {escape_bonus:.3f} {status}")
        
        results['accuracy_tests'] = escape_results
        
        avg_escape_time = sum(escape_times) / len(escape_times)
        print(f"\nEscape Detection Average: {avg_escape_time:.3f}ms")
        
        if avg_escape_time < 3.0:
            print("✅ Escape detection target met (<3ms)")
        else:
            print("⚠️ Escape detection target missed (>3ms)")
        
        # Test 3: Cache efficiency
        print(f"\nTesting cache efficiency...")
        
        # Reset stats and run cache test
        self.lightweight_defense.clear_cache()
        self.lightweight_defense.reset_performance_stats()
        
        # First pass - populate cache
        for fen, _, _ in self.test_positions:
            board = chess.Board(fen)
            self.lightweight_defense.quick_defensive_assessment(board)
        
        # Second pass - should hit cache
        for fen, _, _ in self.test_positions:
            board = chess.Board(fen)
            self.lightweight_defense.quick_defensive_assessment(board)
        
        defense_stats = self.lightweight_defense.get_performance_stats()
        escape_stats = self.tactical_escape.get_performance_stats()
        
        results['cache_efficiency'] = defense_stats
        results['component_stats'] = {
            'defense': defense_stats,
            'escape': escape_stats
        }
        
        print(f"   Cache Hit Rate: {defense_stats['cache_hit_rate']:.1f}%")
        print(f"   Average Time: {defense_stats['avg_time_ms']:.3f}ms")
        print(f"   Cache Size: {defense_stats['cache_size']} entries")
        
        if defense_stats['cache_hit_rate'] > 50:
            print("✅ Cache efficiency acceptable (>50% hit rate)")
        else:
            print("⚠️ Cache efficiency low (<50% hit rate)")
        
        return results
    
    def test_integration_with_search(self) -> Dict:
        """Test defensive analysis integration without impacting search performance"""
        print("\n" + "=" * 70)
        print("TESTING INTEGRATION WITH SEARCH SIMULATION")
        print("=" * 70)
        
        results = {
            'integration_tests': [],
            'performance_impact': {},
            'nps_simulation': {}
        }
        
        # Simulate search-like workload
        print("Simulating search workload with defensive analysis...")
        
        # Test selective application (every 200 nodes)
        node_counts = [100, 200, 500, 1000, 2000]
        
        for node_count in node_counts:
            # Simulate nodes with selective defensive analysis
            start_time = time.time()
            defensive_calls = 0
            
            for node in range(node_count):
                # Use a random test position
                pos_index = node % len(self.test_positions)
                fen, _, _ = self.test_positions[pos_index]
                board = chess.Board(fen)
                
                # Apply defensive analysis every 200 nodes (as planned)
                if node % 200 == 0:
                    defense_score = self.lightweight_defense.quick_defensive_assessment(board)
                    escape_bonus = self.tactical_escape.detect_escape_opportunities(board, board.turn)
                    defensive_calls += 1
            
            elapsed_time = time.time() - start_time
            simulated_nps = node_count / elapsed_time if elapsed_time > 0 else 0
            
            integration_test = {
                'node_count': node_count,
                'defensive_calls': defensive_calls,
                'elapsed_time': elapsed_time,
                'simulated_nps': simulated_nps,
                'overhead_per_node': (elapsed_time / node_count * 1000) if node_count > 0 else 0
            }
            
            results['integration_tests'].append(integration_test)
            
            print(f"   {node_count} nodes: {simulated_nps:.0f} NPS, {defensive_calls} defensive calls")
        
        # Check if we maintain our performance target
        large_test = results['integration_tests'][-1]  # Largest test
        nps_acceptable = large_test['simulated_nps'] > 2000  # Conservative target
        
        results['performance_impact'] = {
            'nps_maintained': nps_acceptable,
            'largest_test_nps': large_test['simulated_nps'],
            'overhead_per_node': large_test['overhead_per_node']
        }
        
        if nps_acceptable:
            print(f"✅ Integration maintains acceptable NPS (>{large_test['simulated_nps']:.0f})")
        else:
            print(f"⚠️ Integration may impact NPS ({large_test['simulated_nps']:.0f})")
        
        return results
    
    def run_comprehensive_phase3a_test(self) -> Dict:
        """Run comprehensive Phase 3A testing"""
        print("V7P3R v11 Phase 3A: Lightweight Defense Comprehensive Test")
        print("=" * 80)
        
        # Run all test suites
        performance_results = self.test_defensive_analysis_performance()
        integration_results = self.test_integration_with_search()
        
        # Compile overall results
        overall_results = {
            'test_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'phase': '3A - Lightweight Defense',
            'performance_analysis': performance_results,
            'integration_analysis': integration_results,
            'phase3a_status': 'UNKNOWN'
        }
        
        # Determine Phase 3A status
        performance_ok = all(t['meets_target'] for t in performance_results['performance_tests'])
        escape_ok = all(t['meets_target'] for t in performance_results['accuracy_tests'])
        cache_ok = performance_results['cache_efficiency']['cache_hit_rate'] > 50
        integration_ok = integration_results['performance_impact']['nps_maintained']
        
        success_criteria = [performance_ok, escape_ok, cache_ok, integration_ok]
        
        if all(success_criteria):
            overall_results['phase3a_status'] = 'PASS'
        elif sum(success_criteria) >= 3:
            overall_results['phase3a_status'] = 'PARTIAL'
        else:
            overall_results['phase3a_status'] = 'FAIL'
        
        # Print summary
        print("\n" + "=" * 80)
        print("PHASE 3A TEST SUMMARY")
        print("=" * 80)
        
        status_emoji = {
            'PASS': '🎉',
            'PARTIAL': '⚠️',
            'FAIL': '❌'
        }
        
        print(f"{status_emoji[overall_results['phase3a_status']]} Phase 3A Status: {overall_results['phase3a_status']}")
        
        print(f"\nTest Results:")
        print(f"   Defensive Performance: {'✅' if performance_ok else '❌'} (<2ms target)")
        print(f"   Escape Detection: {'✅' if escape_ok else '❌'} (<3ms target)")
        print(f"   Cache Efficiency: {'✅' if cache_ok else '❌'} (>50% hit rate)")
        print(f"   Search Integration: {'✅' if integration_ok else '❌'} (NPS maintained)")
        
        if overall_results['phase3a_status'] == 'PASS':
            print(f"\n🚀 Phase 3A ready for integration into main engine!")
        else:
            print(f"\n🔧 Phase 3A needs optimization before integration")
        
        return overall_results
    
    def save_results(self, results: Dict, filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"phase3a_defense_test_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📁 Test results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Error saving results: {e}")


def main():
    """Main testing function"""
    print("Initializing Phase 3A Defensive Analysis Testing...")
    
    tester = Phase3DefenseTester()
    
    try:
        results = tester.run_comprehensive_phase3a_test()
        tester.save_results(results)
        
        # Return success code based on results
        return 0 if results['phase3a_status'] == 'PASS' else 1
        
    except Exception as e:
        print(f"\n❌ Phase 3A testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)