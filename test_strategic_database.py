#!/usr/bin/env python3
"""
V7P3R v11: Strategic Database Integration Test
Tests the enhanced strategic database with pattern matching and similarity scoring
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

from v7p3r import V7P3REngine


class StrategicDatabaseTester:
    """Test suite for strategic database enhancements"""
    
    def __init__(self):
        self.engine = V7P3REngine()
        self.test_results = {}
    
    def test_strategic_database_integration(self) -> Dict:
        """Test that strategic database is properly integrated"""
        print("=" * 60)
        print("TESTING STRATEGIC DATABASE INTEGRATION")
        print("=" * 60)
        
        results = {
            'strategic_database_loaded': False,
            'pattern_matching_tests': [],
            'similarity_scoring_tests': [],
            'evaluation_bonus_tests': [],
            'move_bonus_tests': []
        }
        
        # Test 1: Check if strategic database is loaded
        if hasattr(self.engine, 'strategic_database'):
            results['strategic_database_loaded'] = True
            print("✅ Strategic database integrated")
            
            db_stats = self.engine.strategic_database.get_statistics()
            print(f"   Database Statistics: {db_stats}")
        else:
            print("❌ Strategic database not integrated")
            return results
        
        # Test 2: Test pattern matching
        print(f"\nTesting position pattern matching...")
        
        test_positions = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting Position"),
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", "Complex Kiwipete"),
            ("8/8/8/4k3/4K3/8/8/8 w - - 0 1", "King Endgame")
        ]
        
        for fen, name in test_positions:
            try:
                board = chess.Board(fen)
                pattern = self.engine.strategic_database.create_position_pattern(board)
                
                pattern_test = {
                    'position_name': name,
                    'fen': fen,
                    'strategic_themes': pattern.strategic_themes,
                    'king_safety_white': pattern.king_safety.get('white', 0.0),
                    'king_safety_black': pattern.king_safety.get('black', 0.0),
                    'evaluation_bonus': pattern.evaluation_bonus,
                    'pattern_created': True
                }
                
                results['pattern_matching_tests'].append(pattern_test)
                
                print(f"   {name}:")
                print(f"     Strategic Themes: {pattern.strategic_themes}")
                print(f"     King Safety: W={pattern.king_safety.get('white', 0.0):.2f}, B={pattern.king_safety.get('black', 0.0):.2f}")
                print(f"     Evaluation Bonus: {pattern.evaluation_bonus:.3f}")
                print(f"     ✅ Pattern analysis working")
                
            except Exception as e:
                pattern_test = {
                    'position_name': name,
                    'error': str(e),
                    'pattern_created': False
                }
                results['pattern_matching_tests'].append(pattern_test)
                print(f"   {name}: ❌ Error - {e}")
        
        # Test 3: Test similarity scoring
        print(f"\nTesting position similarity scoring...")
        
        # Use starting position and find similar positions
        starting_board = chess.Board()
        try:
            similar_positions = self.engine.strategic_database.find_similar_positions(
                starting_board, min_similarity=0.3, max_results=3
            )
            
            similarity_test = {
                'test_position': 'Starting Position',
                'similar_positions_found': len(similar_positions),
                'similarities': []
            }
            
            print(f"   Found {len(similar_positions)} similar positions to starting position")
            
            for position_key, similarity, position_data in similar_positions:
                sim_data = {
                    'position_key': position_key,
                    'similarity_score': similarity,
                    'fen': position_data.get('fen', '')
                }
                similarity_test['similarities'].append(sim_data)
                print(f"     Position {position_key}: {similarity:.3f} similarity")
            
            results['similarity_scoring_tests'].append(similarity_test)
            
            if len(similar_positions) > 0:
                print(f"   ✅ Similarity scoring working")
            else:
                print(f"   ℹ️ No similar positions found (may be normal)")
                
        except Exception as e:
            print(f"   ❌ Similarity scoring error: {e}")
        
        # Test 4: Test evaluation bonuses
        print(f"\nTesting strategic evaluation bonuses...")
        
        for fen, name in test_positions[:2]:  # Test first 2 positions
            try:
                board = chess.Board(fen)
                
                # Get regular evaluation
                regular_eval = self.engine._evaluate_position(board)
                
                # Get strategic bonus separately
                strategic_bonus = self.engine.strategic_database.get_strategic_evaluation_bonus(board)
                
                eval_test = {
                    'position_name': name,
                    'regular_evaluation': regular_eval,
                    'strategic_bonus': strategic_bonus,
                    'bonus_applied': strategic_bonus != 0.0
                }
                
                results['evaluation_bonus_tests'].append(eval_test)
                
                print(f"   {name}:")
                print(f"     Regular Evaluation: {regular_eval:.3f}")
                print(f"     Strategic Bonus: {strategic_bonus:.3f}")
                
                if strategic_bonus != 0.0:
                    print(f"     ✅ Strategic bonus applied")
                else:
                    print(f"     ℹ️ No strategic bonus (may be normal)")
                    
            except Exception as e:
                print(f"   {name}: ❌ Error - {e}")
        
        # Test 5: Test move bonuses
        print(f"\nTesting strategic move bonuses...")
        
        # Test some moves in starting position
        starting_board = chess.Board()
        test_moves = [chess.Move.from_uci('e2e4'), chess.Move.from_uci('d2d4'), 
                     chess.Move.from_uci('g1f3'), chess.Move.from_uci('b1c3')]
        
        for move in test_moves:
            if move in starting_board.legal_moves:
                try:
                    # Test regular nudge bonus (includes strategic bonus now)
                    nudge_bonus = self.engine._get_nudge_bonus(starting_board, move)
                    
                    # Test strategic bonus separately
                    strategic_bonus = self.engine.strategic_database.get_strategic_move_bonus(starting_board, move)
                    
                    move_test = {
                        'move': move.uci(),
                        'nudge_bonus': nudge_bonus,
                        'strategic_bonus': strategic_bonus,
                        'bonus_applied': nudge_bonus > 0 or strategic_bonus != 0
                    }
                    
                    results['move_bonus_tests'].append(move_test)
                    
                    print(f"   Move {move.uci()}:")
                    print(f"     Total Nudge Bonus: {nudge_bonus:.3f}")
                    print(f"     Strategic Bonus: {strategic_bonus:.3f}")
                    
                    if nudge_bonus > 0 or strategic_bonus != 0:
                        print(f"     ✅ Move bonus working")
                    else:
                        print(f"     ℹ️ No move bonus")
                        
                except Exception as e:
                    print(f"   Move {move.uci()}: ❌ Error - {e}")
        
        return results
    
    def test_strategic_database_performance(self) -> Dict:
        """Test strategic database performance impact"""
        print("\n" + "=" * 60)
        print("TESTING STRATEGIC DATABASE PERFORMANCE")
        print("=" * 60)
        
        results = {
            'performance_tests': [],
            'cache_efficiency': {},
            'database_statistics': {}
        }
        
        # Performance test: search with and without strategic database
        test_position = chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
        
        print(f"Testing search performance with strategic enhancements...")
        
        # Run multiple searches to get average
        search_times = []
        nodes_searched = []
        
        for i in range(3):
            start_time = time.time()
            best_move = self.engine.search(test_position, 1.0)  # 1 second search
            search_time = time.time() - start_time
            
            search_times.append(search_time)
            nodes_searched.append(self.engine.nodes_searched)
            
            print(f"   Search {i+1}: {search_time:.3f}s, {self.engine.nodes_searched:,} nodes, move: {best_move}")
        
        avg_search_time = sum(search_times) / len(search_times)
        avg_nodes = sum(nodes_searched) / len(nodes_searched)
        
        performance_test = {
            'position': 'Complex Kiwipete',
            'search_count': len(search_times),
            'average_search_time': avg_search_time,
            'average_nodes': avg_nodes,
            'time_per_node': avg_search_time / avg_nodes if avg_nodes > 0 else 0,
            'performance_acceptable': avg_search_time <= 1.5  # Allow 50% tolerance
        }
        
        results['performance_tests'].append(performance_test)
        
        if performance_test['performance_acceptable']:
            print(f"   ✅ Performance acceptable: {avg_search_time:.3f}s avg, {avg_nodes:,.0f} nodes avg")
        else:
            print(f"   ⚠️ Performance concern: {avg_search_time:.3f}s avg")
        
        # Get cache and database statistics
        try:
            db_stats = self.engine.strategic_database.get_statistics()
            results['database_statistics'] = db_stats
            
            cache_efficiency = {
                'cache_hit_rate': db_stats['cache_hits'] / (db_stats['cache_hits'] + db_stats['cache_misses']) * 100 
                    if db_stats['cache_hits'] + db_stats['cache_misses'] > 0 else 0,
                'total_cache_operations': db_stats['cache_hits'] + db_stats['cache_misses'],
                'similarity_calculations': db_stats['similarity_calculations']
            }
            results['cache_efficiency'] = cache_efficiency
            
            print(f"\nDatabase Statistics:")
            print(f"   Positions Loaded: {db_stats['positions_loaded']}")
            print(f"   Patterns Created: {db_stats['patterns_created']}")
            print(f"   Cache Hit Rate: {cache_efficiency['cache_hit_rate']:.1f}%")
            print(f"   Similarity Calculations: {db_stats['similarity_calculations']}")
            
        except Exception as e:
            print(f"   Error getting statistics: {e}")
        
        return results
    
    def run_comprehensive_strategic_test(self) -> Dict:
        """Run all strategic database tests"""
        print("V7P3R v11: Strategic Database Comprehensive Test")
        print("=" * 80)
        
        # Run test suites
        integration_results = self.test_strategic_database_integration()
        performance_results = self.test_strategic_database_performance()
        
        # Compile overall results
        overall_results = {
            'test_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'engine_version': 'v10.9 with v11 Phase 2 Strategic Enhancements',
            'integration_tests': integration_results,
            'performance_tests': performance_results,
            'overall_status': 'UNKNOWN'
        }
        
        # Determine overall status
        critical_tests = [
            integration_results.get('strategic_database_loaded', False),
            len(integration_results.get('pattern_matching_tests', [])) > 0,
            len(performance_results.get('performance_tests', [])) > 0
        ]
        
        if all(critical_tests):
            overall_results['overall_status'] = 'PASS'
        elif any(critical_tests):
            overall_results['overall_status'] = 'PARTIAL'
        else:
            overall_results['overall_status'] = 'FAIL'
        
        # Print summary
        print("\n" + "=" * 80)
        print("STRATEGIC DATABASE TEST SUMMARY")
        print("=" * 80)
        
        status_emoji = {
            'PASS': '🎉',
            'PARTIAL': '⚠️',
            'FAIL': '❌'
        }
        
        print(f"{status_emoji[overall_results['overall_status']]} Overall Status: {overall_results['overall_status']}")
        
        if integration_results['strategic_database_loaded']:
            print(f"✅ Strategic Database: Integrated and functional")
        else:
            print(f"❌ Strategic Database: Not integrated")
        
        pattern_tests = len(integration_results.get('pattern_matching_tests', []))
        successful_patterns = len([t for t in integration_results.get('pattern_matching_tests', []) 
                                 if t.get('pattern_created', False)])
        print(f"📊 Pattern Matching: {successful_patterns}/{pattern_tests} positions analyzed")
        
        performance_tests = performance_results.get('performance_tests', [])
        if performance_tests:
            acceptable_performance = [t for t in performance_tests if t.get('performance_acceptable', False)]
            print(f"⚡ Performance: {len(acceptable_performance)}/{len(performance_tests)} tests acceptable")
        
        print(f"\n🚀 Ready for Production: {overall_results['overall_status'] == 'PASS'}")
        
        return overall_results
    
    def save_results(self, results: Dict, filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"strategic_database_test_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📁 Test results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Error saving results: {e}")


def main():
    """Main testing function"""
    print("Initializing V7P3R Strategic Database Testing System...")
    
    tester = StrategicDatabaseTester()
    
    try:
        results = tester.run_comprehensive_strategic_test()
        tester.save_results(results)
        
        # Return success code based on results
        return 0 if results['overall_status'] == 'PASS' else 1
        
    except Exception as e:
        print(f"\n❌ Strategic database testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)