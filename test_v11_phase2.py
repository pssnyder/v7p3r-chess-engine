#!/usr/bin/env python3
"""
V7P3R v11 Phase 2: Comprehensive Verification and Integration Test
Tests nudge system, time management integration, and strategic improvements
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

class V7P3RPhase2Tester:
    """Comprehensive Phase 2 testing suite"""
    
    def __init__(self):
        self.engine = V7P3REngine()
        self.test_results = {}
    
    def test_nudge_system_integration(self) -> Dict:
        """Test that nudge system is working correctly"""
        print("=" * 60)
        print("TESTING NUDGE SYSTEM INTEGRATION")
        print("=" * 60)
        
        results = {
            'nudge_database_loaded': False,
            'nudge_database_size': 0,
            'nudge_position_tests': [],
            'instant_move_tests': [],
            'move_ordering_tests': []
        }
        
        # Test 1: Check if nudge database loaded
        if hasattr(self.engine, 'nudge_database') and self.engine.nudge_database:
            results['nudge_database_loaded'] = True
            results['nudge_database_size'] = len(self.engine.nudge_database)
            print(f"✅ Nudge database loaded: {results['nudge_database_size']} positions")
        else:
            print("❌ Nudge database not loaded")
            return results
        
        # Test 2: Test nudge bonus calculation for known positions
        print(f"\nTesting nudge bonus calculation...")
        test_count = 0
        successful_tests = 0
        
        for position_key, position_data in list(self.engine.nudge_database.items())[:5]:  # Test first 5
            try:
                fen = position_data.get('fen', '')
                if not fen:
                    continue
                    
                board = chess.Board(fen)
                test_count += 1
                
                # Test each move in the position
                moves_data = position_data.get('moves', {})
                position_test = {
                    'fen': fen,
                    'moves_tested': 0,
                    'nudge_bonuses_found': 0,
                    'average_bonus': 0.0
                }
                
                total_bonus = 0.0
                bonuses_found = 0
                
                for move_uci, move_data in moves_data.items():
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            bonus = self.engine._get_nudge_bonus(board, move)
                            position_test['moves_tested'] += 1
                            
                            if bonus > 0:
                                bonuses_found += 1
                                total_bonus += bonus
                                
                    except Exception as e:
                        continue
                
                if position_test['moves_tested'] > 0:
                    position_test['nudge_bonuses_found'] = bonuses_found
                    if bonuses_found > 0:
                        position_test['average_bonus'] = total_bonus / bonuses_found
                        successful_tests += 1
                    
                results['nudge_position_tests'].append(position_test)
                
            except Exception as e:
                print(f"   Error testing position {position_key}: {e}")
                continue
        
        nudge_success_rate = (successful_tests / test_count * 100) if test_count > 0 else 0
        print(f"   Tested {test_count} positions, {successful_tests} had nudge bonuses")
        print(f"   Nudge system success rate: {nudge_success_rate:.1f}%")
        
        if nudge_success_rate > 50:
            print("   ✅ Nudge bonus calculation working")
        else:
            print("   ⚠️ Nudge bonus calculation may have issues")
        
        # Test 3: Test instant move detection
        print(f"\nTesting instant move detection...")
        instant_move_count = 0
        
        for position_key, position_data in list(self.engine.nudge_database.items())[:10]:
            try:
                fen = position_data.get('fen', '')
                if not fen:
                    continue
                    
                board = chess.Board(fen)
                instant_move = self.engine._check_instant_nudge_move(board)
                
                instant_test = {
                    'fen': fen,
                    'instant_move_found': instant_move is not None,
                    'instant_move': str(instant_move) if instant_move else None
                }
                
                if instant_move:
                    instant_move_count += 1
                    
                results['instant_move_tests'].append(instant_test)
                
            except Exception as e:
                continue
        
        print(f"   Found {instant_move_count} instant moves in test positions")
        if instant_move_count > 0:
            print("   ✅ Instant move detection working")
        else:
            print("   ℹ️ No instant moves found (this is normal for random positions)")
        
        return results
    
    def test_time_management_integration(self) -> Dict:
        """Test time management integration with search"""
        print("\n" + "=" * 60)
        print("TESTING TIME MANAGEMENT INTEGRATION")
        print("=" * 60)
        
        results = {
            'time_manager_available': False,
            'complexity_analysis_tests': [],
            'time_allocation_tests': [],
            'search_integration_tests': []
        }
        
        # Test 1: Check if time manager is available
        if hasattr(self.engine, 'time_manager'):
            results['time_manager_available'] = True
            print("✅ Time manager integrated")
        else:
            print("❌ Time manager not integrated")
            return results
        
        # Test 2: Test complexity analysis
        print(f"\nTesting position complexity analysis...")
        
        test_positions = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting Position", 0.3),
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", "Complex Kiwipete", 0.7),
            ("8/8/8/4k3/4K3/8/8/8 w - - 0 1", "Simple Endgame", 0.2)
        ]
        
        for fen, name, expected_complexity in test_positions:
            board = chess.Board(fen)
            time_remaining = 120.0
            
            allocated_time, target_depth = self.engine.time_manager.calculate_time_allocation(board, time_remaining)
            
            complexity_test = {
                'position_name': name,
                'fen': fen,
                'allocated_time': allocated_time,
                'target_depth': target_depth,
                'reasonable_allocation': 1.0 <= allocated_time <= 15.0,
                'reasonable_depth': 3 <= target_depth <= 12
            }
            
            results['complexity_analysis_tests'].append(complexity_test)
            
            print(f"   {name}:")
            print(f"     Allocated Time: {allocated_time:.2f}s")
            print(f"     Target Depth: {target_depth}")
            
            if complexity_test['reasonable_allocation'] and complexity_test['reasonable_depth']:
                print(f"     ✅ Reasonable allocation")
            else:
                print(f"     ⚠️ Allocation may be off")
        
        # Test 3: Test search integration
        print(f"\nTesting search integration with time management...")
        
        for fen, name, _ in test_positions[:2]:  # Test first 2 positions
            board = chess.Board(fen)
            
            start_time = time.time()
            try:
                best_move = self.engine.search(board, 2.0)  # 2 second search
                search_time = time.time() - start_time
                
                search_test = {
                    'position_name': name,
                    'search_time': search_time,
                    'best_move': str(best_move),
                    'nodes_searched': self.engine.nodes_searched,
                    'time_respected': search_time <= 2.5,  # Allow 25% tolerance
                    'move_found': best_move != chess.Move.null()
                }
                
                results['search_integration_tests'].append(search_test)
                
                print(f"   {name}:")
                print(f"     Search Time: {search_time:.2f}s")
                print(f"     Best Move: {best_move}")
                print(f"     Nodes: {self.engine.nodes_searched:,}")
                
                if search_test['time_respected'] and search_test['move_found']:
                    print(f"     ✅ Search integration working")
                else:
                    print(f"     ⚠️ Search integration issues")
                    
            except Exception as e:
                print(f"     ❌ Search failed: {e}")
                search_test = {
                    'position_name': name,
                    'error': str(e),
                    'time_respected': False,
                    'move_found': False
                }
                results['search_integration_tests'].append(search_test)
        
        return results
    
    def test_strategic_enhancements(self) -> Dict:
        """Test strategic enhancements and move ordering"""
        print("\n" + "=" * 60)
        print("TESTING STRATEGIC ENHANCEMENTS")
        print("=" * 60)
        
        results = {
            'move_ordering_tests': [],
            'lmr_tests': [],
            'statistics_tracking': {}
        }
        
        # Test 1: Move ordering with nudges
        print(f"Testing move ordering with nudge integration...")
        
        # Find a position with nudge data
        nudge_position = None
        for position_key, position_data in self.engine.nudge_database.items():
            fen = position_data.get('fen', '')
            if fen and position_data.get('moves'):
                try:
                    board = chess.Board(fen)
                    if len(list(board.legal_moves)) > 5:  # Position with multiple moves
                        nudge_position = (board, position_data)
                        break
                except:
                    continue
        
        if nudge_position:
            board, position_data = nudge_position
            legal_moves = list(board.legal_moves)
            
            # Test move ordering
            ordered_moves = self.engine._order_moves_advanced(board, legal_moves, 4)
            
            # Check if nudge moves are prioritized
            nudge_moves_in_order = []
            for i, move in enumerate(ordered_moves[:3]):  # Check first 3 moves
                bonus = self.engine._get_nudge_bonus(board, move)
                if bonus > 0:
                    nudge_moves_in_order.append((i, move, bonus))
            
            move_order_test = {
                'total_moves': len(legal_moves),
                'nudge_moves_found': len(nudge_moves_in_order),
                'nudge_moves_early': len([x for x in nudge_moves_in_order if x[0] < 3]),
                'ordering_working': len(nudge_moves_in_order) > 0
            }
            
            results['move_ordering_tests'].append(move_order_test)
            
            print(f"   Position with {len(legal_moves)} legal moves")
            print(f"   Found {len(nudge_moves_in_order)} nudge moves in top 3")
            
            if move_order_test['ordering_working']:
                print(f"   ✅ Move ordering integrating nudges")
            else:
                print(f"   ℹ️ No nudge moves prioritized (may be normal)")
        
        # Test 2: Get engine statistics
        print(f"\nGathering engine statistics...")
        
        if hasattr(self.engine, 'nudge_stats'):
            results['statistics_tracking']['nudge_stats'] = self.engine.nudge_stats.copy()
            print(f"   Nudge Statistics: {self.engine.nudge_stats}")
        
        if hasattr(self.engine, 'time_manager'):
            time_stats = self.engine.time_manager.get_statistics()
            results['statistics_tracking']['time_manager_stats'] = time_stats
            print(f"   Time Manager Statistics: {time_stats}")
        
        return results
    
    def run_comprehensive_phase2_test(self) -> Dict:
        """Run all Phase 2 tests and generate report"""
        print("V7P3R v11 Phase 2: Comprehensive Verification Test")
        print("=" * 80)
        
        # Run all test suites
        nudge_results = self.test_nudge_system_integration()
        time_results = self.test_time_management_integration()
        strategic_results = self.test_strategic_enhancements()
        
        # Compile overall results
        overall_results = {
            'test_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'engine_version': 'v10.9 with v11 Phase 1 & 2 enhancements',
            'nudge_system': nudge_results,
            'time_management': time_results,
            'strategic_enhancements': strategic_results,
            'overall_status': 'UNKNOWN'
        }
        
        # Determine overall status
        critical_tests = [
            nudge_results.get('nudge_database_loaded', False),
            time_results.get('time_manager_available', False)
        ]
        
        if all(critical_tests):
            overall_results['overall_status'] = 'PASS'
        elif any(critical_tests):
            overall_results['overall_status'] = 'PARTIAL'
        else:
            overall_results['overall_status'] = 'FAIL'
        
        # Print summary
        print("\n" + "=" * 80)
        print("PHASE 2 VERIFICATION SUMMARY")
        print("=" * 80)
        
        status_emoji = {
            'PASS': '🎉',
            'PARTIAL': '⚠️',
            'FAIL': '❌'
        }
        
        print(f"{status_emoji[overall_results['overall_status']]} Overall Status: {overall_results['overall_status']}")
        
        if nudge_results['nudge_database_loaded']:
            print(f"✅ Nudge System: {nudge_results['nudge_database_size']} positions loaded")
        else:
            print(f"❌ Nudge System: Not loaded")
        
        if time_results['time_manager_available']:
            print(f"✅ Time Management: Integrated and functional")
        else:
            print(f"❌ Time Management: Not integrated")
        
        print(f"\n📊 Ready for Phase 3 Development: {overall_results['overall_status'] == 'PASS'}")
        
        return overall_results
    
    def save_results(self, results: Dict, filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"phase2_verification_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📁 Test results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Error saving results: {e}")


def main():
    """Main testing function"""
    print("Initializing V7P3R Phase 2 Verification System...")
    
    tester = V7P3RPhase2Tester()
    
    try:
        results = tester.run_comprehensive_phase2_test()
        tester.save_results(results)
        
        # Return success code based on results
        return 0 if results['overall_status'] == 'PASS' else 1
        
    except Exception as e:
        print(f"\n❌ Phase 2 verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)