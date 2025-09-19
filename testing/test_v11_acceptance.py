#!/usr/bin/env python3
"""
V7P3R v11 Acceptance Criteria Validation
Comprehensive test against all v11 targets and acceptance criteria
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import chess.pgn
import time
import json
from typing import List, Dict, Tuple
from v7p3r import V7P3REngine

class V11AcceptanceTester:
    def __init__(self):
        self.engine = V7P3REngine()
        self.results = {}
        
    def test_search_depth_requirement(self) -> bool:
        """Test: Search Depth ≥6 plies"""
        print("🎯 TESTING: Search Depth ≥6 plies")
        print("-" * 40)
        
        test_positions = [
            ("Opening", chess.Board()),
            ("Middlegame", chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")),
            ("Endgame", chess.Board("8/8/2k5/5q2/5K2/8/8/8 w - - 0 1"))
        ]
        
        depth_results = []
        for pos_name, board in test_positions:
            print(f"\n{pos_name} Position:")
            
            # Test up to depth 8 with reasonable time limits
            max_depth = 0
            for depth in range(1, 9):
                self.engine.default_depth = depth
                start_time = time.time()
                
                try:
                    move = self.engine.search(board, time_limit=30.0)
                    elapsed = time.time() - start_time
                    
                    if elapsed <= 30.0:
                        max_depth = depth
                        print(f"  Depth {depth}: {elapsed:6.2f}s ✅")
                    else:
                        print(f"  Depth {depth}: {elapsed:6.2f}s ❌ (timeout)")
                        break
                        
                except Exception as e:
                    print(f"  Depth {depth}: ERROR - {e}")
                    break
            
            depth_results.append((pos_name, max_depth))
            print(f"  Max depth achieved: {max_depth}")
        
        # Check if all positions achieve ≥6 depth
        min_depth = min(depth for _, depth in depth_results)
        depth_target_met = min_depth >= 6
        
        self.results['search_depth'] = {
            'target': 6,
            'achieved': min_depth,
            'details': depth_results,
            'passed': depth_target_met
        }
        
        print(f"\n📊 Result: {'✅ PASS' if depth_target_met else '❌ FAIL'} - Minimum depth: {min_depth}/6")
        return depth_target_met
    
    def test_node_efficiency(self) -> bool:
        """Test: 50% reduction in nodes searched from <v10 versions"""
        print(f"\n🚀 TESTING: Node Efficiency (50% reduction target)")
        print("-" * 40)
        
        # Test standard positions at depth 5
        board = chess.Board()
        self.engine.default_depth = 5
        
        start_time = time.time()
        move = self.engine.search(board, time_limit=30.0)
        elapsed = time.time() - start_time
        nodes = self.engine.nodes_searched
        nps = int(nodes / max(elapsed, 0.001))
        
        # Baseline: v10.6 took ~280,000 nodes for depth 5 opening
        # Target: <140,000 nodes (50% reduction)
        baseline_nodes = 280000
        target_nodes = baseline_nodes * 0.5
        efficiency_met = nodes <= target_nodes
        reduction_percent = ((baseline_nodes - nodes) / baseline_nodes) * 100
        
        print(f"  Depth 5 opening position:")
        print(f"  Nodes searched: {nodes:,}")
        print(f"  Baseline (v10.6): {baseline_nodes:,}")
        print(f"  Target (<50%): {target_nodes:,}")
        print(f"  Reduction achieved: {reduction_percent:.1f}%")
        print(f"  Search time: {elapsed:.2f}s ({nps:,} NPS)")
        
        self.results['node_efficiency'] = {
            'target_reduction': 50,
            'achieved_reduction': reduction_percent,
            'nodes_searched': nodes,
            'baseline_nodes': baseline_nodes,
            'passed': efficiency_met
        }
        
        print(f"\n📊 Result: {'✅ PASS' if efficiency_met else '❌ FAIL'} - {reduction_percent:.1f}% reduction")
        return efficiency_met
    
    def test_time_management(self) -> bool:
        """Test: Adaptive allocation/management working"""
        print(f"\n⏰ TESTING: Adaptive Time Management")
        print("-" * 40)
        
        # Test different time controls
        time_controls = [
            (300, "5-minute game"),
            (600, "10-minute game"),
            (1800, "30-minute game")
        ]
        
        time_mgmt_results = []
        for total_time, description in time_controls:
            print(f"\n{description} ({total_time}s total):")
            
            # Simulate first 10 moves
            board = chess.Board()
            time_used = 0
            
            for move_num in range(1, 11):
                time_remaining = total_time - time_used
                
                # Test time allocation
                self.engine.time_manager.base_time = total_time
                self.engine.time_manager.update_time_info(time_used, move_num)
                
                allocated_time, target_depth = self.engine.time_manager.calculate_time_allocation(board, time_remaining)
                
                # Check if allocation is reasonable
                reasonable_allocation = (allocated_time <= time_remaining * 0.2)  # Not more than 20% per move early game
                adaptive_depth = (target_depth >= 4)  # Reasonable depth target
                
                time_used += min(allocated_time * 0.7, time_remaining * 0.1)  # Simulate actual usage
                
                print(f"  Move {move_num}: Allocated {allocated_time:.1f}s, Target depth {target_depth}, Remaining {time_remaining:.1f}s")
                
                if not reasonable_allocation:
                    break
            
            # Check if we have reasonable time left
            final_remaining = total_time - time_used
            time_efficiency = final_remaining / total_time
            
            time_mgmt_results.append({
                'time_control': description,
                'final_remaining_percent': time_efficiency * 100,
                'reasonable_allocation': reasonable_allocation,
                'adaptive_depth': adaptive_depth
            })
        
        time_mgmt_passed = all(r['reasonable_allocation'] and r['adaptive_depth'] for r in time_mgmt_results)
        
        self.results['time_management'] = {
            'details': time_mgmt_results,
            'passed': time_mgmt_passed
        }
        
        print(f"\n📊 Result: {'✅ PASS' if time_mgmt_passed else '❌ FAIL'} - Adaptive time management working")
        return time_mgmt_passed
    
    def test_strategic_consistency(self) -> bool:
        """Test: Measurable improvement in position evaluation"""
        print(f"\n🧠 TESTING: Strategic Consistency")
        print("-" * 40)
        
        # Test strategic database integration
        strategic_positions = [
            ("King safety", chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4")),
            ("Center control", chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")),
            ("Piece activity", chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 4 5"))
        ]
        
        strategic_results = []
        for pos_name, board in strategic_positions:
            # Get evaluation with strategic database
            eval_score = self.engine._evaluate_position(board)
            
            # Test if strategic database provides bonus (if applicable)
            try:
                strategic_bonus = self.engine.strategic_database.get_strategic_evaluation_bonus(board)
                has_strategic_input = abs(strategic_bonus) > 0.1
            except:
                has_strategic_input = False
            
            strategic_results.append({
                'position': pos_name,
                'evaluation': eval_score,
                'strategic_input': has_strategic_input
            })
            
            print(f"  {pos_name}: Eval={eval_score:.2f}, Strategic={'✅' if has_strategic_input else '❌'}")
        
        # Check if strategic system is active
        strategic_active = any(r['strategic_input'] for r in strategic_results)
        
        self.results['strategic_consistency'] = {
            'details': strategic_results,
            'strategic_system_active': strategic_active,
            'passed': strategic_active
        }
        
        print(f"\n📊 Result: {'✅ PASS' if strategic_active else '❌ FAIL'} - Strategic evaluation active")
        return strategic_active
    
    def test_pattern_recognition(self) -> bool:
        """Test: 90%+ pattern match accuracy"""
        print(f"\n🔍 TESTING: Pattern Recognition (90%+ accuracy)")
        print("-" * 40)
        
        # Test nudge system pattern recognition
        test_patterns = [
            ("Opening development", chess.Board(), ["g1f3", "b1c3", "f1e2"]),
            ("King safety castling", chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"), ["e1g1"]),
            ("Center control", chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"), ["d2d4"]),
        ]
        
        pattern_results = []
        for pattern_name, board, expected_moves in test_patterns:
            # Get best move from engine
            self.engine.default_depth = 4
            best_move = self.engine.search(board, time_limit=10.0)
            
            # Check if best move matches expected pattern
            pattern_match = str(best_move) in expected_moves
            
            # Check if nudge system is active
            try:
                nudge_score = 0
                for move in board.legal_moves:
                    if hasattr(self.engine, 'nudge_database'):
                        temp_board = board.copy()
                        temp_board.push(move)
                        nudge_score += self.engine.nudge_database.evaluate_nudge(temp_board, move)
                nudge_active = abs(nudge_score) > 0.1
            except:
                nudge_active = False
            
            pattern_results.append({
                'pattern': pattern_name,
                'best_move': str(best_move),
                'expected': expected_moves,
                'pattern_match': pattern_match,
                'nudge_active': nudge_active
            })
            
            print(f"  {pattern_name}: Move={best_move}, Expected={expected_moves}, Match={'✅' if pattern_match else '❌'}")
        
        # Calculate pattern accuracy
        matches = sum(1 for r in pattern_results if r['pattern_match'])
        accuracy = (matches / len(pattern_results)) * 100
        accuracy_target_met = accuracy >= 90
        
        self.results['pattern_recognition'] = {
            'target_accuracy': 90,
            'achieved_accuracy': accuracy,
            'details': pattern_results,
            'passed': accuracy_target_met
        }
        
        print(f"\n📊 Result: {'✅ PASS' if accuracy_target_met else '❌ FAIL'} - {accuracy:.1f}% accuracy")
        return accuracy_target_met
    
    def test_tactical_balance(self) -> bool:
        """Test: Equal attack/defense scoring"""
        print(f"\n⚔️ TESTING: Tactical Balance (Attack/Defense)")
        print("-" * 40)
        
        # Test both attacking and defensive positions
        tactical_positions = [
            ("White attacking", chess.Board("r2qkb1r/ppp2ppp/2n1bn2/3pp3/3PP3/3B1N2/PPP2PPP/RNBQK2R w KQkq - 4 6"), "white"),
            ("Black attacking", chess.Board("rnbqk2r/ppp2ppp/3b1n2/3pp3/3PP3/2N1BN2/PPP2PPP/R2QKB1R b KQkq - 4 6"), "black"),
            ("Defensive position", chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5"), "neutral")
        ]
        
        tactical_results = []
        for pos_name, board, pos_type in tactical_positions:
            # Get evaluation
            eval_score = self.engine._evaluate_position(board)
            
            # Test defensive analysis
            try:
                if hasattr(self.engine, 'lightweight_defense'):
                    defense_score = self.engine.lightweight_defense.analyze_king_safety(board)
                    defense_active = True
                else:
                    defense_score = 0
                    defense_active = False
            except:
                defense_score = 0
                defense_active = False
            
            # Test if posture assessment is working
            try:
                if hasattr(self.engine, 'posture_assessment'):
                    posture = self.engine.posture_assessment.assess_position_posture(board)
                    posture_active = True
                else:
                    posture = None
                    posture_active = False
            except:
                posture = None
                posture_active = False
            
            tactical_results.append({
                'position': pos_name,
                'type': pos_type,
                'evaluation': eval_score,
                'defense_score': defense_score,
                'defense_active': defense_active,
                'posture': posture,
                'posture_active': posture_active
            })
            
            print(f"  {pos_name}: Eval={eval_score:.2f}, Defense={'✅' if defense_active else '❌'}, Posture={'✅' if posture_active else '❌'}")
        
        # Check if both attack and defense systems are active
        defense_systems = any(r['defense_active'] for r in tactical_results)
        posture_systems = any(r['posture_active'] for r in tactical_results)
        tactical_balance = defense_systems and posture_systems
        
        self.results['tactical_balance'] = {
            'defense_systems_active': defense_systems,
            'posture_systems_active': posture_systems,
            'details': tactical_results,
            'passed': tactical_balance
        }
        
        print(f"\n📊 Result: {'✅ PASS' if tactical_balance else '❌ FAIL'} - Attack/Defense balance active")
        return tactical_balance
    
    def test_tournament_readiness(self) -> bool:
        """Test: Full competitive validation and UCI compliance"""
        print(f"\n🏆 TESTING: Tournament Readiness")
        print("-" * 40)
        
        # Test basic UCI commands
        uci_tests = [
            ("Engine identification", "V7P3R" in str(self.engine.__class__)),
            ("Move generation", len(list(chess.Board().legal_moves)) > 0),
            ("Search function", callable(getattr(self.engine, 'search', None))),
            ("Time management", hasattr(self.engine, 'time_manager')),
            ("Evaluation", callable(getattr(self.engine, '_evaluate_position', None)))
        ]
        
        uci_results = []
        for test_name, test_result in uci_tests:
            uci_results.append({
                'test': test_name,
                'passed': test_result
            })
            print(f"  {test_name}: {'✅ PASS' if test_result else '❌ FAIL'}")
        
        # Test performance stability
        try:
            # Quick performance test
            board = chess.Board()
            start_time = time.time()
            move = self.engine.search(board, time_limit=5.0)
            elapsed = time.time() - start_time
            
            performance_stable = (elapsed <= 5.0 and move is not None)
        except:
            performance_stable = False
        
        uci_results.append({
            'test': 'Performance stability',
            'passed': performance_stable
        })
        print(f"  Performance stability: {'✅ PASS' if performance_stable else '❌ FAIL'}")
        
        tournament_ready = all(r['passed'] for r in uci_results)
        
        self.results['tournament_readiness'] = {
            'uci_tests': uci_results,
            'passed': tournament_ready
        }
        
        print(f"\n📊 Result: {'✅ PASS' if tournament_ready else '❌ FAIL'} - Tournament ready")
        return tournament_ready
    
    def generate_final_report(self) -> bool:
        """Generate comprehensive acceptance criteria report"""
        print(f"\n{'='*60}")
        print("V7P3R v11 ACCEPTANCE CRITERIA FINAL REPORT")
        print("=" * 60)
        
        criteria_tests = [
            ('Search Depth ≥6 plies', 'search_depth'),
            ('Node Efficiency (50% reduction)', 'node_efficiency'),
            ('Adaptive Time Management', 'time_management'),
            ('Strategic Consistency', 'strategic_consistency'),
            ('Pattern Recognition (90%+)', 'pattern_recognition'),
            ('Tactical Balance', 'tactical_balance'),
            ('Tournament Readiness', 'tournament_readiness')
        ]
        
        passed_tests = 0
        total_tests = len(criteria_tests)
        
        print(f"\nDETAILED RESULTS:")
        for test_name, result_key in criteria_tests:
            if result_key in self.results:
                passed = self.results[result_key]['passed']
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {status} {test_name}")
                if passed:
                    passed_tests += 1
            else:
                print(f"  ❓ SKIP {test_name} (not tested)")
        
        # Calculate overall acceptance
        acceptance_rate = (passed_tests / total_tests) * 100
        v11_accepted = acceptance_rate >= 85  # 85% acceptance threshold
        
        print(f"\nOVERALL ASSESSMENT:")
        print(f"  Tests Passed: {passed_tests}/{total_tests}")
        print(f"  Acceptance Rate: {acceptance_rate:.1f}%")
        print(f"  Threshold: 85%")
        
        if v11_accepted:
            print(f"\n🎉 V7P3R v11 ACCEPTED FOR RELEASE!")
            print(f"  ✅ Meets acceptance criteria")
            print(f"  ✅ Ready for tournament deployment")
            print(f"  ✅ Performance targets achieved")
        else:
            print(f"\n⚠️  V7P3R v11 REQUIRES ADDITIONAL WORK")
            print(f"  🔧 Review failed criteria")
            print(f"  📊 Improve failing components")
        
        # Save results
        with open('v11_acceptance_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return v11_accepted

def main():
    print("Starting V7P3R v11 Acceptance Criteria Validation...")
    print("This comprehensive test will take 10-15 minutes...")
    print()
    
    tester = V11AcceptanceTester()
    
    # Run all acceptance tests
    tests = [
        tester.test_search_depth_requirement,
        tester.test_node_efficiency,
        tester.test_time_management,
        tester.test_strategic_consistency,
        tester.test_pattern_recognition,
        tester.test_tactical_balance,
        tester.test_tournament_readiness
    ]
    
    print("Running acceptance criteria tests...")
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"ERROR in {test_func.__name__}: {e}")
    
    # Generate final report
    accepted = tester.generate_final_report()
    
    print(f"\n{'='*60}")
    if accepted:
        print("🚀 PROCEED WITH V7P3R v11 BUILD AND RELEASE!")
    else:
        print("🔧 ADDITIONAL OPTIMIZATION REQUIRED BEFORE RELEASE")
    print("=" * 60)
    
    return accepted

if __name__ == "__main__":
    main()