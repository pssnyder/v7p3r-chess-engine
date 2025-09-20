#!/usr/bin/env python3
"""
V7P3R v11.1 Puzzle Regression Test
Specifically tests positions where v11 failed
"""

import chess
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from v7p3r_v11_1_simplified import V7P3REngineSimple
    print("✅ Successfully imported V7P3R v11.1 simplified engine")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


class PuzzleRegressionTester:
    """Test v11.1 against specific puzzles where v11 failed"""
    
    def __init__(self):
        self.engine = V7P3REngineSimple()
        
        # Known problematic puzzles from v11 analysis
        self.regression_puzzles = [
            {
                "id": "regression_001",
                "name": "Basic fork tactic",
                "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R b KQkq - 0 3",
                "expected_moves": ["Ng4", "Ne4"],  # Knight fork ideas
                "themes": ["fork", "tactics"],
                "rating": 1200,
                "v11_failed": True,
                "description": "Knight fork opportunity - v11 missed this"
            },
            {
                "id": "regression_002", 
                "name": "Pin break",
                "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
                "expected_moves": ["Bg5", "Bd5", "h3"],
                "themes": ["pin", "tactics"],
                "rating": 1350,
                "v11_failed": True,
                "description": "Breaking pin or creating counter-pin"
            },
            {
                "id": "regression_003",
                "name": "Mate in 2",
                "fen": "r1bqkb1r/pppp1p1p/2n2np1/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
                "expected_moves": ["Bxf7", "Qd5"],  # Forcing moves
                "themes": ["mate", "attack"],
                "rating": 1500,
                "v11_failed": True,
                "description": "Forced mate sequence - v11 played defensively"
            },
            {
                "id": "regression_004",
                "name": "Tactical deflection",
                "fen": "r2qk2r/ppp2ppp/2n1bn2/3pp3/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 7",
                "expected_moves": ["d5", "cxd5", "Nd5"],
                "themes": ["deflection", "tactics"],
                "rating": 1400,
                "v11_failed": True,
                "description": "Deflection to win material"
            },
            {
                "id": "regression_005",
                "name": "Discovery attack",
                "fen": "r1bqk2r/pppp1ppp/2nb1n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 6",
                "expected_moves": ["Nd5", "Nxe5"],
                "themes": ["discovery", "tactics"],
                "rating": 1300,
                "v11_failed": True,
                "description": "Discovery attack with knight"
            },
            {
                "id": "regression_006",
                "name": "Endgame conversion",
                "fen": "8/8/8/3k4/3P4/3K4/8/8 w - - 0 1",
                "expected_moves": ["Kc4", "Ke4", "Kc3"],  # King activation
                "themes": ["endgame", "technique"],
                "rating": 1100,
                "v11_failed": True,
                "description": "Basic pawn endgame technique"
            },
            {
                "id": "regression_007",
                "name": "Time pressure blunder",
                "fen": "r4rk1/pp3ppp/2n1b3/q1pp2B1/8/P1Q2NP1/1PP1PP1P/2KR3R w - - 0 15",
                "expected_moves": ["Qc7", "Bh6", "Nh4"],
                "themes": ["tactics", "time"],
                "rating": 1600,
                "v11_failed": True,
                "description": "Tactical shot under time pressure"
            },
            {
                "id": "regression_008",
                "name": "Positional sacrifice",
                "fen": "r1bq1rk1/pp1n1ppp/2pb1n2/3p4/2PP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 9",
                "expected_moves": ["Bxh7", "e4", "h3"],
                "themes": ["sacrifice", "attack"],
                "rating": 1700,
                "v11_failed": True,
                "description": "Positional piece sacrifice for attack"
            }
        ]
        
        # Time control tests (v11 had time management issues)
        self.time_pressure_tests = [
            {"time_limit": 0.5, "name": "Ultra-bullet"},
            {"time_limit": 1.0, "name": "Bullet"}, 
            {"time_limit": 2.0, "name": "Blitz"},
            {"time_limit": 5.0, "name": "Rapid"}
        ]
    
    def test_regression_puzzles(self):
        """Test specific puzzles where v11 failed"""
        print("\n🔍 REGRESSION PUZZLE TESTS")
        print("=" * 60)
        print("Testing positions where V7P3R v11 specifically failed")
        print("=" * 60)
        
        results = []
        
        for puzzle in self.regression_puzzles:
            print(f"\n🧩 Testing: {puzzle['name']} (ID: {puzzle['id']})")
            print(f"Description: {puzzle['description']}")
            print(f"Themes: {', '.join(puzzle['themes'])}")
            print(f"Rating: {puzzle['rating']}")
            print(f"FEN: {puzzle['fen']}")
            print(f"Expected moves: {', '.join(puzzle['expected_moves'])}")
            
            board = chess.Board(puzzle['fen'])
            
            # Test with different time controls
            for time_control in [2.0, 5.0, 10.0]:
                print(f"\n  ⏱️  Testing with {time_control:.1f}s time limit:")
                
                start_time = time.time()
                try:
                    # Reset engine state
                    self.engine.new_game()
                    
                    engine_move = self.engine.search(board, time_limit=time_control)
                    search_time = time.time() - start_time
                    
                    move_uci = engine_move.uci() if engine_move else "null"
                    nodes = self.engine.search_stats.get('nodes_searched', 0)
                    nps = nodes / search_time if search_time > 0 else 0
                    
                    # Check if move matches expected good moves
                    move_san = board.san(engine_move) if engine_move and engine_move in board.legal_moves else "ILLEGAL"
                    
                    found_good_move = any(
                        move_uci.startswith(expected.lower()) or move_san.startswith(expected)
                        for expected in puzzle['expected_moves']
                    )
                    
                    is_legal = engine_move in board.legal_moves if engine_move else False
                    
                    result = {
                        'puzzle_id': puzzle['id'],
                        'puzzle_name': puzzle['name'],
                        'time_control': time_control,
                        'move_uci': move_uci,
                        'move_san': move_san,
                        'is_legal': is_legal,
                        'found_good_move': found_good_move,
                        'expected_moves': puzzle['expected_moves'],
                        'search_time': search_time,
                        'nodes': nodes,
                        'nps': nps,
                        'themes': puzzle['themes'],
                        'rating': puzzle['rating'],
                        'v11_failed': puzzle['v11_failed']
                    }
                    
                    results.append(result)
                    
                    # Status indicators
                    if found_good_move:
                        status = "✅ FOUND GOOD MOVE"
                    elif is_legal:
                        status = "⚠️  LEGAL BUT SUBOPTIMAL"
                    else:
                        status = "❌ ILLEGAL MOVE"
                    
                    time_status = "⚡ FAST" if search_time < time_control * 0.8 else "⏱️  NORMAL"
                    
                    print(f"    Result: {status}")
                    print(f"    Move: {move_san} ({move_uci})")
                    print(f"    Time: {search_time:.2f}s ({time_status})")
                    print(f"    Performance: {nps:.0f} NPS")
                    
                except Exception as e:
                    result = {
                        'puzzle_id': puzzle['id'],
                        'puzzle_name': puzzle['name'],
                        'time_control': time_control,
                        'error': str(e),
                        'is_legal': False,
                        'found_good_move': False,
                        'search_time': time.time() - start_time,
                        'v11_failed': puzzle['v11_failed']
                    }
                    results.append(result)
                    print(f"    Result: ❌ ERROR - {e}")
        
        return results
    
    def test_time_pressure_scenarios(self):
        """Test engine under extreme time pressure"""
        print("\n⚡ TIME PRESSURE REGRESSION TESTS")
        print("=" * 60)
        print("Testing time management improvements over v11")
        print("=" * 60)
        
        # Use a subset of regression puzzles for time pressure
        time_test_puzzles = self.regression_puzzles[:4]  # First 4 puzzles
        
        results = []
        
        for time_test in self.time_pressure_tests:
            print(f"\n⏱️  Testing {time_test['name']} time control ({time_test['time_limit']:.1f}s)")
            
            time_results = []
            
            for puzzle in time_test_puzzles:
                print(f"  🧩 {puzzle['name']}")
                
                board = chess.Board(puzzle['fen'])
                
                start_time = time.time()
                try:
                    self.engine.new_game()
                    
                    engine_move = self.engine.search(board, time_limit=time_test['time_limit'])
                    search_time = time.time() - start_time
                    
                    move_uci = engine_move.uci() if engine_move else "null"
                    nodes = self.engine.search_stats.get('nodes_searched', 0)
                    
                    # Time compliance (allow 25% buffer for very fast time controls)
                    time_buffer = time_test['time_limit'] * 1.25
                    time_compliant = search_time <= time_buffer
                    
                    is_legal = engine_move in board.legal_moves if engine_move else False
                    
                    found_good_move = any(
                        move_uci.startswith(expected.lower()) 
                        for expected in puzzle['expected_moves']
                    ) if is_legal else False
                    
                    result = {
                        'puzzle_name': puzzle['name'],
                        'time_limit': time_test['time_limit'],
                        'search_time': search_time,
                        'time_compliant': time_compliant,
                        'is_legal': is_legal,
                        'found_good_move': found_good_move,
                        'move': move_uci,
                        'nodes': nodes,
                        'nps': nodes / search_time if search_time > 0 else 0
                    }
                    
                    time_results.append(result)
                    
                    # Status
                    move_status = "✅ GOOD" if found_good_move else "⚠️ OK" if is_legal else "❌ BAD"
                    time_status = "✅ ON TIME" if time_compliant else "❌ OVERTIME"
                    
                    print(f"    Move: {move_status} | Time: {time_status}")
                    print(f"    {move_uci} in {search_time:.2f}s")
                    
                except Exception as e:
                    result = {
                        'puzzle_name': puzzle['name'],
                        'time_limit': time_test['time_limit'],
                        'error': str(e),
                        'time_compliant': False,
                        'is_legal': False,
                        'found_good_move': False
                    }
                    time_results.append(result)
                    print(f"    Result: ❌ ERROR - {e}")
            
            # Calculate time control statistics
            legal_moves = sum(1 for r in time_results if r.get('is_legal', False))
            good_moves = sum(1 for r in time_results if r.get('found_good_move', False))
            time_compliant = sum(1 for r in time_results if r.get('time_compliant', False))
            avg_time = sum(r.get('search_time', 0) for r in time_results) / len(time_results)
            avg_nps = sum(r.get('nps', 0) for r in time_results) / len(time_results)
            
            summary = {
                'time_control': time_test['name'],
                'time_limit': time_test['time_limit'],
                'puzzles_tested': len(time_results),
                'legal_moves': legal_moves,
                'good_moves': good_moves,
                'time_compliant': time_compliant,
                'legal_rate': legal_moves / len(time_results) * 100,
                'good_move_rate': good_moves / len(time_results) * 100,
                'time_compliance_rate': time_compliant / len(time_results) * 100,
                'average_time': avg_time,
                'average_nps': avg_nps,
                'individual_results': time_results
            }
            
            results.append(summary)
            
            print(f"\n  📊 {time_test['name']} Summary:")
            print(f"    Legal moves: {legal_moves}/{len(time_results)} ({legal_moves/len(time_results)*100:.1f}%)")
            print(f"    Good moves: {good_moves}/{len(time_results)} ({good_moves/len(time_results)*100:.1f}%)")
            print(f"    Time compliance: {time_compliant}/{len(time_results)} ({time_compliant/len(time_results)*100:.1f}%)")
            print(f"    Average time: {avg_time:.2f}s")
            print(f"    Average NPS: {avg_nps:.0f}")
        
        return results
    
    def compare_with_v11_baseline(self, results):
        """Compare v11.1 results with known v11 failures"""
        print("\n📊 V11.1 vs V11 COMPARISON")
        print("=" * 60)
        
        # Extract regression puzzle results (longest time control for best comparison)
        regression_results = [r for r in results['regression_puzzles'] if r.get('time_control') == 10.0]
        
        # Count improvements
        total_puzzles = len(self.regression_puzzles)
        puzzles_improved = 0
        
        for puzzle in self.regression_puzzles:
            puzzle_results = [r for r in regression_results if r['puzzle_id'] == puzzle['id']]
            
            if puzzle_results:
                result = puzzle_results[0]
                if result.get('found_good_move', False):
                    puzzles_improved += 1
                    print(f"✅ IMPROVED: {puzzle['name']} - v11.1 found good move")
                elif result.get('is_legal', False):
                    print(f"⚠️  PARTIAL: {puzzle['name']} - v11.1 legal but suboptimal")
                else:
                    print(f"❌ STILL FAILING: {puzzle['name']} - v11.1 still struggles")
            else:
                print(f"❓ NO DATA: {puzzle['name']} - no test results")
        
        improvement_rate = puzzles_improved / total_puzzles * 100
        
        print(f"\n🎯 IMPROVEMENT SUMMARY:")
        print(f"Puzzles improved: {puzzles_improved}/{total_puzzles} ({improvement_rate:.1f}%)")
        
        if improvement_rate >= 75:
            print("🎉 EXCELLENT: Major improvement over v11!")
        elif improvement_rate >= 50:
            print("✅ GOOD: Significant improvement over v11")
        elif improvement_rate >= 25:
            print("⚠️  PARTIAL: Some improvement over v11")
        else:
            print("❌ MINIMAL: Little improvement over v11")
        
        # Time management comparison
        time_results = results.get('time_pressure', [])
        if time_results:
            avg_compliance = sum(r.get('time_compliance_rate', 0) for r in time_results) / len(time_results)
            print(f"\n⏱️  TIME MANAGEMENT:")
            print(f"Average time compliance: {avg_compliance:.1f}%")
            
            if avg_compliance >= 90:
                print("🎉 EXCELLENT: Time management greatly improved!")
            elif avg_compliance >= 75:
                print("✅ GOOD: Time management improved")
            else:
                print("⚠️  NEEDS WORK: Time management still problematic")
        
        return {
            'puzzles_improved': puzzles_improved,
            'total_puzzles': total_puzzles,
            'improvement_rate': improvement_rate,
            'time_compliance': avg_compliance if time_results else None
        }
    
    def run_regression_test_suite(self):
        """Run complete regression test suite"""
        print("🔍 V7P3R v11.1 REGRESSION TEST SUITE")
        print("=" * 80)
        print("Testing improvements over V7P3R v11 known failures")
        print("=" * 80)
        
        all_results = {}
        
        # Run regression tests
        all_results['regression_puzzles'] = self.test_regression_puzzles()
        all_results['time_pressure'] = self.test_time_pressure_scenarios()
        
        # Compare with v11 baseline
        comparison = self.compare_with_v11_baseline(all_results)
        all_results['v11_comparison'] = comparison
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"v7p3r_v11_1_regression_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n💾 Regression test results saved to: {filename}")
        
        return all_results


def main():
    """Main execution"""
    tester = PuzzleRegressionTester()
    results = tester.run_regression_test_suite()
    return results


if __name__ == "__main__":
    main()