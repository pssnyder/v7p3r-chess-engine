#!/usr/bin/env python3
"""
V7P3R v11.1 Comprehensive Test Suite
Direct-to-code testing without executable build
"""

import chess
import sys
import os
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from v7p3r_v11_1_simplified import V7P3REngineSimple
    print("✅ Successfully imported V7P3R v11.1 simplified engine")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


@dataclass
class TestResult:
    """Single test result"""
    test_name: str
    success: bool
    move: Optional[str]
    time_taken: float
    nodes_searched: int
    evaluation: Optional[float]
    error: Optional[str] = None
    details: Optional[Dict] = None


@dataclass
class TestSuite:
    """Test suite results"""
    suite_name: str
    tests: List[TestResult]
    start_time: datetime
    end_time: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_time: float
    total_nodes: int
    average_nps: float


class V7P3RComprehensiveTester:
    """Comprehensive testing suite for V7P3R v11.1"""
    
    def __init__(self):
        self.engine = V7P3REngineSimple()
        self.results = []
        
        # Test positions database
        self.tactical_positions = [
            {
                "name": "Mate in 1",
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
                "expected_theme": "mate",
                "best_moves": ["Qxf7"],
                "avoid_moves": ["Qh4", "Bd5"]
            },
            {
                "name": "Fork tactic",
                "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R b KQkq - 0 3",
                "expected_theme": "fork",
                "best_moves": ["Ng4", "Ne4"],
                "avoid_moves": ["d6", "Bc5"]
            },
            {
                "name": "Pin tactic",
                "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
                "expected_theme": "pin",
                "best_moves": ["Bg5", "Bd5"],
                "avoid_moves": ["h3", "a3"]
            },
            {
                "name": "Discovery attack",
                "fen": "r1bqk2r/pppp1ppp/2nb1n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 6",
                "expected_theme": "discovery",
                "best_moves": ["Nd5", "Nxe5"],
                "avoid_moves": ["Re1", "h3"]
            },
            {
                "name": "Deflection",
                "fen": "r2qk2r/ppp2ppp/2n1bn2/3pp3/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 7",
                "expected_theme": "deflection",
                "best_moves": ["d5", "cxd5"],
                "avoid_moves": ["Be2", "h3"]
            }
        ]
        
        self.positional_positions = [
            {
                "name": "Opening development",
                "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                "expected_theme": "development",
                "good_moves": ["Nf6", "d6", "c5"],
                "poor_moves": ["h6", "a6", "f6"]
            },
            {
                "name": "Middlegame control",
                "fen": "r1bqk2r/pp2nppp/2n1p3/3pP3/1bpP4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 8",
                "expected_theme": "control",
                "good_moves": ["a3", "Bd2", "Qd2"],
                "poor_moves": ["h3", "g3", "Rb1"]
            },
            {
                "name": "Endgame technique",
                "fen": "8/8/3k4/3P4/3K4/8/8/8 w - - 0 1",
                "expected_theme": "endgame",
                "good_moves": ["Kc5", "Ke5"],
                "poor_moves": ["d6", "Kd3"]
            }
        ]
        
        self.time_pressure_positions = [
            {
                "name": "Quick tactical decision",
                "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
                "time_limit": 1.0
            },
            {
                "name": "Medium complexity",
                "fen": "r2qkb1r/pp2nppp/3p1n2/2pP4/4P3/2N2N2/PPP1BPPP/R1BQK2R w KQkq c6 0 8",
                "time_limit": 2.5
            },
            {
                "name": "Complex position",
                "fen": "r1b1k2r/1pqpbppp/p1n1pn2/6B1/2pPP3/2N2N2/PPP2PPP/R2QKB1R w KQkq - 0 10",
                "time_limit": 5.0
            }
        ]
        
        self.puzzle_positions = [
            {
                "name": "Puzzle 1200 rating",
                "fen": "r4rk1/pp3ppp/2n1b3/q1pp2B1/8/P1Q2NP1/1PP1PP1P/2KR3R w - - 0 15",
                "rating": 1200,
                "themes": ["tactics", "pin"]
            },
            {
                "name": "Puzzle 1500 rating",
                "fen": "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P3/3P1N2/PPP2PPP/RN1QK2R w KQ - 0 8",
                "rating": 1500,
                "themes": ["tactics", "discovery"]
            },
            {
                "name": "Puzzle 1800 rating",
                "fen": "r1bq1rk1/pp1n1ppp/2pb1n2/3p4/2PP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 9",
                "rating": 1800,
                "themes": ["positional", "development"]
            }
        ]
    
    def run_tactical_tests(self) -> TestSuite:
        """Test tactical position solving"""
        print("\n🎯 TACTICAL POSITION TESTS")
        print("=" * 50)
        
        start_time = datetime.now()
        tests = []
        
        for pos in self.tactical_positions:
            print(f"\nTesting: {pos['name']}")
            print(f"Position: {pos['fen']}")
            print(f"Expected theme: {pos['expected_theme']}")
            
            test_start = time.time()
            
            try:
                board = chess.Board(pos['fen'])
                move = self.engine.search(board, time_limit=3.0)
                test_time = time.time() - test_start
                
                move_uci = move.uci() if move else "null"
                
                # Check if move is in expected good moves
                success = any(move_uci.startswith(good_move.lower()) for good_move in pos['best_moves'])
                
                # Check if move is in avoided moves (this is bad)
                if any(move_uci.startswith(bad_move.lower()) for bad_move in pos['avoid_moves']):
                    success = False
                
                nodes = self.engine.search_stats.get('nodes_searched', 0)
                
                result = TestResult(
                    test_name=pos['name'],
                    success=success,
                    move=move_uci,
                    time_taken=test_time,
                    nodes_searched=nodes,
                    evaluation=None,
                    details={
                        'theme': pos['expected_theme'],
                        'expected': pos['best_moves'],
                        'avoided': pos['avoid_moves']
                    }
                )
                
                tests.append(result)
                
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"  Result: {status}")
                print(f"  Move: {move_uci}")
                print(f"  Time: {test_time:.2f}s")
                print(f"  Nodes: {nodes:,}")
                
            except Exception as e:
                result = TestResult(
                    test_name=pos['name'],
                    success=False,
                    move=None,
                    time_taken=time.time() - test_start,
                    nodes_searched=0,
                    evaluation=None,
                    error=str(e)
                )
                tests.append(result)
                print(f"  Result: ❌ ERROR - {e}")
        
        end_time = datetime.now()
        
        passed = sum(1 for t in tests if t.success)
        total_time = sum(t.time_taken for t in tests)
        total_nodes = sum(t.nodes_searched for t in tests)
        avg_nps = total_nodes / total_time if total_time > 0 else 0
        
        return TestSuite(
            suite_name="Tactical Tests",
            tests=tests,
            start_time=start_time,
            end_time=end_time,
            total_tests=len(tests),
            passed_tests=passed,
            failed_tests=len(tests) - passed,
            average_time=total_time / len(tests),
            total_nodes=total_nodes,
            average_nps=avg_nps
        )
    
    def run_positional_tests(self) -> TestSuite:
        """Test positional understanding"""
        print("\n🏰 POSITIONAL UNDERSTANDING TESTS")
        print("=" * 50)
        
        start_time = datetime.now()
        tests = []
        
        for pos in self.positional_positions:
            print(f"\nTesting: {pos['name']}")
            print(f"Position: {pos['fen']}")
            print(f"Expected theme: {pos['expected_theme']}")
            
            test_start = time.time()
            
            try:
                board = chess.Board(pos['fen'])
                move = self.engine.search(board, time_limit=4.0)
                test_time = time.time() - test_start
                
                move_uci = move.uci() if move else "null"
                
                # Check if move is positionally sound
                success = any(move_uci.startswith(good_move.lower()) for good_move in pos['good_moves'])
                
                # Penalize poor moves
                if any(move_uci.startswith(poor_move.lower()) for poor_move in pos['poor_moves']):
                    success = False
                
                nodes = self.engine.search_stats.get('nodes_searched', 0)
                
                result = TestResult(
                    test_name=pos['name'],
                    success=success,
                    move=move_uci,
                    time_taken=test_time,
                    nodes_searched=nodes,
                    evaluation=None,
                    details={
                        'theme': pos['expected_theme'],
                        'good_moves': pos['good_moves'],
                        'poor_moves': pos['poor_moves']
                    }
                )
                
                tests.append(result)
                
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"  Result: {status}")
                print(f"  Move: {move_uci}")
                print(f"  Time: {test_time:.2f}s")
                print(f"  Nodes: {nodes:,}")
                
            except Exception as e:
                result = TestResult(
                    test_name=pos['name'],
                    success=False,
                    move=None,
                    time_taken=time.time() - test_start,
                    nodes_searched=0,
                    evaluation=None,
                    error=str(e)
                )
                tests.append(result)
                print(f"  Result: ❌ ERROR - {e}")
        
        end_time = datetime.now()
        
        passed = sum(1 for t in tests if t.success)
        total_time = sum(t.time_taken for t in tests)
        total_nodes = sum(t.nodes_searched for t in tests)
        avg_nps = total_nodes / total_time if total_time > 0 else 0
        
        return TestSuite(
            suite_name="Positional Tests",
            tests=tests,
            start_time=start_time,
            end_time=end_time,
            total_tests=len(tests),
            passed_tests=passed,
            failed_tests=len(tests) - passed,
            average_time=total_time / len(tests),
            total_nodes=total_nodes,
            average_nps=avg_nps
        )
    
    def run_time_pressure_tests(self) -> TestSuite:
        """Test engine under various time pressures"""
        print("\n⏱️  TIME PRESSURE TESTS")
        print("=" * 50)
        
        start_time = datetime.now()
        tests = []
        
        for pos in self.time_pressure_positions:
            print(f"\nTesting: {pos['name']}")
            print(f"Position: {pos['fen']}")
            print(f"Time limit: {pos['time_limit']:.1f}s")
            
            test_start = time.time()
            
            try:
                board = chess.Board(pos['fen'])
                move = self.engine.search(board, time_limit=pos['time_limit'])
                test_time = time.time() - test_start
                
                move_uci = move.uci() if move else "null"
                
                # Success if engine returns a legal move within reasonable time
                time_buffer = pos['time_limit'] * 1.5  # 50% buffer
                success = (move is not None and 
                          move != chess.Move.null() and 
                          test_time <= time_buffer)
                
                nodes = self.engine.search_stats.get('nodes_searched', 0)
                
                result = TestResult(
                    test_name=pos['name'],
                    success=success,
                    move=move_uci,
                    time_taken=test_time,
                    nodes_searched=nodes,
                    evaluation=None,
                    details={
                        'time_limit': pos['time_limit'],
                        'time_buffer': time_buffer,
                        'time_compliance': test_time <= time_buffer
                    }
                )
                
                tests.append(result)
                
                status = "✅ PASS" if success else "❌ FAIL"
                compliance = "✅ ON TIME" if test_time <= time_buffer else "⚠️  OVERTIME"
                print(f"  Result: {status}")
                print(f"  Move: {move_uci}")
                print(f"  Time: {test_time:.2f}s ({compliance})")
                print(f"  Nodes: {nodes:,}")
                
            except Exception as e:
                result = TestResult(
                    test_name=pos['name'],
                    success=False,
                    move=None,
                    time_taken=time.time() - test_start,
                    nodes_searched=0,
                    evaluation=None,
                    error=str(e)
                )
                tests.append(result)
                print(f"  Result: ❌ ERROR - {e}")
        
        end_time = datetime.now()
        
        passed = sum(1 for t in tests if t.success)
        total_time = sum(t.time_taken for t in tests)
        total_nodes = sum(t.nodes_searched for t in tests)
        avg_nps = total_nodes / total_time if total_time > 0 else 0
        
        return TestSuite(
            suite_name="Time Pressure Tests",
            tests=tests,
            start_time=start_time,
            end_time=end_time,
            total_tests=len(tests),
            passed_tests=passed,
            failed_tests=len(tests) - passed,
            average_time=total_time / len(tests),
            total_nodes=total_nodes,
            average_nps=avg_nps
        )
    
    def run_puzzle_performance_tests(self) -> TestSuite:
        """Test puzzle-solving performance"""
        print("\n🧩 PUZZLE PERFORMANCE TESTS")
        print("=" * 50)
        
        start_time = datetime.now()
        tests = []
        
        for pos in self.puzzle_positions:
            print(f"\nTesting: {pos['name']}")
            print(f"Position: {pos['fen']}")
            print(f"Rating: {pos['rating']}")
            print(f"Themes: {', '.join(pos['themes'])}")
            
            test_start = time.time()
            
            try:
                board = chess.Board(pos['fen'])
                
                # Scale time based on puzzle difficulty
                time_allocation = max(2.0, pos['rating'] / 300)  # Harder puzzles get more time
                
                move = self.engine.search(board, time_limit=time_allocation)
                test_time = time.time() - test_start
                
                move_uci = move.uci() if move else "null"
                
                # Get position evaluation
                try:
                    eval_score = self.engine._evaluate_position(board)
                except:
                    eval_score = None
                
                # Success if engine finds a reasonable move
                success = (move is not None and 
                          move != chess.Move.null() and 
                          move in board.legal_moves)
                
                nodes = self.engine.search_stats.get('nodes_searched', 0)
                
                result = TestResult(
                    test_name=pos['name'],
                    success=success,
                    move=move_uci,
                    time_taken=test_time,
                    nodes_searched=nodes,
                    evaluation=eval_score,
                    details={
                        'rating': pos['rating'],
                        'themes': pos['themes'],
                        'time_allocation': time_allocation
                    }
                )
                
                tests.append(result)
                
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"  Result: {status}")
                print(f"  Move: {move_uci}")
                print(f"  Evaluation: {eval_score:.2f}" if eval_score is not None else "  Evaluation: N/A")
                print(f"  Time: {test_time:.2f}s")
                print(f"  Nodes: {nodes:,}")
                
            except Exception as e:
                result = TestResult(
                    test_name=pos['name'],
                    success=False,
                    move=None,
                    time_taken=time.time() - test_start,
                    nodes_searched=0,
                    evaluation=None,
                    error=str(e)
                )
                tests.append(result)
                print(f"  Result: ❌ ERROR - {e}")
        
        end_time = datetime.now()
        
        passed = sum(1 for t in tests if t.success)
        total_time = sum(t.time_taken for t in tests)
        total_nodes = sum(t.nodes_searched for t in tests)
        avg_nps = total_nodes / total_time if total_time > 0 else 0
        
        return TestSuite(
            suite_name="Puzzle Performance Tests",
            tests=tests,
            start_time=start_time,
            end_time=end_time,
            total_tests=len(tests),
            passed_tests=passed,
            failed_tests=len(tests) - passed,
            average_time=total_time / len(tests),
            total_nodes=total_nodes,
            average_nps=avg_nps
        )
    
    def run_performance_benchmark(self) -> TestSuite:
        """Run performance benchmarks"""
        print("\n⚡ PERFORMANCE BENCHMARK")
        print("=" * 50)
        
        start_time = datetime.now()
        tests = []
        
        # Test positions for benchmarking
        benchmark_positions = [
            ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("Italian Game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
            ("Sicilian Defense", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
            ("Complex middlegame", "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P3/3P1N2/PPP2PPP/RN1QK2R w KQ - 0 8")
        ]
        
        for name, fen in benchmark_positions:
            print(f"\nBenchmarking: {name}")
            
            test_start = time.time()
            
            try:
                board = chess.Board(fen)
                
                # Run 3 trials for consistency
                trial_results = []
                
                for trial in range(3):
                    self.engine.new_game()  # Reset stats
                    trial_start = time.time()
                    move = self.engine.search(board, time_limit=3.0)
                    trial_time = time.time() - trial_start
                    trial_nodes = self.engine.search_stats.get('nodes_searched', 0)
                    
                    trial_results.append({
                        'time': trial_time,
                        'nodes': trial_nodes,
                        'nps': trial_nodes / trial_time if trial_time > 0 else 0
                    })
                
                # Calculate averages
                avg_time = sum(t['time'] for t in trial_results) / len(trial_results)
                avg_nodes = sum(t['nodes'] for t in trial_results) / len(trial_results)
                avg_nps = sum(t['nps'] for t in trial_results) / len(trial_results)
                
                test_time = time.time() - test_start
                
                result = TestResult(
                    test_name=f"Benchmark: {name}",
                    success=avg_nps > 5000,  # Success if > 5000 NPS
                    move=move.uci() if move else "null",
                    time_taken=test_time,
                    nodes_searched=int(avg_nodes),
                    evaluation=None,
                    details={
                        'avg_time': avg_time,
                        'avg_nodes': avg_nodes,
                        'avg_nps': avg_nps,
                        'trials': trial_results
                    }
                )
                
                tests.append(result)
                
                status = "✅ PASS" if avg_nps > 5000 else "⚠️  SLOW"
                print(f"  Result: {status}")
                print(f"  Average NPS: {avg_nps:.0f}")
                print(f"  Average Time: {avg_time:.2f}s")
                print(f"  Average Nodes: {avg_nodes:.0f}")
                
            except Exception as e:
                result = TestResult(
                    test_name=f"Benchmark: {name}",
                    success=False,
                    move=None,
                    time_taken=time.time() - test_start,
                    nodes_searched=0,
                    evaluation=None,
                    error=str(e)
                )
                tests.append(result)
                print(f"  Result: ❌ ERROR - {e}")
        
        end_time = datetime.now()
        
        passed = sum(1 for t in tests if t.success)
        total_time = sum(t.time_taken for t in tests)
        total_nodes = sum(t.nodes_searched for t in tests)
        avg_nps = total_nodes / total_time if total_time > 0 else 0
        
        return TestSuite(
            suite_name="Performance Benchmark",
            tests=tests,
            start_time=start_time,
            end_time=end_time,
            total_tests=len(tests),
            passed_tests=passed,
            failed_tests=len(tests) - passed,
            average_time=total_time / len(tests),
            total_nodes=total_nodes,
            average_nps=avg_nps
        )
    
    def save_results(self, results: List[TestSuite]):
        """Save test results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"v7p3r_v11_1_comprehensive_test_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        results_data = {
            'test_session': {
                'engine_version': 'V7P3R v11.1 Simplified',
                'test_timestamp': timestamp,
                'total_suites': len(results),
                'summary': {
                    'total_tests': sum(suite.total_tests for suite in results),
                    'total_passed': sum(suite.passed_tests for suite in results),
                    'total_failed': sum(suite.failed_tests for suite in results),
                    'overall_success_rate': sum(suite.passed_tests for suite in results) / sum(suite.total_tests for suite in results) * 100
                }
            },
            'test_suites': []
        }
        
        for suite in results:
            suite_data = {
                'suite_name': suite.suite_name,
                'start_time': suite.start_time.isoformat(),
                'end_time': suite.end_time.isoformat(),
                'total_tests': suite.total_tests,
                'passed_tests': suite.passed_tests,
                'failed_tests': suite.failed_tests,
                'success_rate': (suite.passed_tests / suite.total_tests * 100) if suite.total_tests > 0 else 0,
                'average_time': suite.average_time,
                'total_nodes': suite.total_nodes,
                'average_nps': suite.average_nps,
                'tests': []
            }
            
            for test in suite.tests:
                test_data = {
                    'test_name': test.test_name,
                    'success': test.success,
                    'move': test.move,
                    'time_taken': test.time_taken,
                    'nodes_searched': test.nodes_searched,
                    'evaluation': test.evaluation,
                    'error': test.error,
                    'details': test.details
                }
                suite_data['tests'].append(test_data)
            
            results_data['test_suites'].append(suite_data)
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename
    
    def print_summary(self, results: List[TestSuite]):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("🎯 V7P3R v11.1 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 80)
        
        total_tests = sum(suite.total_tests for suite in results)
        total_passed = sum(suite.passed_tests for suite in results)
        total_failed = sum(suite.failed_tests for suite in results)
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Overall Results: {total_passed}/{total_tests} tests passed ({overall_success_rate:.1f}%)")
        print()
        
        for suite in results:
            success_rate = (suite.passed_tests / suite.total_tests * 100) if suite.total_tests > 0 else 0
            status = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 60 else "❌"
            
            print(f"{status} {suite.suite_name}:")
            print(f"   Success Rate: {suite.passed_tests}/{suite.total_tests} ({success_rate:.1f}%)")
            print(f"   Average Time: {suite.average_time:.2f}s")
            print(f"   Average NPS: {suite.average_nps:.0f}")
            print(f"   Total Nodes: {suite.total_nodes:,}")
            print()
        
        # Performance analysis
        avg_nps = sum(suite.average_nps for suite in results) / len(results)
        if avg_nps > 10000:
            print("🚀 PERFORMANCE: Excellent (>10,000 NPS)")
        elif avg_nps > 5000:
            print("✅ PERFORMANCE: Good (>5,000 NPS)")
        else:
            print("⚠️ PERFORMANCE: Needs improvement (<5,000 NPS)")
        
        # Readiness assessment
        if overall_success_rate >= 80 and avg_nps > 5000:
            print("\n🎉 ASSESSMENT: V7P3R v11.1 is ready for production!")
            print("   ✅ High test success rate")
            print("   ✅ Good performance metrics")
            print("   ✅ Stable time management")
        elif overall_success_rate >= 60:
            print("\n🔧 ASSESSMENT: V7P3R v11.1 needs minor tuning")
            print("   ⚠️ Some test failures detected")
            print("   🔍 Review failed test cases")
        else:
            print("\n❌ ASSESSMENT: V7P3R v11.1 needs significant work")
            print("   ❌ Multiple test failures")
            print("   🔧 Consider further simplification")
    
    def run_comprehensive_test_suite(self):
        """Run all test suites"""
        print("🎯 V7P3R v11.1 COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print("Testing engine functionality, performance, and reliability")
        print("Direct-to-code testing without executable build")
        print("=" * 80)
        
        results = []
        
        # Run all test suites
        results.append(self.run_tactical_tests())
        results.append(self.run_positional_tests())
        results.append(self.run_time_pressure_tests())
        results.append(self.run_puzzle_performance_tests())
        results.append(self.run_performance_benchmark())
        
        # Save and summarize results
        self.save_results(results)
        self.print_summary(results)
        
        return results


def main():
    """Main test execution"""
    tester = V7P3RComprehensiveTester()
    results = tester.run_comprehensive_test_suite()
    return results


if __name__ == "__main__":
    main()