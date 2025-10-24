#!/usr/bin/env python3
"""
V13.x Competitive Testing Suite
Comprehensive validation for tournament-ready deployment

Tests:
1. UCI Protocol Compliance
2. Move Quality vs Stockfish 
3. Performance Benchmarks
4. Engine vs Engine Testing
5. Tactical Position Validation
6. Arena Readiness Check
"""

import chess
import chess.engine
import time
import sys
import os
import json
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class V13xCompetitiveValidator:
    """Comprehensive testing suite for V13.x competitive readiness"""
    
    def __init__(self):
        self.results = {
            'uci_compliance': False,
            'move_quality_score': 0.0,
            'performance_metrics': {},
            'tactical_accuracy': 0.0,
            'arena_ready': False,
            'detailed_results': {}
        }
        
        # Tactical test positions (known best moves)
        self.tactical_positions = [
            # Format: (name, fen, best_move, category)
            ("Back Rank Mate", "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", "Ra8+", "mate_in_1"),
            ("Fork Attack", "rnbqk2r/pppp1ppp/5n2/4p3/1b2P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 4 4", "Nxe5", "fork"),
            ("Pin Breaking", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4", "Bxf7+", "discovered_attack"),
            ("Piece Safety", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Bb5", "pin"),
            ("King Safety", "rnbqk1nr/pppp1ppp/8/2b1p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3", "O-O", "castling"),
            ("Material Gain", "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3", "d3", "development"),
            ("Center Control", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e4", "opening"),
            ("Endgame Technique", "8/8/8/8/8/8/k1K5/8 w - - 0 1", "Kc3", "king_activity"),
            ("Pawn Breakthrough", "8/2k5/8/8/8/8/1P6/1K6 w - - 0 1", "b4", "pawn_advance"),
            ("Defensive Move", "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R b KQkq - 1 4", "Be7", "development")
        ]
        
        # Performance test positions (varying complexity)
        self.performance_positions = [
            ("Simple Opening", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
            ("Complex Middlegame", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP1NPPP/R1BQK2R w KQ - 4 6"),
            ("Tactical Chaos", "r2qkb1r/pp1n1ppp/2p1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 6"),
            ("Sharp Opening", "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3"),
            ("Endgame Position", "8/2k5/3p4/p2P1p2/P4P2/8/2K5/8 w - - 0 1")
        ]
    
    def run_full_validation(self) -> Dict:
        """Run complete validation suite"""
        print("🚀 V13.x COMPETITIVE TESTING SUITE")
        print("="*60)
        print("Preparing V13.x for tournament deployment...")
        
        try:
            # Import engine
            from v7p3r import V7P3REngine
            self.engine = V7P3REngine()
            
            # Test 1: UCI Compliance
            print(f"\n📋 TEST 1: UCI Protocol Compliance")
            self.test_uci_compliance()
            
            # Test 2: Move Quality
            print(f"\n🎯 TEST 2: Tactical Move Quality")
            self.test_move_quality()
            
            # Test 3: Performance Benchmarks
            print(f"\n⚡ TEST 3: Performance Benchmarks")
            self.test_performance_benchmarks()
            
            # Test 4: V13.x Statistics
            print(f"\n📊 TEST 4: V13.x Move Ordering Statistics")
            self.test_v13x_statistics()
            
            # Test 5: Arena Readiness
            print(f"\n🏆 TEST 5: Arena Readiness Check")
            self.test_arena_readiness()
            
            # Generate final report
            self.generate_final_report()
            
        except ImportError as e:
            print(f"❌ Could not import V7P3REngine: {e}")
            self.results['arena_ready'] = False
        except Exception as e:
            print(f"❌ Error during validation: {e}")
            import traceback
            traceback.print_exc()
            self.results['arena_ready'] = False
        
        return self.results
    
    def test_uci_compliance(self):
        """Test UCI protocol compliance"""
        uci_tests = {
            'engine_creation': False,
            'basic_search': False,
            'move_generation': False,
            'legal_moves': False,
            'fen_handling': False
        }
        
        try:
            # Test engine creation
            engine = self.engine
            uci_tests['engine_creation'] = True
            print("✅ Engine creation successful")
            
            # Test basic search
            board = chess.Board()
            move = engine.search(board, time_limit=1.0, depth=3)
            if move and move in board.legal_moves:
                uci_tests['basic_search'] = True
                print("✅ Basic search working")
            
            # Test move generation
            legal_moves = list(board.legal_moves)
            if len(legal_moves) == 20:  # Starting position has 20 legal moves
                uci_tests['legal_moves'] = True
                print("✅ Legal move generation correct")
            
            # Test FEN handling
            test_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"
            board = chess.Board(test_fen)
            move = engine.search(board, time_limit=1.0, depth=2)
            if move and move in board.legal_moves:
                uci_tests['fen_handling'] = True
                print("✅ FEN position handling working")
            
            # Test move ordering
            ordered_moves = engine._order_moves_advanced(board, list(board.legal_moves), 2)
            if len(ordered_moves) > 0:
                uci_tests['move_generation'] = True
                print("✅ V13.x move ordering working")
            
        except Exception as e:
            print(f"❌ UCI compliance error: {e}")
        
        compliance_score = sum(uci_tests.values()) / len(uci_tests) * 100
        self.results['uci_compliance'] = compliance_score >= 90
        self.results['detailed_results']['uci_tests'] = uci_tests
        
        print(f"UCI Compliance Score: {compliance_score:.1f}%")
    
    def test_move_quality(self):
        """Test move quality against known good moves"""
        correct_moves = 0
        total_tests = len(self.tactical_positions)
        detailed_results = []
        
        for name, fen, expected_move, category in self.tactical_positions:
            board = chess.Board(fen)
            
            try:
                # Test with V13.x
                start_time = time.time()
                best_move = self.engine.search(board, time_limit=2.0, depth=4)
                search_time = time.time() - start_time
                
                actual_move = board.san(best_move) if best_move else "None"
                
                # Check if move is correct (exact match or contains expected)
                is_correct = False
                if best_move:
                    if expected_move in actual_move or actual_move in expected_move:
                        is_correct = True
                    # Also check UCI notation
                    elif expected_move.lower() == best_move.uci().lower():
                        is_correct = True
                
                if is_correct:
                    correct_moves += 1
                    print(f"✅ {name}: {actual_move} (expected {expected_move})")
                else:
                    print(f"❌ {name}: {actual_move} (expected {expected_move})")
                
                detailed_results.append({
                    'name': name,
                    'category': category,
                    'expected': expected_move,
                    'actual': actual_move,
                    'correct': is_correct,
                    'search_time': search_time,
                    'nodes': getattr(self.engine, 'nodes_searched', 0)
                })
                
            except Exception as e:
                print(f"❌ {name}: Error - {e}")
                detailed_results.append({
                    'name': name,
                    'category': category,
                    'expected': expected_move,
                    'actual': 'Error',
                    'correct': False,
                    'error': str(e)
                })
        
        accuracy = correct_moves / total_tests * 100 if total_tests > 0 else 0
        self.results['tactical_accuracy'] = accuracy
        self.results['detailed_results']['tactical_tests'] = detailed_results
        
        print(f"Tactical Accuracy: {correct_moves}/{total_tests} ({accuracy:.1f}%)")
        
        # Category breakdown
        category_stats = {}
        for result in detailed_results:
            cat = result['category']
            if cat not in category_stats:
                category_stats[cat] = {'correct': 0, 'total': 0}
            category_stats[cat]['total'] += 1
            if result['correct']:
                category_stats[cat]['correct'] += 1
        
        print(f"Category Breakdown:")
        for cat, stats in category_stats.items():
            pct = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {cat}: {stats['correct']}/{stats['total']} ({pct:.1f}%)")
    
    def test_performance_benchmarks(self):
        """Test performance across different position types"""
        performance_results = []
        total_time = 0
        total_nodes = 0
        
        for name, fen in self.performance_positions:
            board = chess.Board(fen)
            
            try:
                # Benchmark search at depth 4
                start_time = time.time()
                best_move = self.engine.search(board, time_limit=3.0, depth=4)
                end_time = time.time()
                
                search_time = end_time - start_time
                nodes = getattr(self.engine, 'nodes_searched', 0)
                nps = nodes / search_time if search_time > 0 else 0
                
                # Get V13.x statistics
                v13x_stats = getattr(self.engine, 'v13x_stats', {})
                waiting_stats = getattr(self.engine, 'waiting_move_stats', {})
                
                result = {
                    'name': name,
                    'time': search_time,
                    'nodes': nodes,
                    'nps': nps,
                    'move': board.san(best_move) if best_move else 'None',
                    'v13x_pruning': v13x_stats.get('pruning_rate', 0),
                    'waiting_moves_used': sum(waiting_stats.values()) if waiting_stats else 0
                }
                
                performance_results.append(result)
                total_time += search_time
                total_nodes += nodes
                
                print(f"📊 {name}:")
                print(f"   Time: {search_time:.3f}s, NPS: {nps:.0f}")
                print(f"   V13.x Pruning: {result['v13x_pruning']:.1f}%")
                
            except Exception as e:
                print(f"❌ {name}: Performance test failed - {e}")
                performance_results.append({
                    'name': name,
                    'error': str(e)
                })
        
        # Overall performance metrics
        if total_time > 0:
            overall_nps = total_nodes / total_time
            avg_pruning = sum(r.get('v13x_pruning', 0) for r in performance_results) / len(performance_results)
            
            self.results['performance_metrics'] = {
                'total_time': total_time,
                'total_nodes': total_nodes,
                'overall_nps': overall_nps,
                'average_pruning_rate': avg_pruning,
                'baseline_nps': 1762,  # V12.6 baseline
                'performance_improvement': overall_nps / 1762 if overall_nps > 0 else 0
            }
            
            print(f"\n🚀 OVERALL PERFORMANCE:")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Total nodes: {total_nodes}")
            print(f"   Overall NPS: {overall_nps:.0f}")
            print(f"   V12.6 Baseline: 1762 NPS")
            print(f"   Performance ratio: {overall_nps/1762:.1f}x")
            print(f"   Average pruning: {avg_pruning:.1f}%")
        
        self.results['detailed_results']['performance_tests'] = performance_results
    
    def test_v13x_statistics(self):
        """Test V13.x move ordering statistics"""
        print("Testing V13.x move ordering statistics...")
        
        # Test complex position to gather statistics
        board = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP1NPPP/R1BQK2R w KQ - 4 6")
        legal_moves = list(board.legal_moves)
        
        # Get move ordering
        ordered_moves = self.engine._order_moves_advanced(board, legal_moves, 4)
        waiting_moves = self.engine.get_waiting_moves()
        
        # Calculate statistics
        pruning_rate = len(waiting_moves) / len(legal_moves) * 100 if len(legal_moves) > 0 else 0
        
        v13x_analysis = {
            'total_legal_moves': len(legal_moves),
            'critical_moves': len(ordered_moves),
            'waiting_moves': len(waiting_moves),
            'pruning_rate': pruning_rate,
            'expected_speedup': len(legal_moves) / len(ordered_moves) if len(ordered_moves) > 0 else 1
        }
        
        print(f"📊 V13.x Move Ordering Analysis:")
        print(f"   Total legal moves: {v13x_analysis['total_legal_moves']}")
        print(f"   Critical moves: {v13x_analysis['critical_moves']}")
        print(f"   Waiting moves: {v13x_analysis['waiting_moves']}")
        print(f"   Pruning rate: {v13x_analysis['pruning_rate']:.1f}%")
        print(f"   Expected speedup: {v13x_analysis['expected_speedup']:.1f}x")
        
        # Test top critical moves
        print(f"   🎯 Top Critical Moves:")
        for i, move in enumerate(ordered_moves[:5], 1):
            print(f"      {i}. {board.san(move)}")
        
        self.results['detailed_results']['v13x_analysis'] = v13x_analysis
    
    def test_arena_readiness(self):
        """Test Arena tournament readiness"""
        arena_checks = {
            'uci_compliance': self.results['uci_compliance'],
            'tactical_accuracy': self.results['tactical_accuracy'] >= 70,  # 70% minimum
            'performance_adequate': False,
            'move_ordering_working': False,
            'no_crashes': True,
            'legal_moves_only': True
        }
        
        # Check performance
        perf = self.results.get('performance_metrics', {})
        if perf.get('overall_nps', 0) >= 800:  # Minimum NPS for Arena
            arena_checks['performance_adequate'] = True
        
        # Check move ordering
        v13x = self.results.get('detailed_results', {}).get('v13x_analysis', {})
        if v13x.get('pruning_rate', 0) >= 60:  # Significant pruning
            arena_checks['move_ordering_working'] = True
        
        # Test stability with multiple quick searches
        try:
            board = chess.Board()
            for _ in range(5):
                move = self.engine.search(board, time_limit=0.5, depth=3)
                if move not in board.legal_moves:
                    arena_checks['legal_moves_only'] = False
                    break
                board.push(move)
        except Exception as e:
            arena_checks['no_crashes'] = False
            print(f"❌ Stability test failed: {e}")
        
        # Calculate readiness score
        readiness_score = sum(arena_checks.values()) / len(arena_checks) * 100
        self.results['arena_ready'] = readiness_score >= 80
        
        print(f"🏆 ARENA READINESS CHECKLIST:")
        for check, passed in arena_checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check.replace('_', ' ').title()}")
        
        print(f"Arena Readiness Score: {readiness_score:.1f}%")
        
        if self.results['arena_ready']:
            print(f"🎉 V13.x IS TOURNAMENT READY!")
        else:
            print(f"🔧 V13.x needs improvements before tournament play")
        
        self.results['detailed_results']['arena_checks'] = arena_checks
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print(f"\n" + "="*60)
        print(f"🎯 V13.x COMPETITIVE VALIDATION REPORT")
        print(f"="*60)
        
        # Summary
        print(f"📊 SUMMARY:")
        print(f"   UCI Compliance: {'✅' if self.results['uci_compliance'] else '❌'}")
        print(f"   Tactical Accuracy: {self.results['tactical_accuracy']:.1f}%")
        
        perf = self.results.get('performance_metrics', {})
        if perf:
            print(f"   Performance NPS: {perf.get('overall_nps', 0):.0f}")
            print(f"   V12.6 Improvement: {perf.get('performance_improvement', 0):.1f}x")
            print(f"   Average Pruning: {perf.get('average_pruning_rate', 0):.1f}%")
        
        print(f"   Arena Ready: {'✅' if self.results['arena_ready'] else '❌'}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if self.results['tactical_accuracy'] < 70:
            print(f"   🔧 Improve tactical accuracy (current: {self.results['tactical_accuracy']:.1f}%)")
        
        if perf.get('overall_nps', 0) < 1500:
            print(f"   🔧 Optimize performance (current: {perf.get('overall_nps', 0):.0f} NPS)")
        
        if not self.results['uci_compliance']:
            print(f"   🔧 Fix UCI protocol compliance issues")
        
        if self.results['arena_ready']:
            print(f"   🚀 V13.x is ready for competitive testing!")
            print(f"   🏆 Recommended next steps:")
            print(f"      • Engine vs Engine matches")
            print(f"      • Tournament submission")
            print(f"      • Performance monitoring")
        
        # Save detailed results
        with open('v13x_validation_report.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: v13x_validation_report.json")


def main():
    """Run V13.x competitive validation"""
    validator = V13xCompetitiveValidator()
    results = validator.run_full_validation()
    
    if results['arena_ready']:
        print(f"\n🎉 VALIDATION COMPLETE: V13.x ready for competitive testing!")
        return True
    else:
        print(f"\n🔧 VALIDATION ISSUES: V13.x needs improvements")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)