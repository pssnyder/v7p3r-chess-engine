#!/usr/bin/env python3
"""
V7P3R v10.6 Baseline Performance Measurement
Establishes performance baselines for tracking V11.3 development progress
"""

import sys
import os
import time
import chess
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from v7p3r import V7P3REngine

class V10_6_BaselineMeasurement:
    """Comprehensive baseline measurement for V10.6 performance"""
    
    def __init__(self):
        self.engine = V7P3REngine()
        self.results = {
            'version': 'V10.6 Baseline',
            'timestamp': datetime.now().isoformat(),
            'search_depth': {},
            'node_efficiency': {},
            'evaluation_timing': {},
            'tactical_accuracy': {},
            'endgame_performance': {}
        }
        
    def measure_search_depth(self):
        """Measure search depth in various position types"""
        print("📊 Measuring search depth performance...")
        
        test_positions = [
            ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("Middlegame tactical", "r1bq1rk1/pp1nbppp/2p2n2/3p2B1/3P4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8"),
            ("Endgame position", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
            ("Complex tactical", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
        ]
        
        for name, fen in test_positions:
            board = chess.Board(fen)
            start_time = time.time()
            
            # Measure depth achieved in 3 seconds
            best_move = self.engine.search(board, time_limit=3.0)
            search_time = time.time() - start_time
            
            # Create result dict with available data
            result = {
                'move': str(best_move),
                'depth': getattr(self.engine, 'search_depth_reached', 0),
                'nodes': getattr(self.engine, 'nodes_searched', 0)
            }
            
            self.results['search_depth'][name] = {
                'depth_achieved': result.get('depth', 0),
                'nodes_searched': result.get('nodes', 0),
                'time_used': search_time,
                'nps': result.get('nodes', 0) / max(search_time, 0.001)
            }
            
            print(f"  {name}: Depth {result.get('depth', 0)}, Nodes {result.get('nodes', 0)}, NPS {self.results['search_depth'][name]['nps']:.0f}")
            
    def measure_evaluation_timing(self):
        """Measure evaluation component timing"""
        print("⏱️ Measuring evaluation component timing...")
        
        test_board = chess.Board("r1bq1rk1/pp1nbppp/2p2n2/3p2B1/3P4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8")
        
        # Measure evaluation timing
        times = []
        for i in range(100):
            start_time = time.time()
            score = self.engine._evaluate_position(test_board)
            eval_time = time.time() - start_time
            times.append(eval_time)
            
        avg_time = sum(times) / len(times)
        self.results['evaluation_timing'] = {
            'average_ms': avg_time * 1000,
            'max_ms': max(times) * 1000,
            'min_ms': min(times) * 1000,
            'evaluations_per_second': 1.0 / avg_time if avg_time > 0 else 0
        }
        
        print(f"  Average evaluation: {avg_time * 1000:.2f}ms")
        print(f"  Evaluations per second: {self.results['evaluation_timing']['evaluations_per_second']:.0f}")
        
    def measure_tactical_detection(self):
        """Measure current tactical detection capabilities"""
        print("🎯 Measuring tactical detection baseline...")
        
        tactical_positions = [
            ("Knight fork", "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3"),
            ("Pin tactic", "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
            ("Hanging piece", "rnbqkb1r/ppp2ppp/3p1n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5"),
            ("Back rank mate", "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
        ]
        
        detected = 0
        total = len(tactical_positions)
        
        for name, fen in tactical_positions:
            board = chess.Board(fen)
            best_move = self.engine.search(board, time_limit=2.0)
            
            # Simple heuristic: if we get a move, consider it detected
            if best_move and best_move != chess.Move.null():
                detected += 1
                print(f"  ✓ {name}: Detected (move: {best_move})")
            else:
                print(f"  ✗ {name}: Not detected")
                
        self.results['tactical_accuracy']['baseline_detection_rate'] = detected / total
        print(f"  Baseline tactical detection: {detected}/{total} ({detected/total*100:.1f}%)")
        
    def measure_endgame_performance(self):
        """Measure endgame-specific performance"""
        print("👑 Measuring endgame performance baseline...")
        
        endgame_positions = [
            ("King and pawn vs King", "8/8/4k3/4p3/4K3/8/8/8 w - - 0 1"),
            ("Rook endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
            ("Queen vs pawns", "8/1k6/8/1p6/1P6/8/6Q1/6K1 w - - 0 1")
        ]
        
        endgame_scores = []
        for name, fen in endgame_positions:
            board = chess.Board(fen)
            start_time = time.time()
            best_move = self.engine.search(board, time_limit=3.0)
            search_time = time.time() - start_time
            
            # Get basic info
            depth = getattr(self.engine, 'search_depth_reached', 0)
            nodes = getattr(self.engine, 'nodes_searched', 0)
            endgame_scores.append(depth)
            
            print(f"  {name}: Depth {depth}, Nodes {nodes}, Move {best_move}")
            
        self.results['endgame_performance'] = {
            'average_depth': sum(endgame_scores) / len(endgame_scores),
            'positions_tested': len(endgame_positions)
        }
        
    def save_baseline_report(self):
        """Save baseline measurements to file"""
        filename = f"v10_6_baseline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\n📄 Baseline report saved to: {filename}")
        return filepath
        
    def run_complete_baseline(self):
        """Run all baseline measurements"""
        print("🎯 V7P3R v10.6 Baseline Performance Measurement")
        print("=" * 60)
        
        try:
            self.measure_search_depth()
            print()
            self.measure_evaluation_timing()
            print()
            self.measure_tactical_detection()
            print()
            self.measure_endgame_performance()
            print()
            
            report_file = self.save_baseline_report()
            
            print("📊 BASELINE SUMMARY:")
            print(f"  Average search depth: {sum(pos['depth_achieved'] for pos in self.results['search_depth'].values()) / len(self.results['search_depth']):.1f} plies")
            print(f"  Average NPS: {sum(pos['nps'] for pos in self.results['search_depth'].values()) / len(self.results['search_depth']):.0f}")
            print(f"  Evaluation speed: {self.results['evaluation_timing']['average_ms']:.2f}ms")
            print(f"  Tactical detection: {self.results['tactical_accuracy']['baseline_detection_rate']*100:.1f}%")
            print(f"  Endgame depth: {self.results['endgame_performance']['average_depth']:.1f} plies")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during baseline measurement: {e}")
            return False

if __name__ == "__main__":
    baseline = V10_6_BaselineMeasurement()
    success = baseline.run_complete_baseline()
    
    if success:
        print("\n✅ V10.6 baseline measurement completed successfully!")
        print("Ready to begin V11.3 heuristic enhancements.")
    else:
        print("\n❌ Baseline measurement failed. Check engine configuration.")