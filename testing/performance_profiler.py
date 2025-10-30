#!/usr/bin/env python3
"""
V7P3R v14.4 Performance Profiler
Comprehensive analysis of engine performance bottlenecks
"""

import sys
import os
import time
import cProfile
import pstats
import io
import chess
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from v7p3r import V7P3REngine
except ImportError as e:
    print(f"Failed to import V7P3REngine: {e}")
    sys.exit(1)

class PerformanceProfiler:
    """Profile V7P3R engine performance"""
    
    def __init__(self):
        self.engine = V7P3REngine()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'version': 'V7P3R v14.4',
            'tests': {}
        }
    
    def profile_opening_moves(self, depth: int = 4, num_moves: int = 5) -> Dict:
        """Profile early game performance where time wastage was observed"""
        print(f"Profiling opening moves (depth {depth}, {num_moves} moves)...")
        
        board = chess.Board()  # Starting position
        move_times = []
        total_nodes = 0
        
        for move_num in range(num_moves):
            print(f"  Move {move_num + 1}...", end=" ")
            
            # Time the move search
            start_time = time.time()
            
            # Create profiler for this move
            pr = cProfile.Profile()
            pr.enable()
            
            # Search for best move
            try:
                result = self.engine.search(board, time_limit=10.0, depth=depth)
                if isinstance(result, tuple):
                    best_move, evaluation = result
                else:
                    best_move = result
                    evaluation = 0
                nodes_searched = getattr(self.engine, 'nodes_searched', 0)
            except Exception as e:
                print(f"Error: {e}")
                best_move, evaluation, nodes_searched = None, 0, 0
            
            pr.disable()
            
            end_time = time.time()
            move_time = end_time - start_time
            move_times.append(move_time)
            total_nodes += nodes_searched
            
            print(f"{move_time:.3f}s, {nodes_searched} nodes")
            
            # Make the move for next iteration
            if best_move and best_move in board.legal_moves:
                board.push(best_move)
            else:
                # Fallback to random legal move
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    board.push(legal_moves[0])
                else:
                    break
        
        return {
            'move_times': move_times,
            'average_time': sum(move_times) / len(move_times) if move_times else 0,
            'total_time': sum(move_times),
            'total_nodes': total_nodes,
            'nps': total_nodes / sum(move_times) if sum(move_times) > 0 else 0
        }
    
    def profile_tactical_positions(self) -> Dict:
        """Profile performance on tactical positions"""
        print("Profiling tactical positions...")
        
        # Sample tactical positions from different phases
        positions = [
            # Phase 1 position (queen threat)
            {
                'name': 'Queen Threat Fork',
                'fen': 'r6k/pp4pp/3pQ1n1/2pP1rq1/2N5/8/PPP2PPP/4RRK1 b - - 3 18',
                'expected': 'g6f4'
            },
            # Complex middlegame
            {
                'name': 'Complex Middlegame',
                'fen': '3r1rk1/pp1q1p1R/2n2p1Q/4pb2/4B3/2P2NP1/PP3PP1/R3K3 b Q - 8 20',
                'expected': 'f5e4'
            },
            # Pin detection test
            {
                'name': 'Pin Detection',
                'fen': '3k4/8/8/8/3n4/8/8/3R4 w - - 0 1',
                'expected': 'Rd8'
            }
        ]
        
        results = {}
        
        for pos in positions:
            print(f"  Testing {pos['name']}...")
            board = chess.Board(pos['fen'])
            
            # Profile the search
            pr = cProfile.Profile()
            pr.enable()
            
            start_time = time.time()
            try:
                result = self.engine.search(board, time_limit=5.0, depth=4)
                if isinstance(result, tuple):
                    best_move, evaluation = result
                else:
                    best_move = result
                    evaluation = 0
                nodes_searched = getattr(self.engine, 'nodes_searched', 0)
            except Exception as e:
                print(f"    Error: {e}")
                best_move, evaluation, nodes_searched = None, 0, 0
            
            pr.disable()
            end_time = time.time()
            
            # Capture profile stats
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(10)  # Top 10 functions
            
            results[pos['name']] = {
                'time': end_time - start_time,
                'nodes': nodes_searched,
                'nps': nodes_searched / (end_time - start_time) if end_time > start_time else 0,
                'best_move': best_move.uci() if best_move else None,
                'expected': pos['expected'],
                'correct': best_move.uci() == pos['expected'] if best_move else False,
                'profile_stats': s.getvalue()
            }
        
        return results
    
    def profile_move_ordering(self) -> Dict:
        """Profile move ordering performance"""
        print("Profiling move ordering...")
        
        # Test position with many legal moves
        board = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        legal_moves = list(board.legal_moves)
        
        print(f"  Testing with {len(legal_moves)} legal moves...")
        
        # Profile move ordering
        pr = cProfile.Profile()
        pr.enable()
        
        start_time = time.time()
        ordered_moves = self.engine._order_moves_advanced(board, legal_moves, 4)
        end_time = time.time()
        
        pr.disable()
        
        # Capture profile stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(10)
        
        return {
            'time': end_time - start_time,
            'moves_processed': len(legal_moves),
            'moves_per_second': len(legal_moves) / (end_time - start_time) if end_time > start_time else 0,
            'profile_stats': s.getvalue()
        }
    
    def profile_evaluation_methods(self) -> Dict:
        """Profile individual evaluation components"""
        print("Profiling evaluation methods...")
        
        board = chess.Board('rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1')
        
        methods = [
            ('Position Evaluation', lambda: self.engine._evaluate_position(board)),
            ('Pin Detection', lambda: self.engine._detect_pins(board)),
            ('Move Ordering', lambda: self.engine._order_moves_advanced(board, list(board.legal_moves), 4)),
            ('Tactical Analysis', lambda: self.engine._analyze_position_for_tactics(board))
        ]
        
        results = {}
        
        for method_name, method_func in methods:
            print(f"  Testing {method_name}...")
            
            # Time multiple runs for accuracy
            times = []
            for _ in range(100):  # 100 runs
                start_time = time.time()
                try:
                    result = method_func()
                except Exception as e:
                    print(f"    Error in {method_name}: {e}")
                    result = None
                end_time = time.time()
                times.append(end_time - start_time)
            
            results[method_name] = {
                'average_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'calls_per_second': 1.0 / (sum(times) / len(times)) if times else 0
            }
        
        return results
    
    def profile_time_management(self) -> Dict:
        """Profile time management under different time controls"""
        print("Profiling time management...")
        
        time_controls = [
            {'name': 'Bullet', 'total_time': 60, 'increment': 0},  # 1+0
            {'name': 'Blitz', 'total_time': 300, 'increment': 3},  # 5+3  
            {'name': 'Rapid', 'total_time': 600, 'increment': 5},  # 10+5
            {'name': 'Classical', 'total_time': 1800, 'increment': 30}  # 30+30
        ]
        
        results = {}
        
        for tc in time_controls:
            print(f"  Testing {tc['name']} time control...")
            
            # Simulate early game under time pressure
            board = chess.Board()
            move_count = 0
            remaining_time = tc['total_time']
            
            time_usage = []
            
            for move_num in range(5):  # First 5 moves
                move_count += 1
                
                # Calculate time allocation for this move
                estimated_game_length = 40  # moves
                remaining_moves = max(1, estimated_game_length - move_count)
                base_time = remaining_time / remaining_moves
                allocated_time = base_time + tc['increment']
                
                print(f"    Move {move_num + 1}: {allocated_time:.1f}s allocated, {remaining_time:.1f}s remaining")
                
                # Search with time limit
                start_time = time.time()
                try:
                    # Use iterative deepening with time limit
                    best_move = None
                    for depth in range(1, 8):
                        if time.time() - start_time > allocated_time * 0.8:  # Use 80% of allocated time
                            break
                        result = self.engine.search(board, time_limit=allocated_time * 0.8, depth=depth)
                        if isinstance(result, tuple):
                            move, eval = result
                        else:
                            move = result
                        if move:
                            best_move = move
                
                except Exception as e:
                    print(f"      Error: {e}")
                    best_move = None
                
                end_time = time.time()
                actual_time = end_time - start_time
                time_usage.append(actual_time)
                
                # Update remaining time
                remaining_time = remaining_time - actual_time + tc['increment']
                remaining_time = max(0, remaining_time)
                
                print(f"      Used {actual_time:.3f}s, {remaining_time:.1f}s remaining")
                
                # Make a move
                if best_move and best_move in board.legal_moves:
                    board.push(best_move)
                else:
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        board.push(legal_moves[0])
                
                if remaining_time <= 0:
                    print(f"      TIME OUT after move {move_num + 1}!")
                    break
            
            results[tc['name']] = {
                'time_usage': time_usage,
                'average_time': sum(time_usage) / len(time_usage) if time_usage else 0,
                'total_time_used': sum(time_usage),
                'remaining_time': remaining_time,
                'moves_completed': len(time_usage),
                'flagged': remaining_time <= 0
            }
        
        return results
    
    def run_full_profile(self) -> Dict:
        """Run comprehensive performance profile"""
        print("=" * 60)
        print("V7P3R v14.4 Performance Profile")
        print("=" * 60)
        
        # Run all profiling tests
        self.results['tests']['opening_moves'] = self.profile_opening_moves()
        self.results['tests']['tactical_positions'] = self.profile_tactical_positions()
        self.results['tests']['move_ordering'] = self.profile_move_ordering()
        self.results['tests']['evaluation_methods'] = self.profile_evaluation_methods()
        self.results['tests']['time_management'] = self.profile_time_management()
        
        return self.results
    
    def save_results(self, filename: str = ""):
        """Save profiling results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"v7p3r_v14.4_performance_profile_{timestamp}.json"
        
        filepath = os.path.join("testing", filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nProfile results saved to: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print performance summary"""
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        # Opening moves analysis
        opening = self.results['tests']['opening_moves']
        print(f"\n📈 OPENING MOVES:")
        print(f"  Average time per move: {opening['average_time']:.3f}s")
        print(f"  Total time (5 moves): {opening['total_time']:.3f}s")
        print(f"  Nodes per second: {opening['nps']:,.0f}")
        
        if opening['average_time'] > 10.0:
            print(f"  ⚠️  WARNING: Taking >10s per opening move!")
        elif opening['average_time'] > 5.0:
            print(f"  ⚠️  SLOW: Taking >5s per opening move")
        
        # Time management analysis
        time_mgmt = self.results['tests']['time_management']
        print(f"\n⏰ TIME MANAGEMENT:")
        
        for tc_name, tc_data in time_mgmt.items():
            flagged = tc_data['flagged']
            avg_time = tc_data['average_time']
            status = "🚩 FLAGGED" if flagged else "✅ OK"
            print(f"  {tc_name}: {avg_time:.2f}s avg, {status}")
        
        # Evaluation methods performance
        eval_methods = self.results['tests']['evaluation_methods']
        print(f"\n🔍 EVALUATION METHODS:")
        
        for method, data in eval_methods.items():
            calls_per_sec = data['calls_per_second']
            avg_time = data['average_time'] * 1000  # Convert to ms
            print(f"  {method}: {avg_time:.3f}ms ({calls_per_sec:,.0f}/sec)")
        
        # Move ordering performance  
        move_ordering = self.results['tests']['move_ordering']
        print(f"\n🎯 MOVE ORDERING:")
        print(f"  Time: {move_ordering['time']:.3f}s")
        print(f"  Moves per second: {move_ordering['moves_per_second']:,.0f}")
        
        print("\n" + "=" * 60)

def main():
    """Run performance profiler"""
    profiler = PerformanceProfiler()
    
    try:
        results = profiler.run_full_profile()
        profiler.print_summary()
        profiler.save_results()
        
        return results
        
    except KeyboardInterrupt:
        print("\nProfiling interrupted by user")
        return None
    except Exception as e:
        print(f"\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()