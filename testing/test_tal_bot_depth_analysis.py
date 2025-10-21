#!/usr/bin/env python3
"""
TAL-BOT Depth Performance Analysis & Optimization

Identify and fix the bottlenecks preventing TAL-BOT from reaching 6-10 ply depth.
The "forest vision" advantage requires deep search, not just smart evaluation.
"""

import time
import chess
import sys
import os
import cProfile
import pstats

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vpr import VPREngine


def profile_tal_bot_performance():
    """Profile TAL-BOT to find performance bottlenecks"""
    print("=== TAL-BOT DEPTH PERFORMANCE ANALYSIS ===\n")
    
    engine = VPREngine()
    board = chess.Board()
    
    print("Profiling TAL-BOT search for performance bottlenecks...")
    
    # Profile a 3-second search
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    best_move = engine.search(board, time_limit=3.0)
    actual_time = time.time() - start_time
    
    profiler.disable()
    
    print(f"Search completed in {actual_time:.2f}s")
    print(f"Nodes searched: {engine.nodes_searched:,}")
    print(f"NPS: {int(engine.nodes_searched / actual_time):,}")
    print(f"Best move: {best_move}")
    
    # Analyze profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')
    
    print("\n=== TOP TIME CONSUMERS ===")
    stats.print_stats(10)  # Show top 10 functions
    
    return stats


def benchmark_evaluation_speed():
    """Benchmark evaluation components to find slowest parts"""
    print("\n=== EVALUATION COMPONENT BENCHMARKS ===\n")
    
    engine = VPREngine()
    positions = [
        chess.Board(),  # Starting position
        chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"),  # Middlegame
        chess.Board("r1bq1rk1/pp2bppp/2np1n2/4p3/2B1P3/2NP1N2/PPP1QPPP/R1B2RK1 w - - 0 10")  # Complex
    ]
    
    iterations = 1000
    
    for i, board in enumerate(positions):
        print(f"Position {i+1}: {len(list(board.legal_moves))} legal moves")
        
        # Test full evaluation
        start = time.time()
        for _ in range(iterations):
            score = engine._evaluate_position(board)
        eval_time = time.time() - start
        
        # Test chaos factor
        start = time.time()
        for _ in range(iterations):
            chaos = engine._calculate_chaos_factor(board)
        chaos_time = time.time() - start
        
        # Test piece value calculation
        start = time.time()
        for _ in range(iterations):
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    value = engine._calculate_piece_true_value(board, square, piece.color)
        piece_time = time.time() - start
        
        # Test move ordering
        moves = list(board.legal_moves)
        start = time.time()
        for _ in range(iterations//10):  # Fewer iterations for move ordering
            ordered = engine._order_moves_simple(board, moves)
        move_time = (time.time() - start) * 10  # Scale back up
        
        print(f"  Full evaluation: {eval_time*1000/iterations:.2f}ms per call")
        print(f"  Chaos factor: {chaos_time*1000/iterations:.2f}ms per call")
        print(f"  Piece values: {piece_time*1000/iterations:.2f}ms per call")
        print(f"  Move ordering: {move_time*1000/iterations:.2f}ms per call")
        print()


def test_simplified_evaluation():
    """Test if simplified evaluation improves depth"""
    print("=== SIMPLIFIED EVALUATION TEST ===\n")
    
    # Create a speed-optimized version
    class FastTALBOT(VPREngine):
        def _evaluate_position_fast(self, board):
            """Ultra-fast evaluation for depth testing"""
            if board.is_checkmate():
                return -900000
            if board.is_stalemate() or board.is_insufficient_material():
                return 0
            
            # Simple material count only
            white_material = 0
            black_material = 0
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    value = self.piece_values.get(piece.piece_type, 0)
                    if piece.color == chess.WHITE:
                        white_material += value
                    else:
                        black_material += value
            
            if board.turn == chess.WHITE:
                return white_material - black_material
            else:
                return black_material - white_material
        
        def search_fast(self, board, time_limit=3.0):
            """Fast search with simplified evaluation"""
            self.nodes_searched = 0
            self.search_start_time = time.time()
            
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return chess.Move.null()
            
            if len(legal_moves) == 1:
                return legal_moves[0]
            
            target_time = time_limit * 0.8
            best_move = legal_moves[0]
            
            for current_depth in range(1, 15):  # Try deeper
                if time.time() - self.search_start_time > target_time:
                    break
                
                score = -999999
                for move in legal_moves:
                    if time.time() - self.search_start_time > target_time:
                        break
                    
                    board.push(move)
                    move_score = -self._negamax_fast(board, current_depth - 1, -999999, 999999, target_time)
                    board.pop()
                    
                    if move_score > score:
                        score = move_score
                        best_move = move
                
                print(f"info depth {current_depth} score cp {int(score)} nodes {self.nodes_searched}")
            
            return best_move
        
        def _negamax_fast(self, board, depth, alpha, beta, target_time):
            """Fast negamax with minimal evaluation"""
            self.nodes_searched += 1
            
            if depth <= 0:
                return self._evaluate_position_fast(board)
            
            if board.is_game_over():
                if board.is_checkmate():
                    return -900000 + depth
                return 0
            
            # Time check every 2000 nodes
            if self.nodes_searched % 2000 == 0:
                if time.time() - self.search_start_time > target_time:
                    return 0
            
            for move in board.legal_moves:
                board.push(move)
                score = -self._negamax_fast(board, depth - 1, -beta, -alpha, target_time)
                board.pop()
                
                if score >= beta:
                    return beta
                alpha = max(alpha, score)
            
            return alpha
    
    # Test fast version
    fast_engine = FastTALBOT()
    board = chess.Board()
    
    print("Testing simplified TAL-BOT for depth performance...")
    start_time = time.time()
    move = fast_engine.search_fast(board, time_limit=3.0)
    search_time = time.time() - start_time
    
    print(f"\nFast TAL-BOT Results:")
    print(f"Search time: {search_time:.2f}s")
    print(f"Nodes searched: {fast_engine.nodes_searched:,}")
    print(f"NPS: {int(fast_engine.nodes_searched / search_time):,}")
    print(f"Best move: {move}")


def compare_depth_performance():
    """Compare depth achievement between current and optimized versions"""
    print("\n=== DEPTH COMPARISON TEST ===\n")
    
    engine = VPREngine()
    
    # Test different time limits
    time_limits = [1.0, 2.0, 3.0, 5.0]
    board = chess.Board()
    
    print("Current TAL-BOT depth performance:")
    print(f"{'Time':<6} {'Depth':<6} {'Nodes':<10} {'NPS':<8}")
    print("-" * 35)
    
    for time_limit in time_limits:
        start_time = time.time()
        move = engine.search(board, time_limit=time_limit)
        actual_time = time.time() - start_time
        
        nps = int(engine.nodes_searched / actual_time) if actual_time > 0 else 0
        print(f"{time_limit:<6.1f} {'?':<6} {engine.nodes_searched:<10,} {nps:<8,}")


if __name__ == "__main__":
    print("TAL-BOT Depth Performance Analysis")
    print("=" * 60)
    print("Finding bottlenecks to achieve 6-10 ply depth...\n")
    
    try:
        # Run analysis
        benchmark_evaluation_speed()
        test_simplified_evaluation()
        compare_depth_performance()
        
        print("\n🎯 DEPTH OPTIMIZATION RECOMMENDATIONS:")
        print("1. Simplify evaluation during deep search")
        print("2. Optimize move ordering for better pruning")
        print("3. Reduce chaos calculation frequency")
        print("4. Cache expensive computations")
        print("5. Use iterative deepening more efficiently")
        
        print("\n🔥 NEXT: Implement speed optimizations for forest vision!")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        print("Will need to debug TAL-BOT implementation...")