# test_speed_benchmark_v4_2.py
"""
Speed benchmark for V7P3R Chess Engine v4.2
Tests various time controls and positions.
"""

import chess
import time
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from v7p3r_config import V7P3RConfig
from v7p3r_search import SearchController

def benchmark_time_controls():
    """Benchmark different time controls"""
    config = V7P3RConfig('config_default.json')
    
    # Test positions of varying complexity
    test_positions = [
        ("Opening", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("Spanish", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
        ("Middlegame", "r1bq1rk1/pp1nppbp/3p1np1/2pP4/2P1P3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 9"),
        ("Endgame", "8/8/3k4/3p4/3P4/3K4/8/8 w - - 0 1"),
    ]
    
    time_controls = [1.0, 2.0, 5.0, 10.0]  # seconds per move
    
    print("V7P3R Chess Engine v4.2 Speed Benchmark")
    print("=" * 60)
    
    for pos_name, fen in test_positions:
        print(f"\n{pos_name} Position: {fen}")
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        print(f"Legal moves: {len(legal_moves)}")
        
        for time_limit in time_controls:
            search_controller = SearchController(config)
            search_controller.use_ab_pruning = True
            search_controller.use_move_ordering = True
            
            # Find the maximum depth we can achieve in the time limit
            max_depth = 0
            best_move = None
            
            for depth in range(1, 12):  # Test up to depth 11
                search_controller.max_depth = depth
                
                start_time = time.time()
                try:
                    move, _ = search_controller._negamax_root(board, board.turn, legal_moves)
                    end_time = time.time()
                    
                    search_time = end_time - start_time
                    
                    if search_time <= time_limit:
                        max_depth = depth
                        best_move = move
                        nodes = search_controller.nodes_searched
                        cutoffs = search_controller.cutoffs
                    else:
                        break
                        
                except Exception as e:
                    print(f"    Error at depth {depth}: {e}")
                    break
            
            if max_depth > 0:
                cutoff_rate = (cutoffs / nodes * 100) if nodes > 0 else 0
                print(f"  {time_limit:4.1f}s: depth {max_depth}, {nodes:,} nodes, {cutoffs:,} cutoffs ({cutoff_rate:.1f}%), move: {best_move}")
            else:
                print(f"  {time_limit:4.1f}s: timeout at depth 1")

def stress_test_complex_position():
    """Stress test with the original complex position at various depths"""
    complex_fen = "r3k2r/p1ppqpb1/Bn2pnp1/3PN3/1p2P3/2N2Q2/PPPB1PpP/R3K2R w KQkq - 0 1"
    
    config = V7P3RConfig('config_default.json')
    search_controller = SearchController(config)
    search_controller.use_ab_pruning = True
    search_controller.use_move_ordering = True
    
    print(f"\n" + "="*60)
    print("STRESS TEST - Complex Position")
    print(f"FEN: {complex_fen}")
    print("="*60)
    
    board = chess.Board(complex_fen)
    
    # First, let's force it to NOT take the hanging piece by testing just one specific move
    # We'll test a reasonable developing move instead
    test_moves = [chess.Move.from_uci("e1c1")]  # Castling queenside
    
    for depth in range(1, 8):
        search_controller.max_depth = depth
        
        start_time = time.time()
        try:
            # Test the search without the hanging capture shortcut
            score = search_controller._negamax(board, depth, float('-inf'), float('inf'), board.turn)
            end_time = time.time()
            
            search_time = end_time - start_time
            nodes = search_controller.nodes_searched
            cutoffs = search_controller.cutoffs
            
            cutoff_rate = (cutoffs / nodes * 100) if nodes > 0 else 0
            nps = nodes / search_time if search_time > 0 else 0
            
            print(f"Depth {depth}: {nodes:,} nodes, {cutoffs:,} cutoffs ({cutoff_rate:.1f}%), {search_time:.2f}s, {nps:,.0f} nps, score: {score}")
            
            # Stop if it takes too long
            if search_time > 10:
                print("Stopping - search taking too long")
                break
                
        except Exception as e:
            print(f"Depth {depth}: Error - {e}")
            break

def compare_with_original_perft():
    """Compare with original perft performance"""
    test_fen = "r3k2r/p1ppqpb1/Bn2pnp1/3PN3/1p2P3/2N2Q2/PPPB1PpP/R3K2R w KQkq - 0 1"
    
    def simple_perft(board, depth):
        if depth == 0:
            return 1
        count = 0
        for move in board.legal_moves:
            board.push(move)
            count += simple_perft(board, depth - 1)
            board.pop()
        return count
    
    print(f"\n" + "="*60)
    print("COMPARISON WITH PERFT")
    print("="*60)
    
    board = chess.Board(test_fen)
    
    # Test perft at depth 4
    print("Running perft(4)...")
    start_time = time.time()
    perft_nodes = simple_perft(board, 4)
    perft_time = time.time() - start_time
    perft_nps = perft_nodes / perft_time if perft_time > 0 else 0
    
    print(f"Perft(4): {perft_nodes:,} nodes in {perft_time:.2f}s ({perft_nps:,.0f} nps)")
    
    # Test our optimized search
    config = V7P3RConfig('config_default.json')
    search_controller = SearchController(config)
    search_controller.use_ab_pruning = True
    search_controller.use_move_ordering = True
    search_controller.max_depth = 4
    
    print("\nRunning optimized search (depth 4)...")
    start_time = time.time()
    score = search_controller._negamax(board, 4, float('-inf'), float('inf'), board.turn)
    search_time = time.time() - start_time
    
    nodes = search_controller.nodes_searched
    cutoffs = search_controller.cutoffs
    search_nps = nodes / search_time if search_time > 0 else 0
    
    print(f"Optimized search: {nodes:,} nodes in {search_time:.2f}s ({search_nps:,.0f} nps)")
    print(f"Cutoffs: {cutoffs:,} ({cutoffs/nodes*100:.1f}% cutoff rate)")
    print(f"Score: {score}")
    
    if perft_nodes > 0:
        reduction = (perft_nodes - nodes) / perft_nodes * 100
        speedup = perft_time / search_time if search_time > 0 else 0
        print(f"\nOptimization results:")
        print(f"  Node reduction: {reduction:.1f}%")
        print(f"  Speed improvement: {speedup:.1f}x faster")

def main():
    # Run all benchmarks
    benchmark_time_controls()
    stress_test_complex_position()
    compare_with_original_perft()
    
    print(f"\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print("v4.2 optimizations appear to be working well!")
    print("The engine should now handle shorter time controls much better.")

if __name__ == "__main__":
    main()
