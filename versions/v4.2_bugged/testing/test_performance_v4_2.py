# test_performance_v4_2.py
"""
Performance testing for V7P3R Chess Engine v4.2
Tests legal move generation, alpha-beta pruning, and move ordering efficiency.
"""

import chess
import time
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from v7p3r_config import V7P3RConfig
from v7p3r_search import SearchController
from v7p3r_move_ordering import MoveOrdering

class PerformanceTester:
    def __init__(self):
        self.config = V7P3RConfig('config_default.json')
        self.search_controller = SearchController(self.config)
        self.move_ordering = MoveOrdering()
        
    def perft(self, board, depth):
        """
        Performance test (perft) - counts all possible positions at given depth.
        This is the gold standard for testing move generation accuracy and speed.
        """
        if depth == 0:
            return 1
        
        move_count = 0
        for move in board.legal_moves:
            board.push(move)
            move_count += self.perft(board, depth - 1)
            board.pop()
        
        return move_count
    
    def test_legal_move_generation(self, fen, depth=6):
        """Test raw legal move generation performance"""
        print(f"\n=== LEGAL MOVE GENERATION TEST ===")
        print(f"Position: {fen}")
        print(f"Depth: {depth}")
        
        board = chess.Board(fen)
        
        start_time = time.time()
        node_count = self.perft(board, depth)
        end_time = time.time()
        
        elapsed = end_time - start_time
        nps = node_count / elapsed if elapsed > 0 else 0
        
        print(f"Nodes searched: {node_count:,}")
        print(f"Time taken: {elapsed:.2f}s")
        print(f"Nodes per second: {nps:,.0f}")
        
        return node_count, elapsed, nps
    
    def test_alpha_beta_pruning(self, fen, depth=6):
        """Test alpha-beta pruning effectiveness"""
        print(f"\n=== ALPHA-BETA PRUNING TEST ===")
        print(f"Position: {fen}")
        print(f"Depth: {depth}")
        
        board = chess.Board(fen)
        
        # Test without pruning
        self.search_controller.use_ab_pruning = False
        self.search_controller.use_move_ordering = False
        self.search_controller.max_depth = depth
        
        start_time = time.time()
        best_move = self.search_controller.find_best_move(board, board.turn)
        end_time = time.time()
        
        nodes_without_pruning = self.search_controller.nodes_searched
        time_without_pruning = end_time - start_time
        
        print(f"WITHOUT pruning:")
        print(f"  Nodes searched: {nodes_without_pruning:,}")
        print(f"  Time taken: {time_without_pruning:.2f}s")
        print(f"  Best move: {best_move}")
        
        # Reset and test with pruning
        self.search_controller.use_ab_pruning = True
        self.search_controller.use_move_ordering = False
        
        start_time = time.time()
        best_move = self.search_controller.find_best_move(board, board.turn)
        end_time = time.time()
        
        nodes_with_pruning = self.search_controller.nodes_searched
        time_with_pruning = end_time - start_time
        cutoffs = self.search_controller.cutoffs
        
        print(f"WITH alpha-beta pruning:")
        print(f"  Nodes searched: {nodes_with_pruning:,}")
        print(f"  Time taken: {time_with_pruning:.2f}s")
        print(f"  Cutoffs: {cutoffs:,}")
        print(f"  Best move: {best_move}")
        
        if nodes_without_pruning > 0:
            pruning_efficiency = ((nodes_without_pruning - nodes_with_pruning) / nodes_without_pruning) * 100
            print(f"  Pruning efficiency: {pruning_efficiency:.1f}% reduction")
        
        return {
            'without_pruning': {'nodes': nodes_without_pruning, 'time': time_without_pruning},
            'with_pruning': {'nodes': nodes_with_pruning, 'time': time_with_pruning, 'cutoffs': cutoffs}
        }
    
    def test_move_ordering_effectiveness(self, fen, depth=6):
        """Test move ordering effectiveness"""
        print(f"\n=== MOVE ORDERING TEST ===")
        print(f"Position: {fen}")
        print(f"Depth: {depth}")
        
        board = chess.Board(fen)
        
        # Test with pruning but without move ordering
        self.search_controller.use_ab_pruning = True
        self.search_controller.use_move_ordering = False
        self.search_controller.max_depth = depth
        
        start_time = time.time()
        best_move = self.search_controller.find_best_move(board, board.turn)
        end_time = time.time()
        
        nodes_without_ordering = self.search_controller.nodes_searched
        time_without_ordering = end_time - start_time
        cutoffs_without_ordering = self.search_controller.cutoffs
        
        print(f"WITH pruning, WITHOUT move ordering:")
        print(f"  Nodes searched: {nodes_without_ordering:,}")
        print(f"  Time taken: {time_without_ordering:.2f}s")
        print(f"  Cutoffs: {cutoffs_without_ordering:,}")
        print(f"  Best move: {best_move}")
        
        # Test with both pruning and move ordering
        self.search_controller.use_move_ordering = True
        
        start_time = time.time()
        best_move = self.search_controller.find_best_move(board, board.turn)
        end_time = time.time()
        
        nodes_with_ordering = self.search_controller.nodes_searched
        time_with_ordering = end_time - start_time
        cutoffs_with_ordering = self.search_controller.cutoffs
        
        print(f"WITH pruning AND move ordering:")
        print(f"  Nodes searched: {nodes_with_ordering:,}")
        print(f"  Time taken: {time_with_ordering:.2f}s")
        print(f"  Cutoffs: {cutoffs_with_ordering:,}")
        print(f"  Best move: {best_move}")
        
        if nodes_without_ordering > 0:
            ordering_efficiency = ((nodes_without_ordering - nodes_with_ordering) / nodes_without_ordering) * 100
            print(f"  Move ordering efficiency: {ordering_efficiency:.1f}% additional reduction")
        
        return {
            'without_ordering': {'nodes': nodes_without_ordering, 'time': time_without_ordering, 'cutoffs': cutoffs_without_ordering},
            'with_ordering': {'nodes': nodes_with_ordering, 'time': time_with_ordering, 'cutoffs': cutoffs_with_ordering}
        }
    
    def test_move_ordering_quality(self, fen):
        """Test the quality of move ordering"""
        print(f"\n=== MOVE ORDERING QUALITY TEST ===")
        print(f"Position: {fen}")
        
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        
        print(f"Total legal moves: {len(legal_moves)}")
        
        # Get ordered moves
        ordered_moves = self.move_ordering.order_moves(board, legal_moves)
        
        print(f"Top 10 ordered moves:")
        for i, move in enumerate(ordered_moves[:10]):
            score = self.move_ordering._score_move(board, move)
            print(f"  {i+1}. {move} (score: {score})")
        
        # Check for hanging piece captures
        hanging_captures = self.move_ordering.get_hanging_piece_captures(board)
        if hanging_captures:
            print(f"\nHanging piece captures found: {len(hanging_captures)}")
            for move in hanging_captures:
                print(f"  {move}")
        else:
            print(f"\nNo hanging piece captures found")
        
        return ordered_moves, hanging_captures

def main():
    # The test position you provided
    test_fen = "r3k2r/p1ppqpb1/Bn2pnp1/3PN3/1p2P3/2N2Q2/PPPB1PpP/R3K2R w KQkq - 0 1"
    
    tester = PerformanceTester()
    
    print("V7P3R Chess Engine Performance Test v4.2")
    print("=" * 50)
    
    # Test 1: Legal move generation (perft)
    print(f"\nTesting with classical perft position...")
    perft_results = tester.test_legal_move_generation(test_fen, depth=4)  # Start with depth 4
    
    # Test 2: Alpha-beta pruning effectiveness
    ab_results = tester.test_alpha_beta_pruning(test_fen, depth=4)
    
    # Test 3: Move ordering effectiveness
    ordering_results = tester.test_move_ordering_effectiveness(test_fen, depth=4)
    
    # Test 4: Move ordering quality
    ordering_quality = tester.test_move_ordering_quality(test_fen)
    
    # Summary
    print(f"\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    
    perft_nodes, perft_time, perft_nps = perft_results
    print(f"Perft (depth 4): {perft_nodes:,} nodes in {perft_time:.2f}s ({perft_nps:,.0f} nps)")
    
    without_pruning = ab_results['without_pruning']['nodes']
    with_pruning = ab_results['with_pruning']['nodes']
    if without_pruning > 0:
        pruning_reduction = ((without_pruning - with_pruning) / without_pruning) * 100
        print(f"Alpha-beta pruning: {pruning_reduction:.1f}% node reduction")
    
    without_ordering = ordering_results['without_ordering']['nodes']
    with_ordering = ordering_results['with_ordering']['nodes']
    if without_ordering > 0:
        ordering_reduction = ((without_ordering - with_ordering) / without_ordering) * 100
        print(f"Move ordering: {ordering_reduction:.1f}% additional node reduction")
    
    total_reduction = 0
    if perft_nodes > 0:
        total_reduction = ((perft_nodes - with_ordering) / perft_nodes) * 100
        print(f"Total optimization: {total_reduction:.1f}% reduction from raw perft")

if __name__ == "__main__":
    main()
