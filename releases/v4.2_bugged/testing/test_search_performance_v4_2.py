# test_search_performance_v4_2.py
"""
Targeted search performance testing for V7P3R Chess Engine v4.2
Tests pure search algorithms without tactical shortcuts.
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

class SearchPerformanceTester:
    def __init__(self):
        self.config = V7P3RConfig('config_default.json')
        self.search_controller = SearchController(self.config)
        self.move_ordering = MoveOrdering()
    
    def perft(self, board, depth):
        """Performance test (perft) - counts all possible positions at given depth."""
        if depth == 0:
            return 1
        
        move_count = 0
        for move in board.legal_moves:
            board.push(move)
            move_count += self.perft(board, depth - 1)
            board.pop()
        
        return move_count
    
    def test_raw_negamax_search(self, fen, depth=4):
        """Test the raw negamax search bypassing tactical shortcuts"""
        print(f"\n=== RAW NEGAMAX SEARCH TEST ===")
        print(f"Position: {fen}")
        print(f"Search depth: {depth}")
        
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        
        # Test without pruning, without move ordering
        print(f"\n1. BRUTE FORCE (no pruning, no ordering):")
        self.search_controller.use_ab_pruning = False
        self.search_controller.use_move_ordering = False
        self.search_controller.max_depth = depth
        # Disable time checking for performance tests
        self.search_controller.start_time = 0
        
        start_time = time.time()
        best_move, move_scores = self.search_controller._negamax_root(board, board.turn, legal_moves, debug=True)
        end_time = time.time()
        
        nodes_brute = self.search_controller.nodes_searched
        time_brute = end_time - start_time
        
        print(f"  Nodes searched: {nodes_brute:,}")
        print(f"  Time taken: {time_brute:.2f}s")
        print(f"  NPS: {nodes_brute/time_brute:,.0f}" if time_brute > 0 else "  NPS: N/A")
        print(f"  Best move: {best_move}")
        
        # Test with pruning, without move ordering  
        print(f"\n2. WITH ALPHA-BETA PRUNING:")
        self.search_controller.use_ab_pruning = True
        self.search_controller.use_move_ordering = False
        # Disable time checking for performance tests
        self.search_controller.start_time = 0
        
        start_time = time.time()
        best_move, move_scores = self.search_controller._negamax_root(board, board.turn, legal_moves, debug=True)
        end_time = time.time()
        
        nodes_pruned = self.search_controller.nodes_searched
        time_pruned = end_time - start_time
        cutoffs = self.search_controller.cutoffs
        
        print(f"  Nodes searched: {nodes_pruned:,}")
        print(f"  Time taken: {time_pruned:.2f}s")
        print(f"  NPS: {nodes_pruned/time_pruned:,.0f}" if time_pruned > 0 else "  NPS: N/A")
        print(f"  Cutoffs: {cutoffs:,}")
        print(f"  Best move: {best_move}")
        
        if nodes_brute > 0:
            pruning_reduction = ((nodes_brute - nodes_pruned) / nodes_brute) * 100
            print(f"  Pruning efficiency: {pruning_reduction:.1f}% node reduction")
        
        # Test with pruning and move ordering
        print(f"\n3. WITH PRUNING + MOVE ORDERING:")
        self.search_controller.use_ab_pruning = True
        self.search_controller.use_move_ordering = True
        # Disable time checking for performance tests
        self.search_controller.start_time = 0
        
        start_time = time.time()
        best_move, move_scores = self.search_controller._negamax_root(board, board.turn, legal_moves, debug=True)
        end_time = time.time()
        
        nodes_ordered = self.search_controller.nodes_searched
        time_ordered = end_time - start_time
        cutoffs_ordered = self.search_controller.cutoffs
        
        print(f"  Nodes searched: {nodes_ordered:,}")
        print(f"  Time taken: {time_ordered:.2f}s")
        print(f"  NPS: {nodes_ordered/time_ordered:,.0f}" if time_ordered > 0 else "  NPS: N/A")
        print(f"  Cutoffs: {cutoffs_ordered:,}")
        print(f"  Best move: {best_move}")
        
        if nodes_pruned > 0:
            ordering_reduction = ((nodes_pruned - nodes_ordered) / nodes_pruned) * 100
            print(f"  Move ordering efficiency: {ordering_reduction:.1f}% additional reduction")
        
        return {
            'brute_force': {'nodes': nodes_brute, 'time': time_brute},
            'with_pruning': {'nodes': nodes_pruned, 'time': time_pruned, 'cutoffs': cutoffs},
            'with_ordering': {'nodes': nodes_ordered, 'time': time_ordered, 'cutoffs': cutoffs_ordered}
        }
    
    def test_move_ordering_effectiveness(self, fen):
        """Test move ordering quality and effectiveness"""
        print(f"\n=== MOVE ORDERING ANALYSIS ===")
        print(f"Position: {fen}")
        
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        
        print(f"Total legal moves: {len(legal_moves)}")
        
        # Get moves ordered by our algorithm
        ordered_moves = self.move_ordering.order_moves(board, legal_moves)
        
        print(f"\nTop 15 moves by our ordering:")
        for i, move in enumerate(ordered_moves[:15]):
            score = self.move_ordering._score_move(board, move)
            
            # Analyze move type
            move_type = []
            if board.is_capture(move):
                move_type.append("capture")
            if board.is_check():
                board_copy = board.copy()
                board_copy.push(move)
                if board_copy.is_check():
                    move_type.append("check")
            if move.promotion:
                move_type.append("promotion")
            if board.is_castling(move):
                move_type.append("castling")
            
            type_str = ", ".join(move_type) if move_type else "quiet"
            print(f"  {i+1:2d}. {move} (score: {score:,}) [{type_str}]")
        
        # Test hanging captures
        hanging_captures = self.move_ordering.get_hanging_piece_captures(board)
        if hanging_captures:
            print(f"\nHanging piece captures: {len(hanging_captures)}")
            for move in hanging_captures:
                print(f"  {move}")
        
        return ordered_moves, hanging_captures
    
    def test_time_performance(self, fen, target_time=5.0):
        """Test what depth we can achieve in a target time"""
        print(f"\n=== TIME PERFORMANCE TEST ===")
        print(f"Position: {fen}")
        print(f"Target time: {target_time}s")
        
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        
        # Configure for optimal performance
        self.search_controller.use_ab_pruning = True
        self.search_controller.use_move_ordering = True
        
        max_depth_achieved = 0
        
        for depth in range(1, 10):  # Test depths 1-9
            self.search_controller.max_depth = depth
            # Disable time checking for performance tests
            self.search_controller.start_time = 0
            
            start_time = time.time()
            best_move, move_scores = self.search_controller._negamax_root(board, board.turn, legal_moves)
            end_time = time.time()
            
            search_time = end_time - start_time
            nodes = self.search_controller.nodes_searched
            
            print(f"Depth {depth}: {nodes:,} nodes in {search_time:.2f}s ({nodes/search_time:,.0f} nps)")
            
            if search_time <= target_time:
                max_depth_achieved = depth
            else:
                break
        
        print(f"\nMaximum depth achievable in {target_time}s: {max_depth_achieved}")
        return max_depth_achieved

def main():
    # The test position you provided - a complex middle game position
    test_fen = "r3k2r/p1ppqpb1/Bn2pnp1/3PN3/1p2P3/2N2Q2/PPPB1PpP/R3K2R w KQkq - 0 1"
    
    # Also test a quieter position without hanging pieces
    quiet_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    
    tester = SearchPerformanceTester()
    
    print("V7P3R Chess Engine Search Performance Test v4.2")
    print("=" * 60)
    
    # Test 1: Perft baseline
    print(f"\nPERFT BASELINE:")
    board = chess.Board(test_fen)
    start_time = time.time()
    perft_nodes = tester.perft(board, 4)
    perft_time = time.time() - start_time
    print(f"Perft(4): {perft_nodes:,} nodes in {perft_time:.2f}s ({perft_nodes/perft_time:,.0f} nps)")
    
    # Test 2: Search performance with complex position
    search_results = tester.test_raw_negamax_search(test_fen, depth=4)
    
    # Test 3: Move ordering analysis
    tester.test_move_ordering_effectiveness(test_fen)
    
    # Test 4: Time performance
    max_depth = tester.test_time_performance(test_fen, target_time=5.0)
    
    # Test 5: Quieter position for comparison
    print(f"\n" + "=" * 60)
    print("COMPARISON WITH QUIETER POSITION")
    print("=" * 60)
    search_results_quiet = tester.test_raw_negamax_search(quiet_fen, depth=4)
    
    # Summary
    print(f"\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    complex_results = search_results['with_ordering']
    quiet_results = search_results_quiet['with_ordering']
    
    print(f"Complex position (test FEN):")
    print(f"  Depth 4 search: {complex_results['nodes']:,} nodes in {complex_results['time']:.2f}s")
    print(f"  Effective branching factor: {(complex_results['nodes']/perft_nodes)*100:.1f}% of perft")
    print(f"  Max depth in 5s: {max_depth}")
    
    print(f"\nQuiet position (opening):")
    print(f"  Depth 4 search: {quiet_results['nodes']:,} nodes in {quiet_results['time']:.2f}s")
    
    if complex_results['time'] > 0:
        time_efficiency = perft_time / complex_results['time']
        print(f"\nOptimization factor: {time_efficiency:.1f}x faster than perft")

if __name__ == "__main__":
    main()
