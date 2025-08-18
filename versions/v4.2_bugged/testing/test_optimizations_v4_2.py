# test_optimizations_v4_2.py
"""
Test the v4.2 optimizations to measure performance improvements.
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

def test_position_without_hanging_pieces():
    """Test with a position that doesn't have obvious hanging pieces"""
    # Starting position after 1.e4 e5 2.Nf3 Nc6 3.Bb5 - Spanish opening
    spanish_fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    
    config = V7P3RConfig('config_default.json')
    search_controller = SearchController(config)
    
    print("Testing Spanish Opening Position (no hanging pieces)")
    print(f"FEN: {spanish_fen}")
    
    board = chess.Board(spanish_fen)
    legal_moves = list(board.legal_moves)
    print(f"Legal moves: {len(legal_moves)}")
    
    # Test search at depth 4
    search_controller.max_depth = 4
    search_controller.use_ab_pruning = True
    search_controller.use_move_ordering = True
    
    start_time = time.time()
    best_move, move_scores = search_controller._negamax_root(board, board.turn, legal_moves)
    end_time = time.time()
    
    search_time = end_time - start_time
    nodes = search_controller.nodes_searched
    cutoffs = search_controller.cutoffs
    
    print(f"\nSearch Results:")
    print(f"Best move: {best_move}")
    print(f"Nodes searched: {nodes:,}")
    print(f"Cutoffs: {cutoffs:,}")
    print(f"Time: {search_time:.2f}s")
    print(f"NPS: {nodes/search_time:,.0f}" if search_time > 0 else "NPS: N/A")
    
    if len(move_scores) > 0:
        print(f"\nTop 5 moves:")
        for i, (move, score) in enumerate(move_scores[:5]):
            print(f"  {i+1}. {move}: {score}")
    
    return nodes, search_time, cutoffs

def test_complex_position():
    """Test with the original complex position"""
    complex_fen = "r3k2r/p1ppqpb1/Bn2pnp1/3PN3/1p2P3/2N2Q2/PPPB1PpP/R3K2R w KQkq - 0 1"
    
    config = V7P3RConfig('config_default.json')
    search_controller = SearchController(config)
    
    print("\n" + "="*60)
    print("Testing Complex Position (with tactical opportunities)")
    print(f"FEN: {complex_fen}")
    
    board = chess.Board(complex_fen)
    legal_moves = list(board.legal_moves)
    print(f"Legal moves: {len(legal_moves)}")
    
    # Check for hanging pieces first
    move_ordering = MoveOrdering()
    hanging_captures = move_ordering.get_hanging_piece_captures(board)
    if hanging_captures:
        print(f"Hanging captures found: {[str(move) for move in hanging_captures]}")
        print("Engine will likely choose hanging capture immediately")
        return 0, 0, 0
    
    # Test search at depth 4
    search_controller.max_depth = 4
    search_controller.use_ab_pruning = True
    search_controller.use_move_ordering = True
    
    start_time = time.time()
    best_move, move_scores = search_controller._negamax_root(board, board.turn, legal_moves)
    end_time = time.time()
    
    search_time = end_time - start_time
    nodes = search_controller.nodes_searched
    cutoffs = search_controller.cutoffs
    
    print(f"\nSearch Results:")
    print(f"Best move: {best_move}")
    print(f"Nodes searched: {nodes:,}")
    print(f"Cutoffs: {cutoffs:,}")
    print(f"Time: {search_time:.2f}s")
    print(f"NPS: {nodes/search_time:,.0f}" if search_time > 0 else "NPS: N/A")
    
    return nodes, search_time, cutoffs

def test_endgame_position():
    """Test with an endgame position"""
    endgame_fen = "8/8/3k4/3p4/3P4/3K4/8/8 w - - 0 1"
    
    config = V7P3RConfig('config_default.json')
    search_controller = SearchController(config)
    
    print("\n" + "="*60)
    print("Testing King and Pawn Endgame")
    print(f"FEN: {endgame_fen}")
    
    board = chess.Board(endgame_fen)
    legal_moves = list(board.legal_moves)
    print(f"Legal moves: {len(legal_moves)}")
    
    # Test search at depth 6 (endgames can search deeper)
    search_controller.max_depth = 6
    search_controller.use_ab_pruning = True
    search_controller.use_move_ordering = True
    
    start_time = time.time()
    best_move, move_scores = search_controller._negamax_root(board, board.turn, legal_moves)
    end_time = time.time()
    
    search_time = end_time - start_time
    nodes = search_controller.nodes_searched
    cutoffs = search_controller.cutoffs
    
    print(f"\nSearch Results:")
    print(f"Best move: {best_move}")
    print(f"Nodes searched: {nodes:,}")
    print(f"Cutoffs: {cutoffs:,}")
    print(f"Time: {search_time:.2f}s")
    print(f"NPS: {nodes/search_time:,.0f}" if search_time > 0 else "NPS: N/A")
    
    return nodes, search_time, cutoffs

def main():
    print("V7P3R Chess Engine v4.2 Optimization Test")
    print("=" * 50)
    
    # Test different types of positions
    spanish_nodes, spanish_time, spanish_cutoffs = test_position_without_hanging_pieces()
    complex_nodes, complex_time, complex_cutoffs = test_complex_position()
    endgame_nodes, endgame_time, endgame_cutoffs = test_endgame_position()
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    if spanish_time > 0:
        print(f"Spanish Opening (depth 4): {spanish_nodes:,} nodes, {spanish_cutoffs:,} cutoffs, {spanish_time:.2f}s")
        print(f"  Pruning efficiency: {spanish_cutoffs/spanish_nodes*100:.1f}% cutoff rate")
    
    if complex_time > 0:
        print(f"Complex Position (depth 4): {complex_nodes:,} nodes, {complex_cutoffs:,} cutoffs, {complex_time:.2f}s")
        print(f"  Pruning efficiency: {complex_cutoffs/complex_nodes*100:.1f}% cutoff rate")
    
    if endgame_time > 0:
        print(f"Endgame (depth 6): {endgame_nodes:,} nodes, {endgame_cutoffs:,} cutoffs, {endgame_time:.2f}s")
        print(f"  Pruning efficiency: {endgame_cutoffs/endgame_nodes*100:.1f}% cutoff rate")
    
    # Check if we're meeting our performance targets
    target_nodes_5s = 1_113_895  # From earlier test
    
    if spanish_time > 0:
        projected_5s_nodes = (spanish_nodes / spanish_time) * 5
        print(f"\nSpanish position - Projected nodes in 5s: {projected_5s_nodes:,.0f}")
        if projected_5s_nodes <= target_nodes_5s:
            print("  ✓ Meets 5-second time control target!")
        else:
            print(f"  ✗ Exceeds target by {projected_5s_nodes - target_nodes_5s:,.0f} nodes")

if __name__ == "__main__":
    main()
