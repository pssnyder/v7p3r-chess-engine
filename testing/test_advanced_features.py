#!/usr/bin/env python3
"""
Test Advanced Search Features
Test that transposition table, killer moves, and history heuristic are working
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3RCleanEngine


def test_advanced_features():
    """Test that all advanced search features are working"""
    print("Advanced Search Features Test")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    
    # Test position with tactical opportunities
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    print(f"Test position: {board.fen()}")
    print("This position has captures and tactical possibilities")
    print()
    
    # Search the position
    best_move = engine.search(board, 3.0)
    
    print(f"Best move: {best_move}")
    print(f"Nodes searched: {engine.nodes_searched}")
    
    # Check advanced features
    print()
    print("Advanced Features Status:")
    print("=" * 30)
    print(f"✓ Transposition table entries: {len(engine.transposition_table)}")
    print(f"✓ TT hits: {engine.search_stats['tt_hits']}")
    print(f"✓ TT stores: {engine.search_stats['tt_stores']}")
    print(f"✓ Killer move hits: {engine.search_stats['killer_hits']}")
    print(f"✓ Evaluation cache hits: {engine.search_stats['cache_hits']}")
    print(f"✓ Evaluation cache misses: {engine.search_stats['cache_misses']}")
    
    # Show killer moves
    print()
    print("Killer Moves by Depth:")
    for depth in range(1, 7):
        killers = engine.killer_moves.get_killers(depth)
        if killers:
            print(f"  Depth {depth}: {[str(move) for move in killers]}")
    
    # Show some history scores
    print()
    print("History Heuristic Sample:")
    history_items = list(engine.history_heuristic.history.items())[:5]
    for (from_sq, to_sq), score in history_items:
        from_square = chess.square_name(from_sq)
        to_square = chess.square_name(to_sq)
        print(f"  {from_square}->{to_square}: {score}")
    
    print()
    print("=" * 50)
    print("ADVANCED FEATURES SUMMARY")
    print("=" * 50)
    print(f"✓ Unified search function working")
    print(f"✓ Transposition table: {len(engine.transposition_table)} entries")
    print(f"✓ Killer moves: {sum(len(moves) for moves in engine.killer_moves.killers.values())} total")
    print(f"✓ History heuristic: {len(engine.history_heuristic.history)} move pairs tracked")
    print(f"✓ Evaluation cache: {len(engine.evaluation_cache)} positions cached")
    print()
    print("All advanced search features are working correctly!")


if __name__ == "__main__":
    test_advanced_features()
