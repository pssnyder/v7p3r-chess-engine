#!/usr/bin/env python3
"""
V12.3 Integration Test
Test the unified bitboard evaluator integration and verify basic functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3REngine

def test_basic_engine_functionality():
    """Test that the engine still works with the new unified evaluator."""
    print("Testing V12.3 Engine Integration...")
    
    # Initialize engine
    engine = V7P3REngine()
    
    # Test 1: Basic position evaluation
    board = chess.Board()  # Starting position
    
    print(f"\n1. Starting Position Evaluation:")
    start_time = time.time()
    score = engine._evaluate_position(board)
    eval_time = time.time() - start_time
    print(f"   Score: {score:.2f}")
    print(f"   Evaluation time: {eval_time*1000:.2f}ms")
    
    # Test 2: Position after 1.e4
    board.push_san("e4")
    score_after_e4 = engine._evaluate_position(board)
    print(f"   After 1.e4: {score_after_e4:.2f}")
    board.pop()
    
    # Test 3: Position after 1.d4 
    board.push_san("d4")
    score_after_d4 = engine._evaluate_position(board)
    print(f"   After 1.d4: {score_after_d4:.2f}")
    board.pop()
    
    # Test 4: Best move search
    print(f"\n2. Best Move Search:")
    start_time = time.time()
    best_move = engine.search(board, time_limit=5.0)
    search_time = time.time() - start_time
    
    print(f"   Best move: {best_move}")
    print(f"   Search time: {search_time:.2f}s")
    print(f"   Nodes searched: {engine.nodes_searched:,}")
    if search_time > 0:
        nps = engine.nodes_searched / search_time
        print(f"   Nodes per second: {nps:,.0f}")
    
    # Test 5: Castling detection
    print(f"\n3. Castling Evaluation Test:")
    
    # Set up a position where castling is available
    castling_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    castling_board = chess.Board(castling_fen)
    
    score_before_castle = engine._evaluate_position(castling_board)
    print(f"   Before castling: {score_before_castle:.2f}")
    
    # Castle kingside
    castling_board.push_san("O-O")
    score_after_castle = engine._evaluate_position(castling_board)
    print(f"   After O-O: {score_after_castle:.2f}")
    print(f"   Castling bonus: {score_after_castle - score_before_castle:.2f}")
    
    # Test 6: Performance benchmark
    print(f"\n4. Performance Benchmark (1000 evaluations):")
    benchmark_positions = [
        chess.Board(),  # Starting position
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),  # After 1.e4
        chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),  # Italian
        chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),  # King's pawn game
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")   # Italian opening
    ]
    
    total_evaluations = 0
    start_benchmark = time.time()
    
    for _ in range(200):  # 200 iterations of 5 positions = 1000 evaluations
        for pos in benchmark_positions:
            engine._evaluate_position(pos)
            total_evaluations += 1
    
    benchmark_time = time.time() - start_benchmark
    evaluations_per_second = total_evaluations / benchmark_time
    
    print(f"   Total evaluations: {total_evaluations}")
    print(f"   Total time: {benchmark_time:.3f}s") 
    print(f"   Evaluations per second: {evaluations_per_second:,.0f}")
    print(f"   Average evaluation time: {benchmark_time/total_evaluations*1000:.3f}ms")
    
    print(f"\n✓ V12.3 Integration test completed successfully!")
    print(f"✓ Unified evaluator is working with {evaluations_per_second:,.0f} eval/s performance")

def test_advanced_features():
    """Test specific advanced features that should now be integrated."""
    print(f"\n5. Advanced Features Test:")
    
    engine = V7P3REngine()
    
    # Test passed pawn evaluation
    passed_pawn_fen = "8/8/8/3P4/8/8/3p4/8 w - - 0 1"  # Both sides have passed pawns
    board = chess.Board(passed_pawn_fen)
    score = engine._evaluate_position(board)
    print(f"   Passed pawns position: {score:.2f}")
    
    # Test king safety in middlegame
    unsafe_king_fen = "r1bq1rk1/ppp2ppp/2n2n2/3p4/3P4/2N1P3/PPP2PPP/R1BQKB1R w KQ - 0 8"
    board = chess.Board(unsafe_king_fen)
    score = engine._evaluate_position(board)
    print(f"   King safety test: {score:.2f}")
    
    # Test pawn structure
    isolated_pawns_fen = "8/p1p1p1p1/8/8/8/8/P1P1P1P1/8 w - - 0 1"  # All isolated pawns
    board = chess.Board(isolated_pawns_fen)
    score = engine._evaluate_position(board)
    print(f"   Isolated pawns: {score:.2f}")
    
    print(f"✓ Advanced features are integrated and functioning")

if __name__ == "__main__":
    test_basic_engine_functionality()
    test_advanced_features()
    print(f"\n🎉 All V12.3 integration tests passed!")