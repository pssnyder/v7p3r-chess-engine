#!/usr/bin/env python3
"""
Simple test script for V7P3R Chess Engine
"""

import chess
import time
from v7p3r import V7P3REvaluationEngine

def test_basic_functionality():
    """Test basic engine functionality"""
    print("Testing V7P3R Chess Engine...")
    
    # Create engine instance
    engine = V7P3REvaluationEngine()
    
    # Test with starting position
    board = chess.Board()
    print(f"Starting position: {board.fen()}")
    
    # Test evaluation
    print("\nTesting evaluation...")
    eval_score = engine.evaluate_position(board)
    print(f"Starting position evaluation: {eval_score}")
    
    # Test evaluation from perspective
    white_perspective = engine.evaluate_position_from_perspective(board, chess.WHITE)
    black_perspective = engine.evaluate_position_from_perspective(board, chess.BLACK)
    print(f"White perspective: {white_perspective}")
    print(f"Black perspective: {black_perspective}")
    
    # Test move finding
    print("\nTesting move search...")
    start_time = time.time()
    best_move = engine.find_best_move(board, time_limit=2.0)
    end_time = time.time()
    
    print(f"Best move found: {best_move}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Nodes searched: {engine.nodes_searched}")
    
    # Test if move is legal
    if best_move and board.is_legal(best_move):
        print("✓ Move is legal")
        board.push(best_move)
        print(f"Position after move: {board.fen()}")
    else:
        print("✗ Move is not legal or null")
    
    # Test a few more moves
    for i in range(3):
        if board.is_game_over():
            break
        print(f"\nMove {i+2}:")
        start_time = time.time()
        move = engine.find_best_move(board, time_limit=1.0)
        end_time = time.time()
        
        if move and board.is_legal(move):
            board.push(move)
            print(f"Move: {move}, Time: {end_time - start_time:.2f}s, Nodes: {engine.nodes_searched}")
        else:
            print("No legal move found")
            break
    
    print(f"\nFinal position: {board.fen()}")
    print("Basic functionality test completed!")

def test_position_evaluation():
    """Test evaluation on various positions"""
    print("\n" + "="*50)
    print("Testing position evaluation...")
    
    engine = V7P3REvaluationEngine()
    
    test_positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Queen sacrifice", "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"),
        ("Endgame", "8/8/8/8/8/8/PPP5/R3K3 w Q - 0 1"),
        ("White mate in 1", "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 1 2"),  # White can play Qh5# 
        ("Black already checkmated", "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")  # Black has no moves
    ]
    
    for name, fen in test_positions:
        print(f"\n{name}: {fen}")
        try:
            board = chess.Board(fen)
            eval_score = engine.evaluate_position(board)
            print(f"Evaluation: {eval_score:.2f}")
            
            # Get best move
            start_time = time.time()
            best_move = engine.find_best_move(board, time_limit=1.0)
            end_time = time.time()
            
            if best_move:
                print(f"Best move: {best_move} (Time: {end_time - start_time:.2f}s)")
            else:
                print("No move found")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_basic_functionality()
    test_position_evaluation()
    print("\nAll tests completed!")
