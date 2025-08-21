#!/usr/bin/env python3
"""
Test a real mate in 1 position
"""

import chess
from v7p3r import V7P3REvaluationEngine

def test_real_mate_in_1():
    """Test a position where White can deliver mate in 1"""
    
    # Scholar's mate setup - White can play Qf7#
    fen = "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 1 2"
    board = chess.Board(fen)
    
    print(f"Position: {fen}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"Is check: {board.is_check()}")
    print(f"Is game over: {board.is_game_over()}")
    
    # Check if Qf7 is legal and leads to mate
    potential_mate_move = chess.Move.from_uci("d1f3")  # First bring queen out
    if board.is_legal(potential_mate_move):
        print(f"Qf3 is legal")
        board.push(potential_mate_move)
        print(f"After Qf3, Black in check: {board.is_check()}")
        board.pop()
    
    # Better mate in 1 position: 
    fen2 = "rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 3"
    board2 = chess.Board(fen2)
    print(f"\nBetter mate position: {fen2}")
    
    # Check if Qf7# works
    mate_move = chess.Move.from_uci("h5f7")
    if board2.is_legal(mate_move):
        print(f"Qf7 is legal")
        board2.push(mate_move)
        print(f"After Qf7 - Check: {board2.is_check()}, Mate: {board2.is_checkmate()}")
        board2.pop()
    
    # Test engine on this position
    print(f"\nTesting engine on mate in 1 position...")
    engine = V7P3REvaluationEngine()
    engine.depth = 4
    
    start_time = time.time()
    best_move = engine.find_best_move(board2, time_limit=3.0)
    end_time = time.time()
    
    print(f"Engine move: {best_move}")
    print(f"Time: {end_time - start_time:.2f}s")
    print(f"Nodes: {engine.nodes_searched}")
    
    if best_move and board2.is_legal(best_move):
        board2.push(best_move)
        print(f"After engine move - Check: {board2.is_check()}, Mate: {board2.is_checkmate()}")
        if board2.is_checkmate():
            print("✓ Engine found the mate!")
        else:
            print("✗ Engine didn't find mate")
        board2.pop()

if __name__ == "__main__":
    import time
    test_real_mate_in_1()
