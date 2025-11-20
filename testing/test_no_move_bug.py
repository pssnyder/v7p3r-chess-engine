#!/usr/bin/env python3
"""
V16.1 "No Move Found" Bug Diagnostic
Tests positions where engine returns None instead of a legal move
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

def test_position(engine, name, fen, description):
    """Test a single position and report detailed diagnostics"""
    print(f"\n{'='*70}")
    print(f"Position: {name}")
    print(f"Description: {description}")
    print(f"FEN: {fen}")
    print('='*70)
    
    board = chess.Board(fen)
    engine.board = board
    
    # Check basic board state
    print(f"\nBoard State:")
    print(f"  - Turn: {'White' if board.turn else 'Black'}")
    print(f"  - Legal moves: {board.legal_moves.count()}")
    print(f"  - In check: {board.is_check()}")
    print(f"  - Is checkmate: {board.is_checkmate()}")
    print(f"  - Is stalemate: {board.is_stalemate()}")
    print(f"  - Is game over: {board.is_game_over()}")
    
    # Show some legal moves
    legal_moves = list(board.legal_moves)
    print(f"\nFirst 10 legal moves:")
    for i, move in enumerate(legal_moves[:10]):
        print(f"    {i+1}. {board.san(move)} ({move.uci()})")
    
    # Test evaluation
    print(f"\nEvaluation:")
    try:
        score = engine._evaluate_position(board)
        print(f"  - Static eval: {score:+d}cp")
    except Exception as e:
        print(f"  - ERROR in evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    # Try to get best move with detailed output
    print(f"\nSearching for best move (5 seconds)...")
    print("-" * 70)
    
    try:
        best_move = engine.get_best_move(time_left=5.0)
    except Exception as e:
        print("-" * 70)
        print(f"\n[EXCEPTION] Error during search: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("-" * 70)
    
    if best_move:
        print(f"\n[SUCCESS] Best move: {board.san(best_move)} ({best_move.uci()})")
        
        # Verify it's legal
        if best_move in board.legal_moves:
            print(f"  - Move is LEGAL")
        else:
            print(f"  - ERROR: Move is ILLEGAL!")
            
    else:
        print(f"\n[FAILURE] *** NO MOVE FOUND ***")
        print(f"  - This is the BUG we're investigating!")
        print(f"  - Board has {board.legal_moves.count()} legal moves available")
        print(f"  - Engine should return one of them")
    
    print("\n")

def run_diagnostic_tests():
    """Run comprehensive diagnostic tests"""
    print("\n" + "="*70)
    print("  V16.1 'NO MOVE FOUND' BUG DIAGNOSTIC")
    print("="*70)
    print("\nTesting positions where engine may fail to return a move.")
    print("This is likely the cause of Arena's 'illegal move' error.")
    
    # Initialize engine
    print("\nInitializing engine...")
    engine = V7P3REngine(max_depth=6, tt_size_mb=128, tablebase_path="")
    print("Engine ready.\n")
    
    # Test positions - starting with the one that failed
    test_positions = [
        ("Mate threat defense",
         "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
         "Black must defend against Qxf7#. Previous test showed 'No move found'"),
        
        ("Under check - must respond",
         "rnbqkb1r/pppp1ppp/5n2/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3",
         "Black is in check from Qh5, must block or move king"),
        
        ("Material down but playable",
         "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
         "Black is down a pawn but has moves"),
        
        ("Complex position with threats",
         "r2qkb1r/ppp2ppp/2n1bn2/3pp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 7",
         "Both sides have threats, many legal moves"),
        
        ("Endgame with few pieces",
         "8/5k2/8/3K4/8/8/8/8 w - - 0 1",
         "King and king only - should draw but needs to move"),
        
        ("Opening book exit position",
         "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5",
         "After book moves, engine searches on its own"),
        
        ("Forced sequence position",
         "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
         "Multiple captures available"),
        
        ("Queen trade opportunity",
         "rnbqkb1r/ppp2ppp/3p1n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 0 5",
         "Black can consider development or queen trade"),
        
        ("Tactical position - pin",
         "r1bqk2r/pppp1ppp/2n2n2/2b1p3/1PB1P3/5N2/P1PP1PPP/RNBQK2R b KQkq b3 0 6",
         "Black knight on c6 is pinned to the king"),
        
        ("Normal middlegame",
         "r1bq1rk1/ppp1ppbp/2np1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 9",
         "Standard middlegame position, many options"),
    ]
    
    # Run all tests
    for name, fen, description in test_positions:
        test_position(engine, name, fen, description)
        input("Press Enter to continue to next position...")
    
    # Final summary
    print("\n" + "="*70)
    print("  DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\nReview the output above to identify:")
    print("  1. Which positions return 'No move found'")
    print("  2. Any error messages during evaluation or search")
    print("  3. Patterns in the positions that fail")
    print("  4. Clues about what's causing the bug")
    print("\n")

if __name__ == "__main__":
    run_diagnostic_tests()
