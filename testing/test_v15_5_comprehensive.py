#!/usr/bin/env python3
"""
Comprehensive V15.5 Playing Style Analysis
Tests moves and evaluations under various time pressures and positions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3REngine


def analyze_position(engine, board, description, time_left=0, increment=0):
    """Analyze a position and show the move choice"""
    print(f"\n{'='*70}")
    print(f"Position: {description}")
    print(f"FEN: {board.fen()}")
    print(f"Time: {time_left}s + {increment}s increment")
    print(f"{'='*70}")
    print(board)
    print()
    
    # Get evaluation
    eval_score = engine._evaluate_position(board)
    print(f"Static Eval: {eval_score:+.2f} cp")
    
    # Get best move with timing
    start = time.time()
    best_move = engine.get_best_move(time_left=time_left, increment=increment)
    elapsed = time.time() - start
    
    if best_move:
        print(f"\nChosen Move: {best_move.uci()} ({board.san(best_move)})")
        print(f"Search Time: {elapsed:.2f}s")
        print(f"Nodes: {engine.nodes_searched:,}")
        
        # Show what happens after the move
        board.push(best_move)
        new_eval = engine._evaluate_position(board)
        board.pop()
        print(f"Expected Eval After Move: {new_eval:+.2f} cp")
        
        # Check if move hangs pieces or loses material
        board.push(best_move)
        is_capture = board.is_capture(best_move)
        board.pop()
        
        if is_capture:
            print(f"Move Type: CAPTURE")
        else:
            print(f"Move Type: Quiet move")
    else:
        print("No move available (game over?)")
    
    return best_move


def test_opening_positions():
    """Test opening moves under different time controls"""
    print("\n" + "="*70)
    print("OPENING POSITIONS TEST")
    print("="*70)
    
    engine = V7P3REngine()
    
    test_cases = [
        {
            "name": "Starting Position - Bullet (10s total)",
            "fen": chess.STARTING_FEN,
            "time": 10,
            "increment": 0
        },
        {
            "name": "Starting Position - Blitz (120s total)",
            "fen": chess.STARTING_FEN,
            "time": 120,
            "increment": 1
        },
        {
            "name": "Starting Position - Rapid (600s total)",
            "fen": chess.STARTING_FEN,
            "time": 600,
            "increment": 5
        },
        {
            "name": "After 1.e4 - Black's Response (Bullet)",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "time": 9,
            "increment": 0
        },
        {
            "name": "After 1.e4 c6 - Caro-Kann (Blitz)",
            "fen": "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "time": 115,
            "increment": 1
        },
    ]
    
    for test in test_cases:
        engine.board = chess.Board(test["fen"])
        analyze_position(engine, engine.board, test["name"], 
                        test["time"], test["increment"])


def test_tactical_positions():
    """Test tactical awareness in critical positions"""
    print("\n" + "="*70)
    print("TACTICAL POSITIONS TEST")
    print("="*70)
    
    engine = V7P3REngine()
    
    test_cases = [
        {
            "name": "Mate in 2 - Should find it",
            "fen": "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1",
            "time": 30,
            "increment": 0
        },
        {
            "name": "Free Queen - Should capture",
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPQPPP/RNB1KBNR b KQkq - 0 1",
            "time": 30,
            "increment": 0
        },
        {
            "name": "Hanging Rook - Should capture or defend",
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            "time": 30,
            "increment": 0
        },
        {
            "name": "Fork Opportunity - Can fork king and queen",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 0 1",
            "time": 30,
            "increment": 0
        },
    ]
    
    for test in test_cases:
        engine.board = chess.Board(test["fen"])
        analyze_position(engine, engine.board, test["name"], 
                        test["time"], test["increment"])


def test_material_imbalance():
    """Test how engine handles material imbalances"""
    print("\n" + "="*70)
    print("MATERIAL IMBALANCE TEST")
    print("="*70)
    
    engine = V7P3REngine()
    
    test_cases = [
        {
            "name": "Up a Queen - Should play confidently",
            "fen": "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "time": 30,
            "increment": 0
        },
        {
            "name": "Down a Queen - Should be defensive",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
            "time": 30,
            "increment": 0
        },
        {
            "name": "Rook vs Knight+Bishop - Complex",
            "fen": "4k3/8/8/8/8/8/3R4/4K3 w - - 0 1",
            "time": 30,
            "increment": 0
        },
        {
            "name": "Bishop Pair vs Knights - Should value bishops",
            "fen": "rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1",
            "time": 30,
            "increment": 0
        },
    ]
    
    for test in test_cases:
        engine.board = chess.Board(test["fen"])
        analyze_position(engine, engine.board, test["name"], 
                        test["time"], test["increment"])


def test_endgame_positions():
    """Test endgame technique"""
    print("\n" + "="*70)
    print("ENDGAME POSITIONS TEST")
    print("="*70)
    
    engine = V7P3REngine()
    
    test_cases = [
        {
            "name": "King and Pawn vs King - Should push pawn",
            "fen": "8/8/8/8/4k3/8/4P3/4K3 w - - 0 1",
            "time": 30,
            "increment": 0
        },
        {
            "name": "Rook Endgame - Activity important",
            "fen": "8/5k2/8/8/8/8/R7/6K1 w - - 0 1",
            "time": 30,
            "increment": 0
        },
        {
            "name": "Queen vs Pawn - Should stop pawn",
            "fen": "8/8/8/8/4k3/8/4p3/4KQ2 w - - 0 1",
            "time": 30,
            "increment": 0
        },
    ]
    
    for test in test_cases:
        engine.board = chess.Board(test["fen"])
        analyze_position(engine, engine.board, test["name"], 
                        test["time"], test["increment"])


def test_time_pressure():
    """Test behavior under extreme time pressure"""
    print("\n" + "="*70)
    print("TIME PRESSURE TEST")
    print("="*70)
    
    engine = V7P3REngine()
    
    test_cases = [
        {
            "name": "Complex Position - 1 second",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            "time": 1,
            "increment": 0
        },
        {
            "name": "Complex Position - 5 seconds",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            "time": 5,
            "increment": 0
        },
        {
            "name": "Complex Position - 30 seconds",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            "time": 30,
            "increment": 1
        },
        {
            "name": "Complex Position - 300 seconds (no time limit)",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            "time": 300,
            "increment": 5
        },
    ]
    
    for test in test_cases:
        engine.board = chess.Board(test["fen"])
        analyze_position(engine, engine.board, test["name"], 
                        test["time"], test["increment"])


def test_book_transitions():
    """Test how engine transitions out of book"""
    print("\n" + "="*70)
    print("OPENING BOOK TRANSITION TEST")
    print("="*70)
    
    engine = V7P3REngine()
    
    # Play first few moves following book
    moves = ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3"]
    
    print("\nPlaying opening moves:")
    engine.board = chess.Board()
    for move_uci in moves:
        move = chess.Move.from_uci(move_uci)
        print(f"{engine.board.fullmove_number}. {engine.board.san(move)}", end=" ")
        engine.board.push(move)
    print("\n")
    
    analyze_position(engine, engine.board, 
                    "After opening - First move out of book", 
                    120, 1)


def play_sample_game():
    """Play a quick sample game to see overall behavior"""
    print("\n" + "="*70)
    print("SAMPLE GAME (First 10 moves)")
    print("="*70)
    
    engine_white = V7P3REngine()
    engine_black = V7P3REngine()
    
    board = chess.Board()
    
    for move_num in range(10):
        if board.is_game_over():
            break
            
        print(f"\n--- Move {board.fullmove_number} ---")
        
        if board.turn == chess.WHITE:
            engine_white.board = board.copy()
            move = engine_white.get_best_move(time_left=30, increment=0)
            print(f"White: {board.san(move)}")
        else:
            engine_black.board = board.copy()
            move = engine_black.get_best_move(time_left=30, increment=0)
            print(f"Black: {board.san(move)}")
        
        board.push(move)
        
        # Show evaluation after each move
        engine_white.board = board.copy()
        eval_score = engine_white._evaluate_position(board)
        print(f"Eval: {eval_score:+.2f} cp")
    
    print("\n" + "="*70)
    print("Final Position:")
    print("="*70)
    print(board)
    print(f"\nFEN: {board.fen()}")


if __name__ == "__main__":
    print("="*70)
    print("V7P3R v15.5 COMPREHENSIVE PLAYING STYLE ANALYSIS")
    print("="*70)
    print("\nThis test suite analyzes V15.5's behavior across:")
    print("- Different time controls (bullet to rapid)")
    print("- Tactical positions (mates, captures, forks)")
    print("- Material imbalances (up/down pieces)")
    print("- Endgame technique")
    print("- Time pressure scenarios")
    print("- Opening book transitions")
    
    try:
        test_opening_positions()
        test_tactical_positions()
        test_material_imbalance()
        test_endgame_positions()
        test_time_pressure()
        test_book_transitions()
        play_sample_game()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print("\nReview the results above to see:")
        print("- Move selection patterns")
        print("- Time management behavior")
        print("- Tactical awareness")
        print("- Material evaluation")
        print("- Opening book usage")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
