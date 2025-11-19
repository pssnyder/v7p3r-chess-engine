#!/usr/bin/env python3
"""
Debug why Ke2 is chosen after 1.e4 c6
"""

import sys
import os
sys.path.insert(0, "s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/src")

import chess
from v7p3r import V7P3REngine

def analyze_position_after_e4_c6():
    """Analyze the position after 1.e4 c6"""
    engine = V7P3REngine()
    
    print("="*60)
    print("DEBUGGING: Why does engine play Ke2 after 1.e4 c6?")
    print("="*60)
    print()
    
    # Set up position after 1.e4 c6
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("c7c6"))
    
    print("Position after 1.e4 c6:")
    print(board)
    print()
    
    # Check if book has this position
    book_move = engine.opening_book.get_book_move(board)
    print(f"Book move available: {book_move}")
    print()
    
    if book_move is None:
        print("NOT IN BOOK - Engine will search")
        print()
        
        # Evaluate different candidate moves
        candidates = [
            ("d2d4", "2.d4 (Strong center)"),
            ("g1f3", "2.Nf3 (Development)"),
            ("b1c3", "2.Nc3 (Development)"),
            ("e1e2", "2.Ke2 (BONGCLOUD!)"),
            ("h2h4", "2.h4 (Edge pawn)"),
        ]
        
        eval_before = engine._evaluate_position(board)
        print(f"Position evaluation: {eval_before:+d}")
        print()
        print("Evaluating candidate moves:")
        print("-" * 60)
        
        results = []
        for move_uci, desc in candidates:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                    eval_after = engine._evaluate_position(board)
                    board.pop()
                    
                    change = eval_after - eval_before
                    results.append((desc, move_uci, change, eval_after))
                    print(f"{desc:30s} Eval: {eval_after:+5d}  Change: {change:+4d}")
            except:
                pass
        
        print()
        print("="*60)
        print("BEST MOVES BY PST EVALUATION:")
        print("="*60)
        results.sort(key=lambda x: x[2], reverse=True)
        for desc, move, change, eval_val in results[:3]:
            print(f"{desc:30s} {move:6s} ({change:+4d})")
        
        print()
        print("="*60)
        print("ACTUAL ENGINE SEARCH:")
        print("="*60)
        
        # Disable book temporarily
        engine.opening_book.use_book = False
        engine.board = board.copy()
        
        # Search with time limit
        best_move = engine.get_best_move(time_left=5.0, increment=0.1)
        
        print()
        print(f"Engine chose: {best_move}")
        print()
        
        if best_move and best_move.uci() == "e1e2":
            print("❌ CONFIRMED BUG: Engine chose Ke2 (bongcloud)")
            print()
            print("Root cause: Search horizon issues - engine sees multi-move")
            print("plans that don't account for tactical refutations.")
        elif best_move and best_move.uci() in ["h2h4", "h2h3", "a2a4"]:
            print("❌ CONFIRMED BUG: Engine chose edge pawn move")
            print()
            print("Root cause: Same horizon issue as bongcloud")
        else:
            print(f"✓ Engine chose reasonable move: {best_move}")

if __name__ == "__main__":
    analyze_position_after_e4_c6()
