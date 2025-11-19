#!/usr/bin/env python3
"""
Anti-Bongcloud Test - Ensures V15.3 NEVER plays Ke2 in the opening
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine


def test_no_bongcloud():
    """Test various opening positions to ensure NO Ke2"""
    
    print("=" * 60)
    print("V15.3 Anti-Bongcloud Test")
    print("=" * 60)
    print("\nThis test ensures the engine NEVER plays the bongcloud (Ke2)")
    print("in any common opening position.\n")
    
    engine = V7P3REngine()
    
    # Test positions where Ke2 might be considered
    test_positions = [
        {
            "name": "Starting position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "moves_leading": []
        },
        {
            "name": "After 1.e4 (should not play Ke2)",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            "moves_leading": ["e2e4"]
        },
        {
            "name": "After 1.e4 e5 (King's pawn game)",
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "moves_leading": ["e2e4", "e7e5"]
        },
        {
            "name": "After 1.e4 c6 (THE CRITICAL TEST - Caro-Kann)",
            "fen": "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "moves_leading": ["e2e4", "c7c6"]
        },
        {
            "name": "After 1.e4 c5 (Sicilian)",
            "fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "moves_leading": ["e2e4", "c7c5"]
        },
        {
            "name": "After 1.e4 e6 (French)",
            "fen": "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "moves_leading": ["e2e4", "e7e6"]
        },
        {
            "name": "After 1.e4 d5 (Scandinavian)",
            "fen": "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "moves_leading": ["e2e4", "d7d5"]
        },
        {
            "name": "After 1.d4 (should not play Ke2)",
            "fen": "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
            "moves_leading": ["d2d4"]
        },
    ]
    
    passed = 0
    failed = 0
    bongcloud_detected = False
    
    for test in test_positions:
        print(f"\nTest: {test['name']}")
        print(f"Moves: {' '.join(test['moves_leading']) if test['moves_leading'] else 'None (starting position)'}")
        
        # Set up position
        engine.board = chess.Board(test['fen'])
        
        # First check book
        book_move = engine.opening_book.get_book_move(engine.board)
        if book_move:
            print(f"  Book move: {book_move}")
            
            if book_move == "e1e2":
                print(f"  ❌ BONGCLOUD DETECTED IN BOOK!")
                failed += 1
                bongcloud_detected = True
                continue
            else:
                print(f"  ✓ Book move is safe")
                passed += 1
                continue
        
        # If not in book, test search
        print(f"  Not in book - testing search...")
        best_move = engine.get_best_move()
        
        if best_move:
            move_uci = best_move.uci()
            print(f"  Search move: {move_uci}")
            
            if move_uci == "e1e2":
                print(f"  ❌ BONGCLOUD DETECTED IN SEARCH!")
                failed += 1
                bongcloud_detected = True
            else:
                print(f"  ✓ Search move is safe")
                passed += 1
        else:
            print(f"  ⚠ No move returned")
            passed += 1  # Not a bongcloud at least
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if bongcloud_detected:
        print("\n❌❌❌ BONGCLOUD DETECTED! ❌❌❌")
        print("The engine played Ke2 in at least one test!")
        print("V15.3 is NOT ready for deployment.")
        return False
    else:
        print("\n✓✓✓ NO BONGCLOUD! ✓✓✓")
        print("The engine never played Ke2 in any test position.")
        print("V15.3 passed the anti-bongcloud test!")
        return True


def test_caro_kann_specifically():
    """Focused test on the Caro-Kann position that caused the original bongcloud"""
    
    print("\n" + "=" * 60)
    print("Caro-Kann Specific Test (Original Bongcloud Position)")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # The exact position where bongcloud happened
    print("\n1. e4 c6 - The position where the bongcloud disaster occurred")
    print("-" * 60)
    
    engine.board = chess.Board()
    engine.board.push(chess.Move.from_uci("e2e4"))
    engine.board.push(chess.Move.from_uci("c7c6"))
    
    print(f"Position FEN: {engine.board.fen()}")
    print(f"Position (ASCII):\n{engine.board}")
    
    # Check book first
    book_move = engine.opening_book.get_book_move(engine.board)
    if book_move:
        print(f"\n✓ Book has a move for this position: {book_move}")
        
        if book_move == "e1e2":
            print("❌ DISASTER! Book recommends bongcloud!")
            return False
        else:
            print(f"✓ Book move is sensible: {book_move}")
            
            # Verify it's a standard Caro-Kann response
            standard_moves = ["d2d4", "b1c3", "g1f3", "d2d3"]
            if book_move in standard_moves:
                print(f"✓ Book move is a standard Caro-Kann response")
            else:
                print(f"⚠ Book move is unusual but not terrible")
            
            return True
    else:
        print("\n⚠ No book move found - engine will search")
        print("Testing search to ensure no bongcloud...")
        
        best_move = engine.get_best_move()
        if best_move:
            move_uci = best_move.uci()
            print(f"Search chose: {move_uci}")
            
            if move_uci == "e1e2":
                print("❌ DISASTER! Search found the bongcloud!")
                return False
            else:
                print("✓ Search did not choose bongcloud")
                return True
        else:
            print("❌ No move returned")
            return False


if __name__ == "__main__":
    print("\nV15.3 Anti-Bongcloud Test Suite")
    print("Ensuring we never repeat the Ke2 disaster\n")
    
    success = True
    
    # Main anti-bongcloud test
    if not test_no_bongcloud():
        success = False
    
    # Specific Caro-Kann test
    if not test_caro_kann_specifically():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓✓✓ ANTI-BONGCLOUD TEST PASSED ✓✓✓")
        print("V15.3 will NOT play Ke2 in the opening!")
        print("Safe for deployment.")
    else:
        print("❌ ANTI-BONGCLOUD TEST FAILED")
        print("DO NOT DEPLOY - Fix required!")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
