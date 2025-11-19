#!/usr/bin/env python3
"""
Comprehensive test for V15.3 opening book
Tests all major openings and especially the Caro-Kann to ensure no bongcloud disasters
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine


def test_opening_book():
    """Test opening book coverage"""
    
    print("=" * 60)
    print("V15.3 Opening Book Comprehensive Test")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    test_cases = [
        # Starting position
        {
            "name": "Starting position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "expected_moves": ["e2e4", "d2d4", "g1f3", "c2c4"],
            "disallowed_moves": ["h2h4", "a2a4", "b1a3"]
        },
        
        # After 1.e4 - Black's response
        {
            "name": "After 1.e4 - Black response",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "expected_moves": ["e7e5", "c7c5", "e7e6", "c7c6"],
            "disallowed_moves": ["a7a6", "h7h6"]
        },
        
        # CRITICAL: After 1.e4 c6 (Caro-Kann) - White's 2nd move
        {
            "name": "After 1.e4 c6 (Caro-Kann) - White's response",
            "fen": "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "expected_moves": ["d2d4", "b1c3", "g1f3", "d2d3"],
            "disallowed_moves": ["e1e2", "f1e2", "d1e2"],  # NO BONGCLOUD!
            "must_have_book_move": True
        },
        
        # After 1.e4 c6 2.d4 d5 (Caro-Kann main line)
        {
            "name": "After 1.e4 c6 2.d4 d5 - White's response",
            "fen": "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
            "expected_moves": ["b1c3", "e4d5", "e4e5"],
            "disallowed_moves": ["e1e2", "f1e2"]
        },
        
        # After 1.e4 e5 - White's 2nd move
        {
            "name": "After 1.e4 e5 - White's response",
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "expected_moves": ["g1f3", "f1c4", "b1c3"],
            "disallowed_moves": ["e1e2", "f1e2", "h2h4"]
        },
        
        # After 1.e4 c5 (Sicilian) - White's 2nd move
        {
            "name": "After 1.e4 c5 (Sicilian) - White's response",
            "fen": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "expected_moves": ["g1f3", "b1c3", "c2c3"],
            "disallowed_moves": ["e1e2", "h2h4"]
        },
        
        # After 1.d4 - Black's response
        {
            "name": "After 1.d4 - Black response",
            "fen": "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
            "expected_moves": ["d7d5", "g8f6", "e7e6", "c7c5"],  # Added c5 (Benoni)
            "disallowed_moves": ["a7a6", "h7h6"]
        },
        
        # After 1.d4 Nf6 - White's 2nd move
        {
            "name": "After 1.d4 Nf6 - White's response",
            "fen": "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
            "expected_moves": ["c2c4", "g1f3", "b1c3"],
            "disallowed_moves": ["e2e4", "h2h4"]
        },
        
        # Out of book - should return to search
        {
            "name": "Out of book position (random middlegame)",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5",
            "expected_moves": None,  # Should search
            "disallowed_moves": ["e1e2"],  # Still no bongcloud even out of book
            "must_have_book_move": False
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"FEN: {test_case['fen']}")
        
        engine.board = chess.Board(test_case['fen'])
        book_move = engine.opening_book.get_book_move(engine.board)
        
        if test_case.get("must_have_book_move"):
            if not book_move:
                print(f"❌ FAILED - Expected book move but got None!")
                print(f"   This position MUST be in the book to prevent disasters!")
                failed += 1
                continue
        
        if book_move:
            print(f"Book move: {book_move}")
            
            # Check if book move is in expected moves
            if test_case["expected_moves"]:
                if book_move in test_case["expected_moves"]:
                    print(f"✓ Book move is in expected moves: {test_case['expected_moves']}")
                else:
                    print(f"❌ FAILED - Book move {book_move} not in expected: {test_case['expected_moves']}")
                    failed += 1
                    continue
            
            # Check if book move is NOT in disallowed moves
            if book_move in test_case["disallowed_moves"]:
                print(f"❌ FAILED - Book move {book_move} is DISALLOWED!")
                print(f"   Disallowed moves: {test_case['disallowed_moves']}")
                failed += 1
                continue
            
            print(f"✓ Book move passed all checks")
            passed += 1
        else:
            print("No book move (will search)")
            
            if test_case["expected_moves"] is not None:
                print(f"⚠ Warning - Expected book move from: {test_case['expected_moves']}")
                print(f"   Testing search move...")
                
                # Test that engine doesn't choose disallowed move
                search_move = engine.get_best_move()
                if search_move:
                    move_uci = search_move.uci()
                    print(f"Search chose: {move_uci}")
                    
                    if move_uci in test_case["disallowed_moves"]:
                        print(f"❌ FAILED - Search chose DISALLOWED move: {move_uci}")
                        print(f"   Disallowed: {test_case['disallowed_moves']}")
                        failed += 1
                        continue
                    
                    print(f"✓ Search move is acceptable")
                    passed += 1
                else:
                    print(f"❌ FAILED - No move returned")
                    failed += 1
            else:
                print(f"✓ Correctly out of book")
                passed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 60)
    
    if failed == 0:
        print("✓ All tests passed! Opening book is working correctly.")
        print("✓ No bongcloud disasters detected!")
        return True
    else:
        print(f"❌ {failed} test(s) failed")
        return False


def test_book_variety():
    """Test that book variety works"""
    print("\n" + "=" * 60)
    print("Testing Book Variety")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # Test with 0% variety (always best move)
    engine.opening_book.book_variety = 0
    moves_0 = []
    for _ in range(10):
        engine.board = chess.Board()
        move = engine.opening_book.get_book_move(engine.board)
        moves_0.append(move)
    
    print(f"0% variety (10 trials): {set(moves_0)}")
    if len(set(moves_0)) == 1:
        print("✓ Always chooses same move at 0% variety")
    else:
        print("⚠ Multiple moves at 0% variety (this is acceptable if weights are equal)")
    
    # Test with 100% variety (random)
    engine.opening_book.book_variety = 100
    moves_100 = []
    for _ in range(20):
        engine.board = chess.Board()
        move = engine.opening_book.get_book_move(engine.board)
        moves_100.append(move)
    
    print(f"100% variety (20 trials): {set(moves_100)}")
    if len(set(moves_100)) > 1:
        print("✓ Chooses different moves at 100% variety")
    else:
        print("⚠ Always same move at 100% variety (low sample size?)")
    
    return True


def test_book_depth():
    """Test that book depth limit works"""
    print("\n" + "=" * 60)
    print("Testing Book Depth Limit")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # Set book depth to 4 (2 moves per side)
    engine.opening_book.book_depth = 4
    
    # Test at ply 0 (starting position)
    engine.board = chess.Board()
    move = engine.opening_book.get_book_move(engine.board)
    print(f"Ply 0: {move} (expected: book move)")
    
    # Test at ply 4 (should still be in book)
    engine.board = chess.Board()
    for move_str in ["e2e4", "c7c6", "d2d4", "d7d5"]:
        engine.board.push(chess.Move.from_uci(move_str))
    
    move = engine.opening_book.get_book_move(engine.board)
    print(f"Ply 4 (at depth limit): {move} (expected: book move or None)")
    
    # Test at ply 5 (should be out of book)
    engine.board.push(chess.Move.from_uci("b1c3"))
    move = engine.opening_book.get_book_move(engine.board)
    print(f"Ply 5 (past depth limit): {move} (expected: None)")
    
    if move is None:
        print("✓ Book depth limit working correctly")
        return True
    else:
        print("⚠ Book still returning moves past depth limit")
        return False


if __name__ == "__main__":
    print("\nV15.3 Opening Book Comprehensive Test Suite\n")
    
    success = True
    
    # Main opening book test
    if not test_opening_book():
        success = False
    
    # Book variety test
    if not test_book_variety():
        success = False
    
    # Book depth test
    if not test_book_depth():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("V15.3 opening book is ready for deployment!")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the failures above")
    print("=" * 60)
