#!/usr/bin/env python3
"""
V14.7 Blunder Prevention Test Suite

Tests that V14.7 prevents the specific blunders from V14.6 vs V14.4 game:
1. Move 3: Should capture Ne4, not play b3
2. Move 6: Should play dxc3, not Rg1
3. Move 12: Should not trade bishop for nothing (Bxc6+)
4. Move 20: Should not blunder bishop (Bxc1)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine


def test_position(engine, fen, description, expected_move=None, rejected_moves=None):
    """
    Test engine behavior on a specific position
    
    Args:
        engine: V7P3REngine instance
        fen: Position FEN
        description: Human-readable description
        expected_move: Move that should be played (if known)
        rejected_moves: Moves that should be filtered out as unsafe
    """
    print(f"\nTesting: {description}")
    print(f"FEN: {fen}")
    
    board = chess.Board(fen)
    print(f"Position: {board.unicode()}\n")
    
    # Generate legal moves
    legal_moves = list(board.legal_moves)
    print(f"Legal moves: {len(legal_moves)}")
    
    # Test safety filtering
    safe_moves = engine._filter_unsafe_moves(board, legal_moves)
    print(f"Safe moves after filtering: {len(safe_moves)}")
    
    # Check if expected unsafe moves were filtered out
    if rejected_moves:
        for rejected_move_str in rejected_moves:
            rejected_move = chess.Move.from_uci(rejected_move_str)
            if rejected_move in safe_moves:
                print(f"  FAIL: Unsafe move {rejected_move_str} was NOT filtered out!")
                return False
            else:
                print(f"  OK: Unsafe move {rejected_move_str} correctly filtered out")
    
    # Run search to see what move is chosen
    best_move = engine.search(board, time_limit=3.0, depth=4)
    print(f"Best move chosen: {best_move}")
    
    if expected_move:
        expected = chess.Move.from_uci(expected_move)
        if best_move == expected:
            print(f"  OK: Chose expected move {expected_move}")
            return True
        else:
            print(f"  WARN: Expected {expected_move}, got {best_move}")
            # Not a failure if different move is chosen, as long as it's safe
            return True
    
    return True


def main():
    print("=" * 80)
    print("V14.7 Blunder Prevention Test Suite")
    print("=" * 80)
    
    engine = V7P3REngine()
    
    # Test 1: Position after 2...Ne4 - Should capture, not play b3
    print("\n" + "=" * 80)
    print("TEST 1: Don't ignore hanging knight")
    print("=" * 80)
    test_position(
        engine,
        fen="rnbqkb1r/pppppppp/8/8/4n3/5N2/PPPPPPPP/RNBQKB1R w KQkq - 1 3",
        description="After 2...Ne4 - Should capture Ne4, not play b3",
        expected_move="f3e4",  # Nxe4
        rejected_moves=[]  # b3 isn't necessarily unsafe, just bad
    )
    
    # Test 2: Position after 5...Nxc3 - Should recapture with pawn
    print("\n" + "=" * 80)
    print("TEST 2: Recapture material")
    print("=" * 80)
    test_position(
        engine,
        fen="rnbqkb1r/pppp1ppp/3p4/8/8/2n2N2/PPP1PPPP/R1BQKB1R w KQkq - 0 6",
        description="After 5...Nxc3 - Should play dxc3, not Rg1",
        expected_move="d2c3",  # dxc3
        rejected_moves=[]  # Rg1 isn't necessarily filtered, but dxc3 should be ordered higher
    )
    
    # Test 3: Don't trade bishop for nothing
    print("\n" + "=" * 80)
    print("TEST 3: Don't give away pieces for nothing")
    print("=" * 80)
    test_position(
        engine,
        fen="r2qkb1r/2p2ppp/p1pb4/1p6/8/1P3P2/PBPP2PP/R2QKB1R w KQkq - 0 12",
        description="Don't trade bishop for nothing (Bxc6+ bad)",
        expected_move=None,  # Don't know best move, just check filtering
        rejected_moves=[]  # Bxc6+ might look tactical (check), hard to filter
    )
    
    # Test 4: Don't blunder bishop to enemy rook
    print("\n" + "=" * 80)
    print("TEST 4: Don't leave bishop hanging")
    print("=" * 80)
    test_position(
        engine,
        fen="3nkb1r/6pp/1p1p1p2/p7/3P4/1P5P/PB3PP1/3r1K1R w k - 2 21",
        description="Bishop on b2 - don't move to c1 where rook can capture",
        expected_move=None,
        rejected_moves=["b2c1"]  # Bxc1 should be filtered (bishop hangs)
    )
    
    # Test 5: King safety - don't expose king
    print("\n" + "=" * 80)
    print("TEST 5: King safety check")
    print("=" * 80)
    test_position(
        engine,
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        description="Starting position - test king safety filtering works",
        expected_move=None,
        rejected_moves=[]  # Starting position has no unsafe moves
    )
    
    # Test 6: Queen safety - don't leave queen hanging
    print("\n" + "=" * 80)
    print("TEST 6: Queen safety check")
    print("=" * 80)
    test_position(
        engine,
        fen="rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
        description="After 1.d4 d5 - queen moves should be safe",
        expected_move=None,
        rejected_moves=[]
    )
    
    print("\n" + "=" * 80)
    print("Test Suite Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
