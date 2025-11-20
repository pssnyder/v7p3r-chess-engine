#!/usr/bin/env python3
"""
V16.1 Game Phase Testing
Tests initialization, opening book, middlegame nudges, and tablebase integration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_initialization():
    """Test V16.1 engine initialization"""
    print_section("TEST 1: Engine Initialization")
    
    try:
        # Test basic initialization
        print("\n[1a] Basic initialization (no tablebases)...")
        engine = V7P3REngine(max_depth=6, tt_size_mb=128, tablebase_path="")
        print("✓ Engine initialized successfully")
        print(f"  - Max depth: {engine.max_depth}")
        print(f"  - TT size: {engine.tt_size}")
        print(f"  - Opening book depth: {engine.opening_book.book_depth}")
        print(f"  - Tablebase available: {hasattr(engine, 'tablebase') and engine.tablebase is not None}")
        
        # Check opening book size
        book_size = len(engine.opening_book.book_moves)
        print(f"  - Opening positions loaded: {book_size}")
        
        # Test opening book
        print("\n[1b] Testing opening book...")
        board = chess.Board()
        book_move_uci = engine.opening_book.get_book_move(board)
        if book_move_uci:
            move = chess.Move.from_uci(book_move_uci)
            print(f"✓ Opening book working - suggests: {board.san(move)} ({book_move_uci})")
        else:
            print("⚠ No opening book move found")
        
        # Test position evaluation
        print("\n[1c] Testing evaluation function...")
        score = engine._evaluate_position(board)
        print(f"✓ Starting position evaluation: {score}cp")
        
        return engine
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_opening_phase(engine):
    """Test opening book usage and center control"""
    print_section("TEST 2: Opening Phase (Moves 1-10)")
    
    if not engine:
        print("Skipping - engine not initialized")
        return
    
    test_positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("After 1.e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("After 1.e4 e5", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"),
        ("After 1.e4 e5 2.Nf3", "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"),
        ("After 1.d4 d5", "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2"),
        ("Sicilian: 1.e4 c5", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
    ]
    
    for name, fen in test_positions:
        print(f"\n[2.{test_positions.index((name, fen)) + 1}] {name}")
        print(f"    FEN: {fen}")
        board = chess.Board(fen)
        engine.board = board  # Set engine's board position
        
        # Check opening book
        book_move_uci = engine.opening_book.get_book_move(board)
        if book_move_uci:
            move = chess.Move.from_uci(book_move_uci)
            print(f"    📖 Book move: {board.san(move)} ({book_move_uci})")
        else:
            print(f"    📖 No book move")
        
        # Get best move from engine
        best_move = engine.get_best_move(time_left=1.0)
        if best_move:
            print(f"    🎯 Engine plays: {board.san(best_move)} ({best_move.uci()})")
        
        # Show evaluation
        score = engine._evaluate_position(board)
        print(f"    ⚖️  Evaluation: {score:+d}cp")

def test_middlegame_phase(engine):
    """Test middlegame bonuses (rooks, king safety, pawn structure)"""
    print_section("TEST 3: Middlegame Phase (Bonuses)")
    
    if not engine:
        print("Skipping - engine not initialized")
        return
    
    test_positions = [
        ("Rook on open file", 
         "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQR1K1 w kq - 0 5",
         "White rook on e1 (semi-open e-file)"),
        
        ("Good king safety", 
         "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 8",
         "Both kings castled with pawn shields"),
        
        ("Passed pawn advantage",
         "4k3/8/8/3P4/8/8/8/4K3 w - - 0 1",
         "White has passed d-pawn"),
        
        ("Doubled pawns penalty",
         "rnbqkbnr/p1pppppp/8/8/1p6/P7/1PPPPPPP/RNBQKBNR w KQkq - 0 3",
         "White has doubled a-pawns"),
        
        ("Complex middlegame",
         "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 10",
         "Both sides developed, rooks on semi-open files"),
    ]
    
    for name, fen, description in test_positions:
        print(f"\n[3.{test_positions.index((name, fen, description)) + 1}] {name}")
        print(f"    {description}")
        print(f"    FEN: {fen}")
        board = chess.Board(fen)
        engine.board = board  # Set engine's board position
        
        # Calculate middlegame bonuses
        try:
            bonus = engine._calculate_middlegame_bonuses(board)
            print(f"    🎁 Middlegame bonus: {bonus:+d}cp")
        except Exception as e:
            print(f"    ⚠ Error calculating bonus: {e}")
        
        # Full evaluation
        score = engine._evaluate_position(board)
        print(f"    ⚖️  Total evaluation: {score:+d}cp")
        
        # Get best move
        best_move = engine.get_best_move(time_left=1.0)
        if best_move:
            print(f"    🎯 Best move: {board.san(best_move)} ({best_move.uci()})")

def test_endgame_phase(engine):
    """Test endgame detection and tablebase probing"""
    print_section("TEST 4: Endgame Phase (6-piece and fewer)")
    
    if not engine:
        print("Skipping - engine not initialized")
        return
    
    test_positions = [
        ("KRvK (winning)",
         "8/8/8/8/8/4k3/8/4KR2 w - - 0 1",
         "Basic rook endgame - should be winning for White"),
        
        ("KQvK (winning)",
         "8/8/8/4k3/8/8/8/4KQ2 w - - 0 1",
         "Queen endgame - should be winning for White"),
        
        ("KPvK (winning if advanced)",
         "8/8/8/3k4/3P4/8/8/4K3 w - - 0 1",
         "King and pawn vs king"),
        
        ("KBNvK (difficult win)",
         "8/8/8/8/4k3/8/8/4KBN1 w - - 0 1",
         "Bishop and knight vs king - complex win"),
        
        ("KRPvKR (technical)",
         "8/5k2/8/3R4/3P4/8/5K2/5r2 w - - 0 1",
         "Rook and pawn vs rook"),
    ]
    
    for name, fen, description in test_positions:
        print(f"\n[4.{test_positions.index((name, fen, description)) + 1}] {name}")
        print(f"    {description}")
        print(f"    FEN: {fen}")
        board = chess.Board(fen)
        engine.board = board  # Set engine's board position
        
        # Count pieces
        piece_count = len(board.piece_map())
        print(f"    🎲 Piece count: {piece_count}")
        
        # Check if tablebase available
        has_tablebase = hasattr(engine, 'tablebase') and engine.tablebase is not None
        print(f"    📊 Tablebase loaded: {has_tablebase}")
        
        # Evaluation
        score = engine._evaluate_position(board)
        print(f"    ⚖️  Evaluation: {score:+d}cp")
        
        # Get best move
        print(f"    🎯 Searching for best move...")
        best_move = engine.get_best_move(time_left=2.0)
        if best_move:
            print(f"       Move: {board.san(best_move)} ({best_move.uci()})")
        else:
            print(f"       No move found")

def test_tactical_positions(engine):
    """Test engine's ability to find tactics"""
    print_section("TEST 5: Tactical Positions")
    
    if not engine:
        print("Skipping - engine not initialized")
        return
    
    test_positions = [
        ("Fork opportunity",
         "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
         "Should avoid Nxe5 (Qe7 attacks knight)"),
        
        ("Capture winning material",
         "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3",
         "Black should capture: Nxe4"),
        
        ("Defend against mate threat",
         "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
         "Black must defend against Qxf7#"),
        
        ("Simple checkmate in 1",
         "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
         "Ra8# is checkmate"),
    ]
    
    for name, fen, description in test_positions:
        print(f"\n[5.{test_positions.index((name, fen, description)) + 1}] {name}")
        print(f"    {description}")
        print(f"    FEN: {fen}")
        board = chess.Board(fen)
        engine.board = board  # Set engine's board position
        
        # Evaluation
        score = engine._evaluate_position(board)
        print(f"    ⚖️  Evaluation: {score:+d}cp")
        
        # Get best move with deeper search
        print(f"    🎯 Searching (depth {engine.max_depth})...")
        best_move = engine.get_best_move(time_left=3.0)
        if best_move:
            print(f"       Move: {board.san(best_move)} ({best_move.uci()})")
            
            # Check if it's checkmate
            board_copy = board.copy()
            board_copy.push(best_move)
            if board_copy.is_checkmate():
                print(f"       ✓ CHECKMATE!")
        else:
            print(f"       No move found")

def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*70)
    print("  V7P3R v16.1 - COMPREHENSIVE GAME PHASE TESTING")
    print("="*70)
    print("\nTesting initialization, opening book, middlegame bonuses,")
    print("endgame handling, and tactical awareness across all game phases.")
    print("\nTarget: Beat C0BR4 v3.2 with deep opening knowledge,")
    print("smooth middlegame transition, and perfect endgames.")
    
    # Run tests
    engine = test_initialization()
    
    if engine:
        test_opening_phase(engine)
        test_middlegame_phase(engine)
        test_endgame_phase(engine)
        test_tactical_positions(engine)
    
    # Summary
    print_section("TEST SUMMARY")
    print("\n✓ All tests completed!")
    print("\nNext steps:")
    print("  1. Review opening book moves for center control")
    print("  2. Verify middlegame bonuses are applied correctly")
    print("  3. Download Syzygy tablebases for perfect endgames")
    print("  4. Run tournament match vs C0BR4 v3.2")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    run_all_tests()
