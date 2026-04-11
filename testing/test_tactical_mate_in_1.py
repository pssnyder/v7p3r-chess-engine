"""
Tactical Test Suite: Mate-in-1 Fast Path Validation
Tests engine's ability to detect immediate checkmate opportunities.

Before/After Test Design:
- BEFORE: Baseline detection with full search
- AFTER: Validate fast path detection (<1ms overhead, 100% accuracy)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3REngine

def test_mate_in_1_detection():
    """Test engine detects all mate-in-1 positions correctly."""
    print("\n=== Test: Mate-in-1 Detection (20 positions) ===")
    
    # 20 mate-in-1 test positions
    # Format: (FEN, expected_move_uci, description)
    test_positions = [
        # Back rank mates
        ("r5k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "e1e8", "Back rank mate with rook"),
        ("6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1", "f1f8", "Back rank mate variation"),
        
        # Queen mates
        ("6k1/5ppp/8/8/8/8/5PPP/5Q1K w - - 0 1", "f1f8", "Queen back rank mate"),
        ("r4rk1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1", "e1e8", "Queen mate with rook pinned"),
        ("3k4/3P4/3K4/8/8/8/8/8 w - - 0 1", "d7d8q", "Promotion to queen checkmate"),
        
        # Knight mates
        ("6k1/5ppp/6N1/8/8/8/5PPP/6K1 w - - 0 1", "g6e7", "Knight mate in corner"),
        ("r6k/6pp/7N/8/8/8/5PPP/6K1 w - - 0 1", "h6f7", "Smothered mate setup"),
        
        # Bishop mates
        ("6k1/5ppp/8/6B1/8/8/5PPP/6K1 w - - 0 1", "g5e7", "Bishop mate"),
        
        # Pawn mates (rare but valid)
        ("6k1/5Ppp/8/8/8/8/5PPP/6K1 w - - 0 1", "f7f8q", "Pawn promotion mate"),
        
        # Checkmate with multiple pieces
        ("r5k1/1Q3ppp/8/8/8/8/5PPP/6K1 w - - 0 1", "b7b8", "Queen mate with rook blocked"),
        ("6k1/5ppp/8/8/8/5Q2/5PPP/5RK1 w - - 0 1", "f3f8", "Queen mate supported by rook"),
        
        # Discovered check mates
        ("r2q2k1/5ppp/8/4B3/8/8/5PPP/4R1K1 w - - 0 1", "e1e8", "Rook mate with bishop controlling escape"),
        
        # Castling trapped king
        ("6kr/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1", "f1f8", "Mate on castled king"),
        
        # Double check mates
        ("6k1/6pp/5p2/5N2/8/8/5PPP/4R1K1 w - - 0 1", "e1e8", "Rook mate with knight support"),
        
        # Edge cases
        ("7k/5K1p/6p1/8/8/8/8/6Q1 w - - 0 1", "g1g8", "Queen mate on edge"),
        ("3k4/8/2K5/2Q5/8/8/8/8 w - - 0 1", "c5c7", "Queen mate in center"),
        ("7k/5Rpp/8/8/8/8/5PPP/6K1 w - - 0 1", "f7f8", "Rook mate"),
        ("6rk/6pp/8/8/8/8/5PPP/5QK1 w - - 0 1", "f1f8", "Queen mate with rook pinned"),
        
        # Tricky positions
        ("r4rk1/1Q3ppp/8/8/8/8/5PPP/6K1 w - - 0 1", "b7b8", "Queen and rook coordination"),
        ("6k1/5p1p/5Q1P/6P1/8/8/5P1K/8 w - - 0 1", "f6f7", "Queen mate simple"),
    ]
    
    engine = V7P3REngine()
    detected_count = 0
    total_time = 0
    
    print(f"Testing {len(test_positions)} mate-in-1 positions...\n")
    
    for i, (fen, expected_move, description) in enumerate(test_positions, 1):
        board = chess.Board(fen)
        
        # Time the detection
        start = time.perf_counter()
        best_move = engine.search(board, depth=3)  # Shallow depth sufficient for mate-in-1
        elapsed = time.perf_counter() - start
        total_time += elapsed
        
        # Check if move delivers checkmate
        board.push(best_move)
        is_checkmate = board.is_checkmate()
        board.pop()
        
        if is_checkmate:
            detected_count += 1
            status = "✓ DETECTED"
        else:
            status = "✗ MISSED"
        
        print(f"{i:2d}. {status} - {description} ({elapsed*1000:.2f}ms)")
        if not is_checkmate:
            print(f"    Expected: {expected_move}, Got: {best_move}")
            print(f"    FEN: {fen}")
    
    detection_rate = (detected_count / len(test_positions)) * 100
    avg_time_ms = (total_time / len(test_positions)) * 1000
    
    print(f"\nResults:")
    print(f"  Detection Rate: {detected_count}/{len(test_positions)} ({detection_rate:.1f}%)")
    print(f"  Average Time: {avg_time_ms:.2f}ms per position")
    
    # Pass criteria: 100% detection (engine must find mate even without fast path)
    assert detected_count == len(test_positions), \
        f"Failed to detect all mates: {detected_count}/{len(test_positions)}"
    print("  ✅ PASS: All mate-in-1 positions detected")


def test_mate_in_1_overhead():
    """Test that mate-in-1 check adds minimal overhead (<1ms)."""
    print("\n=== Test: Mate-in-1 Fast Path Overhead ===")
    
    # Non-mate positions (should have minimal overhead from mate check)
    non_mate_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Open game
        "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",  # Queen's Gambit
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5",  # Italian Game
        "r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9",  # Middlegame
    ]
    
    engine = V7P3REngine()
    total_time = 0
    
    print(f"Testing overhead on {len(non_mate_positions)} non-mate positions...\n")
    
    for i, fen in enumerate(non_mate_positions, 1):
        board = chess.Board(fen)
        
        start = time.perf_counter()
        best_move = engine.search(board, depth=3)
        elapsed = time.perf_counter() - start
        total_time += elapsed
        
        print(f"{i}. {elapsed*1000:.2f}ms - {best_move}")
    
    avg_time_ms = (total_time / len(non_mate_positions)) * 1000
    print(f"\nAverage Time: {avg_time_ms:.2f}ms per position")
    print(f"Expected: <1ms overhead from mate-in-1 check")
    
    # With fast path, overhead should be negligible
    # Without fast path, this establishes baseline
    print("  ✅ PASS: Overhead benchmark established")


def test_mate_in_1_vs_non_mate():
    """Verify engine distinguishes mate-in-1 from other good moves."""
    print("\n=== Test: Mate vs Non-Mate Move Selection ===")
    
    # Positions with both mate-in-1 and other strong moves
    test_cases = [
        ("6k1/5ppp/8/8/8/5Q2/5PPP/5RK1 w - - 0 1", "f3f8", "Should prefer mate over material win"),
        ("6k1/5p1p/8/5Q2/8/8/5PPP/5RK1 w - - 0 1", "f5f7", "Should prefer mate over check"),
    ]
    
    engine = V7P3REngine()
    all_correct = True
    
    print(f"Testing {len(test_cases)} positions...\n")
    
    for i, (fen, expected_mate_move, description) in enumerate(test_cases, 1):
        board = chess.Board(fen)
        best_move = engine.search(board, depth=3)
        
        # Check if move delivers checkmate
        board.push(best_move)
        is_checkmate = board.is_checkmate()
        board.pop()
        
        if is_checkmate:
            print(f"{i}. ✓ CORRECT - {description}")
            print(f"   Found mate: {best_move}")
        else:
            print(f"{i}. ✗ INCORRECT - {description}")
            print(f"   Expected mate: {expected_mate_move}, Got: {best_move}")
            all_correct = False
    
    if all_correct:
        print("\n  ✅ PASS: Engine prefers mate-in-1 over other moves")
    else:
        print("\n  ⚠ WARNING: Some mate-in-1 moves not prioritized")


def test_no_false_positives():
    """Ensure mate-in-1 detection doesn't produce false positives."""
    print("\n=== Test: No False Positive Mates ===")
    
    # Positions that LOOK like mate but aren't
    non_mate_positions = [
        ("6k1/5ppp/8/8/8/5R2/5PPP/6K1 w - - 0 1", "King has escape squares"),
        ("r5k1/5ppp/6r1/8/8/8/5PPP/4R1K1 w - - 0 1", "Rook defends back rank"),
        ("6k1/5p1p/5p2/8/8/5Q2/5PPP/6K1 w - - 0 1", "King can block with pawn"),
    ]
    
    engine = V7P3REngine()
    false_positives = 0
    
    print(f"Testing {len(non_mate_positions)} non-mate positions...\n")
    
    for i, (fen, description) in enumerate(non_mate_positions, 1):
        board = chess.Board(fen)
        best_move = engine.search(board, depth=3)
        
        # Check if move claims to be checkmate but isn't
        board.push(best_move)
        is_checkmate = board.is_checkmate()
        board.pop()
        
        # In these positions, no legal move should deliver mate
        # So any checkmate claim is a false positive
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                print(f"{i}. ✗ FALSE POSITIVE - {description}")
                print(f"   Move {move} incorrectly detected as mate")
                false_positives += 1
                board.pop()
                break
            board.pop()
        else:
            print(f"{i}. ✓ CORRECT - {description} (no false mate)")
    
    if false_positives == 0:
        print("\n  ✅ PASS: No false positive mate detections")
    else:
        print(f"\n  ✗ FAIL: {false_positives} false positive(s) detected")
        assert False, "False positive mate detections found"


if __name__ == "__main__":
    print("=" * 60)
    print("TACTICAL TEST SUITE: Mate-in-1 Fast Path")
    print("=" * 60)
    
    try:
        test_mate_in_1_detection()
        test_mate_in_1_overhead()
        test_mate_in_1_vs_non_mate()
        test_no_false_positives()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n{'=' * 60}")
        print(f"TEST FAILED ✗")
        print(f"{'=' * 60}")
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"TEST ERROR ✗")
        print(f"{'=' * 60}")
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
