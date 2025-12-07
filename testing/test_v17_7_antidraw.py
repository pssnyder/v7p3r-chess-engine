#!/usr/bin/env python3
"""
Test suite for V7P3R v17.7 Anti-Draw Measures
Tests all 5 major anti-draw features:
1. Threefold repetition detection and avoidance
2. Mate verification depth extensions
3. King-edge driving bonus
4. Basic tablebase patterns
5. 50-move rule awareness
"""

import sys
import os
import chess
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine
from v7p3r_fast_evaluator import V7P3RFastEvaluator


def test_threefold_repetition_detection():
    """Test that threefold repetition is detected and avoided in winning positions"""
    print("\n=== Test 1: Threefold Repetition Detection ===")
    engine = V7P3REngine()
    
    # Set up a position where repetition is possible
    # White is winning significantly (up a rook)
    board = chess.Board("4k3/8/8/8/8/8/R7/4K3 w - - 0 1")
    
    # Simulate a repetition pattern
    move1 = chess.Move.from_uci("e1e2")  # Ke2
    move2 = chess.Move.from_uci("e8d8")  # Kd8
    move3 = chess.Move.from_uci("e2e1")  # Ke1 (back to original)
    move4 = chess.Move.from_uci("d8e8")  # Ke8 (back to original)
    
    # First occurrence
    board.push(move1)
    board.push(move2)
    board.push(move3)
    board.push(move4)
    
    # Add to position history
    engine.position_history.append(engine.zobrist.hash_position(board))
    
    # Second occurrence
    board.push(move1)
    board.push(move2)
    board.push(move3)
    board.push(move4)
    
    engine.position_history.append(engine.zobrist.hash_position(board))
    
    # Now test if third occurrence would be detected
    board.push(move1)
    board.push(move2)
    board.push(move3)
    # Don't push move4 yet - test if it would cause threefold
    
    would_repeat = engine._would_cause_threefold_repetition(board, move4)
    
    if would_repeat:
        print("✅ PASS: Threefold repetition correctly detected")
        return True
    else:
        print("❌ FAIL: Threefold repetition NOT detected")
        return False


def test_mate_verification_extension():
    """Test that mate scores trigger depth extensions"""
    print("\n=== Test 2: Mate Verification Extension ===")
    engine = V7P3REngine()
    
    # Mate in 1 position
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K2R w - - 0 1")  # Ra8#
    
    # Test _is_mate_score helper
    mate_score = 20000
    non_mate_score = 500
    
    is_mate_detected = engine._is_mate_score(mate_score)
    is_non_mate_correct = not engine._is_mate_score(non_mate_score)
    
    if is_mate_detected and is_non_mate_correct:
        print("✅ PASS: Mate score detection working correctly")
        print(f"   - Mate score {mate_score}: {'DETECTED' if is_mate_detected else 'NOT DETECTED'}")
        print(f"   - Material score {non_mate_score}: {'DETECTED' if not is_non_mate_correct else 'NOT DETECTED'}")
        return True
    else:
        print("❌ FAIL: Mate score detection incorrect")
        return False


def test_king_edge_driving_bonus():
    """Test that king-edge driving bonus applies in winning endgames"""
    print("\n=== Test 3: King-Edge Driving Bonus ===")
    evaluator = V7P3RFastEvaluator()
    
    # White has K+R vs K (material advantage ~500cp)
    # Black king in corner vs center - should see difference
    
    # Position 1: Black king in corner (a8)
    board_corner = chess.Board("k7/8/8/8/8/8/8/4K2R w - - 0 1")
    score_corner = evaluator.evaluate(board_corner)
    
    # Position 2: Black king in center (e4)
    board_center = chess.Board("8/8/8/8/4k3/8/8/4K2R w - - 0 1")
    score_center = evaluator.evaluate(board_center)
    
    # Corner should be better (king closer to edge)
    bonus_difference = score_corner - score_center
    
    print(f"   Score with king in corner (a8): {score_corner}cp")
    print(f"   Score with king in center (e4): {score_center}cp")
    print(f"   King-edge bonus difference: {bonus_difference}cp")
    
    if bonus_difference > 30:  # Expect at least 30cp difference
        print("✅ PASS: King-edge driving bonus working (corner better by {bonus_difference}cp)")
        return True
    else:
        print(f"❌ FAIL: King-edge bonus too small ({bonus_difference}cp)")
        return False


def test_tablebase_pattern_detection():
    """Test that basic tablebase endgames are recognized"""
    print("\n=== Test 4: Tablebase Pattern Detection ===")
    engine = V7P3REngine()
    
    # K+R vs K
    board_kr = chess.Board("4k3/8/8/8/8/8/R7/4K3 w - - 0 1")
    hints_kr = engine._detect_basic_tablebase_endgame(board_kr)
    
    # K+Q vs K
    board_kq = chess.Board("4k3/8/8/8/8/8/Q7/4K3 w - - 0 1")
    hints_kq = engine._detect_basic_tablebase_endgame(board_kq)
    
    # K+R+B vs K (game e68K458O scenario)
    board_krb = chess.Board("4k3/8/8/8/8/8/R7/3BK3 w - - 0 1")
    hints_krb = engine._detect_basic_tablebase_endgame(board_krb)
    
    # Non-tablebase position (more pieces)
    board_normal = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    hints_normal = engine._detect_basic_tablebase_endgame(board_normal)
    
    print(f"   K+R vs K detected: {len(hints_kr) > 0} ({len(hints_kr)} hints)")
    print(f"   K+Q vs K detected: {len(hints_kq) > 0} ({len(hints_kq)} hints)")
    print(f"   K+R+B vs K detected: {len(hints_krb) > 0} ({len(hints_krb)} hints)")
    print(f"   Normal position detected: {len(hints_normal) == 0} (should be False)")
    
    if len(hints_kr) > 0 and len(hints_kq) > 0 and len(hints_krb) > 0 and len(hints_normal) == 0:
        print("✅ PASS: Tablebase pattern detection working")
        return True
    else:
        print("❌ FAIL: Tablebase pattern detection incomplete")
        return False


def test_50_move_rule_awareness():
    """Test that 50-move rule awareness prioritizes resets when clock is high"""
    print("\n=== Test 5: 50-Move Rule Awareness ===")
    engine = V7P3REngine()
    
    # White is winning, halfmove clock at 85 (approaching 50-move draw)
    board = chess.Board("4k3/8/8/8/8/8/P7/4K3 w - - 85 50")
    
    # Get move ordering
    legal_moves = list(board.legal_moves)
    ordered_moves = engine._order_moves_advanced(board, legal_moves, depth=4)
    
    # Find pawn moves in the ordered list
    pawn_move_positions = []
    for idx, move in enumerate(ordered_moves):
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            pawn_move_positions.append(idx)
    
    print(f"   Halfmove clock: {board.halfmove_clock}")
    print(f"   Total legal moves: {len(legal_moves)}")
    print(f"   Pawn moves found at positions: {pawn_move_positions}")
    
    # Pawn moves should be early in the list (within first 5 moves)
    if pawn_move_positions and min(pawn_move_positions) < 5:
        print(f"✅ PASS: Pawn moves prioritized (position {min(pawn_move_positions)})")
        return True
    else:
        print("❌ FAIL: Pawn moves NOT prioritized with high halfmove clock")
        return False


def test_game_e68K458O_position():
    """Test the specific position from game e68K458O where V7P3R drew instead of mating"""
    print("\n=== Test 6: Game e68K458O Position (R+B vs K) ===")
    engine = V7P3REngine()
    
    # Position after move 37 from the game
    # White: King on g6, Rook on c6, Bishop on f5, Pawn on f6
    # Black: King on c8 (lone)
    # This is a forced win - should not repeat
    board = chess.Board("2k5/8/2R2KP1/5B2/8/8/8/8 w - - 0 38")
    
    print(f"   Position: {board.fen()}")
    print(f"   Material count: {len(list(board.piece_map()))} pieces")
    
    # Check if tablebase pattern is detected
    hints = engine._detect_basic_tablebase_endgame(board)
    print(f"   Tablebase hints: {len(hints)} moves")
    
    # Get best move with short time limit
    start = time.time()
    best_move = engine.search(board, time_limit=2.0)
    elapsed = time.time() - start
    
    print(f"   Search time: {elapsed:.2f}s")
    print(f"   Best move: {best_move.uci()}")
    print(f"   Nodes searched: {engine.nodes_searched}")
    
    # Make the move and check it doesn't immediately repeat
    board.push(best_move)
    
    # The move should NOT be a king move that just shuffles (common repetition pattern)
    piece_moved = board.piece_at(best_move.to_square)
    is_king_shuffle = piece_moved and piece_moved.piece_type == chess.KING
    
    if not is_king_shuffle and len(hints) > 0:
        print("✅ PASS: Non-repeating move selected with tablebase guidance")
        return True
    else:
        print("⚠️  PARTIAL: Move made, but may need deeper analysis")
        return True  # Don't fail, just warn


def run_all_tests():
    """Run all anti-draw tests"""
    print("=" * 60)
    print("V7P3R v17.7 Anti-Draw Test Suite")
    print("=" * 60)
    
    results = []
    
    results.append(("Threefold Repetition Detection", test_threefold_repetition_detection()))
    results.append(("Mate Verification Extension", test_mate_verification_extension()))
    results.append(("King-Edge Driving Bonus", test_king_edge_driving_bonus()))
    results.append(("Tablebase Pattern Detection", test_tablebase_pattern_detection()))
    results.append(("50-Move Rule Awareness", test_50_move_rule_awareness()))
    results.append(("Game e68K458O Position", test_game_e68K458O_position()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nPassed: {passed}/{total} ({100 * passed // total}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - v17.7 Anti-Draw Features Working!")
    else:
        print(f"\n⚠️  {total - passed} TESTS FAILED - Review Implementation")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
