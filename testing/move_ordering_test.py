#!/usr/bin/env python3

"""
V7P3R Move Ordering Performance Test

Tests the quality of V7P3R's move ordering against the corrected perft positions.
This validates that our move sorting heuristics are working effectively.

Key metrics:
1. Move ordering consistency
2. Best move ranking (where does the best move appear in our ordering)
3. Capture prioritization
4. Check/threat move prioritization
5. Killer move and history heuristic effectiveness
"""

import chess
import time
import sys
import os

# Add src directory to path for importing V7P3R engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def test_move_ordering():
    """Test V7P3R's move ordering quality"""
    
    # Use the corrected perft test positions
    test_positions = [
        {
            "name": "Position 1 (Initial)",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "expected_good_moves": ["e2e4", "d2d4", "g1f3", "b1c3"]  # Standard opening moves
        },
        {
            "name": "Position 2 (Kiwipete)",
            "fen": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
            "expected_good_moves": ["f3f7", "d5e6", "e5f7"]  # Tactical shots should be first
        },
        {
            "name": "Position 3",
            "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
            "expected_good_moves": ["b4b8", "b4h4", "a5a6"]  # Rook moves and pawn push
        },
        {
            "name": "Position 4 (Corrected)",
            "fen": "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
            "expected_good_moves": ["b5d3", "a5d2", "h3f2"]  # Tactical moves should be prioritized
        },
        {
            "name": "Position 5",
            "fen": "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "expected_good_moves": ["e2f4", "d1d8", "c4f7"]  # Tactical themes
        }
    ]
    
    engine = V7P3REngine()
    
    print("V7P3R Move Ordering Performance Test")
    print("=" * 60)
    
    total_positions = 0
    total_move_ordering_score = 0
    capture_prioritization_score = 0
    check_prioritization_score = 0
    
    for position in test_positions:
        print(f"\n{position['name']}")
        print(f"FEN: {position['fen']}")
        print("-" * 40)
        
        board = chess.Board(position['fen'])
        
        # Get all legal moves (unsorted)
        legal_moves = list(board.legal_moves)
        print(f"Total legal moves: {len(legal_moves)}")
        
        # Get V7P3R's ordered moves
        start_time = time.time()
        ordered_moves = engine._order_moves_advanced(board, legal_moves, depth=2, tt_move=None)
        ordering_time = time.time() - start_time
        
        print(f"Move ordering time: {ordering_time*1000:.2f}ms")
        
        # Analyze move ordering quality
        print(f"\nTop 10 ordered moves:")
        for i, move in enumerate(ordered_moves[:10], 1):
            move_type = classify_move(board, move)
            print(f"  {i:2d}. {move} ({move_type})")
        
        # Check if expected good moves are prioritized
        expected_moves = position.get('expected_good_moves', [])
        if expected_moves:
            print(f"\nExpected good moves ranking:")
            move_ranking_scores = []
            
            for expected_move_uci in expected_moves:
                try:
                    expected_move = chess.Move.from_uci(expected_move_uci)
                    if expected_move in ordered_moves:
                        rank = ordered_moves.index(expected_move) + 1
                        total_moves = len(ordered_moves)
                        # Score: 100% for rank 1, decreasing linearly
                        score = max(0, 100 * (total_moves - rank + 1) / total_moves)
                        move_ranking_scores.append(score)
                        print(f"    {expected_move_uci}: Rank {rank}/{total_moves} (Score: {score:.1f}%)")
                    else:
                        print(f"    {expected_move_uci}: NOT FOUND (illegal move?)")
                except:
                    print(f"    {expected_move_uci}: INVALID UCI")
            
            if move_ranking_scores:
                avg_ranking_score = sum(move_ranking_scores) / len(move_ranking_scores)
                print(f"    Average ranking score: {avg_ranking_score:.1f}%")
                total_move_ordering_score += avg_ranking_score
        
        # Analyze capture prioritization
        captures = [move for move in ordered_moves if board.is_capture(move)]
        if captures:
            # Check if captures appear early in the list
            capture_positions = [ordered_moves.index(cap) + 1 for cap in captures[:5]]  # Top 5 captures
            avg_capture_position = sum(capture_positions) / len(capture_positions)
            capture_score = max(0, 100 * (len(ordered_moves) - avg_capture_position + 1) / len(ordered_moves))
            capture_prioritization_score += capture_score
            print(f"\nCapture prioritization:")
            print(f"  {len(captures)} captures found")
            print(f"  Average position of top captures: {avg_capture_position:.1f}")
            print(f"  Capture prioritization score: {capture_score:.1f}%")
        
        # Analyze check prioritization
        checks = [move for move in ordered_moves if board.gives_check(move)]
        if checks:
            check_positions = [ordered_moves.index(check) + 1 for check in checks[:3]]  # Top 3 checks
            avg_check_position = sum(check_positions) / len(check_positions)
            check_score = max(0, 100 * (len(ordered_moves) - avg_check_position + 1) / len(ordered_moves))
            check_prioritization_score += check_score
            print(f"\nCheck prioritization:")
            print(f"  {len(checks)} checking moves found")
            print(f"  Average position of checks: {avg_check_position:.1f}")
            print(f"  Check prioritization score: {check_score:.1f}%")
        
        # Test move ordering consistency (run multiple times)
        consistency_test_results = []
        for test_run in range(3):
            test_ordered = engine._order_moves_advanced(board, legal_moves, depth=2, tt_move=None)
            # Check if ordering is identical
            is_consistent = test_ordered == ordered_moves
            consistency_test_results.append(is_consistent)
        
        consistency_rate = sum(consistency_test_results) / len(consistency_test_results) * 100
        print(f"\nMove ordering consistency: {consistency_rate:.1f}% (3 runs)")
        
        total_positions += 1
    
    # Overall performance summary
    print(f"\n{'='*60}")
    print("MOVE ORDERING PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    if total_positions > 0:
        avg_move_ordering = total_move_ordering_score / total_positions
        avg_capture_prioritization = capture_prioritization_score / total_positions
        avg_check_prioritization = check_prioritization_score / total_positions
        
        print(f"Average move ranking score: {avg_move_ordering:.1f}%")
        print(f"Average capture prioritization: {avg_capture_prioritization:.1f}%")
        print(f"Average check prioritization: {avg_check_prioritization:.1f}%")
        
        overall_score = (avg_move_ordering + avg_capture_prioritization + avg_check_prioritization) / 3
        print(f"Overall move ordering score: {overall_score:.1f}%")
        
        if overall_score >= 80:
            print("🎯 EXCELLENT move ordering!")
        elif overall_score >= 65:
            print("🎯 GOOD move ordering")
        elif overall_score >= 50:
            print("⚠️  FAIR move ordering - room for improvement")
        else:
            print("❌ POOR move ordering - needs significant improvement")
    
    return total_move_ordering_score / total_positions if total_positions > 0 else 0

def classify_move(board: chess.Board, move: chess.Move) -> str:
    """Classify a move type for analysis"""
    classifications = []
    
    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                          chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
            value = piece_values.get(captured_piece.piece_type, 0)
            classifications.append(f"Capture:{piece_values.get(captured_piece.piece_type, 0)}")
    
    if board.gives_check(move):
        classifications.append("Check")
    
    if move.promotion:
        classifications.append(f"Promotion:{chess.piece_name(move.promotion)}")
    
    # Castling
    if board.is_castling(move):
        classifications.append("Castling")
    
    # En passant
    if board.is_en_passant(move):
        classifications.append("En passant")
    
    if not classifications:
        piece = board.piece_at(move.from_square)
        if piece:
            classifications.append(f"{chess.piece_name(piece.piece_type).title()}")
        else:
            classifications.append("Quiet")
    
    return ", ".join(classifications)

if __name__ == "__main__":
    test_move_ordering()