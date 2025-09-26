#!/usr/bin/env python3
"""
V12.3 Tactical Evaluation Test
Test the engine's ability to detect and evaluate tactical patterns like forks, pins, and skewers
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

def test_knight_fork_detection():
    """Test knight fork detection"""
    print("=== KNIGHT FORK DETECTION TEST ===")
    
    engine = V7P3REngine()
    
    # Test 1: Classic knight fork - knight can fork king and queen
    fork_fen = "r3k3/8/8/8/4N3/8/8/R3K2R w - - 0 1"  # Knight on e4 can fork king on e8 and rook on a8
    print(f"\n1. Knight Fork Test (Ne4 forking king and rook):")
    print(f"   FEN: {fork_fen}")
    
    board = chess.Board(fork_fen)
    
    # Test the knight move that creates a fork
    knight_moves = []
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.KNIGHT:
            knight_moves.append(move)
    
    # Get tactical bonus for knight moves
    for move in knight_moves:
        tactical_score = engine._detect_bitboard_tactics(board, move)
        print(f"   {move}: tactical bonus = {tactical_score:.1f}")
    
    # Test 2: Knight fork in a more practical position
    practical_fork_fen = "rnbqkb1r/ppp2ppp/3p1n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
    print(f"\n2. Practical Knight Fork Position:")
    print(f"   FEN: {practical_fork_fen}")
    
    board2 = chess.Board(practical_fork_fen)
    # Look for Ne7+ which would fork king and rook (if there was a rook on a8)
    for move in board2.legal_moves:
        piece = board2.piece_at(move.from_square)
        if piece and piece.piece_type == chess.KNIGHT and move.to_square == chess.E7:
            tactical_score = engine._detect_bitboard_tactics(board2, move)
            print(f"   {move} (potential fork): tactical bonus = {tactical_score:.1f}")

def test_pin_skewer_detection():
    """Test pin and skewer detection"""
    print(f"\n=== PIN/SKEWER DETECTION TEST ===")
    
    engine = V7P3REngine()
    
    # Test 1: Simple pin - bishop pins knight to king
    pin_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    print(f"\n1. Bishop Pin Test:")
    print(f"   FEN: {pin_fen}")
    
    board = chess.Board(pin_fen)
    
    # Test bishop moves that could create pins
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.BISHOP:
            tactical_score = engine._detect_bitboard_tactics(board, move)
            if tactical_score > 0:
                print(f"   {move}: tactical bonus = {tactical_score:.1f}")
    
    # Test 2: Rook creating a skewer
    skewer_fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"  # Rooks can potentially skewer
    print(f"\n2. Rook Skewer Test:")  
    print(f"   FEN: {skewer_fen}")
    
    board2 = chess.Board(skewer_fen)
    rook_moves_with_bonus = []
    
    for move in board2.legal_moves:
        piece = board2.piece_at(move.from_square)
        if piece and piece.piece_type == chess.ROOK:
            tactical_score = engine._detect_bitboard_tactics(board2, move)
            if tactical_score > 0:
                rook_moves_with_bonus.append((move, tactical_score))
    
    for move, score in rook_moves_with_bonus:
        print(f"   {move}: tactical bonus = {score:.1f}")

def test_move_ordering_with_tactics():
    """Test that tactical moves get priority in move ordering"""
    print(f"\n=== MOVE ORDERING WITH TACTICS TEST ===")
    
    engine = V7P3REngine()
    
    # Position where a tactical move should be highly ranked
    tactical_fen = "rnbqkb1r/ppp2ppp/3p1n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
    board = chess.Board(tactical_fen)
    
    print(f"Position: {tactical_fen}")
    
    # Get all legal moves
    moves = list(board.legal_moves)
    print(f"Total legal moves: {len(moves)}")
    
    # Order moves using the engine's advanced move ordering
    ordered_moves = engine._order_moves_advanced(board, moves, depth=1, tt_move=None)
    
    print(f"\nTop 10 moves by engine ordering:")
    for i, move in enumerate(ordered_moves[:10]):
        # Get tactical bonus for this move
        tactical_score = engine._detect_bitboard_tactics(board, move)
        piece = board.piece_at(move.from_square)
        piece_name = piece.symbol() if piece else "?"
        
        print(f"   {i+1:2}. {piece_name}{move} (tactical: {tactical_score:.1f})")

def test_search_tactical_preference():
    """Test if the engine prefers tactical moves in search"""
    print(f"\n=== SEARCH TACTICAL PREFERENCE TEST ===")
    
    engine = V7P3REngine()
    
    # Position with clear tactical opportunity
    # White can play Nxe7+ forking king and queen
    tactical_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    board = chess.Board(tactical_fen)
    
    print(f"Searching position with tactical opportunity...")
    print(f"FEN: {tactical_fen}")
    
    # Search for best move
    best_move = engine.search(board, time_limit=3.0)
    
    # Get tactical score for the best move
    tactical_score = engine._detect_bitboard_tactics(board, best_move)
    piece = board.piece_at(best_move.from_square)
    
    print(f"Engine's best move: {piece.symbol() if piece else '?'}{best_move}")
    print(f"Tactical bonus for best move: {tactical_score:.1f}")
    print(f"Nodes searched: {engine.nodes_searched:,}")

def summarize_tactical_capabilities():
    """Summarize what tactical patterns the engine can detect"""
    print(f"\n=== TACTICAL CAPABILITIES SUMMARY ===")
    
    print("Based on code analysis, V7P3R v12.3 has the following tactical detection:")
    
    print("\n✅ IMPLEMENTED TACTICS:")
    print("  • Knight Forks: Detects when knight attacks 2+ enemy pieces")
    print("    - Bonus: 50 points + 25 points per high-value target (Q, R, K)")
    print("    - Uses pre-computed knight attack bitboards for fast detection")
    
    print("  • Pin/Skewer Detection: Basic pattern recognition")
    print("    - Bonus: 15 points for sliding pieces aligned with enemy king")
    print("    - Covers bishops, rooks, and queens")
    print("    - Simplified implementation (not full ray-casting)")
    
    print("  • Integration: Tactical bonuses are used in:")
    print("    - Move ordering (prioritizes tactical moves)")
    print("    - Capture moves (MVV-LVA + tactical bonus)")
    print("    - Check moves (additional tactical scoring)")
    print("    - Quiet moves (identifies tactical quiet moves)")
    
    print("\n❌ NOT IMPLEMENTED:")
    print("  • Discovered attacks")
    print("  • Double attacks (non-knight)")
    print("  • Deflection/decoy tactics")
    print("  • Advanced pin/skewer patterns")
    print("  • Sacrifice combinations")
    
    print("\n⚠️  LIMITATIONS:")
    print("  • Advanced tactical detection was disabled in v10.6 due to 70% performance loss")
    print("  • Current implementation uses simplified heuristics")
    print("  • Pin/skewer detection is basic (alignment check only)")
    print("  • No deep tactical combination analysis")
    
    print("\n🔧 PERFORMANCE:")
    print("  • Tactical detection integrated into move ordering")
    print("  • Uses bitboards for fast pattern matching")
    print("  • Balanced between tactical awareness and search speed")

if __name__ == "__main__":
    print("V7P3R Chess Engine - Tactical Evaluation Analysis")
    print("=" * 50)
    
    test_knight_fork_detection()
    test_pin_skewer_detection() 
    test_move_ordering_with_tactics()
    test_search_tactical_preference()
    summarize_tactical_capabilities()
    
    print(f"\n🎯 CONCLUSION:")
    print("The engine HAS tactical evaluation, but it's simplified for performance.")
    print("It can detect basic knight forks and has rudimentary pin/skewer detection.")
    print("More advanced tactical patterns were disabled due to performance concerns.")