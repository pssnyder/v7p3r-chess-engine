#!/usr/bin/env python3
"""
Debug Skewer Detection
Let's understand why the skewer detection isn't working
"""

import chess
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r_scoring_calculation import V7P3RScoringCalculationClean

def debug_skewer_detection():
    """Debug why skewer detection returns 0"""
    
    print("🔧 DEBUGGING SKEWER DETECTION")
    print("=" * 50)
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    scorer = V7P3RScoringCalculationClean(piece_values)
    
    # Test the problematic skewer position
    print("📍 Testing Skewer Position:")
    board = chess.Board("8/8/8/2q2r2/2R5/8/8/K7 w - - 0 1")
    print("FEN: 8/8/8/2q2r2/2R5/8/8/K7 w - - 0 1")
    
    # Let's examine this position piece by piece
    print("\n🔍 Position Analysis:")
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            print(f"  {piece.symbol()} on {chess.square_name(square)}")
    
    # Check what our white rook attacks
    rook_square = chess.C4
    print(f"\n🎯 White Rook on {chess.square_name(rook_square)}:")
    rook_attacks = board.attacks(rook_square)
    print(f"  Attacks squares: {[chess.square_name(sq) for sq in rook_attacks]}")
    
    # Check pieces on attacked squares
    attacked_pieces = []
    for sq in rook_attacks:
        piece = board.piece_at(sq)
        if piece and piece.color == chess.BLACK:
            attacked_pieces.append((chess.square_name(sq), piece.symbol()))
    
    print(f"  Attacks enemy pieces: {attacked_pieces}")
    
    # Check if queen and rook are aligned
    queen_square = chess.C5
    rook_enemy_square = chess.F5
    
    print(f"\n📐 Alignment Check:")
    print(f"  Queen on {chess.square_name(queen_square)}")
    print(f"  Enemy Rook on {chess.square_name(rook_enemy_square)}")
    
    queen_file = chess.square_file(queen_square)
    queen_rank = chess.square_rank(queen_square)
    rook_file = chess.square_file(rook_enemy_square)
    rook_rank = chess.square_rank(rook_enemy_square)
    
    print(f"  Queen: file {queen_file}, rank {queen_rank}")
    print(f"  Enemy Rook: file {rook_file}, rank {rook_rank}")
    print(f"  Same rank? {queen_rank == rook_rank}")
    print(f"  Same file? {queen_file == rook_file}")
    
    # Test our skewer detection method
    print(f"\n🧪 Running Skewer Detection:")
    result = scorer._tactical_skewer_detection(board, chess.WHITE)
    print(f"  Result: {result:.2f}")
    
    # Test specific method for our rook
    print(f"\n🎯 Testing Rook Skewer Opportunities:")
    rook_result = scorer._detect_skewer_opportunities(board, rook_square, chess.WHITE)
    print(f"  Rook result: {rook_result:.2f}")
    
    # Test the behind-check method directly
    print(f"\n🔍 Testing Behind-Check Method:")
    behind_result = scorer._check_for_skewer_behind(board, rook_square, queen_square, chess.WHITE)
    print(f"  Behind-check result: {behind_result:.2f}")
    
    # Let's manually trace through the _check_for_skewer_behind method
    print(f"\n🔍 Manual Behind-Check Tracing:")
    print(f"  Attacker (White Rook): {chess.square_name(rook_square)} = file {chess.square_file(rook_square)}, rank {chess.square_rank(rook_square)}")
    print(f"  Front (Black Queen): {chess.square_name(queen_square)} = file {chess.square_file(queen_square)}, rank {chess.square_rank(queen_square)}")
    
    attacker_file = chess.square_file(rook_square)  # c4 = file 2
    attacker_rank = chess.square_rank(rook_square)  # c4 = rank 3
    front_file = chess.square_file(queen_square)    # c5 = file 2
    front_rank = chess.square_rank(queen_square)    # c5 = rank 4
    
    file_diff = front_file - attacker_file          # 2 - 2 = 0
    rank_diff = front_rank - attacker_rank          # 4 - 3 = 1
    
    print(f"  File diff: {file_diff}, Rank diff: {rank_diff}")
    
    if file_diff != 0:
        file_dir = 1 if file_diff > 0 else -1
    else:
        file_dir = 0
        
    if rank_diff != 0:
        rank_dir = 1 if rank_diff > 0 else -1
    else:
        rank_dir = 0
        
    print(f"  Direction: file_dir = {file_dir}, rank_dir = {rank_dir}")
    
    # Continue in the same direction
    current_file = front_file + file_dir            # 2 + 0 = 2
    current_rank = front_rank + rank_dir            # 4 + 1 = 5
    
    print(f"  Looking behind at: file {current_file}, rank {current_rank}")
    
    if 0 <= current_file <= 7 and 0 <= current_rank <= 7:
        behind_square = chess.square(current_file, current_rank)
        print(f"  Behind square: {chess.square_name(behind_square)}")
        behind_piece = board.piece_at(behind_square)
        if behind_piece:
            print(f"  Behind piece: {behind_piece.symbol()}")
            front_piece = board.piece_at(queen_square)
            if front_piece:
                front_value = scorer.tactical_piece_values.get(front_piece.piece_type, 0)
                behind_value = scorer.tactical_piece_values.get(behind_piece.piece_type, 0)
                print(f"  Front value (queen): {front_value}")
                print(f"  Behind value (rook): {behind_value}")
                print(f"  Front > Behind? {front_value > behind_value}")
                print(f"  Front > 100? {front_value > 100}")
                if front_value > behind_value and front_value > 100:
                    skewer_value = 10.0 + (front_value - behind_value) / 100
                    print(f"  ✅ SKEWER FOUND! Value: {skewer_value}")
                else:
                    print(f"  ❌ No skewer - conditions not met")
        else:
            print(f"  No piece behind")
    else:
        print(f"  Behind square out of bounds")

def create_correct_skewer_position():
    """Create a position where skewer should definitely work"""
    
    print(f"\n🎯 CREATING CORRECT SKEWER POSITION")
    print("=" * 50)
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    scorer = V7P3RScoringCalculationClean(piece_values)
    
    # Create a clear skewer: queen in front, rook behind, on same rank
    print("📍 Clear Skewer Position:")
    board = chess.Board("8/8/8/2q1r3/2R5/8/8/K7 w - - 0 1")
    print("FEN: 8/8/8/2q1r3/2R5/8/8/K7 w - - 0 1")
    print("White Rook c4, Black Queen c5, Black Rook e5")
    print("All on rank 5 and 4! Should be a skewer!")
    
    # Analyze this position
    print(f"\n🔍 Position Analysis:")
    rook_white = chess.C4
    queen_black = chess.C5  
    rook_black = chess.E5
    
    print(f"White Rook: {chess.square_name(rook_white)} (file {chess.square_file(rook_white)}, rank {chess.square_rank(rook_white)})")
    print(f"Black Queen: {chess.square_name(queen_black)} (file {chess.square_file(queen_black)}, rank {chess.square_rank(queen_black)})")
    print(f"Black Rook: {chess.square_name(rook_black)} (file {chess.square_file(rook_black)}, rank {chess.square_rank(rook_black)})")
    
    # Check attacks
    white_rook_attacks = board.attacks(rook_white)
    print(f"White rook attacks: {[chess.square_name(sq) for sq in white_rook_attacks]}")
    print(f"Attacks black queen? {queen_black in white_rook_attacks}")
    
    # Test skewer detection
    result = scorer._tactical_skewer_detection(board, chess.WHITE)
    print(f"Skewer detection result: {result:.2f}")
    
    return result

def test_simple_rank_skewer():
    """Test the simplest possible rank skewer"""
    
    print(f"\n⚡ SIMPLEST RANK SKEWER TEST")
    print("=" * 50)
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    scorer = V7P3RScoringCalculationClean(piece_values)
    
    # Simplest skewer: Rook, Queen, Rook on same rank
    print("📍 Simplest Skewer:")
    board = chess.Board("8/8/8/R1q1r3/8/8/8/K7 w - - 0 1")
    print("FEN: 8/8/8/R1q1r3/8/8/8/K7 w - - 0 1")
    print("White Rook a5, Black Queen c5, Black Rook e5 - all on rank 5")
    
    rook_white = chess.A5
    queen_black = chess.C5
    rook_black = chess.E5
    
    print(f"Pieces aligned on rank {chess.square_rank(rook_white)}")
    
    # Check if white rook attacks queen
    attacks = board.attacks(rook_white)
    print(f"White rook attacks: {[chess.square_name(sq) for sq in attacks]}")
    print(f"Attacks queen on c5? {queen_black in attacks}")
    
    # Test detection
    result = scorer._tactical_skewer_detection(board, chess.WHITE)
    print(f"Skewer detection: {result:.2f}")
    
    if result == 0:
        print("❌ Still not working! Let's debug the method...")
        # Debug the method step by step
        print("\n🔧 Method debugging:")
        result2 = scorer._detect_skewer_opportunities(board, rook_white, chess.WHITE)
        print(f"_detect_skewer_opportunities: {result2:.2f}")

if __name__ == "__main__":
    debug_skewer_detection()
    create_correct_skewer_position()
    test_simple_rank_skewer()
