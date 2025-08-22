#!/usr/bin/env python3
"""
Test V7P3R v5.1 enhanced capture logic and threat detection
"""

import chess
from v7p3r import V7P3REvaluationEngine

def test_capture_logic():
    """Test the enhanced capture evaluation"""
    print("Testing V7P3R v5.1 Enhanced Capture Logic")
    print("=" * 50)
    
    engine = V7P3REvaluationEngine()
    
    # Test 1: Basic capture with good SEE
    print("\n1. Testing basic good capture:")
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    move = chess.Move.from_uci("e4e5")  # Pawn takes pawn
    if board.is_legal(move):
        see_score = engine._static_exchange_evaluation(board, move)
        print(f"  Pawn x Pawn: SEE score = {see_score}")
    
    # Test 2: Bad capture (hanging piece)
    print("\n2. Testing bad capture (hanging queen):")
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 2")
    move = chess.Move.from_uci("d1h5")  # Queen to h5 (can be taken)
    if board.is_legal(move):
        board.push(move)
        # Now test if black takes the queen
        take_queen = chess.Move.from_uci("d8h4")
        if board.is_legal(take_queen):
            see_score = engine._static_exchange_evaluation(board, take_queen)
            print(f"  Queen x Queen: SEE score = {see_score}")
        board.pop()
    
    # Test 3: Threat detection
    print("\n3. Testing threat detection:")
    board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 3")
    
    # Check if pieces are threatened
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == chess.WHITE:
            is_threatened = engine._is_piece_threatened(board, square, piece.color)
            if is_threatened:
                print(f"  White {piece.symbol()} on {chess.square_name(square)} is threatened")
    
    # Test 4: Move ordering with threats
    print("\n4. Testing move ordering with threat awareness:")
    moves = list(board.legal_moves)
    ordered_moves = engine.order_moves(board, moves)
    
    print("  Top 5 moves by score:")
    for i, move in enumerate(ordered_moves[:5]):
        score = engine._order_move_score(board, move)
        print(f"    {i+1}. {move.uci()}: score = {score:.0f}")

def test_quiescence_improvements():
    """Test the enhanced quiescence search"""
    print("\n\nTesting V7P3R v5.1 Enhanced Quiescence Search")
    print("=" * 50)
    
    engine = V7P3REvaluationEngine()
    
    # Test tactical position with captures
    print("\n1. Testing tactical position:")
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5")
    
    # Run quiescence search
    alpha = -float('inf')
    beta = float('inf')
    score = engine._quiescence_search(board, alpha, beta, True)
    print(f"  Quiescence evaluation: {score:.2f}")
    print(f"  Nodes searched in quiescence: {engine.nodes_searched}")

def test_full_position():
    """Test a full position from the tournament games"""
    print("\n\nTesting Full Position Analysis")
    print("=" * 50)
    
    engine = V7P3REvaluationEngine()
    engine.depth = 4
    
    # Position where V7P3R played poorly (from the tournament)
    board = chess.Board()  # Starting position
    
    print("\n1. Starting position analysis:")
    best_move = engine.search(board, chess.WHITE)
    print(f"  Best move: {best_move}")
    print(f"  Nodes searched: {engine.nodes_searched}")
    
    # Test the problematic Nh6 move
    print("\n2. Evaluating Nh6 (bad move from tournament):")
    nh6_move = chess.Move.from_uci("g8h6")
    if board.is_legal(nh6_move):
        score = engine._order_move_score(board, nh6_move)
        print(f"  Nh6 move score: {score:.0f}")
        
        # Compare with a better move like Nf6
        nf6_move = chess.Move.from_uci("g8f6")
        if board.is_legal(nf6_move):
            score = engine._order_move_score(board, nf6_move)
            print(f"  Nf6 move score: {score:.0f}")

if __name__ == "__main__":
    test_capture_logic()
    test_quiescence_improvements()
    test_full_position()
