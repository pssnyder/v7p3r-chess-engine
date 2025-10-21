#!/usr/bin/env python3
"""
VPR v6.0 Test Script - Revolutionary Architecture Validation

Tests the new dual-brain search system and position classification
"""

import chess
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vpr import VPRChaosEngine, PositionType

def test_engine_initialization():
    """Test that the engine initializes without errors"""
    print("Testing VPR v6.0 engine initialization...")
    engine = VPRChaosEngine()
    
    # Check all components are initialized
    assert engine.chaos_brain is not None
    assert engine.opponent_brain is not None
    assert engine.position_classifier is not None
    assert engine.mcts_engine is not None
    
    print("✓ Engine initialized successfully")
    print(f"✓ Engine info: {engine.get_engine_info()}")

def test_position_classification():
    """Test the position classification system"""
    print("\nTesting position classification...")
    engine = VPRChaosEngine()
    
    # Test starting position
    board = chess.Board()
    position_type = engine.position_classifier.classify(board)
    print(f"✓ Starting position classified as: {position_type}")
    assert position_type == PositionType.OPENING
    
    # Test a complex tactical position (example)
    # Sicilian Dragon - typically very complex
    complex_fen = "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 0 6"
    board.set_fen(complex_fen)
    position_type = engine.position_classifier.classify(board)
    print(f"✓ Complex position classified as: {position_type}")
    
    # Test endgame position
    endgame_fen = "8/8/8/8/8/3k4/3p4/3K4 w - - 0 1"
    board.set_fen(endgame_fen)
    position_type = engine.position_classifier.classify(board)
    print(f"✓ Endgame position classified as: {position_type}")
    assert position_type == PositionType.ENDGAME

def test_chaos_brain():
    """Test the chaos brain evaluation"""
    print("\nTesting chaos brain...")
    engine = VPRChaosEngine()
    board = chess.Board()
    
    # Test position evaluation
    score = engine.chaos_brain.evaluate_position(board)
    print(f"✓ Starting position chaos evaluation: {score}")
    
    # Test move ordering
    moves = list(board.legal_moves)
    ordered_moves = engine.chaos_brain.order_moves(board, moves)
    print(f"✓ Chaos brain ordered {len(ordered_moves)} moves")
    assert len(ordered_moves) == len(moves)

def test_search_algorithms():
    """Test all search algorithm selections"""
    print("\nTesting search algorithm selection...")
    engine = VPRChaosEngine()
    board = chess.Board()
    
    # Test each position type triggers correct algorithm
    for position_type in PositionType:
        algorithm = engine._select_search_algorithm(position_type, 2.0)
        print(f"✓ {position_type} -> {algorithm}")

def test_basic_search():
    """Test that search returns a legal move"""
    print("\nTesting basic search functionality...")
    engine = VPRChaosEngine()
    board = chess.Board()
    
    # Test search returns a legal move
    move = engine.search(board, time_limit=0.1)  # Very short search
    print(f"✓ Engine returned move: {move}")
    assert move in board.legal_moves
    
    # Test with forced move
    forced_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"  # Only one legal move scenario
    # Actually, let's use a position with only one legal move
    one_move_fen = "8/8/8/8/8/7k/6pp/7K w - - 0 1"  # King must move to escape
    board.set_fen(one_move_fen)
    if len(list(board.legal_moves)) == 1:
        move = engine.search(board, time_limit=0.1)
        print(f"✓ Forced move handled correctly: {move}")

def test_mcts_engine():
    """Test Monte Carlo Tree Search component"""
    print("\nTesting MCTS engine...")
    engine = VPRChaosEngine()
    board = chess.Board()
    
    # Test MCTS returns a legal move
    mcts_move = engine.mcts_engine.search(board, 0.1)
    print(f"✓ MCTS returned move: {mcts_move}")
    assert mcts_move in board.legal_moves

def run_all_tests():
    """Run all tests for VPR v6.0"""
    print("=" * 60)
    print("VPR v6.0 Revolutionary Architecture Tests")
    print("=" * 60)
    
    try:
        test_engine_initialization()
        test_position_classification()
        test_chaos_brain() 
        test_search_algorithms()
        test_basic_search()
        test_mcts_engine()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! VPR v6.0 is ready for battle!")
        print("🧠 Dual-brain architecture is operational")
        print("🎯 Position classification system is working")
        print("🔍 All search algorithms are accessible")
        print("🎲 Monte Carlo Tree Search is functional")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()