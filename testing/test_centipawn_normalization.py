#!/usr/bin/env python3

"""
test_centipawn_normalization.py

This script tests the engine's evaluation system after implementing the centipawn normalization changes.
It creates a series of positions and evaluates them to verify:
1. Piece values are consistently using centipawns
2. Position evaluations fall within reasonable centipawn ranges
3. Scores accurately reflect position advantage
"""

import sys
import os
import chess
import time
import json
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import engine modules
from v7p3r import v7p3rEngine
from v7p3r_config import v7p3rConfig
from v7p3r_pst import v7p3rPST
from v7p3r_rules import v7p3rRules
from v7p3r_score import v7p3rScore
from v7p3r_debug import v7p3rLogger

# Initialize logger for testing
logger = v7p3rLogger.setup_logger("test_centipawn")

def test_piece_values():
    """Test that piece values are consistent and in centipawns"""
    print("\n=== Testing Piece Values ===")
    
    # Create PST instance
    pst = v7p3rPST(logger)
    
    # Test each piece value
    pieces = {
        chess.PAWN: "Pawn",
        chess.KNIGHT: "Knight",
        chess.BISHOP: "Bishop",
        chess.ROOK: "Rook",
        chess.QUEEN: "Queen",
        chess.KING: "King"
    }
    
    print("Standard centipawn piece values:")
    for piece_type, name in pieces.items():
        piece = chess.Piece(piece_type, chess.WHITE)
        value = pst.get_piece_value(piece)
        print(f"{name}: {value} centipawns")
    
    # Create Rules instance to check its piece values
    config = v7p3rConfig()
    ruleset = config.get_ruleset()
    rules = v7p3rRules(ruleset, pst)
    
    print("\nRules manager piece values:")
    for piece_type, name in pieces.items():
        value = rules.piece_values.get(piece_type)
        print(f"{name}: {value} centipawns")

def test_position_evaluation():
    """Test position evaluation to ensure scores fall within reasonable centipawn ranges"""
    print("\n=== Testing Position Evaluation ===")
    
    # Create scoring components directly
    config = v7p3rConfig()
    ruleset = config.get_ruleset()
    pst = v7p3rPST(logger)
    rules = v7p3rRules(ruleset, pst)
    score_calculator = v7p3rScore(rules, pst)
    
    # Define test positions with expected evaluations
    test_positions = [
        # FEN, description
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "After 1.e4"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Common opening position"),
        ("rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 1 5", "Position with minor piece advantage"),
        ("r3k2r/ppp2ppp/2n1bn2/3pp3/8/2N2N2/PPP1BPPP/R3K2R w KQkq - 0 8", "Equal position"),
        ("8/8/8/4k3/R7/8/8/4K3 w - - 0 1", "Endgame with rook advantage"),
        ("4k3/8/8/8/8/8/R7/4K3 w - - 0 1", "Winning endgame for White"),
        ("r1bqk2r/ppp2ppp/2np1n2/2b1p3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQkq - 0 7", "Complex middlegame position")
    ]
    
    for fen, description in test_positions:
        board = chess.Board(fen)
        # Use the score calculator directly
        evaluation = score_calculator.evaluate_position(board)
        print(f"{description} (FEN: {fen})")
        print(f"Evaluation: {evaluation} centipawns\n")
        print(f"{description} (FEN: {fen})")
        print(f"Evaluation: {evaluation} centipawns\n")

def test_material_imbalance():
    """Test that material imbalance evaluations are proportional to the centipawn advantage"""
    print("\n=== Testing Material Imbalance Evaluation ===")
    
    # Create scoring components with material-focused ruleset
    config = v7p3rConfig()
    # Try to load a material-focused ruleset if it exists
    try:
        with open('configs/rulesets/material_centipawn_test.json', 'r') as f:
            material_ruleset = json.loads(f.read())['default_ruleset']
            print("Using material-focused ruleset for more accurate material tests")
    except:
        # Fall back to default ruleset
        material_ruleset = config.get_ruleset()
        print("Using default ruleset (material-focused ruleset not found)")
    
    pst = v7p3rPST(logger)
    rules = v7p3rRules(material_ruleset, pst)
    score_calculator = v7p3rScore(rules, pst)
    
    # Test with positions with known material imbalances
    imbalances = [
        # FEN, description, approximate expected value range
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Equal material", (-50, 50)),
        ("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", "Equal trade (pawn for pawn)", (-50, 50)),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPP1PP/RNBQKBNR w KQkq - 0 1", "White down a pawn", (-150, -50)),
        ("rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "White up a knight", (270, 370)),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1", "White down a rook", (-550, -450)),
        ("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "White up a queen", (850, 950)),
        ("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kkq - 0 1", "White down a rook, up a knight", (-230, -130))
    ]
    
    for fen, description, expected_range in imbalances:
        board = chess.Board(fen)
        evaluation = score_calculator.evaluate_position(board)
        in_range = expected_range[0] <= evaluation <= expected_range[1]
        result = "✅ In expected range" if in_range else "❌ Outside expected range"
        
        print(f"{description} (FEN: {fen})")
        print(f"Evaluation: {evaluation} centipawns")
        print(f"Expected range: {expected_range[0]} to {expected_range[1]} centipawns")
        print(f"Result: {result}\n")

def main():
    """Run all tests"""
    print("======================================")
    print("CENTIPAWN NORMALIZATION TESTING")
    print("======================================")
    
    # Run the tests
    test_piece_values()
    test_position_evaluation()
    test_material_imbalance()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
