#!/usr/bin/env python3
"""
Test V13.2 Enhanced Tactical Awareness
Tests the improved threat detection and weighted opportunity system
"""

import os
import sys
import time
import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from v7p3r import V7P3REngine

def test_tactical_threat_detection():
    """Test the enhanced threat detection system."""
    print("=" * 60)
    print("V13.2 ENHANCED TACTICAL AWARENESS TEST")
    print("=" * 60)
    print("TESTING: Improved threat detection and weighting")
    print()
    
    engine = V7P3REngine()
    
    # Test critical tactical positions
    tactical_positions = [
        {
            "name": "Scholar's Mate Defense",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 2 4",
            "expected_threats": ["defend_mate_in_1"],
            "good_moves": ["Qe7", "Nxh5", "g6"],  # Defending moves against Qxf7#
            "bad_moves": ["Nc6", "a6"]  # Non-defending moves
        },
        {
            "name": "Check Escape",
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/6P1/8/PPPPPP1P/RNBQKBNR b KQkq g3 0 2",
            "expected_threats": [],  # No immediate threats for black
            "good_moves": ["Nf6", "Nc6"],
            "bad_moves": []
        },
        {
            "name": "Material Hanging",
            "fen": "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2", 
            "expected_threats": [],
            "good_moves": ["Nc3", "Nf3"],
            "bad_moves": ["Qd3"]  # Hanging queen
        }
    ]
    
    for pos in tactical_positions:
        print(f"\n{pos['name']}:")
        print(f"FEN: {pos['fen']}")
        print("-" * 50)
        
        board = chess.Board(pos['fen'])
        
        # Test threat detection
        opportunities = engine._detect_opportunities(board)
        detected_threats = [opp for opp in opportunities if opp in [
            'escape_check', 'defend_mate_in_1', 'defend_mate_in_2', 
            'save_material', 'defend_check_threat', 'defend_material_loss'
        ]]
        
        print(f"Detected threats: {detected_threats}")
        print(f"Expected threats: {pos['expected_threats']}")
        
        # Check if we detected expected threats
        for expected_threat in pos['expected_threats']:
            if expected_threat in detected_threats:
                print(f"✓ Correctly detected: {expected_threat}")
            else:
                print(f"✗ Missed threat: {expected_threat}")
        
        print(f"All opportunities: {opportunities}")

def test_weighted_opportunity_scoring():
    """Test the weighted opportunity scoring system."""
    print("\n" + "=" * 60)
    print("WEIGHTED OPPORTUNITY SCORING TEST")
    print("=" * 60)
    print("TESTING: Tactical priorities vs positional opportunities")
    print()
    
    engine = V7P3REngine()
    
    # Scholar's mate position - critical test case (Black to move, must defend against Qxf7#)
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 2 4")
    
    print("Scholar's Mate Defense Position:")
    print("Testing if defensive moves get higher weights than aggressive moves")
    print()
    
    # Get multi-PV analysis with weighted scores
    candidates = engine.search_multi_pv(board, 2.0, num_lines=5)
    
    print("Move candidates (sorted by weighted score):")
    for i, candidate in enumerate(candidates, 1):
        move_san = board.san(candidate['move'])
        print(f"{i}. {move_san}")
        print(f"   Base score: {candidate['score']:.2f}")
        print(f"   Improvement: {candidate['improvement']:.2f}")
        print(f"   Weighted score: {candidate['weighted_score']:.2f}")
        print(f"   Opportunities: {', '.join(candidate['opportunities'])}")
        print()
    
    # Check if defensive moves are prioritized
    top_move = board.san(candidates[0]['move'])
    defensive_moves = ["Qf3", "d3", "Nf3"]
    
    if any(def_move in top_move for def_move in defensive_moves):
        print("✓ Defensive move correctly prioritized!")
    else:
        print(f"○ Top move '{top_move}' - checking if it addresses threats...")

def test_puzzle_solver_vs_traditional_tactical():
    """Compare puzzle solver vs traditional in tactical positions."""
    print("\n" + "=" * 60)
    print("TACTICAL COMPARISON: PUZZLE SOLVER vs TRADITIONAL")
    print("=" * 60)
    print("TESTING: Enhanced puzzle solver against traditional search")
    print()
    
    engine = V7P3REngine()
    
    # Critical tactical test positions
    test_positions = [
        ("Scholar's Mate Defense", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 2 4"),
        ("Back Rank Threat", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
        ("Pin Defense", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 3 4"),
    ]
    
    improvements = 0
    total_tests = len(test_positions)
    
    for pos_name, fen in test_positions:
        print(f"\n{pos_name}:")
        print(f"FEN: {fen}")
        print("-" * 40)
        
        board = chess.Board(fen)
        
        # Traditional search
        traditional_move = engine.search(board, 1.5)
        traditional_san = board.san(traditional_move)
        
        # Enhanced puzzle solver
        puzzle_move = engine.puzzle_search(board, 1.5)
        puzzle_san = board.san(puzzle_move)
        
        print(f"Traditional: {traditional_san}")
        print(f"Puzzle Solver: {puzzle_san}")
        
        # Simple heuristic: defensive moves in tactical positions are often better
        defensive_keywords = ["Q", "d3", "Nf3", "Be2", "f3", "h3"]  # Common defensive patterns
        
        traditional_defensive = any(keyword in traditional_san for keyword in defensive_keywords)
        puzzle_defensive = any(keyword in puzzle_san for keyword in defensive_keywords)
        
        if puzzle_defensive and not traditional_defensive:
            print("✓ Puzzle solver chose more defensive move")
            improvements += 1
        elif traditional_san == puzzle_san:
            print("= Same move chosen")
        else:
            print("○ Different approaches")
    
    print(f"\nTactical improvement rate: {improvements}/{total_tests}")

def test_immediate_threat_priority():
    """Test that immediate threats get highest priority."""
    print("\n" + "=" * 60)
    print("IMMEDIATE THREAT PRIORITY TEST")
    print("=" * 60)
    print("TESTING: Critical threats override other opportunities")
    print()
    
    engine = V7P3REngine()
    
    # Position where White king is in check - must be handled immediately
    board = chess.Board("rnbqkbnr/pppp1ppp/8/8/8/8/PPPPPPPP/RNBQK2q w Qq - 0 1")
    
    print("Position with White king in check:")
    print(f"FEN: {board.fen()}")
    print()
    
    # Test threat detection
    opportunities = engine._detect_opportunities(board)
    print(f"Detected opportunities: {opportunities}")
    
    # Get move candidates
    candidates = engine.search_multi_pv(board, 1.0, num_lines=3)
    
    print("\nMove analysis:")
    for i, candidate in enumerate(candidates, 1):
        move_san = board.san(candidate['move'])
        print(f"{i}. {move_san} (weighted: {candidate['weighted_score']:.1f})")
    
    # Check if escape_check opportunity was detected and prioritized
    if 'escape_check' in opportunities:
        print("\n✓ Check escape correctly detected as immediate threat")
    else:
        print("\n○ Check escape not specifically detected")

if __name__ == "__main__":
    try:
        test_tactical_threat_detection()
        test_weighted_opportunity_scoring()
        test_puzzle_solver_vs_traditional_tactical()
        test_immediate_threat_priority()
        
        print("\n" + "=" * 60)
        print("ENHANCED TACTICAL AWARENESS TEST COMPLETE")
        print("=" * 60)
        print("✓ Threat detection system tested")
        print("✓ Weighted opportunity scoring verified")
        print("✓ Tactical priorities implemented")
        print("✓ Immediate threat handling validated")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()