#!/usr/bin/env python3
"""
Test V7P3R Adaptive Move Ordering System
Validates intelligent move prioritization based on posture
"""

import sys
import os
import chess

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r_posture_assessment import V7P3RPostureAssessment
from v7p3r_adaptive_move_ordering import V7P3RAdaptiveMoveOrdering

def test_adaptive_move_ordering():
    """Test adaptive move ordering on different position types"""
    
    print("V7P3R Adaptive Move Ordering Test")
    print("=" * 50)
    
    # Initialize systems
    posture_assessor = V7P3RPostureAssessment()
    move_orderer = V7P3RAdaptiveMoveOrdering(posture_assessor)
    
    # Test positions
    test_positions = [
        {
            'name': 'Volatile Position (Emergency Defense)',
            'fen': 'r4rk1/ppp2ppp/8/1q5n/3p4/3P1P1P/PPPQ1P2/R4RK1 w - - 0 16',
            'description': 'White under severe pressure, needs defensive moves'
        },
        {
            'name': 'Balanced Middlegame',
            'fen': 'rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3',
            'description': 'Normal development and captures should be prioritized'
        },
        {
            'name': 'Tactical Opportunity',
            'fen': 'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5',
            'description': 'Should prioritize tactical and attacking moves'
        }
    ]
    
    for i, test_case in enumerate(test_positions):
        print(f"\nTest {i+1}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"FEN: {test_case['fen']}")
        
        board = chess.Board(test_case['fen'])
        
        # Assess posture
        volatility, posture = posture_assessor.assess_position_posture(board)
        print(f"Position Assessment: {volatility.value} volatility, {posture.value} posture")
        
        # Get ordered moves
        legal_moves = list(board.legal_moves)
        scored_moves = move_orderer.order_moves(board, legal_moves)
        
        print(f"Total legal moves: {len(legal_moves)}")
        print("Top 10 moves with classifications:")
        
        for j, (move, score) in enumerate(scored_moves[:10]):
            classification = move_orderer.get_move_classification(board, move)
            print(f"  {j+1:2d}. {move} - {classification} (score: {score})")
        
        # Show move type distribution in top 10
        move_types = {}
        for move, score in scored_moves[:10]:
            move_type, _ = move_orderer._classify_move(board, move, posture, volatility)
            move_types[move_type] = move_types.get(move_type, 0) + 1
        
        print("Move type distribution in top 10:")
        for move_type, count in sorted(move_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {move_type}: {count}")

def test_defensive_prioritization():
    """Test that defensive moves are prioritized correctly in emergency situations"""
    
    print(f"\n" + "=" * 50)
    print("DEFENSIVE PRIORITIZATION TEST")
    print("=" * 50)
    
    # Use the user's volatile position
    posture_assessor = V7P3RPostureAssessment()
    move_orderer = V7P3RAdaptiveMoveOrdering(posture_assessor)
    
    board = chess.Board('r4rk1/ppp2ppp/8/1q5n/3p4/3P1P1P/PPPQ1P2/R4RK1 w - - 0 16')
    
    volatility, posture = posture_assessor.assess_position_posture(board)
    print(f"Position: {volatility.value} volatility, {posture.value} posture")
    
    # Get all moves and classify them
    legal_moves = list(board.legal_moves)
    scored_moves = move_orderer.order_moves(board, legal_moves)
    
    # Count defensive vs offensive moves in top 10
    defensive_types = ['escape_moves', 'blocking_moves', 'defensive_captures', 'defensive_moves']
    offensive_types = ['tactical_moves', 'attacking_moves', 'good_captures']
    
    defensive_count = 0
    offensive_count = 0
    
    print("\nTop 10 moves analysis:")
    for i, (move, score) in enumerate(scored_moves[:10]):
        move_type, _ = move_orderer._classify_move(board, move, posture, volatility)
        
        if move_type in defensive_types:
            defensive_count += 1
            type_class = "DEFENSIVE"
        elif move_type in offensive_types:
            offensive_count += 1
            type_class = "OFFENSIVE"
        else:
            type_class = "NEUTRAL"
        
        print(f"  {i+1:2d}. {move} - {move_type} [{type_class}] (score: {score})")
    
    print(f"\nSummary:")
    print(f"Defensive moves in top 10: {defensive_count}")
    print(f"Offensive moves in top 10: {offensive_count}")
    print(f"Others: {10 - defensive_count - offensive_count}")
    
    # In an emergency position, defensive moves should dominate
    defensive_ratio = defensive_count / 10
    print(f"Defensive ratio: {defensive_ratio:.1%}")
    
    success = defensive_ratio >= 0.6  # At least 60% defensive in emergency
    print(f"Test result: {'PASS' if success else 'FAIL'}")
    
    return success

if __name__ == "__main__":
    print("Starting V7P3R Adaptive Move Ordering Tests...")
    
    try:
        # Run main tests
        test_adaptive_move_ordering()
        
        # Run specific defensive test
        defensive_test_pass = test_defensive_prioritization()
        
        print(f"\n" + "=" * 50)
        print("FINAL TEST RESULTS:")
        print(f"Defensive Prioritization Test: {'PASS' if defensive_test_pass else 'FAIL'}")
        
        if defensive_test_pass:
            print("\n✓ Adaptive move ordering system is working correctly!")
        else:
            print("\n✗ Adaptive move ordering needs improvements.")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()