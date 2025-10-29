#!/usr/bin/env python3

"""
V7P3R v14.4 Phase 2 Validation Test

Test the enhanced pin detection improvements to ensure:
1. Pin detection is working correctly
2. No regression from Phase 1 performance
3. Improved performance on pin-related puzzles
"""

import json
import sys
import os
import chess
from datetime import datetime

# Add v7p3r engine path
v7p3r_path = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\src"
sys.path.append(v7p3r_path)

def test_pin_detection():
    """Test the pin detection on known pin positions"""
    
    print("V7P3R v14.4 Phase 2 - Enhanced Pin Detection Test")
    print("=" * 60)
    
    # Initialize engine
    try:
        from v7p3r import V7P3REngine
        engine = V7P3REngine()
        print("✅ V7P3R v14.4 Phase 2 engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return False
    
    # Test positions with known pins
    pin_test_positions = [
        {
            'name': 'Simple Rook Pin',
            'fen': '3k4/8/8/8/3n4/8/8/3R4 w - - 0 1',
            'description': 'White rook on d1 pins black knight on d4 to black king on d8'
        },
        {
            'name': 'Bishop Pin', 
            'fen': '4k3/8/6n1/8/8/8/2B5/4K3 w - - 0 1',
            'description': 'White bishop on c2 pins black knight on g6 to black king on e8'
        },
        {
            'name': 'Queen Pin',
            'fen': 'r3k2r/8/8/3b4/8/8/8/3QK3 w - - 0 1',
            'description': 'White queen on d1 pins black bishop on d5 to black king on e8'
        }
    ]
    
    print(f"Testing pin detection on {len(pin_test_positions)} positions...")
    print()
    
    success_count = 0
    
    for i, test_pos in enumerate(pin_test_positions, 1):
        print(f"Test {i}: {test_pos['name']}")
        print(f"FEN: {test_pos['fen']}")
        print(f"Description: {test_pos['description']}")
        
        try:
            board = chess.Board(test_pos['fen'])
            
            # Test pin detection
            pin_data = engine._detect_pins(board)
            
            print(f"Pin Detection Results:")
            print(f"  White pins created: {len(pin_data['white_pins'])}")
            print(f"  Black pins created: {len(pin_data['black_pins'])}")
            print(f"  White pieces pinned: {len(pin_data['white_pinned'])}")
            print(f"  Black pieces pinned: {len(pin_data['black_pinned'])}")
            print(f"  White pin score: {pin_data['pin_score_white']:.1f}")
            print(f"  Black pin score: {pin_data['pin_score_black']:.1f}")
            
            # Test evaluation with pins
            evaluation = engine._evaluate_position(board)
            print(f"  Position evaluation: {evaluation:.1f}")
            
            # Count as success if any pins detected (for positions that should have pins)
            total_pins = len(pin_data['white_pins']) + len(pin_data['black_pins'])
            if total_pins > 0:
                print(f"✅ Pin detection working - found {total_pins} pins")
                success_count += 1
            else:
                print(f"⚠️  No pins detected - may need tuning")
            
        except Exception as e:
            print(f"❌ Error testing position: {e}")
        
        print("-" * 40)
        print()
    
    detection_accuracy = (success_count / len(pin_test_positions)) * 100
    print(f"Pin Detection Results: {success_count}/{len(pin_test_positions)} positions with pins detected ({detection_accuracy:.1f}%)")
    
    return success_count >= 2  # At least 2/3 should detect pins

def test_phase1_regression():
    """Ensure Phase 1 improvements are still working"""
    
    print("Phase 1 Regression Test")
    print("=" * 40)
    
    try:
        from v7p3r import V7P3REngine
        engine = V7P3REngine()
        print("✅ Engine ready for regression test")
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return False
    
    # Use same test positions from Phase 1
    phase1_positions = [
        {
            'name': 'Queen Threat Fork',
            'fen': 'r6k/pp4pp/3pQ1n1/2pP1rq1/2N5/8/PPP2PPP/4RRK1 b - - 3 18',
            'expected': 'g6f4',
            'description': 'Should still prioritize Queen threat'
        },
        {
            'name': 'Multi-attack Capture',
            'fen': '3r1rk1/pp1q1p1R/2n2p1Q/4pb2/4B3/2P2NP1/PP3PP1/R3K3 b Q - 8 20',
            'expected': 'f5e4',
            'description': 'Should still prioritize multi-attack capture'
        }
    ]
    
    print(f"Testing Phase 1 regression on {len(phase1_positions)} key positions...")
    print()
    
    success_count = 0
    
    for i, test_pos in enumerate(phase1_positions, 1):
        print(f"Regression Test {i}: {test_pos['name']}")
        print(f"Expected: {test_pos['expected']}")
        
        try:
            board = chess.Board(test_pos['fen'])
            
            # Test move ordering
            legal_moves = list(board.legal_moves)
            ordered_moves = engine._order_moves_advanced(board, legal_moves, 4)
            
            # Check if expected move is in top 5
            expected_move = chess.Move.from_uci(test_pos['expected'])
            if expected_move in ordered_moves[:5]:
                position = ordered_moves.index(expected_move) + 1
                print(f"✅ Expected move found at position {position}")
                success_count += 1
            else:
                try:
                    position = ordered_moves.index(expected_move) + 1
                    print(f"❌ Expected move at position {position} (not in top 5)")
                except ValueError:
                    print(f"❌ Expected move not found in ordering")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    regression_success = (success_count / len(phase1_positions)) * 100
    print(f"Phase 1 Regression Results: {success_count}/{len(phase1_positions)} key moves still in top 5 ({regression_success:.1f}%)")
    
    return success_count == len(phase1_positions)

def test_sample_puzzles():
    """Quick test on sample puzzles to ensure no performance regression"""
    
    print("Sample Puzzle Performance Test")
    print("=" * 40)
    
    # Same sample puzzles from Phase 1
    sample_puzzles = [
        {
            'id': '05Cqj',
            'fen': '5qk1/r2N2p1/2p4p/p2bB3/3P2Q1/7P/PP4P1/6K1 b - - 2 33',
            'moves': 'f8f7 d7f6 f7f6 e5f6',
            'expected_first': 'd7f6'
        },
        {
            'id': '08XLs',
            'fen': 'r3k2r/2p3pp/p1p5/2b1N3/4b3/2P4P/PP4P1/RN1R3K b kq - 0 19',
            'moves': 'a8d8 d1d8 e8d8 e5f7 d8e8 f7h8',
            'expected_first': 'd1d8'
        },
        {
            'id': '0AtVa',
            'fen': '3r4/8/3R2pk/7p/4p2K/4P1PP/pr6/R7 w - - 0 44',
            'moves': 'd6d8 g6g5',
            'expected_first': 'g6g5'
        }
    ]
    
    try:
        from v7p3r import V7P3REngine
        engine = V7P3REngine()
        print("✅ Engine ready for sample test")
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return False
    
    print(f"\nTesting {len(sample_puzzles)} sample puzzles...")
    
    total_positions = 0
    correct_positions = 0
    
    for puzzle in sample_puzzles:
        print(f"\nPuzzle {puzzle['id']}:")
        
        sequence = puzzle['moves'].split()
        if len(sequence) < 2:
            continue
            
        board = chess.Board(puzzle['fen'])
        
        # Test first position in sequence
        opponent_move = sequence[0]
        expected_move = puzzle['expected_first']
        
        try:
            # Apply opponent move
            board.push(chess.Move.from_uci(opponent_move))
            
            # Test engine
            engine_move = engine.search(board, time_limit=2.0)
            total_positions += 1
            
            if engine_move and str(engine_move) == expected_move:
                correct_positions += 1
                print(f"✅ Correct: {engine_move}")
            else:
                print(f"❌ Expected: {expected_move}, Got: {engine_move}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    accuracy = (correct_positions / total_positions * 100) if total_positions > 0 else 0
    print(f"\nSample Results: {correct_positions}/{total_positions} ({accuracy:.1f}%)")
    
    return accuracy >= 65  # Allow slight regression, focus on no major breakdown

if __name__ == "__main__":
    print("V7P3R v14.4 Phase 2 Validation")
    print("Enhanced Pin Detection")
    print("=" * 50)
    
    # Test 1: Pin detection functionality
    print("Test 1: Pin Detection Functionality")
    pin_detection_success = test_pin_detection()
    
    print("\n" + "=" * 50)
    
    # Test 2: Phase 1 regression test
    print("Test 2: Phase 1 Regression Check")
    regression_success = test_phase1_regression()
    
    print("\n" + "=" * 50)
    
    # Test 3: Sample puzzle performance
    print("Test 3: Sample Puzzle Performance")
    sample_success = test_sample_puzzles()
    
    print("\n" + "=" * 50)
    print("PHASE 2 VALIDATION SUMMARY")
    print("=" * 50)
    
    if pin_detection_success and regression_success and sample_success:
        print("✅ Phase 2 improvements validated successfully!")
        print("- Enhanced pin detection working correctly")
        print("- Phase 1 improvements maintained")
        print("- Sample puzzle performance maintained")
        print("\n🎯 Ready to test against full 1500+ diagnostic set")
    else:
        print("❌ Phase 2 validation issues detected")
        if not pin_detection_success:
            print("- Pin detection needs adjustment")
        if not regression_success:
            print("- Phase 1 regression detected")
        if not sample_success:
            print("- Sample puzzle performance regression")
        print("\n⚠️  Recommend fixing issues before proceeding")
    
    print(f"\nNext step: Run tactical_diagnostics_1500.py to measure improvement")
    print(f"Expected: Improve from 79.2% (Phase 1) to 82%+ accuracy")