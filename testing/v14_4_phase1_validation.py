#!/usr/bin/env python3

"""
V7P3R v14.4 Phase 1 Validation Test

Test the enhanced tactical move ordering improvements against
the diagnostic puzzle set to measure improvement.

Expected improvement: +3-5% accuracy on 1500+ puzzles
"""

import json
import sys
import os
import chess
from datetime import datetime

# Add v7p3r engine path
v7p3r_path = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\src"
sys.path.append(v7p3r_path)

def test_enhanced_move_ordering():
    """Test the enhanced move ordering on some sample positions"""
    
    print("V7P3R v14.4 Phase 1 - Enhanced Tactical Move Ordering Test")
    print("=" * 60)
    
    # Initialize engine
    try:
        from v7p3r import V7P3REngine
        engine = V7P3REngine()
        print("✅ V7P3R v14.4 engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return False
    
    # Test positions from diagnostic analysis where move ordering could help
    test_positions = [
        {
            'name': 'Pin Tactic - Puzzle 2TKdX Position 1',
            'fen': '3r1k2/2R4R/7p/p3B3/6rb/1P2n3/2P4P/1K6 w - - 2 31',
            'expected': 'h7h8',
            'description': 'Mate threat should be prioritized'
        },
        {
            'name': 'Fork Tactic - Multi-attack',
            'fen': 'r6k/pp4pp/3pQ1n1/2pP1rq1/2N5/8/PPP2PPP/4RRK1 b - - 3 18',
            'expected': 'g6f4',
            'description': 'Multi-piece attack (fork) should be high priority'
        },
        {
            'name': 'High-value Capture',
            'fen': '3r1rk1/pp1q1p1R/2n2p1Q/4pb2/4B3/2P2NP1/PP3PP1/R3K3 b Q - 8 20',
            'expected': 'f5e4',
            'description': 'High-value capture should be well-ordered'
        }
    ]
    
    print(f"Testing move ordering on {len(test_positions)} positions...")
    print()
    
    success_count = 0
    
    for i, test_pos in enumerate(test_positions, 1):
        print(f"Test {i}: {test_pos['name']}")
        print(f"FEN: {test_pos['fen']}")
        print(f"Expected: {test_pos['expected']}")
        print(f"Description: {test_pos['description']}")
        
        try:
            board = chess.Board(test_pos['fen'])
            
            # Get legal moves and their ordering
            legal_moves = list(board.legal_moves)
            ordered_moves = engine._order_moves_advanced(board, legal_moves, 4)
            
            print(f"Legal moves: {len(legal_moves)}")
            print(f"First 5 ordered moves: {[str(m) for m in ordered_moves[:5]]}")
            
            # Check if expected move is in top 5
            expected_move = chess.Move.from_uci(test_pos['expected'])
            if expected_move in ordered_moves[:5]:
                position = ordered_moves.index(expected_move) + 1
                print(f"✅ Expected move '{test_pos['expected']}' found at position {position}")
                success_count += 1
            else:
                try:
                    position = ordered_moves.index(expected_move) + 1
                    print(f"❌ Expected move '{test_pos['expected']}' found at position {position} (not in top 5)")
                except ValueError:
                    print(f"❌ Expected move '{test_pos['expected']}' not found in ordered moves")
            
            # Test engine search with new ordering
            print("Testing search with enhanced ordering...")
            start_time = datetime.now()
            engine_move = engine.search(board, time_limit=2.0)
            search_time = (datetime.now() - start_time).total_seconds()
            
            if engine_move:
                correct = (str(engine_move) == test_pos['expected'])
                print(f"Engine move: {engine_move} ({'✅ Correct' if correct else '❌ Incorrect'}) - {search_time:.2f}s")
            else:
                print(f"❌ Engine returned no move")
            
        except Exception as e:
            print(f"❌ Error testing position: {e}")
        
        print("-" * 40)
        print()
    
    ordering_accuracy = (success_count / len(test_positions)) * 100
    print(f"Move Ordering Results: {success_count}/{len(test_positions)} expected moves in top 5 ({ordering_accuracy:.1f}%)")
    
    return success_count == len(test_positions)

def run_quick_diagnostic_sample():
    """Run a quick sample of the diagnostic puzzle set to test improvements"""
    
    print("Quick Diagnostic Sample Test")
    print("=" * 40)
    
    # Load a few puzzles from each category for quick testing
    sample_puzzles = [
        # Pin puzzle
        {
            'id': '05Cqj',
            'fen': '5qk1/r2N2p1/2p4p/p2bB3/3P2Q1/7P/PP4P1/6K1 b - - 2 33',
            'moves': 'f8f7 d7f6 f7f6 e5f6',
            'rating': 1200,
            'themes': 'crushing endgame pin short'
        },
        # Fork puzzle  
        {
            'id': '08XLs',
            'fen': 'r3k2r/2p3pp/p1p5/2b1N3/4b3/2P4P/PP4P1/RN1R3K b kq - 0 19',
            'moves': 'a8d8 d1d8 e8d8 e5f7 d8e8 f7h8',
            'rating': 1200,
            'themes': 'attraction crushing exposedKing fork long middlegame'
        },
        # Mate puzzle
        {
            'id': '0AtVa',
            'fen': '3r4/8/3R2pk/7p/4p2K/4P1PP/pr6/R7 w - - 0 44',
            'moves': 'd6d8 g6g5',
            'rating': 1200,
            'themes': 'endgame master mate mateIn1 oneMove rookEndgame'
        }
    ]
    
    try:
        from v7p3r import V7P3REngine
        engine = V7P3REngine()
        print("✅ V7P3R v14.4 engine ready for testing")
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return False
    
    print(f"\nTesting {len(sample_puzzles)} sample puzzles...")
    
    total_positions = 0
    correct_positions = 0
    
    for puzzle in sample_puzzles:
        print(f"\nPuzzle {puzzle['id']} (Rating: {puzzle['rating']})")
        print(f"Themes: {puzzle['themes']}")
        
        sequence = puzzle['moves'].split()
        if len(sequence) < 2:
            continue
            
        board = chess.Board(puzzle['fen'])
        
        # Test first position in sequence
        opponent_move = sequence[0]
        expected_move = sequence[1]
        
        try:
            # Apply opponent move
            board.push(chess.Move.from_uci(opponent_move))
            
            # Test engine
            engine_move = engine.search(board, time_limit=3.0)
            total_positions += 1
            
            if engine_move and str(engine_move) == expected_move:
                correct_positions += 1
                print(f"✅ Correct: {engine_move}")
            else:
                print(f"❌ Expected: {expected_move}, Got: {engine_move}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    accuracy = (correct_positions / total_positions * 100) if total_positions > 0 else 0
    print(f"\nQuick Sample Results: {correct_positions}/{total_positions} ({accuracy:.1f}%)")
    
    return accuracy >= 80  # Expect at least 80% on this simple sample

if __name__ == "__main__":
    print("V7P3R v14.4 Phase 1 Validation")
    print("Enhanced Tactical Move Ordering")
    print("=" * 50)
    
    # Test 1: Move ordering effectiveness
    print("Test 1: Move Ordering Priority Test")
    ordering_success = test_enhanced_move_ordering()
    
    print("\n" + "=" * 50)
    
    # Test 2: Quick diagnostic sample
    print("Test 2: Quick Diagnostic Sample")
    sample_success = run_quick_diagnostic_sample()
    
    print("\n" + "=" * 50)
    print("PHASE 1 VALIDATION SUMMARY")
    print("=" * 50)
    
    if ordering_success and sample_success:
        print("✅ Phase 1 improvements validated successfully!")
        print("- Enhanced move ordering working correctly")
        print("- Sample puzzle performance maintained/improved")
        print("\n🎯 Ready to proceed with full diagnostic test")
    else:
        print("❌ Phase 1 validation issues detected")
        if not ordering_success:
            print("- Move ordering needs adjustment")
        if not sample_success:
            print("- Sample puzzle performance regression")
        print("\n⚠️  Recommend fixing issues before full test")
    
    print(f"\nNext step: Run full diagnostic test against 1500+ puzzle set")
    print(f"Target: Improve from 77.5% to 82.5%+ accuracy")