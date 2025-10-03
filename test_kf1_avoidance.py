#!/usr/bin/env python3
"""
V7P3R v12.4 Kf1 Avoidance Test
===============================
Specifically tests if the engine avoids the problematic Kf1 moves
that v12.2 was making in critical opening positions.
"""

import chess
import chess.engine
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine

def test_kf1_avoidance():
    """Test if engine avoids Kf1 moves in specific positions"""
    
    print("=" * 60)
    print("V7P3R v12.4 Kf1 AVOIDANCE TEST")
    print("=" * 60)
    print("Testing if V12.4 avoids the problematic Kf1 moves that v12.2 made")
    print()
    
    # Test positions where v12.2 made Kf1
    test_cases = [
        {
            "name": "Engine Battle Round 4",
            "fen": "r1bqk1nr/p2ppppp/1p6/8/8/8/PPPP1PPP/RNBQK1NR w KQkq - 0 6",
            "description": "After 5...bxa6, should NOT play Kf1",
            "v12_2_move": "e1f1"
        },
        {
            "name": "Engine Battle Round 5", 
            "fen": "rnbqkb1r/2pppp1p/p5pn/8/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 0 5",
            "description": "After 4...a6, should NOT play Kf1",
            "v12_2_move": "e1f1"
        },
        {
            "name": "Engine Battle Round 8",
            "fen": "rnbqkb1r/pppp1ppp/4pn2/8/2B1P3/2N5/PPPP1PPP/R1BQK1NR w KQkq - 1 4", 
            "description": "After 3...Nf6, should NOT play Kf1",
            "v12_2_move": "e1f1"
        },
        {
            "name": "Regression Battle",
            "fen": "rnbqk1nr/ppppbppp/4p3/8/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 1 5",
            "description": "After 4...f6, should NOT play Kf1", 
            "v12_2_move": "e1f1"
        }
    ]
    
    results = {
        "kf1_avoided": 0,
        "kf1_chosen": 0,
        "total_tests": len(test_cases)
    }
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"=== Test Case {i}: {test_case['name']} ===")
        print(f"Description: {test_case['description']}")
        print(f"FEN: {test_case['fen']}")
        print(f"V12.2 played: {test_case['v12_2_move']} (problematic Kf1)")
        
        # Set up position
        board = chess.Board(test_case['fen'])
        print("Position:")
        print(board)
        print()
        
        # Test engine choice
        engine = V7P3REngine()
        
        try:
            # Get top moves with evaluation
            result = engine.search(board, depth=4)
            best_move = result.get('move')
            
            print(f"V12.4 best move: {best_move}")
            
            # Check if Kf1 was chosen
            kf1_move = chess.Move.from_uci("e1f1")
            
            if best_move == kf1_move:
                print("❌ FAIL: Engine chose Kf1!")
                results["kf1_chosen"] += 1
                test_result = "FAIL"
            else:
                print("✅ PASS: Engine avoided Kf1!")
                results["kf1_avoided"] += 1
                test_result = "PASS"
            
            # Also check if Kf1 is among legal moves and get its evaluation
            legal_moves = list(board.legal_moves)
            if kf1_move in legal_moves:
                print(f"Kf1 is legal but not chosen (good!)")
                
                # Test what engine thinks of Kf1 specifically
                board_after_kf1 = board.copy()
                board_after_kf1.push(kf1_move)
                kf1_eval = engine.bitboard_evaluator.evaluate_bitboard(board_after_kf1, chess.WHITE)
                
                # Test what engine thinks of chosen move
                board_after_best = board.copy()
                board_after_best.push(best_move)
                best_eval = engine.bitboard_evaluator.evaluate_bitboard(board_after_best, chess.WHITE)
                
                print(f"Kf1 evaluation: {kf1_eval:.1f}")
                print(f"Chosen move evaluation: {best_eval:.1f}")
                print(f"Difference: {best_eval - kf1_eval:.1f} (positive = better choice)")
            else:
                print("Kf1 is not legal in this position")
            
            print(f"Result: {test_result}")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results["kf1_chosen"] += 1
            
        print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Kf1 avoided: {results['kf1_avoided']}")
    print(f"❌ Kf1 chosen: {results['kf1_chosen']}")
    print(f"Total tests: {results['total_tests']}")
    print()
    
    success_rate = (results['kf1_avoided'] / results['total_tests']) * 100
    print(f"Kf1 Avoidance Rate: {success_rate:.1f}%")
    
    if success_rate >= 75:
        print("🎉 EXCELLENT: V12.4 successfully avoids most Kf1 moves!")
    elif success_rate >= 50:
        print("👍 GOOD: V12.4 avoids many Kf1 moves, room for improvement")
    else:
        print("⚠️  NEEDS WORK: V12.4 still choosing too many Kf1 moves")
    
    return results

if __name__ == "__main__":
    test_kf1_avoidance()