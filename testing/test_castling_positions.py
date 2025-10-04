#!/usr/bin/env python3
"""
V7P3R v12.4 Castling Test Suite
Tests positions where v12.2 moved king manually instead of castling
"""

import chess
import sys
import os
import time

# Add src directory to path
sys.path.append('src')
from v7p3r import V7P3REngine

def test_position(name, fen, expected_bad_move, description):
    """Test a specific position and return results"""
    print(f"\n=== {name} ===")
    print(f"Description: {description}")
    print(f"FEN: {fen}")
    print(f"Testing V12.4 enhanced castling behavior")
    
    # Set up position
    board = chess.Board(fen)
    engine = V7P3REngine()
    
    print("Position:")
    print(board)
    print()
    
    # Test current v12.4 behavior
    start_time = time.time()
    
    # Search for best move
    best_move = engine.search(board, time_limit=5.0, depth=5)
    
    search_time = time.time() - start_time
    
    print(f"V12.4 move: {best_move}")
    print(f"Search time: {search_time:.1f}s")
    
    # Check if it's castling
    is_castling = board.is_castling(best_move)
    is_king_move = board.piece_at(best_move.from_square).piece_type == chess.KING
    is_manual_king = is_king_move and not is_castling
    
    # Determine result
    if is_castling:
        result = "✅ PASS - Chose castling!"
        status = "PASS"
    elif is_manual_king:
        result = "❌ FAIL - Still moving king manually"
        status = "FAIL"
    else:
        result = "🟡 ACCEPTABLE - Chose other developing move"
        status = "ACCEPTABLE"
    
    print(f"Result: {result}")
    
    return {
        'name': name,
        'fen': fen,
        'v12_2_move': expected_bad_move,
        'v12_4_move': str(best_move),
        'is_castling': is_castling,
        'is_manual_king': is_manual_king,
        'status': status,
        'search_time': search_time
    }

def main():
    """Run all castling tests"""
    print("=" * 60)
    print("V7P3R v12.4 ENHANCED CASTLING TEST SUITE")
    print("=" * 60)
    print("Testing V12.4's enhanced castling behavior in critical positions")
    
    # Test positions extracted from game analysis
    test_cases = [
        {
            'name': 'Test Case 1: Critical Opening Position',
            'fen': 'r1bqk1nr/p2ppppp/1p6/8/8/8/PPPP1PPP/RNBQK1NR w KQkq - 0 6',
            'bad_move': 'Kf1',
            'description': 'Should V12.4 castle or choose strong development? (Previously Kf1 was chosen)'
        },
        {
            'name': 'Test Case 2: Development vs Castling', 
            'fen': 'rnbqkb1r/2pppp1p/p5pn/8/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 0 5',
            'bad_move': 'Kf1',
            'description': 'V12.4 should prefer castling or development over manual king moves'
        },
        {
            'name': 'Test Case 3: King Safety Priority',
            'fen': 'rnbqkb1r/pppp1ppp/4pn2/8/2B1P3/2N5/PPPP1PPP/R1BQK1NR w KQkq - 1 4',
            'bad_move': 'Kf1', 
            'description': 'V12.4 should prioritize king safety through castling or strong moves'
        },
        {
            'name': 'Test Case 4: Complex Position',
            'fen': 'rnbqk1nr/ppppbppp/4p3/8/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 1 5',
            'bad_move': 'Kf1',
            'description': 'After 4...f6, should castle instead of manual king move'
        }
    ]
    
    results = []
    
    # Run all tests
    for test_case in test_cases:
        result = test_position(
            test_case['name'],
            test_case['fen'], 
            test_case['bad_move'],
            test_case['description']
        )
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passes = sum(1 for r in results if r['status'] == 'PASS')
    fails = sum(1 for r in results if r['status'] == 'FAIL') 
    acceptable = sum(1 for r in results if r['status'] == 'ACCEPTABLE')
    
    print(f"✅ PASS (Castling): {passes}")
    print(f"🟡 ACCEPTABLE (Other moves): {acceptable}")
    print(f"❌ FAIL (Manual king): {fails}")
    print(f"Total tests: {len(results)}")
    
    print(f"\nSuccess Rate: {(passes + acceptable) / len(results) * 100:.1f}%")
    
    # Detailed results
    print(f"\nDetailed Results:")
    for result in results:
        castling_note = " (CASTLING!)" if result['is_castling'] else ""
        king_note = " (manual king)" if result['is_manual_king'] else ""
        print(f"  {result['name']}: {result['v12_4_move']}{castling_note}{king_note} [{result['status']}]")
    
    # Overall assessment
    if passes >= 3:
        print(f"\n🎉 SUCCESS: V12.4 shows significant improvement in castling preference!")
    elif passes + acceptable >= 3:
        print(f"\n✅ GOOD: V12.4 avoids most manual king moves, some room for improvement") 
    else:
        print(f"\n⚠️  NEEDS WORK: V12.4 still prefers manual king moves over castling")
    
    return results

if __name__ == "__main__":
    results = main()