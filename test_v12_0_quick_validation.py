#!/usr/bin/env python3
"""
V12.0 Quick Validation Test
===========================

Quick test to validate v12.0 performance vs the v12.1 we just reverted from.
This will help us understand if the regression was in v12.0→v12.1 changes
or if v12.0 itself has issues vs the original v10.8.
"""

import sys
import os
import chess
import time
import json
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from v7p3r import V7P3REngine
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the v7p3r-chess-engine directory")
    sys.exit(1)

def test_tactical_positions():
    """Test a few key tactical positions to see if v12.0 handles them correctly"""
    
    print("V12.0 Tactical Position Test")
    print("=============================")
    
    engine = V7P3REngine()
    
    # Test positions with known good moves
    test_positions = [
        {
            'name': 'Opening Development',
            'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'expected_type': 'development',
            'time_limit': 2.0
        },
        {
            'name': 'Simple Mate in 1',
            'fen': '6k1/5ppp/8/8/8/8/8/7Q w - - 0 1',
            'expected_type': 'mate',
            'time_limit': 1.0
        },
        {
            'name': 'Center Control',
            'fen': 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
            'expected_type': 'center_response',
            'time_limit': 2.0
        }
    ]
    
    results = []
    
    for i, pos in enumerate(test_positions, 1):
        print(f"\nTest {i}: {pos['name']}")
        print(f"Position: {pos['fen']}")
        
        board = chess.Board(pos['fen'])
        start_time = time.time()
        
        move = engine.search(board, time_limit=pos['time_limit'])
        
        search_time = time.time() - start_time
        
        print(f"Move chosen: {move}")
        print(f"Search time: {search_time:.3f}s")
        print(f"Nodes searched: {engine.nodes_searched:,}")
        
        # Basic move validity check
        is_valid = move in board.legal_moves
        
        result = {
            'position': pos['name'],
            'fen': pos['fen'],
            'move': str(move),
            'valid': is_valid,
            'search_time': search_time,
            'nodes': engine.nodes_searched,
            'nps': engine.nodes_searched / search_time if search_time > 0 else 0
        }
        
        results.append(result)
        
        print(f"Status: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    return results

def test_search_consistency():
    """Test that the search gives consistent results"""
    
    print("\nV12.0 Search Consistency Test")
    print("==============================")
    
    engine = V7P3REngine()
    board = chess.Board()
    
    print("Testing same position 3 times with same time limit...")
    
    moves = []
    for i in range(3):
        move = engine.search(board, time_limit=1.0)
        moves.append(str(move))
        print(f"  Run {i+1}: {move}")
    
    # Check if all moves are the same
    consistency = len(set(moves)) == 1
    print(f"Consistency: {'✅ Consistent' if consistency else '❌ Inconsistent'}")
    
    return consistency, moves

def main():
    """Run all v12.0 validation tests"""
    
    print("V7P3R v12.0 Quick Validation Suite")
    print("===================================")
    print()
    
    # Test 1: Tactical positions
    tactical_results = test_tactical_positions()
    
    # Test 2: Search consistency
    consistency, consistency_moves = test_search_consistency()
    
    # Summary
    print("\n" + "="*50)
    print("V12.0 VALIDATION SUMMARY")
    print("="*50)
    
    valid_moves = sum(1 for r in tactical_results if r['valid'])
    avg_nps = sum(r['nps'] for r in tactical_results) / len(tactical_results)
    
    print(f"Valid moves: {valid_moves}/{len(tactical_results)}")
    print(f"Average NPS: {avg_nps:.0f}")
    print(f"Search consistency: {'✅' if consistency else '❌'}")
    
    # Save detailed results
    results = {
        'version': 'V7P3R v12.0',
        'timestamp': datetime.now().isoformat(),
        'tactical_results': tactical_results,
        'consistency_test': {
            'consistent': consistency,
            'moves': consistency_moves
        },
        'summary': {
            'valid_moves': valid_moves,
            'total_moves': len(tactical_results),
            'average_nps': avg_nps,
            'consistent': consistency
        }
    }
    
    filename = f"v12_0_quick_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {filename}")
    
    if valid_moves == len(tactical_results) and consistency:
        print("\n🎉 V12.0 validation: ALL TESTS PASSED")
        return True
    else:
        print("\n⚠️  V12.0 validation: Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)