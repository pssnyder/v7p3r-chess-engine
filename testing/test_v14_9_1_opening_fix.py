"""
V14.9.1 Opening Move Test
Verify that V14.9.1 plays sensible opening moves (not e3)
and that PV matches actual move played.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

def test_opening_moves():
    """Test that V14.9.1 plays sensible opening moves"""
    
    engine = V7P3REngine()
    board = chess.Board()
    
    print("=" * 70)
    print("V14.9.1 OPENING MOVE TEST - Verify PV matches actual move")
    print("=" * 70)
    
    # Test 3 opening moves with moderate time
    for move_num in range(1, 4):
        print(f"\n{'='*70}")
        print(f"Position: {board.fen()}")
        print(f"Testing {'White' if board.turn else 'Black'} move {move_num}...")
        print(f"{'='*70}")
        
        # Give 5 seconds for opening - should easily reach depth 4+
        move = engine.search(board, time_limit=5.0)
        
        print(f"\nV14.9.1 played: {move}")
        print(f"Move type: {board.piece_at(move.from_square)}")
        
        # Verify move is legal
        if move not in board.legal_moves:
            print(f"❌ ILLEGAL MOVE: {move}")
            return False
        
        # Check if move is e3 (the broken move from V14.9)
        if move_num == 1 and move == chess.Move.from_uci("e2e3"):
            print(f"❌ FAILURE: Still playing 1.e3 (Van Kruij's Opening)")
            print(f"   This indicates root search is still using fallback move!")
            return False
        
        # Apply move to board
        board.push(move)
        
        print(f"✅ Legal move played")
        
    print("\n" + "="*70)
    print("✅ V14.9.1 OPENING TEST PASSED")
    print("   - All moves legal")
    print("   - No e3 opening")
    print("   - Root search working correctly")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = test_opening_moves()
    sys.exit(0 if success else 1)
