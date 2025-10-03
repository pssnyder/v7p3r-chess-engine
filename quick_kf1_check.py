#!/usr/bin/env python3
"""
Quick Kf1 Check - V7P3R v12.4
==============================
Simple test to verify Kf1 avoidance in problematic positions
"""

import chess
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_kf1_simple():
    """Simple test of Kf1 avoidance"""
    
    print("V7P3R v12.4 - Kf1 Avoidance Check")
    print("=" * 40)
    
    # Import here to avoid path issues
    from v7p3r import V7P3REngine
    
    # Test the worst case - position where v12.2 definitely chose Kf1
    test_fen = "r1bqk1nr/p2ppppp/1p6/8/8/8/PPPP1PPP/RNBQK1NR w KQkq - 0 6"
    print(f"Testing FEN: {test_fen}")
    print("(This is where v12.2 chose Kf1 instead of castling)")
    
    board = chess.Board(test_fen)
    engine = V7P3REngine()
    
    print("\nPosition:")
    print(board)
    
    # Get engine's choice
    result = engine.search(board, depth=3)
    best_move = result.get('move')
    
    print(f"\nV12.4 chooses: {best_move}")
    
    # Check if it's Kf1
    kf1_move = chess.Move.from_uci("e1f1")
    if best_move == kf1_move:
        print("❌ STILL CHOOSING Kf1 - Enhancement failed!")
        return False
    else:
        print("✅ AVOIDED Kf1 - Enhancement working!")
        
        # Show what it chose instead
        if best_move:
            print(f"Chose {best_move} instead")
            
            # Check if castling is available
            if board.has_kingside_castling_rights(chess.WHITE):
                castling_move = chess.Move.from_uci("e1g1")
                if best_move == castling_move:
                    print("🏰 EXCELLENT: Engine chose to castle!")
                else:
                    print(f"📋 Good: Engine chose development move {best_move}")
        
        return True

if __name__ == "__main__":
    success = test_kf1_simple()
    if success:
        print("\n🎉 V12.4 Enhancement: SUCCESS!")
    else:
        print("\n⚠️  V12.4 Enhancement: NEEDS MORE WORK!")