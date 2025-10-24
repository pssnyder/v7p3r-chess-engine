#!/usr/bin/env python3
"""
Quick V13.x Tactical Test
Test specific tactical positions that were failing
"""

import chess
import time
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from v7p3r import V7P3REngine
    
    def test_enhanced_tactical_scoring():
        """Test enhanced tactical scoring on key positions"""
        print("🎯 V13.x ENHANCED TACTICAL SCORING TEST")
        print("="*55)
        
        engine = V7P3REngine()
        
        # Test the positions that were failing
        test_positions = [
            ("Back Rank Mate", "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", "Ra8+"),
            ("Fork Attack", "rnbqk2r/pppp1ppp/5n2/4p3/1b2P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 4 4", "Nxe5"),
            ("Pin Breaking", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4", "Bxf7+")
        ]
        
        improvements = 0
        
        for name, fen, expected_move in test_positions:
            board = chess.Board(fen)
            print(f"\n📍 {name}")
            print(f"Expected: {expected_move}")
            
            try:
                # Quick search
                start_time = time.time()
                best_move = engine.search(board, time_limit=1.0, depth=3)
                search_time = time.time() - start_time
                
                actual_move = board.san(best_move) if best_move else "None"
                
                # Check if improved
                if expected_move in actual_move or actual_move in expected_move:
                    print(f"✅ FOUND: {actual_move}")
                    improvements += 1
                else:
                    print(f"❌ Got: {actual_move}")
                
                print(f"   Time: {search_time:.3f}s")
                
                # Test move ordering on this position
                legal_moves = list(board.legal_moves)
                ordered_moves = engine._order_moves_advanced(board, legal_moves, 3)
                waiting_moves = engine.get_waiting_moves()
                
                print(f"   Pruning: {len(waiting_moves)/len(legal_moves)*100:.1f}%")
                print(f"   Top moves: {[board.san(m) for m in ordered_moves[:3]]}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n🎯 ENHANCEMENT RESULTS:")
        print(f"Improved positions: {improvements}/3")
        print(f"Tactical accuracy: {improvements/3*100:.1f}%")
        
        if improvements >= 2:
            print(f"✅ Significant improvement in tactical scoring!")
        elif improvements >= 1:
            print(f"⚠️  Some improvement, needs more work")
        else:
            print(f"❌ No improvement, needs debugging")
        
        return improvements >= 1
    
    if __name__ == "__main__":
        success = test_enhanced_tactical_scoring()
        if success:
            print("\n🚀 V13.x tactical enhancements working!")
        else:
            print("\n🔧 V13.x tactical enhancements need more work")
            
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()