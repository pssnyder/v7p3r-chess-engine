#!/usr/bin/env python3
"""
Test V13.x Move Ordering Integration
Quick test to verify the new system works in the engine
"""

import chess
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from v7p3r import V7P3REngine
    
    def test_v13x_integration():
        """Test V13.x move ordering integration in actual engine"""
        print("🚀 TESTING V13.x MOVE ORDERING INTEGRATION")
        print("="*60)
        
        # Initialize engine
        engine = V7P3REngine()
        
        # Test positions
        test_positions = [
            ("Opening", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("Middlegame", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"),
            ("Tactical", "r2qkb1r/pp2nppp/3p1n2/2pP4/2P1P3/2N2N2/PP3PPP/R1BQKB1R w KQq - 0 6")
        ]
        
        total_original = 0
        total_critical = 0
        
        for name, fen in test_positions:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            
            print(f"\n📍 TESTING: {name}")
            print(f"FEN: {fen}")
            print(f"Total legal moves: {len(legal_moves)}")
            
            # Test the new V13.x move ordering
            try:
                critical_moves = engine._order_moves_advanced(board, legal_moves, 0)
                waiting_moves = engine.get_waiting_moves()
                
                print(f"✅ V13.x System Working!")
                print(f"   Critical moves: {len(critical_moves)}")
                print(f"   Waiting moves: {len(waiting_moves)}")
                print(f"   Pruning rate: {len(waiting_moves)/len(legal_moves)*100:.1f}%")
                
                # Show top critical moves
                print(f"   🎯 Top Critical Moves:")
                for i, move in enumerate(critical_moves[:5], 1):
                    print(f"      {i}. {board.san(move)}")
                
                total_original += len(legal_moves)
                total_critical += len(critical_moves)
                
            except Exception as e:
                print(f"❌ Error testing V13.x: {e}")
                return False
        
        # Overall statistics
        if total_original > 0:
            pruning_rate = (total_original - total_critical) / total_original * 100
            speedup = total_original / total_critical if total_critical > 0 else 1
            
            print(f"\n🎯 OVERALL V13.x PERFORMANCE:")
            print(f"Total legal moves: {total_original}")
            print(f"Critical moves selected: {total_critical}")
            print(f"Average pruning rate: {pruning_rate:.1f}%")
            print(f"Expected speedup: {speedup:.1f}x")
            
            if hasattr(engine, 'v13x_stats'):
                stats = engine.v13x_stats
                print(f"\n📊 ENGINE STATISTICS:")
                print(f"V13.x stats collected: {stats}")
        
        print(f"\n✅ V13.x INTEGRATION TEST COMPLETE!")
        return True
    
    if __name__ == "__main__":
        success = test_v13x_integration()
        if success:
            print("\n🚀 V13.x move ordering successfully integrated!")
            print("Ready for Phase 2: Waiting move support and search integration")
        else:
            print("\n❌ V13.x integration needs debugging")
            
except ImportError as e:
    print(f"❌ Could not import V7P3REngine: {e}")
    print("Make sure the engine is properly configured")
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()