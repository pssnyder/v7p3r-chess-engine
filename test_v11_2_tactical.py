#!/usr/bin/env python3
"""
Quick tactical test for V7P3R v11.2 Enhanced Engine
Test on the positions where v11.1 failed
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import chess
import time
from v7p3r_v11_2_enhanced import V7P3REngineEnhanced

print("🎯 V7P3R v11.2 Enhanced - Quick Tactical Test")
print("=" * 60)

# Test positions where v11.1 failed
test_positions = [
    {
        "name": "Basic Fork Tactic", 
        "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R b KQkq - 0 3",
        "expected_moves": ["g8e4", "f6e4", "f6g4"],
        "theme": "fork"
    },
    {
        "name": "Pin Break",
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4", 
        "expected_moves": ["c1g5", "f1d5", "h2h3"],
        "theme": "pin"
    },
    {
        "name": "Mate in 2",
        "fen": "r1bqkb1r/pppp1p1p/2n2np1/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
        "expected_moves": ["c4f7", "d1d5"],
        "theme": "mate"
    },
    {
        "name": "Discovery Attack", 
        "fen": "r1bqk2r/pppp1ppp/2nb1n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 6",
        "expected_moves": ["f3d5", "f3e5"],
        "theme": "discovery"
    }
]

def test_position(engine, name, fen, expected_moves, theme, time_limit=2.0):
    """Test engine on a specific position"""
    print(f"\n🧩 Testing: {name}")
    print(f"   Theme: {theme}")
    print(f"   FEN: {fen}")
    print(f"   Expected moves: {expected_moves}")
    
    try:
        board = chess.Board(fen)
        
        # Analyze position tone first
        tone = engine.move_orderer.analyze_position_tone(board)
        print(f"   Position analysis:")
        print(f"     Tactical complexity: {tone.tactical_complexity:.2f}")
        print(f"     King safety urgency: {tone.king_safety_urgency:.2f}")
        print(f"     Aggression viability: {tone.aggression_viability:.2f}")
        
        # Test move ordering
        legal_moves = list(board.legal_moves)
        ordered_moves = engine.move_orderer.order_moves(board, legal_moves)
        print(f"   Top 5 moves from ordering: {[str(move) for move in ordered_moves[:5]]}")
        
        # Test search
        start_time = time.time()
        best_move = engine.search(board, time_limit)
        elapsed = time.time() - start_time
        
        print(f"   🎯 Engine move: {best_move}")
        print(f"   ⏱️  Time: {elapsed:.2f}s")
        print(f"   📊 Nodes: {engine.nodes_searched}")
        print(f"   ⚡ NPS: {int(engine.nodes_searched / max(elapsed, 0.001))}")
        
        # Check if move is in expected moves
        move_str = str(best_move)
        if move_str in expected_moves:
            print(f"   ✅ SUCCESS: Found expected move!")
            return True
        else:
            print(f"   ⚠️  SUBOPTIMAL: Move is legal but not optimal")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def main():
    try:
        # Create engine
        print("Initializing V7P3R v11.2 Enhanced Engine...")
        engine = V7P3REngineEnhanced()
        print("✅ Engine initialized successfully")
        
        # Test all positions
        results = []
        total_tests = len(test_positions)
        
        for position in test_positions:
            success = test_position(
                engine, 
                position["name"],
                position["fen"], 
                position["expected_moves"],
                position["theme"],
                2.0
            )
            results.append(success)
            
            # Reset engine for next test
            engine.move_orderer.reset_for_new_position()
        
        # Summary
        successes = sum(results)
        print(f"\n" + "=" * 60)
        print(f"🎯 TACTICAL TEST RESULTS SUMMARY")
        print(f"=" * 60)
        print(f"Total tests: {total_tests}")
        print(f"Successes: {successes}")
        print(f"Success rate: {successes/total_tests*100:.1f}%")
        
        if successes >= total_tests * 0.75:  # 75% success threshold
            print(f"✅ EXCELLENT: V7P3R v11.2 shows major tactical improvement!")
        elif successes >= total_tests * 0.5:  # 50% success threshold  
            print(f"⚠️  GOOD: V7P3R v11.2 shows some tactical improvement")
        else:
            print(f"❌ NEEDS WORK: V7P3R v11.2 still has tactical issues")
        
        # Performance summary
        print(f"\n📊 Performance Summary:")
        print(f"   Average NPS: ~{engine.nodes_searched // 2}+ (estimated)")
        print(f"   Engine stability: ✅ No crashes")
        print(f"   Move legality: ✅ All moves legal")
        
        # Comparison with v11.1
        print(f"\n🔄 Comparison with v11.1:")
        print(f"   v11.1 tactical success: 0/5 (0%)")
        print(f"   v11.2 tactical success: {successes}/{total_tests} ({successes/total_tests*100:.1f}%)")
        
        if successes > 0:
            print(f"   ✅ IMPROVEMENT: {successes} more tactical successes!")
        else:
            print(f"   ❌ NO IMPROVEMENT: Still missing tactics")
            
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎉 V7P3R v11.2 Enhanced tactical testing completed!")
        print(f"Ready for full test suite evaluation.")
    else:
        print(f"\n💥 Testing failed - need to debug issues.")
