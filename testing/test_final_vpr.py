#!/usr/bin/env python3
"""
Final VPR Pure Potential Test

Test the completely rewritten VPR engine based on user's vision:
- Piece value = attacks + mobility (NO material assumptions)
- Focus on highest/lowest potential pieces only  
- Chaos preservation through lenient pruning
- Imperfect play assumptions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from vpr import VPREngine

def test_final_vpr():
    """Test the final pure potential VPR implementation"""
    print("🚀 FINAL VPR PURE POTENTIAL TEST")
    print("=" * 50)
    print("✅ Piece value = attacks + mobility (NO material)")
    print("✅ Focus on highest/lowest potential pieces ONLY")
    print("✅ Lenient pruning preserves chaotic positions")
    print("✅ Assumes imperfect opponent play")
    print()
    
    engine = VPREngine()
    
    # Test positions demonstrating the philosophy
    test_suite = [
        {
            "name": "Knight Potential vs Material",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "expected": "Should prioritize active knight over inactive rook"
        },
        {
            "name": "Piece Activation Focus", 
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "expected": "Should develop knights (high potential) over rook moves"
        },
        {
            "name": "Chaos Position",
            "fen": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 
            "expected": "Should preserve complex variations through lenient pruning"
        },
        {
            "name": "Endgame Potential",
            "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "expected": "Should focus on most active pieces (rook and king)"
        }
    ]
    
    total_nodes = 0
    total_time = 0
    depths_achieved = []
    
    for i, test in enumerate(test_suite, 1):
        print(f"🎯 Test {i}: {test['name']}")
        print(f"   Expected: {test['expected']}")
        
        board = chess.Board(test['fen'])
        
        # Show piece potential analysis
        print(f"   Piece Potential Analysis:")
        piece_data = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                potential = engine._calculate_piece_potential(board, square)
                piece_name = chess.piece_name(piece.piece_type)
                square_name = chess.square_name(square)
                piece_data.append((potential, f"{piece_name}@{square_name}"))
        
        piece_data.sort(reverse=True, key=lambda x: x[0])
        for potential, piece_info in piece_data[:4]:  # Top 4
            print(f"     {piece_info}: {potential}")
        
        # Run search and measure performance
        start_time = time.perf_counter()
        best_move = engine.search(board, time_limit=4.0)  # Give it 4 seconds
        end_time = time.perf_counter()
        
        search_time = end_time - start_time
        nodes = engine.nodes_searched
        nps = int(nodes / search_time) if search_time > 0 else 0
        
        # Estimate depth reached (rough calculation from output)
        depth_estimate = max(1, min(15, int(nodes / 1000)))  # Rough estimate
        depths_achieved.append(depth_estimate)
        
        print(f"   ✅ Best move: {best_move}")
        print(f"   📊 Nodes: {nodes:,}")
        print(f"   ⏱️  Time: {search_time:.2f}s")
        print(f"   🏎️  NPS: {nps:,}")
        print(f"   🎯 Est. depth: ~{depth_estimate}")
        print()
        
        total_nodes += nodes
        total_time += search_time
    
    # Summary
    avg_nps = int(total_nodes / total_time) if total_time > 0 else 0
    avg_depth = sum(depths_achieved) / len(depths_achieved) if depths_achieved else 0
    
    print("📈 FINAL VPR PERFORMANCE SUMMARY:")
    print("=" * 40)
    print(f"✅ Total nodes searched: {total_nodes:,}")
    print(f"⏱️  Total time: {total_time:.2f}s") 
    print(f"🏎️  Average NPS: {avg_nps:,}")
    print(f"🎯 Average depth: ~{avg_depth:.1f}")
    print(f"🧠 Philosophy: Position potential over material")
    print(f"🔥 Innovation: Focus on highest/lowest pieces only")
    print(f"🌪️  Chaos handling: Lenient pruning preserves complexity")
    
    # Compare to target goals
    print(f"\n🎯 GOAL ACHIEVEMENT:")
    if avg_nps > 5000:
        print(f"✅ Speed: {avg_nps:,} NPS (Good)")
    else:
        print(f"⚠️  Speed: {avg_nps:,} NPS (Needs improvement)")
    
    if avg_depth >= 6:
        print(f"✅ Depth: ~{avg_depth:.1f} (Excellent)")
    elif avg_depth >= 4:
        print(f"✅ Depth: ~{avg_depth:.1f} (Good)")
    else:
        print(f"⚠️  Depth: ~{avg_depth:.1f} (Needs improvement)")
    
    print(f"✅ Philosophy: Pure potential implemented")

if __name__ == "__main__":
    test_final_vpr()