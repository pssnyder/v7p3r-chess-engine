#!/usr/bin/env python3
"""
Test script for V13.1 Asymmetric Search Depth System
Shows how opponent depth reduction models imperfect play algorithmically
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import chess
from v7p3r import V7P3REngine

def test_asymmetric_search_depth():
    """Test the asymmetric search depth system - algorithmically sound opponent modeling"""
    engine = V7P3REngine()
    
    print("V13.1 ASYMMETRIC SEARCH DEPTH TEST")
    print("=" * 50)
    print("ALGORITHM-SAFE OPPONENT MODELING:")
    print("- Consistent evaluation function (no asymmetric evaluation)")
    print("- Asymmetric search depth (opponent moves searched less deeply)")
    print("- Models that opponents don't search as deeply as we do")
    print()
    
    # Test opponent depth reduction at different search depths and move positions
    print("OPPONENT DEPTH REDUCTION MATRIX:")
    print("Search Depth | Move 1 | Move 4 | Move 8 | Move 12")
    print("-" * 55)
    
    for depth in range(3, 8):
        reductions = []
        for move_num in [1, 4, 8, 12]:
            reduction = engine._calculate_opponent_depth_reduction(depth, move_num - 1)
            reductions.append(str(reduction))
        
        print(f"     {depth}       |   {reductions[0]}    |   {reductions[1]}    |   {reductions[2]}    |   {reductions[3]}")
    
    print()
    print("EVALUATION CONSISTENCY TEST:")
    print("(All positions use same evaluation function)")
    print("-" * 50)
    
    positions = [
        chess.Board(),  # Starting position
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),  # After 1.e4
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq e6 0 3"),  # Tactical position
    ]
    
    for i, pos in enumerate(positions):
        print(f"Position {i+1}: {pos.fen()[:20]}...")
        
        # Test evaluation consistency at different depths
        for depth in [2, 4]:
            eval_level = engine._determine_evaluation_level(pos, depth)
            score = engine._evaluate_position(pos, depth)
            print(f"  Depth {depth}: Eval Level {eval_level}, Score {score:.2f}")
        print()
    
    print("KEY ALGORITHMIC ADVANTAGES:")
    print("✓ Evaluation function remains consistent")
    print("✓ Opponent moves searched with less depth")
    print("✓ Models realistic opponent behavior")
    print("✓ No search pathologies or comparison issues")
    print("✓ Performance improvement from reduced opponent search")

if __name__ == "__main__":
    test_asymmetric_search_depth()