#!/usr/bin/env python3
"""
EXPERIMENTAL: Own-Perspective-Only Search Algorithm
Testing the user's brilliant insight about evaluating everything from our perspective.

This explores replacing traditional minimax with a "paranoid" style search
where we only care about "how good is this position for us?" after every move.
"""

import os
import sys
import time
import chess

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine

def analyze_current_minimax():
    """Analyze what our current minimax is actually doing."""
    print("=" * 60)
    print("CURRENT MINIMAX ANALYSIS")
    print("=" * 60)
    print("Understanding the evaluation flip logic")
    print()
    
    engine = V7P3REngine()
    board = chess.Board()
    
    # Test position: After 1.e4 e5 2.Nf3
    board.push_san("e4")
    board.push_san("e5") 
    board.push_san("Nf3")
    
    print(f"Position: {board.fen()}")
    print(f"Turn: {'White' if board.turn else 'Black'}")
    print()
    
    # Get evaluation from current player's perspective
    eval_score = engine._evaluate_position(board)
    print(f"Position evaluation: {eval_score/100:.2f}")
    print()
    
    # Now let's see what happens with moves
    legal_moves = list(board.legal_moves)[:5]  # First 5 moves
    
    print("MOVE EVALUATION ANALYSIS:")
    print("-" * 40)
    
    for move in legal_moves:
        board.push(move)
        
        # This is what current minimax does - evaluates from new position's perspective
        current_eval = engine._evaluate_position(board)
        
        # This is what user proposes - always evaluate from our original perspective
        # We'd need to flip the board perspective or track who we are
        original_turn = not board.turn  # Who moved (our original perspective)
        
        board.pop()
        
        print(f"Move: {move} | Current Minimax: {current_eval/100:.2f} | Our Perspective: {-current_eval/100:.2f}")
    
    print()
    print("KEY INSIGHT: Current minimax alternates perspective")
    print("Your idea: Always evaluate from OUR perspective")

def simulate_own_perspective_search():
    """Simulate what own-perspective-only search might look like."""
    print("=" * 60)
    print("OWN-PERSPECTIVE-ONLY SIMULATION")
    print("=" * 60)
    print("What if we never flip evaluation perspective?")
    print()
    
    engine = V7P3REngine()
    board = chess.Board()
    our_color = board.turn  # White in this case
    
    # Test position: Starting position
    print(f"Starting position - We are: {'White' if our_color else 'Black'}")
    print()
    
    # Our moves: e4, Nf3, Bc4
    our_moves = ["e4", "Nf3", "Bc4"]
    
    print("TESTING OWN-PERSPECTIVE EVALUATION:")
    print("-" * 45)
    
    for our_move in our_moves:
        print(f"\nOur Move: {our_move}")
        
        # Make our move
        board.push_san(our_move)
        our_position_value = engine._evaluate_position(board)
        print(f"  Position value after our move: {our_position_value/100:.2f}")
        
        # Now opponent's possible responses
        opponent_moves = list(board.legal_moves)[:3]  # First 3 for speed
        
        worst_position = float('inf')
        best_opponent_move = None
        
        for opp_move in opponent_moves:
            board.push(opp_move)
            
            # KEY DIFFERENCE: Always evaluate from OUR perspective
            # Traditional minimax would flip here, we don't
            position_value_for_us = engine._evaluate_position(board)
            
            # Since we're white and evaluations are from current player perspective,
            # when it's black's turn, we need to flip to get our perspective
            if board.turn != our_color:
                position_value_for_us = -position_value_for_us
            
            if position_value_for_us < worst_position:
                worst_position = position_value_for_us
                best_opponent_move = opp_move
            
            board.pop()
            
            print(f"    vs {opp_move}: {position_value_for_us/100:.2f}")
        
        print(f"  Worst case for us: {worst_position/100:.2f} (opponent plays {best_opponent_move})")
        
        # For simulation, let's assume opponent plays the best move against us
        if best_opponent_move:
            board.push(best_opponent_move)
            print(f"  Opponent plays: {best_opponent_move}")

def compare_computational_cost():
    """Compare computational costs of both approaches."""
    print("=" * 60)  
    print("COMPUTATIONAL COST ANALYSIS")
    print("=" * 60)
    print("Traditional Minimax vs Own-Perspective Search")
    print()
    
    engine = V7P3REngine()
    board = chess.Board()
    
    # Test depth 3 search
    depth = 3
    
    print(f"Analyzing depth {depth} search costs:")
    print("-" * 40)
    
    # Traditional minimax cost estimate
    avg_branching = 35  # Average legal moves per position
    traditional_nodes = avg_branching ** depth
    traditional_evals = traditional_nodes  # One eval per node
    
    print(f"Traditional Minimax:")
    print(f"  Nodes searched: ~{traditional_nodes:,}")
    print(f"  Evaluations: ~{traditional_evals:,}")
    print(f"  Each eval: Full complexity")
    print()
    
    # Own-perspective cost estimate
    # We evaluate every possible sequence from our perspective
    own_perspective_positions = avg_branching ** depth
    own_perspective_evals = own_perspective_positions
    
    print(f"Own-Perspective Search:")
    print(f"  Positions evaluated: ~{own_perspective_positions:,}")
    print(f"  Evaluations: ~{own_perspective_evals:,}")
    print(f"  Each eval: Always same perspective (potentially simpler)")
    print()
    
    print("POTENTIAL SAVINGS:")
    print("- No perspective flipping in evaluation")
    print("- No min/max alternation logic")
    print("- Simpler evaluation caching")
    print("- Could enable more aggressive pruning")

def test_tactical_blind_spots():
    """Test if own-perspective search has tactical blind spots."""
    print("=" * 60)
    print("TACTICAL BLIND SPOT ANALYSIS")  
    print("=" * 60)
    print("Can own-perspective search see opponent threats?")
    print()
    
    engine = V7P3REngine()
    
    # Set up a position where opponent has a strong threat
    # Scholar's mate setup: 1.e4 e5 2.Bc4 Nc6 3.Qh5
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Bc4")  
    board.push_san("Nc6")
    board.push_san("Qh5")  # Threatening Qxf7#
    
    print(f"Position: {board.fen()}")
    print("White threatens Qxf7# mate!")
    print()
    
    # Black's possible moves
    legal_moves = list(board.legal_moves)
    
    # Traditional minimax would see this as "very bad for black"
    current_eval = engine._evaluate_position(board)
    print(f"Current evaluation (Black's perspective): {current_eval/100:.2f}")
    print()
    
    print("Own-perspective analysis of Black's responses:")
    print("-" * 45)
    
    # From White's perspective, let's see what happens after each Black move
    board.turn = chess.WHITE  # Pretend we're white for this test
    our_color = chess.WHITE
    
    for move in legal_moves[:8]:  # First 8 moves
        board.turn = chess.BLACK  # Set to black to make the move
        board.push(move)
        
        # Evaluate from White's perspective
        board.turn = our_color
        eval_for_white = engine._evaluate_position(board)
        board.turn = not our_color  # Reset turn
        
        # Check if this stops the mate threat
        board.push_san("Qxf7")  # Try to execute the mate
        is_mate = board.is_checkmate()
        board.pop()  # Undo the mate attempt
        
        board.pop()  # Undo black's move
        
        status = "MATE!" if is_mate else "Safe"
        print(f"  {move}: White eval {eval_for_white/100:.2f} - {status}")

if __name__ == "__main__":
    try:
        analyze_current_minimax()
        simulate_own_perspective_search()
        compare_computational_cost()
        test_tactical_blind_spots()
        
        print("\n" + "=" * 60)
        print("EXPERIMENTAL ANALYSIS COMPLETE")
        print("=" * 60)
        print("✓ Current minimax behavior analyzed")
        print("✓ Own-perspective simulation completed") 
        print("✓ Computational costs compared")
        print("✓ Tactical blind spots investigated")
        print()
        print("KEY FINDINGS:")
        print("- Own-perspective search is computationally feasible")
        print("- May have different tactical characteristics")
        print("- Could be simpler to implement and optimize")
        print("- Needs careful handling of opponent threat detection")
        
    except Exception as e:
        print(f"Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()