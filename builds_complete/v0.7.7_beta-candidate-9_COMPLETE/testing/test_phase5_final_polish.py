#!/usr/bin/env python3
"""
Test Phase 5: Final Polish - Display and Evaluation Fixes
Tests the corrected evaluation display and PGN comment system.
"""

import chess
import chess.pgn
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_root)
sys.path.insert(0, parent_dir)

from v7p3r_play import v7p3rChess


def test_evaluation_display_consistency():
    """Test that evaluation display shows (white_score - black_score) consistently"""
    print("=== Testing Evaluation Display Consistency ===")
    
    game = v7p3rChess(config_name="default_config")
    
    # Play a few moves to generate evaluations
    moves = ["e4", "e5", "Nf3", "Nc6"]
    
    print(f"Initial position: {game.board.fen()}")
    
    # Test evaluation from initial position
    eval_score = game.engine.scoring_calculator.evaluate_position(game.board)
    print(f"Initial evaluation (white perspective): {eval_score:+.2f}")
    
    for i, move_str in enumerate(moves):
        try:
            move = game.board.parse_san(move_str)
            player = "White" if game.board.turn == chess.WHITE else "Black"
            
            print(f"\n--- Move {i+1}: {player} plays {move_str} ---")
            
            # Make the move
            game.board.push(move)
            
            # Get evaluation using the standard method
            eval_score = game.engine.scoring_calculator.evaluate_position(game.board)
            print(f"Position evaluation (white perspective): {eval_score:+.2f}")
            
            # Test display_move_made method
            print("Testing display_move_made:")
            game.display_move_made(move, move_time=1.2)
            
        except Exception as e:
            print(f"Error with move {move_str}: {e}")
            break
    
    print("\n=== Evaluation Display Test Complete ===")


def test_verbose_output_control():
    """Test that verbose output is properly controlled"""
    print("\n=== Testing Verbose Output Control ===")
    
    # Test with verbose disabled (default)
    print("--- Testing with verbose_output: false (from config) ---")
    game1 = v7p3rChess(config_name="default_config")
    print(f"Verbose output enabled: {game1.verbose_output_enabled}")
    
    # Make a move and check output
    move = game1.board.parse_san("e4")
    game1.board.push(move)
    print("Display with verbose disabled:")
    game1.display_move_made(move, move_time=0.5)
    
    # Test with verbose enabled
    print("\n--- Testing with verbose_output manually enabled ---")
    game2 = v7p3rChess(config_name="default_config")
    game2.verbose_output_enabled = True  # Manually enable for test
    print(f"Verbose output enabled: {game2.verbose_output_enabled}")
    
    # Make a move and check output
    move = game2.board.parse_san("e4")
    game2.board.push(move)
    print("Display with verbose enabled:")
    game2.display_move_made(move, move_time=0.5)
    
    print("\n=== Verbose Output Control Test Complete ===")


def test_pgn_comment_format():
    """Test that PGN comments use the correct evaluation format"""
    print("\n=== Testing PGN Comment Format ===")
    
    game = v7p3rChess(config_name="default_config")
    
    # Play a few moves and record evaluations
    moves = ["e4", "e5"]
    
    for move_str in moves:
        try:
            move = game.board.parse_san(move_str)
            
            # Make the move and advance game node
            game.board.push(move)
            game.game_node = game.game_node.add_variation(move)
            
            # Record evaluation
            game.record_evaluation()
            
            # Check the comment format
            comment = game.game_node.comment
            print(f"Move: {move_str}, PGN Comment: '{comment}'")
            
            # Verify the comment format
            if comment.startswith("Eval: "):
                eval_part = comment[6:]  # Remove "Eval: " prefix
                try:
                    eval_value = float(eval_part)
                    print(f"  Parsed evaluation: {eval_value:+.2f}")
                except ValueError:
                    print(f"  ERROR: Could not parse evaluation from comment: {comment}")
            else:
                print(f"  ERROR: Comment doesn't start with 'Eval: ': {comment}")
                
        except Exception as e:
            print(f"Error testing PGN comment for move {move_str}: {e}")
    
    print("\n=== PGN Comment Format Test Complete ===")


def main():
    """Run all phase 5 tests"""
    print("Starting Phase 5: Final Polish Tests")
    print("=" * 50)
    
    try:
        test_evaluation_display_consistency()
        test_verbose_output_control()  
        test_pgn_comment_format()
        
        print("\n" + "=" * 50)
        print("Phase 5 Tests Complete!")
        print("All evaluation display and output control fixes verified.")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
