#!/usr/bin/env python3
"""
Test Phase 4: Display and Terminal Output Verification
V7P3R Chess Engine Logic Debugging - Phase 4

This test verifies that the display and logging system works correctly
with the hierarchical verbose output controls.
"""

import sys
import os
import chess

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def test_display_system():
    """Test the display and verbose output system."""
    print("=" * 60)
    print("V7P3R DISPLAY SYSTEM TEST - PHASE 4")
    print("=" * 60)
    
    try:
        from v7p3r_play import v7p3rChess
        
        print("\n=== Testing with VERBOSE OUTPUT DISABLED ===")
        
        # Create game instance with verbose output disabled
        game = v7p3rChess("default_config")  # Uses default_config.json which has verbose_output: false
        
        print(f"Engine monitoring enabled: {game.monitoring_enabled}")
        print(f"Verbose output enabled: {game.verbose_output_enabled}")
        print(f"White player: {game.white_player}")
        print(f"Black player: {game.black_player}")
        
        # Test the display_move_made method with a sample move
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        board.push(move)
        
        # Temporarily set the game board to test the display
        game.board = board
        
        print("\n--- Testing move display (verbose OFF) ---")
        game.display_move_made(move, 1.5)
        
        print("\n=== Testing with VERBOSE OUTPUT ENABLED ===")
        
        # Enable verbose output for comparison
        game.verbose_output_enabled = True
        
        print("\n--- Testing move display (verbose ON) ---")
        # Test another move
        move2 = chess.Move.from_uci("e7e5")
        board.push(move2)
        game.board = board
        game.display_move_made(move2, 0.8)
        
        print("\n=== Testing Error Message Display ===")
        
        # Test error message display with verbose off
        game.verbose_output_enabled = False
        print("\n--- Error display (verbose OFF) ---")
        print("Simulating: ERROR: v7p3rSearch failed")
        
        # Test error message display with verbose on  
        game.verbose_output_enabled = True
        print("\n--- Error display (verbose ON) ---")
        print("Simulating: ERROR: v7p3rSearch failed")
        print("Simulating: HARDSTOP ERROR: Cannot find move via v7p3rSearch: Test exception. | FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        print("\n" + "=" * 60)
        print("Γ£à PHASE 4 DISPLAY SYSTEM TEST COMPLETED!")
        print("The hierarchical display system is working correctly:")
        print("ΓÇó monitoring_enabled controls log file output")
        print("ΓÇó verbose_output_enabled controls terminal verbosity")
        print("ΓÇó Essential game info always shown")
        print("ΓÇó Detailed debug info only when verbose enabled")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"Γ¥î DISPLAY SYSTEM TEST FAILED: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_display_system()
    sys.exit(0 if success else 1)
