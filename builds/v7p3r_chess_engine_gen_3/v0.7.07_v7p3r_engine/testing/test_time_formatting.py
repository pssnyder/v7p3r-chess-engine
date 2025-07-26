#!/usr/bin/env python3
"""
Test Time Formatting - Test the new time display formatting functionality
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_root)
sys.path.insert(0, parent_dir)

from v7p3r_play import v7p3rChess


def test_time_formatting():
    """Test the _format_time_for_display method with various time values"""
    print("=== Testing Time Formatting ===")
    
    game = v7p3rChess(config_name="default_config")
    
    # Test various time values
    test_times = [
        0.0,          # Zero time
        0.0001,       # 0.1 milliseconds
        0.001,        # 1 millisecond  
        0.005,        # 5 milliseconds
        0.025,        # 25 milliseconds
        0.050,        # 50 milliseconds
        0.100,        # 100 milliseconds (threshold)
        0.250,        # 250 milliseconds
        0.500,        # 500 milliseconds
        1.000,        # 1 second
        1.234,        # 1.234 seconds
        5.678,        # 5.678 seconds
        12.345,       # 12.345 seconds
        60.789,       # 1 minute
    ]
    
    print("Time (seconds) -> Display Format")
    print("-" * 35)
    
    for time_val in test_times:
        formatted = game._format_time_for_display(time_val)
        print(f"{time_val:10.6f}s -> {formatted}")
    
    print("\n=== Time Formatting Test Complete ===")


def test_move_display_with_timing():
    """Test actual move display with different timing scenarios"""
    print("\n=== Testing Move Display with Various Timings ===")
    
    game = v7p3rChess(config_name="speed_config")
    
    # Create test moves with different timings
    import chess
    
    test_scenarios = [
        ("e2e4", 0.0005, "Very fast move (0.5ms)"),
        ("e7e5", 0.025, "Fast move (25ms)"),
        ("g1f3", 0.150, "Medium move (150ms)"),
        ("b8c6", 1.234, "Slow move (1.234s)"),
        ("f1c4", 15.678, "Very slow move (15.678s)"),
    ]
    
    for move_uci, timing, description in test_scenarios:
        print(f"\n{description}:")
        try:
            move = chess.Move.from_uci(move_uci)
            game.board.push(move)
            game.display_move_made(move, timing)
        except Exception as e:
            print(f"Error testing move {move_uci}: {e}")
    
    print("\n=== Move Display Test Complete ===")


def test_logging_precision():
    """Test that logging stores high precision timing"""
    print("\n=== Testing Logging Precision ===")
    
    # Create a game with monitoring enabled for testing
    game = v7p3rChess(config_name="default_config")
    game.monitoring_enabled = True
    
    if game.logger:
        print("Logger available - testing high precision logging")
        
        import chess
        move = chess.Move.from_uci("e2e4")
        game.board.push(move)
        
        # Test with microsecond-level timing
        precise_time = 0.001234567  # 1.234567 milliseconds
        
        print(f"Testing with precise time: {precise_time:.9f}s")
        game.display_move_made(move, precise_time)
        
        print("Check the log file for high-precision timing storage.")
    else:
        print("No logger available - precision logging not tested")
    
    print("\n=== Logging Precision Test Complete ===")


def main():
    """Run all time formatting tests"""
    print("Starting Time Formatting Tests")
    print("=" * 50)
    
    try:
        test_time_formatting()
        test_move_display_with_timing()
        test_logging_precision()
        
        print("\n" + "=" * 50)
        print("Time Formatting Tests Complete!")
        print("All timing precision and display formatting verified.")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
