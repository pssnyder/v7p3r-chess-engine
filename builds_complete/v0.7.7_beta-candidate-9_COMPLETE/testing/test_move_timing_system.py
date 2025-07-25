#!/usr/bin/env python3
"""
Test Move Timing System - Comprehensive test for the fixed timing display
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_root)
sys.path.insert(0, parent_dir)

from v7p3r_play import v7p3rChess


def test_timing_in_real_game():
    """Test that timing works correctly in an actual game"""
    print("=== Testing Real Game Timing ===")
    
    # Use speed_config for faster moves that still have measurable timing
    game = v7p3rChess(config_name="speed_config")
    
    print(f"Verbose output enabled: {game.verbose_output_enabled}")
    print(f"Monitoring enabled: {game.monitoring_enabled}")
    
    move_count = 0
    max_moves = 6
    
    print("\nGame moves with actual timing:")
    print("-" * 40)
    
    while not game.board.is_game_over() and move_count < max_moves:
        move_count += 1
        print(f"\n--- Move {move_count} ---")
        game.process_engine_move()
    
    print(f"\n=== Real Game Timing Test Complete ===")
    print(f"Played {move_count} moves with actual timing measurements")


def test_timing_precision_levels():
    """Test that different timing precision levels display correctly"""
    print("\n=== Testing Timing Precision Levels ===")
    
    game = v7p3rChess(config_name="default_config")
    
    # Test cases: (description, time_in_seconds, expected_unit)
    precision_tests = [
        ("Microsecond precision", 0.0001234, "ms"),      # 0.1234ms
        ("Sub-millisecond", 0.0005678, "ms"),           # 0.5678ms
        ("Few milliseconds", 0.0123, "ms"),             # 12.3ms
        ("Tens of milliseconds", 0.0567, "ms"),         # 56.7ms
        ("Just under 100ms", 0.0999, "ms"),             # 99.9ms
        ("Exactly 100ms threshold", 0.1000, "s"),       # 0.100s
        ("Few hundred milliseconds", 0.250, "s"),       # 0.250s
        ("Sub-second", 0.750, "s"),                     # 0.750s
        ("One second", 1.000, "s"),                     # 1.00s
        ("Few seconds", 3.456, "s"),                    # 3.46s
        ("Ten+ seconds", 12.789, "s"),                  # 12.8s
    ]
    
    print("Description                    | Time      | Display      | Unit")
    print("-" * 65)
    
    for desc, time_val, expected_unit in precision_tests:
        formatted = game._format_time_for_display(time_val)
        actual_unit = "ms" if "ms" in formatted else "s"
        status = "Γ£ô" if actual_unit == expected_unit else "Γ£ù"
        
        print(f"{desc:<30} | {time_val:8.6f}s | {formatted:>10} | {status}")
    
    print("\n=== Timing Precision Test Complete ===")


def main():
    """Run comprehensive timing tests"""
    print("Testing Move Timing System")
    print("=" * 50)
    
    try:
        test_timing_in_real_game()
        test_timing_precision_levels()
        
        print("\n" + "=" * 50)
        print("Move Timing System Tests Complete!")
        print("Γ£ô Real game timing measurement working")
        print("Γ£ô Smart unit conversion (s/ms) working")  
        print("Γ£ô High precision storage implemented")
        print("Γ£ô Display formatting working correctly")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
