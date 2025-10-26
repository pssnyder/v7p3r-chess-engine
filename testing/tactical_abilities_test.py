#!/usr/bin/env python3

"""
V7P3R Quick Puzzle Test

Test V7P3R against 10-20 specific puzzle types:
- Pins
- Forks  
- Checkmates
"""

import sys
import os

# Add the engine tester path for universal puzzle analyzer
engine_tester_path = r"S:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester"
sys.path.append(os.path.join(engine_tester_path, 'engine_utilities'))

from universal_puzzle_analyzer import UniversalPuzzleAnalyzer

def test_v7p3r_tactical_abilities():
    """Test V7P3R's tactical abilities with specific puzzle types"""
    
    # Path to current V7P3R engine
    v7p3r_path = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\v7p3r.py"
    
    print("V7P3R Tactical Abilities Test")
    print("=" * 50)
    print(f"Testing engine: {v7p3r_path}")
    
    if not os.path.exists(v7p3r_path):
        print(f"❌ Engine not found: {v7p3r_path}")
        return
    
    # Create .bat wrapper for the python engine
    bat_path = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\v7p3r_test.bat"
    
    # Create the batch file
    with open(bat_path, 'w') as f:
        f.write(f'@echo off\n')
        f.write(f'cd /d "s:\\Maker Stuff\\Programming\\Chess Engines\\V7P3R Chess Engine\\v7p3r-chess-engine"\n')
        f.write(f'python src\\v7p3r.py\n')
    
    print(f"Created test wrapper: {bat_path}")
    
    try:
        # Initialize analyzer
        analyzer = UniversalPuzzleAnalyzer(bat_path)
        print(f"✅ Analyzer initialized for: {analyzer.engine_name}")
        
        # Test 1: Pin puzzles (10 puzzles)
        print("\n" + "="*50)
        print("TEST 1: PIN PUZZLES (Rating 1200-1800)")
        print("="*50)
        
        pin_results = analyzer.run_analysis(
            num_puzzles=10,
            rating_min=1200,
            rating_max=1800,
            suggested_time=15.0,
            themes_filter=["pin"]
        )
        
        # Generate and print report
        pin_report = analyzer.generate_report(pin_results)
        analyzer.print_report(pin_report)
        
        # Test 2: Fork puzzles (10 puzzles)
        print("\n" + "="*50)
        print("TEST 2: FORK PUZZLES (Rating 1200-1800)")
        print("="*50)
        
        fork_results = analyzer.run_analysis(
            num_puzzles=10,
            rating_min=1200,
            rating_max=1800,
            suggested_time=15.0,
            themes_filter=["fork"]
        )
        
        fork_report = analyzer.generate_report(fork_results)
        analyzer.print_report(fork_report)
        
        # Test 3: Mate puzzles (10 puzzles)
        print("\n" + "="*50)
        print("TEST 3: MATE PUZZLES (Rating 1200-1600)")
        print("="*50)
        
        mate_results = analyzer.run_analysis(
            num_puzzles=10,
            rating_min=1200,
            rating_max=1600,
            suggested_time=20.0,
            themes_filter=["mate"]
        )
        
        mate_report = analyzer.generate_report(mate_results)
        analyzer.print_report(mate_report)
        
        # Summary
        print("\n" + "="*60)
        print("TACTICAL ABILITIES SUMMARY")
        print("="*60)
        
        pin_accuracy = pin_report.get('sequence_metrics', {}).get('avg_weighted_accuracy', 0)
        fork_accuracy = fork_report.get('sequence_metrics', {}).get('avg_weighted_accuracy', 0)
        mate_accuracy = mate_report.get('sequence_metrics', {}).get('avg_weighted_accuracy', 0)
        
        print(f"Pin Puzzles:  {pin_accuracy:.1f}% accuracy")
        print(f"Fork Puzzles: {fork_accuracy:.1f}% accuracy")
        print(f"Mate Puzzles: {mate_accuracy:.1f}% accuracy")
        
        overall_tactical = (pin_accuracy + fork_accuracy + mate_accuracy) / 3
        print(f"\nOverall Tactical Strength: {overall_tactical:.1f}%")
        
        if overall_tactical >= 75:
            print("🎯 EXCELLENT tactical abilities!")
        elif overall_tactical >= 60:
            print("🎯 GOOD tactical abilities")
        elif overall_tactical >= 40:
            print("⚠️  FAIR tactical abilities - needs improvement")
        else:
            print("❌ WEAK tactical abilities - major improvement needed")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up the temp bat file
        if os.path.exists(bat_path):
            os.remove(bat_path)
            print(f"\n🧹 Cleaned up: {bat_path}")

if __name__ == "__main__":
    test_v7p3r_tactical_abilities()