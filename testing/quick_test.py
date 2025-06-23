#!/usr/bin/env python3
"""
Quick Test Runner for V7P3R Chess Engine

A simple script to quickly run individual tests or test categories
for development and debugging purposes.

Usage:
    python quick_test.py                    # Run all tests
    python quick_test.py chess_game         # Run chess game tests
    python quick_test.py --engine           # Run engine tests
    python quick_test.py --utilities        # Run utility tests
    python quick_test.py --list             # List available tests
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def get_available_tests():
    """Get list of available test files."""
    test_dir = Path(__file__).parent / "unit_test_launchers"
    test_files = []
    
    if test_dir.exists():
        for file in test_dir.glob("*_testing.py"):
            test_name = file.stem
            test_files.append(test_name)
    
    return sorted(test_files)

def run_single_test(test_name):
    """Run a single test file."""
    test_file = f"unit_test_launchers/{test_name}.py"
    
    if not test_name.endswith("_testing"):
        test_file = f"unit_test_launchers/{test_name}_testing.py"
    
    test_path = Path(__file__).parent / test_file
    
    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        return False
    
    print(f"🚀 Running test: {test_name}")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "unittest", str(test_path), "-v"
        ], cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"💥 Error running test: {e}")
        return False

def run_category_tests(category):
    """Run tests for a specific category."""
    all_tests = get_available_tests()
    category_tests = []
    
    if category == "engine":
        category_tests = [t for t in all_tests if "v7p3r" in t.lower()]
    elif category == "utilities":
        category_tests = [t for t in all_tests if any(util in t.lower() for util in 
                         ["engine_db", "stockfish", "opening", "time", "etl"])]
    elif category == "metrics":
        category_tests = [t for t in all_tests if "metrics" in t.lower()]
    elif category == "chess":
        category_tests = [t for t in all_tests if "chess_game" in t.lower()]
    else:
        print(f"❌ Unknown category: {category}")
        return False
    
    if not category_tests:
        print(f"❌ No tests found for category: {category}")
        return False
    
    print(f"🚀 Running {category} tests ({len(category_tests)} tests)")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name in category_tests:
        if run_single_test(test_name):
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Results: {passed} passed, {failed} failed")
    
    return failed == 0

def list_tests():
    """List all available tests."""
    tests = get_available_tests()
    
    print("📋 Available Tests:")
    print("=" * 30)
    
    categories = {
        "Engine Tests": [t for t in tests if "v7p3r" in t.lower()],
        "Chess Game Tests": [t for t in tests if "chess_game" in t.lower()],
        "Utility Tests": [t for t in tests if any(util in t.lower() for util in 
                         ["engine_db", "stockfish", "opening", "time", "etl"])],
        "Metrics Tests": [t for t in tests if "metrics" in t.lower()],
        "Firebase Tests": [t for t in tests if "firebase" in t.lower()],
        "Other Tests": []
    }
    
    # Categorize remaining tests
    categorized = set()
    for cat_tests in categories.values():
        categorized.update(cat_tests)
    
    categories["Other Tests"] = [t for t in tests if t not in categorized]
    
    for category, cat_tests in categories.items():
        if cat_tests:
            print(f"\n{category}:")
            for test in sorted(cat_tests):
                print(f"  • {test}")
    
    print(f"\nTotal: {len(tests)} test files")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quick test runner for V7P3R Chess Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_test.py                    # Run full test suite
  python quick_test.py chess_game         # Run specific test
  python quick_test.py --engine           # Run engine tests
  python quick_test.py --utilities        # Run utility tests
  python quick_test.py --list             # List all tests
        """
    )
    
    parser.add_argument('test_name', nargs='?', help='Name of specific test to run')
    parser.add_argument('--engine', action='store_true', help='Run engine tests')
    parser.add_argument('--utilities', action='store_true', help='Run utility tests')
    parser.add_argument('--metrics', action='store_true', help='Run metrics tests')
    parser.add_argument('--firebase', action='store_true', help='Run Firebase tests')
    parser.add_argument('--chess', action='store_true', help='Run chess game tests')
    parser.add_argument('--list', action='store_true', help='List available tests')
    parser.add_argument('--full-suite', action='store_true', help='Run full test suite')
    
    args = parser.parse_args()
    
    if args.list:
        list_tests()
        return
    
    if args.full_suite:
        print("🚀 Running Full Test Suite")
        print("=" * 50)
        result = subprocess.run([
            sys.executable, "launch_unit_testing_suite.py", "--verbose"
        ], cwd=Path(__file__).parent)
        sys.exit(result.returncode)
    
    success = True
    
    if args.engine:
        success &= run_category_tests("engine")
    elif args.utilities:
        success &= run_category_tests("utilities")
    elif args.metrics:
        success &= run_category_tests("metrics")
    elif args.firebase:
        success &= run_category_tests("firebase")
    elif args.chess:
        success &= run_category_tests("chess")
    elif args.test_name:
        success = run_single_test(args.test_name)
    else:
        # Default: run full suite
        print("🚀 Running Full Test Suite (use --help for more options)")
        print("=" * 50)
        result = subprocess.run([
            sys.executable, "launch_unit_testing_suite.py"
        ], cwd=Path(__file__).parent)
        sys.exit(result.returncode)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
