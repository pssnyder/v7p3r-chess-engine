#!/usr/bin/env python3
"""
Build script for V7P3R v7.2 release
Creates standalone executable for tournament distribution
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def build_v7p3r_v7_2():
    """Build V7P3R v7.2 executable"""
    print("Building V7P3R v7.2 Release")
    print("=" * 40)
    
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)
    
    # Check if PyInstaller is available
    try:
        subprocess.run([sys.executable, '-m', 'PyInstaller', '--version'], 
                      check=True, capture_output=True)
        print("✅ PyInstaller found")
    except subprocess.CalledProcessError:
        print("❌ PyInstaller not found. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'], check=True)
        print("✅ PyInstaller installed")
    
    # Clean previous builds
    try:
        if os.path.exists('dist'):
            shutil.rmtree('dist')
            print("🧹 Cleaned previous dist folder")
    except PermissionError:
        print("⚠️  Could not clean dist folder (files may be in use)")
    
    try:
        if os.path.exists('build'):
            shutil.rmtree('build')
            print("🧹 Cleaned previous build folder")
    except PermissionError:
        print("⚠️  Could not clean build folder (files may be in use)")
    
    # Run PyInstaller
    print("\n🔨 Building executable...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'PyInstaller',
            '--onefile',
            '--name', 'V7P3R_v7.2',
            '--console',
            '--optimize', '2',
            'src/v7p3r_uci.py'
        ], check=True, capture_output=True, text=True)
        
        print("✅ Build completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    # Check if executable was created
    exe_path = Path('dist/V7P3R_v7.2.exe')
    if exe_path.exists():
        file_size = exe_path.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"✅ Executable created: {exe_path}")
        print(f"📦 File size: {file_size:.1f} MB")
        
        # Test the executable
        print("\n🧪 Testing executable...")
        try:
            test_result = subprocess.run([
                str(exe_path)
            ], input="uci\nquit\n", text=True, capture_output=True, timeout=10)
            
            if "V7P3R v7.2" in test_result.stdout and "uciok" in test_result.stdout:
                print("✅ Executable test passed!")
                print("\n🎉 V7P3R v7.2 build completed successfully!")
                print(f"📍 Location: {exe_path.absolute()}")
                return True
            else:
                print("❌ Executable test failed - unexpected output")
                print("STDOUT:", test_result.stdout)
                print("STDERR:", test_result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Executable test timed out")
            return False
        except Exception as e:
            print(f"❌ Error testing executable: {e}")
            return False
    else:
        print("❌ Executable not found in dist folder")
        return False

def create_release_info():
    """Create release information file"""
    release_info = """
V7P3R Chess Engine v7.2 Release Notes
=====================================

Version: 7.2
Release Date: August 25, 2025
Author: Pat Snyder

MAJOR IMPROVEMENTS IN V7.2:
---------------------------

Performance Enhancements:
- Reduced average move time from 12-33s to 2-8s
- Competitive time management with SlowMate
- 10,000+ nodes per second search speed

Search Improvements:
- Killer move heuristic for better move ordering
- History heuristic for position memory
- Enhanced MVV-LVA capture prioritization
- Late move reduction in shallow search

Time Management:
- Adaptive time allocation by game phase
- Predictive iteration stopping
- Hard time caps to prevent overruns

Bug Fixes:
- Fixed mate evaluation (no more +M500)
- Proper mate distance calculation
- Corrected evaluation perspective

Tournament Ready:
- UCI compliant interface
- Proper game reset functionality
- Memory management optimizations

TECHNICAL SPECIFICATIONS:
------------------------
- Search: Negamax with alpha-beta pruning
- Evaluation: Material + positional heuristics
- Move Ordering: Captures, killers, history, development
- Time Control: Adaptive based on remaining time
- Depth: Iterative deepening up to configured limit

USAGE:
------
V7P3R_v7.2.exe

The engine communicates via UCI protocol.
Compatible with most chess GUIs and tournament managers.

For support: github.com/pssnyder/engine-tester
"""
    
    with open('dist/V7P3R_v7.2_README.txt', 'w', encoding='utf-8') as f:
        f.write(release_info)
    
    print("Release notes created")

if __name__ == "__main__":
    success = build_v7p3r_v7_2()
    if success:
        create_release_info()
        print("\n🏆 V7P3R v7.2 release package ready for distribution!")
    else:
        print("\n💥 Build failed. Please check errors above.")
        sys.exit(1)
