#!/usr/bin/env python3

"""
Debug Puzzle Database Access

Test accessing the puzzle database to identify import issues.
"""

import sys
import os

# Add the engine tester path for database access
engine_tester_path = r"S:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester"
sys.path.append(engine_tester_path)
sys.path.append(os.path.join(engine_tester_path, 'databases'))

def debug_puzzle_database():
    """Debug puzzle database access"""
    
    print("Puzzle Database Debug")
    print("=" * 30)
    
    # Test 1: Direct SQLite access
    print("1. Direct SQLite access:")
    try:
        import sqlite3
        db_path = os.path.join(engine_tester_path, 'databases', 'puzzles.db')
        print(f"   Database path: {db_path}")
        print(f"   Database exists: {os.path.exists(db_path)}")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get sample puzzle
        cursor.execute("SELECT id, fen, moves, rating, themes FROM puzzles LIMIT 1")
        sample = cursor.fetchone()
        print(f"   Sample puzzle: {sample}")
        
        # Get puzzle count by theme
        cursor.execute("SELECT COUNT(*) FROM puzzles WHERE themes LIKE '%pin%'")
        pin_count = cursor.fetchone()[0]
        print(f"   Pin puzzles: {pin_count}")
        
        cursor.execute("SELECT COUNT(*) FROM puzzles WHERE themes LIKE '%fork%'")
        fork_count = cursor.fetchone()[0]
        print(f"   Fork puzzles: {fork_count}")
        
        cursor.execute("SELECT COUNT(*) FROM puzzles WHERE themes LIKE '%mate%'")
        mate_count = cursor.fetchone()[0]
        print(f"   Mate puzzles: {mate_count}")
        
        conn.close()
        print("   ✅ Direct SQLite access successful")
        
    except Exception as e:
        print(f"   ❌ SQLite error: {e}")
    
    # Test 2: Database module import
    print("\n2. Database module import:")
    try:
        from databases.database import Puzzle
        print("   ✅ Database module imported successfully")
    except Exception as e:
        print(f"   ❌ Database module error: {e}")
        print(f"   Trying alternative import...")
        try:
            import database
            print("   ✅ Alternative database import successful")
        except Exception as e2:
            print(f"   ❌ Alternative import error: {e2}")
    
    # Test 3: Universal Puzzle Analyzer import 
    print("\n3. Universal Puzzle Analyzer import:")
    try:
        from engine_utilities.universal_puzzle_analyzer import UniversalPuzzleAnalyzer
        print("   ✅ Universal Puzzle Analyzer imported successfully")
    except Exception as e:
        print(f"   ❌ Universal Puzzle Analyzer error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Create a minimal puzzle analyzer
    print("\n4. Creating minimal puzzle access:")
    try:
        db_path = os.path.join(engine_tester_path, 'databases', 'puzzles.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get 5 pin puzzles with rating 1200-1600
        cursor.execute("""
            SELECT id, fen, moves, rating, themes 
            FROM puzzles 
            WHERE themes LIKE '%pin%' 
            AND rating BETWEEN 1200 AND 1600 
            LIMIT 5
        """)
        
        pin_puzzles = cursor.fetchall()
        print(f"   Found {len(pin_puzzles)} pin puzzles:")
        
        for puzzle in pin_puzzles:
            puzzle_id, fen, moves, rating, themes = puzzle
            print(f"     • {puzzle_id}: Rating {rating}, Moves: {moves}")
        
        conn.close()
        print("   ✅ Minimal puzzle access successful")
        
    except Exception as e:
        print(f"   ❌ Minimal access error: {e}")

if __name__ == "__main__":
    debug_puzzle_database()