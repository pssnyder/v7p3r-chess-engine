#!/usr/bin/env python3

"""
Tactical Puzzle Tester for V7P3R v14.3

Direct SQLite-based puzzle testing to validate tactical performance
without requiring SQLAlchemy dependencies.
"""

import sqlite3
import os
import sys
import json
import chess
from datetime import datetime

# Add v7p3r engine path
v7p3r_path = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\src"
sys.path.append(v7p3r_path)

def get_puzzle_database():
    """Get path to puzzle database"""
    engine_tester_path = r"S:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester"
    db_path = os.path.join(engine_tester_path, 'databases', 'puzzles.db')
    return db_path

def fetch_puzzles_by_theme(theme, rating_min=1200, rating_max=1600, limit=10):
    """Fetch puzzles of specific theme and rating range"""
    
    db_path = get_puzzle_database()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query puzzles by theme and rating
    query = """
        SELECT id, fen, moves, rating, themes 
        FROM puzzles 
        WHERE themes LIKE ? 
        AND rating BETWEEN ? AND ? 
        ORDER BY rating 
        LIMIT ?
    """
    
    cursor.execute(query, (f'%{theme}%', rating_min, rating_max, limit))
    puzzles = cursor.fetchall()
    
    conn.close()
    
    return puzzles

def parse_puzzle_moves(moves_str):
    """Parse puzzle move sequence"""
    moves = moves_str.strip().split()
    
    # First move is the best move for the current side
    if len(moves) >= 1:
        best_move = moves[0]
        continuation = moves[1:] if len(moves) > 1 else []
        return best_move, continuation
    
    return None, []

def test_v7p3r_tactical_strength():
    """Test V7P3R v14.3 tactical strength on puzzle positions"""
    
    print("V7P3R v14.3 Tactical Puzzle Analysis")
    print("=" * 50)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test categories
    test_themes = [
        ('pin', 'Pin Tactics'),
        ('fork', 'Fork Tactics'), 
        ('mate', 'Checkmate Puzzles')
    ]
    
    results = {}
    
    for theme, theme_name in test_themes:
        print(f"Testing {theme_name}...")
        print("-" * 30)
        
        # Fetch puzzles for this theme
        puzzles = fetch_puzzles_by_theme(theme, limit=10)
        
        if not puzzles:
            print(f"❌ No {theme} puzzles found")
            continue
            
        theme_results = []
        
        for i, puzzle in enumerate(puzzles, 1):
            puzzle_id, fen, moves_str, rating, themes = puzzle
            best_move, continuation = parse_puzzle_moves(moves_str)
            
            print(f"{i:2d}. Puzzle {puzzle_id} (Rating: {rating})")
            print(f"    Position: {fen}")
            print(f"    Expected: {best_move}")
            print(f"    Themes: {themes}")
            
            # Test V7P3R on this position
            try:
                from v7p3r import V7P3REngine
                engine = V7P3REngine()
                
                # Create board from FEN
                board = chess.Board(fen)
                
                # Search for best move (2 second limit)
                engine_move = engine.search(board, time_limit=2.0)
                
                # Check if engine found the correct move
                correct = (engine_move == best_move)
                
                print(f"    V7P3R:    {engine_move} {'✅' if correct else '❌'}")
                
                theme_results.append({
                    'puzzle_id': puzzle_id,
                    'fen': fen,
                    'expected_move': best_move,
                    'engine_move': engine_move,
                    'correct': correct,
                    'rating': rating,
                    'themes': themes
                })
                
            except Exception as e:
                print(f"    Engine Error: {e}")
                theme_results.append({
                    'puzzle_id': puzzle_id,
                    'fen': fen,
                    'expected_move': best_move,
                    'engine_move': None,
                    'correct': False,
                    'rating': rating,
                    'themes': themes,
                    'error': str(e)
                })
            
            print()
        
        # Calculate theme accuracy
        correct_count = sum(1 for r in theme_results if r.get('correct', False))
        total_count = len(theme_results)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        print(f"{theme_name} Results: {correct_count}/{total_count} correct ({accuracy:.1f}%)")
        print()
        
        results[theme] = {
            'theme_name': theme_name,
            'puzzles': theme_results,
            'accuracy': accuracy,
            'correct': correct_count,
            'total': total_count
        }
    
    # Overall summary
    print("Overall Tactical Analysis Summary")
    print("=" * 50)
    
    total_correct = sum(r['correct'] for r in results.values())
    total_puzzles = sum(r['total'] for r in results.values())
    overall_accuracy = (total_correct / total_puzzles * 100) if total_puzzles > 0 else 0
    
    for theme, result in results.items():
        print(f"{result['theme_name']:20}: {result['correct']:2d}/{result['total']:2d} ({result['accuracy']:5.1f}%)")
    
    print("-" * 50)
    print(f"{'Overall Accuracy':20}: {total_correct:2d}/{total_puzzles:2d} ({overall_accuracy:5.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"v7p3r_v14_3_tactical_analysis_{timestamp}.json"
    
    output_data = {
        'engine_version': 'V7P3R v14.3',
        'test_time': datetime.now().isoformat(),
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_puzzles': total_puzzles,
        'theme_results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    # Test database access first
    print("Verifying puzzle database access...")
    db_path = get_puzzle_database()
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)
    
    # Quick database test
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM puzzles")
    puzzle_count = cursor.fetchone()[0]
    conn.close()
    
    print(f"✅ Database found with {puzzle_count:,} puzzles")
    print()
    
    # Run tactical tests
    test_v7p3r_tactical_strength()