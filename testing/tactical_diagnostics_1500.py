#!/usr/bin/env python3

"""
V7P3R Advanced Tactical Diagnostics - 1500+ Rating Puzzles

Systematic puzzle-based improvement workflow:
1. Query higher-rated puzzles (1500-2000+) 
2. Create consistent test sets for before/after comparisons
3. Detailed diagnostic analysis of failures
4. Targeted improvement suggestions
5. Validation runs to measure progress

Focus: Measurable improvements through puzzle performance analysis
"""

import sqlite3
import os
import sys
import json
import chess
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Add v7p3r engine path
v7p3r_path = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\src"
sys.path.append(v7p3r_path)

def get_puzzle_database():
    """Get path to puzzle database"""
    engine_tester_path = r"S:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester"
    db_path = os.path.join(engine_tester_path, 'databases', 'puzzles.db')
    return db_path

def query_puzzles_by_rating_and_theme(rating_min=1500, rating_max=2000, theme=None, limit=10):
    """Query puzzles by rating range and optional theme"""
    
    db_path = get_puzzle_database()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if theme:
        query = """
            SELECT id, fen, moves, rating, themes 
            FROM puzzles 
            WHERE rating BETWEEN ? AND ? 
            AND themes LIKE ?
            ORDER BY rating, RANDOM()
            LIMIT ?
        """
        cursor.execute(query, (rating_min, rating_max, f'%{theme}%', limit))
    else:
        query = """
            SELECT id, fen, moves, rating, themes 
            FROM puzzles 
            WHERE rating BETWEEN ? AND ? 
            ORDER BY rating, RANDOM()
            LIMIT ?
        """
        cursor.execute(query, (rating_min, rating_max, limit))
    
    puzzles = cursor.fetchall()
    conn.close()
    
    return puzzles

def analyze_puzzle_failure(puzzle_data, engine, detailed_analysis=True):
    """
    Detailed analysis of puzzle failures for diagnostic purposes
    """
    puzzle_id, fen, moves_str, rating, themes = puzzle_data
    sequence = moves_str.strip().split() if moves_str else []
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC ANALYSIS: Puzzle {puzzle_id}")
    print(f"Rating: {rating} | Themes: {themes}")
    print(f"{'='*60}")
    
    if len(sequence) < 2:
        return {"error": "Insufficient moves in sequence"}
    
    board = chess.Board(fen)
    print(f"Starting position: {fen}")
    print(f"Turn to move: {'White' if board.turn else 'Black'}")
    
    # Track detailed failure analysis
    analysis = {
        'puzzle_id': puzzle_id,
        'rating': rating,
        'themes': themes.split() if themes else [],
        'sequence': sequence,
        'positions_analyzed': [],
        'failure_patterns': [],
        'suggested_improvements': []
    }
    
    # Process each position in sequence
    for move_index in range(0, len(sequence), 2):
        if move_index + 1 >= len(sequence):
            break
            
        opponent_move_text = sequence[move_index]
        expected_move_text = sequence[move_index + 1]
        position_num = (move_index // 2) + 1
        
        print(f"\n--- Position {position_num} Analysis ---")
        
        # Apply opponent move
        try:
            try:
                opponent_move = chess.Move.from_uci(opponent_move_text)
                if opponent_move not in board.legal_moves:
                    raise ValueError("UCI move not legal")
            except:
                opponent_move = board.parse_san(opponent_move_text)
            
            print(f"After opponent plays {opponent_move_text}:")
            board.push(opponent_move)
            challenge_fen = board.fen()
            print(f"Position: {challenge_fen}")
            
        except Exception as e:
            print(f"❌ Cannot apply opponent move: {e}")
            continue
        
        # Parse expected move
        try:
            try:
                expected_move = chess.Move.from_uci(expected_move_text)
                if expected_move not in board.legal_moves:
                    raise ValueError("UCI move not legal")
                expected_uci = str(expected_move)
            except:
                expected_move = board.parse_san(expected_move_text)
                expected_uci = str(expected_move)
        except Exception as e:
            print(f"❌ Cannot parse expected move: {e}")
            continue
        
        print(f"Expected solution: {expected_uci}")
        
        # Get engine analysis with multiple time controls
        position_analysis = {
            'position_num': position_num,
            'challenge_fen': challenge_fen,
            'opponent_move': opponent_move_text,
            'expected_move': expected_uci,
            'engine_results': {}
        }
        
        # Test with different time limits
        time_limits = [1.0, 3.0, 5.0] if detailed_analysis else [3.0]
        
        for time_limit in time_limits:
            print(f"\nEngine analysis ({time_limit}s):")
            
            try:
                challenge_board = chess.Board(challenge_fen)
                engine_move = engine.search(challenge_board, time_limit=time_limit)
                
                if engine_move:
                    engine_move_str = str(engine_move)
                    correct = (engine_move_str == expected_uci)
                    
                    print(f"V7P3R ({time_limit}s): {engine_move_str} {'✅' if correct else '❌'}")
                    
                    # Get evaluation before and after engine move
                    eval_before = engine._evaluate_position(challenge_board)
                    
                    if not correct:
                        # Analyze the difference
                        challenge_board.push(chess.Move.from_uci(engine_move_str))
                        eval_after_engine = engine._evaluate_position(challenge_board)
                        challenge_board.pop()
                        
                        challenge_board.push(expected_move)
                        eval_after_expected = engine._evaluate_position(challenge_board)
                        challenge_board.pop()
                        
                        eval_diff = eval_after_expected - eval_after_engine
                        
                        print(f"  Evaluation analysis:")
                        print(f"    Before move: {eval_before:+.2f}")
                        print(f"    After engine move: {eval_after_engine:+.2f}")
                        print(f"    After expected move: {eval_after_expected:+.2f}")
                        print(f"    Missed opportunity: {eval_diff:+.2f} centipawns")
                        
                        if abs(eval_diff) > 100:
                            analysis['failure_patterns'].append(f"Position {position_num}: Missed {eval_diff:+.1f}cp opportunity")
                    
                    position_analysis['engine_results'][f'{time_limit}s'] = {
                        'move': engine_move_str,
                        'correct': correct,
                        'eval_before': eval_before,
                        'time_used': time_limit
                    }
                    
                else:
                    print(f"V7P3R ({time_limit}s): No move returned ❌")
                    position_analysis['engine_results'][f'{time_limit}s'] = {
                        'move': None,
                        'correct': False,
                        'error': 'No move returned'
                    }
                    
            except Exception as e:
                print(f"❌ Engine error ({time_limit}s): {e}")
                position_analysis['engine_results'][f'{time_limit}s'] = {
                    'move': None,
                    'correct': False,
                    'error': str(e)
                }
        
        analysis['positions_analyzed'].append(position_analysis)
        
        # Continue with expected move for sequence
        try:
            board.push(expected_move)
        except:
            break
    
    # Generate improvement suggestions based on failure patterns
    if analysis['failure_patterns']:
        print(f"\n--- IMPROVEMENT ANALYSIS ---")
        themes_list = themes.split() if themes else []
        
        # Theme-specific suggestions
        if 'pin' in themes_list and any('Missed' in pattern for pattern in analysis['failure_patterns']):
            analysis['suggested_improvements'].append("Pin recognition: Consider enhancing piece mobility analysis")
        
        if 'fork' in themes_list and any('Missed' in pattern for pattern in analysis['failure_patterns']):
            analysis['suggested_improvements'].append("Fork detection: Improve multi-piece attack evaluation")
        
        if 'mate' in themes_list and any('Missed' in pattern for pattern in analysis['failure_patterns']):
            analysis['suggested_improvements'].append("Mate finding: Enhance mate threat detection in evaluation")
        
        if rating >= 1700 and analysis['failure_patterns']:
            analysis['suggested_improvements'].append("High-level tactics: Consider deeper search or improved move ordering")
        
        for suggestion in analysis['suggested_improvements']:
            print(f"💡 {suggestion}")
    
    return analysis

def create_diagnostic_test_set(rating_ranges=None, themes=None, puzzles_per_category=8):
    """
    Create a consistent test set for before/after comparisons
    """
    if rating_ranges is None:
        rating_ranges = [(1500, 1700), (1700, 1900), (1900, 2200)]
    
    if themes is None:
        themes = ['pin', 'fork', 'mate', 'sacrifice', 'deflection']
    
    test_set = {
        'created': datetime.now().isoformat(),
        'categories': {},
        'total_puzzles': 0
    }
    
    print("Creating V7P3R Diagnostic Test Set")
    print("=" * 50)
    
    for rating_min, rating_max in rating_ranges:
        for theme in themes:
            category_name = f"{theme}_{rating_min}_{rating_max}"
            print(f"Querying {theme} puzzles ({rating_min}-{rating_max})...")
            
            puzzles = query_puzzles_by_rating_and_theme(
                rating_min, rating_max, theme, puzzles_per_category
            )
            
            if puzzles:
                test_set['categories'][category_name] = {
                    'theme': theme,
                    'rating_min': rating_min,
                    'rating_max': rating_max,
                    'puzzles': [
                        {
                            'id': p[0],
                            'fen': p[1], 
                            'moves': p[2],
                            'rating': p[3],
                            'themes': p[4]
                        } for p in puzzles
                    ]
                }
                test_set['total_puzzles'] += len(puzzles)
                print(f"  Found {len(puzzles)} {theme} puzzles")
            else:
                print(f"  No {theme} puzzles found in {rating_min}-{rating_max}")
    
    # Save test set
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    test_set_file = f"v7p3r_diagnostic_test_set_{timestamp}.json"
    
    with open(test_set_file, 'w') as f:
        json.dump(test_set, f, indent=2)
    
    print(f"\nTest set created: {test_set['total_puzzles']} puzzles")
    print(f"Saved to: {test_set_file}")
    
    return test_set, test_set_file

def run_diagnostic_analysis(test_set_file=None, engine_version="v14.3", detailed=True):
    """
    Run comprehensive diagnostic analysis on test set
    """
    # Load test set
    if test_set_file:
        with open(test_set_file, 'r') as f:
            test_set = json.load(f)
    else:
        print("Creating new test set...")
        test_set, test_set_file = create_diagnostic_test_set()
    
    print(f"\nRunning V7P3R {engine_version} Diagnostic Analysis")
    print(f"Test set: {test_set['total_puzzles']} puzzles")
    print("=" * 60)
    
    # Initialize engine
    try:
        from v7p3r import V7P3REngine
        engine = V7P3REngine()
        print("✅ V7P3R engine initialized")
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return
    
    # Results tracking
    results = {
        'engine_version': engine_version,
        'test_set_file': test_set_file,
        'analysis_time': datetime.now().isoformat(),
        'category_results': {},
        'overall_stats': {},
        'failure_analysis': {},
        'improvement_suggestions': []
    }
    
    total_positions = 0
    total_correct = 0
    all_failures = []
    
    # Analyze each category
    for category_name, category_data in test_set['categories'].items():
        print(f"\n{'='*60}")
        print(f"CATEGORY: {category_name}")
        print(f"Theme: {category_data['theme']} | Rating: {category_data['rating_min']}-{category_data['rating_max']}")
        print(f"{'='*60}")
        
        category_positions = 0
        category_correct = 0
        category_failures = []
        
        for i, puzzle_data in enumerate(category_data['puzzles'], 1):
            print(f"\n--- Puzzle {i}/{len(category_data['puzzles'])} ---")
            
            # Convert to expected format
            puzzle_tuple = (
                puzzle_data['id'],
                puzzle_data['fen'],
                puzzle_data['moves'],
                puzzle_data['rating'],
                puzzle_data['themes']
            )
            
            # Run detailed analysis on failures and high-rated puzzles
            should_analyze = detailed and (puzzle_data['rating'] >= 1700)
            
            if should_analyze:
                analysis = analyze_puzzle_failure(puzzle_tuple, engine, detailed_analysis=True)
                if analysis.get('failure_patterns'):
                    category_failures.extend(analysis['failure_patterns'])
                    all_failures.extend(analysis['failure_patterns'])
            else:
                # Quick analysis
                analysis = analyze_basic_puzzle(puzzle_tuple, engine)
            
            if 'positions_analyzed' in analysis:
                category_positions += len(analysis['positions_analyzed'])
                category_correct += sum(1 for pos in analysis['positions_analyzed'] 
                                      if any(result.get('correct', False) 
                                           for result in pos.get('engine_results', {}).values()))
        
        # Category summary
        category_accuracy = (category_correct / category_positions * 100) if category_positions > 0 else 0
        
        results['category_results'][category_name] = {
            'puzzles': len(category_data['puzzles']),
            'positions': category_positions,
            'correct': category_correct,
            'accuracy': category_accuracy,
            'failures': category_failures
        }
        
        total_positions += category_positions
        total_correct += category_correct
        
        print(f"\nCategory Summary: {category_correct}/{category_positions} ({category_accuracy:.1f}%)")
    
    # Overall analysis
    overall_accuracy = (total_correct / total_positions * 100) if total_positions > 0 else 0
    
    results['overall_stats'] = {
        'total_positions': total_positions,
        'total_correct': total_correct,
        'overall_accuracy': overall_accuracy
    }
    
    print(f"\n{'='*60}")
    print(f"OVERALL DIAGNOSTIC RESULTS")
    print(f"{'='*60}")
    print(f"Total positions analyzed: {total_positions}")
    print(f"Total correct: {total_correct}")
    print(f"Overall accuracy: {overall_accuracy:.1f}%")
    
    # Generate targeted improvement suggestions
    improvement_suggestions = generate_improvement_suggestions(results, all_failures)
    results['improvement_suggestions'] = improvement_suggestions
    
    print(f"\n--- TARGETED IMPROVEMENTS ---")
    for i, suggestion in enumerate(improvement_suggestions, 1):
        print(f"{i}. {suggestion}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"v7p3r_{engine_version}_diagnostic_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDiagnostic results saved to: {results_file}")
    
    return results

def analyze_basic_puzzle(puzzle_data, engine):
    """Basic puzzle analysis without detailed diagnostics"""
    puzzle_id, fen, moves_str, rating, themes = puzzle_data
    sequence = moves_str.strip().split() if moves_str else []
    
    if len(sequence) < 2:
        return {"error": "Insufficient moves"}
    
    board = chess.Board(fen)
    positions_analyzed = []
    
    for move_index in range(0, len(sequence), 2):
        if move_index + 1 >= len(sequence):
            break
            
        opponent_move_text = sequence[move_index]
        expected_move_text = sequence[move_index + 1]
        
        try:
            # Apply opponent move
            try:
                opponent_move = chess.Move.from_uci(opponent_move_text)
            except:
                opponent_move = board.parse_san(opponent_move_text)
            
            board.push(opponent_move)
            challenge_fen = board.fen()
            
            # Parse expected move
            try:
                expected_move = chess.Move.from_uci(expected_move_text)
                expected_uci = str(expected_move)
            except:
                expected_move = board.parse_san(expected_move_text)
                expected_uci = str(expected_move)
            
            # Get engine move
            challenge_board = chess.Board(challenge_fen)
            engine_move = engine.search(challenge_board, time_limit=3.0)
            
            correct = (str(engine_move) == expected_uci) if engine_move else False
            
            positions_analyzed.append({
                'position_num': len(positions_analyzed) + 1,
                'challenge_fen': challenge_fen,
                'expected_move': expected_uci,
                'engine_results': {
                    '3.0s': {
                        'move': str(engine_move) if engine_move else None,
                        'correct': correct
                    }
                }
            })
            
            print(f"Position {len(positions_analyzed)}: {str(engine_move) if engine_move else 'None'} vs {expected_uci} {'✅' if correct else '❌'}")
            
            # Continue sequence
            board.push(expected_move)
            
        except Exception as e:
            print(f"Error in position: {e}")
            break
    
    return {
        'puzzle_id': puzzle_id,
        'positions_analyzed': positions_analyzed
    }

def generate_improvement_suggestions(results, all_failures):
    """Generate targeted improvement suggestions based on diagnostic results"""
    suggestions = []
    
    # Analyze category performance
    category_results = results.get('category_results', {})
    
    # Rating-based analysis
    high_rating_categories = [cat for cat, data in category_results.items() 
                             if '1900' in cat and data.get('accuracy', 0) < 70]
    
    if high_rating_categories:
        suggestions.append("Search depth: Consider increasing default depth for high-rating positions (1900+)")
    
    # Theme-based analysis
    pin_performance = [data.get('accuracy', 0) for cat, data in category_results.items() if 'pin' in cat]
    fork_performance = [data.get('accuracy', 0) for cat, data in category_results.items() if 'fork' in cat]
    mate_performance = [data.get('accuracy', 0) for cat, data in category_results.items() if 'mate' in cat]
    
    if pin_performance and sum(pin_performance) / len(pin_performance) < 75:
        suggestions.append("Pin evaluation: Enhance trapped piece and mobility analysis")
    
    if fork_performance and sum(fork_performance) / len(fork_performance) < 75:
        suggestions.append("Fork detection: Improve multi-target attack recognition")
    
    if mate_performance and sum(mate_performance) / len(mate_performance) < 60:
        suggestions.append("Mate finding: Enhance mate threat evaluation and search extension")
    
    # Failure pattern analysis
    if len(all_failures) > 5:
        avg_missed_value = sum(float(f.split()[3].replace('cp', '').replace('+', '')) 
                              for f in all_failures if 'Missed' in f and 'cp' in f) / len(all_failures)
        
        if avg_missed_value > 200:
            suggestions.append(f"Evaluation accuracy: Missing high-value tactics (avg {avg_missed_value:.0f}cp)")
    
    if not suggestions:
        suggestions.append("Performance is strong across all tested categories - consider testing higher ratings")
    
    return suggestions

if __name__ == "__main__":
    print("V7P3R Advanced Tactical Diagnostics")
    print("=" * 50)
    
    # Check database
    db_path = get_puzzle_database()
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)
    
    # Quick database stats
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM puzzles WHERE rating >= 1500")
    high_rating_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM puzzles WHERE rating >= 1700") 
    very_high_rating_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"Database: {high_rating_count:,} puzzles ≥1500 rating")
    print(f"          {very_high_rating_count:,} puzzles ≥1700 rating")
    print()
    
    # Run diagnostic analysis
    results = run_diagnostic_analysis(detailed=True)