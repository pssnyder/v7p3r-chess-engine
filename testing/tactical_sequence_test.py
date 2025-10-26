#!/usr/bin/env python3

"""
V7P3R Tactical Puzzle Tester - Proper Sequence Analysis

Uses the correct puzzle testing methodology where:
1. Start with puzzle position
2. Apply opponent's move from sequence
3. Challenge engine to find the response
4. Check if engine found the expected solution move
5. Continue through the sequence

This matches the methodology from the Universal Puzzle Analyzer.
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

def fetch_puzzles_by_theme(theme, rating_min=1200, rating_max=1600, limit=5):
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

def parse_puzzle_sequence(moves_str):
    """Parse puzzle move sequence into individual moves"""
    if not moves_str:
        return []
    return moves_str.strip().split()

def analyze_puzzle_sequence(puzzle_data, engine, time_limit=2.0):
    """
    Analyze complete puzzle sequence using proper methodology:
    1. Start with puzzle position
    2. For each opponent move + expected response pair:
       - Apply opponent move
       - Challenge engine to find response
       - Check if engine found the expected move
    """
    
    puzzle_id, fen, moves_str, rating, themes = puzzle_data
    sequence = parse_puzzle_sequence(moves_str)
    
    print(f"\nPuzzle {puzzle_id} (Rating: {rating})")
    print(f"Themes: {themes}")
    print(f"Starting FEN: {fen}")
    print(f"Solution sequence: {' '.join(sequence)}")
    
    if len(sequence) < 2:
        print("❌ Insufficient moves in sequence")
        return {
            'puzzle_id': puzzle_id,
            'success': False,
            'error': 'Insufficient moves in sequence',
            'positions_tested': 0,
            'positions_correct': 0
        }
    
    # Initialize tracking
    board = chess.Board(fen)
    positions_tested = 0
    positions_correct = 0
    position_results = []
    
    # Process sequence in pairs: opponent_move, expected_response
    for move_index in range(0, len(sequence), 2):
        if move_index + 1 >= len(sequence):
            break  # Need both opponent move and expected response
            
        opponent_move_text = sequence[move_index]
        expected_move_text = sequence[move_index + 1]
        
        position_num = (move_index // 2) + 1
        print(f"\n--- Position {position_num} ---")
        
        # Apply opponent's move
        try:
            # Try UCI format first, then SAN
            try:
                opponent_move = chess.Move.from_uci(opponent_move_text)
                if opponent_move not in board.legal_moves:
                    raise ValueError("UCI move not legal")
            except:
                opponent_move = board.parse_san(opponent_move_text)
            
            print(f"Opponent plays: {opponent_move_text}")
            board.push(opponent_move)
            challenge_fen = board.fen()
            
        except Exception as e:
            print(f"❌ Cannot apply opponent move {opponent_move_text}: {e}")
            break
        
        # Parse expected response
        try:
            try:
                expected_move = chess.Move.from_uci(expected_move_text)
                if expected_move not in board.legal_moves:
                    raise ValueError("UCI move not legal")
                expected_move_uci = str(expected_move)
            except:
                expected_move = board.parse_san(expected_move_text)
                expected_move_uci = str(expected_move)
                
        except Exception as e:
            print(f"❌ Cannot parse expected move {expected_move_text}: {e}")
            break
        
        # Challenge engine to find the best move
        turn_info = "White" if board.turn else "Black"
        print(f"Position: {turn_info} to move after {opponent_move_text}")
        print(f"Expected: {expected_move_uci}")
        
        try:
            challenge_board = chess.Board(challenge_fen)
            engine_move = engine.search(challenge_board, time_limit=time_limit)
            
            if engine_move:
                engine_move_str = str(engine_move)
                correct = (engine_move_str == expected_move_uci)
                
                print(f"V7P3R:   {engine_move_str} {'✅' if correct else '❌'}")
                
                positions_tested += 1
                if correct:
                    positions_correct += 1
                
                position_results.append({
                    'position_num': position_num,
                    'challenge_fen': challenge_fen,
                    'opponent_move': opponent_move_text,
                    'expected_move': expected_move_uci,
                    'engine_move': engine_move_str,
                    'correct': correct
                })
                
                # Continue sequence with expected move (not engine move)
                board.push(expected_move)
                
            else:
                print("V7P3R:   (no move returned) ❌")
                positions_tested += 1
                
                position_results.append({
                    'position_num': position_num,
                    'challenge_fen': challenge_fen,
                    'opponent_move': opponent_move_text,
                    'expected_move': expected_move_uci,
                    'engine_move': None,
                    'correct': False
                })
                
                # Continue with expected move
                board.push(expected_move)
                
        except Exception as e:
            print(f"❌ Engine error: {e}")
            positions_tested += 1
            
            position_results.append({
                'position_num': position_num,
                'challenge_fen': challenge_fen,
                'opponent_move': opponent_move_text,
                'expected_move': expected_move_uci,
                'engine_move': None,
                'correct': False,
                'error': str(e)
            })
            
            # Continue with expected move
            try:
                board.push(expected_move)
            except:
                break
    
    # Calculate results
    accuracy = (positions_correct / positions_tested * 100) if positions_tested > 0 else 0
    perfect_sequence = (positions_correct == positions_tested) if positions_tested > 0 else False
    
    print(f"\nSequence Results: {positions_correct}/{positions_tested} correct ({accuracy:.1f}%)")
    print(f"Perfect sequence: {'Yes' if perfect_sequence else 'No'}")
    
    return {
        'puzzle_id': puzzle_id,
        'fen': fen,
        'rating': rating,
        'themes': themes.split() if themes else [],
        'sequence': sequence,
        'success': True,
        'positions_tested': positions_tested,
        'positions_correct': positions_correct,
        'accuracy': accuracy,
        'perfect_sequence': perfect_sequence,
        'position_results': position_results
    }

def test_v7p3r_tactical_strength():
    """Test V7P3R v14.3 tactical strength using proper puzzle methodology"""
    
    print("V7P3R v14.3 Tactical Puzzle Analysis")
    print("Using Proper Sequence Testing Methodology")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize engine
    try:
        from v7p3r import V7P3REngine
        engine = V7P3REngine()
        print("✅ V7P3R engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize V7P3R engine: {e}")
        return
    
    # Test categories with smaller counts for initial validation
    test_themes = [
        ('pin', 'Pin Tactics'),
        ('fork', 'Fork Tactics'), 
        ('mate', 'Checkmate Puzzles')
    ]
    
    all_results = []
    theme_summaries = {}
    
    for theme, theme_name in test_themes:
        print(f"\nTesting {theme_name}...")
        print("-" * 40)
        
        # Fetch puzzles for this theme
        puzzles = fetch_puzzles_by_theme(theme, limit=5)  # Start with 5 puzzles per theme
        
        if not puzzles:
            print(f"❌ No {theme} puzzles found")
            continue
            
        theme_results = []
        total_positions = 0
        total_correct = 0
        perfect_sequences = 0
        
        for i, puzzle in enumerate(puzzles, 1):
            print(f"\n{theme_name} Puzzle {i}/5:")
            
            result = analyze_puzzle_sequence(puzzle, engine, time_limit=3.0)
            
            if result['success']:
                theme_results.append(result)
                all_results.append(result)
                total_positions += result['positions_tested']
                total_correct += result['positions_correct']
                if result['perfect_sequence']:
                    perfect_sequences += 1
        
        # Theme summary
        if theme_results:
            theme_accuracy = (total_correct / total_positions * 100) if total_positions > 0 else 0
            perfect_rate = (perfect_sequences / len(theme_results) * 100) if theme_results else 0
            
            theme_summaries[theme] = {
                'name': theme_name,
                'puzzles': len(theme_results),
                'total_positions': total_positions,
                'total_correct': total_correct,
                'accuracy': theme_accuracy,
                'perfect_sequences': perfect_sequences,
                'perfect_rate': perfect_rate
            }
            
            print(f"\n{theme_name} Summary:")
            print(f"  Puzzles: {len(theme_results)}")
            print(f"  Positions: {total_correct}/{total_positions} ({theme_accuracy:.1f}%)")
            print(f"  Perfect sequences: {perfect_sequences}/{len(theme_results)} ({perfect_rate:.1f}%)")
    
    # Overall Summary
    print("\n" + "=" * 60)
    print("OVERALL TACTICAL ANALYSIS SUMMARY")
    print("=" * 60)
    
    grand_total_positions = sum(s['total_positions'] for s in theme_summaries.values())
    grand_total_correct = sum(s['total_correct'] for s in theme_summaries.values())
    grand_total_puzzles = sum(s['puzzles'] for s in theme_summaries.values())
    grand_perfect_sequences = sum(s['perfect_sequences'] for s in theme_summaries.values())
    
    overall_accuracy = (grand_total_correct / grand_total_positions * 100) if grand_total_positions > 0 else 0
    overall_perfect_rate = (grand_perfect_sequences / grand_total_puzzles * 100) if grand_total_puzzles > 0 else 0
    
    for theme, summary in theme_summaries.items():
        print(f"{summary['name']:18}: {summary['total_correct']:2d}/{summary['total_positions']:2d} ({summary['accuracy']:5.1f}%) | Perfect: {summary['perfect_sequences']}/{summary['puzzles']} ({summary['perfect_rate']:4.1f}%)")
    
    print("-" * 60)
    print(f"{'OVERALL':18}: {grand_total_correct:2d}/{grand_total_positions:2d} ({overall_accuracy:5.1f}%) | Perfect: {grand_perfect_sequences}/{grand_total_puzzles} ({overall_perfect_rate:4.1f}%)")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"v7p3r_v14_3_tactical_sequence_analysis_{timestamp}.json"
    
    output_data = {
        'engine_version': 'V7P3R v14.3',
        'test_methodology': 'Proper sequence analysis with opponent moves',
        'test_time': datetime.now().isoformat(),
        'overall_summary': {
            'total_puzzles': grand_total_puzzles,
            'total_positions': grand_total_positions,
            'total_correct': grand_total_correct,
            'overall_accuracy': overall_accuracy,
            'perfect_sequences': grand_perfect_sequences,
            'perfect_sequence_rate': overall_perfect_rate
        },
        'theme_summaries': theme_summaries,
        'detailed_results': all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return all_results

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
    
    # Run tactical tests with proper methodology
    test_v7p3r_tactical_strength()