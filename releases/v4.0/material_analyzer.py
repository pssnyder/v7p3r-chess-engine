import os
import sys
import json
import sqlite3
import chess
import chess.pgn
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def analyze_games_with_material():
    """Analyze games using material-based evaluation"""
    print("=== Starting V7P3R Material-Based Analysis ===")
    
    # Create results directory
    if not os.path.exists('analysis_results'):
        os.makedirs('analysis_results')
    
    # Load games from exported_pgn directory
    games = load_pgn_games()
    if not games:
        print("No games to analyze.")
        return
    
    # Load metrics from database
    game_metrics, move_metrics = load_metrics()
    if not move_metrics:
        print("No move metrics to analyze.")
        return
    
    # Analyze positions using material difference
    position_evals = analyze_positions(games, move_metrics)
    
    # Identify critical positions
    critical_positions = identify_critical_positions(position_evals)
    
    # Debug information
    print(f"Analyzed a total of {len(position_evals)} positions")
    
    # Generate charts
    generate_charts(position_evals, critical_positions)
    
    # Generate report
    generate_report(games, position_evals, critical_positions)
    
    print("=== Analysis Complete ===")
    print("Results saved to 'analysis_results' directory")

def load_pgn_games():
    """Load PGN games from the exported_pgn directory"""
    pgn_dir = "exported_pgn"
    games = []
    
    if not os.path.exists(pgn_dir):
        print(f"PGN directory not found: {pgn_dir}")
        return games
    
    pgn_files = [f for f in os.listdir(pgn_dir) if f.endswith('.pgn')]
    print(f"Found {len(pgn_files)} PGN files in {pgn_dir}")
    
    for pgn_file in pgn_files:
        pgn_path = os.path.join(pgn_dir, pgn_file)
        try:
            with open(pgn_path, 'r') as f:
                game = chess.pgn.read_game(f)
                if game:
                    game_id = int(pgn_file.replace('game_', '').replace('.pgn', ''))
                    games.append((game_id, game))
        except Exception as e:
            print(f"Error loading {pgn_file}: {str(e)}")
    
    print(f"Loaded {len(games)} games")
    return games

def load_metrics():
    """Load metrics from the database"""
    db_path = "engine_metrics.db"
    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found")
        return {}, {}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Load game results
        cursor.execute('''
            SELECT id, result, white_player, black_player, total_moves, game_duration
            FROM game_results
        ''')
        game_results = cursor.fetchall()
        
        game_metrics = {}
        for row in game_results:
            game_metrics[row[0]] = {
                'result': row[1],
                'white_player': row[2],
                'black_player': row[3],
                'total_moves': row[4],
                'duration': row[5]
            }
        
        # Load move analysis
        cursor.execute('''
            SELECT game_id, move_number, player_color, move_uci, evaluation_score, search_time
            FROM move_analysis
        ''')
        move_data = cursor.fetchall()
        
        move_metrics = {}
        for row in move_data:
            game_id = row[0]
            move_number = row[1]
            color = row[2]
            
            if game_id not in move_metrics:
                move_metrics[game_id] = {}
            
            move_key = f"{move_number}_{color}"
            move_metrics[game_id][move_key] = {
                'move': row[3],
                'eval': row[4],
                'time': row[5]
            }
        
        conn.close()
        print(f"Loaded metrics for {len(game_metrics)} games and {sum(len(moves) for moves in move_metrics.values())} moves")
        
        return game_metrics, move_metrics
        
    except Exception as e:
        print(f"Error loading metrics: {str(e)}")
        return {}, {}

def count_material(board, color):
    """Count material value for a specific color"""
    piece_values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.25,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0  # King's value not counted in material
    }
    
    material = 0.0
    for piece_type in piece_values:
        material += len(board.pieces(piece_type, color)) * piece_values[piece_type]
        
    return material

def analyze_positions(games, move_metrics):
    """Analyze positions from games using material difference"""
    position_evals = []
    
    for game_id, game in games:
        if game_id not in move_metrics:
            print(f"No move metrics found for game {game_id}")
            continue
        
        # Get player information
        white = game.headers.get('White', '')
        black = game.headers.get('Black', '')
        is_v7p3r_white = 'v7p3r' in white.lower()
        
        print(f"Analyzing game {game_id}: {white} vs {black}")
        print(f"Move metrics keys: {list(move_metrics[game_id].keys())[:5]} (showing max 5)")
        
        # Replay the game
        board = chess.Board()
        game_positions = []
        
        # For each move in the game
        for i, move in enumerate(game.mainline_moves()):
            # Calculate move number and color
            move_number = (i // 2) + 1
            color = 'white' if i % 2 == 0 else 'black'
            move_key = f"{move_number}_{color}"
            
            # Make the move
            board.push(move)
            
            # Skip if move metrics not available
            if move_key not in move_metrics[game_id]:
                continue
            
            # Get v7p3r evaluation
            v7p3r_eval = move_metrics[game_id][move_key].get('eval')
            if v7p3r_eval is None:
                continue
            
            # Calculate material difference
            white_material = count_material(board, chess.WHITE)
            black_material = count_material(board, chess.BLACK)
            material_diff = white_material - black_material
            
            # Material difference from v7p3r's perspective
            v7p3r_material = material_diff if is_v7p3r_white else -material_diff
            
            # Calculate evaluation difference
            eval_diff = abs(v7p3r_eval - material_diff)
            
            # Store position evaluation
            game_positions.append({
                'game_id': game_id,
                'move_number': move_number,
                'position': board.fen(),
                'move': move_metrics[game_id][move_key].get('move', ''),
                'v7p3r_eval': v7p3r_eval,
                'material_eval': material_diff,
                'eval_diff': eval_diff,
                'v7p3r_material': v7p3r_material,
                'white_material': white_material,
                'black_material': black_material,
                'move_time': move_metrics[game_id][move_key].get('time', 0)
            })
        
        print(f"  Analyzed {len(game_positions)} positions in game {game_id}")
        position_evals.extend(game_positions)
    
    print(f"Analyzed {len(position_evals)} positions across all games")
    return position_evals

def identify_critical_positions(position_evals, threshold_stdev=2.0):
    """Identify critical positions where v7p3r evaluation differs significantly from material-based evaluation"""
    if not position_evals:
        print("No position evaluations available for analysis")
        return []
    
    # Calculate statistics on evaluation differences
    eval_diffs = [pos['eval_diff'] for pos in position_evals]
    mean_diff = sum(eval_diffs) / len(eval_diffs) if eval_diffs else 0
    
    # Calculate standard deviation
    variance = sum((x - mean_diff) ** 2 for x in eval_diffs) / len(eval_diffs) if len(eval_diffs) > 1 else 0
    stdev_diff = variance ** 0.5
    
    # Positions with eval difference more than threshold_stdev standard deviations from mean
    threshold = mean_diff + (threshold_stdev * stdev_diff)
    
    critical_positions = [
        pos for pos in position_evals 
        if pos['eval_diff'] > threshold
    ]
    
    # Sort by evaluation difference (largest first)
    critical_positions.sort(key=lambda pos: pos['eval_diff'], reverse=True)
    
    print(f"Identified {len(critical_positions)} critical positions")
    print(f"Mean evaluation difference: {mean_diff:.2f}")
    print(f"Standard deviation: {stdev_diff:.2f}")
    print(f"Threshold for critical positions: {threshold:.2f}")
    
    return critical_positions

def generate_charts(position_evals, critical_positions):
    """Generate charts for analysis"""
    if not position_evals:
        return
    
    # Create scatter plot of v7p3r evaluation vs material evaluation
    plt.figure(figsize=(10, 6))
    plt.scatter(
        [pos['v7p3r_eval'] for pos in position_evals], 
        [pos['material_eval'] for pos in position_evals],
        alpha=0.5
    )
    plt.xlabel('V7P3R Evaluation')
    plt.ylabel('Material-based Evaluation')
    plt.title('V7P3R vs Material Evaluation Comparison')
    plt.grid(True)
    plt.savefig('analysis_results/eval_comparison.png')
    plt.close()
    
    # Create histogram of evaluation differences
    plt.figure(figsize=(10, 6))
    plt.hist([pos['eval_diff'] for pos in position_evals], bins=20, alpha=0.7)
    plt.xlabel('Evaluation Difference')
    plt.ylabel('Frequency')
    plt.title('Distribution of Evaluation Differences')
    plt.grid(True)
    plt.savefig('analysis_results/eval_diff_histogram.png')
    plt.close()
    
    # Create chart of move timing
    plt.figure(figsize=(12, 6))
    plt.scatter(
        [pos['move_number'] for pos in position_evals], 
        [pos['move_time'] for pos in position_evals],
        alpha=0.5
    )
    plt.xlabel('Move Number')
    plt.ylabel('Time (seconds)')
    plt.title('Move Timing Analysis')
    plt.grid(True)
    plt.savefig('analysis_results/move_timing.png')
    plt.close()
    
    print(f"Generated charts in analysis_results directory")

def generate_report(games, position_evals, critical_positions):
    """Generate a simple HTML report"""
    if not position_evals:
        return
    
    html_path = os.path.join('analysis_results', 'material_analysis_report.html')
    
    # Create HTML content
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>V7P3R Material Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .section {{ margin-bottom: 30px; }}
            .position {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; background-color: #f9f9f9; }}
            img {{ max-width: 100%; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>V7P3R Chess Engine Material Analysis Report</h1>
        
        <div class="section">
            <h2>Analysis Summary</h2>
            <p>Games Analyzed: {len(games)}</p>
            <p>Positions Analyzed: {len(position_evals)}</p>
            <p>Critical Positions: {len(critical_positions)}</p>
            <p>Average Evaluation Difference: {sum(p['eval_diff'] for p in position_evals) / len(position_evals):.2f}</p>
        </div>
        
        <div class="section">
            <h2>Charts</h2>
            <div>
                <h3>Evaluation Comparison</h3>
                <img src="eval_comparison.png" alt="Evaluation Comparison">
            </div>
            <div>
                <h3>Evaluation Difference Distribution</h3>
                <img src="eval_diff_histogram.png" alt="Evaluation Difference Histogram">
            </div>
            <div>
                <h3>Move Timing Analysis</h3>
                <img src="move_timing.png" alt="Move Timing Analysis">
            </div>
        </div>
        
        <div class="section">
            <h2>Critical Positions</h2>
    '''
    
    # Add top 10 critical positions
    for i, pos in enumerate(critical_positions[:10]):
        html_content += f'''
            <div class="position">
                <h3>Critical Position #{i+1} (Game {pos['game_id']}, Move {pos['move_number']})</h3>
                <p>Position FEN: {pos['position']}</p>
                <p>V7P3R Evaluation: {pos['v7p3r_eval']:.2f}</p>
                <p>Material Evaluation: {pos['material_eval']:.2f}</p>
                <p>Evaluation Difference: <strong>{pos['eval_diff']:.2f}</strong></p>
                <p>White Material: {pos['white_material']:.2f}, Black Material: {pos['black_material']:.2f}</p>
            </div>
        '''
    
    # Add game summary table
    html_content += '''
        <div class="section">
            <h2>Game Summaries</h2>
            <table>
                <tr>
                    <th>Game ID</th>
                    <th>White</th>
                    <th>Black</th>
                    <th>Result</th>
                    <th>Critical Positions</th>
                </tr>
    '''
    
    # Add game rows
    for game_id, game in games:
        critical_count = sum(1 for pos in critical_positions if pos['game_id'] == game_id)
        
        html_content += f'''
            <tr>
                <td>{game_id}</td>
                <td>{game.headers.get('White', 'Unknown')}</td>
                <td>{game.headers.get('Black', 'Unknown')}</td>
                <td>{game.headers.get('Result', 'Unknown')}</td>
                <td>{critical_count}</td>
            </tr>
        '''
    
    # Close HTML
    html_content += '''
            </table>
        </div>
    </body>
    </html>
    '''
    
    # Save HTML report
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Generated HTML report at {html_path}")
    
    # Also save data to JSON
    json_path = os.path.join('analysis_results', 'material_analysis.json')
    
    # Prepare data for JSON export
    export_data = {
        'summary': {
            'total_games': len(games),
            'total_positions': len(position_evals),
            'critical_positions': len(critical_positions),
            'average_eval_diff': sum(p['eval_diff'] for p in position_evals) / len(position_evals) if position_evals else 0
        },
        'critical_positions': critical_positions[:20],  # Limit to 20 for JSON size
    }
    
    # Save to JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Generated JSON report at {json_path}")

if __name__ == "__main__":
    analyze_games_with_material()
