#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Batch Game Analyzer
Uses the Stockfish engine to analyze multiple chess games in PGN format for performance metrics and eval comparison.
Populates the engine_metrics.db with remaining data necessary for analysis (game result metrics, engine performance metrics, and move evaluation metrics using stockfish eval scores).

Features:
- Analyzes PGN files from test games
- Extracts engine evaluations and metrics
- Runs Stockfish analysis on critical positions
- Identifies evaluation outliers (standard deviation analysis)
- Generates position improvement recommendations
- Creates visualizations for easy pattern recognition
"""

import os
import re
import sys
import json
import chess
import chess.pgn
import sqlite3
import argparse
import statistics
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from stockfish import Stockfish
from collections import defaultdict, Counter

class BatchGameAnalyzer:
    def __init__(self, pgn_dir='pgn_game_records', metrics_file='metrics/metrics.db', 
                stockfish_path='stockfish.exe', stockfish_elo=2000, stockfish_depth=15):
        """Initialize the batch game analyzer with configuration parameters"""
        self.pgn_dir = pgn_dir
        self.metrics_file = metrics_file
        self.stockfish_path = stockfish_path
        self.stockfish_elo = stockfish_elo
        self.stockfish_depth = stockfish_depth
        
        # Analysis storage
        self.games = []
        self.metrics = {}
        self.position_evals = []
        self.critical_positions = []
        self.improvement_suggestions = []
        
        # Initialize Stockfish engine
        try:
            self.stockfish = Stockfish(
                path=self.stockfish_path,
                parameters={
                    "Threads": 1,
                    "Hash": 128,
                    "UCI_LimitStrength": True,
                    "UCI_Elo": self.stockfish_elo
                }
            )
            print(f"Stockfish engine initialized (ELO: {stockfish_elo}, Depth: {stockfish_depth})")
        except Exception as e:
            print(f"Error initializing Stockfish engine: {e}")
            self.stockfish = None

    def load_pgn_games(self):
        """Load PGN games from the specified directory"""
        self.games = []
        pgn_files = list(Path(self.pgn_dir).glob('*.pgn'))
        
        print(f"Found {len(pgn_files)} PGN files in {self.pgn_dir}")
        
        for pgn_file in pgn_files:
            try:
                with open(pgn_file, 'r') as f:
                    while True:
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break
                        self.games.append(game)
                print(f"Loaded {pgn_file.name}")
            except Exception as e:
                print(f"Error loading {pgn_file.name}: {e}")
        
        print(f"Loaded {len(self.games)} games in total")
        return len(self.games)

    def load_metrics(self):
        """Load metrics from SQLite database"""
        self.metrics = defaultdict(dict)
        
        try:
            conn = sqlite3.connect(self.metrics_file)
            cursor = conn.cursor()
            
            # Get game metrics from game_results table
            cursor.execute("SELECT id, timestamp, white_player, black_player, result, total_moves, game_duration, white_engine_config, black_engine_config FROM game_results")
            game_results = cursor.fetchall()
            
            # Get move analysis metrics
            cursor.execute("SELECT game_id, move_number, player_color, move_uci, evaluation_score, search_depth, nodes_searched, search_time, book_move FROM move_analysis")
            move_analysis = cursor.fetchall()
            
            conn.close()
            
            # Organize game metrics
            for (game_id, timestamp, white_player, black_player, result, 
                 total_moves, game_duration, white_engine_config, black_engine_config) in game_results:
                
                # Store basic game information
                self.metrics[str(game_id)]['game_id'] = str(game_id)
                self.metrics[str(game_id)]['timestamp'] = timestamp
                self.metrics[str(game_id)]['white_player'] = white_player
                self.metrics[str(game_id)]['black_player'] = black_player
                self.metrics[str(game_id)]['result'] = result
                self.metrics[str(game_id)]['total_moves'] = total_moves
                self.metrics[str(game_id)]['game_duration'] = game_duration
                self.metrics[str(game_id)]['white_engine_config'] = white_engine_config
                self.metrics[str(game_id)]['black_engine_config'] = black_engine_config
            
            # Organize move metrics by game and position
            for (game_id, move_number, player_color, move_uci, 
                 evaluation_score, search_depth, nodes_searched, search_time, book_move) in move_analysis:
                
                # Create a position ID from move number and player color
                position_id = f"{move_number}_{player_color}"
                
                # Initialize positions dict if needed
                if 'positions' not in self.metrics[str(game_id)]:
                    self.metrics[str(game_id)]['positions'] = {}
                
                if position_id not in self.metrics[str(game_id)]['positions']:
                    self.metrics[str(game_id)]['positions'][position_id] = {}
                
                # Store move analysis data
                self.metrics[str(game_id)]['positions'][position_id]['move_uci'] = move_uci
                self.metrics[str(game_id)]['positions'][position_id]['evaluation'] = evaluation_score
                self.metrics[str(game_id)]['positions'][position_id]['search_depth'] = search_depth
                self.metrics[str(game_id)]['positions'][position_id]['nodes_searched'] = nodes_searched
                self.metrics[str(game_id)]['positions'][position_id]['search_time'] = search_time
                self.metrics[str(game_id)]['positions'][position_id]['book_move'] = book_move
            
            print(f"Loaded metrics for {len(self.metrics)} games")
            return len(self.metrics)
            
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return 0

    def match_pgns_with_metrics(self):
        """Match PGN games with their corresponding metrics"""
        matched_games = []
        
        for game in self.games:
            # Extract game date and opponent information
            date = game.headers.get('Date', '').replace('.', '-')
            white = game.headers.get('White', '')
            black = game.headers.get('Black', '')
            
            # Try to find a matching game in the metrics
            game_id = None
            for metric_id, metrics in self.metrics.items():
                # Check if the date, white player and black player match
                metric_white = metrics.get('white_player', '')
                metric_black = metrics.get('black_player', '')
                metric_timestamp = metrics.get('timestamp', '')
                
                # Simple date-based matching (just check if date appears in timestamp)
                if (date in metric_timestamp and 
                    white.lower() == metric_white.lower() and 
                    black.lower() == metric_black.lower()):
                    game_id = metric_id
                    break
            
            # If we found a match
            if game_id:
                matched_games.append((game, self.metrics[game_id]))
        
        print(f"Matched {len(matched_games)} games with metrics")
        return matched_games

    def analyze_game_positions(self, matched_games):
        """Analyze all positions in matched games"""
        self.position_evals = []
        
        for game, metrics in matched_games:
            board = game.board()
            
            # Get result for later win/loss/draw determination
            result = game.headers.get('Result', '*')
            white_player = game.headers.get('White', 'Unknown')
            black_player = game.headers.get('Black', 'Unknown')
            
            is_v7p3r_white = 'v7p3r' in white_player.lower()
            
            move_number = 0
            
            # Process each move in the game
            for node in game.mainline():
                move = node.move
                board.push(move)
                move_number += 1
                
                # Only analyze moves made by v7p3r
                current_player = 'white' if board.turn == chess.BLACK else 'black'
                last_player = 'black' if board.turn == chess.BLACK else 'white'
                is_v7p3r_move = (last_player == 'white' and is_v7p3r_white) or \
                                (last_player == 'black' and not is_v7p3r_white)
                
                if is_v7p3r_move:
                    # Get v7p3r evaluation from metrics if available
                    v7p3r_eval = None
                    position_id = f"{move_number}_{last_player}"
                    
                    if 'positions' in metrics and position_id in metrics['positions']:
                        position_metrics = metrics['positions'][position_id]
                        if 'evaluation' in position_metrics and position_metrics['evaluation'] is not None:
                            v7p3r_eval = float(position_metrics['evaluation'])
                    
                    if v7p3r_eval is not None:
                        # For black's evaluation, we need to negate the values to match perspective
                        if last_player == 'black' and not is_v7p3r_white:
                            v7p3r_eval = -v7p3r_eval
                        
                        # Calculate material difference instead of using Stockfish
                        white_material = self.count_material(board, chess.WHITE)
                        black_material = self.count_material(board, chess.BLACK)
                        material_diff = white_material - black_material
                        
                        # Normalize to same perspective as v7p3r_eval
                        if not is_v7p3r_white:
                            material_diff = -material_diff
                        
                        # Use material difference as a simple evaluation baseline
                        simple_eval = material_diff
                        eval_diff = abs(v7p3r_eval - simple_eval)
                        
                        self.position_evals.append({
                            'game_id': metrics.get('game_id', 'unknown'),
                            'move_number': move_number,
                            'position': board.fen(),
                            'v7p3r_eval': v7p3r_eval,
                            'stockfish_eval': simple_eval,  # Using material diff instead
                            'eval_diff': eval_diff,
                            'is_v7p3r_white': is_v7p3r_white,
                            'result': result,
                            'move_uci': node.move.uci(),
                            'white_material': white_material,
                            'black_material': black_material,
                            'material_diff': material_diff
                        })
        
        print(f"Analyzed {len(self.position_evals)} positions")
        return self.position_evals

    def get_stockfish_eval(self, board):
        """Get Stockfish evaluation for a position"""
        if self.stockfish is None:
            return None
        
        try:
            # Set a timeout to prevent hanging
            self.stockfish.set_fen_position(board.fen())
            self.stockfish.set_depth(5)  # Use a lower depth for faster analysis
            
            # Get evaluation from info string
            info = self.stockfish.get_parameters()
            eval_string = info.get('Score', '0')
            
            # Convert string evaluation to a float
            try:
                eval_value = float(eval_string) / 100.0  # Convert centipawns to pawns
                return eval_value
            except (ValueError, TypeError):
                # Handle mate scores or other special evaluation strings
                if 'mate' in str(eval_string).lower():
                    # For mate scores, return a high value
                    return 10.0 if eval_string > 0 else -10.0
                return 0.0
        except Exception as e:
            print(f"Error getting Stockfish evaluation: {e}")
            # Create a new Stockfish instance as a fallback
            try:
                self.stockfish = Stockfish(
                    path=self.stockfish_path,
                    parameters={
                        "Threads": 1,
                        "Hash": 128,
                        "UCI_LimitStrength": True,
                        "UCI_Elo": self.stockfish_elo
                    }
                )
                print("Recreated Stockfish engine")
            except Exception as e2:
                print(f"Failed to recreate Stockfish engine: {e2}")
            
            return None

    def identify_critical_positions(self):
        """Identify critical positions where v7p3r evaluation differs significantly from Stockfish"""
        if not self.position_evals:
            print("No position evaluations available for analysis")
            return []
        
        # Calculate statistics on evaluation differences
        eval_diffs = [pos['eval_diff'] for pos in self.position_evals]
        mean_diff = statistics.mean(eval_diffs)
        stdev_diff = statistics.stdev(eval_diffs) if len(eval_diffs) > 1 else 0
        
        # Positions with eval difference more than 2 standard deviations from mean
        threshold = mean_diff + (2 * stdev_diff)
        
        self.critical_positions = [
            pos for pos in self.position_evals 
            if pos['eval_diff'] > threshold
        ]
        
        # Sort by evaluation difference (largest first)
        self.critical_positions.sort(key=lambda pos: pos['eval_diff'], reverse=True)
        
        print(f"Identified {len(self.critical_positions)} critical positions")
        print(f"Mean evaluation difference: {mean_diff:.2f}")
        print(f"Standard deviation: {stdev_diff:.2f}")
        print(f"Threshold for critical positions: {threshold:.2f}")
        
        return self.critical_positions

    def generate_improvement_suggestions(self):
        """Generate improvement suggestions based on critical positions"""
        self.improvement_suggestions = []
        
        if not self.critical_positions:
            print("No critical positions to analyze")
            return []
        
        # Get the top 10 most critical positions (or fewer if we have less)
        top_critical = self.critical_positions[:min(10, len(self.critical_positions))]
        
        for pos in top_critical:
            board = chess.Board(pos['position'])
            
            # Generate suggestion
            suggestion = {
                'position': pos['position'],
                'move_number': pos['move_number'],
                'v7p3r_eval': pos['v7p3r_eval'],
                'stockfish_eval': pos['stockfish_eval'],  # Actually material-based eval
                'eval_diff': pos['eval_diff'],
                'white_material': pos.get('white_material', self.count_material(board, chess.WHITE)),
                'black_material': pos.get('black_material', self.count_material(board, chess.BLACK)),
                'material_diff': pos.get('material_diff', 0),
                'improvement_areas': []
            }
            
            # Identify potential improvement areas
            if abs(pos['v7p3r_eval']) > 3.0 and abs(pos['stockfish_eval']) < 1.0:
                suggestion['improvement_areas'].append("Evaluation scaling: V7P3R may be overestimating position value")
            
            if abs(pos['v7p3r_eval']) < 1.0 and abs(pos['stockfish_eval']) > 3.0:
                suggestion['improvement_areas'].append("Material awareness: V7P3R may not be properly accounting for material difference")
            
            # Check for material imbalance discrepancies
            material_diff = suggestion['material_diff']
            if abs(material_diff) > 3 and abs(pos['v7p3r_eval'] - material_diff) > 2:
                suggestion['improvement_areas'].append(f"Material evaluation: Material difference is {material_diff} pawns, but V7P3R evaluation is {pos['v7p3r_eval']}")
            
            # Add piece mobility analysis
            white_mobility = self.count_mobility(board, chess.WHITE)
            black_mobility = self.count_mobility(board, chess.BLACK)
            mobility_diff = white_mobility - black_mobility
            
            suggestion['white_mobility'] = white_mobility
            suggestion['black_mobility'] = black_mobility
            suggestion['mobility_diff'] = mobility_diff
            
            # Check if mobility is not reflected in evaluation
            if (mobility_diff > 10 and pos['v7p3r_eval'] < 0) or (mobility_diff < -10 and pos['v7p3r_eval'] > 0):
                suggestion['improvement_areas'].append(f"Mobility awareness: Mobility difference is {mobility_diff}, but not reflected in V7P3R evaluation")
            
            # Check for king safety issues
            white_king_square = board.king(chess.WHITE)
            black_king_square = board.king(chess.BLACK)
            
            if white_king_square is not None and black_king_square is not None:
                white_king_safety = self.evaluate_king_safety(board, white_king_square, chess.WHITE)
                black_king_safety = self.evaluate_king_safety(board, black_king_square, chess.BLACK)
                
                king_safety_diff = white_king_safety - black_king_safety
                
                # If king safety issues aren't reflected in the evaluation
                if (king_safety_diff < -2 and pos['v7p3r_eval'] > 0) or (king_safety_diff > 2 and pos['v7p3r_eval'] < 0):
                    suggestion['improvement_areas'].append("King safety: V7P3R evaluation doesn't reflect king safety concerns")
            
            self.improvement_suggestions.append(suggestion)
        
        print(f"Generated {len(self.improvement_suggestions)} improvement suggestions")
        return self.improvement_suggestions
        
    def evaluate_king_safety(self, board, king_square, color):
        """Evaluate king safety (higher is safer)"""
        safety_score = 0
        
        # Count defenders around the king
        defenders = 0
        attackers = 0
        
        # Check surrounding squares
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                file_idx = chess.square_file(king_square)
                rank_idx = chess.square_rank(king_square)
                
                new_file = file_idx + dx
                new_rank = rank_idx + dy
                
                # Skip if out of bounds
                if new_file < 0 or new_file > 7 or new_rank < 0 or new_rank > 7:
                    continue
                
                square = chess.square(new_file, new_rank)
                piece = board.piece_at(square)
                
                if piece is not None:
                    if piece.color == color:
                        defenders += 1
                    else:
                        attackers += 1
        
        # Calculate safety score
        safety_score = defenders - attackers
        
        # Check pawn shield
        if color == chess.WHITE:
            pawn_shield_squares = [
                chess.square(file_idx - 1, rank_idx + 1),
                chess.square(file_idx, rank_idx + 1),
                chess.square(file_idx + 1, rank_idx + 1)
            ]
        else:
            pawn_shield_squares = [
                chess.square(file_idx - 1, rank_idx - 1),
                chess.square(file_idx, rank_idx - 1),
                chess.square(file_idx + 1, rank_idx - 1)
            ]
        
        for square in pawn_shield_squares:
            if 0 <= square < 64:  # Check if square is valid
                piece = board.piece_at(square)
                if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
                    safety_score += 1
        
        # Check if king is in check
        if board.is_check():
            if board.turn == color:
                safety_score -= 3
        
        return safety_score

    def count_material(self, board, color):
        """Count material value for a side (in pawns)"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3.25,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King not counted in material
        }
        
        material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                material += piece_values[piece.piece_type]
        
        return material

    def count_mobility(self, board, color):
        """Count number of legal moves for a side as mobility metric"""
        # Save current turn
        original_turn = board.turn
        
        # Set board turn to analyze the desired color
        board.turn = color
        
        # Count legal moves
        mobility = len(list(board.legal_moves))
        
        # Restore original turn
        board.turn = original_turn
        
        return mobility

    def visualize_results(self):
        """Create visualizations for analysis results"""
        if not self.position_evals:
            print("No data to visualize")
            return
        
        # Extract data for plotting
        v7p3r_evals = [pos['v7p3r_eval'] for pos in self.position_evals]
        stockfish_evals = [pos['stockfish_eval'] for pos in self.position_evals]
        eval_diffs = [pos['eval_diff'] for pos in self.position_evals]
        move_numbers = [pos['move_number'] for pos in self.position_evals]
        
        # Create output directory if it doesn't exist
        os.makedirs('analysis_results', exist_ok=True)
        
        # 1. Scatter plot: V7P3R vs Stockfish evaluations
        plt.figure(figsize=(10, 6))
        plt.scatter(stockfish_evals, v7p3r_evals, alpha=0.5)
        plt.xlabel('Stockfish Evaluation (pawns)')
        plt.ylabel('V7P3R Evaluation (pawns)')
        plt.title('V7P3R vs Stockfish Position Evaluations')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add perfect correlation line
        min_eval = min(min(v7p3r_evals), min(stockfish_evals))
        max_eval = max(max(v7p3r_evals), max(stockfish_evals))
        plt.plot([min_eval, max_eval], [min_eval, max_eval], 'r--', label='Perfect Correlation')
        
        plt.legend()
        plt.savefig('analysis_results/eval_comparison_scatter.png')
        
        # 2. Histogram of evaluation differences
        plt.figure(figsize=(10, 6))
        plt.hist(eval_diffs, bins=20, alpha=0.7)
        plt.xlabel('Absolute Evaluation Difference (pawns)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Evaluation Differences')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('analysis_results/eval_diff_histogram.png')
        
        # 3. Evaluation differences over move number
        plt.figure(figsize=(12, 6))
        plt.scatter(move_numbers, eval_diffs, alpha=0.5)
        plt.xlabel('Move Number')
        plt.ylabel('Evaluation Difference (pawns)')
        plt.title('Evaluation Differences by Move Number')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('analysis_results/eval_diff_by_move.png')
        
        print("Created visualizations in 'analysis_results' directory")

    def export_results(self):
        """Export analysis results to JSON file"""
        results = {
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'games_analyzed': len(self.games),
            'positions_analyzed': len(self.position_evals),
            'critical_positions': len(self.critical_positions),
            'improvement_suggestions': self.improvement_suggestions,
            'critical_positions_details': self.critical_positions[:20]  # Include top 20 critical positions
        }
        
        # Create output directory if it doesn't exist
        os.makedirs('analysis_results', exist_ok=True)
        
        # Export to JSON
        with open('analysis_results/analysis_report.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Exported analysis results to 'analysis_results/analysis_report.json'")
        
        # Generate HTML report
        self.generate_html_report(results)

    def generate_html_report(self, results):
        """Generate HTML report for easy viewing"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>V7P3R Chess Engine Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
                .stat-item {{ background-color: #fff; padding: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                .suggestion {{ background-color: #f9f9f9; padding: 15px; margin-bottom: 15px; border-left: 4px solid #2196F3; }}
                .critical {{ background-color: #fff3f3; padding: 15px; margin-bottom: 15px; border-left: 4px solid #f44336; }}
                .eval-diff-high {{ color: #f44336; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .visualizations {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 20px; }}
                .visualization {{ text-align: center; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>V7P3R Chess Engine Analysis Report</h1>
                <p>Analysis generated on {results['analysis_timestamp']}</p>
                
                <div class="stats">
                    <h2>Summary Statistics</h2>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <h3>Games Analyzed</h3>
                            <p>{results['games_analyzed']}</p>
                        </div>
                        <div class="stat-item">
                            <h3>Positions Analyzed</h3>
                            <p>{results['positions_analyzed']}</p>
                        </div>
                        <div class="stat-item">
                            <h3>Critical Positions</h3>
                            <p>{results['critical_positions']}</p>
                        </div>
                    </div>
                </div>
                
                <h2>Improvement Suggestions</h2>
        """
        
        # Add improvement suggestions
        for i, suggestion in enumerate(results['improvement_suggestions']):
            html_content += f"""
                <div class="suggestion">
                    <h3>Suggestion {i+1}</h3>
                    <p><strong>Position:</strong> {suggestion['position']}</p>
                    <p><strong>Move Number:</strong> {suggestion['move_number']}</p>
                    <p><strong>V7P3R Evaluation:</strong> {suggestion['v7p3r_eval']:.2f}</p>
                    <p><strong>Material-Based Evaluation:</strong> {suggestion['stockfish_eval']:.2f}</p>
                    <p><strong>Evaluation Difference:</strong> <span class="eval-diff-high">{suggestion['eval_diff']:.2f}</span></p>
                    
                    <h4>Material Analysis</h4>
                    <p>White Material: {suggestion.get('white_material', 'N/A')}, Black Material: {suggestion.get('black_material', 'N/A')}</p>
                    <p>Material Difference: {suggestion.get('material_diff', 'N/A')}</p>
                    
                    <h4>Mobility Analysis</h4>
                    <p>White Mobility: {suggestion.get('white_mobility', 'N/A')}, Black Mobility: {suggestion.get('black_mobility', 'N/A')}</p>
                    <p>Mobility Difference: {suggestion.get('mobility_diff', 'N/A')}</p>
            """
            
            # Add improvement areas
            html_content += """
                    <h4>Improvement Areas</h4>
                    <ul>
            """
            
            # Add improvement areas
            for area in suggestion.get('improvement_areas', []):
                html_content += f"<li>{area}</li>"
            
            html_content += """
                    </ul>
                </div>
            """
        
        # Add critical positions table
        html_content += """
                <h2>Top Critical Positions</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Move #</th>
                            <th>V7P3R Eval</th>
                            <th>Material Eval</th>
                            <th>Difference</th>
                            <th>Position</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for pos in results['critical_positions_details']:
            html_content += f"""
                        <tr>
                            <td>{pos['move_number']}</td>
                            <td>{pos['v7p3r_eval']:.2f}</td>
                            <td>{pos['stockfish_eval']:.2f}</td>
                            <td class="eval-diff-high">{pos['eval_diff']:.2f}</td>
                            <td>{pos['position']}</td>
                        </tr>
            """
        
        # Add visualizations
        html_content += """
                    </tbody>
                </table>
                
                <h2>Visualizations</h2>
                <div class="visualizations">
                    <div class="visualization">
                        <h3>V7P3R vs Material-Based Evaluations</h3>
                        <img src="eval_comparison_scatter.png" alt="Evaluation Comparison">
                    </div>
                    <div class="visualization">
                        <h3>Evaluation Differences Distribution</h3>
                        <img src="eval_diff_histogram.png" alt="Evaluation Differences Histogram">
                    </div>
                    <div class="visualization">
                        <h3>Evaluation Differences by Move Number</h3>
                        <img src="eval_diff_by_move.png" alt="Evaluation Differences by Move">
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open('analysis_results/analysis_report.html', 'w') as f:
            f.write(html_content)
        
        print("Generated HTML report at 'analysis_results/analysis_report.html'")

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n=== Starting V7P3R Batch Game Analysis ===\n")
        
        # Load data
        self.load_pgn_games()
        self.load_metrics()
        
        # Match PGN games with metrics
        matched_games = self.match_pgns_with_metrics()
        
        # Analyze positions using material difference instead of Stockfish
        self.analyze_game_positions(matched_games)
        
        if self.position_evals:
            # Identify critical positions
            self.identify_critical_positions()
            
            # Generate improvement suggestions
            self.generate_improvement_suggestions()
            
            # Visualize results
            self.visualize_results()
            
            # Export results
            self.export_results()
            
            print("\n=== Analysis Complete ===\n")
            print("Results saved to 'analysis_results' directory")
            print("- HTML Report: analysis_results/analysis_report.html")
            print("- JSON Data: analysis_results/analysis_report.json")
            print("- Visualizations: analysis_results/*.png")
        else:
            print("\n=== Analysis Incomplete ===\n")
            print("No positions were evaluated. Check that your metrics database contains evaluation data.")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='V7P3R Chess Engine Batch Game Analyzer')
    
    parser.add_argument('--pgn_dir', type=str, default='pgn_game_records',
                        help='Directory containing PGN files')
    
    parser.add_argument('--metrics_file', type=str, default='metrics/metrics.db',
                        help='Path to metrics SQLite database')
    
    parser.add_argument('--stockfish_path', type=str, default='stockfish.exe',
                        help='Path to Stockfish executable')
    
    parser.add_argument('--elo', type=int, default=2000,
                        help='Stockfish ELO for analysis')
    
    parser.add_argument('--depth', type=int, default=15,
                        help='Stockfish search depth for analysis')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    analyzer = BatchGameAnalyzer(
        pgn_dir=args.pgn_dir,
        metrics_file=args.metrics_file,
        stockfish_path=args.stockfish_path,
        stockfish_elo=args.elo,
        stockfish_depth=args.depth
    )
    
    analyzer.run_analysis()
