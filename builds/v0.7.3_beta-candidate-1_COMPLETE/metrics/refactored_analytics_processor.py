#!/usr/bin/env python3
"""
Refactored Analytics Server for V7P3R Chess Engine
Processes metrics from the new dataset-based enhanced metrics system
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import os

class RefactoredAnalyticsProcessor:
    """
    Analytics processor for the refactored enhanced metrics system
    """
    
    def __init__(self, db_path: str = "metrics/chess_metrics_v2.db", logger: Optional[logging.Logger] = None):
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        
        # Ensure database exists
        if not os.path.exists(self.db_path):
            self.logger.warning(f"Database not found at {self.db_path}")
            
    def get_recent_games(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent games from the database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT game_id, white_player, black_player, result, total_moves, 
                       game_duration, created_at
                FROM games 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            
            games = []
            for row in cursor.fetchall():
                games.append({
                    'game_id': row[0],
                    'white_player': row[1],
                    'black_player': row[2],
                    'result': row[3],
                    'total_moves': row[4],
                    'game_duration': row[5],
                    'created_at': row[6]
                })
            
            conn.close()
            return games
            
        except Exception as e:
            self.logger.error(f"Error getting recent games: {e}")
            return []
    
    def get_game_analysis(self, game_id: str) -> Dict[str, Any]:
        """
        Get comprehensive analysis for a specific game
        """
        analysis = {
            'game_info': {},
            'move_analysis': [],
            'performance_stats': {},
            'scoring_breakdown': {},
            'search_efficiency': {}
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get game info
            cursor.execute("""
                SELECT game_id, white_player, black_player, result, total_moves, 
                       game_duration, termination, created_at
                FROM games 
                WHERE game_id = ?
            """, (game_id,))
            
            game_row = cursor.fetchone()
            if game_row:
                analysis['game_info'] = {
                    'game_id': game_row[0],
                    'white_player': game_row[1],
                    'black_player': game_row[2],
                    'result': game_row[3],
                    'total_moves': game_row[4],
                    'game_duration': game_row[5],
                    'termination': game_row[6],
                    'created_at': game_row[7]
                }
            
            # Get all moves for this game
            cursor.execute("""
                SELECT move_number, player_color, move_uci, move_san, time_taken,
                       search_algorithm, depth_reached, nodes_searched, nps,
                       total_score, material_balance, game_phase, position_type,
                       fen_before, fen_after
                FROM move_metrics 
                WHERE game_id = ?
                ORDER BY move_number
            """, (game_id,))
            
            moves = cursor.fetchall()
            for move in moves:
                analysis['move_analysis'].append({
                    'move_number': move[0],
                    'player_color': move[1],
                    'move_uci': move[2],
                    'move_san': move[3],
                    'time_taken': move[4],
                    'search_algorithm': move[5],
                    'depth_reached': move[6],
                    'nodes_searched': move[7],
                    'nps': move[8],
                    'total_score': move[9],
                    'material_balance': move[10],
                    'game_phase': move[11],
                    'position_type': move[12],
                    'fen_before': move[13],
                    'fen_after': move[14]
                })
            
            # Calculate performance statistics
            if moves:
                times = [m[4] for m in moves if m[4] is not None]
                scores = [m[9] for m in moves if m[9] is not None]
                nodes = [m[7] for m in moves if m[7] is not None and m[7] > 0]
                
                analysis['performance_stats'] = {
                    'avg_time_per_move': np.mean(times) if times else 0,
                    'max_time_per_move': max(times) if times else 0,
                    'min_time_per_move': min(times) if times else 0,
                    'avg_score': np.mean(scores) if scores else 0,
                    'score_variance': np.var(scores) if len(scores) > 1 else 0,
                    'avg_nodes_searched': np.mean(nodes) if nodes else 0,
                    'total_nodes_searched': sum(nodes) if nodes else 0
                }
            
            # Get detailed scoring breakdown for the first few moves
            cursor.execute("""
                SELECT material_score, center_control_score, pawn_structure_score,
                       king_safety_score, piece_activity_score, mobility_score
                FROM move_metrics 
                WHERE game_id = ? AND move_number <= 5
                ORDER BY move_number
            """, (game_id,))
            
            scoring_data = cursor.fetchall()
            if scoring_data:
                analysis['scoring_breakdown'] = {
                    'material_scores': [s[0] for s in scoring_data if s[0] is not None],
                    'center_control_scores': [s[1] for s in scoring_data if s[1] is not None],
                    'pawn_structure_scores': [s[2] for s in scoring_data if s[2] is not None],
                    'king_safety_scores': [s[3] for s in scoring_data if s[3] is not None],
                    'piece_activity_scores': [s[4] for s in scoring_data if s[4] is not None],
                    'mobility_scores': [s[5] for s in scoring_data if s[5] is not None]
                }
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error analyzing game {game_id}: {e}")
            
        return analysis
    
    def get_engine_performance_comparison(self) -> Dict[str, Any]:
        """
        Compare performance between different engines
        """
        comparison = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get performance stats by engine
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN player_color = 'white' THEN g.white_player 
                        ELSE g.black_player 
                    END as engine_name,
                    AVG(m.time_taken) as avg_time,
                    AVG(m.nodes_searched) as avg_nodes,
                    AVG(m.total_score) as avg_score,
                    COUNT(*) as total_moves
                FROM move_metrics m
                JOIN games g ON m.game_id = g.game_id
                WHERE m.time_taken IS NOT NULL
                GROUP BY engine_name
                ORDER BY engine_name
            """)
            
            engines = cursor.fetchall()
            for engine in engines:
                comparison[engine[0]] = {
                    'avg_time_per_move': engine[1],
                    'avg_nodes_searched': engine[2],
                    'avg_evaluation_score': engine[3],
                    'total_moves_played': engine[4]
                }
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error comparing engine performance: {e}")
            
        return comparison
    
    def get_scoring_component_analysis(self, game_id: Optional[str] = None, 
                                     limit: int = 100) -> Dict[str, Any]:
        """
        Analyze individual scoring components across games or for a specific game
        """
        analysis = {
            'component_averages': {},
            'component_ranges': {},
            'component_correlations': {},
            'move_progression': {}
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query based on whether we're analyzing a specific game
            if game_id:
                query = """
                    SELECT move_number, material_score, center_control_score, 
                           pawn_structure_score, king_safety_score, piece_activity_score,
                           mobility_score, total_score, game_phase
                    FROM move_metrics 
                    WHERE game_id = ?
                    ORDER BY move_number
                """
                params = (game_id,)
            else:
                query = """
                    SELECT move_number, material_score, center_control_score, 
                           pawn_structure_score, king_safety_score, piece_activity_score,
                           mobility_score, total_score, game_phase
                    FROM move_metrics 
                    ORDER BY id DESC
                    LIMIT ?
                """
                params = (limit,)
            
            # Use pandas for easier data analysis
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                # Component averages
                numeric_columns = ['material_score', 'center_control_score', 'pawn_structure_score',
                                 'king_safety_score', 'piece_activity_score', 'mobility_score', 'total_score']
                
                for col in numeric_columns:
                    if col in df.columns:
                        analysis['component_averages'][col] = {
                            'mean': df[col].mean(),
                            'median': df[col].median(),
                            'std': df[col].std()
                        }
                        
                        analysis['component_ranges'][col] = {
                            'min': df[col].min(),
                            'max': df[col].max(),
                            'range': df[col].max() - df[col].min()
                        }
                
                # Game phase progression
                if 'game_phase' in df.columns and 'move_number' in df.columns:
                    phase_progression = df.groupby('game_phase')['total_score'].agg(['mean', 'count'])
                    analysis['move_progression'] = phase_progression.to_dict()
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error analyzing scoring components: {e}")
            
        return analysis
    
    def get_search_efficiency_metrics(self) -> Dict[str, Any]:
        """
        Analyze search efficiency across all games
        """
        efficiency = {
            'time_distribution': {},
            'nodes_distribution': {},
            'nps_analysis': {},
            'depth_analysis': {}
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Time distribution
            cursor.execute("""
                SELECT 
                    COUNT(*) as count,
                    CASE 
                        WHEN time_taken < 0.1 THEN 'very_fast'
                        WHEN time_taken < 0.5 THEN 'fast'
                        WHEN time_taken < 2.0 THEN 'medium'
                        WHEN time_taken < 10.0 THEN 'slow'
                        ELSE 'very_slow'
                    END as time_category
                FROM move_metrics 
                WHERE time_taken IS NOT NULL
                GROUP BY time_category
            """)
            
            time_dist = cursor.fetchall()
            efficiency['time_distribution'] = {row[1]: row[0] for row in time_dist}
            
            # Nodes per second analysis
            cursor.execute("""
                SELECT AVG(nps) as avg_nps, MIN(nps) as min_nps, MAX(nps) as max_nps,
                       COUNT(*) as total_moves
                FROM move_metrics 
                WHERE nps IS NOT NULL AND nps > 0
            """)
            
            nps_row = cursor.fetchone()
            if nps_row:
                efficiency['nps_analysis'] = {
                    'average_nps': nps_row[0],
                    'min_nps': nps_row[1],
                    'max_nps': nps_row[2],
                    'total_moves_with_nps': nps_row[3]
                }
            
            # Depth analysis
            cursor.execute("""
                SELECT depth_reached, COUNT(*) as count, AVG(time_taken) as avg_time
                FROM move_metrics 
                WHERE depth_reached IS NOT NULL
                GROUP BY depth_reached
                ORDER BY depth_reached
            """)
            
            depth_data = cursor.fetchall()
            efficiency['depth_analysis'] = {
                f'depth_{row[0]}': {'count': row[1], 'avg_time': row[2]}
                for row in depth_data
            }
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error analyzing search efficiency: {e}")
            
        return efficiency
    
    def generate_performance_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive performance report
        """
        report_lines = []
        report_lines.append("V7P3R Chess Engine - Performance Analysis Report")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Recent games summary
        recent_games = self.get_recent_games(10)
        report_lines.append(f"Recent Games ({len(recent_games)} games):")
        report_lines.append("-" * 30)
        
        for game in recent_games[:5]:  # Show top 5
            duration = game['game_duration'] if game['game_duration'] is not None else 0.0
            report_lines.append(f"  {game['game_id']}: {game['white_player']} vs {game['black_player']}")
            report_lines.append(f"    Result: {game['result']} | Moves: {game['total_moves']} | Duration: {duration:.1f}s")
        
        report_lines.append("")
        
        # Engine performance comparison
        engine_comparison = self.get_engine_performance_comparison()
        if engine_comparison:
            report_lines.append("Engine Performance Comparison:")
            report_lines.append("-" * 35)
            
            for engine, stats in engine_comparison.items():
                avg_time = stats.get('avg_time_per_move', 0) or 0
                avg_nodes = stats.get('avg_nodes_searched', 0) or 0
                avg_score = stats.get('avg_evaluation_score', 0) or 0
                total_moves = stats.get('total_moves_played', 0) or 0
                
                report_lines.append(f"  {engine}:")
                report_lines.append(f"    Avg time/move: {avg_time:.3f}s")
                report_lines.append(f"    Avg nodes: {avg_nodes:.0f}")
                report_lines.append(f"    Avg score: {avg_score:.1f}")
                report_lines.append(f"    Total moves: {total_moves}")
                report_lines.append("")
        
        # Search efficiency
        search_efficiency = self.get_search_efficiency_metrics()
        if search_efficiency.get('time_distribution'):
            report_lines.append("Search Time Distribution:")
            report_lines.append("-" * 25)
            
            for category, count in search_efficiency['time_distribution'].items():
                report_lines.append(f"  {category}: {count} moves")
            
            if search_efficiency.get('nps_analysis'):
                nps = search_efficiency['nps_analysis']
                if nps.get('average_nps') is not None:
                    report_lines.append(f"\nNodes Per Second:")
                    report_lines.append(f"  Average: {nps['average_nps']:.0f} NPS")
                    if nps.get('min_nps') is not None and nps.get('max_nps') is not None:
                        report_lines.append(f"  Range: {nps['min_nps']:.0f} - {nps['max_nps']:.0f} NPS")
            
            report_lines.append("")
        
        # Scoring component analysis
        scoring_analysis = self.get_scoring_component_analysis()
        if scoring_analysis.get('component_averages'):
            report_lines.append("Scoring Component Analysis:")
            report_lines.append("-" * 30)
            
            for component, stats in scoring_analysis['component_averages'].items():
                mean_val = stats.get('mean', 0) or 0
                std_val = stats.get('std', 0) or 0
                if mean_val != 0:  # Only show non-zero components
                    report_lines.append(f"  {component}:")
                    report_lines.append(f"    Mean: {mean_val:.2f} | Std: {std_val:.2f}")
        
        report = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report)
                if self.logger:
                    self.logger.info(f"Performance report saved to {output_file}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to save report: {e}")
        
        return report

def main():
    """
    Test the refactored analytics processor
    """
    print("Testing Refactored Analytics Processor")
    print("=" * 40)
    
    # Initialize processor
    processor = RefactoredAnalyticsProcessor()
    
    # Test recent games
    print("Recent Games:")
    recent_games = processor.get_recent_games(5)
    for game in recent_games:
        print(f"  {game['game_id']}: {game['white_player']} vs {game['black_player']} = {game['result']}")
    
    # Test game analysis
    if recent_games:
        game_id = recent_games[0]['game_id']
        print(f"\nAnalyzing game: {game_id}")
        analysis = processor.get_game_analysis(game_id)
        
        print(f"Game info: {analysis['game_info']}")
        print(f"Moves analyzed: {len(analysis['move_analysis'])}")
        print(f"Performance stats: {analysis['performance_stats']}")
    
    # Test engine comparison
    print(f"\nEngine Performance Comparison:")
    comparison = processor.get_engine_performance_comparison()
    for engine, stats in comparison.items():
        print(f"  {engine}: {stats['avg_time_per_move']:.3f}s avg, {stats['total_moves_played']} moves")
    
    # Generate full report
    print(f"\n" + "="*40)
    print("FULL PERFORMANCE REPORT:")
    print("="*40)
    report = processor.generate_performance_report()
    print(report)

if __name__ == "__main__":
    main()
