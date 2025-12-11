#!/usr/bin/env python3
"""
Analyze V7P3R Regression Tournament Results
Processes PGN files from Arena tournaments to extract version-specific strengths/weaknesses
"""

import chess.pgn
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List
import sys

class RegressionTournamentAnalyzer:
    """Analyze self-play tournament results to identify version strengths/weaknesses"""
    
    def __init__(self):
        self.version_stats = defaultdict(lambda: {
            'total_games': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'wins_as_white': 0,
            'wins_as_black': 0,
            'losses_as_white': 0,
            'losses_as_black': 0,
            'opponents_beaten': defaultdict(int),
            'opponents_lost_to': defaultdict(int),
            'total_moves': [],
            'game_lengths': [],
            'openings': defaultdict(int),
            'results_by_opponent': defaultdict(lambda: {'W': 0, 'L': 0, 'D': 0})
        })
        
        self.head_to_head = defaultdict(lambda: defaultdict(lambda: {'white_wins': 0, 'black_wins': 0, 'draws': 0}))
        
    def extract_version(self, player_name: str) -> str:
        """Extract version number from player name like 'V7P3R_v17.8'"""
        if 'v7p3r' in player_name.lower():
            # Extract version number
            parts = player_name.split('_')
            for part in parts:
                if part.startswith('v') and any(c.isdigit() for c in part):
                    return part
        return player_name
    
    def analyze_pgn(self, pgn_path: Path) -> None:
        """Analyze all games in a PGN file"""
        print(f"Analyzing: {pgn_path.name}")
        
        with open(pgn_path) as pgn_file:
            game_count = 0
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                game_count += 1
                self._analyze_game(game)
        
        print(f"  Processed {game_count} games")
    
    def _analyze_game(self, game: chess.pgn.Game) -> None:
        """Analyze a single game"""
        white = game.headers.get('White', 'Unknown')
        black = game.headers.get('Black', 'Unknown')
        result = game.headers.get('Result', '*')
        opening = game.headers.get('Opening', 'Unknown')
        
        white_version = self.extract_version(white)
        black_version = self.extract_version(black)
        
        # Count moves
        board = game.board()
        move_count = 0
        for move in game.mainline_moves():
            board.push(move)
            move_count += 1
        
        # Update stats for both versions
        for version, color, opponent in [
            (white_version, 'white', black_version),
            (black_version, 'black', white_version)
        ]:
            stats = self.version_stats[version]
            stats['total_games'] += 1
            stats['game_lengths'].append(move_count)
            stats['openings'][opening] += 1
            
            # Determine result from this version's perspective
            if result == '1-0':  # White won
                if color == 'white':
                    stats['wins'] += 1
                    stats['wins_as_white'] += 1
                    stats['opponents_beaten'][opponent] += 1
                    stats['results_by_opponent'][opponent]['W'] += 1
                else:
                    stats['losses'] += 1
                    stats['losses_as_black'] += 1
                    stats['opponents_lost_to'][opponent] += 1
                    stats['results_by_opponent'][opponent]['L'] += 1
            elif result == '0-1':  # Black won
                if color == 'black':
                    stats['wins'] += 1
                    stats['wins_as_black'] += 1
                    stats['opponents_beaten'][opponent] += 1
                    stats['results_by_opponent'][opponent]['W'] += 1
                else:
                    stats['losses'] += 1
                    stats['losses_as_white'] += 1
                    stats['opponents_lost_to'][opponent] += 1
                    stats['results_by_opponent'][opponent]['L'] += 1
            elif result == '1/2-1/2':  # Draw
                stats['draws'] += 1
                stats['results_by_opponent'][opponent]['D'] += 1
        
        # Head-to-head tracking
        h2h = self.head_to_head[white_version][black_version]
        if result == '1-0':
            h2h['white_wins'] += 1
        elif result == '0-1':
            h2h['black_wins'] += 1
        elif result == '1/2-1/2':
            h2h['draws'] += 1
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'versions_analyzed': len(self.version_stats),
            'total_games': sum(s['total_games'] for s in self.version_stats.values()) // 2,  # Divide by 2 (counted twice)
            'version_rankings': [],
            'version_details': {},
            'head_to_head_matrix': {},
            'insights': []
        }
        
        # Calculate win rates and rank versions
        version_scores = []
        for version, stats in self.version_stats.items():
            total = stats['total_games']
            if total == 0:
                continue
            
            win_rate = stats['wins'] / total
            loss_rate = stats['losses'] / total
            draw_rate = stats['draws'] / total
            
            # Calculate color balance
            white_games = stats['wins_as_white'] + stats['losses_as_white']
            black_games = stats['wins_as_black'] + stats['losses_as_black']
            white_win_rate = stats['wins_as_white'] / white_games if white_games > 0 else 0
            black_win_rate = stats['wins_as_black'] / black_games if black_games > 0 else 0
            
            # Average game length
            avg_game_length = sum(stats['game_lengths']) / len(stats['game_lengths']) if stats['game_lengths'] else 0
            
            version_scores.append({
                'version': version,
                'score': stats['wins'] + (stats['draws'] * 0.5),
                'games': total,
                'win_rate': win_rate,
                'loss_rate': loss_rate,
                'draw_rate': draw_rate,
                'white_win_rate': white_win_rate,
                'black_win_rate': black_win_rate,
                'color_balance': white_win_rate - black_win_rate,  # Positive = better as white
                'avg_game_length': avg_game_length
            })
        
        # Sort by score (wins + 0.5*draws)
        version_scores.sort(key=lambda x: x['score'], reverse=True)
        report['version_rankings'] = version_scores
        
        # Detailed stats per version
        for version, stats in self.version_stats.items():
            report['version_details'][version] = {
                'total_games': stats['total_games'],
                'record': f"{stats['wins']}-{stats['losses']}-{stats['draws']}",
                'strongest_against': dict(sorted(stats['opponents_beaten'].items(), key=lambda x: x[1], reverse=True)[:5]),
                'weakest_against': dict(sorted(stats['opponents_lost_to'].items(), key=lambda x: x[1], reverse=True)[:5]),
                'favorite_openings': dict(sorted(stats['openings'].items(), key=lambda x: x[1], reverse=True)[:5]),
                'head_to_head_records': dict(stats['results_by_opponent'])
            }
        
        # Head-to-head matrix
        for v1 in sorted(self.head_to_head.keys()):
            report['head_to_head_matrix'][v1] = {}
            for v2 in sorted(self.head_to_head[v1].keys()):
                h2h = self.head_to_head[v1][v2]
                total = h2h['white_wins'] + h2h['black_wins'] + h2h['draws']
                report['head_to_head_matrix'][v1][v2] = {
                    'as_white': f"{h2h['white_wins']}-{h2h['draws']}-{h2h['black_wins']}",
                    'total_games': total,
                    'win_rate_as_white': h2h['white_wins'] / total if total > 0 else 0
                }
        
        # Generate insights
        report['insights'] = self._generate_insights(version_scores, report['version_details'])
        
        return report
    
    def _generate_insights(self, rankings: List[Dict], details: Dict) -> List[str]:
        """Generate key insights from the data"""
        insights = []
        
        if len(rankings) >= 3:
            top3 = rankings[:3]
            insights.append(f"Top 3 performers: {', '.join([v['version'] + f' ({v['score']:.1f} pts)' for v in top3])}")
            
            # Color balance analysis
            best_white = max(rankings, key=lambda x: x['white_win_rate'])
            best_black = max(rankings, key=lambda x: x['black_win_rate'])
            
            if best_white['version'] != best_black['version']:
                insights.append(f"{best_white['version']} dominates as White ({best_white['white_win_rate']:.1%}), "
                              f"{best_black['version']} excels as Black ({best_black['black_win_rate']:.1%})")
            
            # Biggest color imbalance
            worst_balance = max(rankings, key=lambda x: abs(x['color_balance']))
            if abs(worst_balance['color_balance']) > 0.15:
                side = "White" if worst_balance['color_balance'] > 0 else "Black"
                insights.append(f"{worst_balance['version']} has significant color imbalance (better as {side})")
            
            # Most draws
            most_draws = max(rankings, key=lambda x: x['draw_rate'])
            if most_draws['draw_rate'] > 0.3:
                insights.append(f"{most_draws['version']} draws frequently ({most_draws['draw_rate']:.1%} of games)")
            
            # Game length patterns
            longest_games = max(rankings, key=lambda x: x['avg_game_length'])
            shortest_games = min(rankings, key=lambda x: x['avg_game_length'])
            insights.append(f"Game length: {longest_games['version']} plays longest ({longest_games['avg_game_length']:.1f} moves), "
                          f"{shortest_games['version']} shortest ({shortest_games['avg_game_length']:.1f} moves)")
        
        return insights
    
    def print_summary(self, report: Dict) -> None:
        """Print human-readable summary"""
        print("\n" + "="*80)
        print("V7P3R REGRESSION TOURNAMENT ANALYSIS")
        print("="*80)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Versions Analyzed: {report['versions_analyzed']}")
        print(f"Total Games: {report['total_games']}")
        
        print("\n" + "-"*80)
        print("VERSION RANKINGS")
        print("-"*80)
        print(f"{'Rank':<6} {'Version':<12} {'Score':<8} {'W-L-D':<12} {'Win%':<8} {'Draw%':<8} {'Avg Moves':<10}")
        print("-"*80)
        
        for i, v in enumerate(report['version_rankings'], 1):
            record = f"{int(v['score'] * 2 - v['games'] * v['draw_rate'])}-" \
                    f"{int(v['loss_rate'] * v['games'])}-" \
                    f"{int(v['draw_rate'] * v['games'])}"
            print(f"{i:<6} {v['version']:<12} {v['score']:<8.1f} {record:<12} "
                  f"{v['win_rate']:<8.1%} {v['draw_rate']:<8.1%} {v['avg_game_length']:<10.1f}")
        
        print("\n" + "-"*80)
        print("COLOR PERFORMANCE")
        print("-"*80)
        print(f"{'Version':<12} {'White Win%':<12} {'Black Win%':<12} {'Balance':<10}")
        print("-"*80)
        
        for v in report['version_rankings']:
            balance = v['color_balance']
            balance_str = f"+{balance:.1%}" if balance > 0 else f"{balance:.1%}"
            print(f"{v['version']:<12} {v['white_win_rate']:<12.1%} {v['black_win_rate']:<12.1%} {balance_str:<10}")
        
        print("\n" + "-"*80)
        print("KEY INSIGHTS")
        print("-"*80)
        for insight in report['insights']:
            print(f"• {insight}")
        
        print("\n" + "-"*80)
        print("VERSION STRENGTHS & WEAKNESSES")
        print("-"*80)
        
        for version in sorted(report['version_details'].keys()):
            details = report['version_details'][version]
            print(f"\n{version} ({details['record']})")
            
            if details['strongest_against']:
                print(f"  ✓ Strongest vs: {', '.join([f'{k} ({v}W)' for k, v in list(details['strongest_against'].items())[:3]])}")
            
            if details['weakest_against']:
                print(f"  ✗ Weakest vs: {', '.join([f'{k} ({v}L)' for k, v in list(details['weakest_against'].items())[:3]])}")


def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze V7P3R regression tournament PGN files')
    parser.add_argument('pgn_files', nargs='+', help='PGN files to analyze')
    parser.add_argument('--output', '-o', help='Output JSON file for detailed results')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    analyzer = RegressionTournamentAnalyzer()
    
    # Analyze all provided PGN files
    for pgn_path_str in args.pgn_files:
        pgn_path = Path(pgn_path_str)
        if pgn_path.exists():
            analyzer.analyze_pgn(pgn_path)
        else:
            print(f"Warning: File not found: {pgn_path}")
    
    # Generate report
    report = analyzer.generate_report()
    
    # Print summary
    if not args.quiet:
        analyzer.print_summary(report)
    
    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Detailed report saved to: {output_path}")
    
    return report


if __name__ == '__main__':
    main()
