#!/usr/bin/env python3
"""
Full Analysis Script - Analyzes all games in a PGN file
Creates comprehensive report with all games since v17.4 deployment
"""
import chess.pgn
from pathlib import Path
from datetime import datetime
import sys
from v7p3r_analytics import V7P3RAnalytics
from report_generator import ReportGenerator
import json

# Configuration
STOCKFISH_PATH = "s:/Programming/Chess Engines/Tournament Engines/Stockfish/stockfish-windows-x86-64-avx2.exe"
OUTPUT_DIR = "reports"

def count_games_in_pgn(pgn_file):
    """Quick count of games in PGN file."""
    count = 0
    with open(pgn_file) as f:
        for line in f:
            if line.startswith('[Event '):
                count += 1
    return count

def analyze_all_games(pgn_file, max_games=None, bot_username="v7p3r_bot"):
    """
    Analyze all games in a PGN file.
    
    Args:
        pgn_file: Path to PGN file
        max_games: Maximum number of games to analyze (None = all)
        bot_username: Bot's username
    
    Returns:
        List of GameAnalysisReport objects
    """
    pgn_path = Path(pgn_file)
    
    if not pgn_path.exists():
        print(f"Error: PGN file not found: {pgn_file}")
        return []
    
    # Count total games
    print("="*70)
    print("V7P3R Analytics - Full Game Analysis")
    print("="*70)
    print(f"\nPGN File: {pgn_path}")
    print(f"File Size: {pgn_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print("\nCounting games...")
    total_games = count_games_in_pgn(pgn_path)
    print(f"Found {total_games} games in file")
    
    if max_games:
        games_to_analyze = min(max_games, total_games)
        print(f"Will analyze first {games_to_analyze} games (limit: {max_games})")
    else:
        games_to_analyze = total_games
        print(f"Will analyze all {games_to_analyze} games")
    
    # Analyze games
    print(f"\n{'='*70}")
    print("Starting Stockfish Analysis...")
    print(f"{'='*70}\n")
    
    reports = []
    failed_games = []
    
    with V7P3RAnalytics(STOCKFISH_PATH, bot_username) as analytics:
        with open(pgn_path) as pgn:
            for i in range(games_to_analyze):
                game = chess.pgn.read_game(pgn)
                if not game:
                    print(f"\nReached end of file at game {i}")
                    break
                
                # Get game info
                game_id = game.headers.get("GameId", "unknown")
                white = game.headers.get("White", "")
                black = game.headers.get("Black", "")
                result = game.headers.get("Result", "*")
                date = game.headers.get("UTCDate", "unknown")
                
                # Progress indicator
                progress = (i + 1) / games_to_analyze * 100
                print(f"[{i+1}/{games_to_analyze}] ({progress:.1f}%) {game_id} | {white} vs {black} | {result}")
                
                # Save game to temp file for analysis
                temp_pgn = Path("temp_game.pgn")
                with open(temp_pgn, 'w') as out:
                    print(game, file=out)
                
                # Analyze
                try:
                    report = analytics.analyze_game(str(temp_pgn))
                    if report:
                        reports.append(report)
                        print(f"  ✓ CPL: {report.average_centipawn_loss:.1f} | "
                              f"Moves: {len(report.moves)} | "
                              f"Top1: {report.top1_alignment:.1f}% | "
                              f"Blunders: {report.blunders + report.critical_blunders}")
                    else:
                        failed_games.append((i+1, game_id, "Analysis returned None"))
                        print(f"  ✗ Analysis failed")
                except Exception as e:
                    failed_games.append((i+1, game_id, str(e)))
                    print(f"  ✗ Error: {e}")
                
                # Cleanup temp file
                if temp_pgn.exists():
                    temp_pgn.unlink()
                
                # Progress update every 10 games
                if (i + 1) % 10 == 0:
                    print(f"\n--- Progress: {i+1}/{games_to_analyze} games analyzed ({len(reports)} successful) ---\n")
    
    # Summary
    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"{'='*70}")
    print(f"\nSuccessful: {len(reports)}/{games_to_analyze}")
    if failed_games:
        print(f"Failed: {len(failed_games)}")
        print("\nFailed games:")
        for game_num, game_id, error in failed_games[:5]:
            print(f"  {game_num}. {game_id}: {error}")
        if len(failed_games) > 5:
            print(f"  ... and {len(failed_games) - 5} more")
    
    return reports

def generate_reports(reports, output_dir="reports"):
    """Generate JSON and Markdown reports."""
    if not reports:
        print("\nNo reports to generate")
        return None
    
    print(f"\n{'='*70}")
    print("Generating Reports...")
    print(f"{'='*70}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"v7p3r_analysis_{timestamp}.json"
    
    # Generate report
    generator = ReportGenerator()
    weekly_report = generator.generate_weekly_report(reports, str(report_file))
    
    print(f"✓ JSON Report: {report_file}")
    print(f"✓ Markdown Report: {report_file.with_suffix('.md')}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("Report Summary")
    print(f"{'='*70}\n")
    print(f"Games Analyzed: {weekly_report['games']['total']}")
    print(f"Record: {weekly_report['games']['wins']}W-{weekly_report['games']['losses']}L-{weekly_report['games']['draws']}D")
    print(f"Win Rate: {weekly_report['games']['win_rate']}%")
    print(f"\nPerformance:")
    print(f"  Avg CPL: {weekly_report['performance']['avg_centipawn_loss']:.1f}")
    print(f"  Best Moves: {weekly_report['performance']['move_quality']['best']}")
    print(f"  Blunders: {weekly_report['performance']['move_quality']['blunders']}")
    print(f"  Critical Blunders: {weekly_report['performance']['move_quality']['critical_blunders']}")
    print(f"\nStockfish Alignment:")
    print(f"  Top 1: {weekly_report['performance']['stockfish_alignment']['top1']:.1f}%")
    print(f"  Top 3: {weekly_report['performance']['stockfish_alignment']['top3']:.1f}%")
    print(f"  Top 5: {weekly_report['performance']['stockfish_alignment']['top5']:.1f}%")
    
    print(f"\n{'='*70}")
    print(f"View full report: cat {report_file.with_suffix('.md')}")
    print(f"{'='*70}\n")
    
    return report_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze all games in PGN file")
    parser.add_argument("pgn_file", help="Path to PGN file")
    parser.add_argument("--max-games", type=int, help="Maximum games to analyze")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--bot-username", default="v7p3r_bot", help="Bot username")
    
    args = parser.parse_args()
    
    # Analyze all games
    reports = analyze_all_games(
        args.pgn_file,
        max_games=args.max_games,
        bot_username=args.bot_username
    )
    
    # Generate reports
    if reports:
        generate_reports(reports, output_dir=args.output_dir)
    else:
        print("\nNo games were successfully analyzed")
        sys.exit(1)
