#!/usr/bin/env python3
"""
Quick test script - Analyzes a few games from the PGN file
"""
import chess.pgn
from pathlib import Path
from v7p3r_analytics import V7P3RAnalytics
from report_generator import ReportGenerator

# Configuration
STOCKFISH_PATH = "s:/Programming/Chess Engines/Tournament Engines/Stockfish/stockfish-windows-x86-64-avx2.exe"
PGN_FILE = "test_workspace/pgn_downloads/lichess_v7p3r_bot_2025-11-30.pgn"
OUTPUT_DIR = "test_workspace/reports"
MAX_GAMES = 5  # Analyze first 5 games only

print("="*60)
print("V7P3R Analytics - Quick Test")
print("="*60)
print(f"\nAnalyzing first {MAX_GAMES} games from {PGN_FILE}")
print(f"Using Stockfish: {STOCKFISH_PATH}")
print("")

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Extract first N games to separate PGN files
print(f"Extracting first {MAX_GAMES} games...")
game_files = []
with open(PGN_FILE) as pgn:
    for i in range(MAX_GAMES):
        game = chess.pgn.read_game(pgn)
        if not game:
            print(f"  Only found {i} games in file")
            break
        
        game_id = game.headers.get("GameId", f"game_{i}")
        game_file = Path("test_workspace") / f"game_{game_id}.pgn"
        
        with open(game_file, 'w') as out:
            print(game, file=out)
        
        game_files.append(game_file)
        print(f"  Extracted: {game_id}")

print(f"\n{'='*60}")
print("Analyzing games with Stockfish...")
print(f"{'='*60}\n")

# Analyze each game
reports = []
with V7P3RAnalytics(STOCKFISH_PATH) as analytics:
    for i, game_file in enumerate(game_files, 1):
        print(f"[{i}/{len(game_files)}] Analyzing {game_file.name}...")
        
        report = analytics.analyze_game(str(game_file))
        if report:
            reports.append(report)
            print(f"  ✓ {report.opponent} | {report.result} | CPL: {report.average_centipawn_loss:.1f}")
            print(f"    Moves: {report.best_moves}B {report.excellent_moves}E {report.good_moves}G "
                  f"{report.inaccuracies}I {report.mistakes}M {report.blunders}BL {report.critical_blunders}CB")
        else:
            print(f"  ✗ Analysis failed")
        print("")

# Generate report
if reports:
    print(f"{'='*60}")
    print("Generating Weekly Report...")
    print(f"{'='*60}\n")
    
    generator = ReportGenerator()
    report_path = Path(OUTPUT_DIR) / "test_report.json"
    
    weekly_report = generator.generate_weekly_report(reports, str(report_path))
    
    print(f"✓ Report saved to {report_path}")
    print(f"✓ Markdown: {report_path.with_suffix('.md')}")
    print("")
    print("Summary:")
    print(f"  Games: {weekly_report['games']['total']}")
    print(f"  Win Rate: {weekly_report['games']['win_rate']}%")
    print(f"  Avg CPL: {weekly_report['performance']['avg_centipawn_loss']}")
    print(f"  Top 1 Alignment: {weekly_report['performance']['stockfish_alignment']['top1']}%")
    print("")
    print(f"View full report: cat {report_path.with_suffix('.md')}")
else:
    print("No games were successfully analyzed")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)
