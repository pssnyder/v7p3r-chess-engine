"""
Version-Aware Full Analysis
Analyzes games and tracks which engine version played each game
"""

import sys
import os
from pathlib import Path
import chess.pgn
import json
from datetime import datetime
import logging

# Imports from same directory
from v7p3r_analytics import V7P3RAnalytics
from report_generator import ReportGenerator
from version_tracker import VersionTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)


def count_games_in_pgn(pgn_path: str) -> int:
    """Quickly count games in PGN file."""
    count = 0
    with open(pgn_path, 'r') as f:
        for line in f:
            if line.startswith('[Event '):
                count += 1
    return count


def analyze_all_games_with_versions(
    pgn_path: str,
    output_dir: str,
    limit: int = None
):
    """
    Analyze all games in PGN with version tracking.
    
    Args:
        pgn_path: Path to PGN file
        output_dir: Directory for output files
        limit: Optional limit on number of games to analyze
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    version_tracker = VersionTracker()
    
    # Count games
    print("\nCounting games...")
    total_games = count_games_in_pgn(pgn_path)
    games_to_analyze = min(total_games, limit) if limit else total_games
    
    print(f"Found {total_games} games in file")
    print(f"Will analyze {'all' if not limit else limit} {games_to_analyze} games")
    
    # Analyze games
    analyses = []
    version_stats = {}
    failed_games = []
    
    print("\n" + "=" * 70)
    print("Starting Stockfish Analysis with Version Tracking...")
    print("=" * 70 + "\n")
    
    # Stockfish path
    stockfish_path = "s:/Programming/Chess Engines/Tournament Engines/Stockfish/stockfish-windows-x86-64-avx2.exe"
    
    with V7P3RAnalytics(stockfish_path) as analytics:
        with open(pgn_path) as pgn_file:
            game_num = 0
            
            while game_num < games_to_analyze:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                game_num += 1
                
                # Extract game metadata
                headers = game.headers
                game_id = headers.get("Site", "").split("/")[-1] or f"game_{game_num}"
                white = headers.get("White", "Unknown")
                black = headers.get("Black", "Unknown")
                result = headers.get("Result", "*")
                date_str = headers.get("UTCDate", headers.get("Date", ""))
                
                # Determine engine version for this game
                game_metadata = {"date": date_str}
                version_info = version_tracker.get_version_for_game_id(game_id, game_metadata)
                
                if version_info:
                    version = version_info["version"]
                    version_note = version_info["notes"]
                else:
                    version = "UNKNOWN"
                    version_note = "No version match found"
                
                # Track version statistics
                if version not in version_stats:
                    version_stats[version] = {
                        "count": 0,
                        "notes": version_note,
                        "games": []
                    }
                version_stats[version]["count"] += 1
                version_stats[version]["games"].append(game_id)
                
                # Progress display
                pct = (game_num / games_to_analyze) * 100
                print(f"\n[{game_num}/{games_to_analyze}] ({pct:.1f}%) {game_id} | {white} vs {black} | {result}")
                print(f"  Version: {version} ({version_note})")
                
                # Write game to temp file for analysis
                temp_pgn = output_path / "temp_game.pgn"
                with open(temp_pgn, 'w') as temp_file:
                    print(game, file=temp_file)
                
                # Analyze game
                try:
                    report = analytics.analyze_game(str(temp_pgn))
                    
                    # Add version metadata to report
                    report.metadata["engine_version"] = version
                    report.metadata["version_info"] = version_info
                    
                    analyses.append(report)
                    
                    # Quick stats
                    print(f"    ✓ CPL: {report.average_cpl:.1f} | "
                          f"Moves: {report.total_moves} | "
                          f"Top1: {report.top_1_alignment:.1f}% | "
                          f"Blunders: {report.blunders}")
                    
                except Exception as e:
                    print(f"    ✗ Analysis failed: {e}")
                    failed_games.append({
                        "game_id": game_id,
                        "error": str(e),
                        "version": version
                    })
                
                # Progress checkpoint every 10 games
                if game_num % 10 == 0:
                    print(f"\n--- Checkpoint: {game_num}/{games_to_analyze} complete ---")
                    print(f"Successful: {len(analyses)} | Failed: {len(failed_games)}")
                    print(f"Version breakdown:")
                    for ver, stats in version_stats.items():
                        print(f"  {ver}: {stats['count']} games")
    
    # Clean up temp file
    if temp_pgn.exists():
        temp_pgn.unlink()
    
    # Generate reports with version filtering
    print("\n" + "=" * 70)
    print("Generating Reports...")
    print("=" * 70 + "\n")
    
    generate_reports_by_version(
        analyses,
        output_path,
        version_stats,
        failed_games
    )
    
    print(f"\n{'=' * 70}")
    print("Analysis Complete!")
    print(f"{'=' * 70}\n")
    print(f"Total games analyzed: {len(analyses)}")
    print(f"Failed analyses: {len(failed_games)}")
    print(f"\nVersion breakdown:")
    for ver, stats in version_stats.items():
        print(f"  {ver}: {stats['count']} games - {stats['notes']}")
    print(f"\nReports saved to: {output_path}")


def generate_reports_by_version(
    analyses: list,
    output_path: Path,
    version_stats: dict,
    failed_games: list
):
    """Generate overall report plus per-version reports."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generator = ReportGenerator()
    
    # 1. Overall report (all versions)
    print("\n1. Generating overall report (all versions)...")
    
    overall_report = generator.generate_weekly_report(
        analyses,
        f"v7p3r_bot All Versions Analysis",
        analyses[0].metadata.get("date", "unknown") if analyses else "unknown",
        analyses[-1].metadata.get("date", "unknown") if analyses else "unknown"
    )
    
    # Add version stats to metadata
    overall_report["version_statistics"] = version_stats
    overall_report["failed_games"] = failed_games
    
    # Save overall JSON
    overall_json_path = output_path / f"v7p3r_analysis_all_{timestamp}.json"
    with open(overall_json_path, 'w') as f:
        json.dump(overall_report, f, indent=2)
    print(f"   ✓ Saved: {overall_json_path}")
    
    # Save overall markdown
    overall_md_path = output_path / f"v7p3r_analysis_all_{timestamp}.md"
    with open(overall_md_path, 'w') as f:
        f.write(generator._generate_markdown_summary(overall_report))
    print(f"   ✓ Saved: {overall_md_path}")
    
    # 2. Per-version reports
    print("\n2. Generating per-version reports...")
    
    for version, stats in version_stats.items():
        if stats["count"] == 0:
            continue
        
        # Filter analyses for this version
        version_analyses = [
            a for a in analyses
            if a.metadata.get("engine_version") == version
        ]
        
        if not version_analyses:
            continue
        
        print(f"\n   Version {version}: {len(version_analyses)} games")
        
        # Generate report
        version_report = generator.generate_weekly_report(
            version_analyses,
            f"v7p3r_bot {version} Analysis",
            version_analyses[0].metadata.get("date", "unknown"),
            version_analyses[-1].metadata.get("date", "unknown")
        )
        
        # Add version metadata
        version_report["engine_version"] = version
        version_report["version_notes"] = stats["notes"]
        version_report["game_ids"] = stats["games"]
        
        # Save JSON
        safe_version = version.replace(".", "_")
        version_json_path = output_path / f"v7p3r_analysis_{safe_version}_{timestamp}.json"
        with open(version_json_path, 'w') as f:
            json.dump(version_report, f, indent=2)
        print(f"     ✓ JSON: {version_json_path.name}")
        
        # Save markdown
        version_md_path = output_path / f"v7p3r_analysis_{safe_version}_{timestamp}.md"
        with open(version_md_path, 'w') as f:
            f.write(generator._generate_markdown_summary(version_report))
        print(f"     ✓ MD: {version_md_path.name}")
    
    # 3. Version comparison summary
    print("\n3. Generating version comparison...")
    comparison_path = output_path / f"version_comparison_{timestamp}.md"
    
    with open(comparison_path, 'w') as f:
        f.write("# V7P3R Engine Version Comparison\n\n")
        f.write(f"Analysis Period: {analyses[0].metadata.get('date', 'unknown')} to {analyses[-1].metadata.get('date', 'unknown')}\n\n")
        f.write(f"Total Games Analyzed: {len(analyses)}\n\n")
        f.write("---\n\n")
        
        f.write("## Version Performance Summary\n\n")
        f.write("| Version | Games | Avg CPL | Win Rate | Top1 Align | Blunders/Game | Notes |\n")
        f.write("|---------|-------|---------|----------|------------|---------------|-------|\n")
        
        for version in sorted(version_stats.keys()):
            version_analyses = [a for a in analyses if a.metadata.get("engine_version") == version]
            if not version_analyses:
                continue
            
            avg_cpl = sum(a.average_cpl for a in version_analyses) / len(version_analyses)
            wins = sum(1 for a in version_analyses if a.metadata.get("result") in ["1-0", "0-1"] and "v7p3r" in a.metadata.get("result", ""))
            win_rate = (wins / len(version_analyses)) * 100 if version_analyses else 0
            avg_top1 = sum(a.top_1_alignment for a in version_analyses) / len(version_analyses)
            avg_blunders = sum(a.blunders for a in version_analyses) / len(version_analyses)
            
            notes = version_stats[version]["notes"]
            f.write(f"| {version} | {len(version_analyses)} | {avg_cpl:.1f} | {win_rate:.1f}% | {avg_top1:.1f}% | {avg_blunders:.1f} | {notes} |\n")
        
        f.write("\n---\n\n")
        f.write("## Detailed Analysis by Version\n\n")
        
        for version in sorted(version_stats.keys()):
            f.write(f"### {version}\n\n")
            f.write(f"**Notes**: {version_stats[version]['notes']}\n\n")
            f.write(f"**Games Analyzed**: {version_stats[version]['count']}\n\n")
            
            version_analyses = [a for a in analyses if a.metadata.get("engine_version") == version]
            if version_analyses:
                avg_cpl = sum(a.average_cpl for a in version_analyses) / len(version_analyses)
                f.write(f"**Average CPL**: {avg_cpl:.1f}\n\n")
            
            f.write("---\n\n")
    
    print(f"   ✓ Saved: {comparison_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze PGN games with version tracking")
    parser.add_argument("pgn_file", help="Path to PGN file")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit number of games to analyze")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("V7P3R Analytics - Version-Aware Full Game Analysis")
    print("=" * 70 + "\n")
    
    print(f"PGN File: {args.pgn_file}")
    
    if not os.path.exists(args.pgn_file):
        print(f"ERROR: File not found: {args.pgn_file}")
        sys.exit(1)
    
    file_size_mb = os.path.getsize(args.pgn_file) / (1024 * 1024)
    print(f"File Size: {file_size_mb:.1f} MB")
    
    analyze_all_games_with_versions(
        args.pgn_file,
        args.output_dir,
        args.limit
    )
