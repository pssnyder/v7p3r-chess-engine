"""
Parallel V7P3R Analytics - Multi-process game analysis
Designed for cloud deployment with rapid scale-up and scale-down
"""

import sys
import os
from pathlib import Path
import chess.pgn
import json
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time

from v7p3r_analytics import V7P3RAnalytics, GameAnalysisReport
from report_generator import ReportGenerator
from version_tracker import VersionTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_single_game(game_data: dict) -> dict:
    """
    Analyze a single game in a separate process.
    Each process gets its own Stockfish instance.
    
    Args:
        game_data: Dict with 'pgn_text', 'game_id', 'game_num', 'total_games', 'version_info'
    
    Returns:
        Dict with analysis results or error info
    """
    game_num = game_data['game_num']
    total = game_data['total_games']
    game_id = game_data['game_id']
    pgn_text = game_data['pgn_text']
    version_info = game_data['version_info']
    stockfish_path = game_data['stockfish_path']
    
    try:
        # Create temp PGN file for this game
        temp_dir = Path("temp_analysis")
        temp_dir.mkdir(exist_ok=True)
        temp_pgn = temp_dir / f"game_{game_id}.pgn"
        
        with open(temp_pgn, 'w') as f:
            f.write(pgn_text)
        
        # Analyze with dedicated Stockfish instance
        with V7P3RAnalytics(stockfish_path) as analytics:
            report = analytics.analyze_game(str(temp_pgn))
        
        # Clean up temp file
        temp_pgn.unlink()
        
        # Convert report to dict and add version info
        result = {
            'success': True,
            'game_id': game_id,
            'game_num': game_num,
            'report': report,
            'version': version_info['version'] if version_info else 'UNKNOWN',
            'version_info': version_info
        }
        
        logger.info(f"[{game_num}/{total}] {game_id} | CPL: {report.average_centipawn_loss:.1f} | Top1: {report.top1_alignment:.1f}% | Version: {result['version']}")
        
        return result
        
    except Exception as e:
        logger.error(f"[{game_num}/{total}] {game_id} FAILED: {e}")
        return {
            'success': False,
            'game_id': game_id,
            'game_num': game_num,
            'error': str(e),
            'version': version_info['version'] if version_info else 'UNKNOWN'
        }


def extract_games_from_pgn(pgn_path: str, limit: int = None) -> list:
    """
    Extract all games from PGN file into memory.
    
    Args:
        pgn_path: Path to PGN file
        limit: Optional limit on number of games
    
    Returns:
        List of dicts with game data
    """
    games = []
    version_tracker = VersionTracker()
    
    logger.info(f"Extracting games from {pgn_path}...")
    
    with open(pgn_path) as pgn_file:
        game_num = 0
        while True:
            if limit and game_num >= limit:
                break
                
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            game_num += 1
            
            # Extract metadata
            headers = game.headers
            game_id = headers.get("Site", "").split("/")[-1] or f"game_{game_num}"
            date_str = headers.get("UTCDate", headers.get("Date", ""))
            
            # Determine version
            game_metadata = {"date": date_str}
            version_info = version_tracker.get_version_for_game_id(game_id, game_metadata)
            
            # Store game as PGN text
            pgn_text = str(game)
            
            games.append({
                'game_id': game_id,
                'game_num': game_num,
                'pgn_text': pgn_text,
                'version_info': version_info,
                'headers': dict(headers)
            })
    
    logger.info(f"Extracted {len(games)} games")
    return games


def parallel_analyze_games(
    pgn_path: str,
    output_dir: str,
    max_workers: int = None,
    limit: int = None,
    stockfish_path: str = None
):
    """
    Analyze games in parallel using multiple processes.
    
    Args:
        pgn_path: Path to PGN file
        output_dir: Directory for output
        max_workers: Number of parallel workers (default: CPU count - 1)
        limit: Optional limit on number of games
        stockfish_path: Path to Stockfish executable
    """
    start_time = time.time()
    
    # Set up
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if stockfish_path is None:
        stockfish_path = "s:/Programming/Chess Engines/Tournament Engines/Stockfish/stockfish-windows-x86-64-avx2.exe"
    
    if max_workers is None:
        # Use CPU count - 1 to leave one core for system
        max_workers = max(1, cpu_count() - 1)
    
    logger.info("=" * 70)
    logger.info("V7P3R Parallel Analytics")
    logger.info("=" * 70)
    logger.info(f"PGN File: {pgn_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Workers: {max_workers}")
    logger.info(f"CPU Count: {cpu_count()}")
    logger.info("=" * 70)
    
    # Extract all games first (fast)
    games = extract_games_from_pgn(pgn_path, limit)
    total_games = len(games)
    
    if total_games == 0:
        logger.error("No games found!")
        return
    
    logger.info(f"\nAnalyzing {total_games} games with {max_workers} parallel workers...")
    logger.info("=" * 70 + "\n")
    
    # Add metadata to each game
    for game in games:
        game['total_games'] = total_games
        game['stockfish_path'] = stockfish_path
    
    # Parallel analysis
    analyses = []
    failed_games = []
    version_stats = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = {executor.submit(analyze_single_game, game): game for game in games}
        
        # Process results as they complete
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            
            if result['success']:
                analyses.append(result['report'])
                
                # Track version stats
                version = result['version']
                if version not in version_stats:
                    version_stats[version] = {
                        'count': 0,
                        'notes': result['version_info']['notes'] if result['version_info'] else 'Unknown',
                        'games': []
                    }
                version_stats[version]['count'] += 1
                version_stats[version]['games'].append(result['game_id'])
            else:
                failed_games.append({
                    'game_id': result['game_id'],
                    'error': result['error'],
                    'version': result['version']
                })
            
            # Progress update every 10 games
            if completed % 10 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (total_games - completed) / rate if rate > 0 else 0
                logger.info(f"\nProgress: {completed}/{total_games} ({completed/total_games*100:.1f}%)")
                logger.info(f"Success: {len(analyses)} | Failed: {len(failed_games)}")
                logger.info(f"Rate: {rate:.1f} games/sec | ETA: {remaining/60:.1f} min")
    
    # Clean up temp directory
    temp_dir = Path("temp_analysis")
    if temp_dir.exists():
        for temp_file in temp_dir.glob("*.pgn"):
            temp_file.unlink()
        temp_dir.rmdir()
    
    # Generate reports
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info(f"Analysis Complete in {elapsed/60:.1f} minutes!")
    logger.info("=" * 70)
    logger.info(f"Total: {total_games} | Success: {len(analyses)} | Failed: {len(failed_games)}")
    logger.info(f"Average Rate: {total_games/elapsed:.2f} games/sec")
    logger.info("\nGenerating Reports...")
    
    generate_reports_by_version(
        analyses,
        output_path,
        version_stats,
        failed_games,
        elapsed
    )
    
    logger.info(f"\nAll reports saved to: {output_path}")
    logger.info("=" * 70)


def generate_reports_by_version(
    analyses: list,
    output_path: Path,
    version_stats: dict,
    failed_games: list,
    elapsed_time: float
):
    """Generate overall and per-version reports."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generator = ReportGenerator()
    
    # 1. Overall report
    logger.info("\n1. Overall report (all versions)...")
    
    overall_json = output_path / f"v7p3r_analysis_all_{timestamp}.json"
    overall_report = generator.generate_weekly_report(
        analyses,
        str(overall_json)
    )
    
    # Add metadata
    overall_report["version_statistics"] = version_stats
    overall_report["failed_games"] = failed_games
    overall_report["processing_time_seconds"] = elapsed_time
    overall_report["processing_time_minutes"] = elapsed_time / 60
    overall_report["analysis_rate_games_per_second"] = len(analyses) / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"   Saved: {overall_json.name}")
    
    # Save Markdown
    overall_md = output_path / f"v7p3r_analysis_all_{timestamp}.md"
    with open(overall_md, 'w') as f:
        # Generate custom markdown since _generate_markdown_summary needs output_path
        f.write(f"# V7P3R Analysis Report - All Versions\n\n")
        f.write(f"**Total Games**: {len(analyses)}\n\n")
        f.write(f"**Failed**: {len(failed_games)}\n\n")
        f.write(f"**Processing Time**: {elapsed_time/60:.1f} minutes\n\n")
        f.write(f"**Analysis Rate**: {len(analyses)/elapsed_time:.2f} games/second\n\n")
        f.write(f"## Version Breakdown\n\n")
        for version, stats in version_stats.items():
            f.write(f"- **{version}**: {stats['count']} games - {stats['notes']}\n")
    logger.info(f"   Saved: {overall_md.name}")
    
    # 2. Per-version reports
    logger.info("\n2. Per-version reports...")
    
    for version, stats in version_stats.items():
        if stats["count"] == 0:
            continue
        
        # Filter analyses for this version
        version_analyses = [
            a for a in analyses
            if get_game_version(a, version_stats) == version
        ]
        
        if not version_analyses:
            continue
        
        logger.info(f"   {version}: {len(version_analyses)} games")
        
        # Save files
        safe_version = version.replace(".", "_")
        version_json = output_path / f"v7p3r_analysis_{safe_version}_{timestamp}.json"
        
        version_report = generator.generate_weekly_report(
            version_analyses,
            str(version_json)
        )
        
        # Add metadata
        version_report["engine_version"] = version
        version_report["version_notes"] = stats["notes"]
        version_report["game_ids"] = stats["games"]
        
        version_md = output_path / f"v7p3r_analysis_{safe_version}_{timestamp}.md"
        with open(version_md, 'w') as f:
            f.write(f"# V7P3R {version} Analysis\n\n")
            f.write(f"**Notes**: {stats['notes']}\n\n")
            f.write(f"**Games**: {len(version_analyses)}\n\n")
    
    # 3. Version comparison
    logger.info("\n3. Version comparison...")
    comparison_path = output_path / f"version_comparison_{timestamp}.md"
    
    with open(comparison_path, 'w') as f:
        f.write("# V7P3R Engine Version Comparison\n\n")
        f.write(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Games**: {len(analyses)}\n\n")
        f.write(f"**Processing Time**: {elapsed_time/60:.1f} minutes ({len(analyses)/elapsed_time:.2f} games/sec)\n\n")
        f.write("---\n\n")
        
        f.write("## Version Performance Summary\n\n")
        f.write("| Version | Games | Avg CPL | Top1% | Top3% | Top5% | Blunders/Game | Notes |\n")
        f.write("|---------|-------|---------|-------|-------|-------|---------------|-------|\n")
        
        for version in sorted(version_stats.keys()):
            version_analyses = [a for a in analyses if get_game_version(a, version_stats) == version]
            if not version_analyses:
                continue
            
            avg_cpl = sum(a.average_centipawn_loss for a in version_analyses) / len(version_analyses)
            avg_top1 = sum(a.top1_alignment for a in version_analyses) / len(version_analyses)
            avg_top3 = sum(a.top3_alignment for a in version_analyses) / len(version_analyses)
            avg_top5 = sum(a.top5_alignment for a in version_analyses) / len(version_analyses)
            avg_blunders = sum(a.blunders + a.critical_blunders for a in version_analyses) / len(version_analyses)
            notes = version_stats[version]["notes"]
            
            f.write(f"| {version} | {len(version_analyses)} | {avg_cpl:.1f} | {avg_top1:.1f}% | {avg_top3:.1f}% | {avg_top5:.1f}% | {avg_blunders:.1f} | {notes} |\n")
    
    logger.info(f"   Saved: {comparison_path.name}")


def get_game_version(report: GameAnalysisReport, version_stats: dict) -> str:
    """Helper to get version for a game report."""
    # Check if game_id is in any version's games list
    for version, stats in version_stats.items():
        if report.game_id in stats['games']:
            return version
    return 'UNKNOWN'


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel game analysis with multi-processing")
    parser.add_argument("pgn_file", help="Path to PGN file")
    parser.add_argument("--output-dir", default="reports_parallel", help="Output directory")
    parser.add_argument("--workers", type=int, help=f"Number of parallel workers (default: {max(1, cpu_count()-1)})")
    parser.add_argument("--limit", type=int, help="Limit number of games to analyze")
    parser.add_argument("--stockfish", help="Path to Stockfish executable")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pgn_file):
        print(f"ERROR: File not found: {args.pgn_file}")
        sys.exit(1)
    
    parallel_analyze_games(
        args.pgn_file,
        args.output_dir,
        args.workers,
        args.limit,
        args.stockfish
    )
