"""
Weekly Analytics Runner for V7P3R Engine
Automated weekly analysis focused on theme performance and recurring blunders
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json


def run_weekly_analysis(days_back: int = 7, workers: int = 12):
    """
    Run weekly analysis on recent games.
    
    Args:
        days_back: Number of days to analyze (default: 7 for weekly)
        workers: Number of parallel workers
    """
    analytics_dir = Path(__file__).parent
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    date_str = start_date.strftime("%Y-%m-%d")
    output_dir = f"reports_weekly_{start_date.strftime('%Y%m%d')}"
    
    print("=" * 70)
    print("V7P3R Weekly Analytics")
    print("=" * 70)
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Output: {output_dir}")
    print(f"Workers: {workers}")
    print("=" * 70)
    
    # Step 1: Fetch games from Lichess
    print("\n[1/3] Fetching games from Lichess...")
    pgn_file = analytics_dir / "current_games" / f"v7p3r_weekly_{start_date.strftime('%Y%m%d')}.pgn"
    pgn_file.parent.mkdir(exist_ok=True)
    
    fetch_cmd = [
        sys.executable,
        str(analytics_dir / "fetch_lichess_games.py"),
        "--since", date_str,
        "--output", str(pgn_file)
    ]
    
    result = subprocess.run(fetch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR fetching games: {result.stderr}")
        return False
    
    print(result.stdout)
    
    # Step 2: Run parallel analysis
    print("\n[2/3] Running parallel analysis...")
    analysis_cmd = [
        sys.executable,
        str(analytics_dir / "parallel_analysis.py"),
        str(pgn_file),
        "--output-dir", output_dir,
        "--workers", str(workers)
    ]
    
    result = subprocess.run(analysis_cmd)
    if result.returncode != 0:
        print("ERROR: Analysis failed")
        return False
    
    # Step 3: Generate blunder pattern report
    print("\n[3/3] Analyzing recurring blunder patterns...")
    blunder_cmd = [
        sys.executable,
        str(analytics_dir / "blunder_pattern_analyzer.py"),
        output_dir
    ]
    
    result = subprocess.run(blunder_cmd)
    if result.returncode != 0:
        print("WARNING: Blunder analysis failed (optional)")
    
    print("\n" + "=" * 70)
    print("Weekly Analysis Complete!")
    print("=" * 70)
    print(f"\nReports saved to: {output_dir}/")
    print("\nKey Reports:")
    print(f"  - Overall Performance: v7p3r_analysis_all_*.md")
    print(f"  - Per-Version: v7p3r_analysis_v*_*.md")
    print(f"  - Version Comparison: version_comparison_*.md")
    print(f"  - Blunder Patterns: blunder_patterns_*.md")
    print(f"  - Theme Analysis: theme_adherence_*.md")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run weekly v7p3r analytics")
    parser.add_argument("--days", type=int, default=7, help="Days back to analyze (default: 7)")
    parser.add_argument("--workers", type=int, default=12, help="Parallel workers (default: 12)")
    
    args = parser.parse_args()
    
    success = run_weekly_analysis(args.days, args.workers)
    sys.exit(0 if success else 1)
