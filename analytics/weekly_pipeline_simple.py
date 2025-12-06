"""
V7P3R Weekly Analytics - Simplified Pipeline
Uses existing scripts: fetch_lichess_games.py, parallel_analysis.py
"""
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_weekly_analytics(
    stockfish_path: str,
    work_dir: str = "/workspace",
    days_back: int = 7,
    workers: int = 12
) -> dict:
    """
    Run complete weekly analytics pipeline.
    
    Args:
        stockfish_path: Path to Stockfish executable
        work_dir: Working directory
        days_back: Days to analyze
        workers: Parallel workers
        
    Returns:
        Summary dict
    """
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    
    pgn_dir = work_path / "pgn_downloads"
    reports_dir = work_path / "reports"
    pgn_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    since_str = start_date.strftime("%Y-%m-%d")
    
    logger.info("=" * 70)
    logger.info("V7P3R Weekly Analytics Pipeline")
    logger.info("=" * 70)
    logger.info(f"Period: {since_str} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Stockfish: {stockfish_path}")
    logger.info("=" * 70)
    
    summary = {
        "start_time": datetime.now().isoformat(),
        "period": {"from": since_str, "to": end_date.strftime("%Y-%m-%d")},
        "stages": {}
    }
    
    try:
        # Stage 1: Fetch games
        logger.info("\n[1/2] Fetching games from Lichess...")
        pgn_file = pgn_dir / f"v7p3r_weekly_{since_str}.pgn"
        
        fetch_cmd = [
            sys.executable,
            "fetch_lichess_games.py",
            "--since", since_str,
            "--output", str(pgn_file)
        ]
        
        result = subprocess.run(
            fetch_cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode != 0:
            logger.error(f"Fetch failed: {result.stderr}")
            summary["stages"]["fetch"] = {"success": False, "error": result.stderr}
            return summary
        
        logger.info(result.stdout)
        summary["stages"]["fetch"] = {"success": True, "pgn_file": str(pgn_file)}
        
        # Stage 2: Parallel analysis
        logger.info("\n[2/2] Running parallel analysis...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = reports_dir / f"weekly_{timestamp}"
        
        analysis_cmd = [
            sys.executable,
            "parallel_analysis.py",
            str(pgn_file),
            "--output-dir", str(output_dir),
            "--workers", str(workers)
        ]
        
        # Set environment variable for Stockfish
        env = {"STOCKFISH_PATH": stockfish_path}
        
        result = subprocess.run(
            analysis_cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            env={**os.environ, **env}
        )
        
        if result.returncode != 0:
            logger.error(f"Analysis failed: {result.stderr}")
            summary["stages"]["analysis"] = {"success": False, "error": result.stderr}
            return summary
        
        logger.info(result.stdout)
        summary["stages"]["analysis"] = {
            "success": True,
            "output_dir": str(output_dir)
        }
        
        # Success
        summary["success"] = True
        summary["end_time"] = datetime.now().isoformat()
        
        logger.info("\n" + "=" * 70)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Reports: {output_dir}")
        logger.info("=" * 70)
        
        return summary
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        summary["success"] = False
        summary["error"] = str(e)
        summary["end_time"] = datetime.now().isoformat()
        return summary


def main():
    """Main entry point."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="V7P3R Weekly Analytics")
    parser.add_argument("--stockfish", required=True, help="Stockfish path")
    parser.add_argument("--work-dir", default="/workspace", help="Work directory")
    parser.add_argument("--days-back", type=int, default=7, help="Days to analyze")
    parser.add_argument("--workers", type=int, default=12, help="Parallel workers")
    
    args = parser.parse_args()
    
    # Verify Stockfish
    if not Path(args.stockfish).exists():
        logger.error(f"Stockfish not found: {args.stockfish}")
        sys.exit(1)
    
    # Run pipeline
    summary = run_weekly_analytics(
        stockfish_path=args.stockfish,
        work_dir=args.work_dir,
        days_back=args.days_back,
        workers=args.workers
    )
    
    # Save summary
    summary_file = Path(args.work_dir) / "pipeline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved: {summary_file}")
    
    sys.exit(0 if summary.get("success") else 1)


if __name__ == "__main__":
    main()
