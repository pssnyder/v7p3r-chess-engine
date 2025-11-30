"""
V7P3R Weekly Analytics Pipeline
Main orchestrator for automated game analysis and reporting
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict
import logging
import json

from game_collector import GameCollector  # type: ignore
from v7p3r_analytics import V7P3RAnalytics, GameAnalysisReport  # type: ignore
from report_generator import ReportGenerator  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalyticsPipeline:
    """Main pipeline for weekly analytics."""
    
    def __init__(
        self,
        stockfish_path: str,
        work_dir: str = "./analytics_workspace",
        bot_username: str = "v7p3r_bot",
        days_back: int = 7
    ):
        """
        Initialize analytics pipeline.
        
        Args:
            stockfish_path: Path to Stockfish executable
            work_dir: Working directory for downloads and outputs
            bot_username: Bot's Lichess username
            days_back: Days of game history to analyze
        """
        self.stockfish_path = stockfish_path
        self.work_dir = Path(work_dir)
        self.bot_username = bot_username
        self.days_back = days_back
        
        # Create directories
        self.pgn_dir = self.work_dir / "pgn_downloads"
        self.analysis_dir = self.work_dir / "game_analyses"
        self.report_dir = self.work_dir / "reports"
        
        for dir_path in [self.pgn_dir, self.analysis_dir, self.report_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized: {self.work_dir}")
    
    def run_full_pipeline(self) -> Dict:
        """
        Run complete analytics pipeline.
        
        Returns:
            Summary dictionary with pipeline results
        """
        logger.info("=" * 60)
        logger.info("Starting V7P3R Weekly Analytics Pipeline")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        summary = {
            "start_time": start_time.isoformat(),
            "stages": {},
            "success": False
        }
        
        try:
            # Stage 1: Collect games from GCP
            logger.info("\n[Stage 1/3] Collecting games from GCP...")
            collector = GameCollector()
            pgn_files = collector.fetch_recent_games(
                str(self.pgn_dir),
                days_back=self.days_back
            )
            
            summary["stages"]["collection"] = {
                "games_downloaded": len(pgn_files),
                "success": len(pgn_files) > 0
            }
            
            if not pgn_files:
                logger.warning("No games to analyze")
                return summary
            
            logger.info(f"✓ Downloaded {len(pgn_files)} games")
            
            # Stage 2: Analyze games with Stockfish
            logger.info("\n[Stage 2/3] Analyzing games with Stockfish...")
            reports = []
            
            with V7P3RAnalytics(self.stockfish_path, self.bot_username) as analytics:
                for i, pgn_file in enumerate(pgn_files, 1):
                    logger.info(f"  Analyzing game {i}/{len(pgn_files)}: {pgn_file.name}")
                    
                    report = analytics.analyze_game(str(pgn_file))
                    if report:
                        reports.append(report)
                        
                        # Save individual analysis
                        analysis_file = self.analysis_dir / f"{report.game_id}_analysis.json"
                        with open(analysis_file, 'w') as f:
                            # Convert dataclass to dict for JSON serialization
                            json.dump(self._report_to_dict(report), f, indent=2)
            
            summary["stages"]["analysis"] = {
                "games_analyzed": len(reports),
                "success": len(reports) > 0
            }
            
            if not reports:
                logger.warning("No games successfully analyzed")
                return summary
            
            logger.info(f"✓ Analyzed {len(reports)} games")
            
            # Stage 3: Generate weekly report
            logger.info("\n[Stage 3/3] Generating weekly report...")
            generator = ReportGenerator(self.bot_username)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.report_dir / f"weekly_report_{timestamp}.json"
            
            weekly_report = generator.generate_weekly_report(
                reports,
                str(report_path)
            )
            
            summary["stages"]["reporting"] = {
                "report_generated": True,
                "report_path": str(report_path),
                "markdown_path": str(report_path.with_suffix('.md')),
                "success": True
            }
            
            logger.info(f"✓ Report saved to {report_path}")
            logger.info(f"✓ Markdown summary: {report_path.with_suffix('.md')}")
            
            # Pipeline success
            summary["success"] = True
            summary["end_time"] = datetime.now().isoformat()
            summary["duration_seconds"] = (datetime.now() - start_time).total_seconds()
            
            logger.info("\n" + "=" * 60)
            logger.info("Pipeline completed successfully!")
            logger.info(f"Duration: {summary['duration_seconds']:.1f} seconds")
            logger.info("=" * 60)
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            summary["error"] = str(e)
            summary["end_time"] = datetime.now().isoformat()
            return summary
    
    def _report_to_dict(self, report: GameAnalysisReport) -> Dict:
        """Convert GameAnalysisReport to dictionary for JSON serialization."""
        return {
            "game_id": report.game_id,
            "date": report.date,
            "result": report.result,
            "opponent": report.opponent,
            "color": report.color,
            "time_control": report.time_control,
            "opening": report.opening,
            "performance": {
                "avg_centipawn_loss": report.average_centipawn_loss,
                "best_moves": report.best_moves,
                "excellent_moves": report.excellent_moves,
                "good_moves": report.good_moves,
                "inaccuracies": report.inaccuracies,
                "mistakes": report.mistakes,
                "blunders": report.blunders,
                "critical_blunders": report.critical_blunders,
                "top1_alignment": report.top1_alignment,
                "top3_alignment": report.top3_alignment,
                "top5_alignment": report.top5_alignment
            },
            "themes": report.themes.to_dict(),
            "moves": [
                {
                    "move_number": m.move_number,
                    "move_san": m.move_san,
                    "classification": m.classification,
                    "eval_diff": m.eval_diff,
                    "best_move": m.best_move
                }
                for m in report.moves
            ]
        }


def main():
    """Main entry point for pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="V7P3R Weekly Analytics Pipeline"
    )
    parser.add_argument(
        "--stockfish",
        required=True,
        help="Path to Stockfish executable"
    )
    parser.add_argument(
        "--work-dir",
        default="./analytics_workspace",
        help="Working directory for pipeline outputs"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Number of days of game history to analyze"
    )
    parser.add_argument(
        "--bot-username",
        default="v7p3r_bot",
        help="Bot's Lichess username"
    )
    
    args = parser.parse_args()
    
    # Verify Stockfish exists
    if not Path(args.stockfish).exists():
        logger.error(f"Stockfish not found at: {args.stockfish}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = AnalyticsPipeline(
        stockfish_path=args.stockfish,
        work_dir=args.work_dir,
        bot_username=args.bot_username,
        days_back=args.days_back
    )
    
    summary = pipeline.run_full_pipeline()
    
    # Save summary
    summary_file = Path(args.work_dir) / "pipeline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nPipeline summary saved to: {summary_file}")
    
    # Exit with appropriate code
    sys.exit(0 if summary["success"] else 1)


if __name__ == "__main__":
    main()
