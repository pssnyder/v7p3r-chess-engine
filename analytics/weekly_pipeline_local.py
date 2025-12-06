"""
V7P3R Weekly Analytics - Local Pipeline
Complete automated analysis system for local Docker execution.

Requirements Met:
- Scheduled Lichess API downloads
- Parallel Stockfish analysis agents
- Blunder tracking & tactical theme coverage
- Centipawn loss & deep analysis extraction
- Long-term historical storage
- KPI tracking (W/L/D, accuracy, ELO, terminations)
- Programmatic version tracking
- Modifiable data schema
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import shutil

# Local imports
from fetch_lichess_games import fetch_games_from_lichess
from parallel_analysis import extract_games_from_pgn, analyze_single_game
from version_tracker import VersionTracker
from report_generator import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalAnalyticsPipeline:
    """Local analytics pipeline with full feature set."""
    
    def __init__(
        self,
        stockfish_path: str,
        reports_dir: str = "./analytics_reports",
        days_back: int = 7,
        workers: int = 12
    ):
        """
        Initialize pipeline.
        
        Args:
            stockfish_path: Path to Stockfish executable
            reports_dir: Base directory for all reports
            days_back: Days of history to analyze
            workers: Parallel analysis workers
        """
        self.stockfish_path = stockfish_path
        self.reports_dir = Path(reports_dir)
        self.days_back = days_back
        self.workers = workers
        
        # Initialize components
        self.version_tracker = VersionTracker()
        self.report_generator = ReportGenerator()
        
        # Ensure directories exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info("V7P3R Local Analytics Pipeline")
        logger.info("=" * 70)
        logger.info(f"Reports Dir: {self.reports_dir}")
        logger.info(f"Stockfish: {self.stockfish_path}")
        logger.info(f"Workers: {self.workers}")
        logger.info(f"Days Back: {self.days_back}")
        logger.info("=" * 70)
    
    def run_weekly_analysis(self) -> Dict:
        """
        Execute complete weekly analysis pipeline.
        
        Returns:
            Summary dict with results
        """
        start_time = datetime.now()
        
        # Create week folder
        week_folder = self._get_week_folder()
        week_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n📁 Week Folder: {week_folder}")
        
        summary = {
            "schema_version": "1.0",
            "start_time": start_time.isoformat(),
            "week_folder": str(week_folder),
            "stages": {}
        }
        
        try:
            # Stage 1: Fetch games from Lichess
            logger.info("\n" + "=" * 70)
            logger.info("STAGE 1: Fetching Games from Lichess API")
            logger.info("=" * 70)
            
            pgn_file = self._fetch_games(week_folder)
            if not pgn_file:
                summary["success"] = False
                summary["error"] = "Failed to fetch games"
                return summary
            
            summary["stages"]["fetch"] = {
                "success": True,
                "pgn_file": str(pgn_file),
                "file_size_mb": pgn_file.stat().st_size / 1024 / 1024
            }
            
            # Stage 2: Parse and version-map games
            logger.info("\n" + "=" * 70)
            logger.info("STAGE 2: Parsing PGN & Version Mapping")
            logger.info("=" * 70)
            
            game_data_list = self._parse_and_map_versions(pgn_file)
            if not game_data_list:
                summary["success"] = False
                summary["error"] = "No games to analyze"
                return summary
            
            summary["stages"]["parse"] = {
                "success": True,
                "total_games": len(game_data_list)
            }
            
            # Stage 3: Parallel Stockfish analysis
            logger.info("\n" + "=" * 70)
            logger.info(f"STAGE 3: Parallel Analysis ({self.workers} workers)")
            logger.info("=" * 70)
            
            analysis_results = self._run_parallel_analysis(game_data_list, week_folder)
            
            summary["stages"]["analysis"] = {
                "success": True,
                "games_analyzed": len(analysis_results),
                "games_failed": len([r for r in analysis_results if not r.get("success")])
            }
            
            # Stage 4: Generate reports
            logger.info("\n" + "=" * 70)
            logger.info("STAGE 4: Generating Reports")
            logger.info("=" * 70)
            
            reports = self._generate_reports(analysis_results, week_folder)
            
            summary["stages"]["reports"] = {
                "success": True,
                "reports_generated": len(reports)
            }
            
            # Stage 5: Update historical data
            logger.info("\n" + "=" * 70)
            logger.info("STAGE 5: Updating Historical Data")
            logger.info("=" * 70)
            
            self._update_historical_data(analysis_results, week_folder, summary)
            
            summary["stages"]["historical"] = {"success": True}
            
            # Success!
            summary["success"] = True
            summary["end_time"] = datetime.now().isoformat()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            summary["elapsed_seconds"] = elapsed
            
            logger.info("\n" + "=" * 70)
            logger.info("✓ PIPELINE COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Total Time: {elapsed:.1f}s")
            logger.info(f"Reports: {week_folder}")
            logger.info("=" * 70)
            
            return summary
            
        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {e}", exc_info=True)
            summary["success"] = False
            summary["error"] = str(e)
            summary["end_time"] = datetime.now().isoformat()
            return summary
        
        finally:
            # Save pipeline summary
            summary_file = week_folder / "pipeline_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"\n💾 Pipeline summary: {summary_file}")
    
    def _get_week_folder(self) -> Path:
        """Get folder path for this week's reports."""
        now = datetime.now()
        year = now.year
        week_num = now.isocalendar()[1]
        monday = now - timedelta(days=now.weekday())
        monday_str = monday.strftime("%Y-%m-%d")
        
        return self.reports_dir / str(year) / f"week_{week_num:02d}_{monday_str}"
    
    def _fetch_games(self, week_folder: Path) -> Optional[Path]:
        """Fetch games from Lichess API."""
        pgn_dir = week_folder / "pgn"
        pgn_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)
        
        output_file = pgn_dir / f"v7p3r_weekly_{start_date.strftime('%Y-%m-%d')}.pgn"
        
        try:
            result = fetch_games_from_lichess(
                username="v7p3r_bot",
                since_date=start_date,
                until_date=end_date,
                output_file=str(output_file)
            )
            
            if result and result.exists():
                logger.info(f"✓ Games saved: {result}")
                return result
            else:
                logger.error("✗ No games downloaded")
                return None
                
        except Exception as e:
            logger.error(f"✗ Fetch failed: {e}")
            return None
    
    def _parse_and_map_versions(self, pgn_file: Path) -> List[Dict]:
        """Parse PGN and map games to versions."""
        logger.info(f"Parsing {pgn_file}...")
        
        game_data_list = extract_games_from_pgn(
            str(pgn_file),
            limit=None  # No limit, analyze all
        )
        
        logger.info(f"✓ Parsed {len(game_data_list)} games")
        
        # Add stockfish path and total_games to each game
        total_games = len(game_data_list)
        for game_data in game_data_list:
            game_data['stockfish_path'] = self.stockfish_path
            game_data['total_games'] = total_games
        
        return game_data_list
    
    def _run_parallel_analysis(self, game_data_list: List[Dict], week_folder: Path) -> List[Dict]:
        """Run parallel Stockfish analysis on all games."""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        # Create games output directory
        games_dir = week_folder / "games"
        games_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        total_games = len(game_data_list)
        
        logger.info(f"Analyzing {total_games} games with {self.workers} workers...")
        
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # Submit all games
            futures = {
                executor.submit(analyze_single_game, game_data): game_data 
                for game_data in game_data_list
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{total_games} games analyzed")
                
                # Save individual game analysis
                if result.get("success"):
                    game_id = result["game_id"]
                    game_file = games_dir / f"{game_id}_analysis.json"
                    
                    with open(game_file, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
        
        logger.info(f"✓ Analysis complete: {len(results)} games")
        
        return results
    
    def _generate_reports(self, analysis_results: List[Dict], week_folder: Path) -> List[Path]:
        """Generate all report files."""
        reports = []
        
        # Technical summary report (JSON)
        summary_file = week_folder / "weekly_summary.json"
        summary_data = self._aggregate_metrics(analysis_results)
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        reports.append(summary_file)
        logger.info(f"✓ Summary: {summary_file}")
        
        # Technical report (Markdown)
        md_file = week_folder / "technical_report.md"
        md_content = self._generate_markdown_report(summary_data)
        
        with open(md_file, 'w') as f:
            f.write(md_content)
        reports.append(md_file)
        logger.info(f"✓ Report: {md_file}")
        
        # Version breakdown
        version_file = week_folder / "version_breakdown.json"
        version_data = self._aggregate_by_version(analysis_results)
        
        with open(version_file, 'w') as f:
            json.dump(version_data, f, indent=2, default=str)
        reports.append(version_file)
        logger.info(f"✓ Versions: {version_file}")
        
        return reports
    
    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Aggregate metrics from all games."""
        successful = [r for r in results if r.get("success")]
        
        if not successful:
            return {"error": "No successful games"}
        
        # Basic counts
        total_games = len(successful)
        
        # Extract reports
        reports = [r["report"] for r in successful]
        
        # Win/Loss/Draw
        results_count = {"win": 0, "loss": 0, "draw": 0}
        for report in reports:
            result = report.result.lower()
            if "1-0" in result or "0-1" in result:
                # Need to check which color we played
                if (report.color == "white" and "1-0" in result) or \
                   (report.color == "black" and "0-1" in result):
                    results_count["win"] += 1
                else:
                    results_count["loss"] += 1
            elif "1/2" in result:
                results_count["draw"] += 1
        
        # Accuracy metrics
        cpl_values = [r.average_centipawn_loss for r in reports]
        top1_values = [r.top1_alignment for r in reports]
        
        # Blunders
        total_blunders = sum(r.blunders for r in reports)
        total_critical = sum(r.critical_blunders for r in reports)
        
        # Theme aggregation
        theme_stats = self._aggregate_themes(reports, total_games)
        
        return {
            "schema_version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "games_analyzed": total_games,
            
            "results": {
                "wins": results_count["win"],
                "losses": results_count["loss"],
                "draws": results_count["draw"],
                "win_rate": results_count["win"] / total_games * 100 if total_games > 0 else 0
            },
            
            "accuracy": {
                "average_cpl": sum(cpl_values) / len(cpl_values) if cpl_values else 0,
                "median_cpl": sorted(cpl_values)[len(cpl_values)//2] if cpl_values else 0,
                "average_top1_alignment": sum(top1_values) / len(top1_values) if top1_values else 0
            },
            
            "blunders": {
                "total": total_blunders,
                "per_game": total_blunders / total_games if total_games > 0 else 0,
                "critical_total": total_critical,
                "critical_per_game": total_critical / total_games if total_games > 0 else 0
            },
            
            "move_quality": {
                "best_moves": sum(r.best_moves for r in reports),
                "excellent_moves": sum(r.excellent_moves for r in reports),
                "good_moves": sum(r.good_moves for r in reports),
                "inaccuracies": sum(r.inaccuracies for r in reports),
                "mistakes": sum(r.mistakes for r in reports)
            },
            
            "themes": theme_stats
        }
    
    def _aggregate_themes(self, reports: List, total_games: int) -> Dict:
        """
        Aggregate theme statistics across all games.
        
        Categorizes themes as:
        - Positive: Themes we WANT to increase (good chess concepts)
        - Negative: Themes we WANT to decrease (weaknesses)
        - Tactical: Situational tactics (context-dependent)
        """
        # Initialize counters
        castling_ks = sum(r.themes.castling_king_side for r in reports)
        castling_qs = sum(r.themes.castling_queen_side for r in reports)
        fianchetto = sum(r.themes.fianchetto for r in reports)
        isolated_pawns = sum(r.themes.isolated_pawns for r in reports)
        doubled_pawns = sum(r.themes.doubled_pawns for r in reports)
        passed_pawns = sum(r.themes.passed_pawns for r in reports)
        bishop_pair_count = sum(1 for r in reports if r.themes.bishop_pair)
        knight_outpost = sum(r.themes.knight_outpost for r in reports)
        rook_open_file = sum(r.themes.rook_open_file for r in reports)
        rook_seventh = sum(r.themes.rook_seventh_rank for r in reports)
        battery = sum(r.themes.battery for r in reports)
        pin = sum(r.themes.pin for r in reports)
        skewer = sum(r.themes.skewer for r in reports)
        fork = sum(r.themes.fork for r in reports)
        discovered = sum(r.themes.discovered_attack for r in reports)
        mate_threats = sum(len(r.themes.mate_threats) for r in reports)
        
        return {
            "positive_themes": {
                "description": "Good chess concepts - we want these HIGH",
                "castling": {
                    "kingside": castling_ks,
                    "queenside": castling_qs,
                    "total": castling_ks + castling_qs,
                    "percentage": (castling_ks + castling_qs) / total_games * 100 if total_games > 0 else 0,
                    "per_game": (castling_ks + castling_qs) / total_games if total_games > 0 else 0
                },
                "passed_pawns": {
                    "total": passed_pawns,
                    "per_game": passed_pawns / total_games if total_games > 0 else 0,
                    "description": "Created passed pawns (endgame advantage)"
                },
                "bishop_pair": {
                    "games_with": bishop_pair_count,
                    "percentage": bishop_pair_count / total_games * 100 if total_games > 0 else 0,
                    "description": "Retained both bishops (positional strength)"
                },
                "knight_outpost": {
                    "total": knight_outpost,
                    "per_game": knight_outpost / total_games if total_games > 0 else 0,
                    "description": "Knights on strong squares"
                },
                "rook_activity": {
                    "open_files": rook_open_file,
                    "seventh_rank": rook_seventh,
                    "total": rook_open_file + rook_seventh,
                    "per_game": (rook_open_file + rook_seventh) / total_games if total_games > 0 else 0,
                    "description": "Active rook placement"
                },
                "fianchetto": {
                    "total": fianchetto,
                    "per_game": fianchetto / total_games if total_games > 0 else 0,
                    "description": "Fianchettoed bishops (g2/b2/g7/b7)"
                }
            },
            
            "negative_themes": {
                "description": "Weaknesses - we want these LOW",
                "isolated_pawns": {
                    "total": isolated_pawns,
                    "per_game": isolated_pawns / total_games if total_games > 0 else 0,
                    "severity": "high" if (isolated_pawns / total_games if total_games > 0 else 0) > 2.0 else "medium" if (isolated_pawns / total_games if total_games > 0 else 0) > 1.0 else "low",
                    "description": "Isolated pawns (structural weakness)"
                },
                "doubled_pawns": {
                    "total": doubled_pawns,
                    "per_game": doubled_pawns / total_games if total_games > 0 else 0,
                    "severity": "high" if (doubled_pawns / total_games if total_games > 0 else 0) > 1.5 else "medium" if (doubled_pawns / total_games if total_games > 0 else 0) > 0.5 else "low",
                    "description": "Doubled pawns (often weak)"
                }
            },
            
            "tactical_themes": {
                "description": "Situational tactics - context matters",
                "pins": {
                    "total": pin,
                    "per_game": pin / total_games if total_games > 0 else 0,
                    "description": "Pinned opponent pieces"
                },
                "skewers": {
                    "total": skewer,
                    "per_game": skewer / total_games if total_games > 0 else 0,
                    "description": "Skewered opponent pieces"
                },
                "forks": {
                    "total": fork,
                    "per_game": fork / total_games if total_games > 0 else 0,
                    "description": "Forked opponent pieces"
                },
                "discovered_attacks": {
                    "total": discovered,
                    "per_game": discovered / total_games if total_games > 0 else 0,
                    "description": "Discovered attacks executed"
                },
                "batteries": {
                    "total": battery,
                    "per_game": battery / total_games if total_games > 0 else 0,
                    "description": "Queen/Rook batteries formed"
                },
                "mate_threats": {
                    "total": mate_threats,
                    "per_game": mate_threats / total_games if total_games > 0 else 0,
                    "description": "Positions with mate threats"
                }
            },
            
            "coverage_summary": {
                "positive_score": self._calculate_positive_score(
                    castling_ks + castling_qs, passed_pawns, bishop_pair_count,
                    knight_outpost, rook_open_file + rook_seventh, fianchetto, total_games
                ),
                "weakness_score": self._calculate_weakness_score(
                    isolated_pawns, doubled_pawns, total_games
                ),
                "tactical_score": self._calculate_tactical_score(
                    pin, skewer, fork, discovered, battery, mate_threats, total_games
                )
            }
        }
    
    def _calculate_positive_score(self, castling, passed, bishop_pair, outpost, rook_activity, fianchetto, games):
        """Calculate normalized positive theme score (0-100)."""
        if games == 0:
            return 0
        
        # Weighted scoring (out of 100)
        score = 0
        score += min((castling / games) * 20, 20)  # Max 20 points (expect 1 per game)
        score += min((passed / games) * 15, 15)    # Max 15 points
        score += min((bishop_pair / games) * 15, 15)  # Max 15 points
        score += min((outpost / games) * 15, 15)   # Max 15 points
        score += min((rook_activity / games) * 20, 20)  # Max 20 points
        score += min((fianchetto / games) * 15, 15)  # Max 15 points
        
        return round(score, 1)
    
    def _calculate_weakness_score(self, isolated, doubled, games):
        """Calculate weakness penalty score (lower is better, 0-100)."""
        if games == 0:
            return 0
        
        # Higher score = more weaknesses (BAD)
        penalty = 0
        penalty += min((isolated / games) * 30, 50)  # Heavy penalty
        penalty += min((doubled / games) * 20, 50)   # Moderate penalty
        
        return round(penalty, 1)
    
    def _calculate_tactical_score(self, pin, skewer, fork, discovered, battery, mate, games):
        """Calculate tactical opportunity score (0-100)."""
        if games == 0:
            return 0
        
        score = 0
        score += min((pin / games) * 15, 20)
        score += min((skewer / games) * 15, 20)
        score += min((fork / games) * 20, 20)
        score += min((discovered / games) * 15, 20)
        score += min((battery / games) * 10, 10)
        score += min((mate / games) * 10, 10)
        
        return round(score, 1)
    
    def _aggregate_by_version(self, results: List[Dict]) -> Dict:
        """Aggregate metrics by engine version."""
        by_version = {}
        
        for result in results:
            if not result.get("success"):
                continue
            
            version = result.get("version", "UNKNOWN")
            
            if version not in by_version:
                by_version[version] = []
            
            by_version[version].append(result)
        
        # Aggregate each version
        aggregated = {}
        for version, version_results in by_version.items():
            aggregated[version] = self._aggregate_metrics(version_results)
            aggregated[version]["games_count"] = len(version_results)
        
        return aggregated
    
    def _generate_markdown_report(self, summary: Dict) -> str:
        """Generate human-readable Markdown report with comprehensive theme analysis."""
        themes = summary.get('themes', {})
        
        md = f"""# V7P3R Weekly Analytics Report

**Generated**: {summary.get('generated_at', 'N/A')}  
**Games Analyzed**: {summary.get('games_analyzed', 0)}

---

## 📊 Results Summary

- **Wins**: {summary['results']['wins']}
- **Losses**: {summary['results']['losses']}
- **Draws**: {summary['results']['draws']}
- **Win Rate**: {summary['results']['win_rate']:.1f}%

---

## 🎯 Accuracy Metrics

- **Average CPL**: {summary['accuracy']['average_cpl']:.1f}
- **Median CPL**: {summary['accuracy']['median_cpl']:.1f}
- **Top1 Alignment**: {summary['accuracy']['average_top1_alignment']:.1f}%

---

## ⚠️ Blunder Analysis

- **Total Blunders**: {summary['blunders']['total']}
- **Blunders per Game**: {summary['blunders']['per_game']:.2f}
- **Critical Blunders**: {summary['blunders']['critical_total']}
- **Critical per Game**: {summary['blunders']['critical_per_game']:.2f}

---

## 📈 Move Quality Distribution

- **Best Moves**: {summary['move_quality']['best_moves']}
- **Excellent Moves**: {summary['move_quality']['excellent_moves']}
- **Good Moves**: {summary['move_quality']['good_moves']}
- **Inaccuracies**: {summary['move_quality']['inaccuracies']}
- **Mistakes**: {summary['move_quality']['mistakes']}

---

## 🎨 Theme Coverage Analysis

### ✅ Positive Themes (Goal: INCREASE)

"""
        
        # Positive themes
        if themes.get('positive_themes'):
            pos = themes['positive_themes']
            
            # Castling
            if 'castling' in pos:
                cast = pos['castling']
                md += f"""**Castling** ({cast['percentage']:.1f}% of games)
- Kingside: {cast['kingside']} times
- Queenside: {cast['queenside']} times
- **Average**: {cast['per_game']:.2f} per game

"""
            
            # Passed Pawns
            if 'passed_pawns' in pos:
                pp = pos['passed_pawns']
                md += f"""**Passed Pawns** ⭐
- Total created: {pp['total']}
- **Average**: {pp['per_game']:.2f} per game
- {pp['description']}

"""
            
            # Bishop Pair
            if 'bishop_pair' in pos:
                bp = pos['bishop_pair']
                md += f"""**Bishop Pair Retention**
- Games with both bishops: {bp['games_with']} ({bp['percentage']:.1f}%)
- {bp['description']}

"""
            
            # Knight Outposts
            if 'knight_outpost' in pos:
                ko = pos['knight_outpost']
                md += f"""**Knight Outposts**
- Total: {ko['total']}
- **Average**: {ko['per_game']:.2f} per game
- {ko['description']}

"""
            
            # Rook Activity
            if 'rook_activity' in pos:
                ra = pos['rook_activity']
                md += f"""**Rook Activity**
- Open files: {ra['open_files']}
- Seventh rank: {ra['seventh_rank']}
- **Total**: {ra['total']} ({ra['per_game']:.2f} per game)

"""
            
            # Fianchetto
            if 'fianchetto' in pos:
                fian = pos['fianchetto']
                md += f"""**Fianchetto Development**
- Total: {fian['total']} ({fian['per_game']:.2f} per game)
- {fian['description']}

"""
        
        # Negative themes (weaknesses)
        md += """---

### ⚠️ Negative Themes (Goal: DECREASE)

"""
        
        if themes.get('negative_themes'):
            neg = themes['negative_themes']
            
            # Isolated Pawns
            if 'isolated_pawns' in neg:
                ip = neg['isolated_pawns']
                severity_emoji = "🔴" if ip['severity'] == "high" else "🟡" if ip['severity'] == "medium" else "🟢"
                md += f"""**Isolated Pawns** {severity_emoji} Severity: {ip['severity'].upper()}
- Total: {ip['total']}
- **Average**: {ip['per_game']:.2f} per game
- {ip['description']}

"""
            
            # Doubled Pawns
            if 'doubled_pawns' in neg:
                dp = neg['doubled_pawns']
                severity_emoji = "🔴" if dp['severity'] == "high" else "🟡" if dp['severity'] == "medium" else "🟢"
                md += f"""**Doubled Pawns** {severity_emoji} Severity: {dp['severity'].upper()}
- Total: {dp['total']}
- **Average**: {dp['per_game']:.2f} per game
- {dp['description']}

"""
        
        # Tactical themes
        md += """---

### ⚔️ Tactical Themes (Execution)

"""
        
        if themes.get('tactical_themes'):
            tact = themes['tactical_themes']
            
            if 'pins' in tact:
                md += f"- **Pins**: {tact['pins']['total']} ({tact['pins']['per_game']:.2f}/game)\n"
            if 'skewers' in tact:
                md += f"- **Skewers**: {tact['skewers']['total']} ({tact['skewers']['per_game']:.2f}/game)\n"
            if 'forks' in tact:
                md += f"- **Forks**: {tact['forks']['total']} ({tact['forks']['per_game']:.2f}/game)\n"
            if 'discovered_attacks' in tact:
                md += f"- **Discovered Attacks**: {tact['discovered_attacks']['total']} ({tact['discovered_attacks']['per_game']:.2f}/game)\n"
            if 'batteries' in tact:
                md += f"- **Batteries**: {tact['batteries']['total']} ({tact['batteries']['per_game']:.2f}/game)\n"
            if 'mate_threats' in tact:
                md += f"- **Mate Threats**: {tact['mate_threats']['total']} ({tact['mate_threats']['per_game']:.2f}/game)\n"
        
        # Coverage Summary
        md += """
---

### 📊 Theme Coverage Scores

"""
        
        if themes.get('coverage_summary'):
            cov = themes['coverage_summary']
            
            pos_score = cov.get('positive_score', 0)
            pos_bar = '█' * int(pos_score / 5) + '░' * (20 - int(pos_score / 5))
            
            weak_score = cov.get('weakness_score', 0)
            weak_bar = '█' * int(weak_score / 5) + '░' * (20 - int(weak_score / 5))
            weak_color = "🔴" if weak_score > 40 else "🟡" if weak_score > 20 else "🟢"
            
            tact_score = cov.get('tactical_score', 0)
            tact_bar = '█' * int(tact_score / 5) + '░' * (20 - int(tact_score / 5))
            
            md += f"""**Positive Theme Score**: {pos_score:.1f}/100
```
{pos_bar}
```
*Higher is better - measures coverage of good chess concepts*

**Weakness Penalty**: {weak_score:.1f}/100 {weak_color}
```
{weak_bar}
```
*Lower is better - measures structural weaknesses*

**Tactical Execution**: {tact_score:.1f}/100
```
{tact_bar}
```
*Measures tactical pattern recognition and execution*

"""
        
        md += """---

*Generated by V7P3R Local Analytics System*

**Theme Tracking**: Monitors 15+ chess themes across all games  
**Data Schema**: `docs/Analytics_Data_Schema.md`
"""
        
        return md
    
    def _update_historical_data(self, results: List[Dict], week_folder: Path, summary: Dict):
        """Update historical summary with this week's data."""
        historical_file = self.reports_dir / "historical_summary.json"
        
        # Load existing or create new
        if historical_file.exists():
            with open(historical_file, 'r') as f:
                historical = json.load(f)
        else:
            historical = {
                "schema_version": "1.0",
                "weeks": [],
                "by_version": {}
            }
        
        # Add this week
        week_data = {
            "week_id": week_folder.name,
            "week_folder": str(week_folder),
            "date": datetime.now().isoformat(),
            "games": len([r for r in results if r.get("success")]),
            "summary": summary.get("stages", {}).get("analysis", {})
        }
        
        historical["weeks"].append(week_data)
        historical["total_weeks"] = len(historical["weeks"])
        historical["last_updated"] = datetime.now().isoformat()
        
        # Save
        with open(historical_file, 'w') as f:
            json.dump(historical, f, indent=2)
        
        logger.info(f"✓ Historical data updated: {historical_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="V7P3R Local Weekly Analytics")
    parser.add_argument("--stockfish", required=True, help="Path to Stockfish executable")
    parser.add_argument("--reports-dir", default="./analytics_reports", help="Reports directory")
    parser.add_argument("--days-back", type=int, default=7, help="Days of history to analyze")
    parser.add_argument("--workers", type=int, default=12, help="Parallel workers")
    
    args = parser.parse_args()
    
    # Verify Stockfish exists
    if not Path(args.stockfish).exists():
        logger.error(f"Stockfish not found: {args.stockfish}")
        sys.exit(1)
    
    # Create pipeline
    pipeline = LocalAnalyticsPipeline(
        stockfish_path=args.stockfish,
        reports_dir=args.reports_dir,
        days_back=args.days_back,
        workers=args.workers
    )
    
    # Run analysis
    summary = pipeline.run_weekly_analysis()
    
    # Exit with appropriate code
    sys.exit(0 if summary.get("success") else 1)


if __name__ == "__main__":
    main()
