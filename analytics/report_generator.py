"""
Weekly Report Generator - Creates comprehensive analytics summaries
Aggregates game analysis into actionable insights for engine improvement
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
from dataclasses import dataclass, field
import logging

from v7p3r_analytics import GameAnalysisReport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WeeklyStats:
    """Aggregated statistics for the week."""
    total_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    
    total_moves_analyzed: int = 0
    avg_centipawn_loss: float = 0.0
    
    best_moves: int = 0
    excellent_moves: int = 0
    good_moves: int = 0
    inaccuracies: int = 0
    mistakes: int = 0
    blunders: int = 0
    critical_blunders: int = 0
    
    top1_alignment: float = 0.0
    top3_alignment: float = 0.0
    top5_alignment: float = 0.0
    
    # Opening performance
    opening_performance: Dict[str, Dict] = field(default_factory=dict)
    
    # Opponent analysis
    best_opponents: List[Dict] = field(default_factory=list)
    worst_opponents: List[Dict] = field(default_factory=list)
    
    # Theme adherence
    theme_stats: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "games": {
                "total": self.total_games,
                "wins": self.wins,
                "losses": self.losses,
                "draws": self.draws,
                "win_rate": round(self.wins / max(1, self.total_games) * 100, 1)
            },
            "performance": {
                "moves_analyzed": self.total_moves_analyzed,
                "avg_centipawn_loss": round(self.avg_centipawn_loss, 1),
                "move_quality": {
                    "best": self.best_moves,
                    "excellent": self.excellent_moves,
                    "good": self.good_moves,
                    "inaccuracies": self.inaccuracies,
                    "mistakes": self.mistakes,
                    "blunders": self.blunders,
                    "critical_blunders": self.critical_blunders
                },
                "stockfish_alignment": {
                    "top1": round(self.top1_alignment, 1),
                    "top3": round(self.top3_alignment, 1),
                    "top5": round(self.top5_alignment, 1)
                }
            },
            "openings": self.opening_performance,
            "opponents": {
                "best_performances": self.best_opponents,
                "worst_performances": self.worst_opponents
            },
            "themes": self.theme_stats
        }


class ReportGenerator:
    """Generates weekly analytics reports."""
    
    def __init__(self, bot_username: str = "v7p3r_bot"):
        """Initialize report generator."""
        self.bot_username = bot_username
    
    def generate_weekly_report(
        self, 
        reports: List[GameAnalysisReport],
        output_path: str
    ) -> Dict:
        """
        Generate comprehensive weekly report from game analyses.
        
        Args:
            reports: List of GameAnalysisReport objects
            output_path: Path to save report JSON
            
        Returns:
            Report dictionary
        """
        if not reports:
            logger.warning("No reports to analyze")
            return {}
        
        stats = WeeklyStats()
        stats.total_games = len(reports)
        
        # Aggregate data
        centipawn_losses = []
        top1_scores = []
        top3_scores = []
        top5_scores = []
        opponent_stats = {}
        opening_stats = {}
        theme_totals = {
            "castling_kingside": 0,
            "castling_queenside": 0,
            "fianchetto": 0,
            "isolated_pawns": 0,
            "passed_pawns": 0,
            "bishop_pair_games": 0,
            "knight_outpost": 0,
            "rook_open_file": 0,
            "tactical_themes": 0
        }
        
        for report in reports:
            # Results
            if report.result == "1-0" and report.color == "white":
                stats.wins += 1
            elif report.result == "0-1" and report.color == "black":
                stats.wins += 1
            elif report.result == "1/2-1/2":
                stats.draws += 1
            else:
                stats.losses += 1
            
            # Move statistics
            stats.total_moves_analyzed += len(report.moves)
            stats.best_moves += report.best_moves
            stats.excellent_moves += report.excellent_moves
            stats.good_moves += report.good_moves
            stats.inaccuracies += report.inaccuracies
            stats.mistakes += report.mistakes
            stats.blunders += report.blunders
            stats.critical_blunders += report.critical_blunders
            
            centipawn_losses.append(report.average_centipawn_loss)
            top1_scores.append(report.top1_alignment)
            top3_scores.append(report.top3_alignment)
            top5_scores.append(report.top5_alignment)
            
            # Opponent tracking
            opp = report.opponent
            if opp not in opponent_stats:
                opponent_stats[opp] = {
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "avg_cpl": []
                }
            
            opponent_stats[opp]["games"] += 1
            opponent_stats[opp]["avg_cpl"].append(report.average_centipawn_loss)
            
            if report.result == "1-0" and report.color == "white":
                opponent_stats[opp]["wins"] += 1
            elif report.result == "0-1" and report.color == "black":
                opponent_stats[opp]["wins"] += 1
            elif report.result == "1/2-1/2":
                opponent_stats[opp]["draws"] += 1
            else:
                opponent_stats[opp]["losses"] += 1
            
            # Opening tracking
            opening = report.opening
            if opening not in opening_stats:
                opening_stats[opening] = {
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "avg_cpl": []
                }
            
            opening_stats[opening]["games"] += 1
            opening_stats[opening]["avg_cpl"].append(report.average_centipawn_loss)
            
            if report.result == "1-0" and report.color == "white":
                opening_stats[opening]["wins"] += 1
            elif report.result == "0-1" and report.color == "black":
                opening_stats[opening]["wins"] += 1
            elif report.result == "1/2-1/2":
                opening_stats[opening]["draws"] += 1
            else:
                opening_stats[opening]["losses"] += 1
            
            # Theme tracking
            themes = report.themes.to_dict()
            theme_totals["castling_kingside"] += themes["castling"]["kingside"]
            theme_totals["castling_queenside"] += themes["castling"]["queenside"]
            theme_totals["isolated_pawns"] += themes["pawn_structure"]["isolated"]
            theme_totals["passed_pawns"] += themes["pawn_structure"]["passed"]
            if themes["pieces"]["bishop_pair"]:
                theme_totals["bishop_pair_games"] += 1
            theme_totals["knight_outpost"] += themes["pieces"]["knight_outpost"]
            theme_totals["rook_open_file"] += themes["pieces"]["rook_open_file"]
            theme_totals["tactical_themes"] += (
                themes["tactics"]["pin"] +
                themes["tactics"]["fork"] +
                themes["tactics"]["skewer"] +
                themes["tactics"]["battery"]
            )
        
        # Calculate averages
        stats.avg_centipawn_loss = sum(centipawn_losses) / len(centipawn_losses)
        stats.top1_alignment = sum(top1_scores) / len(top1_scores)
        stats.top3_alignment = sum(top3_scores) / len(top3_scores)
        stats.top5_alignment = sum(top5_scores) / len(top5_scores)
        
        # Process opening performance
        for opening, data in opening_stats.items():
            win_rate = data["wins"] / data["games"] * 100
            avg_cpl = sum(data["avg_cpl"]) / len(data["avg_cpl"])
            stats.opening_performance[opening] = {
                "games": data["games"],
                "win_rate": round(win_rate, 1),
                "avg_cpl": round(avg_cpl, 1),
                "record": f"{data['wins']}-{data['losses']}-{data['draws']}"
            }
        
        # Best and worst opponents
        for opp, data in opponent_stats.items():
            win_rate = data["wins"] / data["games"] * 100
            avg_cpl = sum(data["avg_cpl"]) / len(data["avg_cpl"])
            opp_data = {
                "name": opp,
                "games": data["games"],
                "win_rate": round(win_rate, 1),
                "avg_cpl": round(avg_cpl, 1),
                "record": f"{data['wins']}-{data['losses']}-{data['draws']}"
            }
            stats.best_opponents.append(opp_data)
            stats.worst_opponents.append(opp_data)
        
        # Sort opponents
        stats.best_opponents.sort(key=lambda x: (x["win_rate"], -x["avg_cpl"]), reverse=True)
        stats.worst_opponents.sort(key=lambda x: (x["win_rate"], -x["avg_cpl"]))
        stats.best_opponents = stats.best_opponents[:5]
        stats.worst_opponents = stats.worst_opponents[:5]
        
        # Theme statistics
        stats.theme_stats = theme_totals
        
        # Convert to dict and save
        report_dict = stats.to_dict()
        report_dict["metadata"] = {
            "generated": datetime.now().isoformat(),
            "period": f"Last {len(reports)} games",
            "bot": self.bot_username
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Report saved to {output_file}")
        
        # Generate markdown summary
        self._generate_markdown_summary(report_dict, output_file.with_suffix('.md'))
        
        return report_dict
    
    def _generate_markdown_summary(self, report: Dict, output_path: Path):
        """Generate human-readable markdown summary."""
        md_lines = [
            f"# V7P3R Weekly Analytics Report",
            f"\n**Generated:** {report['metadata']['generated']}",
            f"\n**Period:** {report['metadata']['period']}",
            f"\n---",
            f"\n## Overall Performance",
            f"\n- **Games Played:** {report['games']['total']}",
            f"- **Record:** {report['games']['wins']}W / {report['games']['losses']}L / {report['games']['draws']}D",
            f"- **Win Rate:** {report['games']['win_rate']}%",
            f"\n### Move Quality",
            f"\n- **Average Centipawn Loss:** {report['performance']['avg_centipawn_loss']}",
            f"- **Best Moves:** {report['performance']['move_quality']['best']}",
            f"- **Excellent Moves:** {report['performance']['move_quality']['excellent']}",
            f"- **Good Moves:** {report['performance']['move_quality']['good']}",
            f"- **Inaccuracies:** {report['performance']['move_quality']['inaccuracies']}",
            f"- **Mistakes:** {report['performance']['move_quality']['mistakes']}",
            f"- **Blunders:** {report['performance']['move_quality']['blunders']}",
            f"- **Critical Blunders:** {report['performance']['move_quality']['critical_blunders']}",
            f"\n### Stockfish Alignment",
            f"\n- **Top 1 Move:** {report['performance']['stockfish_alignment']['top1']}%",
            f"- **Top 3 Moves:** {report['performance']['stockfish_alignment']['top3']}%",
            f"- **Top 5 Moves:** {report['performance']['stockfish_alignment']['top5']}%",
            f"\n## Opening Performance",
            f"\n| Opening | Games | Win Rate | Avg CPL | Record |",
            f"|---------|-------|----------|---------|--------|"
        ]
        
        for opening, data in sorted(
            report['openings'].items(), 
            key=lambda x: x[1]['games'], 
            reverse=True
        )[:10]:
            md_lines.append(
                f"| {opening[:40]} | {data['games']} | {data['win_rate']}% | {data['avg_cpl']} | {data['record']} |"
            )
        
        md_lines.extend([
            f"\n## Opponent Analysis",
            f"\n### Best Performances",
            f"\n| Opponent | Games | Win Rate | Avg CPL | Record |",
            f"|----------|-------|----------|---------|--------|"
        ])
        
        for opp in report['opponents']['best_performances']:
            md_lines.append(
                f"| {opp['name']} | {opp['games']} | {opp['win_rate']}% | {opp['avg_cpl']} | {opp['record']} |"
            )
        
        md_lines.extend([
            f"\n### Needs Improvement",
            f"\n| Opponent | Games | Win Rate | Avg CPL | Record |",
            f"|----------|-------|----------|---------|--------|"
        ])
        
        for opp in report['opponents']['worst_performances']:
            md_lines.append(
                f"| {opp['name']} | {opp['games']} | {opp['win_rate']}% | {opp['avg_cpl']} | {opp['record']} |"
            )
        
        md_lines.extend([
            f"\n## Chess Themes Detected",
            f"\n- **Castling (Kingside):** {report['themes']['castling_kingside']} instances",
            f"- **Castling (Queenside):** {report['themes']['castling_queenside']} instances",
            f"- **Isolated Pawns:** {report['themes']['isolated_pawns']} detected",
            f"- **Passed Pawns:** {report['themes']['passed_pawns']} detected",
            f"- **Bishop Pair:** {report['themes']['bishop_pair_games']} games",
            f"- **Knight Outposts:** {report['themes']['knight_outpost']} instances",
            f"- **Rook on Open File:** {report['themes']['rook_open_file']} instances",
            f"- **Tactical Themes:** {report['themes']['tactical_themes']} (pins/forks/skewers/batteries)",
            f"\n---",
            f"\n## Recommendations",
            f"\n### Strengths to Maintain",
            f"- Continue exploiting best-performing openings",
            f"- Maintain high top-5 alignment with Stockfish recommendations",
            f"\n### Areas for Improvement",
            f"- Reduce critical blunders through deeper position evaluation",
            f"- Improve theme recognition for better positional play",
            f"- Focus on opponents with losing records for strategic adjustments",
            f"\n---",
            f"\n*Generated by V7P3R Analytics System*"
        ])
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(md_lines))
        
        logger.info(f"Markdown summary saved to {output_path}")


if __name__ == "__main__":
    # Test report generation
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python report_generator.py <analysis_dir> <output_path>")
        sys.exit(1)
    
    analysis_dir = Path(sys.argv[1])
    output_path = sys.argv[2]
    
    # Load analysis reports
    reports = []
    for json_file in analysis_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            # Convert back to GameAnalysisReport (simplified for testing)
            # In production, use proper deserialization
            reports.append(data)
    
    generator = ReportGenerator()
    # Note: This test would need proper GameAnalysisReport objects
    # generator.generate_weekly_report(reports, output_path)
    print(f"Would generate report from {len(reports)} games")
