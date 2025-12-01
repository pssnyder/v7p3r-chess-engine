"""
Blunder Pattern Analyzer for V7P3R
Identifies recurring tactical mistakes and theme weaknesses
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Any


class BlunderPatternAnalyzer:
    """Analyzes recurring blunder patterns across games."""
    
    def __init__(self, reports_dir: str):
        """Initialize with reports directory."""
        self.reports_dir = Path(reports_dir)
        self.blunder_patterns = defaultdict(list)
        self.theme_weaknesses = defaultdict(int)
        self.position_types = defaultdict(int)
        
    def analyze_reports(self) -> Dict[str, Any]:
        """Analyze all JSON reports in directory."""
        
        # Find all JSON reports
        json_files = list(self.reports_dir.glob("v7p3r_analysis_*.json"))
        
        if not json_files:
            print(f"No JSON reports found in {self.reports_dir}")
            return {}
        
        print(f"Analyzing {len(json_files)} reports...")
        
        all_data = []
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
                all_data.append(data)
        
        # Aggregate blunder patterns
        total_blunders = 0
        total_critical_blunders = 0
        blunder_phases = {"opening": 0, "middlegame": 0, "endgame": 0}
        
        for report in all_data:
            perf = report.get("performance", {})
            moves = perf.get("move_quality", {})
            
            total_blunders += moves.get("blunders", 0)
            total_critical_blunders += moves.get("critical_blunders", 0)
        
        # Analyze theme performance
        theme_data = {}
        for report in all_data:
            themes = report.get("themes", {})
            for theme, count in themes.items():
                if theme not in theme_data:
                    theme_data[theme] = []
                theme_data[theme].append(count)
        
        # Calculate averages
        theme_averages = {}
        for theme, counts in theme_data.items():
            theme_averages[theme] = sum(counts) / len(counts) if counts else 0
        
        return {
            "total_blunders": total_blunders,
            "total_critical_blunders": total_critical_blunders,
            "blunders_per_game": total_blunders / len(all_data) if all_data else 0,
            "critical_blunders_per_game": total_critical_blunders / len(all_data) if all_data else 0,
            "theme_performance": theme_averages,
            "reports_analyzed": len(all_data)
        }
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate markdown blunder pattern report."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        md = f"""# V7P3R Blunder Pattern Analysis

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Reports Analyzed**: {analysis['reports_analyzed']}

---

## Blunder Summary

### Overall Statistics

- **Total Blunders**: {analysis['total_blunders']}
- **Total Critical Blunders**: {analysis['total_critical_blunders']}
- **Blunders per Game**: {analysis['blunders_per_game']:.1f}
- **Critical Blunders per Game**: {analysis['critical_blunders_per_game']:.1f}

### Severity Breakdown

| Type | Count | Per Game | Severity |
|------|-------|----------|----------|
| Blunders (200-400cp) | {analysis['total_blunders']} | {analysis['blunders_per_game']:.1f} | High |
| Critical Blunders (>400cp) | {analysis['total_critical_blunders']} | {analysis['critical_blunders_per_game']:.1f} | Critical |

---

## Theme Performance Analysis

### Tactical Theme Adherence

"""
        
        # Theme performance
        themes = analysis['theme_performance']
        
        md += "| Theme | Avg Count | Performance |\n"
        md += "|-------|-----------|-------------|\n"
        
        # Categorize themes
        for theme, avg_count in sorted(themes.items(), key=lambda x: x[1], reverse=True):
            if "castling" in theme:
                status = "✓ Good" if avg_count > 0.7 else "⚠ Low"
            elif "bishop_pair" in theme:
                status = "✓ Maintained"
            elif "isolated_pawns" in theme:
                status = "⚠ High" if avg_count > 50 else "✓ Acceptable"
            elif "passed_pawns" in theme:
                status = "✓ Good" if avg_count > 20 else "⚠ Low"
            else:
                status = "-"
            
            md += f"| {theme.replace('_', ' ').title()} | {avg_count:.1f} | {status} |\n"
        
        md += """

---

## Recommendations for Next Version

### High Priority Fixes

1. **Reduce Critical Blunders**
   - Current: {crit_per_game:.1f} per game
   - Target: <5.0 per game
   - Focus: Endgame evaluation, tactical awareness

2. **Improve Blunder Rate**
   - Current: {blunder_per_game:.1f} per game
   - Target: <6.0 per game
   - Focus: Position evaluation, move ordering

### Theme Improvements

""".format(
            crit_per_game=analysis['critical_blunders_per_game'],
            blunder_per_game=analysis['blunders_per_game']
        )
        
        # Identify weak themes
        weak_themes = []
        for theme, avg_count in themes.items():
            if "castling" in theme and avg_count < 0.5:
                weak_themes.append(f"- **{theme.replace('_', ' ').title()}**: Increase castling frequency")
            elif "isolated_pawns" in theme and avg_count > 100:
                weak_themes.append(f"- **{theme.replace('_', ' ').title()}**: Reduce pawn structure weaknesses")
        
        if weak_themes:
            md += "\n".join(weak_themes)
        else:
            md += "- Theme performance is within acceptable ranges\n"
        
        md += """

---

## Next Steps

1. **Heuristic Adjustments**
   - Review evaluation function for high-blunder positions
   - Adjust endgame evaluation weights
   - Increase tactical search depth for critical positions

2. **Theme Enforcement**
   - Strengthen castling incentives
   - Improve pawn structure evaluation
   - Enhance passed pawn detection and bonus

3. **Testing Protocol**
   - Run test suite against previous version
   - Validate blunder reduction
   - Confirm theme adherence improvements

---

*This report should guide the next iteration of engine development.*
"""
        
        return md
    
    def save_report(self, report: str, filename: str = None):
        """Save report to markdown file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"blunder_patterns_{timestamp}.md"
        
        output_path = self.reports_dir / filename
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Blunder pattern report saved to: {output_path}")
        return output_path


def main():
    """Run blunder pattern analysis."""
    if len(sys.argv) < 2:
        print("Usage: python blunder_pattern_analyzer.py <reports_directory>")
        sys.exit(1)
    
    reports_dir = sys.argv[1]
    
    print("=" * 70)
    print("V7P3R Blunder Pattern Analyzer")
    print("=" * 70)
    print(f"Reports Directory: {reports_dir}\n")
    
    analyzer = BlunderPatternAnalyzer(reports_dir)
    analysis = analyzer.analyze_reports()
    
    if not analysis:
        print("No analysis data generated")
        sys.exit(1)
    
    report = analyzer.generate_report(analysis)
    output_path = analyzer.save_report(report)
    
    print("\n" + "=" * 70)
    print("Blunder Analysis Complete!")
    print("=" * 70)
    print(f"\nKey Findings:")
    print(f"  - Blunders per game: {analysis['blunders_per_game']:.1f}")
    print(f"  - Critical blunders per game: {analysis['critical_blunders_per_game']:.1f}")
    print(f"  - Reports analyzed: {analysis['reports_analyzed']}")
    print(f"\nReport: {output_path}")


if __name__ == "__main__":
    main()
