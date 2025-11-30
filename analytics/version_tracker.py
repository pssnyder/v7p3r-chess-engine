"""
Version Tracker for V7P3R Analytics
Maps games to engine versions based on game timestamps
"""

from datetime import datetime
from typing import Optional, Dict, Any
import json
from pathlib import Path


class VersionTracker:
    """Maps game timestamps to engine versions based on deployment history."""
    
    # Deployment timeline based on CHANGELOG.md
    VERSION_TIMELINE = [
        # Format: (start_date, end_date, version, notes)
        ("2025-11-30", None, "v17.1", "Rolled back from v17.4 - CURRENT"),
        ("2025-11-26", "2025-11-29", "v17.4", "ROLLED BACK - endgame issues"),
        ("2025-11-21", "2025-11-26", "v17.2.0", "Stable deployment"),
        ("2025-11-21", "2025-11-21", "v17.1.1", "Short-lived patch"),
        ("2025-11-21", "2025-11-21", "v17.1", "Initial v17.1 deployment"),
        ("2025-11-20", "2025-11-21", "v17.0", "Initial v17 series"),
        ("2025-11-19", "2025-11-20", "v16.1", "Last of v16 series"),
        ("2025-10-25", "2025-11-19", "v14.1", "Stable 25-day run"),
        ("2025-10-25", "2025-10-25", "v14.0", "Short-lived"),
        ("2025-10-04", "2025-10-25", "v12.6", "Stable 21-day run"),
        ("2025-10-03", "2025-10-04", "v12.4", "Short-lived"),
        ("2025-10-03", "2025-10-03", "v12.2", "Initial v12 series"),
    ]
    
    def __init__(self):
        """Initialize version tracker."""
        self.timeline = self._parse_timeline()
    
    def _parse_timeline(self) -> list:
        """Parse timeline into datetime objects."""
        parsed = []
        for start_str, end_str, version, notes in self.VERSION_TIMELINE:
            start_date = datetime.strptime(start_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_str, "%Y-%m-%d") if end_str else None
            parsed.append({
                "start": start_date,
                "end": end_date,
                "version": version,
                "notes": notes
            })
        return parsed
    
    def get_version_for_game(self, game_date: datetime) -> Optional[Dict[str, Any]]:
        """
        Determine which engine version played a game based on timestamp.
        
        Args:
            game_date: Game datetime
            
        Returns:
            Dict with version info, or None if no match
        """
        for entry in self.timeline:
            start = entry["start"]
            end = entry["end"]
            
            # Check if game falls within this version's timeframe
            if end is None:
                # Current version (no end date)
                if game_date >= start:
                    return {
                        "version": entry["version"],
                        "notes": entry["notes"],
                        "deployment_date": start.strftime("%Y-%m-%d"),
                        "status": "current"
                    }
            else:
                # Historical version
                if start <= game_date <= end:
                    return {
                        "version": entry["version"],
                        "notes": entry["notes"],
                        "deployment_date": start.strftime("%Y-%m-%d"),
                        "retirement_date": end.strftime("%Y-%m-%d"),
                        "status": "retired"
                    }
        
        # No match found
        return None
    
    def get_version_for_game_id(self, game_id: str, game_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get version info for a game using its metadata.
        
        Args:
            game_id: Lichess game ID
            game_metadata: Game metadata dict with 'date' or 'datetime' field
            
        Returns:
            Dict with version info, or None if no match
        """
        # Extract date from metadata
        date_str = game_metadata.get("date") or game_metadata.get("datetime")
        if not date_str:
            return None
        
        # Parse date
        try:
            # Try full datetime first
            if "T" in date_str or " " in date_str:
                # ISO format or similar
                game_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                # Date only - try multiple formats
                for fmt in ["%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"]:
                    try:
                        game_date = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # No format matched
                    return None
        except ValueError:
            return None
        
        return self.get_version_for_game(game_date)
    
    def filter_games_by_version(self, games: list, target_version: str) -> list:
        """
        Filter list of games to only those played by specific version.
        
        Args:
            games: List of game dicts with metadata
            target_version: Version to filter for (e.g., "v17.4")
            
        Returns:
            Filtered list of games
        """
        filtered = []
        for game in games:
            version_info = self.get_version_for_game_id(
                game.get("id", "unknown"),
                game
            )
            if version_info and version_info["version"] == target_version:
                filtered.append(game)
        return filtered
    
    def get_version_statistics(self, games: list) -> Dict[str, Any]:
        """
        Generate statistics about which versions played which games.
        
        Args:
            games: List of game dicts with metadata
            
        Returns:
            Dict with version statistics
        """
        version_counts = {}
        unmatched = 0
        
        for game in games:
            version_info = self.get_version_for_game_id(
                game.get("id", "unknown"),
                game
            )
            if version_info:
                version = version_info["version"]
                if version not in version_counts:
                    version_counts[version] = {
                        "count": 0,
                        "notes": version_info["notes"],
                        "games": []
                    }
                version_counts[version]["count"] += 1
                version_counts[version]["games"].append(game.get("id", "unknown"))
            else:
                unmatched += 1
        
        return {
            "total_games": len(games),
            "matched_games": len(games) - unmatched,
            "unmatched_games": unmatched,
            "versions": version_counts
        }
    
    def export_version_timeline(self, output_path: Path):
        """
        Export version timeline to JSON for reference.
        
        Args:
            output_path: Path to save JSON file
        """
        timeline_export = []
        for entry in self.timeline:
            timeline_export.append({
                "version": entry["version"],
                "start_date": entry["start"].strftime("%Y-%m-%d"),
                "end_date": entry["end"].strftime("%Y-%m-%d") if entry["end"] else "CURRENT",
                "notes": entry["notes"]
            })
        
        with open(output_path, 'w') as f:
            json.dump({
                "generated": datetime.now().isoformat(),
                "timeline": timeline_export
            }, f, indent=2)


def main():
    """Demo version tracker functionality."""
    tracker = VersionTracker()
    
    # Test with some sample dates
    test_dates = [
        ("2025-11-30", "Should be v17.1 (rollback)"),
        ("2025-11-28", "Should be v17.4 (rolled back)"),
        ("2025-11-25", "Should be v17.2.0"),
        ("2025-11-21", "Should be v17.1.1 or v17.1"),
        ("2025-11-01", "Should be v14.1"),
    ]
    
    print("Version Tracker - Demo\n")
    print("=" * 60)
    
    for date_str, expected in test_dates:
        game_date = datetime.strptime(date_str, "%Y-%m-%d")
        version_info = tracker.get_version_for_game(game_date)
        
        print(f"\nDate: {date_str}")
        print(f"Expected: {expected}")
        if version_info:
            print(f"Result: {version_info['version']} - {version_info['notes']}")
        else:
            print("Result: No match found")
    
    print("\n" + "=" * 60)
    
    # Export timeline
    output_path = Path(__file__).parent / "version_timeline.json"
    tracker.export_version_timeline(output_path)
    print(f"\n✓ Exported timeline to: {output_path}")


if __name__ == "__main__":
    main()
