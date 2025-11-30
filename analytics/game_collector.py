"""
Game Collector - Fetch PGN files from GCP VM storage
Retrieves recent games from v7p3r-lichess-bot production environment
"""
import subprocess
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameCollector:
    """Collects PGN files from GCP infrastructure."""
    
    def __init__(
        self, 
        project_id: str = "v7p3r-lichess-bot",
        instance_name: str = "v7p3r-production-bot",
        zone: str = "us-central1-a",
        container_name: str = "v7p3r-production",
        remote_game_dir: str = "/lichess-bot/game_records"
    ):
        """Initialize collector with GCP configuration."""
        self.project_id = project_id
        self.instance_name = instance_name
        self.zone = zone
        self.container_name = container_name
        self.remote_game_dir = remote_game_dir
    
    def fetch_recent_games(
        self, 
        local_dir: str, 
        days_back: int = 7
    ) -> list[Path]:
        """
        Fetch recent PGN files from GCP VM.
        
        Args:
            local_dir: Local directory to store PGN files
            days_back: How many days of games to fetch
            
        Returns:
            List of downloaded PGN file paths
        """
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate date threshold
        date_threshold = datetime.now() - timedelta(days=days_back)
        date_pattern = date_threshold.strftime("%Y%m")  # YYYYMM format
        
        logger.info(f"Fetching games from last {days_back} days (since {date_threshold.date()})")
        
        # Step 1: Copy entire game_records directory to VM temp
        logger.info("Copying game records from container to VM...")
        copy_cmd = [
            "gcloud", "compute", "ssh", self.instance_name,
            f"--zone={self.zone}",
            f"--project={self.project_id}",
            "--command",
            f"sudo docker cp {self.container_name}:{self.remote_game_dir} /tmp/game_records_export"
        ]
        
        result = subprocess.run(copy_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to copy from container: {result.stderr}")
            return []
        
        # Step 2: Use gcloud compute scp to download to local
        logger.info(f"Downloading to {local_path}...")
        scp_cmd = [
            "gcloud", "compute", "scp",
            "--recurse",
            f"{self.instance_name}:/tmp/game_records_export/*",
            str(local_path),
            f"--zone={self.zone}",
            f"--project={self.project_id}"
        ]
        
        result = subprocess.run(scp_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to download files: {result.stderr}")
            return []
        
        # Step 3: Clean up VM temp directory
        cleanup_cmd = [
            "gcloud", "compute", "ssh", self.instance_name,
            f"--zone={self.zone}",
            f"--project={self.project_id}",
            "--command",
            "sudo rm -rf /tmp/game_records_export"
        ]
        subprocess.run(cleanup_cmd, capture_output=True, text=True)
        
        # Step 4: Find recent PGN files
        pgn_files = []
        for pgn_file in local_path.rglob("*.pgn"):
            # Check if file is recent enough
            file_date = datetime.fromtimestamp(pgn_file.stat().st_mtime)
            if file_date >= date_threshold:
                pgn_files.append(pgn_file)
        
        logger.info(f"Found {len(pgn_files)} recent PGN files")
        return sorted(pgn_files)
    
    def fetch_specific_game(self, game_id: str, local_dir: str) -> Optional[Path]:
        """
        Fetch a specific game by ID.
        
        Args:
            game_id: Lichess game ID
            local_dir: Local directory to store PGN
            
        Returns:
            Path to downloaded PGN or None
        """
        # Search for game in remote directory
        search_cmd = [
            "gcloud", "compute", "ssh", self.instance_name,
            f"--zone={self.zone}",
            f"--project={self.project_id}",
            "--command",
            f"sudo docker exec {self.container_name} find {self.remote_game_dir} -name '*{game_id}*.pgn'"
        ]
        
        result = subprocess.run(search_cmd, capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            logger.warning(f"Game {game_id} not found")
            return None
        
        remote_file = result.stdout.strip().split('\n')[0]
        local_path = Path(local_dir) / f"{game_id}.pgn"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy from container to VM
        copy_cmd = [
            "gcloud", "compute", "ssh", self.instance_name,
            f"--zone={self.zone}",
            f"--project={self.project_id}",
            "--command",
            f"sudo docker cp {self.container_name}:{remote_file} /tmp/{game_id}.pgn"
        ]
        
        result = subprocess.run(copy_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to copy game: {result.stderr}")
            return None
        
        # Download to local
        scp_cmd = [
            "gcloud", "compute", "scp",
            f"{self.instance_name}:/tmp/{game_id}.pgn",
            str(local_path),
            f"--zone={self.zone}",
            f"--project={self.project_id}"
        ]
        
        result = subprocess.run(scp_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to download: {result.stderr}")
            return None
        
        # Cleanup
        cleanup_cmd = [
            "gcloud", "compute", "ssh", self.instance_name,
            f"--zone={self.zone}",
            f"--project={self.project_id}",
            "--command",
            f"sudo rm /tmp/{game_id}.pgn"
        ]
        subprocess.run(cleanup_cmd, capture_output=True, text=True)
        
        logger.info(f"Downloaded game {game_id} to {local_path}")
        return local_path


if __name__ == "__main__":
    # Test collector
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python game_collector.py <local_directory> [days_back]")
        sys.exit(1)
    
    local_dir = sys.argv[1]
    days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    
    collector = GameCollector()
    files = collector.fetch_recent_games(local_dir, days_back)
    
    print(f"\nDownloaded {len(files)} games:")
    for f in files[:10]:  # Show first 10
        print(f"  {f.name}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")
