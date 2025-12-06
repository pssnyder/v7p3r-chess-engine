"""
Cloud Storage Manager for V7P3R Analytics
Handles persistent storage of reports and historical data
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import subprocess

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages Cloud Storage operations for analytics reports."""
    
    def __init__(self, bucket_name: str = "v7p3r-analytics-reports"):
        """
        Initialize storage manager.
        
        Args:
            bucket_name: GCS bucket name
        """
        self.bucket_name = bucket_name
        self.bucket_uri = f"gs://{bucket_name}"
        
    def get_week_folder(self, date: datetime = None) -> str:
        """
        Get the folder path for a specific week.
        
        Args:
            date: Date to get week folder for (defaults to now)
            
        Returns:
            Path like "weekly/2025/week_49_2025-12-01"
        """
        if date is None:
            date = datetime.now()
            
        year = date.year
        week_num = date.isocalendar()[1]
        # Get Monday of the week
        monday = date - timedelta(days=date.weekday())
        monday_str = monday.strftime("%Y-%m-%d")
        
        return f"weekly/{year}/week_{week_num:02d}_{monday_str}"
    
    def upload_report_folder(self, local_dir: Path, week_folder: str = None) -> bool:
        """
        Upload entire report folder to Cloud Storage.
        
        Args:
            local_dir: Local directory containing reports
            week_folder: Week folder path (auto-generated if None)
            
        Returns:
            True if successful
        """
        if week_folder is None:
            week_folder = self.get_week_folder()
            
        dest_uri = f"{self.bucket_uri}/{week_folder}/"
        
        try:
            logger.info(f"Uploading {local_dir} to {dest_uri}")
            
            cmd = [
                "gsutil", "-m", "cp", "-r",
                f"{local_dir}/*",
                dest_uri
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Upload successful: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Upload failed: {e.stderr}")
            return False
    
    def download_historical_summary(self) -> Optional[Dict]:
        """
        Download historical summary from Cloud Storage.
        
        Returns:
            Historical summary dict or None if not found
        """
        uri = f"{self.bucket_uri}/historical_summary.json"
        local_file = Path("/tmp/historical_summary.json")
        
        try:
            cmd = ["gsutil", "cp", uri, str(local_file)]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            with open(local_file, 'r') as f:
                return json.load(f)
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not download historical summary: {e.stderr}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in historical summary: {e}")
            return None
    
    def upload_historical_summary(self, summary: Dict) -> bool:
        """
        Upload updated historical summary to Cloud Storage.
        
        Args:
            summary: Historical summary dict
            
        Returns:
            True if successful
        """
        uri = f"{self.bucket_uri}/historical_summary.json"
        local_file = Path("/tmp/historical_summary_upload.json")
        
        try:
            # Save locally first
            with open(local_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Upload
            cmd = ["gsutil", "cp", str(local_file), uri]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("Historical summary uploaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload historical summary: {e}")
            return False
    
    def update_historical_summary(self, week_data: Dict) -> bool:
        """
        Update historical summary with new week's data.
        
        Args:
            week_data: Week summary containing metrics
            
        Returns:
            True if successful
        """
        # Download current summary
        summary = self.download_historical_summary()
        
        if summary is None:
            logger.info("Creating new historical summary")
            summary = {
                "weeks": [],
                "last_updated": datetime.now().isoformat(),
                "total_weeks": 0
            }
        
        # Add new week
        summary["weeks"].append(week_data)
        summary["total_weeks"] = len(summary["weeks"])
        summary["last_updated"] = datetime.now().isoformat()
        
        # Keep only last 52 weeks (1 year)
        if len(summary["weeks"]) > 52:
            summary["weeks"] = summary["weeks"][-52:]
        
        # Upload updated summary
        return self.upload_historical_summary(summary)
    
    def list_weeks(self, limit: int = 10) -> List[str]:
        """
        List available weeks in storage.
        
        Args:
            limit: Maximum number of weeks to return
            
        Returns:
            List of week folder paths
        """
        try:
            cmd = ["gsutil", "ls", f"{self.bucket_uri}/weekly/"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse output
            folders = []
            for line in result.stdout.strip().split('\n'):
                if line and 'week_' in line:
                    folders.append(line.strip())
            
            return sorted(folders, reverse=True)[:limit]
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list weeks: {e.stderr}")
            return []
    
    def download_week_data(self, week_folder: str, local_dir: Path) -> bool:
        """
        Download a specific week's data.
        
        Args:
            week_folder: Week folder path
            local_dir: Local directory to download to
            
        Returns:
            True if successful
        """
        src_uri = f"{self.bucket_uri}/{week_folder}/*"
        
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                "gsutil", "-m", "cp", "-r",
                src_uri,
                str(local_dir)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Downloaded {week_folder} to {local_dir}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e.stderr}")
            return False
    
    def get_recent_weeks_summary(self, num_weeks: int = 4) -> List[Dict]:
        """
        Get summary data for recent weeks.
        
        Args:
            num_weeks: Number of recent weeks to retrieve
            
        Returns:
            List of week summary dicts
        """
        summary = self.download_historical_summary()
        
        if not summary or not summary.get("weeks"):
            return []
        
        return summary["weeks"][-num_weeks:]


def main():
    """Test storage manager functionality."""
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Test Storage Manager")
    parser.add_argument("--list", action="store_true", help="List weeks")
    parser.add_argument("--download-summary", action="store_true", help="Download historical summary")
    parser.add_argument("--upload-test", type=str, help="Upload test folder")
    
    args = parser.parse_args()
    
    manager = StorageManager()
    
    if args.list:
        weeks = manager.list_weeks()
        print(f"\nAvailable weeks ({len(weeks)}):")
        for week in weeks:
            print(f"  {week}")
    
    if args.download_summary:
        summary = manager.download_historical_summary()
        if summary:
            print(f"\nHistorical Summary:")
            print(f"  Total weeks: {summary.get('total_weeks', 0)}")
            print(f"  Last updated: {summary.get('last_updated', 'N/A')}")
            if summary.get("weeks"):
                print(f"  Recent weeks:")
                for week in summary["weeks"][-5:]:
                    print(f"    - {week.get('week_folder', 'N/A')}: {week.get('games', 0)} games")
    
    if args.upload_test:
        test_dir = Path(args.upload_test)
        if test_dir.exists():
            success = manager.upload_report_folder(test_dir)
            print(f"\nUpload {'successful' if success else 'failed'}")
        else:
            print(f"\nError: {test_dir} does not exist")


if __name__ == "__main__":
    main()
