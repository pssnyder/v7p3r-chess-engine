#!/usr/bin/env python3
"""
Monitor V7P3R Analytics Progress
Quick status check for running analysis
"""
import subprocess
import sys
import time
from pathlib import Path

def check_progress():
    """Check analytics container progress."""
    try:
        # Check if container is running
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=v7p3r-analytics", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        
        if not result.stdout.strip():
            print("❌ Container not running")
            return False
        
        print(f"✓ Container status: {result.stdout.strip()}")
        
        # Get last 20 lines of logs
        logs = subprocess.run(
            ["docker", "logs", "v7p3r-analytics", "--tail", "20"],
            capture_output=True,
            text=True
        )
        
        print("\n📊 Recent Progress:")
        print("-" * 70)
        print(logs.stdout)
        
        # Check for completion
        if "✓ All stages complete" in logs.stdout or "COMPLETE" in logs.stdout:
            print("\n🎉 Analysis COMPLETE!")
            
            # Show report location
            reports_dir = Path("analytics_reports/2025")
            if reports_dir.exists():
                latest = max(reports_dir.glob("week_*"), key=lambda p: p.stat().st_mtime)
                print(f"\n📁 Reports: {latest}")
                print(f"   - {latest / 'technical_report.md'}")
                print(f"   - {latest / 'weekly_summary.json'}")
            
            return True
        
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("V7P3R Analytics Progress Monitor")
    print("=" * 70)
    
    complete = check_progress()
    
    if not complete:
        print("\n⏳ Analysis still running...")
        print("   Run this script again to check progress")
        print("   Or: docker logs v7p3r-analytics -f (to follow live)")
