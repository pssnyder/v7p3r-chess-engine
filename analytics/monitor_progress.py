#!/usr/bin/env python3
"""
Check progress of running analysis
"""
import time
from pathlib import Path

print("Monitoring analysis progress...\n")
print("Press Ctrl+C to stop monitoring (analysis will continue)\n")

temp_file = Path("temp_game.pgn")
last_size = 0

try:
    while True:
        # Check if temp file exists (means analysis is active)
        if temp_file.exists():
            current_size = temp_file.stat().st_size
            if current_size != last_size:
                print(f"[{time.strftime('%H:%M:%S')}] Analysis in progress...")
                last_size = current_size
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Waiting for analysis to start or completed...")
        
        time.sleep(10)
except KeyboardInterrupt:
    print("\n\nMonitoring stopped. Analysis continues in background.")
