"""Quick progress checker for long-running analysis"""
import time
import os
from pathlib import Path

analysis_log = Path("analysis_log.txt")

if not analysis_log.exists():
    print("No analysis log found yet...")
    exit(0)

# Read last 30 lines
with open(analysis_log) as f:
    lines = f.readlines()
    recent = lines[-30:]
    
print("\n" + "="*60)
print("V7P3R Analysis Progress")
print("="*60)

for line in recent:
    if "[" in line and "/" in line and "]" in line:
        # Progress line
        print(line.strip())
    elif "✓" in line or "Version:" in line:
        print("  " + line.strip())
    elif "Checkpoint" in line:
        print("\n" + line.strip())
    elif "ERROR" in line or "Failed" in line:
        print("⚠️  " + line.strip())

print("\n" + "="*60)

# Show file size
log_size = os.path.getsize(analysis_log) / 1024
print(f"\nLog file size: {log_size:.1f} KB")
print("\nRun this script again to see updated progress.")
