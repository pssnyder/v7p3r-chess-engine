"""Monitor parallel analysis progress"""
import time
from pathlib import Path

log_file = Path("parallel_analysis_log.txt")

if not log_file.exists():
    print("No log file found yet...")
    exit(0)

with open(log_file) as f:
    lines = f.readlines()

print("\n" + "="*70)
print("V7P3R Parallel Analysis Progress")
print("="*70)

# Find key metrics
for line in reversed(lines[-50:]):
    if "Progress:" in line:
        print("\n" + line.strip())
    elif "Rate:" in line and "games/sec" in line:
        print(line.strip())
    elif "ETA:" in line:
        print(line.strip())
        break

# Show recent completions
print("\nRecent completions:")
count = 0
for line in reversed(lines):
    if "CPL:" in line and "Top1:" in line:
        print("  " + line.strip())
        count += 1
        if count >= 5:
            break

print("\n" + "="*70)
