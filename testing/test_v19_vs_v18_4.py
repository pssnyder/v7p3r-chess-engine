#!/usr/bin/env python3
"""
V19.0 vs V18.4 Validation Test

WHY THIS EXISTS: Automated validation that v19.0 improvements actually work
and that the engine is ready for lichess deployment.

WHAT IT DOES:
- Runs 30 games between v19.0 and v18.4
- Uses 5min+4s blitz (where v18.4 had 75% timeout rate)
- Tracks wins/losses/draws and critical issues
- Provides deployment recommendation

SUCCESS CRITERIA:
- ✓ v19.0 scores ≥45% (maintains strength)
- ✓ v19.0 has 0 timeouts (fixes 75% timeout bug)
- ✓ No crashes or UCI errors (stability)
"""

import sys
import subprocess
from pathlib import Path

# Engine paths
V19_ENGINE = Path("src/v7p3r_uci.py")
V18_4_ENGINE = Path("../Tournament Engines/V7P3R/V7P3R_v18.4/src/v7p3r_uci.py")

# Alternative paths if not found
V18_4_ALT = Path("lichess/engines/V7P3R_v18.4_20260417/src/v7p3r_uci.py")

def find_engine(primary: Path, alternative: Path = None) -> Path:
    """Find engine file, checking alternative paths"""
    if primary.exists():
        return primary.resolve()
    if alternative and alternative.exists():
        return alternative.resolve()
    return None

def main():
    print("=" * 80)
    print("V19.0 VS V18.4 VALIDATION TEST")
    print("=" * 80)
    
    # Locate engines
    print("\n1. Locating engines...")
    
    v19_path = find_engine(V19_ENGINE)
    if not v19_path:
        print(f"  ✗ ERROR: v19.0 engine not found at {V19_ENGINE}")
        print(f"    Expected location: E:\\Programming Stuff\\Chess Engines\\V7P3R Chess Engine\\v7p3r-chess-engine\\src\\v7p3r_uci.py")
        sys.exit(1)
    print(f"  ✓ v19.0 found: {v19_path}")
    
    v18_4_path = find_engine(V18_4_ENGINE, V18_4_ALT)
    if not v18_4_path:
        print(f"  ✗ ERROR: v18.4 engine not found")
        print(f"    Tried: {V18_4_ENGINE}")
        if V18_4_ALT:
            print(f"    Tried: {V18_4_ALT}")
        print(f"\n  Please ensure v18.4 is available in Tournament Engines or lichess/engines/")
        sys.exit(1)
    print(f"  ✓ v18.4 found: {v18_4_path}")
    
    # Prepare python command
    python_cmd = sys.executable  # Use same python as this script
    
    # Build tournament command
    tournament_script = Path("testing/tournament_runner.py")
    if not tournament_script.exists():
        print(f"  ✗ ERROR: Tournament runner not found at {tournament_script}")
        sys.exit(1)
    
    cmd = [
        python_cmd,
        str(tournament_script),
        "--engine1", python_cmd, str(v19_path),
        "--engine2", python_cmd, str(v18_4_path),
        "--games", "30",
        "--time", "300",  # 5 minutes
        "--increment", "4",  # 4 seconds
        "--output", "tournament_results"
    ]
    
    print("\n2. Tournament Configuration:")
    print(f"  Games: 30 (15 as White, 15 as Black)")
    print(f"  Time Control: 5min + 4s increment (blitz)")
    print(f"  Engine 1 (v19.0): {v19_path.name}")
    print(f"  Engine 2 (v18.4): {v18_4_path.name}")
    
    print("\n3. Starting tournament...")
    print("  (This will take approximately 45-60 minutes)")
    print("=" * 80)
    
    # Run tournament
    try:
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "=" * 80)
        print("DEPLOYMENT RECOMMENDATION")
        print("=" * 80)
        print("\nReview the tournament results above:")
        print("  ✓ If v19.0 scored ≥45% and had 0 timeouts → READY for deployment")
        print("  ✗ If v19.0 scored <45% → Review evaluation/search changes")
        print("  ✗ If v19.0 had timeouts → Review TimeManager allocations")
        print("  ✗ If crashes occurred → Review stability issues")
        
        print("\nNext steps if READY:")
        print("  1. Review docs/V19_PHASE1_2_COMPLETION.md")
        print("  2. Tag v19.0.0 release candidate")
        print("  3. Deploy to lichess following version_management.instructions.md")
        print("=" * 80)
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Tournament failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTournament interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
