# V7P3R Chess Engine - Testing Guide

This document outlines testing procedures, regression prevention, and quality assurance practices.

## Testing Philosophy

**Every deployment must pass ALL tests.** No exceptions.

Testing serves three purposes:
1. **Regression Prevention**: Ensure new versions don't break existing functionality
2. **Performance Validation**: Verify improvements are real, not statistical noise
3. **Production Readiness**: Confirm version is stable enough for 24/7 deployment

---

## Test Hierarchy

### Level 1: Regression Suite (REQUIRED)
**Purpose**: Catch known failure modes before deployment  
**Runtime**: <1 minute  
**Pass Criteria**: 100% (every test must pass)

#### Current Regression Tests
1. **Mate-in-3 Detection** (v17.4 failure case)
   - Position: Game 9i883UOF, move 23
   - Engine must find mate-in-3, NOT play Be2??
   
2. **Endgame Conversion: R+B vs K**
   - Starting position with R+B vs lone K
   - Must mate within 50 moves
   - Tests tablebase patterns and king-edge driving
   
3. **Threefold Repetition Avoidance**
   - Position where repetition draw is available
   - Engine should avoid if evaluation >50cp
   
4. **Time Forfeit Prevention**
   - Simulated time pressure scenario
   - Must make move before time runs out
   
5. **PV Instant Move Validation** (v17.0 failure)
   - Ensure PV instant moves disabled
   - Verify engine doesn't hang on instant moves

#### Creating Regression Suite
```python
# testing/regression_suite.py
import chess
from src.v7p3r import V7P3REngine

def test_mate_in_3_detection():
    """v17.4 failed this - Be2?? instead of finding mate"""
    board = chess.Board("8/8/8/8/8/4k3/8/4K2Q w - - 0 1")  # Example
    engine = V7P3REngine()
    move, score = engine.search(board, max_depth=8)
    
    # Verify mate found within 3 moves
    assert score > 9000, f"Mate not detected: score={score}"
    # Verify specific move (if known good move)
    # assert move in [chess.Move(...), ...], "Wrong move chosen"
    
    return True

def test_endgame_rb_vs_k():
    """Must mate with R+B vs K within 50 moves"""
    board = chess.Board("8/8/8/8/8/3k4/8/R2KB3 w - - 0 1")
    engine = V7P3REngine()
    
    moves = 0
    while not board.is_game_over() and moves < 50:
        move, score = engine.search(board, max_depth=10)
        board.push(move)
        moves += 1
    
    assert board.is_checkmate(), f"Failed to mate in 50 moves (took {moves})"
    return True

def run_all_tests():
    tests = [
        ("Mate-in-3 Detection", test_mate_in_3_detection),
        ("R+B vs K Endgame", test_endgame_rb_vs_k),
        # Add more tests here
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"✅ {name}")
            passed += 1
        except AssertionError as e:
            print(f"❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"💥 {name}: {e}")
            failed += 1
    
    print(f"\nResults: {passed}/{len(tests)} passed")
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
```

---

### Level 2: Performance Benchmark (REQUIRED)
**Purpose**: Validate version improves or maintains performance  
**Runtime**: 1-3 hours (50-100 games)  
**Pass Criteria**: Meets acceptance criteria

#### Acceptance Criteria (ALL must pass)
- **Win Rate**: ≥48% against equal-strength baseline
- **Blunders/Game**: ≤6.0
- **Time Forfeit Rate**: <10%
- **CPL (Centipawn Loss)**: <150 average
- **No Critical Errors**: Zero crashes, hangs, or illegal moves

#### Running Performance Benchmark

##### Option 1: Arena GUI (Windows)
```
1. Open Arena Chess GUI
2. Engines → Manage → Add engines:
   - v17.7.0 (baseline stable version)
   - v17.8.0 (new version to test)
3. Tournaments → New Tournament
   - Engines: Select both versions
   - Games per pairing: 50 minimum (100 recommended)
   - Time control: 5min+4s (blitz standard)
   - Opening book: Disabled or same for both
   - Save results to: results/v17.8_vs_v17.7_benchmark.res
4. Start tournament
5. Wait for completion (1-3 hours)
6. Analyze results
```

##### Option 2: Automated Script
```python
# testing/performance_benchmark.py
import argparse
import json
from datetime import datetime

def run_benchmark(new_version, baseline_version, games=50):
    """
    Run performance benchmark comparing two versions
    Uses Arena CLI or python-chess for automation
    """
    results = {
        "date": datetime.now().isoformat(),
        "new_version": new_version,
        "baseline_version": baseline_version,
        "games_played": games,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "blunders": [],
        "time_forfeits": 0,
        "cpl_per_game": []
    }
    
    # Run games and collect results
    # (Implementation depends on setup - Arena, cutechess-cli, or custom)
    
    # Calculate metrics
    total = results["wins"] + results["losses"] + results["draws"]
    win_rate = results["wins"] / total if total > 0 else 0
    blunders_per_game = sum(results["blunders"]) / len(results["blunders"])
    time_forfeit_rate = results["time_forfeits"] / total
    avg_cpl = sum(results["cpl_per_game"]) / len(results["cpl_per_game"])
    
    # Check acceptance criteria
    passed = (
        win_rate >= 0.48 and
        blunders_per_game <= 6.0 and
        time_forfeit_rate < 0.10 and
        avg_cpl < 150
    )
    
    results["metrics"] = {
        "win_rate": win_rate,
        "blunders_per_game": blunders_per_game,
        "time_forfeit_rate": time_forfeit_rate,
        "average_cpl": avg_cpl,
        "acceptance_criteria_met": passed
    }
    
    # Save results
    with open(f"results/benchmark_{new_version}_vs_{baseline_version}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--games", type=int, default=50)
    args = parser.parse_args()
    
    passed = run_benchmark(args.version, args.baseline, args.games)
    exit(0 if passed else 1)
```

---

### Level 3: Time Control Validation (RECOMMENDED)
**Purpose**: Ensure performance across all time formats  
**Runtime**: 2-4 hours  
**Pass Criteria**: No regression in any time control

#### Time Controls to Test
1. **Bullet**: 1min+2s (20 games minimum)
   - Fast decision-making
   - Time management critical
   
2. **Blitz**: 5min+4s (20 games minimum)
   - Balanced speed/depth
   - Standard competitive format
   
3. **Rapid**: 15min+10s (20 games minimum)
   - Deeper search possible
   - Fewer time forfeits expected

#### Expected Performance by Time Control
- Bullet: Higher blunders acceptable (6-8/game), time forfeits critical
- Blitz: Balanced performance (5-6 blunders/game)
- Rapid: Lower blunders expected (4-5/game), more tactical accuracy

---

## Pre-Deployment Checklist

Before ANY production deployment:

- [ ] **Regression Suite**: 100% pass (Level 1)
- [ ] **Performance Benchmark**: Acceptance criteria met (Level 2)
- [ ] **Time Control Validation**: No regression in any format (Level 3)
- [ ] **CHANGELOG.md**: Updated with version entry
- [ ] **deployment_log.json**: Updated with test results
- [ ] **Git Tag**: Created for version
- [ ] **Code Review**: Diff reviewed for unintended changes
- [ ] **Version Numbers**: Updated in v7p3r.py and v7p3r_uci.py
- [ ] **Rollback Plan**: Documented and tested
- [ ] **Deployment Window**: Scheduled during low-traffic hours

**If ANY item fails, DO NOT DEPLOY.**

---

## Post-Deployment Monitoring

After deployment to production:

### First 24 Hours (Critical Monitoring)
- [ ] **First 5 games**: Watch manually on Lichess
- [ ] **Docker logs**: Check for errors every 2 hours
- [ ] **Move times**: Verify no unusual delays
- [ ] **Blunders**: Monitor for spike in tactical errors
- [ ] **Time forfeits**: Should be <10% of games

### First 48 Hours (Performance Validation)
- [ ] **Win rate**: Compare to baseline (expect ±5% variance)
- [ ] **Draw rate**: Should be stable (±5% of baseline)
- [ ] **Blunders/game**: Should meet acceptance criteria
- [ ] **CPL**: Should be consistent with testing

### First Week (Stability Confirmation)
- [ ] **ELO**: Should stabilize within ±50 of expected
- [ ] **Crash rate**: Zero crashes acceptable
- [ ] **Game count**: Minimum 50 games for statistical significance
- [ ] **User feedback**: Monitor Lichess profile comments

**If any red flags appear, initiate rollback procedure.**

---

## Known Failure Patterns

### Pattern 1: Endgame Evaluation Regression (v17.4)
**Symptom**: Missed mates, high CPL in endgames  
**Example**: Game 9i883UOF, move 23. Be2?? (missed mate-in-3)  
**Test**: Mate-in-3 detection regression test  
**Prevention**: Never raise endgame threshold without extensive testing

### Pattern 2: Color Imbalance (v17.0)
**Symptom**: 100% win as White, 64% as Black  
**Example**: All 3 tournament losses from PV instant move bug  
**Test**: 50/50 White/Black split in benchmarks  
**Prevention**: Disable PV instant moves, test color balance

### Pattern 3: Time Management Issues (v17.7)
**Symptom**: 30% of losses from time forfeits  
**Example**: Games MTXMr5rL, ZIYO32mt (lost with 4+ minutes remaining)  
**Test**: Time pressure scenarios in regression suite  
**Prevention**: Dynamic time allocation (TimeManager module)

### Pattern 4: Draw Rate Anomalies (v17.7)
**Symptom**: High draw rate in winning positions  
**Example**: Accepting draws at +100cp (1 pawn advantage)  
**Test**: Track draw rate in benchmarks, compare vs baseline  
**Prevention**: Test repetition threshold changes across time controls

---

## Adding New Regression Tests

When a production failure occurs:

1. **Document the failure** in CHANGELOG.md
2. **Create position test** for the exact failure case
3. **Add to regression suite** in testing/regression_suite.py
4. **Verify new version passes** the test
5. **Keep test forever** to prevent regression

Example:
```python
def test_v17_4_mate_in_3_failure():
    """
    v17.4 rolled back for this failure
    Position from game 9i883UOF, move 23
    Engine played Be2?? instead of finding mate-in-3
    """
    # FEN for position before move 23
    board = chess.Board("...")
    engine = V7P3REngine()
    move, score = engine.search(board, max_depth=8)
    
    # Must find mate
    assert score > 9000, f"Failed to detect mate: {score}"
    
    # Must NOT play the blunder
    assert move != chess.Move.from_uci("e1e2"), "Played the v17.4 blunder!"
    
    return True
```

---

## Testing Best Practices

### DO
- ✅ Run full regression suite before EVERY deployment
- ✅ Test across ALL time controls (bullet, blitz, rapid)
- ✅ Compare against stable baseline, not experimental versions
- ✅ Use consistent opening positions for benchmarks
- ✅ Monitor first 24 hours after deployment closely
- ✅ Document all test results in deployment_log.json
- ✅ Create regression test for every production failure

### DON'T
- ❌ Deploy without 100% regression test pass rate
- ❌ Accept <48% win rate in benchmarks
- ❌ Ignore time forfeit spikes in testing
- ❌ Compare against unstable or experimental versions
- ❌ Deploy during peak traffic hours
- ❌ Skip post-deployment monitoring
- ❌ Reuse version numbers even if rolled back

---

## Rollback Decision Tree

```
Production issue detected
    ↓
Is it a critical bug? (crashes, illegal moves, mate-in-1 misses)
    YES → IMMEDIATE ROLLBACK
    NO → Continue monitoring
        ↓
    Is win rate <40% after 20 games?
        YES → ROLLBACK within 24 hours
        NO → Continue monitoring
            ↓
        Is time forfeit rate >20%?
            YES → ROLLBACK within 24 hours
            NO → Continue monitoring
                ↓
            Is blunder rate >8/game?
                YES → ROLLBACK within 48 hours
                NO → Monitor for 1 week, then mark stable
```

---

## Summary

**Testing is not optional.** Every production deployment must:
1. Pass 100% of regression tests
2. Meet performance benchmark acceptance criteria
3. Show no regression across time controls
4. Be monitored for first 24-48 hours
5. Have documented rollback plan

**When in doubt, don't deploy.** It's better to delay deployment for more testing than to rollback a failed version.

For detailed deployment procedures, see: `.github/instructions/version_management.instructions.md`
