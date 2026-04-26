# V19.0 Tournament Testing - In Progress

**Date**: April 22, 2026  
**Status**: 🏁 RUNNING - 30 games @ 5min+4s blitz  
**Estimated Duration**: 45-60 minutes  
**Terminal**: Running in background (async mode)

---

## What's Happening

A programmatic tournament is currently running between:
- **Engine 1 (v19.0)**: Spring cleaning version with modular eval removed + C0BR4-style time management
- **Engine 2 (v18.4)**: Current production version (experiencing performance downturn)

### Tournament Configuration
- **Games**: 30 total (15 as White, 15 as Black for each engine)
- **Time Control**: 5 minutes + 4 seconds increment (blitz)
- **Validation**: Both engines passed UCI validation before tournament start
- **Why This Time Control**: v18.4 had 75% timeout rate at this speed - critical test for v19.0's time management

---

## Success Criteria

For v19.0 to be deployment-ready:

### ✅ Minimum Requirements (MUST PASS)
- **0 timeouts** for v19.0 (fixes 75% bug from v18.4)
- **No crashes or UCI errors** (stability)
- **Score ≥45%** vs v18.4 (maintains or improves strength)

### 🎯 Good Performance
- **Score 48-52%** (similar strength, no regression)
- **0-1 timeouts** for v18.4 (confirms their timeout issues)
- **Similar move quality** (no evaluation regression)

### ⭐ Excellent Performance
- **Score >55%** (actual improvement from spring cleaning)
- **Faster average move times** (30-50% speedup expected)
- **Better tactical play** (no modular overhead)

---

## Checking Progress

You can check tournament progress at any time:

```powershell
# View terminal output (shows current game status)
# The tournament is running in terminal ID: d6d0bb4c-0eba-4e5b-8665-0760f693b600
```

Or wait for the tournament to complete (45-60 minutes from start).

---

## What to Expect

### During Tournament
- Each game takes ~2-4 minutes to complete
- You'll see game numbers counting up (1/30, 2/30, etc.)
- Running totals update after each game
- Any timeouts or crashes will be immediately visible

### After Tournament
The script will display:
1. **Final Results**
   - Win/Loss/Draw record for each engine
   - Score percentages
   - Timeout and crash counts
   
2. **Deployment Recommendation**
   - ✓ READY for deployment (if criteria met)
   - ✗ NOT READY (if issues found)
   
3. **Saved Results**
   - JSON file in `tournament_results/` directory
   - Full game records with move lists
   - Detailed statistics

---

## What the Results Mean

### If v19.0 Wins (Score >50%)
✅ **Spring cleaning improved strength** - modular eval removal helped  
✅ **Time management working** - no timeouts means fix is solid  
→ **DEPLOY to lichess** as v7p3r_bot

### If v19.0 Ties (Score 45-50%)
✅ **No regression** - code cleanup didn't hurt strength  
✅ **Time management working** - critical fix validated  
→ **DEPLOY to lichess** - at minimum fixes timeout issues

### If v19.0 Loses (Score <45%)
⚠️ **Investigate regression** - something went wrong  
- Review evaluation changes
- Check search depth consistency
- Test in isolation (single games)
→ **DO NOT DEPLOY** - fix issues first

### If v19.0 Has Timeouts
❌ **TimeManager needs tuning** - allocations too aggressive  
- Review phase multipliers
- Increase safety margins
- Test specific scenarios
→ **DO NOT DEPLOY** - critical bug remains

---

## After Results

### If READY for Deployment
1. Review full tournament results in `tournament_results/` JSON
2. Tag v19.0.0 release candidate: `git tag -a v19.0.0 -m "v19.0.0: Spring cleaning + time fix"`
3. Follow deployment workflow in [docs/V19_VALIDATION_TOURNAMENT_SETUP.md](V19_VALIDATION_TOURNAMENT_SETUP.md)
4. Update CHANGELOG.md with tournament stats
5. Deploy to lichess GCP VM as v7p3r_bot

### If NOT READY
1. Review failure patterns in tournament results
2. Run targeted tests (single games, specific positions)
3. Iterate on fixes (TimeManager tuning, evaluation checks)
4. Re-run tournament after fixes

---

## Current Status

🏁 **Tournament is RUNNING in background**

The tournament will complete automatically and display final results. You can:
- Let it run to completion (~45-60 min)
- Check progress periodically via terminal
- Review results when "TOURNAMENT RESULTS" banner appears

---

## Tournament Infrastructure

### Created Files (Now Committed)
1. **[testing/tournament_runner.py](../testing/tournament_runner.py)** (464 lines)
   - Full UCI engine tournament system
   - Handles time controls, move tracking, result analysis
   - Generates JSON reports

2. **[testing/test_v19_vs_v18_4.py](../testing/test_v19_vs_v18_4.py)** (105 lines)
   - Convenience launcher for v19.0 vs v18.4
   - Finds engines automatically
   - Provides deployment recommendations

3. **[testing/validate_engines.py](../testing/validate_engines.py)** (99 lines)
   - Pre-tournament UCI validation
   - Tests basic communication before full tournament
   - Catches setup issues early

### Why Programmatic Testing?
- **Faster setup** - no Arena GUI configuration needed
- **Automated** - runs without user interaction
- **Reproducible** - consistent time controls and settings
- **Detailed results** - JSON output for analysis
- **CI/CD ready** - can be integrated into automated testing

---

## Expected Timeline

- **Tournament Start**: Now (just launched)
- **Game 1 Complete**: ~2-4 minutes
- **Halfway (15 games)**: ~25-35 minutes
- **Tournament Complete**: ~45-60 minutes
- **Results Analysis**: Instant (automatic)

---

## Next Actions

**NOW**: Let tournament run to completion (45-60 minutes)

**THEN**: Review results and follow deployment recommendation

**IF READY**: Deploy v19.0 to lichess as production v7p3r_bot

---

Good luck! The tournament is designed to validate that v19.0 is:
1. **Stable** (no crashes or timeouts)
2. **Fast** (improved time management)
3. **Strong** (maintains or improves on v18.4)

If all three criteria pass, v19.0 is ready to replace v18.4 as the production bot on lichess.
