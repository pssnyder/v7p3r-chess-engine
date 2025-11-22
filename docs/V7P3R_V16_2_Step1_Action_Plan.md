# V16.2 Step 1: Deployment & Validation Action Plan

**Date:** November 20, 2025  
**Status:** 🔴 V16.2 NOT YET DEPLOYED  
**Current Deployed Version:** V12.6.exe

## Critical Finding

Analysis of Lichess game records shows:
- **Config file points to: `V7P3R_v12.6.exe`** (not v14.1 or v16.2)
- **Most recent games: October 21, 2025** (over a month old)
- **Games analyzed: 50 games from v12.6 era**
  - Score: 17% (6W-39L-5D)
  - ELO: 1349 (lowest recorded)
  - This was **v12.6 data**, not current

**YOU SAID:** "we currently have v16.2 deployed on lichess and just replaced v14.1 which had an elo around 1496"

**REALITY:** V16.2 has not been deployed yet. The Lichess bot is either:
1. Not running (last games from October 21)
2. Still running v12.6.exe from config

## Immediate Actions Required

### Action 1: Verify Current Deployment Status

```bash
# Check if lichess bot is running
cd "s:\Programming\Chess Engines\Deployed Engines\v7p3r-lichess-engine"
python lichess-bot.py --version  # or check process list

# Check what engine version is actually configured
cat config.yml | grep "name:"
```

**Expected output:** Should show what's currently configured to run

### Action 2: Deploy V16.2 to Lichess

#### Option A: Update Existing Configuration (Recommended)

```yaml
# Edit: s:\Programming\Chess Engines\Deployed Engines\v7p3r-lichess-engine\config.yml
engine:
  dir: "./engines/V7P3R_v16.2/"
  name: "src"  # For Python source
  # OR
  name: "V7P3R_v16.2"  # If there's an exe build
  protocol: "homemade"  # v7p3r uses homemade protocol, not UCI
```

#### Option B: Fresh Deployment Script

The repository has deployment scripts:
- `deploy-v16.2-quick.sh`
- `deploy_v16.2.sh`
- `DEPLOY_V16_2_MANUAL_CHECKLIST.md`

Check these files to follow proper deployment procedure.

### Action 3: Ensure v16.2 Source is Complete

```bash
# Verify v16.2 engine files exist
cd "s:\Programming\Chess Engines\Deployed Engines\v7p3r-lichess-engine\engines\V7P3R_v16.2"
ls -la src/
```

**Required files in src/:**
- `v7p3r_engine.py` (main engine)
- `v7p3r_evaluator.py` (evaluation function with 60/40 PST+Material blend)
- `v7p3r_search.py` (search with NULL MOVE PRUNING FIX)
- `v7p3r_move_generator.py`
- `v7p3r_time_manager.py`
- All other supporting modules

### Action 4: Configure PGN Comment Output

To track depth in games, ensure v16.2 writes depth to PGN comments:

**Check in `v7p3r_engine.py` or UCI/homemade interface:**
```python
# Should include depth in info output
print(f"info depth {depth} score cp {score} pv {pv_string}")
```

The lichess-bot framework should automatically capture this and write to PGN.

### Action 5: Start Bot and Play Test Games

```bash
# Start lichess bot with v16.2
cd "s:\Programming\Chess Engines\Deployed Engines\v7p3r-lichess-engine"
python lichess-bot.py

# Monitor startup logs for:
# - Engine version loaded
# - Successful UCI/homemade protocol handshake
# - Bot goes online on lichess.org
```

**Play 5-10 test games manually first:**
1. Challenge the bot from another account
2. Observe behavior:
   - Does it respond?
   - What time per move?
   - Does it make sensible moves?
   - Check depth in game analysis

### Action 6: Run Monitoring After 20+ Games

Once v16.2 is deployed and playing:

```bash
# After 20+ games have been played
cd "s:\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine"
python scripts/monitor_v16_2_lichess_deployment.py
```

This will analyze:
- Search depth consistency (target: 5-8 consistently, not 2-3)
- Win/loss record
- ELO progression vs v14.1 baseline (1496)
- Material safety
- Tactical quality

## Step 1 Success Criteria

Before proceeding to Step 2, we need:

✅ **Deployment Confirmation:**
- [ ] v16.2 successfully deployed and bot is online
- [ ] Config points to v16.2 source/exe
- [ ] Bot accepts and plays challenges

✅ **Game Data Collection:**
- [ ] 20+ completed games with v16.2
- [ ] PGN files include depth information in comments
- [ ] Games timestamped after deployment date

✅ **Depth Validation:**
- [ ] Average depth ≥ 5 (confirms bug fix)
- [ ] <10% of moves at depth 2-3 (old bug range)
- [ ] Depth consistency (low std deviation preferred)

✅ **Performance Baseline:**
- [ ] Score rate measured (target: >50%)
- [ ] ELO trend established vs v14.1's 1496
- [ ] If score <45%, immediately rollback

## Current Status Summary

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Deployed Version | v16.2 | v12.6.exe | 🔴 NOT DEPLOYED |
| Bot Online | Yes | Unknown (likely offline) | 🔴 CHECK STATUS |
| Recent Games | 20+ new | 0 (last: Oct 21) | 🔴 NO DATA |
| Depth Data | Available | Not available | 🔴 N/A - not deployed |

## Next Steps

**FIRST:** You need to:
1. Confirm v16.2 deployment status
2. If not deployed, deploy it now following proper procedure
3. Restart lichess bot
4. Let it play 20+ games
5. Re-run monitoring script

**OR** if you prefer to wait:
- We can proceed directly to Step 2 (controlled local testing)
- Test v16.2 vs v14.4/PositionalOpponent/MaterialOpponent locally
- Validate depth fix and performance before deploying

## Questions to Answer

1. **When did you intend to deploy v16.2?** Today? Already done (but config not updated)?
2. **Is the lichess bot currently running?** Check if process is active
3. **Do you want to deploy now, or test locally first?** (Recommend local testing given uncertainty)
4. **Is there a v16.2 exe build, or running from Python source?**

---

**RECOMMENDATION:** Let's proceed to Step 2 (local controlled testing) to validate v16.2's improvements before deploying to Lichess. This avoids another potential public regression like the October games showed.
