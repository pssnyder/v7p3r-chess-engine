# V7P3R Lichess Bot - Deployment Changelog

**Purpose**: Track all engine deployments, rollbacks, and configuration changes for v7p3r_bot on Lichess  
**Maintainer**: pssnyder  
**Last Updated**: 2026-05-03

---

## Quick Reference

**Current Active Version**: v18.6.3 (deployed 2026-05-13 [DEPLOYING])  
**GCP Project**: v7p3r-lichess-bot  
**Instance**: v7p3r-production-bot (e2-micro, us-central1-a)  
**ELO Rating**: [TBD - monitoring] - Target: 1500+ stable, depth 5-6, zero timeouts

---

## Version History

<!-- 
TEMPLATE FOR NEW VERSIONS:
Copy this block, fill in the fields, and paste at the top of this section.

### vX.X
- **Deployed**: YYYY-MM-DD
- **Retired**: YYYY-MM-DD | [ACTIVE] | [UNKNOWN]
- **Status**: active | retired | rolled_back
- **Rollback**: true | false
- **Duration Days**: N | [TBD]
- **Deployment Method**: automated | manual | emergency
- **Environment**: production | staging
- **ELO Rating**: NNNN | [TBD]
- **Games Played**: NNNN | [TBD]
- **Features**:
  - Feature 1
  - Feature 2
- **Known Issues**:
  - Issue 1
  - Issue 2
- **Rollback Reason**: [if rollback=true] Reason text
- **Notes**: Additional context

-->

### v18.6.3
- **Deployed**: 2026-05-13 [DEPLOYING]
- **Retired**: [ACTIVE]
- **Status**: active
- **Rollback**: false
- **Duration Days**: [TBD]
- **Deployment Method**: manual
- **Environment**: production
- **ELO Rating**: [TBD] - Target: 1500+ stable, strong tactical play
- **Games Played**: [TBD]
- **Features**:
  - **ULTRA-PERFORMANCE MODE**: Disabled move_safety checker (93ms overhead per move ordering)
  - **REDUCED QUIESCENCE DEPTH**: 4 → 2 (was searching to depth 8 total!)
  - **FREEZE PREVENTION**: Removed repetition re-search bug, added UCI emergency fallback
  - Now reaches **depth 5-6** consistently (was stuck at depth 3-4)
  - NPS improved to **10800+** (was 6000-8000)
  - All v18.6.2 timeout fixes kept (exception-based, movetime respected)
- **Known Issues**:
  - Move safety disabled - blunder rate unknown (monitoring required)
- **Rollback Reason**: N/A
- **Notes**: **CRITICAL PERFORMANCE BREAKTHROUGH** - Identified move_safety.evaluate_move_safety() as major bottleneck (93ms per move ordering × 30k nodes = 45+ seconds wasted). Disabled safety checker for raw speed test. Also reduced quiescence depth 4→2 (depth 4 main + depth 4 q-search = depth 8 total was excessive). Fixed FREEZE BUG: repetition avoidance re-ran search with 10 minutes on clock and no timeout protection. Added UCI emergency fallback to always return valid move. Performance: Depth 5 in 8.6s (was depth 4 in 4s), depth 6 in endgames. NPS: 10802 (was 6164). Testing speed vs safety tradeoff - may need selective safety re-enabling if blunders spike. ROOT CAUSE: Safety checker called board.push/pop + scanned 64 squares + generated legal moves on EVERY non-TT move at depth >= 3.

### v18.6.2
- **Deployed**: 2026-05-12 [PENDING TEST]
- **Retired**: [ACTIVE]
- **Status**: active
- **Rollback**: false
- **Duration Days**: [TBD]
- **Deployment Method**: manual
- **Environment**: production
- **ELO Rating**: [TBD - monitoring] - Target: 1500+ stable, zero time forfeits
- **Games Played**: [TBD]
- **Features**:
  - **FINAL TIME FIX**: Exception-based timeout enforcement
  - Implemented SearchTimeoutException that aborts entire search tree
  - v18.6.1 fix (movetime hard limit) kept
  - v18.6.0 fix (bitboard overhead removal) kept
  - v18.5 cleanup (dead code removal) kept
- **Known Issues**:
  - None identified in testing (validated with 200ms, 500ms, 1s, 3s limits)
- **Rollback Reason**: N/A
- **Notes**: **COMPREHENSIVE TIME FIX** - Solves 3-layer time management bug cascade: (1) v18.6.0 removed bitboard tactical detection overhead (30k-40k ops/sec), (2) v18.6.1 fixed movetime parameter being overridden by adaptive allocation, (3) v18.6.2 fixes TIME ABORT returning from individual branches instead of aborting entire search. Previous versions showed 23 "TIME ABORT" messages but search continued, completing depth 4 in 4059ms instead of stopping at 2250ms. Exception-based timeout properly exits search tree when time exceeded. Tested: 200ms limit fired exception at 180ms (depth 1), 500ms stopped at 142ms (depth 2), 1s stopped at 262ms (depth 2), 3s stopped at 966ms (depth 3). No TIME ABORT spam, clean output. Deploy pending user validation. Root cause analysis: v18.6.1 TIME ABORT used `return` which only exited current function call, parent iterative deepening loop continued evaluating other moves.

### v18.5


- **Deployed**: 2026-05-11 17:10 UTC
- **Retired**: [ACTIVE]
- **Status**: active
- **Rollback**: false
- **Duration Days**: [TBD]
- **Deployment Method**: manual
- **Environment**: production
- **ELO Rating**: [TBD - monitoring first 5 games] - Target: 1500+ stable
- **Games Played**: [TBD]
- **Features**:
  - Stripped v18.3 codebase (removed unused modular evaluation system)
  - Fixed threefold threshold (50cp, was dynamic but unused)
  - Removed dead code overhead (4 unused module files deleted)
  - Identical time management to v18.0/v18.3 (proven stable)
  - Kept core: Fast evaluator, move safety, opening book
- **Known Issues**:
  - None identified yet (active monitoring)
- **Rollback Reason**: N/A
- **Notes**: **TIME FORFEIT FIX** - Created to address 37.5% time forfeit rate observed in v18.3 redeployment (May 3-10, 2026). Root cause analysis identified modular evaluation system was never enabled but still computing context calculations on every search. v18.5 strips all dead code while preserving v18.3's core evaluation "brain". Files reduced from 10 to 6 (same as v18.0). Successfully deployed May 11, 2026. Monitoring for time management improvements and ELO recovery.

### v18.4
- **Deployed**: 2026-04-17
- **Retired**: 2026-05-03
- **Status**: rolled_back
- **Rollback**: true
- **Duration Days**: 16
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1633 (start) → 1614 (end, -19 points)
- **Games Played**: [TBD]
- **Features**:
  - Aspiration windows
  - Latest stable release attempt
- **Known Issues**:
  - ELO decline from 1633 to ~1614 over 16 days
  - Performance regression vs v18.3 (28 ELO gap)
  - Underperforming in blitz/bullet formats
- **Rollback Reason**: Continued ELO decline and performance regression. v18.4 dropped from 1633 to ~1614 ELO (28-47 points below v18.3's stable 1661). Reverting to proven v18.3 all-star engine to recover rating and expand to blitz/bullet formats.
- **Notes**: Aspiration windows may have introduced search inefficiency. Need to investigate why v18.4 underperformed vs v18.3 despite theoretical improvements.

### v18.3
- **Deployed**: 2025-12-29 (initial), 2026-05-03 (redeployed as rollback)
- **Retired**: 2026-04-17 (initial retirement), 2026-05-10 (second retirement - replaced by v18.5)
- **Status**: retired
- **Rollback**: false
- **Duration Days**: 110 (first deployment), 7 (second deployment - replaced due to time forfeits)
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1661 (stable first run), 1318-1539 (second run with time forfeits)
- **Games Played**: [TBD]
- **Features**:
  - Adaptive time allocation (game phase awareness)
  - Smart early exit (stable move detection)
  - Emergency time mode
  - PST optimization (28% speedup)
  - Position complexity adjustments
- **Deployment Notes**:
  - **2026-05-03**: Successfully redeployed as rollback from v18.4. Engine files copied directly into Docker container filesystem (not mounted) to resolve execute permission issues. Bot online and awaiting challenges.
  - **2025-12-29**: Initial deployment with successful 110-day run reaching 1722 peak ELO.
- **Known Issues**: 
  - **CRITICAL (May 3-10 deployment)**: 37.5% time forfeit rate (3/8 games). Unused modular evaluation system computing context calculations on every root search despite being disabled. ELO dropped to 1318-1539 range.
  - Smart matchmaking system caused rating slide Jan 21-Feb 15 (1722→1661) in first deployment. Mitigated by reverting to random matchmaking.
- **Rollback Reason**: N/A (was rollback TARGET from v18.4, now superseded by v18.5)
- **Notes**: **MIXED RESULTS** - First deployment (Dec 29 - Apr 17): Longest stable run (110 days), highest peak ELO (1722). Second deployment (May 3-10): Critical time management failure due to dead code overhead. Analysis revealed modular evaluation system was never enabled but still executing expensive context calculations. v18.5 created to strip dead code while preserving core evaluation.

### v18.0
- **Deployed**: 2025-12-20
- **Retired**: 2025-12-29
- **Status**: retired
- **Rollback**: false
- **Duration Days**: 9
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1654
- **Games Played**: [TBD]
- **Features**: []
- **Known Issues**: []
- **Rollback Reason**: N/A
- **Notes**: First v18 series deployment

### v17.7
- **Deployed**: 2025-12-06
- **Retired**: 2025-12-20
- **Status**: retired
- **Rollback**: false
- **Duration Days**: 14
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1623
- **Games Played**: [TBD]
- **Features**: []
- **Known Issues**: []
- **Rollback Reason**: N/A
- **Notes**: Replaced buggy v17.6

### v17.5
- **Deployed**: 2025-12-02
- **Retired**: 2025-12-06
- **Status**: retired
- **Rollback**: false
- **Duration Days**: 4
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1604
- **Games Played**: [TBD]
- **Features**: []
- **Known Issues**: []
- **Rollback Reason**: N/A
- **Notes**: Deployed via universal deployment script

### v17.4
- **Deployed**: 2025-11-26
- **Retired**: 2025-11-30
- **Status**: rolled_back
- **Rollback**: true
- **Duration Days**: 4
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1606
- **Games Played**: [TBD]
- **Features**:
  - Endgame improvements (rook bonus)
  - Improved endgame detection
  - Enhanced endgame evaluation
- **Known Issues**:
  - Critical endgame blunders
  - Mate-in-3 misses
  - High centipawn loss (~2000+)
- **Rollback Reason**: Critical blunder detected - move 23. Be2?? in game 9i883UOF (missed mate in 3). Lost to human player. High centipawn loss in endgames. Endgame evaluation regression.
- **Notes**: Rolled back to v17.1 for stability

### v17.2.0
- **Deployed**: 2025-11-21
- **Retired**: 2025-11-26
- **Status**: retired
- **Rollback**: false
- **Duration Days**: 5
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1614
- **Games Played**: [TBD]
- **Features**:
  - Positional evaluation improvements
  - Opening book enhancements
- **Known Issues**: []
- **Rollback Reason**: N/A
- **Notes**: Stable version, replaced by v17.4

### v17.1.1
- **Deployed**: 2025-11-21
- **Retired**: 2025-11-21
- **Status**: retired
- **Rollback**: false
- **Duration Days**: 0
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1487
- **Games Played**: [TBD]
- **Features**: []
- **Known Issues**: []
- **Rollback Reason**: N/A
- **Notes**: Minor patch, quickly superseded by v17.2.0 (same-day deployment)

### v17.1
- **Deployed**: 2025-11-21
- **Retired**: 2025-11-21
- **Status**: retired
- **Rollback**: false
- **Duration Days**: 0
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: n/a (unrated)
- **Games Played**: [TBD]
- **Features**:
  - Stable endgame play
  - Reliable evaluation
- **Known Issues**: []
- **Rollback Reason**: N/A
- **Notes**: Initial deployment superseded same day. Re-deployed 2025-11-30 as rollback target from v17.4. Proven stable version. CPL: Moderate (~1500-1800 average).

### v17.0
- **Deployed**: 2025-11-20
- **Retired**: 2025-11-21
- **Status**: retired
- **Rollback**: false
- **Duration Days**: 1
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1496
- **Games Played**: [TBD]
- **Features**: []
- **Known Issues**: []
- **Rollback Reason**: N/A
- **Notes**: Initial v17 series deployment

### v16.1
- **Deployed**: 2025-11-19
- **Retired**: 2025-11-20
- **Status**: retired
- **Rollback**: false
- **Duration Days**: 1
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1503
- **Games Played**: [TBD]
- **Features**: []
- **Known Issues**: []
- **Rollback Reason**: N/A
- **Notes**: Last of v16 series

### v14.1
- **Deployed**: 2025-10-25
- **Retired**: 2025-11-19
- **Status**: retired
- **Rollback**: false
- **Duration Days**: 25
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1488
- **Games Played**: [TBD]
- **Features**:
  - Major evaluation overhaul
  - Improved positional understanding
- **Known Issues**: []
- **Rollback Reason**: N/A
- **Notes**: Long-running stable version (~25 days)

### v14.0
- **Deployed**: 2025-10-25
- **Retired**: 2025-10-25
- **Status**: retired
- **Rollback**: false
- **Duration Days**: 0
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: n/a (unrated)
- **Games Played**: [TBD]
- **Features**: []
- **Known Issues**: []
- **Rollback Reason**: N/A
- **Notes**: Quickly patched to v14.1 (same-day deployment)

### v12.6
- **Deployed**: 2025-10-04
- **Retired**: 2025-10-25
- **Status**: retired
- **Rollback**: false
- **Duration Days**: 21
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1544
- **Games Played**: [TBD]
- **Features**: []
- **Known Issues**: []
- **Rollback Reason**: N/A
- **Notes**: Last of v12 series, stable for ~21 days

### v12.4
- **Deployed**: 2025-10-03
- **Retired**: 2025-10-04
- **Status**: retired
- **Rollback**: false
- **Duration Days**: 1
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1509
- **Games Played**: [TBD]
- **Features**: []
- **Known Issues**: []
- **Rollback Reason**: N/A
- **Notes**: Quick iteration in v12 series

### v12.2
- **Deployed**: 2025-10-03
- **Retired**: 2025-10-03
- **Status**: retired
- **Rollback**: false
- **Duration Days**: 0
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1497
- **Games Played**: [TBD]
- **Features**: []
- **Known Issues**: []
- **Rollback Reason**: N/A
- **Notes**: Initial v12 series deployment (same-day replacement)

---

## Deployment Procedures

### Standard Deployment Checklist
- [ ] Build engine executable
- [ ] Create dated directory: `V7P3R_vX.X_YYYYMMDD/`
- [ ] Test locally with lichess-bot
- [ ] Deploy to GCP container
- [ ] Monitor first 10 games for issues
- [ ] Update this CHANGELOG.md (add new version at top)
- [ ] Update **Last Updated** date in header
- [ ] Update **Current Active Version** in Quick Reference
- [ ] If issues found, execute rollback procedure
- [ ] Run analytics after 24 hours to validate performance

### Rollback Procedure
1. Stop lichess-bot container: `sudo docker stop v7p3r-production`
2. Update `config.yml` engine path to previous stable version
3. Restart container: `sudo docker restart v7p3r-production`
4. Verify version in logs: `sudo docker logs v7p3r-production --tail 50`
5. Update CHANGELOG.md:
   - Set rolled-back version **Retired** date to today
   - Set **Status** to `rolled_back`
   - Set **Rollback** to `true`
   - Fill in **Rollback Reason** with detailed explanation
   - Update **Current Active Version** to rollback target
6. Run analytics to confirm issue and regression

---

## Analytics Quick Reference

### Filter Games by Version

Use deployed/retired dates to filter game analysis:

```sql
-- Games for specific version
SELECT * FROM games
WHERE game_date BETWEEN '2025-11-26' AND '2025-11-30'  -- v17.4 period
```

### Common Analysis Queries

```bash
# Current version performance (v18.4)
python full_analysis.py --since 2026-04-17

# Rolled back version analysis (v17.4)
python full_analysis.py --since 2025-11-26 --until 2025-11-30

# Long-term stable version (v14.1)
python full_analysis.py --since 2025-10-25 --until 2025-11-19
```

---

## Notes

- All dates are in YYYY-MM-DD format for easy parsing
- Duration Days calculated as: (Retired - Deployed).days
- [ACTIVE] means version is currently deployed
- [TBD] means data not yet available (fill in after analytics run)
- [UNKNOWN] means data lost or not recorded
- Same-day deployments have Duration Days = 0
- Rollback Reason field only populated when Rollback = true
