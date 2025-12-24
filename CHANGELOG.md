# V7P3R Chess Engine - Version History

All notable changes to the V7P3R chess engine are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Version Numbering Convention

Starting with v18.0.0, we use semantic versioning:
- **MAJOR.MINOR.PATCH** (e.g., 18.2.1)
  - **MAJOR**: Breaking changes or significant evaluation rewrites
  - **MINOR**: New features, evaluation improvements, algorithm changes
  - **PATCH**: Bug fixes, performance tweaks, parameter tuning

Legacy v17.x series used incremental numbering without semantic meaning.

---

## [Unreleased]

### Planned
- TimeManager integration for dynamic time allocation
- Mate-in-1 fast path detection
- Enhanced king safety with back-rank mate detection

---

## [18.2.0] - 2025-12-22 [TACTICAL + POSITIONAL COMBINED] [TESTING]

### Summary
Merges v18.0's tactical safety system with v18.1's evaluation tuning for comprehensive improvement.

### Added (from v18.0)
- **MoveSafetyChecker**: Lightweight defensive tactical awareness system
- Anti-tactical penalty system in move ordering (-50 to -800cp for unsafe moves)
- Hanging piece detection (checks if pieces are undefended after move)
- Immediate capture threat evaluation (prevents leaving high-value pieces exposed)

### Changed (from v18.1)
- **King Safety**: Added -100cp penalty for high-value attackers (Q/R) in king zone
- **King Safety**: Added -80cp penalty for unmoved king on center files (d/e) in middlegame
- **Bishop Pair**: Increased bonus from implicit handling to explicit +50cp
- **Passed Pawns**: Changed to exponential scaling (20 × 2^advancement) instead of linear
- **King Centralization**: Enhanced endgame bonus to +40-70cp based on central squares
- **Move ordering**: Integrated safety scores into all move categories (captures, checks, killers, quiet moves)
- **Threefold repetition**: Final bestmove validation avoids repetition when winning (>100cp)

### Rationale
- **v18.1 Tournament Results**: 64% vs v17.1 (+100 ELO) - evaluation tuning works
- **v18.0 Tournament Results**: 58% vs v17.1 (+56 ELO) - tactical safety works
- **v18.1 vs v18.0**: 48% vs 52% (-14 ELO) - evaluation alone cannot beat tactics
- **Hypothesis**: Combining both systems should achieve 65-70%+ vs v17.1
- **Tactical safety prevents blunders** (v18.0's strength), **evaluation wins positions** (v18.1's strength)

### Testing
- ✅ Unit tests (from v18.1): 5/5 passed
  - King safety center penalty: 115cp difference
  - Passed pawn exponential: 280cp difference
  - Bishop pair bonus: 622cp applied
  - King centralization: 100cp difference
  - High-value attacker penalty: 1800cp difference
- ⏳ Performance benchmark: NOT YET RUN (requires 50+ games vs v18.0 and v17.1)

### Performance
- **Expected**: 65-70% vs v17.1 (combining +100 ELO from evaluation + +56 ELO from tactics)
- **Expected**: 55-60% vs v18.0 (adding evaluation tuning to tactical base)
- Bitboard evaluator path only (fast evaluator unchanged)

---

## [18.1.0] - 2025-12-21 [EVALUATION TUNING] [TESTED]

### Changed
- **King Safety**: Added -100cp penalty for high-value attackers (Q/R) in king zone
- **King Safety**: Added -80cp penalty for unmoved king on center files (d/e) in middlegame
- **Bishop Pair**: Increased bonus from implicit handling to explicit +50cp
- **Passed Pawns**: Changed to exponential scaling (20 × 2^advancement) instead of linear
- **King Centralization**: Enhanced endgame bonus to +40-70cp based on central squares

### Rationale
- **Game Analysis Foundation**: Analyzed 10 recent Lichess games (2W-2D-6L, Dec 14-21) using Stockfish 17.1 at depth 20
- **41 Total Mistakes**: 21 inaccuracies, 13 mistakes, 7 blunders across 10 games
- **Theme Breakdown**: 18 unknown (44%), 14 material imbalance (34%), 11 king safety (27%), 9 endgame (22%), 8 pawn play (20%)
- **Pattern Identification**: King safety errors in 6/10 games, passed pawn mishandling in 4/10 games, bishop pair undervaluation in 3/10 games
- **Specific Examples**:
  - Game z2x7qO6f: Failed to prioritize passed pawn advancement (linear bonus insufficient)
  - Game U9E2iNmr: Allowed rook penetration near king (no high-value attacker penalty)
  - Game CHF4FHSy: Gave up bishop pair advantage (no explicit bonus)
  - Game vl2mCQ5b: Kept king in center too long (no center file penalty)

### Testing
- ✅ Unit tests: 5/5 passed
  - King safety center penalty: 115cp difference (e1 unmoved vs g1 castled)
  - Passed pawn exponential: 280cp difference (a3 vs a6)
  - Bishop pair bonus: 622cp applied
  - King centralization: 100cp difference (e4 center vs h1 corner)
  - High-value attacker penalty: 1800cp difference (queen near king vs not)

### Performance
- **NOT YET BENCHMARKED**: Requires 50-game tournament vs v18.0.0
- Bitboard evaluator path only (fast evaluator unchanged)

---

## [18.0.0] - 2025-12-20 [ANTI-TACTICAL DEFENSE] [PRODUCTION]

### Added
- **MoveSafetyChecker**: Lightweight defensive tactical awareness system
- Anti-tactical penalty system in move ordering (-50 to -800cp for unsafe moves)
- Hanging piece detection (checks if pieces are undefended after move)
- Immediate capture threat evaluation (prevents leaving high-value pieces exposed)

### Changed
- **Move ordering**: Integrated safety scores into all move categories (captures, checks, killers, quiet moves)
- **Killer move sorting**: Now sorted by safety score (prevents promoting unsafe killers)
- **Tactical move evaluation**: Combined bitboard tactical bonus with safety penalty for accurate scoring

### Rationale
- Manual PGN analysis of Dec 17 games revealed middlegame tactical errors as primary weakness
- Game 4ZzIc3g6: 18...Rf6 (loses knight on d5 to 19.Bxd5) - safety checker penalty: -520cp
- Game 4ZzIc3g6: 29...Ke7 (allows 30.Ba3+ winning bishop) - safety checker penalty: -264cp
- Lichess yearend metrics show middlegame has highest error density (12/20 total errors)
- Bishop weakness hypothesis REJECTED (only 1 error in 4 games) - issue was tactical calculation, not piece handling

### Performance
- Safety checker: ~9,888 checks/second (101 microseconds per check)
- Search overhead: ~0.1s at depth 4, ~0.3s at depth 5 (negligible)
- Only activates at depth >= 2 (shallow searches skip for speed)

### Testing
- ✅ Test 1: 18...Rf6 correctly flagged as unsafe (-520cp < -100cp threshold)
- ✅ Test 2: 29...Ke7 correctly flagged as unsafe (-264cp < -50cp threshold)
- ✅ Performance: 9888 checks/sec >> 500 checks/sec requirement

### Expected Impact
- Target: +50-80 ELO via blunder prevention
- Reduces middlegame material losses (current primary weakness)
- Preserves existing strengths: checkmate rate (66.9%), opening play (83.1%), rating climb (1200→1575 ELO)

### Deployment
- Status: PRODUCTION (deployed 2025-12-20 to GCP v7p3r-production-bot)
- Tournament validation: 30 games vs v17.1, 58% win rate (17.5/30 points)
- Results: 11W-6L-13D, 67% as White, 50% as Black
- Acceptance criteria: ✅ 58% > 48% threshold, 0% time forfeits
- Known limitation: High draw rate (43%, all threefold repetitions) - flagged for future tuning

---

## [17.1.1] - 2025-12-10 [PRODUCTION ROLLBACK]

### Changed
- **ROLLBACK**: Restored v17.1.1 to production after v17.4-v17.8 failures

### Rationale
- Complete tournament analysis (298 games, 5 tournaments) revealed v17.1/v17.3 as best performers
- v17.1: 75.8% average win rate (70-81% across all time controls) - MOST CONSISTENT
- v17.3: 77.0% average (86.8% peak in 5min+2s) - BEST PEAK
- v17.4+: Progressive regression (32-63% win rates)
- v17.6-v17.8: Complete failures (0-52% win rates, illegal moves)
- **Root cause identified**: v17.2 introduced broken two-tier bucket TT system using integer indices instead of zobrist hashes, corrupting move selection

### Deployment
- Date: 2025-12-10
- Platform: GCP VM (v7p3r-production-bot)
- Engine verified: `id name V7P3R v17.1.1`
- Status: Live on Lichess

### Testing
- Tournament data: v17.1.1 proven stable across bullet (1min+2s), blitz (5min+4s), rapid (30min+5s)
- No regression tests run (rolling back to known stable version)

### Notes
- v17.8-clean abandoned (catastrophic tournament failure, 0/6 score)
- v17.3 analysis revealed it's functionally identical to v17.1.1 (only UCI version string differs)
- Skipping remaining v17 line development
- Next version will be v18.0.0 (major overhaul from v17.1.1 baseline)

---

## [17.8-clean] - 2025-12-10 [ABANDONED]

### Philosophy
Based on tournament analysis showing v17.1-v17.3 as top performers (42.5 pts), v17.8-clean
removes failed v17.4-v17.7 experimental features and builds on proven stable foundation.

### Changed
- **BASELINE**: v17.1 codebase (proven stable, 800cp endgame threshold)
- **REPETITION THRESHOLD**: 50cp (simplified - uses board.is_repetition(), no position history)
- **PAWN STRUCTURE**: Kept v17.6 improvements (bishop pair +50cp, isolated pawns -15cp, connected pawns +5cp, knight outposts +20cp)

### Fixed
- **CRITICAL**: Reverted 1300cp endgame threshold → 800cp (v17.4 bug that caused mate-in-3 miss)

### Removed
- v17.7 complexity: Tablebase patterns (dead code), king-edge driving, 50-move awareness, mate verification extensions
- v17.7 position history tracking (simplified to use python-chess built-in)
- v17.5 mate threat detection (redundant with quiescence search)
- 437 lines of code bloat (-30% from v17.8)

### Rationale
- Tournament data: v17.1 (42.5 pts, 56.9% win), v17.2 (39.5 pts, 50%), v17.3 (42.5 pts, 58.9%)
- v17.4-v17.7 scored 21.5-35 pts (regression from v17.1-v17.3)
- v17.4's 1300cp threshold caused endgame blunders (game 9i883UOF mate-in-3 miss)
- v17.6 pawn structure fixes address real issue (159 isolated pawns/game)
- v17.8's 50cp repetition threshold keeps aggressive draw avoidance without complexity

### Testing
- Regression tournament pending (v17.8-clean vs v17.1 baseline)
- Expected: Match or exceed v17.1 performance with better draw handling
- Acceptance: Win rate ≥48%, blunders ≤6.0/game, no time forfeits

---

## [17.8.0] - 2025-12-10 [SUPERSEDED BY 17.8-clean]

### Changed
- **CRITICAL FIX**: Lowered repetition avoidance threshold from 200cp to 50cp
- Engine now pushes advantages more aggressively in rapid games
- Updated version identifier in UCI to "V7P3R v17.8"

### Rationale
- v17.7 was accepting draws in positions with +100cp (1 pawn advantage)
- 200cp threshold too conservative for competitive play
- New 50cp threshold (0.5 pawns) aligns with "avoid draws unless losing" philosophy

### Testing
- Regression tournament vs v17.1-v17.7 pending
- Local validation required before production deployment

### Known Issues
- Rapid game performance regression needs monitoring
- May need adaptive threshold by time control (50cp rapid, 100cp blitz/bullet)

---

## [17.7.0] - 2025-12-06

### Added
- Threefold repetition avoidance when winning (>200cp)
- Mate verification with +2 ply depth extensions
- King-edge driving bonus (+10cp × distance when material >600cp)
- Basic tablebase patterns (K+R/Q/R+B vs K)
- 50-move rule awareness (prioritize resets when halfmove_clock >80)

### Deployment
- **Status**: Stable production deployment
- **Duration**: 4+ days (Dec 6-10)
- **Platform**: GCP v7p3r-production-bot

### Goal
- Never draw from winning positions
- Convert R+B vs K endgames reliably

---

## [17.6.0] - 2025-12-06

### Added
- Bishop pair bonus (+30cp)
- **CRITICAL**: Isolated pawn penalty (-15cp) - previously missing entirely
- Connected pawns bonus (+5cp)
- Knight outpost bonus (+20cp)

### Deployment
- **Status**: Short-lived (replaced by v17.7 same day)
- **Platform**: GCP v7p3r-production-bot

### Impact
- Reduced isolated pawns from 159/game target to <50/game
- Improved pawn structure understanding

---

## [17.5.0] - 2025-12-03

### Added
- Pure endgame detection (≤6 pieces total)
- Skip PST in simplified endgames (38% speedup)
- Mate threat detection for opponent mate-in-1/2

### Fixed
- Endgame blunders reduced from 7.0 to 5.29/game
- Improved tactical awareness in simplified positions

### Deployment
- **Status**: Stable deployment (Dec 3-6)
- **Platform**: GCP v7p3r-production-bot
- **Replaced**: v17.1 (rollback recovery version)

---

## [17.4.0] - 2025-11-26 [ROLLED BACK]

### Added
- Endgame evaluation changes
- Rook bonus modifications
- **RAISED** endgame detection threshold from 800cp to 1300cp

### Deployment
- **Status**: **FAILED - ROLLED BACK after 3-4 days**
- **Duration**: Nov 26-29
- **Platform**: GCP v7p3r-production-bot
- **Rollback Date**: ~Nov 29-30
- **Rollback Target**: v17.1

### Critical Issues
- Endgame blunders detected
- Game 9i883UOF: Move 23. Be2?? (missed mate in 3)
- Lost to human player (v7p3r) in test game
- Very high CPL (~2000+)

### Root Cause
- Endgame threshold change (800cp → 1300cp) caused evaluation bugs
- Enhanced endgame features introduced tactical oversights

### Lesson Learned
- Never raise endgame detection threshold without extensive testing
- Regression suite needed for mate detection
- Require 50+ game validation before production deployment

---

## [17.3.0] - 2025-11-26 [EXPERIMENTAL - NOT DEPLOYED]

### Added
- **MAJOR REWRITE**: SEE-based quiescence search
- Static Exchange Evaluation instead of deep recursive search
- Maximum quiescence depth: 3 plies (reduced from 10)
- 83% move stability (vs 50% in v17.1.1)

### Status
- **Tested locally, NEVER deployed to production Lichess bot**
- Experimental branch separate from production lineage
- Good results in local testing

### Notes
- This version exists outside the main production timeline
- v17.4 was NOT based on v17.3
- Consider integrating SEE improvements in future version

---

## [17.2.0] - 2025-11-21

### Added
- UCI enhancements (seldepth, hashfull reporting)
- Performance optimizations

### Deployment
- **Status**: Stable 5-day deployment
- **Duration**: Nov 21-26
- **Platform**: GCP v7p3r-production-bot
- **Replaced by**: v17.4 (which failed and was rolled back)

---

## [17.1.1] - 2025-11-21

### Fixed
- Time management for Lichess cloud deployment
- Minor UCI patches

### Deployment
- **Status**: Short-lived (hours)
- **Replaced by**: v17.2.0 same day

---

## [17.1.0] - 2025-11-21

### Fixed
- **CRITICAL**: Disabled PV instant moves (caused all 3 tournament losses)
- Fixed severe Black-side weakness from v17.0

### Added
- Opening book from v16.2
- Tournament reliability improvements

### Deployment
- **Initial**: Nov 21 (few hours, replaced by v17.1.1 → v17.2.0)
- **Rollback Deployment**: Nov 30 - Dec 2 (after v17.4 failure)
- **Status**: Proven stable, used twice in production

### Impact
- Fixed Black-side weakness (v17.0 had 100% win as White, only 64% as Black)
- Tournament reliable

---

## [17.0.0] - 2025-11-20

### Added
- **BREAKTHROUGH**: Relaxed time management → deeper search
- Average depth: 4.8 (vs 4.5 in previous versions)
- History heuristic
- Killer moves
- MVV-LVA move ordering
- Enhanced transposition table

### Deployment
- **Status**: Retired after 1 day
- **Duration**: Nov 20-21
- **Replaced by**: v17.1.0

### Performance
- **Tournament Result**: 1st place, 44.0/50 points (88% win rate)
- **ELO**: ~1600
- **Critical Issue**: Severe Black-side weakness
  - White: 100% win rate
  - Black: 64% win rate
- All 3 tournament losses from PV instant move bug

---

## [16.1.0] - 2025-11-19

### Added
- Opening book experiments
- Tablebase integration attempts

### Deployment
- **Duration**: Nov 19-20 (1 day)
- **Replaced by**: v17.0

---

## [14.1.0] - 2025-10-25

### Added
- Major evaluation overhaul
- Improved positional understanding
- Smart time management

### Deployment
- **Status**: Stable long-running deployment
- **Duration**: 25 days (Oct 25 - Nov 19)
- **Platform**: GCP v7p3r-production-bot
- **ELO**: 1496

### Performance
- Longest stable deployment in v14-v17 series
- Consistent results across time controls

---

## [12.6.0] - 2025-10-04

### Changed
- Removed nudge system
- Clean performance build
- Consolidated evaluation

### Notes
- Foundation for v14.x and v17.x series
- Many v13.x-v16.x versions were experimental
- This represents the "true v7" lineage continuation

---

## Version Lineage Truth

The version numbers don't reflect true linear evolution:

**Actual Evolution Path**:
- v10.8 (recovery baseline) 
- → v12.0-v12.6 (clean builds)
- → v14.0-v14.1 (smart time management)
- → v17.0 (breakthrough performance)
- → v17.1 (PV fix)
- → v17.2.0 (stable)
- → [v17.3 experimental branch - not deployed]
- → v17.4 (failed, rolled back)
- → v17.5-v17.7 (progressive recovery)
- → v17.8 (current)

**Skipped/Experimental**: v11.x, v13.x, v15.x, v16.x were experiments or copies.

**True Count**: v17.8 is approximately the 8th or 9th major iteration of working code.

---

## Deployment Log

See `deployment_log.json` for machine-readable deployment history with:
- Exact deployment timestamps
- Production vs testing deployments
- ELO ratings per version
- Rollback history
- Performance metrics

## Testing Requirements

Before production deployment, all versions must pass:
1. **Regression Suite**: 20+ tactical positions including mate-in-3 detection
2. **Performance Benchmark**: 50-game tournament vs last stable version
3. **Acceptance Criteria**: Win% ≥48%, Blunders ≤6.0/game, Time Forfeit <10%
4. **Time Control Validation**: Test in bullet, blitz, and rapid
5. **Git Tag**: Create annotated tag before deployment

---

## Links

- **Production Deployment**: `lichess/CHANGELOG.md`
- **Deployment Guide**: `lichess/docs/CLOUD_DEPLOYMENT_GUIDE.md`
- **Testing Guide**: `docs/TESTING_GUIDE.md`
- **Version Management**: `.github/instructions/version_management.instructions.md`
