# V7P3R Lichess Bot - Deployment Changelog

This document tracks all engine deployments, rollbacks, and configuration changes for v7p3r_bot on Lichess.

---

## 2025 Deployment History

### November 2025

#### **2025-11-30 (Current)**
- **Version**: v17.1 (ROLLED BACK)
- **Status**: ACTIVE on production
- **Reason**: v17.4 had endgame blunders, rolled back to stable v17.1
- **Games Period**: 2025-11-26 onwards (after v17.4 rollback)
- **Notes**: 
  - v17.4 showed critical blunders in endgame (mate in 3 missed)
  - Lost to human player (v7p3r) in test game
  - Rolled back to v17.1 for stability
  - Smart matchmaking system deployed and active

#### **2025-11-26 to 2025-11-30**
- **Version**: v17.4 (DEPLOYED THEN ROLLED BACK)
- **Status**: RETIRED (rollback after ~4 days)
- **Games Period**: 2025-11-26 to ~2025-11-29
- **Deployment**: Deployed to GCP production
- **Features**:
  - Endgame improvements (rook bonus, improved detection)
  - Enhanced endgame evaluation
- **Issues**:
  - Critical blunder detected: move 23. Be2?? (Mate in 3 missed)
  - Lost to v7p3r human player (game 9i883UOF)
  - High centipawn loss in endgames
- **Rollback Date**: ~2025-11-29/30
- **Rollback Reason**: Endgame evaluation regression

#### **2025-11-21 to 2025-11-26**
- **Version**: v17.2.0
- **Status**: RETIRED
- **Games Period**: 2025-11-21 to 2025-11-26
- **Deployment**: GCP production deployment
- **Features**:
  - Positional evaluation improvements
  - Opening book enhancements
- **Notes**: Stable version, replaced by v17.4

#### **2025-11-21** (Earlier)
- **Version**: v17.1.1
- **Status**: RETIRED (short-lived)
- **Games Period**: Brief deployment on 2025-11-21
- **Notes**: Minor patch, quickly superseded by v17.2.0

#### **2025-11-21** (Initial)
- **Version**: v17.1
- **Status**: RETIRED (deployed 2025-11-21, now ACTIVE AGAIN after rollback)
- **Games Period**: 
  - Initial: 2025-11-21 (few hours)
  - Current: 2025-11-26+ (after rollback from v17.4)
- **Deployment**: Initial GCP deployment, now active again
- **Features**:
  - Stable endgame play
  - Reliable evaluation
- **Notes**: 
  - Proven stable version
  - Currently active after v17.4 rollback
  - **This is the engine analyzing games Nov 26-30**

#### **2025-11-20**
- **Version**: v17.0
- **Status**: RETIRED
- **Games Period**: 2025-11-20 to 2025-11-21
- **Notes**: Initial v17 series deployment

#### **2025-11-19**
- **Version**: v16.1
- **Status**: RETIRED
- **Games Period**: 2025-11-19 to 2025-11-20
- **Notes**: Last of v16 series

---

### October 2025

#### **2025-10-25**
- **Version**: v14.1
- **Status**: RETIRED
- **Games Period**: 2025-10-25 to 2025-11-19
- **Deployment**: Deployed to GCP
- **Features**:
  - Major evaluation overhaul
  - Improved positional understanding
- **Notes**: Long-running stable version (~25 days)

#### **2025-10-25** (Earlier)
- **Version**: v14.0
- **Status**: RETIRED (short-lived)
- **Games Period**: Brief deployment on 2025-10-25
- **Notes**: Quickly patched to v14.1

#### **2025-10-04**
- **Version**: v12.6
- **Status**: RETIRED
- **Games Period**: 2025-10-04 to 2025-10-25
- **Notes**: Last of v12 series, stable for ~21 days

#### **2025-10-03** (Later)
- **Version**: v12.4
- **Status**: RETIRED (short-lived)
- **Games Period**: Brief deployment on 2025-10-03
- **Notes**: Quick iteration in v12 series

#### **2025-10-03** (Initial)
- **Version**: v12.2
- **Status**: RETIRED
- **Games Period**: 2025-10-03 (few hours)
- **Notes**: Initial v12 series deployment

---

## Current Production Environment

### Active Configuration
- **Version**: v17.1 (rolled back from v17.4)
- **GCP Project**: v7p3r-lichess-bot
- **Instance**: v7p3r-production-bot (e2-micro, us-central1-a)
- **Container**: v7p3r-production
- **ELO**: ~1614 (Rapid)
- **Active Since**: 2025-11-26 (rollback date)

### Smart Matchmaking
- **Deployed**: 2025-11-30
- **Status**: ACTIVE
- **Features**:
  - Intelligent opponent selection
  - 30% improvement targets, 50% variety, 20% priority
  - Slot reservation for incoming challenges
  - Followed accounts integration

---

## Engine Version Timeline Summary

```
Oct 3  ──┬── v12.2 (hours)
         ├── v12.4 (hours)
         └── v12.6 ────────────────────────┐
                                            │ (21 days)
Oct 25 ──┬── v14.0 (hours)                 │
         └── v14.1 ─────────────────────────┴────────────┐
                                                          │ (25 days)
Nov 19 ─── v16.1 ────────┐                              │
                          │ (1 day)                      │
Nov 20 ─── v17.0 ────────┤                              │
                          │ (1 day)                      │
Nov 21 ──┬── v17.1 ──────┤                              │
         ├── v17.1.1 (hours)                            │
         └── v17.2.0 ─────────────────┐                 │
                                       │ (5 days)        │
Nov 26 ─── v17.4 ────────────────┐    │                 │
                                  │ (4 days)             │
Nov 30 ─── v17.1 (ROLLBACK) ◄────┴────┴─────────────────┘
           [CURRENT VERSION]
```

---

## Game Analysis Mapping

Use this table to filter analytics by engine version:

| Date Range | Version | Status | Notes |
|------------|---------|--------|-------|
| 2025-11-30+ | v17.1 | ACTIVE | Rolled back from v17.4 |
| 2025-11-26 to 2025-11-29 | v17.4 | ROLLED BACK | Endgame issues |
| 2025-11-21 to 2025-11-26 | v17.2.0 | Stable | - |
| 2025-11-21 (brief) | v17.1.1 | Short-lived | - |
| 2025-11-21 (initial) | v17.1 | Initial deployment | - |
| 2025-11-20 to 2025-11-21 | v17.0 | Stable | - |
| 2025-11-19 to 2025-11-20 | v16.1 | Stable | - |
| 2025-10-25 to 2025-11-19 | v14.1 | Stable (25 days) | - |
| 2025-10-25 (brief) | v14.0 | Short-lived | - |
| 2025-10-04 to 2025-10-25 | v12.6 | Stable (21 days) | - |
| 2025-10-03 (later) | v12.4 | Short-lived | - |
| 2025-10-03 (initial) | v12.2 | Short-lived | - |

---

## Analytics Filtering Guide

When analyzing games, use these filters to isolate specific versions:

### Current Version (v17.1 - Post-Rollback)
```bash
# Games since rollback
python full_analysis.py --since 2025-11-30
```

### v17.4 Performance (ROLLED BACK)
```bash
# Games during v17.4 deployment (to identify issues)
python full_analysis.py --since 2025-11-26 --until 2025-11-29
```

### v17.2.0 Performance
```bash
# Stable v17.2.0 period
python full_analysis.py --since 2025-11-21 --until 2025-11-26
```

### v14.1 Long-Term Performance
```bash
# 25-day stable run
python full_analysis.py --since 2025-10-25 --until 2025-11-19
```

---

## Deployment Checklist

When deploying new engine versions:

- [ ] Build engine executable
- [ ] Create dated directory: `V7P3R_vX.X_YYYYMMDD/`
- [ ] Test locally with lichess-bot
- [ ] Deploy to GCP container
- [ ] Monitor first 10 games for issues
- [ ] Update this CHANGELOG.md
- [ ] If issues found, rollback and document here
- [ ] Run analytics after 24 hours to validate performance

---

## Rollback Procedure

When rolling back (as done with v17.4):

1. Stop lichess-bot container
2. Update `config.yml` engine path to previous version
3. Restart container
4. Verify version in logs
5. Document rollback in this changelog
6. Run analytics to confirm regression

---

## Known Issues by Version

### v17.4 (ROLLED BACK)
- **Issue**: Critical endgame blunders
- **Example**: Game 9i883UOF - move 23. Be2?? (Mate in 3 missed)
- **CPL**: Very high average (~2000+)
- **Resolution**: Rolled back to v17.1

### v17.1 (CURRENT)
- **Status**: Stable
- **CPL**: Moderate (~1500-1800 average)
- **Notes**: Reliable endgame play, consistent performance

---

**Last Updated**: 2025-11-30  
**Current Version**: v17.1 (rolled back from v17.4)  
**Maintainer**: pssnyder
