# V7P3R Lichess Bot - Deployment Changelog

**Purpose**: Track all engine deployments, rollbacks, and configuration changes for v7p3r_bot on Lichess  
**Maintainer**: pssnyder  
**Last Updated**: 2026-04-26

---

## Quick Reference

**Current Active Version**: v18.4 (deployed 2026-04-17)  
**GCP Project**: v7p3r-lichess-bot  
**Instance**: v7p3r-production-bot (e2-micro, us-central1-a)  
**ELO Rating**: ~1633 (Rapid)

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

### v18.4
- **Deployed**: 2026-04-17
- **Retired**: [ACTIVE]
- **Status**: active
- **Rollback**: false
- **Duration Days**: [TBD]
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1633
- **Games Played**: [TBD]
- **Features**:
  - Latest stable release
- **Known Issues**: []
- **Rollback Reason**: N/A
- **Notes**: Currently active in production

### v18.3
- **Deployed**: 2025-12-29
- **Retired**: 2026-04-17
- **Status**: retired
- **Rollback**: false
- **Duration Days**: 110
- **Deployment Method**: automated
- **Environment**: production
- **ELO Rating**: 1661
- **Games Played**: [TBD]
- **Features**: []
- **Known Issues**: Smart matchmaking system believed to cause rating slide from 1722 Jan 21st-Feb 15th. Mitigated by reverting to random matchmaking for 2 weeks. Rating recovered to 1660s.
- **Rollback Reason**: N/A
- **Notes**: Stable deployment, replaced by v18.4

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
