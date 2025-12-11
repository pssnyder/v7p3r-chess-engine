---
applyTo: '**'
---
Coding standards, domain knowledge, and preferences that AI should follow.

# Version Management Instructions

This document defines the version management, testing, and deployment workflow for V7P3R Chess Engine development.

## Semantic Versioning (v18.0.0+)

Starting with v18.0.0, use semantic versioning: **MAJOR.MINOR.PATCH**

### Version Components
- **MAJOR** (18.x.x): Breaking changes, significant evaluation rewrites, algorithm overhauls
  - Example: Complete quiescence search rewrite (SEE-based)
  - Example: New neural network evaluation integration
  
- **MINOR** (x.2.x): New features, evaluation improvements, non-breaking algorithm changes
  - Example: Added bishop pair bonus, isolated pawn penalty
  - Example: Time manager integration
  - Example: Mate-in-1 fast path detection
  
- **PATCH** (x.x.1): Bug fixes, parameter tuning, performance tweaks
  - Example: Repetition threshold change (200cp → 50cp)
  - Example: Fix PV instant move bug
  - Example: UCI reporting improvements

### Legacy Versions (v17.x and earlier)
- v17.x series used incremental numbering without semantic meaning
- Many versions (v11.x, v13.x, v15.x, v16.x) were experiments or rollbacks
- True evolutionary count: v17.8 ≈ 8th-9th major iteration
- Continue v17.x numbering for minor fixes, transition to v18.0.0 for next major feature

## Git Workflow

### Branch Strategy
```
main (production)
  ├── develop (integration testing)
  │   ├── feature/time-manager
  │   ├── feature/mate-detection
  │   └── feature/see-quiescence
  └── hotfix/repetition-threshold
```

### Branch Types
- **main**: Production-deployed code only, always stable
- **develop**: Integration branch for testing before production
- **feature/\***: Individual features (merge to develop, never main)
- **hotfix/\***: Critical fixes (can merge to main after validation)
- **experimental/\***: Research branches (like v17.3 SEE rewrite)

### Commit Messages
Use conventional commits format:
```
feat: add bishop pair bonus (+30cp)
fix: lower repetition threshold from 200cp to 50cp
perf: skip PST in pure endgames (38% speedup)
docs: update CHANGELOG for v17.8 release
test: add mate-in-3 regression test
```

### Git Tags
Create annotated tags for every production deployment:
```bash
git tag -a v17.8.0 -m "v17.8.0: Repetition threshold fix for rapid game improvement"
git push origin v17.8.0
```

## Version Lifecycle

### 1. Development Phase
- Work on `feature/*` or `develop` branch
- Run local tests continuously
- Document changes in feature branch

### 2. Testing Phase
**REQUIRED before any production deployment:**

#### A. Regression Suite (Must Pass 100%)
```bash
python testing/regression_suite.py
```
Tests must include:
- Mate-in-3 detection (v17.4 failure case)
- Endgame conversion (R+B vs K)
- Tactical positions (pins, forks, skewers)
- Opening book positions
- Time management scenarios

#### B. Performance Benchmark (50+ games)
```bash
python testing/performance_benchmark.py --version v17.8.0 --baseline v17.7.0 --games 50
```

**Acceptance Criteria** (ALL must pass):
- Win Rate: ≥48% (against equal-strength baseline)
- Blunders/Game: ≤6.0
- Time Forfeit Rate: <10%
- CPL (Centipawn Loss): <150
- No critical errors in logs

#### C. Time Control Validation
Test in all formats:
- Bullet: 1min+2s (20 games minimum)
- Blitz: 5min+4s (20 games minimum)
- Rapid: 15min+10s (20 games minimum)

### 3. Documentation Phase
**REQUIRED before merge to main:**

#### Update CHANGELOG.md
```markdown
## [17.8.0] - 2025-12-10

### Changed
- Lowered repetition threshold from 200cp to 50cp

### Rationale
- v17.7 accepting draws at +100cp caused rapid game regression
- 50cp threshold more aggressive, aligns with competitive philosophy

### Testing
- ✅ Regression suite: 100% pass
- ✅ Performance: 52% win rate vs v17.7 (52W-48L in 100 games)
- ✅ Time controls: All formats tested
```

#### Update deployment_log.json
```json
{
  "version": "17.8.0",
  "deployed": "2025-12-10",
  "status": "production",
  "regression_tests_passed": true,
  "acceptance_criteria": {
    "win_rate": 0.52,
    "blunders_per_game": 5.1,
    "time_forfeit_rate": 0.08,
    "tested": true
  }
}
```

### 4. Merge & Tag
```bash
# Merge to develop first
git checkout develop
git merge feature/repetition-threshold
python testing/regression_suite.py  # Validate on develop

# Merge to main
git checkout main
git merge develop

# Tag release
git tag -a v17.8.0 -m "v17.8.0: Repetition threshold fix"
git push origin main --tags
```

### 5. Deployment Phase
See **Production Deployment Workflow** section below.

## Production Deployment Workflow

### Pre-Deployment Checklist
- [ ] All regression tests pass (100%)
- [ ] Performance benchmark meets acceptance criteria
- [ ] CHANGELOG.md updated
- [ ] deployment_log.json updated
- [ ] Git tag created
- [ ] Rollback plan documented
- [ ] Deployment window scheduled (low-traffic hours)

### Deployment Steps

#### Step 1: Create Deployment Package
```bash
cd "s:/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine"

# Create timestamped deployment folder
mkdir -p "lichess/engines/V7P3R_v17.8.0_$(date +%Y%m%d)"

# Copy engine files
cp -r src/ "lichess/engines/V7P3R_v17.8.0_$(date +%Y%m%d)/"
cp requirements.txt "lichess/engines/V7P3R_v17.8.0_$(date +%Y%m%d)/"
cp README.md "lichess/engines/V7P3R_v17.8.0_$(date +%Y%m%d)/"

# Create tarball for upload
cd "lichess/engines/V7P3R_v17.8.0_$(date +%Y%m%d)"
tar -czf ../v17.8.0-src.tar.gz src/
```

#### Step 2: Upload to GCP VM
```bash
# Upload tarball to VM
gcloud compute scp ../v17.8.0-src.tar.gz v7p3r-production-bot:/home/patss/ \
  --zone=us-central1-a

# Verify upload
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a \
  --command="ls -lh /home/patss/v17.8.0-src.tar.gz"
```

#### Step 3: Backup Current Production Version
```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
  # Backup current version with timestamp
  BACKUP_DATE=\$(date +%Y%m%d_%H%M%S)
  sudo docker exec v7p3r-production bash -c \
    'tar -czf /tmp/v7p3r_backup_\$BACKUP_DATE.tar.gz /lichess-bot/engines/v7p3r'
  
  # Copy backup out of container
  sudo docker cp v7p3r-production:/tmp/v7p3r_backup_\$BACKUP_DATE.tar.gz /home/patss/backups/
  
  # Verify backup
  ls -lh /home/patss/backups/v7p3r_backup_\$BACKUP_DATE.tar.gz
"
```

#### Step 4: Deploy New Version
```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
  # Copy new version into container
  sudo docker cp v17.8.0-src.tar.gz v7p3r-production:/tmp/
  
  # Move current version to backup location inside container
  sudo docker exec v7p3r-production bash -c \
    'mv /lichess-bot/engines/v7p3r /lichess-bot/engines/v7p3r.backup'
  
  # Create new engine directory
  sudo docker exec v7p3r-production mkdir -p /lichess-bot/engines/v7p3r
  
  # Extract new version
  sudo docker exec v7p3r-production bash -c \
    'cd /lichess-bot/engines/v7p3r && tar -xzf /tmp/v17.8.0-src.tar.gz --strip-components=1'
  
  # Verify extraction
  sudo docker exec v7p3r-production ls -la /lichess-bot/engines/v7p3r/
"
```

#### Step 5: Restart Bot & Monitor
```bash
# Restart container to load new engine
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a \
  --command="sudo docker restart v7p3r-production"

# Wait for container to start
sleep 10

# Check logs for successful startup
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a \
  --command="sudo docker logs v7p3r-production --tail 50"

# Verify engine version in UCI
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
  sudo docker exec v7p3r-production bash -c \
    'echo uci | python /lichess-bot/engines/v7p3r/v7p3r_uci.py | grep \"id name\"'
"
# Expected output: id name V7P3R v17.8
```

#### Step 6: Post-Deployment Validation
- **Monitor first 5 games closely** via Lichess web interface
- Check for errors in Docker logs: `sudo docker logs v7p3r-production -f`
- Verify move times are reasonable (not timing out)
- Watch for blunders or unusual play
- Check game results (win/loss/draw ratios)

### Rollback Procedure (If Issues Detected)

```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
  # Remove failed version
  sudo docker exec v7p3r-production bash -c \
    'rm -rf /lichess-bot/engines/v7p3r'
  
  # Restore backup
  sudo docker exec v7p3r-production bash -c \
    'mv /lichess-bot/engines/v7p3r.backup /lichess-bot/engines/v7p3r'
  
  # Restart container
  sudo docker restart v7p3r-production
"

# Update deployment_log.json
# Mark version as "rollback": true, document reason
```

### Post-Deployment Tasks
1. Update `lichess/CHANGELOG.md` with deployment timestamp
2. Update `deployment_log.json` with final status
3. Monitor performance for 24-48 hours
4. Compare stats against baseline (win%, blunders, CPL)
5. If stable after 48 hours, mark as "stable" in deployment_log.json

## GCP Production Environment

### Instance Details
- **Name**: v7p3r-production-bot
- **Type**: e2-medium (4GB RAM, 2 vCPUs)
- **Region**: us-central1-a
- **OS**: Debian-based Linux
- **Cost**: ~$24/month

### Docker Container
- **Container Name**: v7p3r-production
- **Base Image**: lichess-bot (lichess-bot framework 2025.10.1.2)
- **Engine Path**: `/lichess-bot/engines/v7p3r/`
- **Config**: `/lichess-bot/config.yml`

### Access Commands
```bash
# SSH into VM
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a

# Access Docker container
sudo docker exec -it v7p3r-production bash

# View bot logs
sudo docker logs v7p3r-production -f

# View engine directory
sudo docker exec v7p3r-production ls -la /lichess-bot/engines/v7p3r/

# Check bot status
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a \
  --command="sudo docker exec v7p3r-production supervisorctl status"
```

### File Paths
- **VM Backups**: `/home/patss/backups/`
- **Container Engine**: `/lichess-bot/engines/v7p3r/`
- **Container Config**: `/lichess-bot/config.yml`
- **Container Logs**: `/lichess-bot/logs/`

## Regression Prevention

### Never Deploy Without
1. **Regression Test Suite**: Automated tests for all known failure modes
   - v17.4 mate-in-3 miss (game 9i883UOF, move 23. Be2??)
   - Endgame conversion positions (R+B vs K)
   - Threefold repetition handling
   - Time forfeit prevention
   
2. **Performance Benchmark**: 50+ game tournament vs stable baseline
   - Must meet acceptance criteria
   - Test all time controls
   
3. **Code Review**: Review diffs before deployment
   - Check for unintended changes
   - Verify version numbers updated
   - Confirm evaluation changes are intentional

### Known Failure Patterns
Document all production failures for regression prevention:

**v17.4 Endgame Failure**:
- **Issue**: Raised endgame threshold from 800cp to 1300cp
- **Result**: Missed mate-in-3, high CPL, blunders
- **Prevention**: Never raise endgame detection threshold without extensive testing
- **Test**: Add mate-in-3 position to regression suite

**v17.0 Black-Side Weakness**:
- **Issue**: PV instant move caused all 3 tournament losses
- **Result**: 100% win as White, 64% as Black
- **Prevention**: Disable PV instant moves, test color balance
- **Test**: Run 50/50 White/Black split in benchmarks

**v17.7 Rapid Regression**:
- **Issue**: 200cp repetition threshold too conservative
- **Result**: Accepting draws with +1 pawn advantage
- **Prevention**: Test threshold changes across time controls
- **Test**: Track draw rate in benchmarks, compare vs baseline

## Version Comparison & Tournament Testing

### Local Regression Tournament
Before deploying any new version, run a local tournament:

```bash
# Arena GUI setup (Windows)
# 1. Add all test versions to Arena engines list
# 2. Create tournament with these settings:
#    - Engines: v17.1, v17.7, v17.8 (new version)
#    - Games: 100 per pairing (300 total)
#    - Time Control: 5min+4s (blitz)
#    - Opening Book: Disabled or consistent across all
#    - Positions: Varied (use opening suite)
```

### Expected Results
- **New version should score ≥48% against stable baseline**
- If new version scores <45%, investigate regression
- If new version scores >55%, validate improvement is real (not luck)
- Draw rate should be stable (±5% of baseline)

### Multi-Time-Control Validation
```bash
# Bullet tournament (1min+2s)
# Blitz tournament (5min+4s)
# Rapid tournament (15min+10s)

# Compare performance across formats
# New version should be stable or improved in ALL formats
```

## Documentation Standards

### Every Version Must Have
1. **CHANGELOG.md entry**
   - Version number
   - Date
   - Changes (Added/Changed/Fixed/Removed)
   - Rationale
   - Testing results
   - Known issues

2. **deployment_log.json entry**
   - Version, date, status
   - Environment, platform
   - Duration, ELO
   - Changes, rollback status
   - Regression tests results
   - Acceptance criteria results

3. **Git tag**
   - Annotated tag with version and summary
   - Pushed to origin

4. **Release notes** (for major versions)
   - High-level summary
   - Performance improvements
   - Breaking changes
   - Upgrade instructions

## Version Comparison Guide

When user asks "which version should I use?":

### Current Recommendations
- **Production Stable**: v17.7.0 (4+ day stable deployment)
- **Rollback Target**: v17.1.0 (proven stable, deployed twice)
- **Emergency Fallback**: v14.1.0 (25-day stable deployment)
- **Testing**: v17.8.0 (requires validation)

### Version Selection Criteria
1. **Stability**: Days in production without issues
2. **Performance**: ELO rating, win rate, blunders/game
3. **Recency**: Newer versions have more features
4. **Testing**: Regression tests passed, acceptance criteria met

### Never Use
- **v17.4.0**: Endgame blunders, rolled back (PERMANENT BLACKLIST)
- **v17.3.0**: Experimental, never deployed to production
- **v15.x, v16.x**: Skipped/experimental versions

## AI Assistant Workflow

When implementing new features:

1. **Check current version** in src/v7p3r.py and src/v7p3r_uci.py
2. **Create feature branch** if substantial change
3. **Implement changes** following code preservation instructions
4. **Update version number** following semantic versioning
5. **Update CHANGELOG.md** with detailed entry
6. **Update deployment_log.json** with "testing" status
7. **Run regression tests** (if available)
8. **Document testing plan** for user validation
9. **Create git tag** after user approval
10. **Update deployment instructions** if needed

### Version Number Updates Required In
- `src/v7p3r.py`: Header comment and VERSION_LINEAGE
- `src/v7p3r_uci.py`: UCI "id name" response
- `CHANGELOG.md`: New entry at top
- `deployment_log.json`: New entry in deployment_history
- Git tag: Annotated tag with version number

### Before Any Production Deployment
- [ ] Read current CHANGELOG.md to understand version history
- [ ] Check deployment_log.json for last stable version
- [ ] Verify all tests pass
- [ ] Confirm user has validated changes
- [ ] Document rollback procedure
- [ ] Update all version references consistently

## Emergency Rollback

If critical issue discovered in production:

```bash
# IMMEDIATE ROLLBACK (use backup inside container)
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
  sudo docker exec v7p3r-production bash -c \
    'rm -rf /lichess-bot/engines/v7p3r && mv /lichess-bot/engines/v7p3r.backup /lichess-bot/engines/v7p3r'
  sudo docker restart v7p3r-production
"

# POST-ROLLBACK TASKS
# 1. Update deployment_log.json: Mark failed version with rollback=true
# 2. Document failure reason in CHANGELOG.md
# 3. Create regression test for failure case
# 4. Investigate root cause before attempting re-deployment
```

## Summary

**Golden Rules**:
1. Never deploy without regression tests passing
2. Never reuse version numbers
3. Always create git tags for deployments
4. Always maintain rollback backup in container
5. Document every deployment in CHANGELOG.md and deployment_log.json
6. Test across all time controls before production
7. Monitor first 24 hours after deployment closely
8. When in doubt, rollback and investigate

**Version Truth**: The version number is less important than the code quality. v17.8 might be the "8th real iteration" but what matters is stability, testing, and proper rollback capability.
