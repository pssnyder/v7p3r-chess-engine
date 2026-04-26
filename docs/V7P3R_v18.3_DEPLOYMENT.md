# V7P3R v18.3.0 Production Deployment

**Version**: v18.3.0 - PST Optimization  
**Date**: 2025-12-29  
**Deployment Target**: v7p3r-production-bot (GCP e2-medium)  
**Status**: ✅ Ready for deployment

## Pre-Deployment Summary

### Changes in v18.3.0
- **PST Optimization**: Direct square indexing (28% PST speedup)
- **PST Optimization**: Pre-computed flipped tables for Black
- **Architecture**: Decomposed fast evaluator (modular components)
- **Performance**: 23% faster full evaluation (0.046ms → 0.037ms)

### Validation Results
- ✅ **25-game tournament**: 58% vs v17.1 (+56 ELO)
- ✅ **Win/Loss/Draw**: 8-4-13 (14.5-10.5 points)
- ✅ **No regressions**: Identical depth (4.2-4.3), stable play
- ✅ **Quality gate**: Exceeds 48% minimum (58% achieved)
- ✅ **Draw rate**: 52% indicates sound chess
- ✅ **Win ratio**: 2:1 when positions diverge

### Files Prepared
- ✅ Engine files: `lichess/engines/V7P3R_v18.3_20251229/src/`
- ✅ Deployment tarball: `lichess/engines/v18.3.0-src.tar.gz` (123KB)
- ✅ Config updated: `lichess/config.yml` → v18.3 directory
- ✅ CHANGELOG updated: v18.3.0 entry added
- ✅ Deployment log updated: v18.3.0 production entry

## Deployment Steps

### Step 1: Upload Tarball to GCP VM

```powershell
# Upload tarball to VM
gcloud compute scp "s:\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\lichess\engines\v18.3.0-src.tar.gz" v7p3r-production-bot:/home/patss/ --zone=us-central1-a

# Verify upload
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="ls -lh /home/patss/v18.3.0-src.tar.gz"
```

**Expected output**: File size ~123KB, timestamp today

### Step 2: Backup Current Production Version

```powershell
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
  # Create timestamped backup
  BACKUP_DATE=\$(date +%Y%m%d_%H%M%S)
  sudo docker exec v7p3r-production bash -c \
    'tar -czf /tmp/v7p3r_backup_\${BACKUP_DATE}.tar.gz /lichess-bot/engines/v7p3r'
  
  # Copy backup out of container
  sudo docker cp v7p3r-production:/tmp/v7p3r_backup_\${BACKUP_DATE}.tar.gz /home/patss/backups/
  
  # Verify backup
  ls -lh /home/patss/backups/v7p3r_backup_\${BACKUP_DATE}.tar.gz
"
```

**Expected output**: Backup file created in `/home/patss/backups/`

### Step 3: Deploy v18.3.0

```powershell
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
  # Copy tarball into container
  sudo docker cp /home/patss/v18.3.0-src.tar.gz v7p3r-production:/tmp/
  
  # Move current version to backup location inside container
  sudo docker exec v7p3r-production bash -c \
    'mv /lichess-bot/engines/v7p3r /lichess-bot/engines/v7p3r.backup'
  
  # Create new engine directory
  sudo docker exec v7p3r-production mkdir -p /lichess-bot/engines/v7p3r
  
  # Extract new version
  sudo docker exec v7p3r-production bash -c \
    'cd /lichess-bot/engines/v7p3r && tar -xzf /tmp/v18.3.0-src.tar.gz --strip-components=1'
  
  # Verify extraction
  sudo docker exec v7p3r-production ls -la /lichess-bot/engines/v7p3r/
"
```

**Expected output**: Source files visible in `/lichess-bot/engines/v7p3r/` directory

### Step 4: Update Configuration

```powershell
# Upload updated config.yml
gcloud compute scp "s:\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\lichess\config.yml" v7p3r-production-bot:/home/patss/ --zone=us-central1-a

# Copy config into container
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
  sudo docker cp /home/patss/config.yml v7p3r-production:/lichess-bot/config.yml
  
  # Verify config
  sudo docker exec v7p3r-production grep 'V7P3R_v18.3' /lichess-bot/config.yml
"
```

**Expected output**: `dir: "./engines/V7P3R_v18.3_20251229/src/"`

### Step 5: Restart Bot & Validate

```powershell
# Restart container to load new engine
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker restart v7p3r-production"

# Wait for container to start
Start-Sleep -Seconds 15

# Check logs for successful startup
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker logs v7p3r-production --tail 50"

# Verify engine version via UCI
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
  sudo docker exec v7p3r-production bash -c \
    'echo uci | python /lichess-bot/engines/v7p3r/v7p3r_uci.py | grep \"id name\"'
"
```

**Expected output**: `id name V7P3R v18.3`

### Step 6: Monitor First Games

- **Watch first 5 games** via Lichess web interface: https://lichess.org/@/v7p3r_bot
- **Check for errors**: `sudo docker logs v7p3r-production -f`
- **Verify move times**: Should be similar to v17.1 (not timing out)
- **Check for blunders**: Watch for unusual play or tactical mistakes
- **Monitor results**: Win/loss/draw ratio should be stable

### Post-Deployment Validation (24-48 hours)

- [ ] No critical errors in logs
- [ ] No time forfeits
- [ ] Move times reasonable (not timing out)
- [ ] Performance stable (win rate near expected)
- [ ] Depth achieved comparable to v17.1 (4.2-4.3 average)

## Rollback Procedure (If Issues Detected)

```powershell
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
```

**After rollback**: 
1. Update `deployment_log.json` with rollback status
2. Document failure reason in CHANGELOG.md
3. Create regression test for failure case
4. Investigate root cause before re-attempting

## Success Criteria

- ✅ Engine starts without errors
- ✅ UCI version reports "V7P3R v18.3"
- ✅ Bot accepts and plays games normally
- ✅ No time forfeits in first 10 games
- ✅ Move quality similar to previous version
- ✅ No critical errors in 24-hour monitoring period

## Expected Performance

Based on 25-game validation tournament:
- **Win rate vs equal opponents**: 50-55% (conservative)
- **Draw rate**: 45-55% (sound positional play)
- **Average depth**: 4.2-4.3 plies
- **Blunders per game**: <6.0 (maintained from v17.1)
- **Time forfeit rate**: <5%

## Notes

- **Clean optimization**: Same algorithm, just faster implementation
- **No behavioral changes**: PST values unchanged, only lookup optimized
- **Foundation prepared**: For lazy evaluation and cache improvements (Tier 1 roadmap)
- **Low risk**: 52% draw rate in testing indicates identical play in most positions
- **Proven gains**: 2:1 win ratio when positions diverge from baseline

## Contact

- **Engineer**: Pat Snyder
- **Deployment Date**: 2025-12-29
- **Git Tag**: v18.3.0
- **Deployment Log**: Updated with production status

---

**Deployment Authorization**: Ready to proceed ✅
