# V7P3R Chess Engine - Quick Reference

## Version Management Quick Commands

### Check Current Version
```bash
# In source code
grep "id name" src/v7p3r_uci.py

# In production
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker exec v7p3r-production bash -c 'echo uci | python /lichess-bot/engines/v7p3r/v7p3r_uci.py | grep \"id name\"'"
```

### Create New Version
```bash
# 1. Update version in source files
# - src/v7p3r.py (header comment, VERSION_LINEAGE)
# - src/v7p3r_uci.py (UCI "id name")

# 2. Update documentation
# - CHANGELOG.md (new entry at top)
# - deployment_log.json (new entry with "testing" status)

# 3. Commit and tag
git add .
git commit -m "feat: implement v17.8 repetition threshold fix"
git tag -a v17.8.0 -m "v17.8.0: Lowered repetition threshold from 200cp to 50cp"
git push origin main --tags
```

### Deploy to Production
```bash
# Automated deployment script
cd "s:/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine"
bash scripts/deploy_to_production.sh v17.8.0
```

### Rollback to Previous Version
```bash
# Emergency rollback (uses backup inside container)
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
  sudo docker exec v7p3r-production bash -c \
    'rm -rf /lichess-bot/engines/v7p3r && mv /lichess-bot/engines/v7p3r.backup /lichess-bot/engines/v7p3r'
  sudo docker restart v7p3r-production
"
```

## Testing Quick Commands

### Run Regression Tests
```bash
python testing/regression_suite.py
```

### Run Performance Benchmark (Arena GUI)
```
1. Arena → Engines → Manage → Add v17.8.0 and v17.7.0
2. Tournaments → New Tournament
3. Select both engines, 50 games per pairing
4. Time control: 5min+4s
5. Start and wait for results
```

### Check Test Results
```bash
# View last benchmark results
cat results/benchmark_v17.8.0_vs_v17.7.0.json

# Check acceptance criteria
# - Win rate: ≥48%
# - Blunders/game: ≤6.0
# - Time forfeit rate: <10%
# - CPL: <150
```

## GCP Production Access

### SSH into VM
```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a
```

### Access Docker Container
```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker exec -it v7p3r-production bash"
```

### View Logs
```bash
# Real-time logs
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker logs v7p3r-production -f"

# Last 50 lines
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker logs v7p3r-production --tail 50"
```

### Check Bot Status
```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker exec v7p3r-production supervisorctl status"
```

## File Locations

### Local Development
- Source: `s:/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/src/`
- Documentation: `s:/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/docs/`
- Instructions: `s:/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/.github/instructions/`
- Deployment Packages: `s:/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/lichess/engines/`

### GCP VM (v7p3r-production-bot)
- VM Backups: `/home/patss/backups/`
- VM Upload Location: `/home/patss/`

### Docker Container (v7p3r-production)
- Engine: `/lichess-bot/engines/v7p3r/`
- Backup: `/lichess-bot/engines/v7p3r.backup`
- Config: `/lichess-bot/config.yml`
- Logs: `/lichess-bot/logs/`

## Common Issues & Solutions

### Issue: Version mismatch after deployment
```bash
# Verify files extracted correctly
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker exec v7p3r-production ls -la /lichess-bot/engines/v7p3r/"

# Check version in extracted files
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker exec v7p3r-production grep 'id name' /lichess-bot/engines/v7p3r/v7p3r_uci.py"
```

### Issue: Bot not responding
```bash
# Check container status
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker ps"

# Restart container
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker restart v7p3r-production"

# Check logs for errors
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker logs v7p3r-production --tail 100"
```

### Issue: High blunder rate after deployment
```bash
# 1. Check if regression tests were run
cat deployment_log.json | grep -A 10 "\"version\": \"v17.8.0\""

# 2. Compare performance vs baseline
# Run local tournament: v17.8 vs v17.7 (50+ games)

# 3. If blunders >8/game after 20 games → ROLLBACK
# Use rollback command from above
```

## Monitoring Checklist

### First 24 Hours After Deployment
- [ ] Watch first 5 games manually on Lichess
- [ ] Check logs every 2 hours for errors
- [ ] Verify move times are reasonable (no timeouts)
- [ ] Monitor blunder count (should be ≤6/game)
- [ ] Check time forfeit rate (should be <10%)

### First 48 Hours
- [ ] Compare win rate vs baseline (expect ±5% variance)
- [ ] Verify draw rate stable (±5% of baseline)
- [ ] Confirm blunders/game meets acceptance criteria
- [ ] Check CPL consistency with testing

### First Week
- [ ] Monitor ELO stability (±50 of expected)
- [ ] Verify zero crashes
- [ ] Minimum 50 games for statistical significance
- [ ] Check Lichess profile for user feedback

### If Any Red Flags → Initiate Rollback

## Documentation Files

- **CHANGELOG.md**: Version history with changes, rationale, testing
- **deployment_log.json**: Machine-readable deployment history
- **.github/instructions/version_management.instructions.md**: Full version management workflow
- **docs/TESTING_GUIDE.md**: Testing procedures and regression prevention
- **lichess/CHANGELOG.md**: Lichess-specific deployment history

## Rollback Targets

- **Primary**: v17.7.0 (current stable, 4+ days deployment)
- **Secondary**: v17.1.0 (proven stable, deployed twice)
- **Emergency**: v14.1.0 (25-day stable deployment)

## Acceptance Criteria

ALL must pass before production deployment:
- ✅ Regression tests: 100% pass
- ✅ Win rate: ≥48% (vs equal baseline)
- ✅ Blunders/game: ≤6.0
- ✅ Time forfeit rate: <10%
- ✅ CPL: <150
- ✅ CHANGELOG.md updated
- ✅ deployment_log.json updated
- ✅ Git tag created
- ✅ User validation completed
