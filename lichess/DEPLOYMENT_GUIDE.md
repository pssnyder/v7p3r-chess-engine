# V7P3R Lichess Bot - Production Deployment Guide

**Last Updated**: 2026-05-03  
**Current Production Version**: v18.3  
**Environment**: GCP (v7p3r-lichess-bot project)

---

## Table of Contents
1. [Quick Reference](#quick-reference)
2. [Prerequisites](#prerequisites)
3. [Deployment Procedure](#deployment-procedure)
4. [Rollback Procedure](#rollback-procedure)
5. [Troubleshooting](#troubleshooting)
6. [Known Issues & Solutions](#known-issues--solutions)

---

## Quick Reference

### Essential Commands

```bash
# Check bot status
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker logs v7p3r-production --tail 30"

# View engine files
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker exec v7p3r-production ls -la /lichess-bot/engines/v7p3r/"

# Check engine version
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker exec v7p3r-production head -5 /lichess-bot/engines/v7p3r/v7p3r.py"
```

### GCP Environment
- **Project**: v7p3r-lichess-bot
- **VM Instance**: v7p3r-production-bot
- **Zone**: us-central1-a
- **Instance Type**: e2-micro (1GB RAM, 2 vCPUs)
- **Container**: v7p3r-production (lichess-bot framework 2025.10.1.2)
- **Cost**: ~$24/month

### File Paths
- **VM Staging Directory**: `/home/v7p3r/engines/v7p3r/`
- **Container Engine Path**: `/lichess-bot/engines/v7p3r/` (internal filesystem, NOT mounted)
- **Config File**: `/home/v7p3r/config.yml` (mounted read-only)
- **Game Records**: `/home/v7p3r/game_records/` (mounted)
- **Logs**: `/home/v7p3r/logs/` (mounted)

---

## Prerequisites

### Local Setup
1. **gcloud CLI** installed and authenticated:
   ```bash
   gcloud auth login
   gcloud config set project v7p3r-lichess-bot
   ```

2. **Engine Files** prepared in local deployment directory:
   ```
   lichess/engines/V7P3R_vXX.X_YYYYMMDD/src/
   ├── v7p3r.py
   ├── v7p3r_uci.py
   ├── v7p3r_bitboard_evaluator.py
   ├── v7p3r_eval_modules.py
   ├── v7p3r_eval_selector.py
   ├── v7p3r_fast_evaluator.py
   ├── v7p3r_modular_eval.py
   ├── v7p3r_move_safety.py
   ├── v7p3r_openings_v161.py
   └── v7p3r_position_context.py
   ```

3. **Deployment Package** created:
   ```bash
   cd "lichess/engines/V7P3R_vXX.X_YYYYMMDD"
   # Create ZIP (easier for Windows PowerShell)
   Compress-Archive -Path src\* -DestinationPath v18.3-src.zip
   ```

### Pre-Deployment Checklist
- [ ] Version number updated in both `v7p3r.py` and `v7p3r_uci.py`
- [ ] CHANGELOG.md updated with deployment entry
- [ ] deployment_log.json updated with new version
- [ ] Engine tested locally via UCI protocol
- [ ] Regression tests passed (if available)
- [ ] Backup plan documented

---

## Deployment Procedure

### Step 1: Upload Engine Files to VM

```bash
# Upload ZIP to VM staging area
gcloud compute scp "lichess/engines/V7P3R_vXX.X_YYYYMMDD/v18.3-src.zip" \
  v7p3r-production-bot:/tmp/v18.3-src.zip \
  --zone=us-central1-a \
  --project=v7p3r-lichess-bot

# Verify upload
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="ls -lh /tmp/v18.3-src.zip"
```

**Expected Output**: File size should match local ZIP (e.g., ~185KB)

---

### Step 2: Extract to VM Staging Directory

```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="cd /tmp ; \
    sudo python3 -m zipfile -e v18.3-src.zip v18.3_extracted ; \
    sudo rm -rf /home/v7p3r/engines/v7p3r/* ; \
    sudo cp -r v18.3_extracted/* /home/v7p3r/engines/v7p3r/ ; \
    sudo chown -R v7p3r:v7p3r /home/v7p3r/engines/v7p3r/ ; \
    sudo chmod +x /home/v7p3r/engines/v7p3r/v7p3r_uci.py ; \
    sudo ls -lh /home/v7p3r/engines/v7p3r/ ; \
    sudo head -5 /home/v7p3r/engines/v7p3r/v7p3r.py"
```

**Expected Output**: 
- 10 Python files listed
- Version header in `v7p3r.py` shows correct version (e.g., "V7P3R Chess Engine v18.3.0")

---

### Step 3: Create Wrapper Script (Required for Execute Permissions)

**CRITICAL**: Mounted volumes have execute restrictions in Docker. We must use a wrapper script.

```bash
# Create wrapper script locally
cat > run_v7p3r.sh << 'EOF'
#!/bin/bash
cd /lichess-bot/engines/v7p3r
exec python3 v7p3r_uci.py
EOF

# Upload wrapper to VM
gcloud compute scp run_v7p3r.sh v7p3r-production-bot:/tmp/run_v7p3r.sh \
  --zone=us-central1-a \
  --project=v7p3r-lichess-bot

# Deploy wrapper to staging directory
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo cp /tmp/run_v7p3r.sh /home/v7p3r/engines/v7p3r/run_v7p3r.sh ; \
    sudo chmod +x /home/v7p3r/engines/v7p3r/run_v7p3r.sh ; \
    sudo chown v7p3r:v7p3r /home/v7p3r/engines/v7p3r/run_v7p3r.sh ; \
    sudo cat /home/v7p3r/engines/v7p3r/run_v7p3r.sh"
```

---

### Step 4: Update config.yml (If Needed)

```bash
# Download current config
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo cp /home/v7p3r/config.yml /tmp/config.yml ; sudo chmod 644 /tmp/config.yml"

gcloud compute scp v7p3r-production-bot:/tmp/config.yml lichess/config.yml.backup \
  --zone=us-central1-a \
  --project=v7p3r-lichess-bot
```

**Edit config.yml locally** (if engine name changed):
```yaml
engine:
  dir: "./engines/v7p3r/"  # V7P3R v18 engine directory
  name: "run_v7p3r.sh"     # Wrapper script (NOT v7p3r_uci.py)
  protocol: "uci"
```

**Upload updated config** (only if changes needed):
```bash
gcloud compute scp lichess/config.yml.backup v7p3r-production-bot:/tmp/config.yml \
  --zone=us-central1-a \
  --project=v7p3r-lichess-bot

gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo cp /tmp/config.yml /home/v7p3r/config.yml ; \
    sudo chown v7p3r:v7p3r /home/v7p3r/config.yml"
```

---

### Step 5: Stop Container and Copy Engine Files

**CRITICAL**: Engine files must be copied into container's **internal filesystem** (NOT mounted volume).

```bash
# Stop container
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker stop v7p3r-production"

# Copy engine files from VM staging to container
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker exec v7p3r-production mkdir -p /lichess-bot/engines/v7p3r ; \
    sudo docker cp /home/v7p3r/engines/v7p3r/. v7p3r-production:/lichess-bot/engines/v7p3r/ ; \
    sudo docker exec v7p3r-production chmod +x /lichess-bot/engines/v7p3r/run_v7p3r.sh /lichess-bot/engines/v7p3r/v7p3r_uci.py"
```

**Why copy instead of mount?**
- Mounted volumes have `noexec` flag in Docker (security restriction)
- Scripts on mounted filesystems cannot be executed directly
- Copying into container's internal filesystem bypasses this restriction

---

### Step 6: Verify Files in Container

```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker exec v7p3r-production ls -lh /lichess-bot/engines/v7p3r/ ; \
    sudo docker exec v7p3r-production head -5 /lichess-bot/engines/v7p3r/v7p3r.py"
```

**Expected Output**:
- All 10 source files present with correct dates
- `run_v7p3r.sh` has execute permissions (`-rwxr-xr-x`)
- Version header shows correct version

---

### Step 7: Start Container and Monitor

```bash
# Start container
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker start v7p3r-production ; sleep 10 ; sudo docker logs v7p3r-production --tail 80"
```

**Expected Output**:
```
INFO     Engine configuration OK
INFO     Welcome v7p3r_bot!
INFO     You're now connected to https://lichess.org/ and awaiting challenges.
```

**Error States**:
- ❌ "engine file does not exist" → Files not copied to container
- ❌ "doesn't have execute permission" → Wrapper script not executable
- ❌ "ModuleNotFoundError" → Missing dependency file

---

### Step 8: Test Engine via UCI

```bash
# Test engine responds to UCI protocol
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker exec v7p3r-production bash -c 'cd /lichess-bot && echo uci | ./engines/v7p3r/run_v7p3r.sh | head -20'"
```

**Expected Output**:
```
info string Using Fast Evaluator (v16.1 speed)
info string Modular evaluation: DISABLED (parallel testing)
info string Opening book loaded (v16.1 repertoire)
id name V7P3R v18.3
id author pssnyder
uciok
```

---

### Step 9: Post-Deployment Validation

#### Monitor Bot Activity
```bash
# Watch logs in real-time
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker logs v7p3r-production -f"
```

#### Check First Games
1. Visit **https://lichess.org/@/v7p3r_bot** to monitor live games
2. Verify move times are reasonable (not timing out)
3. Watch for blunders or unusual play
4. Check game results and ELO changes

#### Update Deployment Logs
1. **lichess/CHANGELOG.md**: Add deployment timestamp and notes
2. **deployment_log.json**: Update status to "active" with deployment date

---

## Rollback Procedure

### Emergency Rollback (Container Has Backup)

If you created a `.backup` directory during deployment:

```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker stop v7p3r-production ; \
    sudo docker exec v7p3r-production bash -c 'rm -rf /lichess-bot/engines/v7p3r && mv /lichess-bot/engines/v7p3r.backup /lichess-bot/engines/v7p3r' ; \
    sudo docker start v7p3r-production ; \
    sleep 10 ; \
    sudo docker logs v7p3r-production --tail 50"
```

### Full Rollback (Redeploy Previous Version)

1. Follow **Deployment Procedure** with previous version's ZIP file
2. Example: Rollback from v18.4 to v18.3:
   ```bash
   # Upload v18.3 ZIP and follow Steps 1-9
   gcloud compute scp "lichess/engines/V7P3R_v18.3_20251229/v18.3-src.zip" ...
   ```

3. Update deployment logs:
   - Mark failed version as `"rolled_back": true` in deployment_log.json
   - Document rollback reason in CHANGELOG.md

---

## Troubleshooting

### Problem: "engine file does not exist"

**Cause**: Files not copied into container's internal filesystem

**Solution**:
```bash
# Verify files are in VM staging area
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo ls -lh /home/v7p3r/engines/v7p3r/"

# Copy files to container (Step 5)
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker cp /home/v7p3r/engines/v7p3r/. v7p3r-production:/lichess-bot/engines/v7p3r/"
```

---

### Problem: "doesn't have execute (x) permission"

**Cause**: Wrapper script or UCI script lacks execute permissions

**Solution**:
```bash
# Add execute permissions inside container
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker exec v7p3r-production chmod +x /lichess-bot/engines/v7p3r/run_v7p3r.sh /lichess-bot/engines/v7p3r/v7p3r_uci.py"
```

---

### Problem: "Permission denied" when executing script

**Cause**: Trying to execute script from mounted volume (has `noexec` flag)

**Solution**: Files MUST be in container's internal filesystem, not mounted volume
```bash
# DO NOT mount /engines directory
# CORRECT: Copy files into container with docker cp (Step 5)
sudo docker cp /home/v7p3r/engines/v7p3r/. v7p3r-production:/lichess-bot/engines/v7p3r/
```

---

### Problem: Container restart loop

**Cause**: Config validation failing (missing engine file)

**Solution**:
```bash
# Stop container to break restart loop
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker stop v7p3r-production"

# Check logs to identify issue
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker logs v7p3r-production --tail 100"

# Fix issue (missing files, permissions, etc.) then start
```

---

### Problem: Bot offline but container running

**Cause**: Lichess authentication failure or network issue

**Solution**:
```bash
# Check logs for auth errors
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker logs v7p3r-production | grep -i 'error\|token\|auth'"

# Verify config has correct token
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo cat /home/v7p3r/config.yml | grep token"
```

---

## Known Issues & Solutions

### Issue: Tarball Corruption (29 bytes)

**Symptom**: Uploaded tarball only 29 bytes instead of ~185KB

**Cause**: PowerShell path quoting issues

**Solution**: Use ZIP format instead of TAR.GZ for Windows deployments:
```bash
# Create ZIP (Windows-friendly)
Compress-Archive -Path src\* -DestinationPath v18.3-src.zip

# Extract on Linux VM with Python
sudo python3 -m zipfile -e v18.3-src.zip v18.3_extracted
```

---

### Issue: Version Header Mismatch

**Symptom**: `v7p3r.py` header shows different version than expected (e.g., v14.0 comment but v18.3 functionality)

**Cause**: Version comment not updated in code

**Diagnosis**: Check file dates and functionality, not just header comment
```bash
# File dates are authoritative
ls -lh /lichess-bot/engines/v7p3r/v7p3r.py

# Test engine functionality
echo uci | ./engines/v7p3r/run_v7p3r.sh
```

**Solution**: Update header comment in future versions, but file dates/functionality take precedence

---

## Best Practices

1. **Always Test Locally First**
   - Run engine via UCI protocol
   - Test in Arena GUI or Cute Chess
   - Verify version string

2. **Use ZIP for Deployment Packages**
   - More reliable than TAR.GZ on Windows
   - Built-in Python extraction on Linux VMs

3. **Never Mount /engines Directory**
   - Docker `noexec` flag prevents script execution
   - Always copy files into container's internal filesystem

4. **Create Wrapper Scripts**
   - Use `run_v7p3r.sh` to execute Python engine
   - Ensures proper working directory and Python invocation

5. **Monitor First 10-20 Games**
   - Check for time management issues
   - Watch for tactical blunders
   - Verify ELO stability

6. **Document Everything**
   - Update CHANGELOG.md immediately
   - Record deployment timestamp
   - Note any issues encountered

---

## Quick Commands Reference

### Deployment
```bash
# Upload files
gcloud compute scp v18.3-src.zip v7p3r-production-bot:/tmp/ --zone=us-central1-a --project=v7p3r-lichess-bot

# Extract and deploy
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="cd /tmp ; sudo python3 -m zipfile -e v18.3-src.zip v18.3_extracted ; sudo cp -r v18.3_extracted/* /home/v7p3r/engines/v7p3r/ ; sudo chown -R v7p3r:v7p3r /home/v7p3r/engines/v7p3r/"

# Copy to container
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker stop v7p3r-production ; sudo docker cp /home/v7p3r/engines/v7p3r/. v7p3r-production:/lichess-bot/engines/v7p3r/ ; sudo docker exec v7p3r-production chmod +x /lichess-bot/engines/v7p3r/run_v7p3r.sh"

# Start and verify
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker start v7p3r-production ; sleep 10 ; sudo docker logs v7p3r-production --tail 50"
```

### Monitoring
```bash
# Real-time logs
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker logs v7p3r-production -f"

# Check engine version
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker exec v7p3r-production head -5 /lichess-bot/engines/v7p3r/v7p3r.py"
```

---

**Last Deployment**: v18.3 (2026-05-03) - Successful rollback from v18.4  
**Next Review**: After 24-48 hours of gameplay monitoring
