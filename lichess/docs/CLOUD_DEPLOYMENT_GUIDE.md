# V7P3R Lichess Bot: Complete Cloud Deployment Guide

## 🚀 Executive Summary

This guide documents the complete cloud deployment architecture for the V7P3R chess engine on Google Cloud Platform. The system uses Python-based UCI implementation, Docker containerization, and volume-mounted configuration for production deployment on GCP Compute Engine.

**Current Production Status (as of Nov 20, 2025):**
- **Engine Version**: V7P3R v16.2 (Bug Fix Release - Depth + Illegal Move Fixes)
- **Instance Type**: e2-micro (GCP Compute Engine, Container-Optimized OS)
- **Deployment Method**: Manual container update (fastest for small engine changes)
- **Location**: us-central1-a zone

## ☁️ Why Cloud + Python is the Perfect Combination

### Python Benefits Amplified in Cloud:
- **No .exe compatibility issues** in Linux containers
- **Native Docker support** - Python is first-class in containerized environments  
- **Better resource efficiency** - crucial for cost optimization in cloud
- **Professional deployment patterns** - follows cloud-native best practices
- **Easier horizontal scaling** if you ever want multiple bots

### Cloud Benefits for Chess Bots:
- **Lower latency** - Cloud datacenters have better connectivity to Lichess servers
- **99.9% uptime** - No more worrying about home internet/power outages
- **Professional infrastructure** - Automatic restarts, monitoring, logging
- **Cost-effective** - e2-micro starts at ~$5-7/month
- **Easy updates** - Deploy new engine versions with zero downtime

## 🎯 GCP Instance Recommendations

### Current Production: **e2-micro** 

**Actual Production Configuration:**
- **Instance**: e2-micro (Container-Optimized OS)
- **Zone**: us-central1-a
- **Instance Name**: v7p3r-production-bot
- **Container Name**: v7p3r-production
- **Monthly Cost**: ~$5-7

| Instance | vCPUs | RAM | Monthly Cost* | V7P3R Production Experience |
|----------|-------|-----|---------------|-------------------|
| e2-micro | 0.25-2 | 1GB | $5-7 | ✅ **Working well** - 61% win rate, 3 concurrent games |
| e2-medium | 1-2 | 4GB | $20-30 | 🔄 Future upgrade option for higher concurrency |

**Production Insights:**
- **e2-micro is sufficient** for V7P3R with 3 concurrent games
- **Burstable CPU** handles tactical calculation spikes well
- **1GB RAM** adequate for Python engine + lichess-bot framework
- **Cost-effective** for 24/7 operation at ~$6/month

*Costs are approximate and vary by region/usage

## 🐳 Docker Deployment Architecture

### **CRITICAL: Volume-Mounted Configuration Pattern**

The production deployment uses a **volume-mounted config file** pattern, not the config baked into the Docker image. This is essential to understand:

```bash
# Production container run command (from manage-production.sh)
docker run -d \
    --name v7p3r-production \
    --restart unless-stopped \
    --log-driver=gcplogs \
    -v /home/v7p3r/config.yml:/lichess-bot/config.yml:ro \  # ← External config mounts OVER internal
    -v /home/v7p3r/game_records:/lichess-bot/game_records \
    -v /home/v7p3r/logs:/lichess-bot/logs \
    v7p3r-bot:production
```

**Key Points:**
1. **External config takes precedence**: `/home/v7p3r/config.yml` on the VM overrides any config in the Docker image
2. **Updates require two steps**: 
   - Rebuild Docker image with new engine version
   - Upload updated config to VM at `/home/v7p3r/config.yml`
3. **Engine path in config**: Must match the path inside the container (`./engines/v7p3r/`), NOT the source path
4. **Read-only mount**: Config is mounted read-only (`:ro`) for security

### Docker Image Structure

The Dockerfile copies engine files to a **normalized path** inside the container:

```dockerfile
# Dockerfile copies V7P3R_v14.0/src/ to ./engines/v7p3r/
COPY engines/V7P3R_v14.0/src/ ./engines/v7p3r/
```

**Therefore, config.yml must reference:**
```yaml
engine:
  dir: "./engines/v7p3r/"  # ← Path INSIDE container, not source path
  name: "v7p3r_uci.py"
```

### Cloud Optimizations:
- **Security**: Non-root user execution
- **Monitoring**: Built-in health checks for GCP monitoring
- **Efficiency**: Minimal base image (python:3.13-slim)
- **Caching**: Optimized layer structure for faster rebuilds
- **Logging**: Cloud-friendly output with gcplogs driver

### V7P3R Integration:
- **Native Python**: No .exe complexity
- **Embedded engine**: V7P3R source code built into container
- **Normalized paths**: Engine always at `./engines/v7p3r/` regardless of source version

## 📋 Engine Update Deployment Process (Tested & Verified)

### Prerequisites
1. **GCP Setup**: Project created with billing enabled
2. **gcloud CLI**: Installed and authenticated (`gcloud auth login`)
3. **VM Access**: SSH access to `v7p3r-production-bot` in zone `us-central1-a`
4. **Engine Files**: New engine version in `engines/V7P3R_vX.X/src/` directory

---

## 🚀 QUICK START: Engine Update in 5 Minutes

**When you just need to update engine files (recommended for most updates):**

### Step 1: Create Engine Tarball
```bash
cd "s:/Maker Stuff/Programming/Chess Engines/Deployed Engines/v7p3r-lichess-engine"
tar -czf v16.2-src.tar.gz -C engines/V7P3R_v16.2 src
```

### Step 2: Upload to VM
```bash
gcloud compute scp v16.2-src.tar.gz v7p3r-production-bot:/home/v7p3r/ --zone=us-central1-a
```

### Step 3: Deploy to Container
```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
    cd /home/v7p3r
    sudo docker cp v16.2-src.tar.gz v7p3r-production:/tmp/
    sudo docker exec v7p3r-production bash -c 'mv /lichess-bot/engines/v7p3r /lichess-bot/engines/v7p3r.backup'
    sudo docker exec v7p3r-production mkdir -p /lichess-bot/engines/v7p3r
    sudo docker exec v7p3r-production bash -c 'cd /lichess-bot/engines/v7p3r && tar -xzf /tmp/v16.2-src.tar.gz --strip-components=1'
    sudo docker exec v7p3r-production chmod +x /lichess-bot/engines/v7p3r/*.py
    sudo docker exec v7p3r-production rm /tmp/v16.2-src.tar.gz
    rm v16.2-src.tar.gz
    sudo docker restart v7p3r-production
"
```

### Step 4: Verify Deployment
```bash
# Wait 10 seconds for startup
sleep 10

# Check container status
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker ps --filter name=v7p3r-production"

# Check logs for successful startup
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker logs v7p3r-production --tail=30"
```

**Success indicators:**
- ✅ Container shows "Up X seconds" in status
- ✅ Logs show "Engine configuration OK"
- ✅ Logs show "You're now connected to https://lichess.org/"
- ✅ No error messages about missing directories

**Total time: ~5 minutes** (vs 30-45 minutes for full Docker rebuild)

---

## 🐳 FULL REBUILD: When Docker Image Needs Updates

**Use this method when:**
- Updating lichess-bot framework version
- Changing Python dependencies
- Modifying Dockerfile configuration
- First-time deployment

### Prerequisites for Full Rebuild
- **Docker Desktop**: Must be running (cannot build without it)
- **Updated Dockerfile**: `docker/Dockerfile.v7p3r-local` with new engine version
- **manage-production.sh**: Deployment automation script

### Full Rebuild Workflow

**Option A: Use Automation Script (Recommended)**

```bash
./manage-production.sh update
```

This script automates:
1. Building Docker image locally with latest engine version
2. Saving and compressing image to `v7p3r-update.tar.gz`
3. Transferring to GCP VM via `gcloud compute scp`
4. Loading image on VM
5. Stopping old container
6. Starting new container with volume mounts

**Option B: Manual Full Rebuild Steps**

#### 1. Update Engine Version in Dockerfile

Edit `docker/Dockerfile.v7p3r-local` and change the COPY line:

```dockerfile
# Change from:
COPY engines/V7P3R_v16.1/src/ ./engines/v7p3r/

# To (replace with your version):
COPY engines/V7P3R_v16.2/src/ ./engines/v7p3r/
```

**CRITICAL**: Engine is **always** copied to `./engines/v7p3r/` inside container (normalized path)

#### 2. Update Greeting in Config (Optional)

If you want to show version in bot greeting, update `config-docker-cloud.yml`:

```yaml
greeting:
  hello: "Hello! I'm v7p3r_bot powered by V7P3R v16.2 running on Google Cloud. Good luck!"
```

**DO NOT change the engine dir path** - it must remain:
```yaml
engine:
  dir: "./engines/v7p3r/"  # ← ALWAYS this path, never version-specific
  name: "v7p3r_uci.py"
```

#### 3. Build Docker Image

**REQUIREMENT**: Docker Desktop must be running

```bash
cd "s:/Maker Stuff/Programming/Chess Engines/Deployed Engines/v7p3r-lichess-engine"
docker build -f docker/Dockerfile.v7p3r-local -t v7p3r-bot:production .
```

#### 4. Transfer to VM

```bash
# Save and compress image
docker save v7p3r-bot:production | gzip > v7p3r-update.tar.gz

# Upload to VM (takes 5-10 minutes depending on connection)
gcloud compute scp v7p3r-update.tar.gz v7p3r-production-bot:/home/v7p3r/ --zone=us-central1-a
```

#### 5. Deploy on VM

```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
    cd /home/v7p3r
    echo 'Loading new image...'
    gunzip -c v7p3r-update.tar.gz | sudo docker load
    echo 'Stopping old container...'
    sudo docker stop v7p3r-production
    sudo docker rm v7p3r-production
    echo 'Starting new container...'
    sudo docker run -d \
        --name v7p3r-production \
        --restart unless-stopped \
        --log-driver=gcplogs \
        -v /home/v7p3r/config.yml:/lichess-bot/config.yml:ro \
        -v /home/v7p3r/game_records:/lichess-bot/game_records \
        -v /home/v7p3r/logs:/lichess-bot/logs \
        v7p3r-bot:production
    echo 'Cleaning up...'
    rm v7p3r-update.tar.gz
    echo 'Deployment complete!'
"
```

#### 6. Update Config on VM (If Changed)

If you updated the greeting or other config settings:

```bash
gcloud compute scp config-docker-cloud.yml v7p3r-production-bot:/home/v7p3r/config.yml --zone=us-central1-a
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker restart v7p3r-production"
```

#### 7. Verify Deployment

```bash
# Wait for startup
sleep 10

# Check container status
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker ps --filter name=v7p3r-production"

# Check logs for successful startup
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker logs v7p3r-production --tail=30"

# Verify engine files
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker exec v7p3r-production ls -lh /lichess-bot/engines/v7p3r/"

# Test engine responds to UCI
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker exec v7p3r-production python3 /lichess-bot/engines/v7p3r/v7p3r_uci.py < <(echo 'uci')"
```

## 💰 Actual Production Cost Analysis (Nov 2025)

### Monthly Costs (e2-micro):
- **e2-micro instance**: ~$6-7/month (24/7 operation)
- **20GB persistent disk**: ~$0.80/month  
- **Network egress**: ~$0.50-1/month (Lichess API calls minimal)
- **Total**: **~$7-9/month**

### Performance vs Cost:
- **61.5% win rate** in first 24 hours
- **26 games** played with matchmaking
- **Zero downtime** since deployment
- **$0.30/day** for professional infrastructure

### Cost Optimization (Proven):
- **e2-micro sufficient**: No need for e2-medium upgrade yet
- **Sustained use discounts**: Automatic ~30% savings after 1 month
- **Preemptible NOT recommended**: Bot requires 24/7 uptime for rating stability

### Value Comparison:
- **Home deployment**: $0/month but unreliable, higher latency, power costs
- **VPS alternatives**: $10-20/month but less reliable than GCP
- **Current production**: $7-9/month with enterprise reliability

## 🔧 Common Issues & Solutions (Production-Tested)

### Issue 1: Container Restart Loop - "engine directory not found"

**Symptom**: Container keeps restarting, logs show:
```
Exception: Your engine directory './engines/V7P3R_v16.2/src/' is not a directory
```

**Cause**: Config file on VM references source path instead of container's normalized path

**Root Cause**: The config at `/home/v7p3r/config.yml` has wrong engine directory path

**Solution**:
```bash
# 1. Fix config-docker-cloud.yml locally
# Change from: dir: "./engines/V7P3R_v16.2/src/"
# To:          dir: "./engines/v7p3r/"

# 2. Upload corrected config
gcloud compute scp config-docker-cloud.yml v7p3r-production-bot:/home/v7p3r/config.yml --zone=us-central1-a

# 3. Recreate container to force config reload
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
    sudo docker stop v7p3r-production
    sudo docker rm v7p3r-production
    sudo docker run -d \
        --name v7p3r-production \
        --restart unless-stopped \
        --log-driver=gcplogs \
        -v /home/v7p3r/config.yml:/lichess-bot/config.yml:ro \
        -v /home/v7p3r/game_records:/lichess-bot/game_records \
        -v /home/v7p3r/logs:/lichess-bot/logs \
        v7p3r-bot:production
"
```

**Prevention**: Always use `./engines/v7p3r/` in config, never version-specific paths

### Issue 2: Config Not Updating After Deployment

**Symptom**: Deployed new Docker image but bot still shows old greeting or behavior

**Cause**: Volume-mounted config at `/home/v7p3r/config.yml` overrides config in Docker image

**Solution**:
```bash
# Update config on VM (not just in Docker image)
gcloud compute scp config-docker-cloud.yml v7p3r-production-bot:/home/v7p3r/config.yml --zone=us-central1-a
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker restart v7p3r-production"
```

**Remember**: External config file takes precedence - always update both image AND VM file

### Issue 3: Docker Desktop Not Running

**Symptom**: `docker build` fails with "cannot connect to Docker daemon" or pipe error

**Cause**: Docker Desktop application is not running on local machine

**Solution**:
```bash
# Option A: Start Docker Desktop and retry full rebuild
# Windows: Start Docker Desktop from Start Menu
# Then: docker build -f docker/Dockerfile.v7p3r-local -t v7p3r-bot:production .

# Option B: Use manual container update instead (no Docker Desktop needed)
# See "QUICK START: Engine Update in 5 Minutes" section above
```

### Issue 4: Permission Errors on .py Files

**Symptom**: Engine files extracted but container shows "permission denied" when running engine

**Cause**: Files extracted without execute permissions

**Solution**:
```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
    sudo docker exec v7p3r-production chmod +x /lichess-bot/engines/v7p3r/*.py
    sudo docker restart v7p3r-production
"
```

### Issue 5: Tarball Extraction Path Issues

**Symptom**: Files end up in wrong directory or nested subdirectories

**Cause**: Incorrect `--strip-components` value or wrong base directory

**Solution**:
```bash
# Verify tarball structure first
tar -tzf v16.2-src.tar.gz | head -10

# Should show: src/v7p3r.py, src/v7p3r_uci.py, etc.
# Use --strip-components=1 to remove "src/" prefix

# If tarball has different structure, adjust accordingly
# Example: engines/V7P3R_v16.2/src/v7p3r.py would need --strip-components=3
```

### Issue 6: Google Cloud Logging Permission Denied

**Symptom**: Container fails with `logging.logEntries.create permission denied`

**Cause**: VM service account lacks Cloud Logging write permissions

**Solution**:
```bash
# Get VM service account
gcloud compute instances describe v7p3r-production-bot --zone=us-central1-a --format="value(serviceAccounts[0].email)"

# Grant logging permission (replace PROJECT_ID and SERVICE_ACCOUNT_EMAIL)
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
    --role="roles/logging.logWriter"

# Restart container
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker restart v7p3r-production"
```

## 🎯 Migration Strategy

### Week 1: Preparation
- [x] Switch to Python locally (completed)
- [x] Test Docker container locally (completed)
- [x] Set up GCP project and billing (completed)

### Week 2: Deployment
- [x] Deploy to GCP e2-micro (completed - v12.6 initial)
- [x] Run parallel testing (completed)
- [x] Monitor performance and costs (completed)

### Week 3: Optimization & V14.0 Upgrade
- [x] Fine-tune cloud configuration (completed)
- [x] Set up monitoring and alerting (completed)
- [x] Deploy V7P3R v14.0 engine (completed Oct 31, 2025)
- [x] Verify improved performance (+70 Elo in testing)

### Current Status (Nov 2025): ✅ **PRODUCTION READY**
- V7P3R v14.0 running successfully on e2-micro
- 61.5% win rate, 26 games in first 24 hours
- Matchmaking active (85% acceptance rate)
- Cost: ~$7-9/month

## 🏆 Production Results Summary

### V16.2 Deployment (Nov 20, 2025)
- **Deployment Method**: Manual container update (5 minutes)
- **Critical Fixes**:
  - Depth bug fixed (was stuck at 2-3, now reaches 5-10)
  - "No move found" bug fixed (explicit checkmate/stalemate detection)
  - Opening book active (52 positions, 15 moves deep)
- **Deployment Success**: ✅ Container running, bot online, awaiting challenges

### V14.0 Performance Baseline (Nov 2025)
- **Win Rate**: 61.5% (16 wins, 6 losses, 4 draws in first 24 hours)
- **Games Played**: 26 total games
- **Engine Speed**: ~1-2 seconds per move (1.0-2.4 Knps)
- **Search Depth**: 3-5 plies (limited by depth bug - now fixed in v16.2)
- **Matchmaking Success**: 85% (28 accepted, 5 declined)
- **System Uptime**: 99.9% (zero downtime incidents)

### Cloud Deployment Benefits:
1. **5-minute updates** with manual container method (vs 30-45 min full rebuild)
2. **No Docker Desktop required** for engine-only updates
3. **Professional monitoring** with Cloud Logging
4. **99.9% uptime** vs home internet reliability
5. **Cost-effective** at ~$7-9/month for 24/7 operation
6. **Easy rollback** with backup directory preserved

## 🚀 Conclusion & Critical Lessons Learned

The cloud deployment system for V7P3R has evolved into a **production-grade, easy-to-update platform**. Key achievements:

1. **5-minute engine updates** (manual container method)
2. **No local Docker required** for most updates
3. **99.9% uptime** with professional infrastructure
4. **Cost-effective** at ~$7-9/month for 24/7 operation

### Critical Deployment Lessons (MUST REMEMBER):

#### 1. **Config Path is Always `./engines/v7p3r/`**
- ❌ WRONG: `dir: "./engines/V7P3R_v16.2/src/"` (version-specific)
- ✅ CORRECT: `dir: "./engines/v7p3r/"` (normalized container path)
- Container restart loops if this is wrong

#### 2. **Volume-Mounted Config Overrides Everything**
- External config at `/home/v7p3r/config.yml` takes precedence over Docker image config
- Must update both Docker image AND VM config file for changes to take effect
- Restart container after config updates: `sudo docker restart v7p3r-production`

#### 3. **Two Deployment Methods - Choose Wisely**

| Method | Time | Docker Desktop Required? | Use When |
|--------|------|-------------------------|----------|
| Manual Container Update | 5 min | ❌ No | Engine files only changed |
| Full Docker Rebuild | 30-45 min | ✅ Yes | Framework/dependencies changed |

**Recommendation**: Use manual container update for 90% of engine versions

#### 4. **Always Backup Old Engine**
```bash
sudo docker exec v7p3r-production bash -c 'mv /lichess-bot/engines/v7p3r /lichess-bot/engines/v7p3r.backup'
```
Allows instant rollback if new version has issues

#### 5. **Verify Before Celebrating**
Don't assume deployment worked - always check:
- Container status: `docker ps` shows "Up X seconds" not "Restarting"
- Logs show: "Engine configuration OK" and "connected to https://lichess.org/"
- No error messages about missing directories or permission denied

### Deployment Workflow Comparison

**For Most Engine Updates (Recommended):**
```
1. Create tarball (30 sec)
2. Upload to VM (1 min)
3. Deploy to container (2 min)
4. Verify (1 min)
Total: ~5 minutes
```

**For Framework/Dependency Updates:**
```
1. Update Dockerfile (2 min)
2. Build Docker image (10-15 min)
3. Transfer to VM (5-10 min)
4. Deploy on VM (5 min)
5. Update config if needed (2 min)
6. Verify (2 min)
Total: ~30-45 minutes
```

### Quick Reference Commands

```bash
# ============================================
# QUICK ENGINE UPDATE (5 minutes)
# ============================================
cd "s:/Maker Stuff/Programming/Chess Engines/Deployed Engines/v7p3r-lichess-engine"
tar -czf v16.2-src.tar.gz -C engines/V7P3R_v16.2 src
gcloud compute scp v16.2-src.tar.gz v7p3r-production-bot:/home/v7p3r/ --zone=us-central1-a
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
    cd /home/v7p3r
    sudo docker cp v16.2-src.tar.gz v7p3r-production:/tmp/
    sudo docker exec v7p3r-production bash -c 'mv /lichess-bot/engines/v7p3r /lichess-bot/engines/v7p3r.backup'
    sudo docker exec v7p3r-production mkdir -p /lichess-bot/engines/v7p3r
    sudo docker exec v7p3r-production bash -c 'cd /lichess-bot/engines/v7p3r && tar -xzf /tmp/v16.2-src.tar.gz --strip-components=1'
    sudo docker exec v7p3r-production chmod +x /lichess-bot/engines/v7p3r/*.py
    sudo docker exec v7p3r-production rm /tmp/v16.2-src.tar.gz
    rm v16.2-src.tar.gz
    sudo docker restart v7p3r-production
"

# ============================================
# MONITORING & VERIFICATION
# ============================================

# Check container status
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker ps --filter name=v7p3r-production"

# View recent logs
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker logs v7p3r-production --tail=50"

# Check engine files
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker exec v7p3r-production ls -lh /lichess-bot/engines/v7p3r/"

# Verify config path
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker exec v7p3r-production grep 'dir:' /lichess-bot/config.yml"

# ============================================
# EMERGENCY ROLLBACK
# ============================================
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
    sudo docker exec v7p3r-production bash -c 'rm -rf /lichess-bot/engines/v7p3r'
    sudo docker exec v7p3r-production bash -c 'mv /lichess-bot/engines/v7p3r.backup /lichess-bot/engines/v7p3r'
    sudo docker restart v7p3r-production
"

# ============================================
# CONFIG-ONLY UPDATE
# ============================================
gcloud compute scp config-docker-cloud.yml v7p3r-production-bot:/home/v7p3r/config.yml --zone=us-central1-a
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker restart v7p3r-production"

# ============================================
# GAME STATISTICS
# ============================================
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
    echo '=== Game Statistics ==='
    echo -n 'Wins: ' && sudo docker logs v7p3r-production 2>&1 | grep -c 'v7p3r_bot won'
    echo -n 'Losses: ' && sudo docker logs v7p3r-production 2>&1 | grep -c 'v7p3r_bot resigned'
    echo -n 'Draws: ' && sudo docker logs v7p3r-production 2>&1 | grep -c 'Game ended in a draw'
"
```

---

## 📚 Additional Resources

- **GCP Documentation**: https://cloud.google.com/compute/docs
- **lichess-bot Framework**: https://github.com/lichess-bot-devs/lichess-bot
- **Docker Best Practices**: https://docs.docker.com/develop/dev-best-practices/

**V7P3R is now a professional cloud-deployed chess bot with fast, reliable updates!** 🎉