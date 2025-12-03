# V7P3R Analytics System - Production Deployment Plan

**Goal**: Live automated weekly analytics in GCP by Sunday, Dec 8, 2025  
**Current Date**: Wednesday, Dec 3, 2025  
**Target**: v17.5 Initial Report by Weekend (Dec 7-8)

---

## System Overview

The V7P3R Analytics System automatically:
1. **Downloads** games from Lichess API (v7p3r_bot)
2. **Maps** games to engine versions using CHANGELOG timeline
3. **Analyzes** with Stockfish (12 parallel workers)
4. **Generates** performance reports (MD + JSON)
5. **Compares** versions (Top1 alignment, CPL, blunders)
6. **Delivers** weekly reports via email

---

## Current Status (Dec 3, 2025)

### ✅ Completed Components

1. **Core Analytics Engine** (v7p3r_analytics.py)
   - Stockfish integration
   - Move-by-move analysis
   - Blunder classification (critical, major, minor)
   - Theme adherence tracking

2. **Parallel Processing** (parallel_analysis.py)
   - 12-worker processing
   - Cloud-optimized (tested: 320 games in 217 min)
   - Per-version splitting

3. **Version Tracking** (version_tracker.py)
   - Timeline: v12.2 → v17.5
   - Automatic game→version mapping
   - **Updated**: v17.5 (Dec 2+) now included

4. **Game Fetching** (fetch_lichess_games.py)
   - Lichess API integration
   - Date range filtering
   - **Tested**: 36 v17.5 games downloaded

5. **Report Generation** (report_generator.py)
   - Markdown reports
   - JSON data exports
   - Version comparison tables

6. **Cloud Infrastructure** (Dockerfile, deploy_gcp.sh)
   - Docker container with Stockfish
   - Cloud Run job configuration
   - Service account setup

### 🔄 In Progress

1. **v17.5 Initial Analysis** (Running Now)
   - 36 games since Dec 2
   - Expected: 5-10 minutes
   - Output: `reports_v17_5_initial/`

### ❌ Remaining Tasks

1. **Email Delivery Integration**
2. **Cloud Scheduler Configuration**
3. **GCP Deployment Execution**
4. **Production Validation**

---

## Weekly Analytics Pipeline

### Automated Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  SUNDAY MIDNIGHT (UTC)                                      │
│  Cloud Scheduler Trigger                                    │
└────────────────────┬────────────────────────────────────────┘
                     ▼
        ┌────────────────────────┐
        │  Cloud Run Job Start   │
        └────────┬───────────────┘
                 ▼
    ┌────────────────────────────────┐
    │  1. Fetch Last 7 Days Games    │
    │     - Lichess API              │
    │     - v7p3r_bot account        │
    │     - Save to /workspace/      │
    └────────┬───────────────────────┘
             ▼
    ┌────────────────────────────────┐
    │  2. Version Mapping            │
    │     - Read CHANGELOG timeline  │
    │     - Match games to versions  │
    │     - Group by version         │
    └────────┬───────────────────────┘
             ▼
    ┌────────────────────────────────┐
    │  3. Parallel Analysis          │
    │     - 12 Stockfish workers     │
    │     - Process all games        │
    │     - Generate per-version     │
    └────────┬───────────────────────┘
             ▼
    ┌────────────────────────────────┐
    │  4. Report Generation          │
    │     - Overall stats            │
    │     - Per-version breakdowns   │
    │     - Version comparisons      │
    │     - Blunder patterns         │
    └────────┬───────────────────────┘
             ▼
    ┌────────────────────────────────┐
    │  5. Email Delivery             │
    │     - Send report              │
    │     - Attach JSON data         │
    │     - Include key metrics      │
    └────────────────────────────────┘
```

### Report Contents

**Weekly Performance Report**:
- **Overall Stats**: Win rate, games played, average CPL
- **Version Breakdown**: Performance by each version deployed
- **Comparison Table**: v17.5 vs baseline (v17.1)
- **Top Improvements**: Moves where performance increased
- **Recurring Blunders**: Positions needing attention
- **Theme Adherence**: Opening/middlegame/endgame breakdown

---

## Deployment Schedule

### Day 1-2: Wednesday-Thursday (Dec 3-4) - LOCAL TESTING
- [x] Update version_tracker.py with v17.5
- [x] Fetch v17.5 games (36 games)
- [ ] Complete v17.5 initial analysis
- [ ] Review results locally
- [ ] Document findings

### Day 3: Friday (Dec 5) - CLOUD PREP
- [ ] Update Dockerfile with latest code
- [ ] Test Docker build locally
- [ ] Configure email delivery (SendGrid/Gmail API)
- [ ] Update deploy_gcp.sh with email settings
- [ ] Test email delivery locally

### Day 4: Saturday (Dec 6) - DEPLOY TO GCP
- [ ] Execute: `bash deploy_gcp.sh`
- [ ] Verify Cloud Run job created
- [ ] Manual test run: `gcloud run jobs execute v7p3r-weekly-analytics --region us-central1`
- [ ] Verify logs and output
- [ ] Confirm email delivery works

### Day 5: Sunday (Dec 7) - PRODUCTION GO-LIVE
- [ ] Monitor first scheduled run (midnight UTC = 7pm EST Sat)
- [ ] Verify email report received
- [ ] Review v17.5 weekend report (Dec 2-7 games)
- [ ] Document any issues
- [ ] Adjust scheduling if needed

---

## V17.5 Report Expectations

### Key Metrics to Track

**Baseline: v17.1 (Nov 30 - Dec 2)**
- Win Rate: ~49.4%
- Critical Blunders: 7.0/game
- Top1 Alignment: ~43-44%
- Average CPL: ~1500-1800

**Target: v17.5 (Dec 2+)**
- Win Rate: ≥50% (goal)
- Critical Blunders: <5.0/game (30% reduction)
- Top1 Alignment: ≥47% (Phase 1 improvement)
- Average CPL: <1400 (endgame speedup benefit)

### Validation Points

1. **Endgame Performance**
   - Pure endgames (≤6 pieces): Should show improvement
   - Mate threat detection: Fewer mate-in-2/3 misses
   - PST pruning: Faster evaluation = better decisions

2. **Castling Optimization**
   - Endgames: No wasted time on castling eval
   - Middlegame: Normal castling logic preserved

3. **Overall Strength**
   - Top1 alignment increase indicates better move selection
   - CPL decrease shows improved position evaluation
   - Blunder reduction validates tactical improvements

---

## Cloud Deployment Details

### GCP Configuration

**Project**: `v7p3r-lichess-bot`  
**Region**: `us-central1`  
**Job Name**: `v7p3r-weekly-analytics`

**Resources**:
- CPU: 2 cores
- Memory: 2 GB
- Timeout: 1 hour (3600s)
- Max Retries: 2

**Schedule**:
- Trigger: Every Sunday at midnight UTC
- Cron: `0 0 * * 0`
- Timezone: UTC

**Estimated Costs**:
- Cloud Run execution: ~$0.10/week (1 hour @ 2 CPU)
- Cloud Storage: ~$0.01/month
- **Total**: ~$0.50/month

### Environment Variables

```bash
PROJECT_ID=v7p3r-lichess-bot
STOCKFISH_PATH=/usr/local/bin/stockfish
WORK_DIR=/workspace
LICHESS_USERNAME=v7p3r_bot
DAYS_BACK=7
WORKERS=12
EMAIL_TO=your-email@example.com
EMAIL_FROM=analytics@v7p3r.com
```

### Service Account Permissions

**v7p3r-analytics@v7p3r-lichess-bot.iam.gserviceaccount.com**:
- `roles/compute.instanceAdmin` - Access to GCP resources
- `roles/storage.objectAdmin` - Save reports to Cloud Storage
- `roles/logging.logWriter` - Write execution logs

---

## Manual Testing Commands

### Local Testing

```bash
# 1. Fetch recent games
cd analytics
python fetch_lichess_games.py --since 2025-12-02 --output current_games/test.pgn

# 2. Run analysis
python parallel_analysis.py current_games/test.pgn --output-dir reports_test --workers 12

# 3. Check results
ls -lh reports_test/
cat reports_test/version_comparison_*.md
```

### Cloud Testing

```bash
# Build and test Docker image locally
cd analytics
docker build -t v7p3r-analytics:test .
docker run --rm v7p3r-analytics:test --days-back 7

# Deploy to GCP
bash deploy_gcp.sh

# Manual execution
gcloud run jobs execute v7p3r-weekly-analytics --region us-central1 --project v7p3r-lichess-bot

# Check logs
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=v7p3r-weekly-analytics" --limit 50 --project v7p3r-lichess-bot
```

---

## Success Criteria

### Week 1 (Dec 2-8)

- [x] v17.5 deployed to production (Dec 2)
- [ ] ≥50 v17.5 games collected by weekend
- [ ] Initial analysis complete and reviewed
- [ ] Cloud Run job deployed and tested
- [ ] First automated weekly report delivered Sunday night

### Week 2 (Dec 9-15)

- [ ] Second weekly report automatically generated
- [ ] v17.5 performance validated over 100+ games
- [ ] Decision made on Phase 2 (parallel mate search)
- [ ] System running autonomously

### Long-term (Ongoing)

- [ ] Weekly reports delivered every Monday morning
- [ ] Version comparisons track engine evolution
- [ ] Blunder patterns inform development priorities
- [ ] Zero manual intervention required

---

## Rollback Plan

If analytics system encounters issues:

1. **Cloud Run Job Fails**
   - Check logs: `gcloud logging read`
   - Review error messages
   - Test Docker image locally
   - Redeploy with fixes

2. **Analysis Takes Too Long**
   - Reduce workers (12 → 8)
   - Increase timeout (1h → 2h)
   - Reduce game limit temporarily

3. **Email Delivery Fails**
   - Verify SendGrid/Gmail API credentials
   - Check Cloud Run env variables
   - Test email locally first
   - Fall back to Cloud Storage only

4. **Data Quality Issues**
   - Verify Lichess API access
   - Check version_tracker timeline
   - Validate Stockfish installation
   - Review game download logs

---

## Next Steps

1. **Monitor Current Analysis** (Running)
   - Wait for 36 v17.5 games to complete (~5-10 min)
   - Review reports in `reports_v17_5_initial/`

2. **Review v17.5 Results**
   - Compare to v17.1 baseline
   - Validate endgame improvements
   - Document findings

3. **Prepare Cloud Deployment**
   - Configure email delivery
   - Test Docker build
   - Update deployment scripts

4. **Deploy to GCP** (Saturday)
   - Execute deployment
   - Test manual run
   - Verify email works

5. **Monitor First Scheduled Run** (Sunday)
   - Receive first automated report
   - Validate accuracy
   - Celebrate success! 🎉

---

**Last Updated**: Dec 3, 2025  
**Status**: Phase 1 (Local Testing) - In Progress  
**Owner**: Pat Snyder
