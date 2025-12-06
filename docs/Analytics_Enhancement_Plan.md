# V7P3R Analytics System - Enhancement Plan
## Historical Persistence & Email Delivery

**Date**: December 4, 2025  
**Status**: Design Phase  
**Goal**: Persistent storage, historical metrics, and automated email delivery

---

## Current State Assessment

### ✅ What Works
- **Pipeline**: Fetch games → Analyze → Generate reports
- **Cloud Deployment**: Docker + Cloud Run job
- **Scheduling**: Every Sunday midnight UTC
- **Processing**: 12 parallel Stockfish workers
- **Local Reports**: JSON + Markdown files generated

### ❌ Current Limitations
1. **Ephemeral Storage**: Reports lost when container completes
2. **No Historical Tracking**: Can't compare week-over-week trends
3. **No Email Delivery**: Reports trapped in container
4. **No Persistence Layer**: Every run starts from scratch

---

## Architecture Changes

### Phase 1: Cloud Storage Integration (This Session)

#### 1.1 Google Cloud Storage Bucket

**Purpose**: Persist reports for historical analysis

**Structure**:
```
v7p3r-analytics-reports/
├── weekly/
│   ├── 2025/
│   │   ├── week_48_2025-11-24/
│   │   │   ├── summary.json
│   │   │   ├── v17_1_analysis.json
│   │   │   ├── v17_1_analysis.md
│   │   │   ├── version_comparison.md
│   │   │   └── pgn/
│   │   │       └── v7p3r_weekly_2025-11-24.pgn
│   │   ├── week_49_2025-12-01/
│   │   │   ├── summary.json
│   │   │   ├── v17_5_analysis.json
│   │   │   ├── v17_5_analysis.md
│   │   │   ├── version_comparison.md
│   │   │   └── pgn/
│   │   │       └── v7p3r_weekly_2025-12-01.pgn
│   │   └── week_50_2025-12-08/
│   │       └── ...
├── historical_summary.json      # Aggregated week-over-week data
└── metadata.json                # Tracking info
```

**Bucket Configuration**:
- Name: `v7p3r-analytics-reports`
- Region: `us-central1` (same as Cloud Run)
- Storage Class: Standard (frequent access)
- Lifecycle: Keep all data (no auto-deletion)
- Cost: ~$0.02/GB/month (~$0.10/month for first year)

#### 1.2 Modified Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│  SUNDAY MIDNIGHT (UTC) - Cloud Scheduler Trigger            │
└────────────────────┬────────────────────────────────────────┘
                     ▼
        ┌────────────────────────┐
        │  Cloud Run Job Start   │
        └────────┬───────────────┘
                 ▼
    ┌────────────────────────────────┐
    │  1. Download Historical Data   │  ← NEW
    │     - Fetch historical_summary │
    │     - Get last 4 weeks data    │
    └────────┬───────────────────────┘
             ▼
    ┌────────────────────────────────┐
    │  2. Fetch Games (Lichess API)  │
    │     - Last 7 days              │
    │     - Save to local /workspace │
    └────────┬───────────────────────┘
             ▼
    ┌────────────────────────────────┐
    │  3. Version Mapping            │
    │     - Match to CHANGELOG       │
    └────────┬───────────────────────┘
             ▼
    ┌────────────────────────────────┐
    │  4. Parallel Analysis          │
    │     - 12 workers × Stockfish   │
    └────────┬───────────────────────┘
             ▼
    ┌────────────────────────────────┐
    │  5. Generate Reports           │
    │     - Current week stats       │
    │     - Week-over-week trends    │  ← ENHANCED
    │     - Historical charts data   │  ← NEW
    └────────┬───────────────────────┘
             ▼
    ┌────────────────────────────────┐
    │  6. Upload to Cloud Storage    │  ← NEW
    │     - Week folder with reports │
    │     - Update historical_summary│
    └────────┬───────────────────────┘
             ▼
    ┌────────────────────────────────┐
    │  7. Generate Email Report      │  ← NEW
    │     - HTML email with trends   │
    │     - Embedded charts/tables   │
    └────────┬───────────────────────┘
             ▼
    ┌────────────────────────────────┐
    │  8. Send Email (SendGrid)      │  ← NEW
    │     - To: patssnyder@gmail.com │
    │     - Attach summary JSON      │
    └────────────────────────────────┘
```

### Phase 2: Email Delivery System

#### 2.1 Email Provider: SendGrid

**Why SendGrid?**
- Free tier: 100 emails/day (need 1/week)
- Google Cloud integration
- HTML email support
- Email API (no SMTP complexity)
- Reliable delivery

**Setup**:
1. Create SendGrid account
2. Verify sender email (patssnyder@gmail.com or custom domain)
3. Generate API key
4. Store in Cloud Run environment variable

#### 2.2 Email Template Structure

**Subject**: `V7P3R Weekly Analytics - Week [XX] - [Date Range]`

**HTML Email Contents**:

```html
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; }
        .header { background: #2c3e50; color: white; padding: 20px; }
        .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px; }
        .improvement { color: green; }
        .regression { color: red; }
        table { border-collapse: collapse; width: 100%; }
        td, th { border: 1px solid #ddd; padding: 8px; text-align: left; }
    </style>
</head>
<body>
    <div class="header">
        <h1>V7P3R Weekly Analytics Report</h1>
        <p>Week 49: Dec 1-8, 2025</p>
        <p>Version: v17.5</p>
    </div>
    
    <h2>📊 Weekly Summary</h2>
    <div class="metric-card">
        <h3>Games Played: 223</h3>
        <ul>
            <li>Win Rate: <strong>51.6%</strong> <span class="improvement">↑ 2.2%</span></li>
            <li>Top1 Alignment: <strong>47.3%</strong> <span class="improvement">↑ 3.3%</span></li>
            <li>Avg CPL: <strong>1,342</strong> <span class="improvement">↓ 156</span></li>
            <li>Critical Blunders: <strong>4.2/game</strong> <span class="improvement">↓ 40%</span></li>
        </ul>
    </div>
    
    <h2>📈 4-Week Trend</h2>
    <table>
        <tr>
            <th>Week</th>
            <th>Version</th>
            <th>Games</th>
            <th>Win Rate</th>
            <th>Top1 %</th>
            <th>Blunders</th>
        </tr>
        <tr>
            <td>Week 46</td>
            <td>v17.1</td>
            <td>187</td>
            <td>49.4%</td>
            <td>44.0%</td>
            <td>7.0</td>
        </tr>
        <tr>
            <td>Week 47</td>
            <td>v17.2</td>
            <td>201</td>
            <td>50.1%</td>
            <td>45.2%</td>
            <td>6.3</td>
        </tr>
        <tr>
            <td>Week 48</td>
            <td>v17.4</td>
            <td>195</td>
            <td>48.7%</td>
            <td>43.8%</td>
            <td>7.5</td>
        </tr>
        <tr style="background-color: #e8f5e9;">
            <td><strong>Week 49</strong></td>
            <td><strong>v17.5</strong></td>
            <td><strong>223</strong></td>
            <td><strong>51.6%</strong></td>
            <td><strong>47.3%</strong></td>
            <td><strong>4.2</strong></td>
        </tr>
    </table>
    
    <h2>🎯 Key Highlights</h2>
    <ul>
        <li><strong>Endgame Improvement</strong>: v17.5 shows 30% faster endgame evaluation</li>
        <li><strong>Fewer Mate Misses</strong>: Critical blunders down 40% from baseline</li>
        <li><strong>Top1 Alignment Up</strong>: Best performance since v17.1</li>
    </ul>
    
    <h2>⚠️ Areas to Monitor</h2>
    <ul>
        <li>Opening diversity: Still playing same lines</li>
        <li>Time management: Some games flagging in bullet</li>
    </ul>
    
    <p>Full reports available in Cloud Storage: <a href="https://console.cloud.google.com/storage/browser/v7p3r-analytics-reports">View Reports</a></p>
    
    <p>-- V7P3R Analytics System</p>
</body>
</html>
```

---

## Implementation Steps

### Step 1: Create Cloud Storage Bucket

```bash
# Create bucket
gsutil mb -p v7p3r-lichess-bot -c STANDARD -l us-central1 gs://v7p3r-analytics-reports

# Set permissions
gsutil iam ch serviceAccount:v7p3r-analytics@v7p3r-lichess-bot.iam.gserviceaccount.com:objectAdmin gs://v7p3r-analytics-reports

# Initialize structure
echo '{"weeks": [], "last_updated": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}' | gsutil cp - gs://v7p3r-analytics-reports/historical_summary.json
```

### Step 2: Add Cloud Storage to Pipeline

**New File**: `storage_manager.py`

Features:
- Upload reports to dated folders
- Download historical summaries
- Update aggregated metrics
- List available weeks

### Step 3: Enhanced Report Generator

**Update**: `report_generator.py`

New features:
- Week-over-week comparison
- 4-week trend tables
- Historical charts data (JSON for future visualization)
- Regression detection (alert if metrics drop)

### Step 4: Email Generator

**New File**: `email_generator.py`

Features:
- Generate HTML email from report data
- Embed trend tables
- Highlight improvements/regressions
- Attach full JSON summary

### Step 5: SendGrid Integration

**New File**: `email_sender.py`

Features:
- SendGrid API client
- HTML email delivery
- Error handling and retries
- Delivery confirmation logging

### Step 6: Update Pipeline

**Update**: `weekly_pipeline_simple.py`

Add stages:
1. Download historical data
2. Enhanced report generation with trends
3. Upload to Cloud Storage
4. Generate email
5. Send email

### Step 7: Update Dockerfile

Add dependencies:
```dockerfile
RUN pip install google-cloud-storage sendgrid
```

### Step 8: Deploy to Cloud Run

Update environment variables:
- `GCS_BUCKET=v7p3r-analytics-reports`
- `SENDGRID_API_KEY=<from SendGrid>`
- `EMAIL_TO=patssnyder@gmail.com`
- `EMAIL_FROM=analytics@v7p3r-engine.com`

---

## Historical Metrics Dashboard (Future Phase)

### Static HTML Dashboard

Generate a static HTML page uploaded to Cloud Storage:

```
https://storage.googleapis.com/v7p3r-analytics-reports/dashboard.html
```

**Features**:
- Interactive charts (Chart.js)
- Version timeline
- Win rate trends
- Blunder rate over time
- Top1 alignment progression
- Auto-refreshed weekly

**Benefits**:
- No server needed (static file)
- Shareable link
- Fast loading
- Mobile-friendly

---

## Cost Estimate

### Monthly Costs

| Service | Usage | Cost |
|---------|-------|------|
| Cloud Run | 4 executions × 1 hour | $0.40 |
| Cloud Storage | ~1 GB data | $0.02 |
| SendGrid | 4 emails/month | $0.00 (free tier) |
| Cloud Scheduler | 4 triggers | $0.40 |
| **Total** | | **~$0.82/month** |

### Annual Cost: ~$10/year

---

## Success Metrics

### Week 1 (Dec 8-15)
- [ ] Cloud Storage bucket created and tested
- [ ] Historical data persistence working
- [ ] First email delivered to patssnyder@gmail.com
- [ ] Week-over-week comparison showing v17.5 vs v17.1

### Week 2 (Dec 16-22)
- [ ] Second automated email with 2-week trend
- [ ] Historical summary accumulating correctly
- [ ] Email format validated and refined

### Week 4 (Dec 30-Jan 5)
- [ ] 4-week trend table fully populated
- [ ] Static dashboard generated
- [ ] System running autonomously with zero intervention

---

## Rollback Safety

### Data Preservation
- Cloud Storage is append-only (no overwrites)
- Each week creates new folder
- Historical summary maintains full history
- Can manually recover from any week's data

### Email Delivery Fallback
- If SendGrid fails, logs contain report summary
- Can manually retrieve from Cloud Storage
- Option to add Slack/Discord webhook as backup

### Pipeline Failures
- Failed analysis doesn't delete historical data
- Partial reports still uploaded
- Email includes error notifications
- Can re-run manually for failed weeks

---

## Next Actions

### Immediate (This Session)
1. Create Cloud Storage bucket
2. Implement `storage_manager.py`
3. Add historical tracking to report generator
4. Test upload/download locally

### Tonight/Tomorrow
5. Set up SendGrid account
6. Implement email generator + sender
7. Update pipeline with new stages
8. Test end-to-end locally

### This Weekend
9. Rebuild Docker image
10. Deploy to Cloud Run
11. Test manual execution
12. Validate first email delivery

### Next Week
13. Monitor automated Sunday run
14. Receive first production email
15. Review 2-week comparison
16. Begin dashboard development

---

**Status**: Ready to implement  
**Owner**: Pat Snyder  
**Priority**: High  
**Risk**: Low (incremental changes, no breaking existing pipeline)
