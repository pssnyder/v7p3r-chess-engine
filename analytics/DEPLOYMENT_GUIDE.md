# V7P3R Analytics - GCP Deployment Guide

## Quick Start Deployment

### Prerequisites
1. Docker Desktop running
2. gcloud CLI authenticated: `gcloud auth login`
3. Project set: `gcloud config set project v7p3r-lichess-bot`

### Deploy to GCP (One Command)

```bash
cd "s:/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/analytics"
bash deploy_gcp.sh
```

This will:
- Enable required GCP APIs
- Create service account with permissions
- Build Docker image with Stockfish 17.1
- Push to Google Container Registry
- Create Cloud Run job
- Configure Cloud Scheduler (Sundays midnight UTC)

### Manual Test Run

```bash
gcloud run jobs execute v7p3r-weekly-analytics \
  --region us-central1 \
  --project v7p3r-lichess-bot
```

### Monitor Execution

```bash
# View logs
gcloud logging read \
  "resource.type=cloud_run_job AND resource.labels.job_name=v7p3r-weekly-analytics" \
  --limit 50 \
  --project v7p3r-lichess-bot \
  --format="table(timestamp,textPayload)"

# Check job status
gcloud run jobs describe v7p3r-weekly-analytics \
  --region us-central1 \
  --project v7p3r-lichess-bot
```

## System Architecture

```
┌─────────────────────────────────────────┐
│  Cloud Scheduler (Sundays midnight UTC) │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  Cloud Run Job: v7p3r-weekly-analytics  │
│  ├─ CPU: 2 cores                        │
│  ├─ Memory: 2 GB                        │
│  ├─ Timeout: 1 hour                     │
│  └─ Workers: 12 parallel Stockfish      │
└──────────────┬──────────────────────────┘
               ▼
        ┌──────────────┐
        │  Pipeline    │
        └──────┬───────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌──────────────┐
│ Lichess API │  │ Stockfish    │
│ (v7p3r_bot) │  │ Analysis     │
└──────┬──────┘  └──────┬───────┘
       │                │
       └────────┬───────┘
                ▼
      ┌──────────────────┐
      │  Reports         │
      │  /workspace/     │
      │  ├─ pgn/         │
      │  └─ reports/     │
      └──────────────────┘
```

## Cost Estimate

**Monthly Cost: ~$0.50**
- Cloud Run: $0.10/week × 4 = $0.40
- Cloud Scheduler: $0.10/month
- Cloud Storage: $0.01/month
- **Total: ~$0.50/month**

## Deployment Checklist

### Pre-Deployment
- [x] Dockerfile updated with Stockfish 17.1
- [x] weekly_pipeline_simple.py created
- [x] version_tracker.py includes v17.5
- [x] deploy_gcp.sh configured for v7p3r-lichess-bot project
- [ ] Docker Desktop running

### Deployment Steps
1. [ ] Start Docker Desktop
2. [ ] Run `bash deploy_gcp.sh`
3. [ ] Verify build completes (5-10 minutes)
4. [ ] Test manual execution
5. [ ] Verify logs show success
6. [ ] Confirm scheduler created

### Post-Deployment
- [ ] Test run completes successfully
- [ ] Reports generated in /workspace/reports/
- [ ] Wait for Sunday midnight UTC
- [ ] Verify first automated run
- [ ] Download and review first report

## Configuration

### Environment Variables (Set in Cloud Run Job)
```bash
STOCKFISH_PATH=/usr/local/bin/stockfish
WORK_DIR=/workspace
PYTHONUNBUFFERED=1
```

### Pipeline Arguments
- `--stockfish`: Path to Stockfish (default: /usr/local/bin/stockfish)
- `--work-dir`: Working directory (default: /workspace)
- `--days-back`: Days to analyze (default: 7)
- `--workers`: Parallel workers (default: 12)

### Schedule
- **Trigger**: Every Sunday at 00:00 UTC (7pm EST Saturday)
- **Cron**: `0 0 * * 0`
- **Timezone**: UTC

## Troubleshooting

### Docker Build Fails
```bash
# Ensure Docker Desktop is running
docker ps

# If not running, start Docker Desktop
# Try build again
cd analytics
docker build -t v7p3r-analytics:test .
```

### GCP Permissions Error
```bash
# Grant necessary permissions to service account
gcloud projects add-iam-policy-binding v7p3r-lichess-bot \
  --member="serviceAccount:v7p3r-analytics@v7p3r-lichess-bot.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

### Job Execution Timeout
```bash
# Increase timeout to 2 hours
gcloud run jobs update v7p3r-weekly-analytics \
  --region us-central1 \
  --task-timeout 7200 \
  --project v7p3r-lichess-bot
```

### Reduce Workers if Memory Issues
```bash
# Update job to use 8 workers instead of 12
gcloud run jobs update v7p3r-weekly-analytics \
  --region us-central1 \
  --update-args="--workers=8" \
  --project v7p3r-lichess-bot
```

## Accessing Results

### Cloud Run Job Logs
```bash
gcloud logging read \
  "resource.type=cloud_run_job" \
  --project v7p3r-lichess-bot \
  --limit 100
```

### Export Reports
Reports are stored in the job's filesystem but are ephemeral. Future enhancement: upload to Cloud Storage.

```bash
# TODO: Add Cloud Storage integration
gsutil cp /workspace/reports/* gs://v7p3r-analytics-reports/
```

## Next Steps

1. **Deploy System** (Today/Tomorrow)
   - Run `bash deploy_gcp.sh`
   - Test manual execution
   - Verify success

2. **First Scheduled Run** (Sunday midnight)
   - Monitor logs
   - Verify analysis completes
   - Review generated reports

3. **Phase 2 Enhancements** (Next Week)
   - Add Cloud Storage for persistent reports
   - Set up email delivery (SendGrid/Gmail API)
   - Create web dashboard for viewing results
   - Add Slack notifications

## Support

- GCP Project: `v7p3r-lichess-bot`
- Cloud Run Job: `v7p3r-weekly-analytics`
- Service Account: `v7p3r-analytics@v7p3r-lichess-bot.iam.gserviceaccount.com`
- Scheduler: `v7p3r-analytics-weekly`

For issues, check Cloud Run logs first:
```bash
gcloud logging read "resource.type=cloud_run_job" --project v7p3r-lichess-bot --limit 50
```
