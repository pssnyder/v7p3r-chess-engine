# V7P3R Analytics System

## Overview

Comprehensive automated analytics system for v7p3r_bot chess games using Stockfish analysis. Generates weekly reports with theme detection, move quality analysis, and actionable insights for engine improvement.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GCP Cloud Scheduler                       │
│              (Triggers every Monday 9 AM)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Cloud Run Job Container                     │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  1. Game Collector                                     │ │
│  │     └─> Fetch PGNs from v7p3r-production-bot VM       │ │
│  │                                                         │ │
│  │  2. Stockfish Analyzer                                 │ │
│  │     └─> Deep analysis with theme detection            │ │
│  │                                                         │ │
│  │  3. Report Generator                                   │ │
│  │     └─> Weekly summary with insights                  │ │
│  │                                                         │ │
│  │  4. Email Delivery                                     │ │
│  │     └─> Send report via SendGrid                      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Features

### 1. **Stockfish-Powered Analysis**
- Deep position evaluation (20 ply depth)
- Move classification: Best, Excellent, Good, Inaccuracy, Mistake, Blunder, Critical Blunder
- Best move differential (how close to Stockfish's top recommendation)
- Top 5 move alignment tracking

### 2. **Chess Theme Detection**
Automatically detects and tracks:
- **Castling**: King-side and queen-side
- **Pawn Structure**: Isolated pawns, doubled pawns, passed pawns
- **Piece Activity**: Bishop pair, knight outposts, rooks on open files
- **Tactical Patterns**: Pins, forks, skewers, batteries
- **Mate Threats**: Tracks positions with mate opportunities

### 3. **Performance Metrics**
- Average centipawn loss (CPL)
- Move quality distribution
- Stockfish alignment percentages (Top 1, Top 3, Top 5)
- Opening performance breakdown
- Opponent-specific analysis

### 4. **Weekly Reports**
- **JSON Format**: Structured data for programmatic analysis
- **Markdown Format**: Human-readable summary with tables
- **Email Delivery**: Automatic delivery every Monday morning

## Installation

### Local Development

```bash
cd analytics

# Install dependencies
pip install -r requirements.txt

# Install Stockfish
# Linux/Mac:
wget https://stockfishchess.org/files/stockfish-16-linux-x64-avx2.zip
unzip stockfish-16-linux-x64-avx2.zip
sudo mv stockfish /usr/local/bin/

# Windows:
# Download from https://stockfishchess.org/download/
# Add to PATH or note the full path
```

### GCP Deployment

```bash
# Prerequisites
gcloud auth login
gcloud config set project v7p3r-lichess-bot

# Deploy (builds container, creates Cloud Run job, sets up scheduler)
chmod +x deploy_gcp.sh
./deploy_gcp.sh

# Configure environment variables in Cloud Run
gcloud run jobs update v7p3r-weekly-analytics \
  --region us-central1 \
  --set-env-vars="SENDGRID_API_KEY=your_key,TO_EMAIL=your@email.com"
```

## Usage

### Manual Local Run

```bash
# Analyze last 7 days of games
python weekly_pipeline.py \
  --stockfish /usr/local/bin/stockfish \
  --work-dir ./analytics_workspace \
  --days-back 7

# Results will be in:
# - analytics_workspace/reports/weekly_report_TIMESTAMP.json
# - analytics_workspace/reports/weekly_report_TIMESTAMP.md
```

### Test Individual Components

```bash
# Test game collection
python game_collector.py ./test_downloads 7

# Test single game analysis
python v7p3r_analytics.py /usr/local/bin/stockfish game.pgn

# Test report generation
python report_generator.py ./game_analyses ./output_report.json

# Test email delivery (requires SendGrid API key)
export SENDGRID_API_KEY="your_key"
export TO_EMAIL="your@email.com"
python email_delivery.py report.md report.json
```

### Manual GCP Run

```bash
# Trigger analytics job manually
gcloud run jobs execute v7p3r-weekly-analytics \
  --region us-central1 \
  --project v7p3r-lichess-bot

# Check execution logs
gcloud logging read \
  "resource.type=cloud_run_job AND resource.labels.job_name=v7p3r-weekly-analytics" \
  --limit 100 \
  --format json
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `STOCKFISH_PATH` | Path to Stockfish executable | `/usr/local/bin/stockfish` | Yes |
| `WORK_DIR` | Working directory for outputs | `./analytics_workspace` | No |
| `BOT_USERNAME` | Bot's Lichess username | `v7p3r_bot` | No |
| `DAYS_BACK` | Days of game history | `7` | No |
| `SENDGRID_API_KEY` | SendGrid API key for email | - | For email |
| `TO_EMAIL` | Report recipient email | - | For email |
| `FROM_EMAIL` | Sender email address | `analytics@v7p3r.com` | No |

### GCP Configuration

```bash
# Update schedule (default: Monday 9 AM)
gcloud scheduler jobs update http v7p3r-analytics-weekly \
  --location us-central1 \
  --schedule "0 9 * * 1"  # Cron format

# Update analysis depth (edit Dockerfile)
ENV ANALYSIS_DEPTH=20

# Update time allocation per move (edit v7p3r_analytics.py)
self.analysis_time = 0.5  # seconds
```

## Report Format

### JSON Report Structure

```json
{
  "metadata": {
    "generated": "2025-11-30T09:00:00",
    "period": "Last 15 games",
    "bot": "v7p3r_bot"
  },
  "games": {
    "total": 15,
    "wins": 8,
    "losses": 5,
    "draws": 2,
    "win_rate": 53.3
  },
  "performance": {
    "avg_centipawn_loss": 35.2,
    "move_quality": {
      "best": 45,
      "excellent": 32,
      "good": 18,
      "inaccuracies": 12,
      "mistakes": 8,
      "blunders": 3,
      "critical_blunders": 1
    },
    "stockfish_alignment": {
      "top1": 45.2,
      "top3": 72.8,
      "top5": 89.1
    }
  },
  "openings": {
    "Queen's Gambit Declined": {
      "games": 5,
      "win_rate": 60.0,
      "avg_cpl": 32.1,
      "record": "3-1-1"
    }
  },
  "themes": {
    "castling_kingside": 12,
    "passed_pawns": 23,
    "bishop_pair_games": 8
  }
}
```

### Markdown Report Sections

1. **Overall Performance**: Win rate, move quality distribution
2. **Move Quality Analysis**: Best/Excellent/Good/Inaccuracies/Mistakes/Blunders
3. **Stockfish Alignment**: Top 1/3/5 move percentages
4. **Opening Performance**: Best and worst openings with statistics
5. **Opponent Analysis**: Best and worst matchups
6. **Theme Detection**: Chess pattern adherence
7. **Recommendations**: Actionable insights for improvement

## Theme Detection Details

### Implemented Themes

| Theme | Detection Method | Purpose |
|-------|------------------|---------|
| **Castling** | Move type check | Safety evaluation |
| **Fianchetto** | Bishop on long diagonal | Opening theory adherence |
| **Isolated Pawns** | Pawn chain analysis | Structural weaknesses |
| **Passed Pawns** | Opponent pawn blocking | Endgame potential |
| **Bishop Pair** | Two bishops present | Positional advantage |
| **Knight Outpost** | Protected knight on enemy territory | Piece coordination |
| **Rook Open File** | Rook with no friendly pawns | Activity measurement |
| **Tactical Patterns** | Pin/Fork/Skewer detection | Tactical awareness |

### Future Theme Enhancements

- Fianchetto detection (bishop on b2, g2, b7, g7)
- Rook on 7th rank detection
- Battery detection (queen + rook/bishop alignment)
- Discovered attack identification
- Advanced mate threat tracking (mate in 3+)

## Integration with Engine Development

### Workflow

1. **Weekly Analysis**: Automated run every Monday
2. **Report Review**: Examine move quality and theme adherence
3. **Identify Patterns**: Find recurring mistakes or missed themes
4. **Engine Enhancement**: Implement heuristics to address weaknesses
5. **Validation**: Monitor next week's report for improvement

### Example Use Cases

**Scenario 1: High Blunder Rate in Endgames**
- Report shows 60% of critical blunders occur after move 40
- Action: Enhance endgame evaluation in `v7p3r_evaluator.py`
- Validation: Next report should show reduced endgame CPL

**Scenario 2: Low Passed Pawn Recognition**
- Report shows only 15% of games develop passed pawns
- Action: Increase passed pawn bonuses in evaluation
- Validation: Monitor passed_pawns theme count increase

**Scenario 3: Opening Performance Disparity**
- Report shows 70% win rate in Queen's Gambit vs 30% in Sicilian
- Action: Review opening book for Sicilian lines
- Validation: Track Sicilian performance improvement

## Monitoring

### Cloud Run Job Monitoring

```bash
# View recent executions
gcloud run jobs executions list \
  --job v7p3r-weekly-analytics \
  --region us-central1

# Get execution details
gcloud run jobs executions describe EXECUTION_NAME \
  --region us-central1

# Stream logs during execution
gcloud logging tail \
  "resource.type=cloud_run_job AND resource.labels.job_name=v7p3r-weekly-analytics"
```

### Email Delivery Status

- Check SendGrid dashboard for delivery status
- Monitor bounce rates and open rates
- Configure SendGrid webhooks for detailed tracking

### Cost Estimation

**Cloud Run Job:**
- CPU: 2 vCPU × 1 hour/week = ~$0.08/week
- Memory: 2GB × 1 hour/week = ~$0.01/week
- **Total: ~$0.09/week or ~$4.68/year**

**Cloud Scheduler:**
- 1 job = Free tier (3 jobs/month free)

**SendGrid:**
- Free tier: 100 emails/day
- Weekly reports: Well within free tier

**Total estimated cost: ~$5/year**

## Troubleshooting

### Common Issues

**Issue: Game collection fails**
```bash
# Verify VM access
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a

# Check container status
docker ps | grep v7p3r-production

# Verify game records directory
docker exec v7p3r-production ls -la /lichess-bot/game_records
```

**Issue: Stockfish not found**
```bash
# Verify Stockfish installation
which stockfish
stockfish --version

# Update path in pipeline
export STOCKFISH_PATH=/path/to/stockfish
```

**Issue: Email not sending**
```bash
# Test SendGrid API key
curl -X POST https://api.sendgrid.com/v3/mail/send \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json"

# Check environment variables
gcloud run jobs describe v7p3r-weekly-analytics \
  --region us-central1 \
  --format="value(spec.template.spec.template.spec.containers[0].env)"
```

## Development

### Adding New Themes

1. Edit `v7p3r_analytics.py`:
   ```python
   # In ThemeDetection dataclass
   new_theme: int = 0
   
   # In _detect_themes method
   if self._detect_new_theme(board, move):
       themes.new_theme += 1
   ```

2. Update `report_generator.py` to track new theme

3. Test with sample games

### Custom Analysis Metrics

Extend `MoveAnalysis` dataclass:
```python
@dataclass
class MoveAnalysis:
    # ... existing fields
    custom_metric: float = 0.0
```

Update analyzer to calculate metric during `_analyze_move()`.

## Support

For issues or questions:
- GitHub Issues: v7p3r-chess-engine repository
- Email: (your contact email)

## License

Part of the V7P3R Chess Engine project.
