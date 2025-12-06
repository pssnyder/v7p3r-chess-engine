# 🚀 Quick Start - V7P3R Analytics

## Run Now (Background-Friendly)

**Windows:** Double-click `run_weekly_analytics.bat`  
**Linux/Mac:** Run `./run_weekly_analytics.sh`

**Resource Usage:**
- **4 CPU cores** (leaves 12+ cores free for you)
- **4GB RAM max**
- **4 parallel workers** (safe for 16-core system)
- **Expected time:** 3-4 hours (runs in background)

**Tip:** Run overnight or while AFK - it won't slow down your system!

---

## First Run - SUCCESS ✅

- **183 games** analyzed (Nov 28 - Dec 5, 2025)
- **56.8% win rate** (104W-59L-20D)
- **96 minutes** with 12 workers
- **All requirements met** including theme analysis

Reports saved to: `analytics_reports/2025/week_49_2025-12-01/`

---

## Automate Weekly (Optional)

### Windows Task Scheduler

1. Open Task Scheduler: `Win+R` → `taskschd.msc`
2. Create Basic Task: "V7P3R Weekly Analytics"
3. Trigger: Weekly, Sunday 2:00 AM
4. Action: Start a program
   - Program: `C:\Program Files\Docker\Docker\resources\bin\docker-compose.exe`
   - Arguments: `up --build`
   - Start in: `S:\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\analytics`

### Linux/Mac Cron

```bash
# Add to crontab (crontab -e)
0 2 * * 0 cd /path/to/analytics && docker-compose up --build
```

---

## View Reports

```bash
# Summary report
cat analytics_reports/2025/week_49_2025-12-01/technical_report.md

# Detailed JSON
cat analytics_reports/2025/week_49_2025-12-01/weekly_summary.json

# Individual games
ls analytics_reports/2025/week_49_2025-12-01/games/
```

---

## Troubleshooting

**Check logs:**
```bash
docker logs v7p3r-analytics
```

**Rebuild:**
```bash
docker-compose down
docker-compose up --build
```

---

## Documentation

📘 **Full Guide**: `docs/Analytics_Automation_Guide.md`  
📊 **Data Schema**: `docs/Analytics_Data_Schema.md`  
📋 **System Overview**: `docs/Local_Analytics_System_READY.md`

---

**Status**: ✅ Production Ready | Zero Cloud Costs | Fully Automated
