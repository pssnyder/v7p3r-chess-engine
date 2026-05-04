# V7P3R Deployment Quick Reference Card

**Last Updated**: 2026-05-03  
**Current Production**: v18.3.0

---

## 🚀 Quick Deployment Commands

### 1. Create Package (Windows)
```powershell
cd "E:\Programming Stuff\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine"
$version = "18.3"
$date = Get-Date -Format "yyyyMMdd"
Compress-Archive -Path "lichess\engines\V7P3R_v${version}_${date}\src\*" -DestinationPath "v${version}-src.zip"
```

### 2. Upload to VM
```bash
gcloud compute scp v18.3-src.zip v7p3r-production-bot:/tmp/ \
  --zone=us-central1-a --project=v7p3r-lichess-bot
```

### 3. Extract & Stage
```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="cd /tmp ; \
    sudo python3 -m zipfile -e v18.3-src.zip v18.3_extracted ; \
    sudo cp -r v18.3_extracted/* /home/v7p3r/engines/v7p3r/ ; \
    sudo chown -R v7p3r:v7p3r /home/v7p3r/engines/v7p3r/"
```

### 4. Deploy Wrapper
```bash
# Create locally
cat > run_v7p3r.sh << 'EOF'
#!/bin/bash
cd /lichess-bot/engines/v7p3r
exec python3 v7p3r_uci.py
EOF

# Upload
gcloud compute scp run_v7p3r.sh v7p3r-production-bot:/tmp/ \
  --zone=us-central1-a --project=v7p3r-lichess-bot

# Deploy
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo cp /tmp/run_v7p3r.sh /home/v7p3r/engines/v7p3r/ ; \
    sudo chmod +x /home/v7p3r/engines/v7p3r/run_v7p3r.sh"
```

### 5. Copy to Container (CRITICAL!)
```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker stop v7p3r-production ; \
    sudo docker cp /home/v7p3r/engines/v7p3r/. v7p3r-production:/lichess-bot/engines/v7p3r/ ; \
    sudo docker exec v7p3r-production chmod +x /lichess-bot/engines/v7p3r/run_v7p3r.sh"
```

### 6. Start & Verify
```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker start v7p3r-production ; \
    sleep 10 ; \
    sudo docker logs v7p3r-production --tail 50"
```

---

## 🔍 Quick Status Checks

### Check Bot Status
```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker logs v7p3r-production --tail 30"
```

### Check Engine Version
```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker exec v7p3r-production head -5 /lichess-bot/engines/v7p3r/v7p3r.py"
```

### Test UCI
```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker exec v7p3r-production bash -c 'echo uci | ./engines/v7p3r/run_v7p3r.sh | head -10'"
```

---

## 🆘 Emergency Rollback

```bash
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --project=v7p3r-lichess-bot \
  --command="sudo docker stop v7p3r-production ; \
    sudo docker cp /home/v7p3r/engines/v7p3r.backup/. v7p3r-production:/lichess-bot/engines/v7p3r/ ; \
    sudo docker exec v7p3r-production chmod +x /lichess-bot/engines/v7p3r/run_v7p3r.sh ; \
    sudo docker start v7p3r-production"
```

---

## ⚠️ Critical Rules

1. **NEVER mount /engines directory** - Use `docker cp` instead
2. **ALWAYS use wrapper script** - Config must reference `run_v7p3r.sh`
3. **ALWAYS stop container** before copying files
4. **ALWAYS use ZIP on Windows** - TAR.GZ causes corruption
5. **ALWAYS test UCI** after deployment

---

## 📁 File Paths

- **VM Staging**: `/home/v7p3r/engines/v7p3r/`
- **Container Path**: `/lichess-bot/engines/v7p3r/` (internal filesystem)
- **Config**: `/home/v7p3r/config.yml`
- **Logs**: `/home/v7p3r/logs/`

---

## 🎯 Success Indicators

After deployment, you should see:
```
INFO     Engine configuration OK
INFO     Welcome v7p3r_bot!
INFO     You're now connected to https://lichess.org/
```

UCI test should return:
```
id name V7P3R v18.3
id author pssnyder
uciok
```

---

**For detailed troubleshooting**: See `lichess/DEPLOYMENT_GUIDE.md`  
**For version history**: See `.github/instructions/version_management.instructions.md`
