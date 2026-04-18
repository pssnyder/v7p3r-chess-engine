# v7p3r-labs Quick Start Guide

**Goal**: Get v7p3r-labs chess interface running this weekend  
**Time**: ~8-10 hours total  
**Status**: Backend code ready, VM deployment next

---

## What's Been Created

✅ **Backend API** (FastAPI + WebSocket)
- `backend/main.py` - WebSocket server with game loop
- `backend/uci_manager.py` - Docker container lifecycle + UCI communication  
- `backend/game_manager.py` - Concurrent game manager (max 3 games)
- `backend/models.py` - Data types (GameConfig, Move, GameState)
- `backend/requirements.txt` - Python dependencies

✅ **Docker Configuration**
- `docker/Dockerfile` - Engine container template
- `docker/build-engines.ps1` - Build all 5 versions locally (Windows)
- `docker/push-engines.ps1` - Push to Google Container Registry

✅ **Deployment Scripts**
- `scripts/setup-vm.sh` - Install Docker/Python on VM
- `scripts/setup-backend.sh` - Deploy backend as systemd service
- `IMPLEMENTATION_PLAN.md` - Detailed phase breakdown

---

## Step-by-Step: Weekend Sprint

### **Step 1: Build Docker Images** (15 min)

```powershell
cd "e:\Programming Stuff\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\v7p3r-labs\docker"

.\build-engines.ps1   # Builds 5 engine versions
.\push-engines.ps1    # Pushes to gcr.io/rts-labs-f3981/v7p3r:VERSION
```

**What this does**: Creates Docker images from existing engine source in `lichess/engines/V7P3R_v*`

---

### **Step 2: Create VM** (5 min)

```powershell
gcloud compute instances create v7p3r-labs-vm `
  --project=rts-labs-f3981 `
  --zone=us-central1-a `
  --machine-type=e2-micro `
  --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default `
  --tags=http-server,https-server `
  --create-disk=auto-delete=yes,boot=yes,image=projects/debian-cloud/global/images/debian-12-bookworm-v20250113,size=30 `
  --labels=gce-deployment-group=v7p3r-labs
```

**Cost**: $0/month (free tier - first e2-micro in us-central1)

---

### **Step 3: Configure Firewall** (2 min)

```powershell
gcloud compute firewall-rules create v7p3r-labs-allow-http `
  --direction=INGRESS `
  --priority=1000 `
  --network=default `
  --action=ALLOW `
  --rules=tcp:80,tcp:443,tcp:8080 `
  --source-ranges=0.0.0.0/0 `
  --target-tags=http-server
```

---

### **Step 4: Setup VM** (5 min)

```bash
# SSH into VM
gcloud compute ssh v7p3r-labs-vm --zone=us-central1-a

# Upload and run setup script
gcloud compute scp scripts/setup-vm.sh v7p3r-labs-vm:/tmp/ --zone=us-central1-a
gcloud compute ssh v7p3r-labs-vm --zone=us-central1-a --command="bash /tmp/setup-vm.sh"
```

**What this does**: Installs Docker, Python 3.12, configures GCR auth

---

### **Step 5: Deploy Backend** (10 min)

```powershell
# Upload backend code from local machine
gcloud compute scp --recurse backend/ v7p3r-labs-vm:/opt/v7p3r-labs/backend --zone=us-central1-a

# Upload and run backend setup script
gcloud compute scp scripts/setup-backend.sh v7p3r-labs-vm:/opt/v7p3r-labs/backend/ --zone=us-central1-a
gcloud compute ssh v7p3r-labs-vm --zone=us-central1-a --command="bash /opt/v7p3r-labs/backend/setup-backend.sh"
```

**What this does**: 
- Installs Python dependencies
- Pulls all 5 Docker images to VM disk
- Creates systemd service (auto-restart on crash/reboot)
- Starts API server on port 8080

---

### **Step 6: Test Backend** (2 min)

```bash
# On VM
curl http://localhost:8080/

# Expected: {"service":"v7p3r-labs-api","status":"online","active_games":0}
```

Get VM external IP for frontend connection:

```powershell
gcloud compute instances describe v7p3r-labs-vm --zone=us-central1-a --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
```

---

### **Step 7: Frontend Development** (2-3 hours)

**Files to create**:
1. `src/App.tsx` - Add routing for `/play` page
2. `src/pages/Play.tsx` - Game config UI + chessboard
3. `src/hooks/useChessEngine.ts` - WebSocket communication
4. `src/components/ChessBoard.tsx` - Chessboard.js wrapper

**Dependencies**:
```powershell
npm install react-router-dom@7.13.0
npm install chessboardjsx@5.5.2
npm install chess.js@1.0.0-beta.8
```

**Vite config** (add WebSocket proxy):
```typescript
// vite.config.ts
export default defineConfig({
  server: {
    proxy: {
      '/api': 'http://VM_EXTERNAL_IP:8080',
      '/ws': {
        target: 'ws://VM_EXTERNAL_IP:8080',
        ws: true
      }
    }
  }
})
```

---

### **Step 8: Deploy Frontend** (5 min)

```powershell
npm run build
firebase deploy --only hosting:v7p3r-labs
```

**Live at**: https://v7p3r-chess-engine-f3981.web.app

---

## Testing Checklist

- [ ] Human vs v18.4 (5+3 blitz)
- [ ] Engine vs Engine (v17.7 vs v18.4, 2+1 bullet)
- [ ] Concurrent games (start 3 simultaneously)
- [ ] Container cleanup (verify `docker ps` after games)
- [ ] Time control accuracy (countdown matches increment)
- [ ] Game end detection (checkmate, stalemate, time forfeit)

---

## Quick Commands Reference

```bash
# View logs
gcloud compute ssh v7p3r-labs-vm --zone=us-central1-a --command="sudo journalctl -u v7p3r-api -f"

# Restart backend
gcloud compute ssh v7p3r-labs-vm --zone=us-central1-a --command="sudo systemctl restart v7p3r-api"

# Check running containers
gcloud compute ssh v7p3r-labs-vm --zone=us-central1-a --command="docker ps"

# Monitor resource usage
gcloud compute ssh v7p3r-labs-vm --zone=us-central1-a --command="docker stats"
```

---

## Troubleshooting

**Backend won't start**:
```bash
sudo journalctl -u v7p3r-api -n 50  # View last 50 log lines
```

**Docker images not found**:
```bash
docker pull gcr.io/rts-labs-f3981/v7p3r:18.4  # Re-pull specific version
```

**WebSocket connection failed**:
- Verify firewall allows port 8080
- Check VM external IP matches frontend config
- Test with: `curl http://VM_IP:8080/`

---

## What's Next?

After successful deployment:
1. **Analytics**: Add Google Analytics to track usage
2. **Move History**: Display PGN notation
3. **Position Analysis**: Show engine evaluation scores  
4. **Mobile Optimization**: Responsive chessboard
5. **Spectator Mode**: Shareable game links

---

**Ready?** Start with Step 1 (build Docker images) and work through sequentially!
