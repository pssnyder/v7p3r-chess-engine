# v7p3r-labs Implementation Plan

**Project**: Web interface for V7P3R chess engine  
**Timeline**: Weekend sprint (~8-10 hours)  
**Architecture**: Separate e2-micro VM, FastAPI backend, React frontend

---

## Phase 1: VM Setup (5 minutes)

### Create e2-micro VM Instance

```powershell
gcloud compute instances create v7p3r-labs-vm `
  --project=rts-labs-f3981 `
  --zone=us-central1-a `
  --machine-type=e2-micro `
  --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default `
  --maintenance-policy=MIGRATE `
  --provisioning-model=STANDARD `
  --tags=http-server,https-server `
  --create-disk=auto-delete=yes,boot=yes,device-name=v7p3r-labs-vm,image=projects/debian-cloud/global/images/debian-12-bookworm-v20250113,mode=rw,size=30,type=pd-balanced `
  --no-shielded-secure-boot `
  --shielded-vtpm `
  --shielded-integrity-monitoring `
  --labels=gce-deployment-group=v7p3r-labs `
  --reservation-affinity=any
```

**Cost**: $0/month (free tier eligible - first e2-micro in us-central1)

### Configure Firewall

```powershell
# Allow HTTP/HTTPS traffic
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

## Phase 2: Docker Images (30 minutes)

### Local Build & Push

```powershell
cd "e:\Programming Stuff\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\v7p3r-labs\docker"

# Build all 5 engine versions
.\build-engines.ps1

# Push to Google Container Registry
.\push-engines.ps1
```

**Output**: 5 Docker images in GCR
- `gcr.io/rts-labs-f3981/v7p3r:12.6`
- `gcr.io/rts-labs-f3981/v7p3r:14.1`
- `gcr.io/rts-labs-f3981/v7p3r:16.1`
- `gcr.io/rts-labs-f3981/v7p3r:17.7`
- `gcr.io/rts-labs-f3981/v7p3r:18.4`

---

## Phase 3: Deploy Backend (15 minutes)

### Install Dependencies on VM

```bash
# SSH into VM
gcloud compute ssh v7p3r-labs-vm --zone=us-central1-a

# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker

# Add user to docker group
sudo usermod -aG docker $USER

# Install Python 3.12
sudo apt-get install -y python3.12 python3.12-venv python3-pip

# Configure Docker for GCR
gcloud auth configure-docker gcr.io
```

### Upload Backend Code

```powershell
# From local machine
cd "e:\Programming Stuff\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\v7p3r-labs"

gcloud compute scp --recurse backend/ v7p3r-labs-vm:/tmp/backend --zone=us-central1-a
```

### Setup Backend Service

```bash
# On VM
sudo mkdir -p /opt/v7p3r-labs
sudo mv /tmp/backend /opt/v7p3r-labs/

cd /opt/v7p3r-labs/backend
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Create Systemd Service

```bash
sudo tee /etc/systemd/system/v7p3r-api.service > /dev/null <<EOF
[Unit]
Description=v7p3r-labs FastAPI Backend
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/v7p3r-labs/backend
Environment="PATH=/opt/v7p3r-labs/backend/venv/bin"
ExecStart=/opt/v7p3r-labs/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable v7p3r-api
sudo systemctl start v7p3r-api

# Check status
sudo systemctl status v7p3r-api
```

### Pull Docker Images to VM

```bash
# Pre-pull all engine images (speeds up first game)
docker pull gcr.io/rts-labs-f3981/v7p3r:12.6
docker pull gcr.io/rts-labs-f3981/v7p3r:14.1
docker pull gcr.io/rts-labs-f3981/v7p3r:16.1
docker pull gcr.io/rts-labs-f3981/v7p3r:17.7
docker pull gcr.io/rts-labs-f3981/v7p3r:18.4
```

### Test Backend

```bash
# Check health endpoint
curl http://localhost:8080/

# Expected: {"service":"v7p3r-labs-api","status":"online","active_games":0}
```

---

## Phase 4: React Frontend (2-3 hours)

### Install Dependencies

```powershell
cd "e:\Programming Stuff\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\v7p3r-labs"

npm install react-router-dom@7.13.0
npm install chessboardjsx@5.5.2
npm install chess.js@1.0.0-beta.8
```

### Create Components

**Files to create:**
1. `src/App.tsx` - Router and layout
2. `src/pages/Play.tsx` - Game configuration and play page
3. `src/components/ChessBoard.tsx` - Chessboard component
4. `src/hooks/useChessEngine.ts` - WebSocket hook
5. `src/utils/api.ts` - API client

### Update Vite Config

Add WebSocket proxy to `vite.config.ts`:

```typescript
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

### Test Locally

```powershell
npm run dev
# Navigate to http://localhost:5173/play
```

---

## Phase 5: Deploy Frontend (10 minutes)

### Build & Deploy

```powershell
npm run build
firebase deploy --only hosting:v7p3r-labs
```

### Update DNS (if needed)

Configure `v7p3r.labs.rtsts.tech` to point to Firebase Hosting:
- Firebase Console → Hosting → Add custom domain
- Follow DNS verification steps

---

## Phase 6: End-to-End Testing (30 minutes)

### Test Scenarios

1. **Human vs Engine (v18.4)**
   - Time control: 5+3 blitz
   - Play 3 moves, verify UCI communication
   - Check time countdown

2. **Engine vs Engine (v17.7 vs v18.4)**
   - Time control: 2+1 bullet
   - Watch full game to completion
   - Verify result detection (checkmate/draw)

3. **Concurrent Games**
   - Start 3 games simultaneously
   - Verify resource limits (max 3 containers)
   - Check `docker ps` for cleanup after games

4. **Error Handling**
   - Invalid move submission
   - WebSocket disconnect/reconnect
   - Engine crash simulation

### Monitoring

```bash
# On VM
sudo journalctl -u v7p3r-api -f  # Watch logs
docker ps  # Check running containers
docker stats  # Monitor resource usage
```

---

## Cost Breakdown

| Resource | Cost/Month |
|----------|------------|
| e2-micro VM (us-central1) | $0 (free tier) |
| Firebase Hosting | $0 (Spark plan) |
| GCR Storage (5 images ~500MB) | ~$0.02 |
| **Total** | **~$0** |

---

## Rollback Plan

If issues arise, revert to coming-soon page:

```powershell
cd v7p3r-labs
git checkout HEAD~1 index.html
npm run deploy
```

---

## Production Checklist

- [ ] VM created and Docker installed
- [ ] Docker images built and pushed to GCR
- [ ] Backend deployed as systemd service
- [ ] Frontend built and deployed to Firebase
- [ ] WebSocket connection tested
- [ ] All time controls validated
- [ ] Concurrent game limits verified
- [ ] Container cleanup confirmed
- [ ] DNS configured (v7p3r.labs.rtsts.tech)
- [ ] Monitoring setup (logs, metrics)

---

## Next Steps After Launch

1. **Analytics Integration**: Add Google Analytics to track usage
2. **Move History Display**: Show PGN/move list in UI
3. **Position Analysis**: Show engine evaluation scores
4. **Save/Resume Games**: Add game persistence
5. **Opening Book**: Display opening names
6. **Mobile Optimization**: Responsive chessboard
7. **Spectator Mode**: Share game links
8. **ELO Tracking**: Track engine performance over time

---

**Timeline Estimate**: 8-10 hours total for weekend sprint  
**Risk Level**: Low (isolated from production lichess-bot)  
**Deployment Window**: Anytime (no production impact)
