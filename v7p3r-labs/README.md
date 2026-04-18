# V7P3R Chess Engine - RTSLabs Implementation

## Project Overview

A web-based chess interface hosted at **v7p3r.labs.rtsts.tech** that allows users to play against the V7P3R chess engine or watch historical engine versions compete against each other. This project serves as both a portfolio demonstration and a learning platform for containerized resource management.

## Project Goals

### Core Features
1. **Interactive Chess Interface**
   - Clean, modern web UI with custom-styled chessboard
   - Custom chess piece images (provided in `/images/`)
   - Responsive design for desktop and mobile play

2. **Game Modes**
   - **Human vs Engine**: Play against the latest V7P3R version
   - **Engine vs Engine**: Watch historical versions compete
   - Selectable players via dropdown: "Human" or "V7P3R v#.#"

3. **Time Controls**
   - 2 minutes + 1 second increment (bullet)
   - 5 minutes + 3 second increment (blitz)
   - 10 minutes + 0 second increment (rapid)
   - 10 minutes + 5 second increment (rapid+)
   - Maximum game duration: 10 minutes with appropriate increments

4. **Engine Versions Supported**
   - v12.6 (Final v12 series)
   - v14.1 (Stable v14 series, 25-day deployment)
   - v16.1 (v16 series)
   - v17.7 (Stable v17 series, 4+ day deployment)
   - v18.4 (Latest v18 series)

## Hosting Infrastructure

**Firebase Hosting**
- Project: `rts-labs-f3981`
- Hosting Site: `v7p3r-chess-engine-f3981.web.app`
- Custom Domain: `v7p3r.labs.rtsts.tech`

## Architecture Options

### Option 1: Lichess Bot Integration (Simpler)
**Approach**: Piggyback off existing containerized lichess bot infrastructure

**Pros**:
- Leverages existing VM and container setup
- Engine already running in production environment
- Minimal new infrastructure needed
- Proven stability and performance

**Cons**:
- Limited to currently deployed engine version
- Requires API calls to external lichess bot instance
- Cannot easily run historical engine matchups
- Dependent on lichess bot availability

**Implementation**:
- Frontend sends game requests to GCP VM endpoint
- Lichess bot container handles UCI communication
- Results streamed back to web interface
- Challenge current V7P3R version via internal API

---

### Option 2: On-Demand Containerization (Learning-Focused)
**Approach**: Containerized engine instances spun up per game session

**Pros**:
- Full control over engine versions (v12.6, v14.1, v16.1, v17.7, v18.4)
- Engine vs Engine matchups possible
- Excellent learning platform for resource management
- Scalable architecture for future expansion
- Isolated game sessions (security & stability)

**Cons**:
- More complex infrastructure setup
- Container orchestration required (Docker/Kubernetes)
- Cold start latency (container spin-up time)
- Higher resource management overhead
- Cost considerations for compute resources

**Implementation**:
- Frontend requests game → Backend API receives request
- Docker container spins up with selected engine version
- UCI interface bridged to web frontend via WebSocket
- Game concludes → Container destroyed, resources freed
- Firebase Functions or GCP Cloud Run for orchestration

**Technology Stack**:
- **Containers**: Docker with pre-built engine images
- **Orchestration**: Cloud Run (serverless) or GKE (Kubernetes)
- **Communication**: WebSocket for real-time move updates
- **Backend**: Python Flask/FastAPI or Node.js Express
- **Frontend**: React/Vue.js with chessboard.js library

## Technical Components

### Frontend (v7p3r-labs/)
- **Framework**: React or Vue.js (TBD)
- **Chess Library**: chessboard.js or react-chessboard
- **Styling**: Custom CSS with provided board/piece assets
- **Communication**: WebSocket client for real-time updates

### Backend/API
- **Language**: Python (FastAPI) or Node.js (Express)
- **Engine Interface**: UCI protocol communication
- **Container Management**: Docker SDK (Option 2)
- **Game State**: In-memory or Redis for active games

### Engine Versions
Each version containerized with:
- Python 3.12+ runtime
- V7P3R engine files (from `/lichess/engines/V7P3R_v*/`)
- UCI interface (`v7p3r_uci.py`)
- Minimal dependencies (python-chess)

## Development Phases

### Phase 1: Foundation (MVP)
- [ ] Setup Firebase hosting configuration
- [ ] Create basic web UI with chessboard
- [ ] Implement player selection dropdowns
- [ ] Implement time control selection
- [ ] Test custom piece/board rendering

### Phase 2: Backend Architecture Decision
- [ ] Evaluate Option 1 vs Option 2 trade-offs
- [ ] Document chosen architecture with rationale
- [ ] Setup development environment (local Docker testing)
- [ ] Create API specification (endpoints, WebSocket protocol)

### Phase 3: Engine Integration
- [ ] Containerize engine versions (Option 2) OR Setup lichess bot proxy (Option 1)
- [ ] Implement UCI bridge to web interface
- [ ] Test move submission and engine response
- [ ] Implement time control enforcement

### Phase 4: Game Flow
- [ ] Start game → Initialize engine(s)
- [ ] Move validation and submission
- [ ] Real-time move updates (WebSocket)
- [ ] Game conclusion detection (checkmate, draw, timeout)
- [ ] Resource cleanup (container shutdown if Option 2)

### Phase 5: Polish & Deployment
- [ ] Mobile responsive design
- [ ] Game history/PGN export
- [ ] Error handling and user feedback
- [ ] Performance optimization
- [ ] Production deployment to Firebase
- [ ] Custom domain configuration (v7p3r.labs.rtsts.tech)

## Learning Objectives

1. **Resource Management**: Understand container lifecycle, spin-up/down timing, resource allocation
2. **Real-Time Communication**: WebSocket implementation for bidirectional game updates
3. **Chess Engine Integration**: UCI protocol, move validation, time management
4. **Serverless/Containerized Architectures**: Cloud Run, Firebase Functions, Docker orchestration
5. **Frontend/Backend Integration**: API design, state management, error handling

## Assets

### Custom Chess Pieces
Located in `/images/`:
- Black pieces: `bB.png`, `bK.png`, `bN.png`, `bp.png`, `bQ.png`, `bR.png`
- White pieces: `wB.png`, `wK.png`, `wN.png`, `wp.png`, `wQ.png`, `wR.png`
- Board theme: `chess_board_theme.jpg`

## Next Steps

1. **Decision**: Choose architecture option (Option 1 or Option 2)
2. **Prototype**: Build simple chess UI with custom pieces
3. **Backend Setup**: Implement chosen engine integration approach
4. **Testing**: Local development testing with single engine version
5. **Iterate**: Expand to multiple versions and full feature set

---

**Project Status**: Planning Phase  
**Last Updated**: April 17, 2026  
**Portfolio Site**: https://labs.rtsts.tech  
**Target Domain**: https://v7p3r.labs.rtsts.tech
