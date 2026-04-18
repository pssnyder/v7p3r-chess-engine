# v7p3r-labs Backend

FastAPI WebSocket server for V7P3R chess engine interface.

## Architecture

- **FastAPI**: REST + WebSocket endpoints
- **Docker**: Ephemeral containers for each engine instance
- **UCI Protocol**: Chess engine communication via stdin/stdout
- **Asyncio**: Concurrent game management with resource limits

## Components

### `main.py`
FastAPI application with WebSocket game loop:
- `POST /games` - Create new game
- `GET /engines` - List available versions
- `WS /ws/game/{game_id}` - Real-time game communication

### `uci_manager.py`
Docker container lifecycle + UCI communication:
- Spawns engine in Docker container (`--rm` for auto-cleanup)
- Manages UCI protocol handshake (`uci` → `uciok`)
- Sends position/time and receives bestmove

### `game_manager.py`
Concurrent game orchestration:
- Limits max concurrent games (default: 3 for e2-micro)
- Manages chess board state (python-chess library)
- Time control tracking with increment
- Game end detection (checkmate, stalemate, time forfeit, etc.)

### `models.py`
Pydantic data models:
- `GameConfig` - Player types, engine versions, time control
- `GameState` - FEN, moves, status, result
- `WebSocketMessage` - Message envelope for client/server

## Deployment

### Local Testing
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Production (Systemd Service)
See `../scripts/setup-backend.sh` for automated deployment.

**Service Management**:
```bash
sudo systemctl start v7p3r-api
sudo systemctl stop v7p3r-api
sudo systemctl restart v7p3r-api
sudo journalctl -u v7p3r-api -f  # View logs
```

## Resource Limits

Per-engine Docker container:
- **Memory**: 256MB
- **CPU**: 0.3 vCPU
- **Lifecycle**: Ephemeral (auto-removed after game)

VM capacity (e2-micro: 1GB RAM, 2 vCPU):
- **Max concurrent games**: 3
- **Overhead**: ~200MB for system + API server

## Engine Versions

Docker images from Google Container Registry:
- `gcr.io/rts-labs-f3981/v7p3r:12.6`
- `gcr.io/rts-labs-f3981/v7p3r:14.1`
- `gcr.io/rts-labs-f3981/v7p3r:16.1`
- `gcr.io/rts-labs-f3981/v7p3r:17.7`
- `gcr.io/rts-labs-f3981/v7p3r:18.4`

## WebSocket Protocol

### Client → Server
```json
{
  "type": "move",
  "data": {"uci": "e2e4"}
}
```

### Server → Client
```json
{
  "type": "game_update",
  "data": {
    "game_id": "abc123",
    "status": "active",
    "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "moves": [{"uci": "e2e4", "san": "e4", "time_spent_ms": 1234}],
    "white_time_ms": 298766,
    "black_time_ms": 300000
  }
}
```

## Logging

Structured logging with levels:
- **INFO**: Game lifecycle (create, start, end)
- **DEBUG**: UCI commands and responses (only in development)
- **ERROR**: Engine failures, invalid moves

Access logs via systemd:
```bash
sudo journalctl -u v7p3r-api -f
```

## Troubleshooting

**Container not starting**:
```bash
docker logs <container_name>
docker inspect <container_name>
```

**Engine not responding**:
- Check UCI communication in logs
- Verify Docker image exists: `docker images | grep v7p3r`
- Test manual UCI: `docker run -it gcr.io/rts-labs-f3981/v7p3r:18.4`

**Memory issues**:
```bash
docker stats  # Monitor container resource usage
free -h       # Check VM memory
```

## Future Enhancements

- [ ] Engine evaluation scores in WebSocket updates
- [ ] Opening book integration
- [ ] Position analysis endpoint
- [ ] PGN export/import
- [ ] Game persistence (database)
- [ ] Prometheus metrics
