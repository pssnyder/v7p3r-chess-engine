"""
v7p3r-labs FastAPI Backend
WebSocket-based chess engine server
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import logging
import json

from models import GameConfig, WebSocketMessage, PlayerType, EngineVersion
from game_manager import GameManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global game manager
game_manager: Optional[GameManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    global game_manager
    
    # Startup
    logger.info("Starting v7p3r-labs API server")
    game_manager = GameManager(max_concurrent_games=3)
    
    yield
    
    # Shutdown
    logger.info("Shutting down v7p3r-labs API server")
    # Clean up all active games
    for game_id in list(game_manager.active_games.keys()):
        await game_manager.cleanup_game(game_id)


app = FastAPI(
    title="v7p3r-labs API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (allow frontend origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict to v7p3r.labs.rtsts.tech in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    if not game_manager:
        return {"service": "v7p3r-labs-api", "status": "initializing"}
    return {
        "service": "v7p3r-labs-api",
        "status": "online",
        "active_games": len(game_manager.active_games)
    }


@app.get("/engines")
async def list_engines():
    """List available engine versions"""
    return {
        "engines": [
            {"version": v.value, "name": f"V7P3R {v.value}"} 
            for v in EngineVersion
        ]
    }


@app.post("/games")
async def create_game(config: GameConfig):
    """Create new game"""
    if not game_manager:
        raise HTTPException(status_code=503, detail="Server not ready")
    try:
        game_id = await game_manager.create_game(config)
        return {"game_id": game_id}
    except Exception as e:
        logger.error(f"Error creating game: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws/game/{game_id}")
async def game_websocket(websocket: WebSocket, game_id: str):
    """
    WebSocket endpoint for real-time game communication
    
    Message format:
        Client -> Server: {"type": "move", "data": {"uci": "e2e4"}}
        Server -> Client: {"type": "game_update", "data": {...GameState}}
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for game {game_id}")
    
    if not game_manager:
        await websocket.send_json({
            "type": "error",
            "data": {"message": "Server not ready"}
        })
        await websocket.close()
        return
    
    try:
        game = game_manager.get_game(game_id)
        if not game:
            await websocket.send_json({
                "type": "error",
                "data": {"message": f"Game {game_id} not found"}
            })
            await websocket.close()
            return
        
        # Start game (spawn engines)
        await game_manager.start_game(game_id)
        
        # Send initial game state
        await websocket.send_json({
            "type": "game_update",
            "data": game.get_state().dict()
        })
        
        # Game loop
        while game.status.value == "active":
            # Check whose turn it is
            is_white_turn = game.board.turn
            current_player = game.config.white if is_white_turn else game.config.black
            
            if current_player.type == PlayerType.HUMAN:
                # Wait for human move via WebSocket
                try:
                    message = await websocket.receive_json()
                    
                    if message.get("type") == "move":
                        uci_move = message.get("data", {}).get("uci")
                        if not uci_move:
                            raise ValueError("Missing UCI move")
                        
                        # Make human move
                        move = await game.make_move(uci_move)
                        
                        # Send updated state
                        await websocket.send_json({
                            "type": "game_update",
                            "data": game.get_state().dict()
                        })
                        
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected for game {game_id}")
                    break
                    
            else:
                # Engine move (autonomous)
                move = await game.make_move()
                
                # Send updated state
                await websocket.send_json({
                    "type": "game_update",
                    "data": game.get_state().dict()
                })
        
        # Game finished
        if game.status.value == "finished":
            await websocket.send_json({
                "type": "game_over",
                "data": {
                    "result": game.result.value if game.result else None,
                    "reason": game.result_reason
                }
            })
        
    except Exception as e:
        logger.error(f"Error in game {game_id}: {e}")
        await websocket.send_json({
            "type": "error",
            "data": {"message": str(e)}
        })
    
    finally:
        # Clean up game
        await game_manager.cleanup_game(game_id)
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
