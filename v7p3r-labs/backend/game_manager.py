"""
Game Manager - Handles concurrent games and resource allocation
Implements concurrency limits like lichess-bot (max 3 games on e2-micro)
"""
import asyncio
import chess
import uuid
from typing import Optional, Dict
from datetime import datetime
import logging

from models import (
    GameConfig, GameState, GameStatus, GameResult, Move, 
    PlayerType, TimeControl
)
from uci_manager import UCIEngine

logger = logging.getLogger(__name__)


class GameManager:
    """Manages concurrent chess games with resource limits"""
    
    def __init__(self, max_concurrent_games: int = 3):
        self.max_concurrent_games = max_concurrent_games
        self.semaphore = asyncio.Semaphore(max_concurrent_games)
        self.active_games: Dict[str, 'ChessGame'] = {}
        
    async def create_game(self, config: GameConfig) -> str:
        """
        Create new game
        Returns game_id
        """
        game_id = str(uuid.uuid4())[:8]
        
        # Validate configuration
        config.validate_model()
        
        # Create game instance
        game = ChessGame(game_id, config, self.semaphore)
        self.active_games[game_id] = game
        
        logger.info(f"Created game {game_id}: {config.white.type} vs {config.black.type}")
        return game_id
    
    def get_game(self, game_id: str) -> Optional['ChessGame']:
        """Get game instance"""
        return self.active_games.get(game_id)
    
    async def start_game(self, game_id: str):
        """Start game (spawn engines if needed)"""
        game = self.get_game(game_id)
        if not game:
            raise ValueError(f"Game {game_id} not found")
        
        await game.start()
    
    async def cleanup_game(self, game_id: str):
        """Clean up finished game"""
        game = self.active_games.pop(game_id, None)
        if game:
            await game.cleanup()
            logger.info(f"Cleaned up game {game_id}")


class ChessGame:
    """Individual chess game with engine management"""
    
    def __init__(self, game_id: str, config: GameConfig, semaphore: asyncio.Semaphore):
        self.game_id = game_id
        self.config = config
        self.semaphore = semaphore
        
        self.board = chess.Board()
        self.moves: list[Move] = []
        self.status = GameStatus.WAITING
        self.result: Optional[GameResult] = None
        self.result_reason: Optional[str] = None
        
        # Time tracking (in milliseconds)
        self.white_time_ms = config.time_control.minutes * 60 * 1000
        self.black_time_ms = config.time_control.minutes * 60 * 1000
        self.last_move_time: Optional[datetime] = None
        
        # Engine instances
        self.white_engine: Optional[UCIEngine] = None
        self.black_engine: Optional[UCIEngine] = None
        
    async def start(self):
        """Start game and spawn engines"""
        async with self.semaphore:
            try:
                self.status = GameStatus.ACTIVE
                
                # Spawn white engine if needed
                if self.config.white.type == PlayerType.ENGINE:
                    if not self.config.white.engine_version:
                        raise ValueError("White engine player must have engine_version")
                    self.white_engine = UCIEngine(
                        self.config.white.engine_version, 
                        f"{self.game_id}-white"
                    )
                    await self.white_engine.start()
                
                # Spawn black engine if needed
                if self.config.black.type == PlayerType.ENGINE:
                    if not self.config.black.engine_version:
                        raise ValueError("Black engine player must have engine_version")
                    self.black_engine = UCIEngine(
                        self.config.black.engine_version,
                        f"{self.game_id}-black"
                    )
                    await self.black_engine.start()
                
                logger.info(f"Game {self.game_id} started")
                
            except Exception as e:
                logger.error(f"Error starting game {self.game_id}: {e}")
                self.status = GameStatus.ERROR
                await self.cleanup()
                raise
    
    async def make_move(self, uci_move: Optional[str] = None) -> Move:
        """
        Make a move (human or engine)
        Args:
            uci_move: UCI move string (for human moves), None for engine moves
        Returns:
            Move object with timing info
        """
        if self.status != GameStatus.ACTIVE:
            raise RuntimeError("Game not active")
        
        move_start = datetime.now()
        
        # Determine whose turn it is
        is_white_turn = self.board.turn == chess.WHITE
        current_engine = self.white_engine if is_white_turn else self.black_engine
        
        # Get move (from human input or engine)
        if uci_move:
            # Human move
            move_uci = uci_move
        else:
            # Engine move
            if not current_engine:
                raise RuntimeError("No engine available for current player")
            
            move_uci = await current_engine.get_move(
                self.board.fen(),
                self.white_time_ms,
                self.black_time_ms,
                self.config.time_control.increment * 1000,
                self.config.time_control.increment * 1000
            )
            
            if not move_uci:
                raise RuntimeError("Engine failed to return move")
        
        # Validate and apply move
        try:
            chess_move = chess.Move.from_uci(move_uci)
            if chess_move not in self.board.legal_moves:
                raise ValueError(f"Illegal move: {move_uci}")
            
            self.board.push(chess_move)
            
        except Exception as e:
            logger.error(f"Invalid move {move_uci}: {e}")
            raise
        
        # Calculate time spent
        move_end = datetime.now()
        time_spent_ms = int((move_end - move_start).total_seconds() * 1000)
        
        # Update time remaining
        if is_white_turn:
            self.white_time_ms -= time_spent_ms
            self.white_time_ms += self.config.time_control.increment * 1000
        else:
            self.black_time_ms -= time_spent_ms
            self.black_time_ms += self.config.time_control.increment * 1000
        
        # Create move object
        move = Move(
            uci=move_uci,
            san=self.board.san(chess_move),
            time_spent_ms=time_spent_ms
        )
        self.moves.append(move)
        
        # Check game end conditions
        self._check_game_end()
        
        return move
    
    def _check_game_end(self):
        """Check if game has ended"""
        # Checkmate
        if self.board.is_checkmate():
            self.status = GameStatus.FINISHED
            self.result = GameResult.BLACK_WINS if self.board.turn == chess.WHITE else GameResult.WHITE_WINS
            self.result_reason = "checkmate"
        
        # Stalemate
        elif self.board.is_stalemate():
            self.status = GameStatus.FINISHED
            self.result = GameResult.DRAW
            self.result_reason = "stalemate"
        
        # Insufficient material
        elif self.board.is_insufficient_material():
            self.status = GameStatus.FINISHED
            self.result = GameResult.DRAW
            self.result_reason = "insufficient_material"
        
        # Threefold repetition
        elif self.board.is_repetition(3):
            self.status = GameStatus.FINISHED
            self.result = GameResult.DRAW
            self.result_reason = "threefold_repetition"
        
        # Fifty move rule
        elif self.board.is_fifty_moves():
            self.status = GameStatus.FINISHED
            self.result = GameResult.DRAW
            self.result_reason = "fifty_move_rule"
        
        # Time forfeit
        elif self.white_time_ms <= 0:
            self.status = GameStatus.FINISHED
            self.result = GameResult.BLACK_WINS
            self.result_reason = "white_time_forfeit"
        
        elif self.black_time_ms <= 0:
            self.status = GameStatus.FINISHED
            self.result = GameResult.WHITE_WINS
            self.result_reason = "black_time_forfeit"
    
    def get_state(self) -> GameState:
        """Get current game state"""
        return GameState(
            game_id=self.game_id,
            status=self.status,
            fen=self.board.fen(),
            moves=self.moves,
            result=self.result,
            result_reason=self.result_reason,
            white_time_ms=self.white_time_ms,
            black_time_ms=self.black_time_ms
        )
    
    async def cleanup(self):
        """Stop engines and clean up resources"""
        if self.white_engine:
            await self.white_engine.stop()
        if self.black_engine:
            await self.black_engine.stop()
        
        logger.info(f"Game {self.game_id} cleaned up")
