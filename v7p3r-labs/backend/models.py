"""
Data models for v7p3r-labs chess engine API
"""
from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel


class EngineVersion(str, Enum):
    """Available engine versions"""
    V12_6 = "12.6"
    V14_1 = "14.1"
    V16_1 = "16.1"
    V17_7 = "17.7"
    V18_4 = "18.4"


class TimeControl(BaseModel):
    """Time control configuration"""
    minutes: int
    increment: int
    
    @property
    def display(self) -> str:
        return f"{self.minutes}+{self.increment}"


class PlayerType(str, Enum):
    """Player type: human or engine"""
    HUMAN = "human"
    ENGINE = "engine"


class Player(BaseModel):
    """Player configuration"""
    type: PlayerType
    engine_version: Optional[EngineVersion] = None
    
    def validate_model(self):
        if self.type == PlayerType.ENGINE and not self.engine_version:
            raise ValueError("Engine player must have engine_version")


class GameConfig(BaseModel):
    """Game configuration"""
    white: Player
    black: Player
    time_control: TimeControl
    
    def validate_model(self):
        self.white.validate_model()
        self.black.validate_model()


class Move(BaseModel):
    """Chess move in UCI format"""
    uci: str  # e.g., "e2e4"
    san: Optional[str] = None  # e.g., "e4"
    time_spent_ms: Optional[int] = None


class GameStatus(str, Enum):
    """Game status"""
    WAITING = "waiting"
    ACTIVE = "active"
    FINISHED = "finished"
    ERROR = "error"


class GameResult(str, Enum):
    """Game result"""
    WHITE_WINS = "white_wins"
    BLACK_WINS = "black_wins"
    DRAW = "draw"
    ONGOING = "ongoing"


class GameState(BaseModel):
    """Current game state"""
    game_id: str
    status: GameStatus
    fen: str
    moves: list[Move]
    result: Optional[GameResult] = None
    result_reason: Optional[str] = None
    white_time_ms: Optional[int] = None
    black_time_ms: Optional[int] = None


class WebSocketMessage(BaseModel):
    """WebSocket message envelope"""
    type: Literal["game_update", "move", "error", "game_over", "chat"]
    data: dict
