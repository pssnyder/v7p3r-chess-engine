# metrics.py

"""V7P3R Chess Engine Metrics System
A simple in-memory metrics collection system for the V7P3R chess engine.
"""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from datetime import datetime
import json

@dataclass
class MoveMetric:
    """Data structure for move-level metrics"""
    game_id: str
    move_number: int
    player: str
    move_notation: str
    position_fen: str
    evaluation_score: float
    search_depth: Optional[int]
    nodes_evaluated: Optional[int]
    time_taken: float
    best_move: str
    pv_line: Optional[List[str]] = None
    quiescence_nodes: Optional[int] = None
    transposition_hits: Optional[int] = None
    move_ordering_efficiency: Optional[float] = None
    remaining_time: Optional[float] = None
    time_control_enabled: bool = False
    increment: float = 0.0

@dataclass
class GameMetric:
    """Data structure for game-level metrics"""
    game_id: str
    timestamp: str
    v7p3r_color: str
    opponent: str
    result: str
    total_moves: int
    game_duration: float
    time_control_enabled: bool = False
    game_time: float = 0.0
    increment: float = 0.0
    opening_name: Optional[str] = None
    final_position_fen: Optional[str] = None
    termination_reason: Optional[str] = None

class v7p3rMetrics:
    """Simple in-memory metrics system for V7P3R chess engine."""
    
    def __init__(self):
        """Initialize metrics system."""
        self.games: Dict[str, GameMetric] = {}
        self.moves: Dict[str, List[MoveMetric]] = {}
    
    def add_game(self, game: GameMetric):
        """Record a game.
        
        Args:
            game: Game metrics to record
        """
        self.games[game.game_id] = game
        self.moves[game.game_id] = []
        self._save_to_file(game.game_id)
    
    def add_move(self, move: MoveMetric):
        """Record a move.
        
        Args:
            move: Move metrics to record
        """
        if move.game_id not in self.moves:
            self.moves[move.game_id] = []
        self.moves[move.game_id].append(move)
        self._save_to_file(move.game_id)
    
    def _save_to_file(self, game_id: str):
        """Save metrics to file for a game.
        
        Args:
            game_id: ID of game to save
        """
        game = self.games.get(game_id)
        if not game:
            return
            
        data = {
            'game': asdict(game),
            'moves': [asdict(m) for m in self.moves[game_id]]
        }
        
        # Save to file with game ID
        filename = f"metrics/metrics_{game_id}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

def get_metrics_instance() -> v7p3rMetrics:
    """Get a metrics system instance.
    
    Returns:
        v7p3rMetrics: Metrics system instance
    """
    return v7p3rMetrics()