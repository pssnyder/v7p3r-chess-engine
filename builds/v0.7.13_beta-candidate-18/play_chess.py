# play_chess.py

import os
import pygame
import chess
import chess.pgn
import datetime
import time
from typing import Optional, Dict, Any
from io import StringIO
from v7p3r_config import v7p3rConfig
from v7p3r_utilities import get_timestamp
from metrics import get_metrics_instance, GameMetric, MoveMetric
from chess_core import ChessCore
from v7p3r_engine import v7p3rEngine
from stockfish_handler import StockfishHandler
import asyncio

CONFIG_NAME = "default_config"
MAX_FPS = 60

class playChess:
    def __init__(self, config_name: Optional[str] = None):
        """Initialize the chess game with configuration."""
        # Initialize Pygame
        pygame.init()
        self.clock = pygame.time.Clock()
        
        try:
            # Load configuration
            self.config = v7p3rConfig(config_path=config_name if config_name else None)
            self.game_config = self.config.get_game_config()
            self.engine_config = self.config.get_engine_config()
            self.stockfish_config = dict(self.config.get_stockfish_config())
            
            # Game configuration
            self.white_player = self.game_config.get('white_player', 'v7p3r')
            self.black_player = self.game_config.get('black_player', 'stockfish')
            self.starting_position = self.game_config.get('starting_position', 'default')
            
            # Time control
            self.time_control = self.game_config.get('time_control', False)
            self.game_time = self.game_config.get('game_time', 300)
            self.time_increment = self.game_config.get('time_increment', 0)
            
            # Game state
            self.headless = self.game_config.get('headless', True)
            self.metrics_enabled = self.game_config.get('record_metrics', True)
            self.current_player = chess.WHITE
            self.game_count = self.game_config.get('game_count', 1)
            self.current_game_id = get_timestamp()
            self.game_start_time = time.time()
            self.current_eval = 0.0
            self.move_duration = 0
            
            # Initialize components
            self.chess_core = ChessCore()
            self.engine = v7p3rEngine(self.engine_config)
            self.stockfish = StockfishHandler(self.stockfish_config)
            self.metrics = get_metrics_instance()
            
        except Exception as e:
            print(f"Error initializing game: {str(e)}")
            raise

    async def _setup_game(self):
        """Set up a new game."""
        # Initialize chess core with starting position
        self.chess_core.new_game(self.starting_position)
        
        # Set up initial board state
        self.board = self.chess_core.board
        self.current_player = chess.WHITE
        self.game_start_time = time.time()
        
        # Initialize engines if needed
        if self.stockfish and self.black_player == 'stockfish':
            await self.stockfish.initialize()
            
        # Record game start metrics
        if self.metrics_enabled:
            game_metric = GameMetric(
                game_id=self.current_game_id,
                timestamp=get_timestamp(),
                v7p3r_color='white' if self.white_player == 'v7p3r' else 'black',
                opponent='stockfish' if self.white_player == 'v7p3r' else 'v7p3r',
                result='*',
                total_moves=0,
                game_duration=0.0,
                time_control_enabled=self.time_control,
                game_time=self.game_time,
                increment=self.time_increment,
                opening_name=None,
                final_position_fen=None,
                termination_reason=None
            )
            self.metrics.add_game(game_metric)
            
    async def _get_engine_move(self) -> Optional[chess.Move]:
        """Get a move from the appropriate engine."""
        start_time = time.time()
        
        # Determine which engine to use
        is_white = self.current_player == chess.WHITE
        player = self.white_player if is_white else self.black_player
        
        # Print thinking message
        print(f"\n{player.upper()} is thinking...")
        
        # Get move from appropriate engine
        move = None
        if player == 'v7p3r':
            move = self.engine.get_move(self.board)
        elif player == 'stockfish':
            move = await self.stockfish.get_move(self.board)
            
        duration = time.time() - start_time
            
        # Record move metrics
        if self.metrics_enabled and move:
            move_metric = MoveMetric(
                game_id=self.current_game_id,
                move_number=len(self.board.move_stack) + 1,
                player=player,
                move_notation=move.uci(),
                position_fen=self.board.fen(),
                evaluation_score=self.current_eval,
                search_depth=self.engine_config.get('depth', 6),
                nodes_evaluated=None,
                time_taken=duration,
                best_move=move.uci(),
                pv_line=None,
                quiescence_nodes=None,
                transposition_hits=None,
                move_ordering_efficiency=None,
                remaining_time=self.game_time if self.time_control else None,
                time_control_enabled=self.time_control,
                increment=self.time_increment
            )
            self.metrics.add_move(move_metric)
            
            # Print move info
            print(f"{player.upper()} plays: {self.board.san(move)} (eval: {self.current_eval:.2f}, time: {duration:.2f}s)")
            print(f"Position: {self.board.fen()}")
            
            # Write PGN after each move
            game = chess.pgn.Game()
            game.headers["Event"] = "V7P3R vs Stockfish"
            game.headers["Site"] = "Local"
            game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
            game.headers["Round"] = "1"
            game.headers["White"] = self.white_player
            game.headers["Black"] = self.black_player
            
            # Add all moves
            node = game
            for m in self.board.move_stack[:-1]:  # Exclude the move we just made
                node = node.add_variation(m)
                
            # Add the latest move
            if move:
                node = node.add_variation(move)
                
            # Write to active_game.pgn
            with open("active_game.pgn", "w") as f:
                print(game, file=f, end="\n\n")
            
        return move

    async def run(self):
        """Run the chess game."""
        try:
            print("\nStarting new game...")
            print(f"White: {self.white_player.upper()}")
            print(f"Black: {self.black_player.upper()}\n")
            
            # Setup initial game state
            await self._setup_game()
            
            # Game loop
            while not self.board.is_game_over():
                # Get move from current player
                move = await self._get_engine_move()
                
                # Apply move if valid
                if move and self.board.is_legal(move):
                    self.board.push(move)
                    
                    # Switch players
                    self.current_player = not self.current_player
                else:
                    print(f"Invalid move from {'White' if self.current_player else 'Black'}")
                    break
                    
                # Update display at reasonable FPS
                self.clock.tick(MAX_FPS)
                
            # Announce game over
            outcome = self.board.outcome()
            if outcome:
                winner = "White" if outcome.winner == chess.WHITE else "Black"
                print(f"\nGame Over! {winner} wins by {outcome.termination}")
                print(f"Final position: {self.board.fen()}")
            else:
                print("\nGame Over! Draw")
                
            # Record game end metrics
            if self.metrics_enabled:
                game_duration = time.time() - self.game_start_time
                game_metric = GameMetric(
                    game_id=self.current_game_id,
                    timestamp=get_timestamp(),
                    v7p3r_color='white' if self.white_player == 'v7p3r' else 'black',
                    opponent='stockfish' if self.white_player == 'v7p3r' else 'v7p3r',
                    result=self.board.result(),
                    total_moves=len(self.board.move_stack),
                    game_duration=game_duration,
                    time_control_enabled=self.time_control,
                    game_time=self.game_time,
                    increment=self.time_increment,
                    opening_name=None,
                    final_position_fen=self.board.fen(),
                    termination_reason=str(outcome.termination) if outcome else None
                )
                self.metrics.add_game(game_metric)
                
        except Exception as e:
            print(f"Error during game: {str(e)}")
            raise
        finally:
            # Cleanup
            if self.stockfish:
                await self.stockfish.quit()

async def main():
    game = playChess()
    await game.run()

if __name__ == "__main__":
    asyncio.run(main())
