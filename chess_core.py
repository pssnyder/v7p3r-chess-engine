# chess_core.py
""" Core Components for python chess.
Generic chess functionality independent of specific engines.
"""
import os
import chess
import chess.pgn
import chess.engine
import datetime
import socket
import time
from typing import Optional
from io import StringIO
from v7p3r_debug import v7p3rLogger, v7p3rUtilities

class ChessCore:
    def __init__(self, logger_name: str = "chess_core"):
        """Initialize core chess components"""
        # Setup logging
        self.logger = v7p3rLogger.setup_logger(logger_name)
        
        # Initialize core chess state
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.game_node = self.game
        self.selected_square = chess.SQUARES[0]
        self.last_move = chess.Move.null()
        self.move_history = []
        self.game_start_time = 0
        self.game_start_timestamp = ""
        
        # Game state tracking
        self.current_player = self.board.turn
        
        if self.logger:
            self.logger.info("ChessCore initialized")

    def new_game(self, starting_position: str = "default"):
        """Reset the game state for a new game"""
        self.board = chess.Board()
        if starting_position != "default":
            self.board.set_fen(starting_position)
        self.game = chess.pgn.Game()
        self.game_node = self.game
        self.selected_square = chess.SQUARES[0]
        self.last_move = chess.Move.null()
        self.current_player = self.board.turn
        self.move_history = []
        self.game_start_time = time.time()
        self.game_start_timestamp = v7p3rUtilities.get_timestamp()

        # Set default PGN headers
        self.set_headers()
        
        if self.logger:
            self.logger.info(f"New game started with position: {starting_position}")

    def set_headers(self, white_player: str = "White", black_player: str = "Black", event: str = "Chess Game"):
        """Set PGN headers for the game"""
        self.game.headers["Event"] = event
        self.game.headers["White"] = white_player
        self.game.headers["Black"] = black_player
        self.game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        self.game.headers["Site"] = socket.gethostbyname(socket.gethostname())
        self.game.headers["Round"] = "#"

    def get_board_result(self) -> str:
        """Return the result string for the current board state."""
        # Explicitly handle all draw and win/loss cases, fallback to "*"
        if self.board.is_checkmate():
            # The side to move is checkmated, so the other side wins
            return "1-0" if self.board.turn == chess.BLACK else "0-1"
        # Explicit draw conditions
        if (
            self.board.is_stalemate()
            or self.board.is_insufficient_material()
            or self.board.can_claim_fifty_moves()
            or self.board.can_claim_threefold_repetition()
            or self.board.is_seventyfive_moves()
            or self.board.is_fivefold_repetition()
            or self.board.is_variant_draw()
        ):
            return "1/2-1/2"
        # If the game is over but not by checkmate or above draws, fallback to chess.Board.result()
        if self.board.is_game_over():
            result = self.board.result()
            # Defensive: If result is not a valid string, force draw string
            if result not in ("1-0", "0-1", "1/2-1/2"):
                return "1/2-1/2"
            return result
        return "*"

    def handle_game_end(self) -> bool:
        """Check if the game is over and handle end conditions."""
        if self.board.is_game_over():
            # Ensure the result is set in the PGN headers and game node
            result = self.get_board_result()
            self.game.headers["Result"] = result
            self.game_node = self.game.end()
            if self.logger:
                self.logger.info(f"Game ended with result: {result}")
            return True
        return False

    def push_move(self, move) -> bool:
        """Test and push a move to the board and game node"""
        if not self.board.is_valid():
            if self.logger:
                self.logger.error(f"Invalid board state detected! | FEN: {self.board.fen()}")
            return False
        
        if self.logger:
            self.logger.debug(f"Attempting to push move: {move} | FEN: {self.board.fen()}")

        if isinstance(move, str):
            try:
                move = chess.Move.from_uci(move)
                if self.logger:
                    self.logger.debug(f"Converted UCI string to chess.Move: {move}")
            except ValueError:
                if self.logger:
                    self.logger.error(f"Invalid UCI string received: {move}")
                return False
        
        if not self.board.is_legal(move):
            if self.logger:
                self.logger.error(f"Illegal move blocked: {move}")
            return False
        
        try:
            if self.logger:
                self.logger.debug(f"Pushing move: {move} | FEN: {self.board.fen()}")
            
            self.board.push(move)
            self.game_node = self.game_node.add_variation(move)
            self.last_move = move
            self.move_history.append(move)
            
            if self.logger:
                self.logger.debug(f"Move pushed successfully: {move} | FEN: {self.board.fen()}")
            
            self.current_player = self.board.turn
            
            # If the move ends the game, set the result header and end the game node
            if self.board.is_game_over():
                result = self.get_board_result()
                self.game.headers["Result"] = result
                self.game_node = self.game.end()
            else:
                self.game.headers["Result"] = "*"
            
            return True
        except ValueError as e:
            if self.logger:
                self.logger.error(f"ValueError pushing move {move}: {e}")
            return False
        except Exception as e:
            if self.logger:
                self.logger.error(f"Exception pushing move {move}: {e}")
            return False

    def import_fen(self, fen_string: str) -> bool:
        """Import a position from FEN notation"""
        try:
            new_board = chess.Board(fen_string)
            
            if not new_board.is_valid():
                if self.logger:
                    self.logger.error(f"Invalid FEN position: {fen_string}")
                return False

            self.board = new_board
            self.game = chess.pgn.Game()
            self.game.setup(new_board)
            self.game_node = self.game            
            self.selected_square = None
            self.game.headers["FEN"] = fen_string

            if self.logger:
                self.logger.info(f"Successfully imported FEN: {fen_string}")
            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected problem importing FEN: {e}")
            return False

    def quick_save_pgn(self, filename: str) -> bool:
        """Save PGN to local file."""
        try:            
            with open(filename, 'w', encoding='utf-8') as f:
                # Get PGN text
                buf = StringIO()
                exporter = chess.pgn.FileExporter(buf)
                self.game.accept(exporter)
                f.write(buf.getvalue())
            
            if self.logger:
                self.logger.debug(f"PGN saved to {filename}")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save PGN to {filename}: {e}")
            return False

    def quick_save_pgn_to_file(self, filename: str):
        """Quick save the current game to a PGN file"""
        # Inject or update Result header so the PGN shows the game outcome
        if self.board.is_game_over():
            self.game.headers["Result"] = self.get_board_result()
            self.game_node = self.game.end()
        else:
            self.game.headers["Result"] = "*"
        
        with open(filename, "w") as f:
            exporter = chess.pgn.FileExporter(f)
            self.game.accept(exporter)

    def get_pgn_text(self) -> str:
        """Get the PGN text for the current game"""
        buf = StringIO()
        exporter = chess.pgn.FileExporter(buf)
        self.game.accept(exporter)
        return buf.getvalue()

    def save_local_game_files(self, game_id: str, additional_data: Optional[dict] = None) -> bool:
        """Save both PGN and JSON config files locally for metrics processing."""
        try:
            import json
            
            # Ensure games directory exists
            os.makedirs("games", exist_ok=True)
            
            # Save PGN file
            pgn_path = f"games/{game_id}.pgn"
            pgn_text = self.get_pgn_text()
            with open(pgn_path, 'w', encoding='utf-8') as f:
                f.write(pgn_text)
            
            # Save JSON config file for metrics processing
            json_path = f"games/{game_id}.json"
            config_data = {
                'game_id': game_id,
                'timestamp': self.game_start_timestamp,
                'total_moves': len(self.move_history),
                'game_duration': time.time() - self.game_start_time,
                'final_result': self.get_board_result(),
                'final_fen': self.board.fen()
            }
            
            # Add any additional data provided
            if additional_data:
                config_data.update(additional_data)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4)
            
            if self.logger:
                self.logger.info(f"Saved local files: {pgn_path}, {json_path}")
            return True
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save local game files for {game_id}: {e}")
            return False

    def _format_time_for_display(self, move_time: float) -> str:
        """
        Format move time for display with appropriate units.
        
        Args:
            move_time: Time in seconds (stored with high precision)
            
        Returns:
            Formatted time string with appropriate units (s or ms)
        """
        if move_time <= 0:
            return "0.000s"
        
        # If time is less than 0.1 seconds (100ms), display in milliseconds
        if move_time < 0.1:
            time_ms = move_time * 1000
            if time_ms < 1.0:
                # For very fast moves (sub-millisecond), show microseconds with higher precision
                return f"{time_ms:.3f}ms"
            else:
                # For sub-100ms moves, show milliseconds with 1 decimal place
                return f"{time_ms:.1f}ms"
        else:
            # For moves 100ms and above, display in seconds
            if move_time < 1.0:
                # Sub-second but >= 100ms: show 3 decimal places
                return f"{move_time:.3f}s"
            elif move_time < 10.0:
                # 1-10 seconds: show 2 decimal places
                return f"{move_time:.2f}s"
            else:
                # 10+ seconds: show 1 decimal place
                return f"{move_time:.1f}s"

    def get_game_info(self) -> dict:
        """Get current game information"""
        return {
            'move_count': len(self.move_history),
            'current_player': 'white' if self.current_player == chess.WHITE else 'black',
            'game_over': self.board.is_game_over(),
            'result': self.get_board_result(),
            'fen': self.board.fen(),
            'game_duration': time.time() - self.game_start_time if self.game_start_time > 0 else 0
        }