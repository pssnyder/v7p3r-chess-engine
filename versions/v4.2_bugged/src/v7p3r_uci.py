"""
v7p3r_uci.py - UCI protocol handler for V7P3R Chess Engine
Allows the engine to communicate with chess GUIs like                     # More aggressive time allocation for tournament play
                    if remaining_seconds > 60:
                        # Early game: use 12-15% of remaining time  
                        time_limit = max(0.3, remaining_seconds * 0.13)
                    elif remaining_seconds > 30:
                        # Mid game: use 15-18% of remaining time  
                        time_limit = max(0.2, remaining_seconds * 0.16)
                    elif remaining_seconds > 10:
                        # Late game: use 20-25% of remaining time
                        time_limit = max(0.15, remaining_seconds * 0.22)
                    else:
                        # Critical time: use 30% but minimum 0.1s
                        time_limit = max(0.1, remaining_seconds * 0.30) Universal Chess Interface (UCI).
"""

import sys
import traceback
import chess
from pathlib import Path

from v7p3r_engine import V7P3REngine

ENGINE_NAME = "V7P3R"
ENGINE_AUTHOR = "Pat Snyder"


class UCIEngine:
    def __init__(self):
        # Core engine that performs searches and evaluations
        # Handle both development and PyInstaller bundled execution
        import sys
        
        if getattr(sys, 'frozen', False):
            # Running as PyInstaller bundle
            bundle_dir = Path(getattr(sys, '_MEIPASS', Path.cwd()))
            default_conf = bundle_dir / "config_default.json"
            config_file = str(default_conf)
        else:
            # Running as script
            pkg_dir = Path(__file__).resolve().parent
            user_conf = pkg_dir / "config.json"
            default_conf = pkg_dir / "config_default.json"
            config_file = str(user_conf) if user_conf.exists() else str(default_conf)
        
        self.engine = V7P3REngine(config_file=config_file)
        self.isready = True
        self.position_fen = None
        self.moves = []

    def uci(self):
        print(f"id name {ENGINE_NAME}")
        print(f"id author {ENGINE_AUTHOR}")
        # Option list can be expanded later using `option name <name> type ...`
        print("uciok")
        sys.stdout.flush()

    def is_ready(self):
        # The engine may perform any lazy initialization here
        print("readyok")
        sys.stdout.flush()

    def ucinewgame(self):
        # Reset engine/game state for a new game
        try:
            self.engine.reset_game()
        except Exception:
            # Some engine builds may not expose reset; ignore if not present
            pass
        self.position_fen = None
        self.moves = []

    def position(self, args: str):
        # Parse 'position [fen ...] moves ...' command and store tokens
        tokens = args.split()
        if not tokens:
            return

        if tokens[0] == "startpos":
            self.position_fen = None
            move_idx = 1
        elif tokens[0] == "fen":
            # fen consists of 6 space-separated fields
            if len(tokens) < 7:
                return
            fen = " ".join(tokens[1:7])
            self.position_fen = fen
            move_idx = 7
        else:
            return

        if len(tokens) > move_idx and tokens[move_idx] == "moves":
            self.moves = tokens[move_idx + 1 :]
        else:
            self.moves = []

    def _build_board_from_position(self) -> chess.Board:
        # Construct a python-chess Board from stored position and moves
        board = chess.Board() if self.position_fen is None else chess.Board(self.position_fen)
        for mv in self.moves:
            try:
                board.push_uci(mv)
            except Exception:
                # Ignore illegal moves here; engine will handle invalid positions later
                pass
        return board

    def go(self, args: str):
        # Parse simple time controls and depth from 'go' arguments
        tokens = args.split()
        # Default time limit (seconds)
        time_limit = 5.0
        depth = None

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "movetime" and i + 1 < len(tokens):
                try:
                    time_limit = float(tokens[i + 1]) / 1000.0
                except Exception:
                    pass
                i += 2
                continue
            if tok in ("wtime", "btime") and i + 1 < len(tokens):
                # Improved time management for fast games
                try:
                    ms = float(tokens[i + 1])
                    remaining_seconds = ms / 1000.0
                    
                    # More aggressive time allocation based on remaining time
                    if remaining_seconds > 60:
                        # Early game: use 6-8% of remaining time
                        time_limit = max(0.5, remaining_seconds * 0.07)
                    elif remaining_seconds > 30:
                        # Mid game: use 8-10% of remaining time  
                        time_limit = max(0.3, remaining_seconds * 0.09)
                    elif remaining_seconds > 10:
                        # Late game: use 12-15% of remaining time
                        time_limit = max(0.2, remaining_seconds * 0.13)
                    else:
                        # Critical time: use 20% but minimum 0.1s
                        time_limit = max(0.1, remaining_seconds * 0.20)
                        
                except Exception:
                    pass
                i += 2
                continue
            if tok == "depth" and i + 1 < len(tokens):
                try:
                    depth = int(tokens[i + 1])
                except Exception:
                    depth = None
                i += 2
                continue
            i += 1

        # Build board and call engine
        board = self._build_board_from_position()

        # If depth was supplied, try to set engine max depth if available
        if depth is not None and hasattr(self.engine.search, 'max_depth'):
            try:
                self.engine.search.max_depth = depth
            except Exception:
                pass

        try:
            best_move = self.engine.find_move(board, time_limit=time_limit)
        except Exception:
            traceback.print_exc()
            best_move = None

        if best_move is None:
            print("bestmove (none)")
            sys.stdout.flush()
            return

        # Convert chess.Move to UCI string if necessary
        mv_str = best_move.uci() if hasattr(best_move, 'uci') else str(best_move)
        print(f"bestmove {mv_str}")
        sys.stdout.flush()

    def loop(self):
        sys.stdout.flush()  # Ensure output is flushed
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if line == "":
                    continue
                if line == "uci":
                    self.uci()
                    sys.stdout.flush()
                elif line == "isready":
                    self.is_ready()
                    sys.stdout.flush()
                elif line.startswith("setoption"):
                    # Option handling can be added here
                    pass
                elif line == "ucinewgame":
                    self.ucinewgame()
                elif line.startswith("position"):
                    self.position(line[len("position") :].strip())
                elif line.startswith("go"):
                    self.go(line[len("go") :].strip())
                    sys.stdout.flush()
                elif line == "quit":
                    break
                # Additional UCI commands can be implemented as needed
            except Exception:
                traceback.print_exc()
                break


def main():
    uci_engine = UCIEngine()
    uci_engine.loop()


if __name__ == "__main__":
    main()
