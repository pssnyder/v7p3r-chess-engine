
"""
v7p3r_uci.py - UCI protocol handler for V7P3R Chess Engine
Allows the engine to communicate with chess GUIs like Arena via the Universal Chess Interface (UCI).
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
        # Try to use user config.json if present, otherwise fall back to config_default.json
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

    def is_ready(self):
        # The engine may perform any lazy initialization here
        print("readyok")

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
                # We could use remaining time to set a search budget; use a small fraction
                try:
                    ms = float(tokens[i + 1])
                    # Use a conservative fraction of remaining time
                    time_limit = max(0.05, ms / 1000.0 * 0.02)
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
            return

        # Convert chess.Move to UCI string if necessary
        mv_str = best_move.uci() if hasattr(best_move, 'uci') else str(best_move)
        print(f"bestmove {mv_str}")

    def loop(self):
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
                elif line == "isready":
                    self.is_ready()
                elif line.startswith("setoption"):
                    # Option handling can be added here
                    pass
                elif line == "ucinewgame":
                    self.ucinewgame()
                elif line.startswith("position"):
                    self.position(line[len("position") :].strip())
                elif line.startswith("go"):
                    self.go(line[len("go") :].strip())
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
