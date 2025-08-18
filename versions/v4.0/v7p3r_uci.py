
"""
v7p3r_uci.py - UCI protocol handler for V7P3R Chess Engine
Allows the engine to communicate with chess GUIs like Arena via the Universal Chess Interface (UCI).
"""

import sys
import threading
import traceback

# Import your engine's move generation and game state modules
# from v7p3r_engine import V7P3REngine  # Example, adjust as needed
# from v7p3r_game import V7P3RGame      # Example, adjust as needed

ENGINE_NAME = "V7P3R Chess Engine"
ENGINE_AUTHOR = "pssnyder"

class UCIEngine:
    def __init__(self):
        # self.engine = V7P3REngine()  # Integrate your engine here
        # self.game = V7P3RGame()      # Integrate your game state here
        self.isready = True
        self.position_fen = None
        self.moves = []

    def uci(self):
        print(f"id name {ENGINE_NAME}")
        print(f"id author {ENGINE_AUTHOR}")
        print("uciok")

    def is_ready(self):
        print("readyok")

    def ucinewgame(self):
        # Reset engine/game state for a new game
        # self.engine.reset()
        # self.game.reset()
        self.position_fen = None
        self.moves = []

    def position(self, args):
        # Parse 'position [fen ...] moves ...' command
        tokens = args.split()
        if tokens[0] == "startpos":
            self.position_fen = None  # Use default starting position
            move_idx = 1
        elif tokens[0] == "fen":
            fen = " ".join(tokens[1:7])
            self.position_fen = fen
            move_idx = 7
        else:
            return
        if len(tokens) > move_idx and tokens[move_idx] == "moves":
            self.moves = tokens[move_idx+1:]
        else:
            self.moves = []
        # TODO: Set up board position and apply moves

    def go(self, args):
        # For now, just output a dummy move (e2e4)
        # Integrate with your engine's search here
        # best_move = self.engine.get_best_move()
        best_move = "e2e4"  # Placeholder
        print(f"bestmove {best_move}")

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
                    # Optionally handle engine options
                    pass
                elif line == "ucinewgame":
                    self.ucinewgame()
                elif line.startswith("position"):
                    self.position(line[len("position"):].strip())
                elif line.startswith("go"):
                    self.go(line[len("go"):].strip())
                elif line == "quit":
                    break
                # Add more UCI commands as needed
            except Exception as e:
                traceback.print_exc()
                break

def main():
    uci_engine = UCIEngine()
    uci_engine.loop()

if __name__ == "__main__":
    main()
