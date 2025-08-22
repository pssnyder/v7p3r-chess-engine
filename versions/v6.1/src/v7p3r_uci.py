"""
v7p3r_uci.py - UCI protocol handler for V7P3R Chess Engine
Allows the engine to communicate with chess GUIs like Arena via the Universal Chess Interface (UCI).
Enhanced with tournament-level output including depth, nodes, nps, pv, and proper time control handling.
"""

import sys
import traceback
import chess
import time
import threading
from typing import Optional

from v7p3r import V7P3REvaluationEngine

ENGINE_NAME = "V7P3R v6.1"
ENGINE_AUTHOR = "Pat Snyder"
ENGINE_VERSION = "5.4"


class UCIEngine:
    def __init__(self):
        # Core engine that performs searches and evaluations
        # Initialize engine
        self.engine = V7P3REvaluationEngine()
        self.isready = True
        self.position_fen = None
        self.moves = []
        self.searching = False
        self.search_thread = None
        self.stop_search = False
        
        # UCI options
        self.options = {
            'Hash': {'type': 'spin', 'default': 256, 'min': 1, 'max': 2048, 'value': 256},
            'Threads': {'type': 'spin', 'default': 1, 'min': 1, 'max': 16, 'value': 1},
            'Clear Hash': {'type': 'button'},
            'Move Overhead': {'type': 'spin', 'default': 30, 'min': 0, 'max': 1000, 'value': 30},
        }

    def uci(self):
        print(f"id name {ENGINE_NAME}")
        print(f"id author {ENGINE_AUTHOR}")
        
        # UCI options for Arena compatibility
        for option_name, option_info in self.options.items():
            if option_info['type'] == 'spin':
                print(f"option name {option_name} type spin default {option_info['default']} min {option_info['min']} max {option_info['max']}")
            elif option_info['type'] == 'button':
                print(f"option name {option_name} type button")
                
        print("uciok")
        sys.stdout.flush()

    def setoption(self, args: str):
        """Handle UCI option settings"""
        tokens = args.split()
        if len(tokens) >= 4 and tokens[0] == "name":
            option_name = tokens[1]
            if len(tokens) >= 4 and tokens[2] == "value":
                option_value = " ".join(tokens[3:])
                
                if option_name in self.options:
                    if option_name == "Hash":
                        try:
                            hash_mb = int(option_value)
                            self.options["Hash"]["value"] = hash_mb
                            # Update engine hash size
                            self.engine.hash_size = hash_mb * 1024 * 1024
                        except ValueError:
                            pass
                    elif option_name == "Clear Hash":
                        self.engine.transposition_table.clear()
                    elif option_name == "Move Overhead":
                        try:
                            overhead = int(option_value)
                            self.options["Move Overhead"]["value"] = overhead
                        except ValueError:
                            pass

    def is_ready(self):
        # The engine may perform any lazy initialization here
        print("readyok")
        sys.stdout.flush()

    def ucinewgame(self):
        # Reset engine/game state for a new game
        try:
            self.engine.reset()
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
        """Enhanced go command with tournament-level output"""
        # Stop any ongoing search
        self.stop_search = False
        
        # Parse time controls and depth from 'go' arguments
        tokens = args.split()
        time_control = {}
        depth = None
        nodes_limit = None
        infinite = False

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "movetime" and i + 1 < len(tokens):
                try:
                    time_control['movetime'] = int(tokens[i + 1])
                except Exception:
                    pass
                i += 2
                continue
            if tok in ("wtime", "btime") and i + 1 < len(tokens):
                try:
                    time_control[tok] = int(tokens[i + 1])
                except Exception:
                    pass
                i += 2
                continue
            if tok in ("winc", "binc") and i + 1 < len(tokens):
                try:
                    time_control[tok] = int(tokens[i + 1])
                except Exception:
                    pass
                i += 2
                continue
            if tok == "depth" and i + 1 < len(tokens):
                try:
                    depth = int(tokens[i + 1])
                    time_control['depth'] = depth
                except Exception:
                    depth = None
                i += 2
                continue
            if tok == "nodes" and i + 1 < len(tokens):
                try:
                    nodes_limit = int(tokens[i + 1])
                    time_control['nodes'] = nodes_limit
                except Exception:
                    pass
                i += 2
                continue
            if tok == "infinite":
                infinite = True
                time_control['infinite'] = True
                i += 1
                continue
            if tok == "movestogo" and i + 1 < len(tokens):
                try:
                    time_control['movestogo'] = int(tokens[i + 1])
                except Exception:
                    pass
                i += 2
                continue
            i += 1

        # Execute search directly (synchronous for simplicity)
        self._search_and_report(time_control, depth, nodes_limit, infinite)

    def _search_and_report(self, time_control: dict, depth: Optional[int] = None, nodes_limit: Optional[int] = None, infinite: bool = False):
        """V6.1 Enhanced search with integrated time management and real-time UCI reporting"""
        try:
            # Build board from position
            board = self._build_board_from_position()
            search_start_time = time.time()
            
            # Use the new integrated search function for v6.1
            best_move, final_depth, total_nodes, search_time_ms = self.engine.search_with_time_management(board, time_control)
            
            # Calculate final statistics
            nps = int(total_nodes / max(search_time_ms / 1000.0, 0.001)) if search_time_ms > 0 else 0
            
            # Convert score to centipawns (evaluate final position)
            if best_move and best_move != chess.Move.null():
                temp_board = board.copy()
                temp_board.push(best_move)
                score = self.engine.evaluate_position_from_perspective(temp_board, board.turn)
                score_cp = int(score * 100)
            else:
                score_cp = 0
                
            # Output final info string
            pv_str = best_move.uci() if best_move and best_move != chess.Move.null() else ""
            print(f"info depth {final_depth} score cp {score_cp} nodes {total_nodes} nps {nps} time {search_time_ms} pv {pv_str}")
            sys.stdout.flush()
            
            # Output best move
            if best_move and best_move != chess.Move.null():
                print(f"bestmove {best_move.uci()}")
            else:
                # Emergency fallback
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    fallback = legal_moves[0]
                    print(f"bestmove {fallback.uci()}")
                else:
                    print("bestmove 0000")
                    
            sys.stdout.flush()
            
        except Exception as e:
            # Emergency fallback on any error
            try:
                board = self._build_board_from_position()
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    print(f"bestmove {legal_moves[0].uci()}")
                else:
                    print("bestmove 0000")
            except:
                print("bestmove 0000")
            sys.stdout.flush()

    def stop(self):
        """Stop the current search"""
        self.stop_search = True

    def loop(self):
        """Main UCI command loop with enhanced tournament support"""
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
                    self.setoption(line[len("setoption") :].strip())
                elif line == "ucinewgame":
                    self.ucinewgame()
                elif line.startswith("position"):
                    self.position(line[len("position") :].strip())
                elif line.startswith("go"):
                    self.go(line[len("go") :].strip())
                    sys.stdout.flush()
                elif line == "stop":
                    self.stop()
                elif line == "quit":
                    self.stop_search = True
                    if self.search_thread and self.search_thread.is_alive():
                        self.search_thread.join(timeout=0.5)
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
