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
from pathlib import Path
from typing import Optional

from v7p3r import V7P3REvaluationEngine

ENGINE_NAME = "V7P3R v6.0"
ENGINE_AUTHOR = "Pat Snyder"
ENGINE_VERSION = "5.4"


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
        """Execute search and report progress with UCI info strings"""
        try:
            # Build board from position
            board = self._build_board_from_position()
            search_start_time = time.time()
            
            # Set engine parameters
            if depth is not None:
                self.engine.depth = depth
                max_depth = depth
            else:
                max_depth = self.engine.depth if self.engine.depth else 6
            
            # Use iterative deepening for tournament play
            best_move = None
            best_score = 0
            
            # Calculate time allocation
            if not infinite and 'depth' not in time_control and 'nodes' not in time_control and 'movetime' not in time_control:
                allocated_time = self.engine.time_manager.allocate_time(time_control, board)
                self.engine.time_manager.start_timer(allocated_time)
            else:
                allocated_time = float('inf')
            
            # Iterative deepening search with info output
            total_nodes = 0
            for current_depth in range(1, max_depth + 1):
                if self.stop_search:
                    break
                    
                iteration_start = time.time()
                nodes_before = self.engine.nodes_searched
                
                # Set current search depth
                self.engine.depth = current_depth
                
                # Perform search for this depth
                try:
                    # Only use time management callback for time-based searches
                    if infinite or 'depth' in time_control or 'nodes' in time_control or 'movetime' in time_control:
                        stop_callback = lambda: self.stop_search
                    else:
                        stop_callback = lambda: self.stop_search or self.engine.time_manager.should_stop()
                        
                    move_result = self.engine.search(board, board.turn, stop_callback=stop_callback)
                    
                    if move_result and move_result != chess.Move.null():
                        best_move = move_result
                        # Get score from evaluation
                        temp_board = board.copy()
                        temp_board.push(best_move)
                        best_score = self.engine.evaluate_position_from_perspective(temp_board, board.turn)
                        
                except Exception as e:
                    # Continue with current best move if search fails
                    break  # Stop on errors
                
                # Calculate search statistics
                iteration_time = time.time() - iteration_start
                total_time = time.time() - search_start_time
                nodes_this_iteration = self.engine.nodes_searched - nodes_before
                total_nodes += nodes_this_iteration
                nps = int(nodes_this_iteration / max(iteration_time, 0.001))
                
                # Convert score to centipawns
                score_cp = int(best_score * 100) if best_score is not None else 0
                
                # Generate principal variation (simplified - just the best move)
                pv_str = best_move.uci() if best_move else ""
                
                # Output UCI info string (like Stockfish)
                print(f"info depth {current_depth} score cp {score_cp} nodes {total_nodes} nps {nps} time {int(total_time * 1000)} pv {pv_str}")
                sys.stdout.flush()
                
                # Check stopping conditions
                if nodes_limit and total_nodes >= nodes_limit:
                    break
                    
                if 'movetime' in time_control:
                    # For movetime, check if we've exceeded the time
                    if (time.time() - search_start_time) * 1000 >= time_control['movetime']:
                        break
                        
                if not infinite and 'depth' not in time_control and 'nodes' not in time_control and 'movetime' not in time_control:
                    if self.engine.time_manager.should_stop():
                        break
                        
                # For fixed depth searches, stop after reaching target depth
                if depth is not None and current_depth >= depth:
                    break

            # Output final result
            if best_move is None or best_move == chess.Move.null():
                print("bestmove (none)")
            else:
                mv_str = best_move.uci() if hasattr(best_move, 'uci') else str(best_move)
                print(f"bestmove {mv_str}")
            
            sys.stdout.flush()
            
        except Exception as e:
            traceback.print_exc()
            print("bestmove (none)")
            sys.stdout.flush()
        finally:
            self.searching = False

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
