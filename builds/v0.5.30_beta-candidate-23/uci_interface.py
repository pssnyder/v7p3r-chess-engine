# uci_interface.py
"""
UCI (Universal Chess Interface) Protocol Implementation
This module handles communication between the chess engine and UCI-compatible GUIs
"""

import sys
import threading
import time
from typing import Optional
import chess
from main_engine import ChessEngine

class UCIInterface:
    def __init__(self):
        self.engine = ChessEngine()
        self.board = chess.Board()
        self.thinking = False
        self.stop_thinking = False

    def run(self):
        """Main UCI loop - reads commands and responds"""
        while True:
            try:
                line = input().strip()
                if line:
                    self.handle_command(line)
            except EOFError:
                break
            except KeyboardInterrupt:
                break

    def handle_command(self, command: str):
        """Handle UCI commands"""
        parts = command.split()
        if not parts:
            return

        cmd = parts[0]

        if cmd == "uci":
            self.handle_uci()
        elif cmd == "isready":
            self.handle_isready()
        elif cmd == "ucinewgame":
            self.handle_ucinewgame()
        elif cmd == "position":
            self.handle_position(parts[1:])
        elif cmd == "go":
            self.handle_go(parts[1:])
        elif cmd == "stop":
            self.handle_stop()
        elif cmd == "quit":
            self.handle_quit()
        elif cmd == "setoption":
            self.handle_setoption(parts[1:])

    def handle_uci(self):
        """Respond to uci command with engine info"""
        print("id name ChessBot 1.0")
        print("id author YourName")
        print("option name Hash type spin default 64 min 1 max 1024")
        print("option name Threads type spin default 1 min 1 max 8")
        print("option name Depth type spin default 6 min 1 max 20")
        print("uciok")
        sys.stdout.flush()

    def handle_isready(self):
        """Respond when engine is ready"""
        print("readyok")
        sys.stdout.flush()

    def handle_ucinewgame(self):
        """Start new game"""
        self.board = chess.Board()
        self.engine.new_game()

    def handle_position(self, args):
        """Set up board position"""
        if not args:
            return

        if args[0] == "startpos":
            self.board = chess.Board()
            move_index = 1
        elif args[0] == "fen":
            # Find where moves start
            move_index = 1
            fen_parts = []
            while move_index < len(args) and args[move_index] != "moves":
                fen_parts.append(args[move_index])
                move_index += 1

            fen = " ".join(fen_parts)
            self.board = chess.Board(fen)
        else:
            return

        # Apply moves if present
        if move_index < len(args) and args[move_index] == "moves":
            for move_str in args[move_index + 1:]:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                except ValueError:
                    pass

    def handle_go(self, args):
        """Start thinking about the position"""
        # Parse time control
        time_control = self.parse_time_control(args)

        # Stop any current thinking
        self.stop_thinking = True
        time.sleep(0.01)  # Give time for previous search to stop

        # Start new search in separate thread
        self.stop_thinking = False
        search_thread = threading.Thread(
            target=self.search_and_respond,
            args=(time_control,)
        )
        search_thread.daemon = True
        search_thread.start()

    def parse_time_control(self, args):
        """Parse time control from go command"""
        time_control = {
            'wtime': None,
            'btime': None,
            'winc': 0,
            'binc': 0,
            'movestogo': 30,
            'depth': None,
            'movetime': None,
            'infinite': False
        }

        i = 0
        while i < len(args):
            if args[i] == "wtime" and i + 1 < len(args):
                time_control['wtime'] = int(args[i + 1])
                i += 2
            elif args[i] == "btime" and i + 1 < len(args):
                time_control['btime'] = int(args[i + 1])
                i += 2
            elif args[i] == "winc" and i + 1 < len(args):
                time_control['winc'] = int(args[i + 1])
                i += 2
            elif args[i] == "binc" and i + 1 < len(args):
                time_control['binc'] = int(args[i + 1])
                i += 2
            elif args[i] == "movestogo" and i + 1 < len(args):
                time_control['movestogo'] = int(args[i + 1])
                i += 2
            elif args[i] == "depth" and i + 1 < len(args):
                time_control['depth'] = int(args[i + 1])
                i += 2
            elif args[i] == "movetime" and i + 1 < len(args):
                time_control['movetime'] = int(args[i + 1])
                i += 2
            elif args[i] == "infinite":
                time_control['infinite'] = True
                i += 1
            else:
                i += 1

        return time_control

    def search_and_respond(self, time_control):
        """Search for best move and respond"""
        try:
            self.thinking = True
            best_move = self.engine.search(self.board, time_control, self.stop_thinking_callback)

            if best_move and not self.stop_thinking:
                print(f"bestmove {best_move.uci()}")
                sys.stdout.flush()
        except Exception as e:
            print(f"info string Error during search: {e}")
            # Return random legal move as fallback
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                print(f"bestmove {legal_moves[0].uci()}")
            sys.stdout.flush()
        finally:
            self.thinking = False

    def stop_thinking_callback(self):
        """Callback to check if search should stop"""
        return self.stop_thinking

    def handle_stop(self):
        """Stop current search"""
        self.stop_thinking = True

    def handle_quit(self):
        """Quit the engine"""
        self.stop_thinking = True
        sys.exit(0)

    def handle_setoption(self, args):
        """Handle option setting"""
        if len(args) >= 4 and args[0] == "name" and args[2] == "value":
            option_name = args[1]
            option_value = args[3]

            if option_name == "Hash":
                try:
                    hash_size = int(option_value)
                    self.engine.set_hash_size(hash_size)
                except ValueError:
                    pass
            elif option_name == "Threads":
                try:
                    threads = int(option_value)
                    self.engine.set_threads(threads)
                except ValueError:
                    pass
            elif option_name == "Depth":
                try:
                    depth = int(option_value)
                    self.engine.set_max_depth(depth)
                except ValueError:
                    pass

if __name__ == "__main__":
    uci = UCIInterface()
    uci.run()
