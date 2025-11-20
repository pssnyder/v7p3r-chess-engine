#!/usr/bin/env python3
"""
V7P3R v16.3 UCI Interface
Bug Fix: PV Display and Following
"""

import sys
from v7p3r_v163 import V7P3REngine
import chess

class UCIEngine:
    def __init__(self):
        self.tablebase_path = ""
        self.max_depth = 10
        self.tt_size = 256
        self.engine = V7P3REngine(max_depth=self.max_depth, tt_size_mb=self.tt_size, 
                                   tablebase_path=self.tablebase_path)
        self.debug = False
        
    def run(self):
        """Main UCI loop"""
        while True:
            try:
                line = input().strip()
                if not line:
                    continue
                
                parts = line.split()
                command = parts[0]
                
                if command == "uci":
                    self._handle_uci()
                elif command == "isready":
                    print("readyok", flush=True)
                elif command == "ucinewgame":
                    self.engine = V7P3REngine(max_depth=self.max_depth, tt_size_mb=self.tt_size, 
                                               tablebase_path=self.tablebase_path)
                elif command == "position":
                    self._handle_position(parts[1:])
                elif command == "go":
                    self._handle_go(parts[1:])
                elif command == "quit":
                    break
                elif command == "setoption":
                    self._handle_setoption(parts[1:])
                elif command == "debug":
                    self.debug = parts[1] == "on" if len(parts) > 1 else False
                    
            except EOFError:
                break
            except Exception as e:
                if self.debug:
                    print(f"info string Error: {e}", flush=True)
    
    def _handle_uci(self):
        """Handle UCI identification"""
        print("id name V7P3R v16.3", flush=True)
        print("id author V7P3R Team", flush=True)
        print("option name MaxDepth type spin default 10 min 1 max 20", flush=True)
        print("option name TTSize type spin default 256 min 16 max 2048", flush=True)
        print("option name SyzygyPath type string default <empty>", flush=True)
        print("uciok", flush=True)
    
    def _handle_position(self, args):
        """Handle position command"""
        if not args:
            return
        
        if args[0] == "startpos":
            self.engine.board = chess.Board()
            moves_idx = args.index("moves") if "moves" in args else -1
            if moves_idx != -1:
                for move_str in args[moves_idx + 1:]:
                    try:
                        move = chess.Move.from_uci(move_str)
                        self.engine.board.push(move)
                    except:
                        if self.debug:
                            print(f"info string Invalid move: {move_str}", flush=True)
        
        elif args[0] == "fen":
            moves_idx = args.index("moves") if "moves" in args else len(args)
            fen = " ".join(args[1:moves_idx])
            try:
                self.engine.board = chess.Board(fen)
                if moves_idx < len(args):
                    for move_str in args[moves_idx + 1:]:
                        try:
                            move = chess.Move.from_uci(move_str)
                            self.engine.board.push(move)
                        except:
                            if self.debug:
                                print(f"info string Invalid move: {move_str}", flush=True)
            except Exception as e:
                if self.debug:
                    print(f"info string Invalid FEN: {e}", flush=True)
    
    def _handle_go(self, args):
        """Handle go command"""
        time_left = 0
        increment = 0
        depth = None
        
        i = 0
        while i < len(args):
            if args[i] == "wtime" and i + 1 < len(args):
                if self.engine.board.turn == chess.WHITE:
                    time_left = int(args[i + 1]) / 1000.0
                i += 2
            elif args[i] == "btime" and i + 1 < len(args):
                if self.engine.board.turn == chess.BLACK:
                    time_left = int(args[i + 1]) / 1000.0
                i += 2
            elif args[i] == "winc" and i + 1 < len(args):
                if self.engine.board.turn == chess.WHITE:
                    increment = int(args[i + 1]) / 1000.0
                i += 2
            elif args[i] == "binc" and i + 1 < len(args):
                if self.engine.board.turn == chess.BLACK:
                    increment = int(args[i + 1]) / 1000.0
                i += 2
            elif args[i] == "depth" and i + 1 < len(args):
                depth = int(args[i + 1])
                i += 2
            else:
                i += 1
        
        # Override depth if specified
        original_depth = self.engine.max_depth
        if depth is not None:
            self.engine.max_depth = depth
        
        best_move = self.engine.get_best_move(time_left, increment)
        
        # Restore original depth
        self.engine.max_depth = original_depth
        
        if best_move:
            print(f"bestmove {best_move.uci()}", flush=True)
        else:
            print("bestmove 0000", flush=True)
    
    def _handle_setoption(self, args):
        """Handle setoption command"""
        if len(args) < 2 or args[0] != "name":
            return
        
        # Find value index
        try:
            value_idx = args.index("value")
            option_name = " ".join(args[1:value_idx])
            option_value = " ".join(args[value_idx + 1:])
        except (ValueError, IndexError):
            return
        
        try:
            if option_name == "MaxDepth":
                self.max_depth = int(option_value)
                self.engine.max_depth = self.max_depth
            elif option_name == "TTSize":
                self.tt_size = int(option_value)
                self.engine.tt_size = (self.tt_size * 1024 * 1024) // 64
                self.engine.transposition_table = {}
            elif option_name == "SyzygyPath":
                self.tablebase_path = option_value
                # Reload engine with new tablebase path
                self.engine = V7P3REngine(max_depth=self.max_depth, tt_size_mb=self.tt_size, 
                                           tablebase_path=self.tablebase_path)
        except Exception as e:
            if self.debug:
                print(f"info string Error setting option: {e}", flush=True)


if __name__ == "__main__":
    uci_engine = UCIEngine()
    uci_engine.run()
