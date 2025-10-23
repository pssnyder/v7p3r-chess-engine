#!/usr/bin/env python3
"""
V7P3R v13.0 UCI Interface - Clean Arena Build
"""

import sys
import time
import chess
import os

# Suppress all warnings and prints that could interfere with UCI
import warnings
warnings.filterwarnings("ignore")

# Redirect stderr to devnull to suppress engine warnings
if os.name == 'nt':  # Windows
    import subprocess
    sys.stderr = open(os.devnull, 'w')

from v7p3r import V7P3REngine


def main():
    """UCI interface - Clean for Arena compatibility"""
    engine = V7P3REngine()
    board = chess.Board()
    
    while True:
        try:
            line = input().strip()
            if not line:
                continue
                
            parts = line.split()
            command = parts[0]
            
            if command == "quit":
                break
                
            elif command == "uci":
                print("id name V7P3R v13.0")
                print("id author Pat Snyder")
                print("uciok")
                
            elif command == "setoption":
                # Acknowledge but ignore options
                pass
                
            elif command == "isready":
                print("readyok")
                
            elif command == "ucinewgame":
                board = chess.Board()
                engine.new_game()
                
            elif command == "position":
                if len(parts) > 1:
                    if parts[1] == "startpos":
                        board = chess.Board()
                        move_start = 2
                        if len(parts) > 2 and parts[2] == "moves":
                            move_start = 3
                    elif parts[1] == "fen":
                        fen_parts = parts[2:8]  # FEN has 6 parts
                        fen = " ".join(fen_parts)
                        board = chess.Board(fen)
                        move_start = 8
                        if len(parts) > 8 and parts[8] == "moves":
                            move_start = 9
                    
                    # Apply moves
                    if len(parts) > move_start:
                        for move_uci in parts[move_start:]:
                            try:
                                move = chess.Move.from_uci(move_uci)
                                if board.is_legal(move):
                                    board.push(move)
                                else:
                                    break
                            except:
                                break
                                
            elif command == "go":
                # Search parameter parsing
                time_limit = 3.0  # Default
                depth_limit = None
                
                for i, part in enumerate(parts):
                    if part == "movetime" and i + 1 < len(parts):
                        try:
                            time_limit = int(parts[i + 1]) / 1000.0
                        except:
                            pass
                    elif part == "depth" and i + 1 < len(parts):
                        try:
                            depth_limit = int(parts[i + 1])
                            engine.default_depth = depth_limit
                        except:
                            pass
                    elif part == "wtime" and i + 1 < len(parts):
                        try:
                            if board.turn == chess.WHITE:
                                remaining_time = int(parts[i + 1]) / 1000.0
                                moves_played = len(board.move_stack)
                                if moves_played < 10:
                                    time_factor = 30.0
                                elif moves_played < 20:
                                    time_factor = 25.0
                                elif moves_played < 40:
                                    time_factor = 20.0
                                else:
                                    time_factor = 15.0
                                
                                calculated_time = remaining_time / time_factor
                                if remaining_time > 120:
                                    time_limit = min(calculated_time, 30.0)
                                elif remaining_time > 60:
                                    time_limit = min(calculated_time, 15.0)
                                else:
                                    time_limit = min(calculated_time, 8.0)
                        except:
                            pass
                    elif part == "btime" and i + 1 < len(parts):
                        try:
                            if board.turn == chess.BLACK:
                                remaining_time = int(parts[i + 1]) / 1000.0
                                moves_played = len(board.move_stack)
                                if moves_played < 10:
                                    time_factor = 30.0
                                elif moves_played < 20:
                                    time_factor = 25.0
                                elif moves_played < 40:
                                    time_factor = 20.0
                                else:
                                    time_factor = 15.0
                                
                                calculated_time = remaining_time / time_factor
                                if remaining_time > 120:
                                    time_limit = min(calculated_time, 30.0)
                                elif remaining_time > 60:
                                    time_limit = min(calculated_time, 15.0)
                                else:
                                    time_limit = min(calculated_time, 8.0)
                        except:
                            pass
                
                # Normal search
                best_move = engine.search(board, time_limit)
                print(f"bestmove {best_move}")
                sys.stdout.flush()
                
        except (EOFError, KeyboardInterrupt):
            break
        except Exception:
            # Silently ignore errors for clean UCI
            pass


if __name__ == "__main__":
    main()
