#!/usr/bin/env python3
"""V7P3R v15.2 UCI Interface

Version 15.2: Fixed material blindness with proper SEE
- Removed broken material floor from evaluation
- Added Static Exchange Evaluation for captures and attacked squares
- Enhanced move safety filtering to prevent hanging pieces
"""

import sys
import chess
from v7p3r import V7P3REngine


def main():
    engine = V7P3REngine()
    
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
                print("id name V7P3R v15.2")
                print("id author Pat Snyder")
                print("option name MaxDepth type spin default 8 min 1 max 20")
                print("option name TTSize type spin default 128 min 16 max 1024")
                print("uciok")
                sys.stdout.flush()
                
            elif command == "setoption":
                if len(parts) >= 5 and parts[1] == "name" and parts[3] == "value":
                    option_name = parts[2]
                    option_value = parts[4]
                    
                    if option_name == "MaxDepth":
                        engine.max_depth = max(1, min(20, int(option_value)))
                    elif option_name == "TTSize":
                        tt_size = max(16, min(1024, int(option_value)))
                        engine = V7P3REngine(max_depth=engine.max_depth, tt_size_mb=tt_size)
                sys.stdout.flush()
                
            elif command == "isready":
                print("readyok")
                sys.stdout.flush()
                
            elif command == "ucinewgame":
                engine = V7P3REngine(max_depth=engine.max_depth, tt_size_mb=128)
                sys.stdout.flush()
                
            elif command == "position":
                if len(parts) > 1:
                    if parts[1] == "startpos":
                        engine.board = chess.Board()
                        move_start = 3 if len(parts) > 2 and parts[2] == "moves" else 2
                    elif parts[1] == "fen":
                        fen_parts = []
                        i = 2
                        while i < len(parts) and parts[i] != "moves":
                            fen_parts.append(parts[i])
                            i += 1
                        engine.board = chess.Board(" ".join(fen_parts))
                        move_start = i + 1 if i < len(parts) and parts[i] == "moves" else len(parts)
                    else:
                        move_start = len(parts)
                    
                    if move_start < len(parts):
                        for move_uci in parts[move_start:]:
                            try:
                                move = chess.Move.from_uci(move_uci)
                                if engine.board.is_legal(move):
                                    engine.board.push(move)
                                else:
                                    break
                            except:
                                break
                                
            elif command == "go":
                time_left = 0.0
                increment = 0.0
                depth_override = None
                
                for i, part in enumerate(parts):
                    if part == "wtime" and i + 1 < len(parts) and engine.board.turn == chess.WHITE:
                        time_left = float(parts[i + 1]) / 1000.0
                    elif part == "btime" and i + 1 < len(parts) and engine.board.turn == chess.BLACK:
                        time_left = float(parts[i + 1]) / 1000.0
                    elif part == "winc" and i + 1 < len(parts) and engine.board.turn == chess.WHITE:
                        increment = float(parts[i + 1]) / 1000.0
                    elif part == "binc" and i + 1 < len(parts) and engine.board.turn == chess.BLACK:
                        increment = float(parts[i + 1]) / 1000.0
                    elif part == "depth" and i + 1 < len(parts):
                        depth_override = int(parts[i + 1])
                    elif part == "movetime" and i + 1 < len(parts):
                        time_left = float(parts[i + 1]) / 1000.0
                        increment = 0.0
                
                original_depth = engine.max_depth
                if depth_override:
                    engine.max_depth = depth_override
                    time_left = 0
                
                best_move = engine.get_best_move(time_left, increment)
                
                if depth_override:
                    engine.max_depth = original_depth
                
                if best_move:
                    print(f"bestmove {best_move.uci()}")
                else:
                    print("bestmove 0000")
                sys.stdout.flush()
                
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            print(f"info string Error: {e}", file=sys.stderr)
            sys.stderr.flush()


if __name__ == "__main__":
    main()
