#!/usr/bin/env python3
"""
V7P3R v14.8 UCI Interface - Simplified & Back to Basics
Disabled aggressive safety filtering, focusing on search depth and tactical play
Based on V14.0 foundation (best tournament performance: 67.1%)
"""

import sys
import time
import chess
from v7p3r import V7P3REngine


def main():
    """UCI interface"""
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
                print("id name V7P3R v14.8")
                print("id author Pat Snyder")
                print("uciok")
                
            elif command == "setoption":
                if len(parts) >= 4 and parts[1] == "name":
                    option_name = parts[2]
                    if len(parts) >= 5 and parts[3] == "value":
                        option_value = parts[4]
                        print(f"info string Option {option_name}={option_value} acknowledged but not used")
                
            elif command == "isready":
                print("readyok")
                
            elif command == "ucinewgame":
                board = chess.Board()
                engine.new_game()
                # V12.2: Skip tactical cache clearing (not used in simplified version)
                
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
                    
                    # Apply moves and notify engine for PV following
                    if len(parts) > move_start:
                        for i, move_uci in enumerate(parts[move_start:]):
                            try:
                                move = chess.Move.from_uci(move_uci)
                                if board.is_legal(move):
                                    # Notify engine before making the move (for PV following)
                                    engine.notify_move_played(move, board)
                                    board.push(move)
                                else:
                                    break
                            except:
                                break
                                
            elif command == "go":
                # Search parameter parsing - V14.5: Parse ALL parameters first, then calculate time
                time_limit = 3.0  # Default
                depth_limit = None
                perft_depth = None
                increment = 0.0
                wtime_ms = None
                btime_ms = None
                
                # FIRST PASS: Parse all parameters
                for i, part in enumerate(parts):
                    if part == "perft" and i + 1 < len(parts):
                        try:
                            perft_depth = int(parts[i + 1])
                        except:
                            print("info string Invalid perft depth")
                            continue
                    elif part == "movetime" and i + 1 < len(parts):
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
                    elif part == "winc" and i + 1 < len(parts):
                        try:
                            if board.turn == chess.WHITE:
                                increment = int(parts[i + 1]) / 1000.0
                        except:
                            pass
                    elif part == "binc" and i + 1 < len(parts):
                        try:
                            if board.turn == chess.BLACK:
                                increment = int(parts[i + 1]) / 1000.0
                        except:
                            pass
                    elif part == "wtime" and i + 1 < len(parts):
                        try:
                            if board.turn == chess.WHITE:
                                wtime_ms = int(parts[i + 1])
                        except:
                            pass
                    elif part == "btime" and i + 1 < len(parts):
                        try:
                            if board.turn == chess.BLACK:
                                btime_ms = int(parts[i + 1])
                        except:
                            pass
                
                # SECOND PASS: Calculate time allocation with all info available
                if wtime_ms is not None and board.turn == chess.WHITE:
                    remaining_time = wtime_ms / 1000.0
                    moves_played = len(board.move_stack)
                    
                    # V14.5: Opening time allocation - use VERY LITTLE time
                    if moves_played < 8:
                        time_factor = 80.0
                        time_limit = (remaining_time / time_factor) + (increment * 0.8)
                    elif moves_played < 15:
                        time_factor = 50.0
                        time_limit = (remaining_time / time_factor) + increment
                    elif moves_played < 30:
                        time_factor = 30.0
                        time_limit = (remaining_time / time_factor) + increment
                    else:
                        time_factor = 20.0
                        time_limit = (remaining_time / time_factor) + increment
                    
                    # Apply caps based on remaining time to prevent flagging
                    if remaining_time > 180:
                        time_limit = min(time_limit, 25.0)
                    elif remaining_time > 120:
                        time_limit = min(time_limit, 15.0)
                    elif remaining_time > 60:
                        time_limit = min(time_limit, 10.0)
                    elif remaining_time > 30:
                        time_limit = min(time_limit, 5.0)
                    else:
                        time_limit = min(time_limit, 2.0)
                
                elif btime_ms is not None and board.turn == chess.BLACK:
                    remaining_time = btime_ms / 1000.0
                    moves_played = len(board.move_stack)
                    
                    # V14.5: Same time management for black
                    if moves_played < 8:
                        time_factor = 80.0
                        time_limit = (remaining_time / time_factor) + (increment * 0.8)
                    elif moves_played < 15:
                        time_factor = 50.0
                        time_limit = (remaining_time / time_factor) + increment
                    elif moves_played < 30:
                        time_factor = 30.0
                        time_limit = (remaining_time / time_factor) + increment
                    else:
                        time_factor = 20.0
                        time_limit = (remaining_time / time_factor) + increment
                    
                    # Apply caps
                    if remaining_time > 180:
                        time_limit = min(time_limit, 25.0)
                    elif remaining_time > 120:
                        time_limit = min(time_limit, 15.0)
                    elif remaining_time > 60:
                        time_limit = min(time_limit, 10.0)
                    elif remaining_time > 30:
                        time_limit = min(time_limit, 5.0)
                    else:
                        time_limit = min(time_limit, 2.0)
                
                # V11 ENHANCEMENT: Handle perft command
                if perft_depth is not None:
                    print(f"info string Starting perft {perft_depth}")
                    try:
                        start_time = time.time()
                        nodes = engine.perft(board, perft_depth, divide=False)
                        elapsed = time.time() - start_time
                        nps = int(nodes / max(elapsed, 0.001))
                        print(f"info string Perft {perft_depth}: {nodes} nodes in {elapsed:.3f}s ({nps} nps)")
                        print(f"perft {perft_depth}: {nodes}")
                    except Exception as e:
                        print(f"info string Perft error: {e}")
                    sys.stdout.flush()
                else:
                    # Normal search
                    best_move = engine.search(board, time_limit)
                    print(f"bestmove {best_move}")
                    sys.stdout.flush()  # Ensure output is sent immediately
                
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            print(f"info error {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
