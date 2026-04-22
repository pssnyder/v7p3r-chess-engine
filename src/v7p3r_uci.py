#!/usr/bin/env python3
"""
V7P3R v19.0.0 UCI Interface - Spring Cleaning
"""

import sys
import time
import chess
from v7p3r import V7P3REngine
from v7p3r_time_manager import TimeManager


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
                print("id name V7P3R v19.0")
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
                # V19.0: Simplified time management using TimeManager
                time_limit = 5.0  # Default fallback
                depth_limit = None
                perft_depth = None
                
                # Parse time control parameters
                movetime = None
                wtime = None
                btime = None
                winc = 0
                binc = 0
                
                for i, part in enumerate(parts):
                    try:
                        if part == "perft" and i + 1 < len(parts):
                            perft_depth = int(parts[i + 1])
                        elif part == "movetime" and i + 1 < len(parts):
                            movetime = int(parts[i + 1]) / 1000.0
                        elif part == "depth" and i + 1 < len(parts):
                            depth_limit = int(parts[i + 1])
                            engine.default_depth = depth_limit
                        elif part == "wtime" and i + 1 < len(parts):
                            wtime = int(parts[i + 1]) / 1000.0
                        elif part == "btime" and i + 1 < len(parts):
                            btime = int(parts[i + 1]) / 1000.0
                        elif part == "winc" and i + 1 < len(parts):
                            winc = int(parts[i + 1]) / 1000.0
                        elif part == "binc" and i + 1 < len(parts):
                            binc = int(parts[i + 1]) / 1000.0
                    except (ValueError, IndexError):
                        pass
                
                # V19.0: Use TimeManager for clean time allocation
                if movetime is not None:
                    # Fixed time per move
                    time_limit = movetime
                else:
                    # Use time control with TimeManager
                    remaining_time = wtime if board.turn == chess.WHITE else btime
                    increment = winc if board.turn == chess.WHITE else binc
                    
                    if remaining_time is not None:
                        moves_played = len(board.move_stack)
                        target_time, max_time = TimeManager.calculate_time_allocation(
                            remaining_time, increment, moves_played, board
                        )
                        time_limit = target_time
                
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
