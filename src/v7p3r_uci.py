#!/usr/bin/env python3
"""
V7P3R v12.6 UCI Interface - Clean Performance Build
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
                print("id name V7P3R v12.6")
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
                # Search parameter parsing
                time_limit = 3.0  # Default
                depth_limit = None
                perft_depth = None
                
                for i, part in enumerate(parts):
                    if part == "perft" and i + 1 < len(parts):
                        # V11 ENHANCEMENT: Perft command support
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
                    elif part == "wtime" and i + 1 < len(parts):
                        try:
                            if board.turn == chess.WHITE:
                                # V12.2: Balanced aggressive time management
                                remaining_time = int(parts[i + 1]) / 1000.0
                                # Skip tactical detector for simplified version
                                
                                # V12.2: Reasonable time usage - aim for using time but not burning it
                                moves_played = len(board.move_stack)
                                if moves_played < 10:
                                    time_factor = 30.0  # Opening: use 1/30th (10min game = 20s/move)
                                elif moves_played < 20:
                                    time_factor = 25.0  # Early game: use 1/25th  
                                elif moves_played < 40:
                                    time_factor = 20.0  # Mid game: use 1/20th (more time for complex positions)
                                else:
                                    time_factor = 15.0  # End game: use 1/15th (precision matters)
                                
                                # Apply reasonable time cap to prevent burning too much time
                                calculated_time = remaining_time / time_factor
                                if remaining_time > 120:  # More than 2 minutes remaining
                                    time_limit = min(calculated_time, 30.0)  # Max 30s per move
                                elif remaining_time > 60:   # 1-2 minutes remaining
                                    time_limit = min(calculated_time, 15.0)  # Max 15s per move  
                                else:  # Less than 1 minute - be more careful
                                    time_limit = min(calculated_time, 8.0)   # Max 8s per move
                        except:
                            pass
                    elif part == "btime" and i + 1 < len(parts):
                        try:
                            if board.turn == chess.BLACK:
                                # V12.2: Balanced aggressive time management
                                remaining_time = int(parts[i + 1]) / 1000.0
                                # Skip tactical detector for simplified version
                                
                                # V12.2: Reasonable time usage - aim for using time but not burning it
                                moves_played = len(board.move_stack)
                                if moves_played < 10:
                                    time_factor = 30.0  # Opening: use 1/30th (10min game = 20s/move)
                                elif moves_played < 20:
                                    time_factor = 25.0  # Early game: use 1/25th  
                                elif moves_played < 40:
                                    time_factor = 20.0  # Mid game: use 1/20th (more time for complex positions)
                                else:
                                    time_factor = 15.0  # End game: use 1/15th (precision matters)
                                
                                # Apply reasonable time cap to prevent burning too much time
                                calculated_time = remaining_time / time_factor
                                if remaining_time > 120:  # More than 2 minutes remaining
                                    time_limit = min(calculated_time, 30.0)  # Max 30s per move
                                elif remaining_time > 60:   # 1-2 minutes remaining
                                    time_limit = min(calculated_time, 15.0)  # Max 15s per move  
                                else:  # Less than 1 minute - be more careful
                                    time_limit = min(calculated_time, 8.0)   # Max 8s per move
                        except:
                            pass
                
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
