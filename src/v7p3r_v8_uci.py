#!/usr/bin/env python3
"""
V7P3R v8.0 UCI Interface
Enhanced UCI interface with strength setting and performance monitoring
"""

import sys
import chess
from v7p3r_v8 import V7P3REngineV8


def main():
    """Enhanced UCI interface with v8.0 features"""
    engine = V7P3REngineV8()
    board = chess.Board()
    
    print("id name V7P3R v8.0")
    print("id author Pat Snyder")
    
    # UCI options
    print("option name Strength type spin default 75 min 50 max 95")
    print("option name Threads type spin default 4 min 1 max 8")
    
    print("uciok")
    
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
                print("id name V7P3R v8.0")
                print("id author Pat Snyder")
                print("option name Strength type spin default 75 min 50 max 95")
                print("option name Threads type spin default 4 min 1 max 8")
                print("uciok")
                
            elif command == "isready":
                print("readyok")
                
            elif command == "ucinewgame":
                board = chess.Board()
                engine.new_game()
                print("info string New game started")
                
            elif command == "setoption":
                if len(parts) >= 5 and parts[1] == "name":
                    option_name = parts[2]
                    if parts[3] == "value":
                        option_value = parts[4]
                        
                        if option_name == "Strength":
                            try:
                                strength = int(option_value)
                                engine.set_strength(strength)
                            except ValueError:
                                print("info string Invalid strength value")
                        
                        elif option_name == "Threads":
                            try:
                                threads = int(option_value)
                                # Update thread pool size
                                engine.thread_pool._max_workers = min(threads, 8)
                                print(f"info string Threads set to {threads}")
                            except ValueError:
                                print("info string Invalid threads value")
                
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
                            except:
                                break
                                
            elif command == "go":
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
                                if moves_played < 20:
                                    time_factor = 25.0
                                elif moves_played < 40:
                                    time_factor = 30.0
                                else:
                                    time_factor = 40.0
                                time_limit = min(remaining_time / time_factor, 10.0)
                        except:
                            pass
                    elif part == "btime" and i + 1 < len(parts):
                        try:
                            if board.turn == chess.BLACK:
                                remaining_time = int(parts[i + 1]) / 1000.0
                                moves_played = len(board.move_stack)
                                if moves_played < 20:
                                    time_factor = 25.0
                                elif moves_played < 40:
                                    time_factor = 30.0
                                else:
                                    time_factor = 40.0
                                time_limit = min(remaining_time / time_factor, 10.0)
                        except:
                            pass
                
                best_move = engine.search(board, time_limit)
                print(f"bestmove {best_move}")
                
                # Print performance stats
                stats = engine.get_search_stats()
                print(f"info string nodes {engine.nodes_searched} nps {stats['nodes_per_second']} cache_hits {stats['cache_hits']} timeouts {stats['evaluation_timeouts']}")
                
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            print(f"info error {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
