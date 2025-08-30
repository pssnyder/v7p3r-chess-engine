#!/usr/bin/env python3
"""
V7P3R v9.3 UCI Interface
Clean UCI interface with deterministic evaluation
"""

import sys
import chess
from v7p3r import V7P3RCleanEngine


def main():
    """Clean UCI interface for V9.3"""
    engine = V7P3RCleanEngine()
    board = chess.Board()
    
    while True:
        try:
            line = input().strip()
            if not line:
                continue
                
            parts = line.split()
            command = parts[0]
            
            if command == "quit":
                                # V9.3: No configuration options (enhanced heuristics built-in)
                break
                
            elif command == "uci":
                # V9.3: Simplified UCI options - enhanced heuristics built-in
                print("id name V7P3R v9.3")
                print("id author Pat Snyder")
                print("uciok")
                
            elif command == "setoption":
                # V9.3: Enhanced heuristics are built-in, no configuration needed
                if len(parts) >= 4 and parts[1] == "name":
                    option_name = parts[2]
                    if len(parts) >= 5 and parts[3] == "value":
                        option_value = parts[4]
                        print(f"info string Option {option_name}={option_value} acknowledged but not used in v9.3")
                
            elif command == "isready":
                print("readyok")
                
            elif command == "ucinewgame":
                board = chess.Board()
                engine.new_game()
                print("info string New game started - V9.3 enhanced engine")
                
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
                # Clear previous UCI info when starting to think about a new position
                # This ensures previous analysis stays visible until we start the next search
                print("info string Starting search...")
                sys.stdout.flush()
                
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
                                # More aggressive time management - compete with SlowMate
                                remaining_time = int(parts[i + 1]) / 1000.0
                                # Use more time early game, less time in endgame
                                moves_played = len(board.move_stack)
                                if moves_played < 20:
                                    time_factor = 25.0  # Early game: use 1/25th
                                elif moves_played < 40:
                                    time_factor = 30.0  # Mid game: use 1/30th  
                                else:
                                    time_factor = 40.0  # End game: use 1/40th
                                time_limit = min(remaining_time / time_factor, 10.0)
                        except:
                            pass
                    elif part == "btime" and i + 1 < len(parts):
                        try:
                            if board.turn == chess.BLACK:
                                # More aggressive time management - compete with SlowMate
                                remaining_time = int(parts[i + 1]) / 1000.0
                                # Use more time early game, less time in endgame
                                moves_played = len(board.move_stack)
                                if moves_played < 20:
                                    time_factor = 25.0  # Early game: use 1/25th
                                elif moves_played < 40:
                                    time_factor = 30.0  # Mid game: use 1/30th  
                                else:
                                    time_factor = 40.0  # End game: use 1/40th
                                time_limit = min(remaining_time / time_factor, 10.0)
                        except:
                            pass
                
                best_move = engine.search(board, time_limit)
                print(f"bestmove {best_move}")
                
                # Print enhanced performance stats from V9.0
                stats = engine.get_search_stats()
                print(f"info string nodes {engine.nodes_searched} cache_hits {stats['cache_hits']} memory_usage {stats.get('memory_mb', 0):.1f}MB")
                sys.stdout.flush()  # Ensure output is sent immediately
                
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            print(f"info error {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
