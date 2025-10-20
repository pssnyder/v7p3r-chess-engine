#!/usr/bin/env python3
"""
VPR UCI Interface - Minimal UCI protocol implementation

Provides UCI communication for the VPR (barebones) engine.
"""

import sys
import chess
from vpr import VPREngine


def main():
    """UCI interface for VPR engine"""
    engine = VPREngine()
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
                info = engine.get_engine_info()
                print(f"id name {info['name']} v{info['version']}")
                print(f"id author {info['author']}")
                print("uciok")
            
            elif command == "isready":
                print("readyok")
            
            elif command == "ucinewgame":
                board = chess.Board()
                engine.new_game()
            
            elif command == "position":
                if len(parts) < 2:
                    continue
                
                # Parse position command
                if parts[1] == "startpos":
                    board = chess.Board()
                    moves_index = 3 if len(parts) > 2 and parts[2] == "moves" else -1
                elif parts[1] == "fen":
                    # Find where moves start
                    moves_index = -1
                    fen_parts = []
                    for i in range(2, len(parts)):
                        if parts[i] == "moves":
                            moves_index = i + 1
                            break
                        fen_parts.append(parts[i])
                    
                    fen = " ".join(fen_parts)
                    try:
                        board = chess.Board(fen)
                    except:
                        continue
                else:
                    continue
                
                # Apply moves if present
                if moves_index > 0 and moves_index < len(parts):
                    for move_str in parts[moves_index:]:
                        try:
                            move = chess.Move.from_uci(move_str)
                            if move in board.legal_moves:
                                board.push(move)
                        except:
                            break
            
            elif command == "go":
                # Parse go command
                time_limit = 3.0  # Default
                depth = None
                
                i = 1
                while i < len(parts):
                    if parts[i] == "movetime":
                        if i + 1 < len(parts):
                            time_limit = int(parts[i + 1]) / 1000.0  # Convert ms to seconds
                        i += 2
                    elif parts[i] == "depth":
                        if i + 1 < len(parts):
                            depth = int(parts[i + 1])
                        i += 2
                    elif parts[i] == "wtime" and board.turn == chess.WHITE:
                        if i + 1 < len(parts):
                            wtime = int(parts[i + 1])
                            # Use a fraction of remaining time
                            time_limit = min(wtime / 30000.0, 5.0)  # Use 1/30 of time, max 5s
                        i += 2
                    elif parts[i] == "btime" and board.turn == chess.BLACK:
                        if i + 1 < len(parts):
                            btime = int(parts[i + 1])
                            time_limit = min(btime / 30000.0, 5.0)
                        i += 2
                    else:
                        i += 1
                
                # Search for best move
                best_move = engine.search(board, time_limit=time_limit, depth=depth)
                print(f"bestmove {best_move}")
            
            elif command == "stop":
                # For now, we don't support stopping mid-search
                # Just acknowledge
                pass
            
            elif command == "setoption":
                # No options supported yet
                pass
            
        except EOFError:
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            # Silently continue on errors
            print(f"info string Error: {e}")


if __name__ == "__main__":
    main()
