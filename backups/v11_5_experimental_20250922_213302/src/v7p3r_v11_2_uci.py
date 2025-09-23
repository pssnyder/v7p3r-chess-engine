#!/usr/bin/env python3
"""
V7P3R v11.2 Enhanced UCI Interface
Standard UCI interface for tournament play with enhanced dynamic positional intelligence
"""

import sys
import time
import chess
from v7p3r_v11_2_enhanced import V7P3REngineEnhanced


def main():
    """UCI interface"""
    engine = V7P3REngineEnhanced()
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
                print("id name V7P3R v11.2 Enhanced")
                print("id author Pat Snyder")
                print("uciok")
                
            elif command == "setoption":
                if len(parts) >= 4 and parts[1] == "name":
                    option_name = parts[2]
                    if len(parts) >= 5 and parts[3] == "value":
                        option_value = parts[4]
                        print(f"info string Option {option_name}={option_value} acknowledged")
                
            elif command == "isready":
                print("readyok")
                
            elif command == "ucinewgame":
                board = chess.Board()
                engine.new_game()
                
            elif command == "position":
                if len(parts) > 1:
                    if parts[1] == "startpos":
                        board = chess.Board()
                        
                        # Handle moves
                        if len(parts) > 2 and parts[2] == "moves":
                            for move_str in parts[3:]:
                                try:
                                    move = chess.Move.from_uci(move_str)
                                    if move in board.legal_moves:
                                        board.push(move)
                                except:
                                    print(f"info string Invalid move: {move_str}")
                                    
                    elif parts[1] == "fen":
                        # Position from FEN
                        fen_parts = []
                        i = 2
                        # Collect FEN string (up to 6 parts)
                        while i < len(parts) and i < 8:
                            if parts[i] == "moves":
                                break
                            fen_parts.append(parts[i])
                            i += 1
                        
                        try:
                            fen = " ".join(fen_parts)
                            board = chess.Board(fen)
                        except:
                            print(f"info string Invalid FEN: {fen}")
                            continue
                        
                        # Handle moves after FEN
                        if i < len(parts) and parts[i] == "moves":
                            for move_str in parts[i+1:]:
                                try:
                                    move = chess.Move.from_uci(move_str)
                                    if move in board.legal_moves:
                                        board.push(move)
                                except:
                                    print(f"info string Invalid move: {move_str}")
                
            elif command == "go":
                # Parse go command
                time_limit = 5.0  # Default
                
                i = 1
                while i < len(parts):
                    if parts[i] == "movetime":
                        if i + 1 < len(parts):
                            time_limit = int(parts[i + 1]) / 1000.0
                            i += 2
                        else:
                            i += 1
                    elif parts[i] == "wtime" and board.turn == chess.WHITE:
                        if i + 1 < len(parts):
                            wtime = int(parts[i + 1]) / 1000.0
                            time_limit = min(wtime / 30, 10.0)  # Conservative time management
                            i += 2
                        else:
                            i += 1
                    elif parts[i] == "btime" and board.turn == chess.BLACK:
                        if i + 1 < len(parts):
                            btime = int(parts[i + 1]) / 1000.0
                            time_limit = min(btime / 30, 10.0)  # Conservative time management
                            i += 2
                        else:
                            i += 1
                    else:
                        i += 1
                
                # Search for best move
                try:
                    best_move = engine.search(board, time_limit)
                    if best_move:
                        print(f"bestmove {best_move}")
                    else:
                        # Fallback to any legal move
                        legal_moves = list(board.legal_moves)
                        if legal_moves:
                            print(f"bestmove {legal_moves[0]}")
                        else:
                            print("bestmove 0000")
                except Exception as e:
                    print(f"info string Search error: {e}")
                    # Fallback to any legal move
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        print(f"bestmove {legal_moves[0]}")
                    else:
                        print("bestmove 0000")
                
            elif command == "stop":
                # Stop search (not implemented in this version)
                pass
                
            else:
                print(f"info string Unknown command: {command}")
                
        except EOFError:
            break
        except Exception as e:
            print(f"info string Error: {e}")


if __name__ == "__main__":
    main()