#!/usr/bin/env python3
"""
VPR Chaos Capitalization Engine v6.0 - UCI Interface

UCI communication for VPR v6.0 Revolutionary Dual-Brain Chess Engine.
Features asymmetric thinking: "We create chaos, they respond traditionally, we capitalize."

Architecture: Dual-Brain Search System (Chaos Brain + Opponent Model)
Technology: Position-aware algorithm selection with MCTS for ultra-chaotic positions
Philosophy: Traditional engines assume the opponent thinks like them. We assume they don't.
"""

import sys
import chess
from vpr import VPREngine


def uci_print(message):
    """Print with immediate flush for Arena compatibility"""
    print(message)
    sys.stdout.flush()


def main():
    """UCI interface for VPR Chaos Capitalization Engine v6.0"""
    engine = VPREngine()
    board = chess.Board()
    
    # Set unbuffered output for Arena compatibility
    sys.stdout = sys.__stdout__
    
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
                uci_print("id name VPR Chaos Capitalization v6.0")
                uci_print("id author V7P3R Project")
                uci_print("option name Chaos_Threshold type spin default 80 min 50 max 200")
                uci_print("option name MCTS_Time_Threshold type spin default 1000 min 100 max 5000")
                uci_print("option name Complexity_Bonus type spin default 25 min 0 max 100")
                uci_print("uciok")
            
            elif command == "isready":
                uci_print("readyok")
            
            elif command == "ucinewgame":
                board = chess.Board()
                # Reset engine state
                engine = VPREngine()
                uci_print("info string VPR v6.0: Revolutionary dual-brain chaos capitalization ready")
            
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
                        uci_print("info string Error: Invalid FEN")
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
                            else:
                                uci_print(f"info string Error: Illegal move {move_str}")
                                break
                        except:
                            uci_print(f"info string Error: Invalid move {move_str}")
                            break
            
            elif command == "go":
                # Parse go command
                time_limit = 3.0  # Default
                depth = None
                
                i = 1
                while i < len(parts):
                    if parts[i] == "movetime":
                        if i + 1 < len(parts):
                            try:
                                time_limit = int(parts[i + 1]) / 1000.0  # Convert ms to seconds
                            except:
                                pass
                        i += 2
                    elif parts[i] == "depth":
                        if i + 1 < len(parts):
                            try:
                                depth = int(parts[i + 1])
                            except:
                                pass
                        i += 2
                    elif parts[i] == "wtime" and board.turn == chess.WHITE:
                        if i + 1 < len(parts):
                            try:
                                wtime = int(parts[i + 1])
                                # Use a fraction of remaining time
                                time_limit = min(wtime / 25000.0, 8.0)  # Use 1/25 of time, max 8s
                            except:
                                pass
                        i += 2
                    elif parts[i] == "btime" and board.turn == chess.BLACK:
                        if i + 1 < len(parts):
                            try:
                                btime = int(parts[i + 1])
                                time_limit = min(btime / 25000.0, 8.0)
                            except:
                                pass
                        i += 2
                    elif parts[i] == "winc" and board.turn == chess.WHITE:
                        if i + 1 < len(parts):
                            try:
                                winc = int(parts[i + 1])
                                time_limit += winc / 2000.0  # Add half the increment
                            except:
                                pass
                        i += 2
                    elif parts[i] == "binc" and board.turn == chess.BLACK:
                        if i + 1 < len(parts):
                            try:
                                binc = int(parts[i + 1])
                                time_limit += binc / 2000.0
                            except:
                                pass
                        i += 2
                    else:
                        i += 1
                
                # Ensure minimum time
                time_limit = max(time_limit, 0.1)
                
                try:
                    # Search for best move
                    best_move = engine.search(board, time_limit=time_limit, depth=depth)
                    if best_move and best_move != chess.Move.null():
                        uci_print(f"bestmove {best_move}")
                    else:
                        # Fallback to any legal move
                        legal_moves = list(board.legal_moves)
                        if legal_moves:
                            uci_print(f"bestmove {legal_moves[0]}")
                        else:
                            uci_print("bestmove 0000")
                except Exception as e:
                    uci_print(f"info string Error in search: {e}")
                    # Fallback to any legal move
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        uci_print(f"bestmove {legal_moves[0]}")
                    else:
                        uci_print("bestmove 0000")
            
            elif command == "stop":
                # For now, we don't support stopping mid-search
                # Just acknowledge
                pass
            
            elif command == "setoption":
                # Parse setoption command
                if len(parts) >= 4 and parts[1] == "name":
                    option_name = parts[2]
                    if len(parts) >= 5 and parts[3] == "value":
                        option_value = parts[4]
                        if option_name == "Chaos_Threshold":
                            try:
                                threshold = int(option_value)
                                engine.chaos_threshold = max(50, min(200, threshold))
                                uci_print(f"info string Chaos threshold set to {engine.chaos_threshold}")
                            except:
                                pass
                        elif option_name == "MCTS_Time_Threshold":
                            try:
                                mcts_time = int(option_value) / 1000.0  # Convert ms to seconds
                                engine.mcts_time_threshold = max(0.1, min(5.0, mcts_time))
                                uci_print(f"info string MCTS time threshold set to {engine.mcts_time_threshold:.1f}s")
                            except:
                                pass
                        elif option_name == "Complexity_Bonus":
                            try:
                                bonus = int(option_value)
                                engine.complexity_bonus = max(0, min(100, bonus))
                                uci_print(f"info string Complexity bonus set to {engine.complexity_bonus}")
                            except:
                                pass
            
        except EOFError:
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            # Silently continue on errors for Arena compatibility
            uci_print(f"info string Error: {e}")


if __name__ == "__main__":
    main()