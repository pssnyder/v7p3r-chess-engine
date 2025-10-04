#!/usr/bin/env python3
"""
V7P3R Arena Testing: V12.6 vs V12.2
Simulates tournament games between the two versions
"""

import subprocess
import time
import random
import chess
import json
from pathlib import Path


class ArenaMatch:
    def __init__(self):
        self.base_path = Path("s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine")
        self.v12_2_path = self.base_path / "dist" / "V7P3R_v12.2.exe"
        self.v12_6_path = self.base_path / "dist" / "V7P3R_v12.6.exe"
        self.games_played = []
        
    def start_engine(self, engine_path):
        """Start an engine process"""
        return subprocess.Popen(
            str(engine_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
    def send_command(self, engine, command):
        """Send command to engine"""
        engine.stdin.write(f"{command}\n")
        engine.stdin.flush()
        
    def get_move(self, engine, board, time_limit_ms=3000):
        """Get a move from the engine"""
        # Set position
        if len(board.move_stack) == 0:
            self.send_command(engine, "position startpos")
        else:
            moves = " ".join([move.uci() for move in board.move_stack])
            self.send_command(engine, f"position startpos moves {moves}")
            
        # Request move
        self.send_command(engine, f"go movetime {time_limit_ms}")
        
        # Read response
        start_time = time.time()
        while time.time() - start_time < (time_limit_ms / 1000.0) + 2:
            try:
                line = engine.stdout.readline().strip()
                if line.startswith("bestmove"):
                    move_str = line.split()[1]
                    if move_str == "(none)" or move_str == "none":
                        return None
                    try:
                        return chess.Move.from_uci(move_str)
                    except:
                        return None
            except:
                continue
        return None
        
    def play_game(self, white_engine, black_engine, white_name, black_name, time_per_move=3000):
        """Play a complete game between two engines"""
        board = chess.Board()
        move_count = 0
        max_moves = 200  # Prevent infinite games
        
        print(f"🎯 Starting game: {white_name} (White) vs {black_name} (Black)")
        
        # Initialize engines
        for engine in [white_engine, black_engine]:
            self.send_command(engine, "uci")
            time.sleep(0.1)
            self.send_command(engine, "isready")
            time.sleep(0.1)
            self.send_command(engine, "ucinewgame")
            time.sleep(0.1)
            
        game_moves = []
        
        while not board.is_game_over() and move_count < max_moves:
            current_engine = white_engine if board.turn == chess.WHITE else black_engine
            current_name = white_name if board.turn == chess.WHITE else black_name
            
            print(f"Move {move_count + 1}: {current_name} to move", end=" ")
            
            start_time = time.time()
            move = self.get_move(current_engine, board, time_per_move)
            think_time = time.time() - start_time
            
            if move is None or move not in board.legal_moves:
                print(f"❌ {current_name} failed to provide legal move!")
                result = "1-0" if board.turn == chess.BLACK else "0-1"
                break
                
            board.push(move)
            game_moves.append(move.uci())
            move_count += 1
            
            print(f"-> {move.uci()} ({think_time:.2f}s)")
            
        # Determine result
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                result = "0-1"  # Black wins
                winner = black_name
            else:
                result = "1-0"  # White wins
                winner = white_name
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
            result = "1/2-1/2"  # Draw
            winner = "Draw"
        elif move_count >= max_moves:
            result = "1/2-1/2"  # Draw by move limit
            winner = "Draw (Move Limit)"
        else:
            result = "1/2-1/2"  # Draw by engine failure
            winner = "Draw (Engine Error)"
            
        print(f"🏁 Game finished: {result} - Winner: {winner}")
        print(f"📊 Total moves: {move_count}")
        
        outcome = board.outcome()
        termination = outcome.termination.name if outcome else "ENGINE_ERROR"
        
        return {
            "white": white_name,
            "black": black_name,
            "result": result,
            "winner": winner,
            "moves": game_moves,
            "move_count": move_count,
            "termination": termination
        }
        
    def run_arena_tournament(self, num_games=6):
        """Run a small tournament between V12.6 and V12.2"""
        print("🏆 V7P3R ARENA TOURNAMENT")
        print("="*40)
        print(f"🎮 Games to play: {num_games}")
        print(f"⚪ V12.2: {self.v12_2_path}")
        print(f"⚫ V12.6: {self.v12_6_path}")
        print()
        
        # Verify engines exist
        if not self.v12_2_path.exists():
            print(f"❌ V12.2 engine not found!")
            return
        if not self.v12_6_path.exists():
            print(f"❌ V12.6 engine not found!")
            return
            
        v12_2_wins = 0
        v12_6_wins = 0
        draws = 0
        
        for game_num in range(1, num_games + 1):
            print(f"\n🎯 GAME {game_num}/{num_games}")
            print("-" * 30)
            
            # Alternate colors
            if game_num % 2 == 1:
                # V12.6 plays white
                white_engine = self.start_engine(self.v12_6_path)
                black_engine = self.start_engine(self.v12_2_path)
                white_name = "V12.6"
                black_name = "V12.2"
            else:
                # V12.2 plays white
                white_engine = self.start_engine(self.v12_2_path)
                black_engine = self.start_engine(self.v12_6_path)
                white_name = "V12.2"
                black_name = "V12.6"
                
            try:
                game_result = self.play_game(white_engine, black_engine, white_name, black_name)
                self.games_played.append(game_result)
                
                # Update scores
                if game_result["winner"] == "V12.2":
                    v12_2_wins += 1
                elif game_result["winner"] == "V12.6":
                    v12_6_wins += 1
                else:
                    draws += 1
                    
            except Exception as e:
                print(f"❌ Game {game_num} error: {e}")
                draws += 1
                
            finally:
                try:
                    white_engine.terminate()
                    black_engine.terminate()
                    white_engine.wait(timeout=2)
                    black_engine.wait(timeout=2)
                except:
                    pass
                    
            # Show current standings
            total_games = game_num
            print(f"\n📊 STANDINGS AFTER {total_games} GAMES:")
            print(f"V12.2: {v12_2_wins} wins ({v12_2_wins/total_games*100:.1f}%)")
            print(f"V12.6: {v12_6_wins} wins ({v12_6_wins/total_games*100:.1f}%)")
            print(f"Draws: {draws} ({draws/total_games*100:.1f}%)")
            
        # Final report
        print("\n" + "="*50)
        print("🏆 FINAL TOURNAMENT RESULTS")
        print("="*50)
        print(f"Total Games: {num_games}")
        print(f"V12.2 Score: {v12_2_wins} wins ({v12_2_wins/num_games*100:.1f}%)")
        print(f"V12.6 Score: {v12_6_wins} wins ({v12_6_wins/num_games*100:.1f}%)")
        print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
        
        if v12_6_wins > v12_2_wins:
            print("🎉 V12.6 WINS THE TOURNAMENT!")
            performance = "BETTER"
        elif v12_2_wins > v12_6_wins:
            print("🔄 V12.2 wins - V12.6 needs improvement")
            performance = "WORSE"
        else:
            print("🤝 Tournament tied - roughly equal performance")
            performance = "EQUAL"
            
        # Save results
        results_file = self.base_path / "testing" / f"arena_v12_6_vs_v12_2_{int(time.time())}.json"
        tournament_data = {
            "tournament_summary": {
                "total_games": num_games,
                "v12_2_wins": v12_2_wins,
                "v12_6_wins": v12_6_wins,
                "draws": draws,
                "v12_6_performance": performance
            },
            "games": self.games_played
        }
        
        with open(results_file, 'w') as f:
            json.dump(tournament_data, f, indent=2)
            
        print(f"\n📁 Results saved to: {results_file}")
        return performance


if __name__ == "__main__":
    arena = ArenaMatch()
    arena.run_arena_tournament(num_games=6)