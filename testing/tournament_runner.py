#!/usr/bin/env python3
"""
Tournament Runner - Automated Engine vs Engine Testing

WHY THIS EXISTS: Programmatically test v19.0 vs v18.4 to validate improvements
without requiring Arena GUI setup.

WHAT IT DOES:
- Runs multiple games between two UCI engines
- Handles time controls (blitz, rapid, classical)
- Tracks wins/losses/draws
- Detects timeouts, crashes, illegal moves
- Generates comprehensive statistics

USAGE:
    python tournament_runner.py --engine1 path/to/v19.0 --engine2 path/to/v18.4 --games 30
"""

import chess
import chess.engine
import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class TournamentRunner:
    """Runs automated tournament between two UCI engines"""
    
    def __init__(
        self,
        engine1_cmd: list,
        engine2_cmd: list,
        time_control: Tuple[float, float] = (300, 4),  # 5min+4s
        games: int = 30,
        output_dir: str = "tournament_results"
    ):
        self.engine1_cmd = engine1_cmd
        self.engine2_cmd = engine2_cmd
        self.base_time, self.increment = time_control
        self.total_games = games
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Results tracking
        self.results = {
            "engine1_wins": 0,
            "engine2_wins": 0,
            "draws": 0,
            "engine1_timeouts": 0,
            "engine2_timeouts": 0,
            "engine1_crashes": 0,
            "engine2_crashes": 0,
            "games": []
        }
    
    def run_tournament(self):
        """Run complete tournament"""
        print("=" * 80)
        print(f"TOURNAMENT: {self.total_games} games")
        print("=" * 80)
        print(f"Engine 1: {' '.join(self.engine1_cmd)}")
        print(f"Engine 2: {' '.join(self.engine2_cmd)}")
        print(f"Time Control: {self.base_time/60:.1f}min + {self.increment:.0f}s")
        print("=" * 80)
        
        start_time = time.time()
        
        # Play games alternating colors
        for game_num in range(1, self.total_games + 1):
            # Alternate who plays white
            white_engine = self.engine1_cmd if game_num % 2 == 1 else self.engine2_cmd
            black_engine = self.engine2_cmd if game_num % 2 == 1 else self.engine1_cmd
            
            print(f"\n--- Game {game_num}/{self.total_games} ---")
            print(f"White: {white_engine[-1].split('/')[-1] if white_engine else 'Unknown'}")
            print(f"Black: {black_engine[-1].split('/')[-1] if black_engine else 'Unknown'}")
            
            result = self.play_game(white_engine, black_engine, game_num)
            
            # Track results
            self._record_result(result, game_num % 2 == 1)
            
            # Display running totals
            self._display_progress()
            
            # Brief pause between games
            time.sleep(0.5)
        
        elapsed = time.time() - start_time
        
        # Generate final report
        self._generate_report(elapsed)
    
    def play_game(
        self,
        white_cmd: list,
        black_cmd: list,
        game_num: int
    ) -> Dict:
        """Play a single game between two engines"""
        
        board = chess.Board()
        game_result = {
            "game_number": game_num,
            "white": white_cmd[-1].split('/')[-1] if white_cmd else 'Unknown',
            "black": black_cmd[-1].split('/')[-1] if black_cmd else 'Unknown',
            "result": None,
            "termination": None,
            "moves": 0,
            "pgn": "",
            "white_timeout": False,
            "black_timeout": False,
            "white_crash": False,
            "black_crash": False,
        }
        
        move_list = []
        
        try:
            # Initialize engines
            with chess.engine.SimpleEngine.popen_uci(white_cmd) as white_engine, \
                 chess.engine.SimpleEngine.popen_uci(black_cmd) as black_engine:
                
                # Set up time controls
                white_time = self.base_time
                black_time = self.base_time
                
                move_count = 0
                
                while not board.is_game_over() and move_count < 200:
                    move_count += 1
                    
                    # Select current engine
                    if board.turn == chess.WHITE:
                        current_engine = white_engine
                        remaining_time = white_time
                    else:
                        current_engine = black_engine
                        remaining_time = black_time
                    
                    # Calculate time limit for this move
                    move_time = min(remaining_time / 20, remaining_time - 1)  # Conservative
                    move_time = max(0.1, move_time)  # Minimum 100ms
                    
                    try:
                        # Get move from engine
                        move_start = time.time()
                        result = current_engine.play(
                            board,
                            chess.engine.Limit(time=move_time),
                            info=chess.engine.INFO_ALL
                        )
                        move_elapsed = time.time() - move_start
                        
                        # Update time remaining
                        if board.turn == chess.WHITE:
                            white_time -= move_elapsed
                            white_time += self.increment
                            if white_time < 0:
                                game_result["result"] = "0-1"
                                game_result["termination"] = "time forfeit (white)"
                                game_result["white_timeout"] = True
                                break
                        else:
                            black_time -= move_elapsed
                            black_time += self.increment
                            if black_time < 0:
                                game_result["result"] = "1-0"
                                game_result["termination"] = "time forfeit (black)"
                                game_result["black_timeout"] = True
                                break
                        
                        # Make the move
                        move_list.append(board.san(result.move))
                        board.push(result.move)
                        
                    except chess.engine.EngineTerminatedError:
                        # Engine crashed
                        if board.turn == chess.WHITE:
                            game_result["result"] = "0-1"
                            game_result["termination"] = "crash (white)"
                            game_result["white_crash"] = True
                        else:
                            game_result["result"] = "1-0"
                            game_result["termination"] = "crash (black)"
                            game_result["black_crash"] = True
                        break
                    
                    except Exception as e:
                        print(f"  Error on move {move_count}: {e}")
                        game_result["result"] = "error"
                        game_result["termination"] = f"error: {str(e)}"
                        break
                
                # Check for normal game over
                if board.is_game_over() and game_result["result"] is None:
                    result_str = board.result()
                    game_result["result"] = result_str
                    
                    if board.is_checkmate():
                        game_result["termination"] = "checkmate"
                    elif board.is_stalemate():
                        game_result["termination"] = "stalemate"
                    elif board.is_insufficient_material():
                        game_result["termination"] = "insufficient material"
                    elif board.can_claim_fifty_moves():
                        game_result["termination"] = "fifty-move rule"
                    elif board.can_claim_threefold_repetition():
                        game_result["termination"] = "threefold repetition"
                    else:
                        game_result["termination"] = "unknown"
                
                # Store move count and PGN
                game_result["moves"] = move_count
                game_result["pgn"] = " ".join(move_list)
                
        except Exception as e:
            print(f"  CRITICAL ERROR: {e}")
            game_result["result"] = "error"
            game_result["termination"] = f"critical error: {str(e)}"
        
        # Display result
        print(f"  Result: {game_result['result']} ({game_result['termination']})")
        print(f"  Moves: {game_result['moves']}")
        
        return game_result
    
    def _record_result(self, result: Dict, engine1_is_white: bool):
        """Record game result in tournament statistics"""
        
        self.results["games"].append(result)
        
        # Determine winner
        if result["result"] == "1-0":
            if engine1_is_white:
                self.results["engine1_wins"] += 1
            else:
                self.results["engine2_wins"] += 1
        elif result["result"] == "0-1":
            if engine1_is_white:
                self.results["engine2_wins"] += 1
            else:
                self.results["engine1_wins"] += 1
        elif result["result"] == "1/2-1/2":
            self.results["draws"] += 1
        
        # Track timeouts/crashes
        if result.get("white_timeout"):
            if engine1_is_white:
                self.results["engine1_timeouts"] += 1
            else:
                self.results["engine2_timeouts"] += 1
        
        if result.get("black_timeout"):
            if engine1_is_white:
                self.results["engine2_timeouts"] += 1
            else:
                self.results["engine1_timeouts"] += 1
        
        if result.get("white_crash"):
            if engine1_is_white:
                self.results["engine1_crashes"] += 1
            else:
                self.results["engine2_crashes"] += 1
        
        if result.get("black_crash"):
            if engine1_is_white:
                self.results["engine2_crashes"] += 1
            else:
                self.results["engine1_crashes"] += 1
    
    def _display_progress(self):
        """Display running tournament statistics"""
        total = len(self.results["games"])
        e1_wins = self.results["engine1_wins"]
        e2_wins = self.results["engine2_wins"]
        draws = self.results["draws"]
        
        e1_score = e1_wins + draws * 0.5
        e2_score = e2_wins + draws * 0.5
        
        e1_pct = (e1_score / total) * 100 if total > 0 else 0
        e2_pct = (e2_score / total) * 100 if total > 0 else 0
        
        print(f"\n  Current Standings ({total} games):")
        print(f"    Engine 1: {e1_wins}W {e2_wins}L {draws}D ({e1_score:.1f}/{total} = {e1_pct:.1f}%)")
        print(f"    Engine 2: {e2_wins}W {e1_wins}L {draws}D ({e2_score:.1f}/{total} = {e2_pct:.1f}%)")
        
        if self.results["engine1_timeouts"] > 0 or self.results["engine2_timeouts"] > 0:
            print(f"    Timeouts: E1={self.results['engine1_timeouts']} E2={self.results['engine2_timeouts']}")
        
        if self.results["engine1_crashes"] > 0 or self.results["engine2_crashes"] > 0:
            print(f"    Crashes: E1={self.results['engine1_crashes']} E2={self.results['engine2_crashes']}")
    
    def _generate_report(self, elapsed_time: float):
        """Generate final tournament report"""
        
        print("\n" + "=" * 80)
        print("TOURNAMENT RESULTS")
        print("=" * 80)
        
        total = len(self.results["games"])
        e1_wins = self.results["engine1_wins"]
        e2_wins = self.results["engine2_wins"]
        draws = self.results["draws"]
        
        e1_score = e1_wins + draws * 0.5
        e2_score = e2_wins + draws * 0.5
        
        e1_pct = (e1_score / total) * 100 if total > 0 else 0
        e2_pct = (e2_score / total) * 100 if total > 0 else 0
        
        print(f"\nEngine 1: {self.engine1_cmd[-1].split('/')[-1] if self.engine1_cmd else 'Unknown'}")
        print(f"  Score: {e1_score:.1f}/{total} ({e1_pct:.1f}%)")
        print(f"  Record: {e1_wins}W - {e2_wins}L - {draws}D")
        print(f"  Timeouts: {self.results['engine1_timeouts']}")
        print(f"  Crashes: {self.results['engine1_crashes']}")
        
        print(f"\nEngine 2: {self.engine2_cmd[-1].split('/')[-1] if self.engine2_cmd else 'Unknown'}")
        print(f"  Score: {e2_score:.1f}/{total} ({e2_pct:.1f}%)")
        print(f"  Record: {e2_wins}W - {e1_wins}L - {draws}D")
        print(f"  Timeouts: {self.results['engine2_timeouts']}")
        print(f"  Crashes: {self.results['engine2_crashes']}")
        
        print(f"\nTournament Duration: {elapsed_time/60:.1f} minutes")
        print(f"Average Game Time: {elapsed_time/total:.1f} seconds")
        
        # Determine winner
        print("\n" + "-" * 80)
        if e1_score > e2_score:
            margin = e1_score - e2_score
            print(f"✓ ENGINE 1 WINS by {margin:.1f} points ({e1_pct - e2_pct:.1f}% margin)")
        elif e2_score > e1_score:
            margin = e2_score - e1_score
            print(f"✓ ENGINE 2 WINS by {margin:.1f} points ({e2_pct - e1_pct:.1f}% margin)")
        else:
            print("✓ TOURNAMENT TIED")
        
        # Critical issues
        if self.results["engine1_timeouts"] > 0 or self.results["engine2_timeouts"] > 0:
            print("\n⚠ TIME FORFEIT ISSUES DETECTED:")
            if self.results["engine1_timeouts"] > 0:
                pct = (self.results["engine1_timeouts"] / total) * 100
                print(f"  Engine 1: {self.results['engine1_timeouts']} timeouts ({pct:.1f}%)")
            if self.results["engine2_timeouts"] > 0:
                pct = (self.results["engine2_timeouts"] / total) * 100
                print(f"  Engine 2: {self.results['engine2_timeouts']} timeouts ({pct:.1f}%)")
        
        if self.results["engine1_crashes"] > 0 or self.results["engine2_crashes"] > 0:
            print("\n⚠ STABILITY ISSUES DETECTED:")
            if self.results["engine1_crashes"] > 0:
                pct = (self.results["engine1_crashes"] / total) * 100
                print(f"  Engine 1: {self.results['engine1_crashes']} crashes ({pct:.1f}%)")
            if self.results["engine2_crashes"] > 0:
                pct = (self.results["engine2_crashes"] / total) * 100
                print(f"  Engine 2: {self.results['engine2_crashes']} crashes ({pct:.1f}%)")
        
        print("=" * 80)
        
        # Save results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"tournament_{timestamp}.json"
        
        report = {
            "timestamp": timestamp,
            "engine1": ' '.join(self.engine1_cmd),
            "engine2": ' '.join(self.engine2_cmd),
            "time_control": {"base": self.base_time, "increment": self.increment},
            "total_games": total,
            "elapsed_time": elapsed_time,
            "results": self.results,
            "summary": {
                "engine1_score": e1_score,
                "engine2_score": e2_score,
                "engine1_percentage": e1_pct,
                "engine2_percentage": e2_pct,
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run automated tournament between two UCI chess engines"
    )
    parser.add_argument(
        "--engine1",
        required=True,
        nargs='+',
        help="Command for engine 1 (e.g., python src/v7p3r_uci.py)"
    )
    parser.add_argument(
        "--engine2",
        required=True,
        nargs='+',
        help="Command for engine 2 (e.g., python path/to/v18.4/v7p3r_uci.py)"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=30,
        help="Number of games to play (default: 30)"
    )
    parser.add_argument(
        "--time",
        type=float,
        default=300,
        help="Base time in seconds (default: 300 = 5 minutes)"
    )
    parser.add_argument(
        "--increment",
        type=float,
        default=4,
        help="Increment per move in seconds (default: 4)"
    )
    parser.add_argument(
        "--output",
        default="tournament_results",
        help="Output directory for results (default: tournament_results)"
    )
    
    args = parser.parse_args()
    
    # Validate engine commands (check last argument is a file)
    if not Path(args.engine1[-1]).exists():
        print(f"ERROR: Engine 1 script not found: {args.engine1[-1]}")
        sys.exit(1)
    
    if not Path(args.engine2[-1]).exists():
        print(f"ERROR: Engine 2 script not found: {args.engine2[-1]}")
        sys.exit(1)
    
    # Run tournament
    runner = TournamentRunner(
        engine1_cmd=args.engine1,
        engine2_cmd=args.engine2,
        time_control=(args.time, args.increment),
        games=args.games,
        output_dir=args.output
    )
    
    try:
        runner.run_tournament()
    except KeyboardInterrupt:
        print("\n\nTournament interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
