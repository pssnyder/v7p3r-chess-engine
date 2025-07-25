# testing/test_vs_stockfish.py

"""Test V7P3R Engine vs Stockfish
Runs automated games to test win rate against Stockfish at ELO 400.
Uses the main ChessGame class to ensure proper PGN recording and metrics collection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import chess
from v7p3r_game import ChessGame
from v7p3r_config import V7P3RConfig
from metrics import ChessMetrics

class AutomatedTester:
    def __init__(self, config_file="config.json"):
        self.config = V7P3RConfig(config_file)
        self.metrics = ChessMetrics()
        
    def run_test_games(self, num_games=10, stockfish_elo=400, time_limit=30.0):
        """Run automated test games using the main ChessGame system"""
        print(f"=== Running {num_games} test games vs Stockfish ELO {stockfish_elo} ===")
        
        # Store original config values
        original_game_count = self.config.get_setting('game_config', 'game_count')
        original_elo = self.config.get_setting('stockfish_config', 'elo_rating')
        
        # Update config for testing
        self.config.config['game_config']['game_count'] = num_games
        self.config.config['stockfish_config']['elo_rating'] = stockfish_elo
        self.config.config['stockfish_config']['depth'] = 4  # Reasonable depth for testing
        
        # Ensure v7p3r alternates colors
        results = {'wins': 0, 'losses': 0, 'draws': 0}
        game_times = []
        
        try:
            for game_num in range(num_games):
                print(f"\nGame {game_num + 1}/{num_games}")
                
                # Alternate colors each game
                if game_num % 2 == 0:
                    # V7P3R plays White
                    self.config.config['game_config']['white_player'] = 'v7p3r'
                    self.config.config['game_config']['black_player'] = 'stockfish'
                    v7p3r_color = "White"
                else:
                    # V7P3R plays Black
                    self.config.config['game_config']['white_player'] = 'stockfish'
                    self.config.config['game_config']['black_player'] = 'v7p3r'
                    v7p3r_color = "Black"
                
                print(f"  V7P3R playing {v7p3r_color}")
                
                # Create game with updated config
                self.config.config['game_config']['game_count'] = 1
                
                # Write the config temporarily and reload it
                import tempfile
                import json
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(self.config.config, f, indent=2)
                    temp_config_path = f.name
                
                try:
                    # Create a custom game class that will display move, duration, and evaluation
                    game = self.create_monitored_game(temp_config_path, v7p3r_color)
                    
                    # Run single game and time it
                    game_start = time.time()
                    game.run_games()  # This will run exactly 1 game
                    game_time = time.time() - game_start
                    game_times.append(game_time)
                    
                    # Get the result from the last game in the database
                    result = self.get_last_game_result(game_num % 2 == 0)  # True if v7p3r was white
                    
                    if result == 'win':
                        results['wins'] += 1
                        print("Γ£ô V7P3R WINS!")
                    elif result == 'loss':
                        results['losses'] += 1
                        print("Γ£ù V7P3R loses")
                    else:
                        results['draws'] += 1
                        print("= Draw")
                        
                except Exception as e:
                    print(f"  Error in game {game_num + 1}: {e}")
                    results['losses'] += 1
                    print("Γ£ù V7P3R loses")
                    game_times.append(30.0)  # Default time
                        
                finally:
                    # Clean up temp file
                    import os
                    try:
                        os.unlink(temp_config_path)
                    except:
                        pass
                    
        finally:
            # Restore original config
            self.config.config['game_config']['game_count'] = original_game_count
            self.config.config['stockfish_config']['elo_rating'] = original_elo
        
        # Calculate statistics
        total_games = sum(results.values())
        non_draw_games = results['wins'] + results['losses']
        
        win_rate_total = (results['wins'] / total_games * 100) if total_games > 0 else 0
        win_rate_decisive = (results['wins'] / non_draw_games * 100) if non_draw_games > 0 else 0
        
        avg_game_time = sum(game_times) / len(game_times) if game_times else 0
        
        print(f"\n=== Test Results ===")
        print(f"Games played: {total_games}")
        print(f"V7P3R: {results['wins']}W {results['losses']}L {results['draws']}D")
        print(f"Win rate (all games): {win_rate_total:.1f}%")
        print(f"Win rate (decisive games): {win_rate_decisive:.1f}%")
        print(f"Average game time: {avg_game_time:.1f}s")
        
        # Check if goal is met
        goal_met = win_rate_decisive >= 10.0
        print(f"\nGoal (10% win rate vs ELO 400): {'Γ£ô MET' if goal_met else 'Γ£ù NOT MET'}")
        
        return results, win_rate_decisive
    
    def create_monitored_game(self, config_path, v7p3r_color):
        """Create a game with a custom run_single_game method that displays move, duration, and evaluation"""
        from v7p3r_game import ChessGame
        import chess
        import pygame
        
        # Create a subclass of ChessGame with enhanced move reporting
        class MonitoredChessGame(ChessGame):
            def run_single_game(self):
                """Override run_single_game to add enhanced move reporting"""
                self.reset_for_new_game()
                
                # Record game start
                white_config = self.config.get_engine_config() if self.white_player == 'v7p3r' else None
                black_config = self.config.get_engine_config() if self.black_player == 'v7p3r' else None
                self.game_id = self.metrics.record_game_start(
                    self.white_player, self.black_player, white_config, black_config
                )
                
                self.game_start_time = time.time()
                running = True
                
                while running and not self._is_game_over():
                    # Handle pygame events
                    if not self.headless:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                                break
                    
                    if not running:
                        break
                    
                    # Determine current player
                    current_player = self.white_player if self.board.turn == chess.WHITE else self.black_player
                    
                    print(f"\nMove {self.move_number} - {current_player} to play")
                    
                    # Get move from appropriate engine
                    move_start_time = time.time()
                    move = self.get_engine_move(current_player)
                    move_time = time.time() - move_start_time
                    
                    if move is None:
                        print(f"No move available for {current_player}")
                        break
                    
                    # Get evaluation before making the move
                    v7p3r_eval = self.v7p3r_engine.get_evaluation(self.board)
                    
                    # Make the move
                    if move in self.board.legal_moves:
                        # Record move analysis
                        evaluation = self.v7p3r_engine.get_evaluation(self.board) if current_player == 'v7p3r' else None
                        player_color = 'white' if self.board.turn == chess.WHITE else 'black'
                        
                        self.metrics.record_move(
                            self.game_id, self.move_number, player_color, move.uci(),
                            evaluation_score=evaluation, search_time=move_time
                        )
                        
                        # Enhanced move reporting - show move, duration, and evaluation
                        eval_sign = "+" if v7p3r_eval > 0 else "" if v7p3r_eval == 0 else "-"
                        eval_abs = abs(v7p3r_eval)
                        eval_display = f"{eval_sign}{eval_abs:.2f}"
                        
                        # Add evaluation interpretation
                        eval_interp = ""
                        if abs(v7p3r_eval) < 0.3:
                            eval_interp = "Equal"
                        elif abs(v7p3r_eval) < 0.7:
                            eval_interp = "Slight advantage"
                        elif abs(v7p3r_eval) < 1.5:
                            eval_interp = "Advantage"
                        elif abs(v7p3r_eval) < 3.0:
                            eval_interp = "Clear advantage"
                        elif abs(v7p3r_eval) < 5.0:
                            eval_interp = "Winning position"
                        else:
                            eval_interp = "Decisive advantage"
                            
                        # Add who has the advantage
                        if v7p3r_eval > 0.3:
                            eval_interp += " for White"
                        elif v7p3r_eval < -0.3:
                            eval_interp += " for Black"
                            
                        # Format based on whether this is V7P3R's move or not
                        if current_player == 'v7p3r':
                            print(f"V7P3R Move: {move} | Time: {move_time:.2f}s | Evaluation: {eval_display} ({eval_interp})")
                        else:
                            print(f"Stockfish Move: {move} | V7P3R Evaluation: {eval_display} ({eval_interp})")
                        
                        self.board.push(move)
                        
                        if self.board.turn == chess.WHITE:
                            self.move_number += 1
                        
                        # Update display
                        self.update_display()
                        self.write_pgn()
                        
                    else:
                        print(f"Illegal move attempted: {move}")
                        break
                
                # Game finished
                self.finish_game()
                
        # Create and return our enhanced game
        return MonitoredChessGame(config_path, headless=True)
    
    def get_last_game_result(self, v7p3r_was_white):
        """Get the result of the last game from the database"""
        try:
            # Query the last game from the database
            recent_games = self.metrics.get_recent_games(1)
            if not recent_games:
                return 'loss'  # Default to loss if no game found
            
            game = recent_games[0]
            result = game[4]  # result column
            white_player = game[2]  # white_player column
            
            # Determine if v7p3r won
            if result == 'draw':
                return 'draw'
            elif result == 'white_wins':
                return 'win' if (white_player == 'v7p3r') else 'loss'
            elif result == 'black_wins':
                return 'win' if (white_player != 'v7p3r') else 'loss'
            else:
                return 'loss'  # Default
                
        except Exception as e:
            print(f"Error getting game result: {e}")
            return 'loss'  # Default to loss on error

def test_single_game():
    """Run a single test game with enhanced move reporting"""
    tester = AutomatedTester()
    
    # Create temporary config with v7p3r as white
    config = V7P3RConfig()
    config.config['game_config']['game_count'] = 1
    config.config['game_config']['white_player'] = 'v7p3r'
    config.config['game_config']['black_player'] = 'stockfish'
    config.config['stockfish_config']['elo_rating'] = 400
    
    import tempfile
    import json
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config.config, f, indent=2)
        temp_config_path = f.name
    
    try:
        # Create and run game with move/eval display
        game = tester.create_monitored_game(temp_config_path, "White")
        game.run_games()
    except Exception as e:
        print(f"Error running test game: {e}")
    finally:
        # Clean up
        import os
        try:
            os.unlink(temp_config_path)
        except:
            pass

def main():
    """Run the test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test V7P3R vs Stockfish')
    parser.add_argument('--games', '-g', type=int, default=10,
                       help='Number of test games (default: 10)')
    parser.add_argument('--elo', '-e', type=int, default=400,
                       help='Stockfish ELO rating (default: 400)')
    parser.add_argument('--time-limit', '-t', type=float, default=30.0,
                       help='Time limit per move in seconds (default: 30)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Display more detailed output')
    parser.add_argument('--single', '-s', action='store_true',
                       help='Run a single test game with enhanced move reporting')
    
    args = parser.parse_args()
    
    # Run a single test game if requested
    if args.single:
        test_single_game()
        return 0
    
    tester = AutomatedTester()
    try:
        result = tester.run_test_games(
            num_games=args.games,
            stockfish_elo=args.elo,
            time_limit=args.time_limit
        )
        
        if result:
            results, win_rate = result
            # Return appropriate exit code
            return 0 if win_rate >= 10.0 else 1
        else:
            print("Error: Test failed to complete")
            return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
