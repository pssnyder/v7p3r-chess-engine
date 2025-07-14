# testing/test_vs_stockfish.py

"""Test V7P3R Engine vs Stockfish
Runs automated games to test win rate against Stockfish at ELO 400.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import chess
from v7p3r_engine import V7P3REngine
from v7p3r_stockfish import StockfishHandler
from v7p3r_config import V7P3RConfig
from metrics import ChessMetrics

class AutomatedTester:
    def __init__(self, config_file="config.json"):
        self.config = V7P3RConfig(config_file)
        self.metrics = ChessMetrics()
        
    def run_test_games(self, num_games=10, stockfish_elo=400, time_limit=30.0):
        """Run automated test games"""
        print(f"=== Running {num_games} test games vs Stockfish ELO {stockfish_elo} ===")
        
        # Initialize engines
        v7p3r = V7P3REngine()
        
        # Configure Stockfish
        stockfish_config = self.config.get_stockfish_config()
        stockfish_config['elo_rating'] = stockfish_elo
        stockfish = StockfishHandler(self.config)
        stockfish.set_elo(stockfish_elo)
        
        if not stockfish.is_available():
            print("Error: Stockfish not available")
            return
        
        results = {'wins': 0, 'losses': 0, 'draws': 0}
        game_times = []
        
        for game_num in range(num_games):
            print(f"\nGame {game_num + 1}/{num_games}")
            
            # Alternate colors
            v7p3r_is_white = (game_num % 2 == 0)
            
            result, game_time = self.play_single_game(
                v7p3r, stockfish, v7p3r_is_white, time_limit
            )
            
            game_times.append(game_time)
            
            if result == 'win':
                results['wins'] += 1
                print("✓ V7P3R WINS!")
            elif result == 'loss':
                results['losses'] += 1
                print("✗ V7P3R loses")
            else:
                results['draws'] += 1
                print("= Draw")
        
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
        print(f"\nGoal (10% win rate vs ELO 400): {'✓ MET' if goal_met else '✗ NOT MET'}")
        
        return results, win_rate_decisive
    
    def play_single_game(self, v7p3r, stockfish, v7p3r_is_white, time_limit):
        """Play a single automated game"""
        board = chess.Board()
        game_start = time.time()
        moves_played = 0
        max_moves = 200  # Prevent infinite games
        
        print(f"  V7P3R playing {'White' if v7p3r_is_white else 'Black'}")
        
        while not board.is_game_over() and moves_played < max_moves:
            current_player_is_v7p3r = (board.turn == chess.WHITE) == v7p3r_is_white
            
            if current_player_is_v7p3r:
                # V7P3R's turn
                move = v7p3r.find_move(board, time_limit)
                player_name = "V7P3R"
            else:
                # Stockfish's turn
                move = stockfish.get_move(board)
                player_name = "Stockfish"
            
            if move and move in board.legal_moves:
                board.push(move)
                moves_played += 1
                
                if moves_played <= 10 or moves_played % 20 == 0:
                    print(f"    Move {moves_played}: {player_name} - {move}")
            else:
                print(f"    Invalid move from {player_name}: {move}")
                # Forfeit the game
                return 'loss' if current_player_is_v7p3r else 'win', time.time() - game_start
        
        game_time = time.time() - game_start
        
        # Determine result
        if board.is_game_over():
            if board.is_checkmate():
                winner_is_white = not board.turn  # The player who just moved won
                if winner_is_white == v7p3r_is_white:
                    result = 'win'
                else:
                    result = 'loss'
            else:
                result = 'draw'
        else:
            # Game too long
            result = 'draw'
        
        print(f"    Game finished: {result} in {moves_played} moves ({game_time:.1f}s)")
        return result, game_time

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
    
    args = parser.parse_args()
    
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
