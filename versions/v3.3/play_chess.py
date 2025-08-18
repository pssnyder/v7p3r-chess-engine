# play_chess.py

"""Main Entry Point for V7P3R Chess Engine
Run this file to start chess games between v7p3r and other engines.
"""

import sys
import argparse
from v7p3r_game import ChessGame
from v7p3r_config import V7P3RConfig

def main():
    parser = argparse.ArgumentParser(description='V7P3R Chess Engine')
    parser.add_argument('--config', '-c', default='config.json',
                       help='Configuration file to use (default: config.json)')
    parser.add_argument('--games', '-g', type=int,
                       help='Number of games to play (overrides config)')
    parser.add_argument('--white', '-w', choices=['v7p3r', 'stockfish'],
                       help='White player (overrides config)')
    parser.add_argument('--black', '-b', choices=['v7p3r', 'stockfish'],
                       help='Black player (overrides config)')
    parser.add_argument('--depth', '-d', type=int,
                       help='Search depth for v7p3r (overrides config)')
    parser.add_argument('--stockfish-elo', type=int,
                       help='Stockfish ELO rating (overrides config)')
    parser.add_argument('--background', action='store_true',
                       help='Run in background mode without visual display (default now)')
    
    args = parser.parse_args()
    
    try:
        # Load and modify configuration if needed
        config = V7P3RConfig(args.config)
        
        # Apply command line overrides
        if args.games:
            config.config['game_config']['game_count'] = args.games
        if args.white:
            config.config['game_config']['white_player'] = args.white
        if args.black:
            config.config['game_config']['black_player'] = args.black
        if args.depth:
            config.config['engine_config']['depth'] = args.depth
        if args.stockfish_elo:
            config.config['stockfish_config']['elo_rating'] = args.stockfish_elo
        
        # Print configuration
        print("=== V7P3R Chess Engine ===")
        print(f"Configuration: {args.config}")
        print(f"Games to play: {config.get_setting('game_config', 'game_count')}")
        print(f"White player: {config.get_setting('game_config', 'white_player')}")
        print(f"Black player: {config.get_setting('game_config', 'black_player')}")
        print(f"V7P3R depth: {config.get_setting('engine_config', 'depth')}")
        print(f"Stockfish ELO: {config.get_setting('stockfish_config', 'elo_rating')}")
        print()
        
        # Create and run the game
        game = ChessGame(args.config)
        game.run_games()
        
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
