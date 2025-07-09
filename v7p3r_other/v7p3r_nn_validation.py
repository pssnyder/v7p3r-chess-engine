# v7p3r_nn_validation.py

import os
import argparse
import datetime
import chess
import chess.pgn
import time
import yaml
from v7p3r_nn import v7p3rNeuralNetwork
from v7p3r import v7p3rEngine
from v7p3r_stockfish_handler import StockfishHandler

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def play_game(white_engine, black_engine, game_options=None):
    """
    Play a full game between two engines and return the game record
    
    Args:
        white_engine: Engine playing as white
        black_engine: Engine playing as black
        game_options: Optional game configuration options
        
    Returns:
        chess.pgn.Game object representing the completed game
    """
    if game_options is None:
        game_options = {}
    
    # Create a new board
    board = chess.Board()
    
    # Create a new game
    game = chess.pgn.Game()
    game.headers["Event"] = "v7p3r NN Evaluation"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = game_options.get("white_name", "v7p3r Neural Network")
    game.headers["Black"] = game_options.get("black_name", "Opponent")
    
    # Set up the main variation
    node = game
    
    # Initialize engines with the starting position
    if hasattr(white_engine, 'reset'):
        white_engine.reset(board)
    if hasattr(black_engine, 'reset'):
        black_engine.reset(board)
    
    # Play the game
    while not board.is_game_over(claim_draw=True):
        # Determine which engine's turn it is
        if board.turn == chess.WHITE:
            engine = white_engine
            player_color = chess.WHITE
        else:
            engine = black_engine
            player_color = chess.BLACK
        
        # Get the move from the engine
        start_time = time.time()
        
        try:
            # Handle different engine interfaces
            if hasattr(engine, 'search'):
                move = engine.search(board, player_color)
            elif hasattr(engine, 'get_best_move'):
                uci_move = engine.get_best_move(board.fen())
                move = chess.Move.from_uci(uci_move)
            else:
                break
        except Exception as e:
            break
        
        elapsed = time.time() - start_time
        
        # Make the move
        board.push(move)
        node = node.add_variation(move)
        
        # Add evaluation as a comment if available
        if hasattr(engine, 'evaluate_position'):
            try:
                eval_score = engine.evaluate_position(board.fen())
                node.comment = f"Eval: {eval_score:.2f}"
            except:
                pass
    
    # Add the game result
    result = board.result(claim_draw=True)
    game.headers["Result"] = result
    
    return game

def save_game(game, output_dir="games"):
    """Save a game to a PGN file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = get_timestamp()
    output_file = os.path.join(output_dir, f"eval_game_{timestamp}.pgn")
    
    with open(output_file, "w") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Evaluate the v7p3r Neural Network engine")
    
    parser.add_argument("--opponent", default="v7p3r", choices=["v7p3r", "stockfish"], 
                        help="Opponent engine to play against")
    parser.add_argument("--games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--nn_config", default="config/v7p3r_nn_config.yaml", 
                        help="Path to NN configuration file")
    parser.add_argument("--v7p3r_config", default="config/v7p3r_config.yaml", 
                        help="Path to v7p3r configuration file")
    parser.add_argument("--stockfish_config", default="config/stockfish_handler.yaml", 
                        help="Path to Stockfish configuration file")
    parser.add_argument("--alternate", action="store_true", 
                        help="Alternate colors between games")
    
    args = parser.parse_args()
    
    # Initialize the Neural Network engine
    nn_engine = v7p3rNeuralNetwork(config_path=args.nn_config)
    
    # Initialize the opponent engine
    opponent_engine = None
    opponent_name = ""
    
    if args.opponent == "v7p3r":
        # Load v7p3r configuration
        try:
            with open(args.v7p3r_config, 'r') as f:
                v7p3r_config = yaml.safe_load(f)
        except Exception as e:
            raise
        
        # Initialize v7p3r engine
        opponent_engine = v7p3rEngine()
        opponent_name = "v7p3r"
    
    elif args.opponent == "stockfish":
        # Load Stockfish configuration
        try:
            with open(args.stockfish_config, 'r') as f:
                stockfish_config = yaml.safe_load(f)
        except Exception as e:
            stockfish_config = {}
        
        stockfish_path = stockfish_config.get('stockfish_path', '')
        if not stockfish_path or not os.path.exists(stockfish_path):
            return
        
        # Initialize Stockfish engine
        opponent_engine = StockfishHandler(stockfish_config=stockfish_path)
        opponent_name = "Stockfish"
    
    if opponent_engine is None:
        return
    
    # Play the specified number of games
    results = {"white_wins": 0, "black_wins": 0, "draws": 0}
    nn_results = {"wins": 0, "losses": 0, "draws": 0}
    
    try:
        for game_num in range(1, args.games + 1):
            # Determine engine colors
            if args.alternate:
                nn_plays_white = (game_num % 2 == 1)
            else:
                nn_plays_white = True
            
            white_engine = nn_engine if nn_plays_white else opponent_engine
            black_engine = opponent_engine if nn_plays_white else nn_engine
            
            white_name = "v7p3r Neural Network" if nn_plays_white else opponent_name
            black_name = opponent_name if nn_plays_white else "v7p3r Neural Network"
            
            game_options = {
                "white_name": white_name,
                "black_name": black_name
            }
            
            # Play the game
            game = play_game(white_engine, black_engine, game_options)
            
            # Save the game
            save_game(game)
            
            # Update results
            result = game.headers["Result"]
            if result == "1-0":
                results["white_wins"] += 1
                if nn_plays_white:
                    nn_results["wins"] += 1
                else:
                    nn_results["losses"] += 1
            elif result == "0-1":
                results["black_wins"] += 1
                if not nn_plays_white:
                    nn_results["wins"] += 1
                else:
                    nn_results["losses"] += 1
            else:
                results["draws"] += 1
                nn_results["draws"] += 1
            
    finally:
        # Close engines to release resources
        if hasattr(nn_engine, 'close'):
            nn_engine.close()
        if hasattr(opponent_engine, 'close'):
            opponent_engine.close()
    
    # Print final results
    print("Evaluation complete")
    print(f"Final results - v7p3r NN vs {opponent_name}:")
    print(f"v7p3r NN: {nn_results['wins']} wins, {nn_results['losses']} losses, {nn_results['draws']} draws")
    win_rate = (nn_results['wins'] + nn_results['draws'] * 0.5) / args.games * 100
    print(f"v7p3r NN score: {win_rate:.2f}%")

if __name__ == "__main__":
    main()
