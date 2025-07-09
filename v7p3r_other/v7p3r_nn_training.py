# v7p3r_nn_training.py

import os
import argparse
import glob
from v7p3r_nn import v7p3rNeuralNetwork

def main():
    parser = argparse.ArgumentParser(description="Train the v7p3r Neural Network engine")
    
    parser.add_argument("--pgn_dir", default="games", help="Directory containing PGN files")
    parser.add_argument("--analyze", action="store_true", help="Analyze games with Stockfish")
    parser.add_argument("--stockfish", help="Path to Stockfish executable")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--config", default="config/v7p3r_nn_config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Find PGN files
    pgn_files = []
    if os.path.exists(args.pgn_dir):
        pgn_files = glob.glob(os.path.join(args.pgn_dir, "*.pgn"))
    
    if not pgn_files:
        return
    
    # Initialize the neural network engine
    v7p3r_nn = v7p3rNeuralNetwork(config_path=args.config)
    
    try:
        try:
            v7p3r_nn.train(pgn_files=pgn_files, epochs=args.epochs)
        except Exception as e:
            raise
        
        print("Training completed successfully")
        
        # Analyze games with Stockfish if requested
        if args.analyze:
            stockfish_path = args.stockfish
            
            # If no Stockfish path provided, try to get it from config
            if not stockfish_path:
                try:
                    import yaml
                    with open("config/stockfish_handler.yaml", 'r') as f:
                        stockfish_config = yaml.safe_load(f)
                        stockfish_path = stockfish_config.get('stockfish_path', '')
                except Exception as e:
                    raise
            
            if not stockfish_path or not os.path.exists(stockfish_path):
                return
            
            # Analyze each PGN file
            for pgn_file in pgn_files:
                v7p3r_nn.analyze_game_with_stockfish(pgn_file, stockfish_path=stockfish_path)
            
            print("Stockfish analysis completed")
    
    finally:
        # Close the engine to release resources
        v7p3r_nn.close()

if __name__ == "__main__":
    main()
