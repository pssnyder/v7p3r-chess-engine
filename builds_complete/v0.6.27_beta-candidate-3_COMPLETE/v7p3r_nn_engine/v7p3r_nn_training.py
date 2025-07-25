# training/train_v7p3r_nn.py
# Sample script to train the v7p3r Neural Network engine

import os
import argparse
import glob
import logging
from v7p3r_nn_engine.v7p3r_nn import v7p3rNeuralNetwork

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("train_v7p3r_nn")

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
        logger.warning(f"No PGN files found in {args.pgn_dir}")
        return
    
    logger.info(f"Found {len(pgn_files)} PGN files for training")
    
    # Initialize the neural network engine
    v7p3r_nn = v7p3rNeuralNetwork(config_path=args.config)
    
    try:
        # Train on the PGN files
        logger.info("Starting training...")
        logger.info(f"Training config: epochs={args.epochs}, PGN files={len(pgn_files)}")
        for idx, pgn_file in enumerate(pgn_files, 1):
            logger.info(f"[{idx}/{len(pgn_files)}] Preparing {pgn_file}")
        try:
            v7p3r_nn.train(pgn_files=pgn_files, epochs=args.epochs)
        except Exception as e:
            logger.error(f"Error during training: {e}")
        logger.info("Training completed")
        
        # Analyze games with Stockfish if requested
        if args.analyze:
            logger.info("Starting Stockfish analysis...")
            stockfish_path = args.stockfish
            
            # If no Stockfish path provided, try to get it from config
            if not stockfish_path:
                try:
                    import yaml
                    with open("config/stockfish_handler.yaml", 'r') as f:
                        stockfish_config = yaml.safe_load(f)
                        stockfish_path = stockfish_config.get('stockfish_path', '')
                except Exception as e:
                    logger.error(f"Failed to load stockfish config: {e}")
            
            if not stockfish_path or not os.path.exists(stockfish_path):
                logger.error("Stockfish path not provided or executable not found")
                return
            
            # Analyze each PGN file
            for pgn_file in pgn_files:
                logger.info(f"Analyzing {pgn_file}...")
                v7p3r_nn.analyze_game_with_stockfish(pgn_file, stockfish_path=stockfish_path)
            
            logger.info("Stockfish analysis completed")
    
    finally:
        # Close the engine to release resources
        logger.info("Closing neural network engine...")
        v7p3r_nn.close()

if __name__ == "__main__":
    main()
