<<<<<<< HEAD
import chess
from v7p3r_nn_engine.v7p3r_nn import ChessDataset, ChessAI
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import numpy as np
import pickle
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine_utilities.firebase_cloud_store import FirebaseCloudStore

torch.backends.cudnn.benchmark = True  # Enable CuDNN auto-tuner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open("../config/v7p3r_nn_config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize Firebase connection
firebase_store = FirebaseCloudStore()

class MoveEncoder:
    def __init__(self):
        # Generate all possible chess moves
        self.move_to_index = {}
        self.index_to_move = {}
        idx = 0
        
        # Generate all possible from/to squares
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                if from_sq == to_sq:
                    continue
                
                # Add non-promotion moves
                self.move_to_index[f"{chess.square_name(from_sq)}{chess.square_name(to_sq)}"] = idx
                self.index_to_move[idx] = f"{chess.square_name(from_sq)}{chess.square_name(to_sq)}"
                idx += 1
                
                # Add promotion moves
                for promo in ['q', 'r', 'b', 'n']:
                    self.move_to_index[f"{chess.square_name(from_sq)}{chess.square_name(to_sq)}{promo}"] = idx
                    self.index_to_move[idx] = f"{chess.square_name(from_sq)}{chess.square_name(to_sq)}{promo}"
                    idx += 1

    def encode_move(self, move):
        """Encodes a move into its corresponding index."""
        return self.move_to_index.get(move, -1)  # Return -1 for unknown moves

# Calculate move weights based on frequency
def calculate_move_weights(moves_list, num_classes):
    counts = np.bincount(moves_list, minlength=num_classes)
    weights = 1.0 / (np.sqrt(counts) + 1e-6)  # Inverse sqrt frequency
    return torch.tensor(weights / weights.max(), dtype=torch.float32)

# Initialize move-to-index mapping
def create_move_vocab(dataset):
    return {move: idx for idx, move in enumerate(np.unique(dataset.moves))}

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data and create vocabulary
    dataset = ChessDataset("training")
    move_to_index = create_move_vocab(dataset)
    num_classes = len(move_to_index)
    
    # Save the move vocabulary to a file
    with open("move_vocab.pkl", "wb") as f:
        pickle.dump(move_to_index, f)
    
    # Initialize move encoder with full chess move vocabulary
    move_encoder = MoveEncoder()

    # Convert moves to indices
    dataset.moves = [move_encoder.encode_move(m) for m in dataset.moves]
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    model = ChessAI(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=calculate_move_weights(np.array(dataset.moves), num_classes)
    )    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Track metrics for Firebase
    training_metrics = {
        'model_name': 'v7p3r_chess_ai',
        'train_start_time': datetime.now().isoformat(),
        'device': str(device),
        'dataset_size': len(dataset),
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'num_classes': num_classes,
        'epochs': config['training']['epochs'],
        'epoch_metrics': []
    }
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        batches = 0
        
        for positions, moves in dataloader:
            positions = positions.to(device, non_blocking=True)
            moves = moves.to(device, non_blocking=True)
            
            outputs = model(positions)
            loss = criterion(outputs, moves)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        # Calculate epoch metrics
        avg_loss = total_loss / batches
        logger.info(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Store epoch metrics
        training_metrics['epoch_metrics'].append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'timestamp': datetime.now().isoformat()
        })
    
    # Save model locally
    model_filename = "v7p3r_chess_ai_model.pth"
    torch.save(model.state_dict(), model_filename)
    logger.info(f"Model saved to {model_filename}")
    
    # Add final metrics
    training_metrics['train_end_time'] = datetime.now().isoformat()
    training_metrics['final_loss'] = training_metrics['epoch_metrics'][-1]['loss']
    
    # Upload model to Firebase
    if firebase_store.is_connected:
        # Get current version or set to 1 if no previous models
        if firebase_store.db is not None:
            try:
                models = firebase_store.db.collection('models').where('model_name', '==', 'v7p3r_chess_ai').stream()
                models_list = list(models)
                new_version = max([m.get('version', 0) for m in [doc.to_dict() for doc in models_list if doc.to_dict() is not None] if m is not None], default=0) + 1
            except Exception as e:
                logger.error(f"Error querying Firebase models: {e}")
                new_version = 1
        else:
            logger.warning("Firebase database not initialized. Defaulting model version to 1.")
            new_version = 1
        
        # Upload model
        success = firebase_store.upload_model(
            model_path=model_filename,
            model_name='v7p3r_chess_ai',
            version=new_version,
            metadata=training_metrics
        )
        
        if success:
            logger.info(f"Uploaded model to Firebase as v7p3r_chess_ai v{new_version}")
        else:
            logger.error("Failed to upload model to Firebase")
    else:
        logger.warning("Firebase not connected - model not uploaded to cloud")

if __name__ == "__main__":
    train_model()
=======
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
        v7p3r_nn.train(pgn_files=pgn_files, epochs=args.epochs)
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
        v7p3r_nn.close()

if __name__ == "__main__":
    main()
>>>>>>> 07a8bd8b88a40e25c3039c45e202a1c15bd0bce9
