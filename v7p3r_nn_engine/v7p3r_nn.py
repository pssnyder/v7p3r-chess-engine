# v7p3r_nn_engine/v7p3r_nn.py
# v7p3r Chess Engine Neural Network Module
import os
import sys
import numpy as np
import chess
import chess.pgn
import sqlite3
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import yaml
import logging
import random
from io import StringIO
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union

# Add the project root to the Python path to allow imports from anywhere in the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v7p3r_engine.stockfish_handler import StockfishHandler

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("v7p3r_nn")

class ChessPositionDataset(Dataset):
    """Dataset for chess positions and their evaluations"""
    
    def __init__(self, positions, evaluations, transform=None):
        """
        Args:
            positions: List of FEN positions
            evaluations: List of evaluation scores
            transform: Optional transform to apply to the data
        """
        self.positions = positions
        self.evaluations = evaluations
        self.transform = transform
        
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        position = self.positions[idx]
        evaluation = self.evaluations[idx]
        
        # Convert FEN to feature representation
        features = self._fen_to_features(position)
        
        if self.transform:
            features = self.transform(features)
            
        return features, evaluation
    
    def _fen_to_features(self, fen):
        """Convert FEN string to feature vector"""
        # Initialize a chess board with the FEN
        board = chess.Board(fen)
        
        # Create a 12 x 8 x 8 tensor (12 piece types, 8x8 board)
        # 6 piece types for each color (pawn, knight, bishop, rook, queen, king)
        features = np.zeros((12, 8, 8), dtype=np.float32)
        
        # Map each piece type to an index in the feature array
        piece_to_index = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        # Fill the feature array
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Get the row and column (0-7) from the square (0-63)
                row, col = divmod(square, 8)
                
                # Get the index for the piece type
                piece_idx = piece_to_index[piece.piece_type]
                
                # If the piece is black, add 6 to the index
                if not piece.color:
                    piece_idx += 6
                
                # Set the feature to 1 at the piece's position
                features[piece_idx, row, col] = 1.0
        
        # Add a feature for the side to move
        # Create a new array for turn information (1 if white to move, 0 if black)
        turn_feature = np.ones((1, 8, 8), dtype=np.float32) if board.turn else np.zeros((1, 8, 8), dtype=np.float32)
        
        # Concatenate with the piece features
        features = np.concatenate((features, turn_feature), axis=0)
        
        return torch.tensor(features, dtype=torch.float32)

class ChessNN(nn.Module):
    """Neural network for chess position evaluation"""
    
    def __init__(self, hidden_layers=[256, 128, 64], dropout_rate=0.3):
        """
        Args:
            hidden_layers: List of neurons in each hidden layer
            dropout_rate: Dropout probability for regularization
        """
        super(ChessNN, self).__init__()
        
        # Input is a 13 x 8 x 8 tensor (12 piece types + turn indicator)
        self.conv1 = nn.Conv2d(13, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of flattened features after convolutions and pooling
        flat_features = 64 * 1 * 1  # After 3 pooling operations: 8x8 -> 4x4 -> 2x2 -> 1x1
        
        # Fully connected layers
        fc_layers = []
        input_size = flat_features
        
        for hidden_size in hidden_layers:
            fc_layers.append(nn.Linear(input_size, hidden_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
            
        # Output layer (single value for evaluation)
        fc_layers.append(nn.Linear(input_size, 1))
        fc_layers.append(nn.Tanh())  # Tanh activation for [-1, 1] output range
        
        self.fc = nn.Sequential(*fc_layers)
        
    def forward(self, x):
        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc(x)
        
        # Scale the output to a reasonable evaluation range
        # Multiply by 10 to get evaluations in the -10 to 10 range
        return x * 10.0

class MoveLibrary:
    """A class to manage a library of chess moves and positions in an SQLite database."""
    def __init__(self, db_path="v7p3r_move_library.db"):
        """Initialize the MoveLibrary and create the database and tables if they don't exist."""
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create the necessary tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Positions table for storing FEN positions and their evaluations
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fen TEXT UNIQUE,
            evaluation REAL,
            source TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Best moves table for storing the best move for each position
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS best_moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            position_id INTEGER,
            move TEXT,
            evaluation REAL,
            source TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (position_id) REFERENCES positions (id)
        )
        ''')
        
        self.conn.commit()
        
    def add_position(self, fen, evaluation, source="nn"):
        """Add or update a position in the library"""
        cursor = self.conn.cursor()
        
        # Insert or replace the position
        cursor.execute('''
        INSERT OR REPLACE INTO positions (fen, evaluation, source, timestamp)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (fen, evaluation, source))
        
        position_id = cursor.lastrowid
        # Removed self.conn.commit() for batch commit
        
        return position_id
        
    def add_best_move(self, fen, move, evaluation, source="nn", confidence=1.0):
        """Add a best move for a position"""
        cursor = self.conn.cursor()
        
        # Get or create the position
        cursor.execute("SELECT id FROM positions WHERE fen=?", (fen,))
        row = cursor.fetchone()
        
        if row:
            position_id = row[0]
        else:
            position_id = self.add_position(fen, evaluation, source)
        
        # Insert the best move
        cursor.execute('''
        INSERT INTO best_moves (position_id, move, evaluation, source, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (position_id, move, evaluation, source, confidence))
        
        # Removed self.conn.commit() for batch commit
        
    def get_best_move(self, fen):
        """Get the best move for a position"""
        cursor = self.conn.cursor()
        
        # Get the position ID
        cursor.execute("SELECT id FROM positions WHERE fen=?", (fen,))
        row = cursor.fetchone()
        
        if not row:
            return None
            
        position_id = row[0]
        
        # Get the best move with the highest evaluation
        cursor.execute('''
        SELECT move, evaluation, source, confidence
        FROM best_moves
        WHERE position_id=?
        ORDER BY evaluation DESC, confidence DESC, timestamp DESC
        LIMIT 1
        ''', (position_id,))
        
        row = cursor.fetchone()
        
        if not row:
            return None
            
        return {
            "move": row[0],
            "evaluation": row[1],
            "source": row[2],
            "confidence": row[3]
        }
        
    def get_position_evaluation(self, fen):
        """Get the evaluation for a position"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT evaluation FROM positions WHERE fen=?", (fen,))
        row = cursor.fetchone()
        
        return row[0] if row else None
        
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

class v7p3rNeuralNetwork:
    """Neural network-based chess engine for v7p3r"""
    
    def __init__(self, config_path="config/v7p3r_nn_config.yaml"):
        """Initialize the neural network engine with configuration"""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize move library with config
        move_library_config = self.config.get("move_library", {})
        db_path = move_library_config.get("db_path", "v7p3r_nn_engine/v7p3r_nn_move_vocab/move_library.db")
        self.move_library = MoveLibrary(db_path=db_path)
        
        self.model = self._create_model()
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        logger.warning(f"Config file {config_path} not found. Using default configuration.")
        return {}
        
    def _create_model(self):
        """Create and initialize the neural network model"""
        model_config = self.config.get("training", {}).get("model", {})
        hidden_layers = model_config.get("hidden_layers", [256, 128, 64])
        dropout_rate = model_config.get("dropout_rate", 0.3)
        
        model = ChessNN(hidden_layers=hidden_layers, dropout_rate=dropout_rate)
        model.to(self.device)
        
        # Load pre-trained model if available
        model_path = self._get_latest_model_path()
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading pre-trained model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            
        return model
        
    def _get_latest_model_path(self):        
        """Get the path to the latest saved model"""
        storage_config = self.config.get("training", {}).get("storage", {})
        if not storage_config.get("enabled", False):
            return None
            
        model_dir = storage_config.get("model_path", "v7p3r_nn_engine/models")
        os.makedirs(model_dir, exist_ok=True)
        
        # Find the most recent model file
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
        if not model_files:
            return None
            
        latest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
        return os.path.join(model_dir, latest_model)
        
    def train(self, pgn_files=None, db_path=None, epochs=None):
        """Train the neural network on historical PGN games"""
        logger.info("Starting neural network training...")
        
        # Get training parameters from config
        train_config = self.config.get("training", {})
        batch_size = train_config.get("batch_size", 64)
        learning_rate = train_config.get("learning_rate", 0.001)
        weight_decay = train_config.get("weight_decay", 0.0001)
        epochs = epochs or train_config.get("epochs", 50)
        
        # Collect training data from PGNs and/or database
        positions, evaluations = self._collect_training_data(pgn_files, db_path)
        
        if not positions:
            logger.warning("No training data found. Skipping training.")
            return
            
        logger.info(f"Collected {len(positions)} positions for training")
        
        # Create dataset and dataloader
        dataset = ChessPositionDataset(positions, evaluations)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            batch_count = 0
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            for i, data in enumerate(dataloader):
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).float().view(-1, 1)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batch_count += 1
                logger.debug(f"Epoch {epoch+1} Batch {i+1} Loss: {loss.item():.4f}")
                if i % 10 == 9:
                    logger.info(f"Epoch {epoch+1} [{i+1}/{len(dataloader)}] avg loss: {running_loss / (i+1):.4f}")
            avg_loss = running_loss / batch_count if batch_count else 0.0
            logger.info(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
            # Save checkpoint if configured
            self._save_checkpoint(epoch)
        logger.info("Finished training")
        
        # Save the final model
        self._save_model()
        
    def _collect_training_data(self, pgn_files=None, db_path=None):
        """Collect training data from PGN files and/or database"""
        positions = []
        evaluations = []
        
        # Collect from PGN files
        if pgn_files:
            for pgn_file in pgn_files:
                try:
                    with open(pgn_file, 'r') as f:
                        file_positions = 0
                        while True:
                            game = chess.pgn.read_game(f)
                            if game is None:
                                break
                            positions_from_game, evals_from_game = self._extract_positions_from_game(game)
                            positions.extend(positions_from_game)
                            evaluations.extend(evals_from_game)
                            file_positions += len(positions_from_game)
                        if file_positions == 0:
                            logger.warning(f"No positions found in PGN file {pgn_file}")
                        else:
                            logger.info(f"Extracted {file_positions} positions from {pgn_file}")
                        # Commit all DB changes for this file at once (batch commit)
                        if hasattr(self, 'move_library') and hasattr(self.move_library, 'conn'):
                            self.move_library.conn.commit()
                except Exception as e:
                    logger.error(f"Error reading PGN file {pgn_file}: {e}")
        
        # Collect from database
        if db_path:
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Fetch all game records from the database
                cursor.execute("SELECT game_pgn FROM game_results")
                
                for row in cursor.fetchall():
                    pgn_text = row[0]
                    if not pgn_text:
                        continue
                        
                    try:
                        pgn_io = StringIO(pgn_text)
                        game = chess.pgn.read_game(pgn_io)
                        if game is None:
                            continue
                            
                        positions_from_game, evals_from_game = self._extract_positions_from_game(game)
                        positions.extend(positions_from_game)
                        evaluations.extend(evals_from_game)
                    except Exception as e:
                        logger.error(f"Error parsing PGN from database: {e}")
                
                conn.close()
            except Exception as e:
                logger.error(f"Error accessing database {db_path}: {e}")
        
        return positions, evaluations
    
    def _extract_positions_from_game(self, game):
        """Extract positions and evaluations from a chess game"""
        positions = []
        evaluations = []
        
        board = game.board()
        
        for node in game.mainline():
            # Skip positions from the first 5 moves (opening book territory)
            if board.fullmove_number <= 5:
                board.push(node.move)
                continue
                
            # Get the FEN of the position before the move
            fen = board.fen()
            
            # Get the evaluation from the comment, if available
            eval_score = self._extract_evaluation_from_comment(node.comment)
            
            if eval_score is not None:
                positions.append(fen)
                evaluations.append(eval_score)
                
                # Add the position to the move library
                self.move_library.add_position(fen, eval_score, source="pgn")
                
                # Add the best move to the move library
                self.move_library.add_best_move(
                    fen=fen,
                    move=node.move.uci(),
                    evaluation=eval_score,
                    source="pgn",
                    confidence=0.9
                )
            
            # Make the move on the board
            board.push(node.move)
        
        return positions, evaluations
    
    def _extract_evaluation_from_comment(self, comment):
        """Extract the evaluation score from a move comment"""
        if not comment:
            return None
            
        # Look for patterns like "Eval: 0.72" or similar
        import re
        eval_match = re.search(r"Eval:\s*([-+]?\d+\.\d+)", comment)
        
        if eval_match:
            return float(eval_match.group(1))
            
        return None
    
    def _save_checkpoint(self, epoch):
        """Save a checkpoint of the model during training"""
        storage_config = self.config.get("training", {}).get("storage", {})        
        if not storage_config.get("enabled", False) or not storage_config.get("store_checkpoints", False):
            return
            
        checkpoint_freq = storage_config.get("checkpoint_frequency", 5)
        if (epoch + 1) % checkpoint_freq != 0:
            return
            
        model_dir = storage_config.get("model_path", "v7p3r_nn_engine/models")
        os.makedirs(model_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(model_dir, f"v7p3r_nn_checkpoint_epoch_{epoch+1}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _save_model(self):
        """Save the trained model"""
        storage_config = self.config.get("training", {}).get("storage", {})        
        if not storage_config.get("enabled", False) or not storage_config.get("save_model", False):
            return
            
        model_dir = storage_config.get("model_path", "v7p3r_nn_engine/models")
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"v7p3r_nn_model_{timestamp}.pt")
        
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
    
    def evaluate_position(self, fen):
        """Evaluate a chess position using the neural network"""
        # First check the move library
        eval_from_library = self.move_library.get_position_evaluation(fen)
        if eval_from_library is not None:
            return eval_from_library
            
        # Otherwise, use the neural network
        self.model.eval()
        with torch.no_grad():
            # Convert FEN to features
            features = ChessPositionDataset([fen], [0])._fen_to_features(fen)
            features = features.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Get the evaluation from the model
            evaluation = self.model(features).item()
            
            # Store the evaluation in the move library for future reference
            self.move_library.add_position(fen, evaluation, source="nn")
            
            return evaluation
    
    def get_best_move(self, fen):
        """Get the best move for a position using the neural network"""
        # First check the move library
        best_move_info = self.move_library.get_best_move(fen)
        if best_move_info is not None:
            return best_move_info["move"]
            
        # If not in the library, use the engine to evaluate all legal moves
        board = chess.Board(fen)
        
        best_move = None
        best_eval = float('-inf') if board.turn == chess.WHITE else float('inf')
        
        for move in board.legal_moves:
            # Make the move and evaluate the resulting position
            board.push(move)
            eval_score = self.evaluate_position(board.fen())
            board.pop()
            
            # Update best move if this move is better
            if (board.turn == chess.WHITE and eval_score > best_eval) or \
               (board.turn == chess.BLACK and eval_score < best_eval):
                best_eval = eval_score
                best_move = move
        
        if best_move:
            # Store the best move in the library
            self.move_library.add_best_move(
                fen=fen,
                move=best_move.uci(),                evaluation=best_eval,
                source="nn_search",
                confidence=0.8
            )
            
            return best_move.uci()
        
        return None
        
    def analyze_game_with_stockfish(self, pgn_path, stockfish_path=None):
        """Analyze a game with Stockfish and update the move library"""
        # Get stockfish path from config if not provided
        if not stockfish_path:
            try:
                with open("config/stockfish_handler.yaml", 'r') as f:
                    stockfish_config = yaml.safe_load(f)
                    stockfish_path = stockfish_config.get('stockfish_path', '')
            except Exception as e:
                logger.error(f"Failed to load stockfish config: {e}")
                return False
        
        if not stockfish_path or not os.path.exists(stockfish_path):
            logger.error(f"Stockfish executable not found at: {stockfish_path}")
            return False
        
        # Initialize Stockfish handler
        try:
            stockfish = StockfishHandler(
                stockfish_config= {
                    'stockfish_path': stockfish_path, 
                    'depth': 15,
                }
            )
            
            # Load the game from PGN
            with open(pgn_path, 'r') as f:
                game = chess.pgn.read_game(f)
                
            if not game:
                logger.error(f"Could not load game from {pgn_path}")
                return False
                
            # Get analysis settings from config
            stockfish_config = self.config.get('training', {}).get('stockfish', {})
            depth = stockfish_config.get('depth', 15)
            positions_per_game = stockfish_config.get('positions_per_game', 20)
            
            # Initialize board and position counter
            board = game.board()
            positions_analyzed = 0
            moves_list = list(game.mainline_moves())
            
            # Skip very short games
            if len(moves_list) < 10:
                logger.warning(f"Game {pgn_path} is too short to analyze ({len(moves_list)} moves)")
                stockfish.quit()
                return False
            
            # Select positions evenly distributed throughout the game
            position_indices = []
            if len(moves_list) <= positions_per_game:
                position_indices = range(len(moves_list))
            else:
                # Skip the first 5 moves (opening book territory)
                step = max(1, (len(moves_list) - 5) // positions_per_game)
                position_indices = range(5, len(moves_list), step)
                position_indices = list(position_indices)[:positions_per_game]
            
            logger.info(f"Analyzing {len(position_indices)} positions from game {pgn_path}")
            
            # Analyze each selected position
            for move_index, move in enumerate(moves_list):
                # Make the move on the board
                if move_index not in position_indices:
                    board.push(move)
                    continue
                
                # Analyze the position before the move
                fen = board.fen()
                  # Set up Stockfish for analysis
                analysis_board = chess.Board(fen)
                stockfish.set_position(analysis_board)
                
                # Run search for depth moves
                engine_config = {"depth": depth}
                stockfish.search(analysis_board, analysis_board.turn, engine_config)
                
                # Get analysis results
                search_info = stockfish.get_last_search_info()
                
                # Extract evaluation
                score = search_info.get('score', 0.0)
                best_move = None
                
                # Get best move from PV
                pv = search_info.get('pv', '')
                if pv:
                    best_move = pv.split()[0]
                
                # Normalize score to [-1, 1] range
                evaluation = max(-1.0, min(1.0, score / 10.0))
                
                # Add to move library
                if best_move:
                    self.move_library.add_best_move(
                        fen=fen,
                        move=best_move,
                        evaluation=evaluation,
                        source="stockfish",
                        confidence=1.0  # Highest confidence for Stockfish
                    )
                
                # Store the position evaluation
                self.move_library.add_position(
                    fen=fen,
                    evaluation=evaluation,
                    source="stockfish"
                )
                
                # Make the move
                board.push(move)
                positions_analyzed += 1
                
                if positions_analyzed % 5 == 0:
                    logger.info(f"Analyzed {positions_analyzed} positions")
            
            logger.info(f"Completed Stockfish analysis of {positions_analyzed} positions from {pgn_path}")
            stockfish.quit()
            return True            
        except Exception as e:
            logger.error(f"Error in Stockfish analysis: {e}")
            if 'stockfish' in locals() and stockfish:
                try:
                    stockfish.quit()
                except:
                    pass
            return False
            
    def close(self):
        """Close the move library and release resources"""
        if hasattr(self, 'move_library'):
            self.move_library.close()
            
    def cleanup(self):
        """Cleanup resources - alias for close() to match engine interface"""
        self.close()
            
    def reset(self, board: Optional[chess.Board] = None):
        """
        Reset the engine state for a new board position.
        Compatible with the v7p3rEngine interface.
        
        Args:
            board: chess.Board object with the new position
        """
        # Nothing to reset for the NN engine
        pass
        
    def evaluate_position_from_perspective(self, board, perspective):
        """
        Evaluate a chess position from a specific perspective.
        Compatible with the v7p3rEngine interface.
        
        Args:
            board: chess.Board object
            perspective: Player color (chess.WHITE or chess.BLACK)
            
        Returns:
            Evaluation score (positive is good for the player with the given perspective)
        """
        # Get raw evaluation (from white's perspective)
        eval_score = self.evaluate_position(board.fen())
        
        # Adjust perspective if needed
        if perspective == chess.BLACK:
            eval_score = -eval_score
            
        return eval_score
    
    def search(self, board, player_color, engine_config=None):
        """
        Search for the best move in the current position.
        This method is compatible with the v7p3rEngine interface
        expected by chess_game.py.
        
        Args:
            board: chess.Board object
            player_color: Player's color (chess.WHITE or chess.BLACK)
            engine_config: Optional engine configuration
            
        Returns:
            chess.Move object
        """
        best_move_uci = self.get_best_move(board.fen())
        
        # If no best move found in library, evaluate all legal moves
        if not best_move_uci:
            best_move = None
            best_eval = float('-inf') if board.turn == chess.WHITE else float('inf')
            
            for move in board.legal_moves:
                # Make the move and evaluate the resulting position
                board.push(move)
                eval_score = self.evaluate_position(board.fen())
                board.pop()
                
                # Update best move if this move is better
                if (board.turn == chess.WHITE and eval_score > best_eval) or \
                   (board.turn == chess.BLACK and eval_score < best_eval):
                    best_eval = eval_score
                    best_move = move
            
            if best_move:
                return best_move
                
            # If we get here, something went wrong - return a random legal move
            moves = list(board.legal_moves)
            if moves:
                return random.choice(moves)
            return None
        
        # Convert UCI string to chess.Move object
        try:
            move = chess.Move.from_uci(best_move_uci)
            if move in board.legal_moves:
                return move
        except ValueError:
            pass
        
        # Fallback to random move if UCI parsing fails or move is illegal
        moves = list(board.legal_moves)
        if moves:
            return random.choice(moves)
        return None

# Main function for direct execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="v7p3r Neural Network Chess Engine")
    parser.add_argument("--train", action="store_true", help="Train the neural network")
    parser.add_argument("--pgn", nargs="+", help="PGN files to train on")
    parser.add_argument("--db", help="Database path for game records")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--analyze", help="Analyze a PGN file with Stockfish")
    parser.add_argument("--stockfish", help="Path to Stockfish executable")
    
    args = parser.parse_args()
    
    v7p3r_nn = v7p3rNeuralNetwork()
    
    try:
        if args.train:
            v7p3r_nn.train(pgn_files=args.pgn, db_path=args.db, epochs=args.epochs)
            
        if args.analyze:
            v7p3r_nn.analyze_game_with_stockfish(args.analyze, stockfish_path=args.stockfish or "engine_utilities/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe")
    finally:
        v7p3r_nn.close()
