"""
CUDA/GPU acceleration utilities for GA training.
Provides GPU-accelerated evaluation and batch processing capabilities.
"""

import torch
import numpy as np
import chess
from typing import List, Tuple, Optional
import logging

class CUDAAccelerator:
    """Handles CUDA acceleration for batch position evaluation and fitness calculation."""
    
    def __init__(self, use_cuda: Optional[bool] = None, batch_size: int = 64):
        """
        Initialize CUDA accelerator.
        
        Args:
            use_cuda: Force CUDA usage. If None, auto-detect availability.
            batch_size: Number of positions to process in parallel
        """
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        
        if self.use_cuda and torch.cuda.is_available():
            print(f"[CUDA] GPU acceleration enabled: {torch.cuda.get_device_name()}")
            print(f"[CUDA] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("[CUDA] Using CPU (GPU not available or disabled)")
    
    def batch_fitness_calculation(self, v7p3r_evals: List[float], 
                                stockfish_evals: List[float]) -> float:
        """
        GPU-accelerated batch fitness calculation using MSE.
        
        Args:
            v7p3r_evals: List of v7p3r evaluation scores
            stockfish_evals: List of corresponding Stockfish scores
            
        Returns:
            Negative MSE fitness score (higher = better)
        """
        if not v7p3r_evals or not stockfish_evals:
            return float('-inf')
        
        # Convert to tensors and move to GPU
        v7p3r_tensor = torch.tensor(v7p3r_evals, dtype=torch.float32, device=self.device)
        stockfish_tensor = torch.tensor(stockfish_evals, dtype=torch.float32, device=self.device)
        
        # Calculate MSE using GPU
        mse = torch.nn.functional.mse_loss(v7p3r_tensor, stockfish_tensor)
        
        # Return negative MSE (higher fitness = better)
        return -mse.item()
    
    def batch_position_features(self, boards: List[chess.Board]) -> torch.Tensor:
        """
        Extract position features in batches for neural network evaluation.
        
        Args:
            boards: List of chess boards to extract features from
            
        Returns:
            Tensor of shape (batch_size, feature_dim) with position features
        """
        features = []
        
        for board in boards:
            # Extract basic position features
            board_features = self._extract_board_features(board)
            features.append(board_features)
        
        # Convert to tensor and move to GPU
        features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        return features_tensor
    
    def _extract_board_features(self, board: chess.Board) -> List[float]:
        """
        Extract numerical features from a chess position.
        
        Returns:
            List of numerical features representing the position
        """
        features = []
        
        # Material features (12 values: 6 piece types x 2 colors)
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                          chess.ROOK, chess.QUEEN, chess.KING]:
            features.append(len(board.pieces(piece_type, chess.WHITE)))
            features.append(len(board.pieces(piece_type, chess.BLACK)))
        
        # Positional features
        features.append(float(board.turn))  # Side to move
        features.append(float(board.has_kingside_castling_rights(chess.WHITE)))
        features.append(float(board.has_queenside_castling_rights(chess.WHITE)))
        features.append(float(board.has_kingside_castling_rights(chess.BLACK)))
        features.append(float(board.has_queenside_castling_rights(chess.BLACK)))
        features.append(float(board.is_check()))
        features.append(float(len(list(board.legal_moves))))  # Mobility
        
        # Center control (simplified)
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        white_center = 0
        black_center = 0
        for sq in center_squares:
            piece = board.piece_at(sq)
            if piece is not None:
                if piece.color == chess.WHITE:
                    white_center += 1
                elif piece.color == chess.BLACK:
                    black_center += 1
        features.extend([white_center, black_center])
        
        return features
    
    def parallel_stockfish_evaluation(self, boards: List[chess.Board], 
                                    stockfish_handler) -> List[float]:
        """
        Optimized Stockfish evaluation with persistent engine and batching.
        """
        evaluations = []
        
        # Process boards in batches to avoid memory issues
        for i in range(0, len(boards), self.batch_size):
            batch = boards[i:i + self.batch_size]
            batch_evals = []
            
            for board in batch:
                try:
                    # Use the existing stockfish handler but with optimizations
                    score = stockfish_handler.evaluate_position_from_perspective(
                        board, board.turn)
                    batch_evals.append(score)
                except Exception as e:
                    batch_evals.append(0.0)
            
            evaluations.extend(batch_evals)
        
        return evaluations
    
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage information."""
        if self.use_cuda and torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'cached': torch.cuda.memory_reserved() / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated() / 1e9
            }
        return {'cpu_mode': True}
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if self.use_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()


class NeuralNetworkEvaluator:
    """
    Uses a pre-trained neural network for position evaluation instead of/alongside Stockfish.
    This can be much faster than running Stockfish for every position.
    """
    
    def __init__(self, model_path: Optional[str] = None, cuda_accelerator: Optional[CUDAAccelerator] = None):
        """
        Initialize NN evaluator.
        
        Args:
            model_path: Path to pre-trained neural network model
            cuda_accelerator: CUDA accelerator instance
        """
        self.cuda_accelerator = cuda_accelerator or CUDAAccelerator()
        self.model = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a pre-trained evaluation model."""
        try:
            # Try to load from the NN engine if available
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v7p3r_nn_engine'))
            from v7p3r_nn import V7P3RNeuralNetwork  # type: ignore
            
            self.model = V7P3RNeuralNetwork()
            self.model.load_model(model_path)
            self.model.model.to(self.cuda_accelerator.device)
            self.model.model.eval()
            print(f"[NN] Loaded neural network model from {model_path}")
        except (ImportError, Exception) as e:
            print(f"[NN] Could not load model: {e}")
            self.model = None
    
    def batch_evaluate_positions(self, boards: List[chess.Board]) -> List[float]:
        """
        Evaluate positions using neural network in batches.
        
        Args:
            boards: List of chess boards to evaluate
            
        Returns:
            List of evaluation scores
        """
        if not self.model:
            raise ValueError("No neural network model loaded")
        
        # Extract features for all positions
        features = self.cuda_accelerator.batch_position_features(boards)
        
        # Evaluate in batches
        evaluations = []
        batch_size = self.cuda_accelerator.batch_size
        
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i + batch_size]
                
                # Get model predictions
                outputs = self.model.model(batch_features)
                batch_evals = outputs.cpu().numpy().flatten()
                evaluations.extend(batch_evals)
        
        return evaluations
