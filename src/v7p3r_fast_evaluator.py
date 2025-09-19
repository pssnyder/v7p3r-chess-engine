#!/usr/bin/env python3
"""
V7P3R v11 Performance Optimization: Fast Evaluation System
Lightweight evaluator for non-critical nodes to fix exponential search explosion
Author: Pat Snyder
"""

import chess
from typing import Dict, Optional


class V7P3RFastEvaluator:
    """
    Lightweight evaluator for performance-critical search nodes
    Provides quick material + basic positional evaluation
    """
    
    def __init__(self):
        # Standard piece values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 10000
        }
        
        # Simple positional tables for piece-square values
        self.pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        self.knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
        
        self.bishop_table = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]
        
        self.rook_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ]
        
        self.queen_table = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]
        
        self.king_table = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]
        
        # Simple evaluation cache for repeated positions
        self.evaluation_cache: Dict[str, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def evaluate_position_fast(self, board: chess.Board) -> float:
        """
        Fast evaluation for performance-critical nodes
        Returns evaluation from current player's perspective
        """
        # Check cache first
        position_hash = str(board.board_fen())  # Simple position hash
        if position_hash in self.evaluation_cache:
            self.cache_hits += 1
            return self.evaluation_cache[position_hash]
        
        self.cache_misses += 1
        
        # Quick material + piece-square evaluation
        white_score = 0
        black_score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Material value
                material_value = self.piece_values[piece.piece_type]
                
                # Piece-square table value
                positional_value = self._get_piece_square_value(piece, square)
                
                total_value = material_value + positional_value
                
                if piece.color:  # White
                    white_score += total_value
                else:  # Black
                    black_score += total_value
        
        # Simple mobility bonus
        mobility_bonus = 0
        if not board.is_game_over():
            try:
                num_moves = len(list(board.legal_moves))
                mobility_bonus = min(num_moves * 5, 100)  # Cap at 100 centipawns
            except:
                mobility_bonus = 0
        
        # Apply mobility to current player
        if board.turn:  # White to move
            white_score += mobility_bonus
        else:  # Black to move
            black_score += mobility_bonus
        
        # Calculate final score from current player's perspective
        if board.turn:  # White to move
            final_score = (white_score - black_score) / 100.0
        else:  # Black to move
            final_score = (black_score - white_score) / 100.0
        
        # Cache the result
        self.evaluation_cache[position_hash] = final_score
        
        return final_score
    
    def _get_piece_square_value(self, piece: chess.Piece, square: int) -> int:
        """Get piece-square table value for piece at square"""
        # Convert square to table index (flip for black pieces)
        if piece.color:  # White
            table_index = square
        else:  # Black - flip the board
            table_index = square ^ 56  # Flip rank
        
        # Select appropriate table
        if piece.piece_type == chess.PAWN:
            return self.pawn_table[table_index]
        elif piece.piece_type == chess.KNIGHT:
            return self.knight_table[table_index]
        elif piece.piece_type == chess.BISHOP:
            return self.bishop_table[table_index]
        elif piece.piece_type == chess.ROOK:
            return self.rook_table[table_index]
        elif piece.piece_type == chess.QUEEN:
            return self.queen_table[table_index]
        elif piece.piece_type == chess.KING:
            return self.king_table[table_index]
        
        return 0
    
    def clear_cache(self):
        """Clear evaluation cache"""
        self.evaluation_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / max(total_requests, 1)) * 100
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.evaluation_cache)
        }


# Example usage and testing
if __name__ == "__main__":
    fast_eval = V7P3RFastEvaluator()
    
    # Test positions
    test_positions = [
        chess.Board(),  # Starting position
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),  # After 1.e4
        chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),  # Complex
    ]
    
    print("V7P3R Fast Evaluator Test")
    print("=" * 30)
    
    import time
    
    for i, board in enumerate(test_positions):
        start_time = time.time()
        
        # Run evaluation multiple times to test performance
        for _ in range(1000):
            score = fast_eval.evaluate_position_fast(board)
        
        elapsed = time.time() - start_time
        
        print(f"Position {i+1}: {score:+6.2f} (1000 evals in {elapsed:.3f}s, {1000/elapsed:.0f} evals/sec)")
    
    print(f"\nCache stats: {fast_eval.get_cache_stats()}")