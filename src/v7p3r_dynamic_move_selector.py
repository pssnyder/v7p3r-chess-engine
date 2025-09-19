#!/usr/bin/env python3
"""
V7P3R Dynamic Depth-Based Move Pruning
Progressive move selection that becomes more aggressive with search depth
"""

import chess
from typing import List, Tuple

class V7P3RDynamicMoveSelector:
    """
    V11 ENHANCEMENT: Dynamic move selection based on search depth
    Becomes increasingly selective as depth increases for speed
    """
    
    def __init__(self):
        self.depth_thresholds = {
            1: 50,   # Depth 1-2: Consider most moves
            2: 50,
            3: 20,   # Depth 3-4: More aggressive pruning
            4: 15,
            5: 8,    # Depth 5-6: Very aggressive
            6: 6,
            7: 4,    # Depth 7+: Only critical moves
            8: 3
        }
    
    def get_move_limit_for_depth(self, depth: int) -> int:
        """Get maximum number of moves to consider at given depth"""
        if depth <= 2:
            return self.depth_thresholds[1]
        elif depth <= 4:
            return self.depth_thresholds[min(depth, 4)]
        elif depth <= 6:
            return self.depth_thresholds[min(depth, 6)]
        else:
            return self.depth_thresholds[8]
    
    def should_prune_move(self, board: chess.Board, move: chess.Move, depth: int, move_index: int) -> bool:
        """
        Determine if a move should be pruned based on depth and position
        Returns True if move should be skipped
        """
        # Never prune in shallow depths
        if depth <= 2:
            return False
        
        # Get move limit for this depth
        move_limit = self.get_move_limit_for_depth(depth)
        
        # Always keep moves within the limit based on their ordering
        if move_index < move_limit:
            return False
        
        # For deeper searches, be very aggressive about pruning
        if depth >= 5:
            # Always keep captures and checks
            if board.is_capture(move) or board.gives_check(move):
                return False
            
            # Always keep moves that escape check
            if board.is_check():
                return False
            
            # Prune quiet moves beyond the limit
            return True
        
        # For medium depths (3-4), prune less aggressively
        if depth >= 3:
            # Keep tactical moves
            if board.is_capture(move) or board.gives_check(move):
                return False
            
            # Keep castling
            if board.is_castling(move):
                return False
            
            # Prune quiet moves beyond limit
            return move_index >= move_limit
        
        return False
    
    def filter_moves_by_depth(self, board: chess.Board, ordered_moves: List[chess.Move], depth: int) -> List[chess.Move]:
        """
        Filter moves based on search depth - more aggressive pruning at deeper levels
        """
        if depth <= 2:
            return ordered_moves  # No pruning for shallow depths
        
        filtered_moves = []
        move_limit = self.get_move_limit_for_depth(depth)
        
        # Categorize moves for depth-based selection
        critical_moves = []
        tactical_moves = []
        positional_moves = []
        quiet_moves = []
        
        for move in ordered_moves:
            if board.is_check() or board.is_capture(move) or board.gives_check(move):
                critical_moves.append(move)
            elif board.is_castling(move) or self._is_promotion(move):
                tactical_moves.append(move)
            elif self._is_central_move(board, move) or self._is_development_move(board, move):
                positional_moves.append(move)
            else:
                quiet_moves.append(move)
        
        # Select moves based on depth
        if depth >= 6:
            # Depth 6+: Only critical moves
            filtered_moves = critical_moves[:move_limit]
        elif depth >= 5:
            # Depth 5: Critical + very few tactical
            filtered_moves = critical_moves[:int(move_limit * 0.8)]
            remaining = move_limit - len(filtered_moves)
            filtered_moves.extend(tactical_moves[:remaining])
        elif depth >= 4:
            # Depth 4: Critical + some tactical
            filtered_moves = critical_moves[:int(move_limit * 0.7)]
            remaining = move_limit - len(filtered_moves)
            filtered_moves.extend(tactical_moves[:remaining])
        else:
            # Depth 3: Critical + tactical + some positional
            filtered_moves = critical_moves[:int(move_limit * 0.6)]
            remaining = move_limit - len(filtered_moves)
            filtered_moves.extend(tactical_moves[:int(remaining * 0.7)])
            remaining = move_limit - len(filtered_moves)
            filtered_moves.extend(positional_moves[:remaining])
        
        # Ensure we have at least one move
        if not filtered_moves and ordered_moves:
            filtered_moves = [ordered_moves[0]]
        
        return filtered_moves
    
    def _is_promotion(self, move: chess.Move) -> bool:
        """Check if move is a promotion"""
        return move.promotion is not None
    
    def _is_central_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move involves central squares"""
        central_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        return move.to_square in central_squares or move.from_square in central_squares
    
    def _is_development_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move develops a piece"""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        
        # Check if piece is moving from back rank for first time
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            back_rank = 0 if piece.color == chess.BLACK else 7
            return chess.square_rank(move.from_square) == back_rank
        
        return False