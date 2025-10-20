#!/usr/bin/env python3
"""
VPR Chess Engine - Barebones Maximum Depth Experimental Build

An experimental rebuild of the V7P3R chess engine stripped down to
barebones heuristics and functions for maximum depth.

DESIGN PHILOSOPHY:
- Minimal computational overhead per node
- Simple, fast evaluation (material + basic positioning)
- Aggressive pruning without complex move ordering
- Focus on raw search depth over sophisticated evaluation

REMOVED FEATURES (vs v12.x):
- Transposition table (hash table overhead)
- Killer moves and history heuristic
- PV following system
- Nudge system
- Advanced pawn structure evaluation
- King safety evaluation
- Quiescence search (optional minimal version)
- Evaluation caching
- Complex move ordering (just MVV-LVA for captures)

RETAINED FEATURES:
- Alpha-beta pruning (negamax)
- Iterative deepening
- Basic time management
- Simple material evaluation
- Piece-square tables for positioning
- MVV-LVA for capture ordering

TARGET: 8-10 ply depth vs 6 ply in full engine

Author: Pat Snyder
Version: VPR v1.0 (Experimental)
"""

import time
import chess
from typing import Optional, List


class VPREngine:
    """VPR - Barebones chess engine optimized for maximum search depth"""
    
    def __init__(self):
        # Basic piece values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        # Simple piece-square tables for positional awareness
        # Values are from white's perspective, will be flipped for black
        self._init_piece_square_tables()
        
        # Search configuration
        self.default_depth = 8  # Target deeper search
        self.nodes_searched = 0
        self.search_start_time = 0
        
    def _init_piece_square_tables(self):
        """Initialize basic piece-square tables for positional evaluation"""
        
        # Pawn table - encourage central pawns and advancement
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
        
        # Knight table - prefer center, avoid edges
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
        
        # Bishop table - prefer long diagonals
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
        
        # Rook table - prefer open files and 7th rank
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
        
        # Queen table - slight center preference
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
        
        # King middlegame table - stay safe, prefer castled position
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
    
    def search(self, board: chess.Board, time_limit: float = 3.0, 
               depth: Optional[int] = None) -> chess.Move:
        """
        Main search function with iterative deepening
        
        Args:
            board: Current position
            time_limit: Time limit in seconds
            depth: Optional fixed depth (otherwise uses default)
            
        Returns:
            Best move found
        """
        self.nodes_searched = 0
        self.search_start_time = time.time()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Calculate time allocation (use 80% of available time)
        target_time = time_limit * 0.8
        
        # Iterative deepening
        max_depth = depth if depth else self.default_depth
        best_move = legal_moves[0]
        best_score = -999999
        
        for current_depth in range(1, max_depth + 1):
            iteration_start = time.time()
            elapsed = time.time() - self.search_start_time
            
            # Stop if we're running out of time
            if elapsed > target_time:
                break
            
            # Search at current depth
            score = -999999
            current_best = best_move
            time_exceeded = False
            
            for move in legal_moves:
                # Check time before starting move search
                elapsed = time.time() - self.search_start_time
                if elapsed > target_time:
                    time_exceeded = True
                    break
                
                board.push(move)
                
                # Negamax search with time tracking
                move_score = -self._negamax(board, current_depth - 1, -999999, 999999, target_time)
                
                board.pop()
                
                if move_score > score:
                    score = move_score
                    current_best = move
            
            # Update best move only if we completed this depth
            if not time_exceeded:
                best_move = current_best
                best_score = score
                
                # UCI info output
                elapsed = time.time() - self.search_start_time
                nps = int(self.nodes_searched / elapsed) if elapsed > 0 else 0
                print(f"info depth {current_depth} score cp {int(score)} "
                      f"nodes {self.nodes_searched} time {int(elapsed * 1000)} "
                      f"nps {nps} pv {best_move}")
        
        return best_move
    
    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, target_time: float) -> float:
        """
        Negamax search with alpha-beta pruning and time checking
        
        Args:
            board: Current position
            depth: Remaining depth to search
            alpha: Alpha bound
            beta: Beta bound
            target_time: Target time limit for search
            
        Returns:
            Evaluation score from current player's perspective
        """
        self.nodes_searched += 1
        
        # Check time every 1000 nodes
        if self.nodes_searched % 1000 == 0:
            elapsed = time.time() - self.search_start_time
            if elapsed > target_time:
                return 0  # Return neutral score if time exceeded
        
        # Terminal conditions
        if depth <= 0:
            return self._evaluate_position(board)
        
        if board.is_game_over():
            if board.is_checkmate():
                return -900000 + (self.default_depth - depth)  # Prefer faster mates
            return 0  # Draw
        
        # Move generation and ordering
        legal_moves = list(board.legal_moves)
        ordered_moves = self._order_moves_simple(board, legal_moves)
        
        best_score = -999999
        
        for move in ordered_moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, target_time)
            board.pop()
            
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            
            if alpha >= beta:
                break  # Beta cutoff
        
        return best_score
    
    def _order_moves_simple(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """
        Simple move ordering: captures first (MVV-LVA), then quiet moves
        
        Args:
            board: Current position
            moves: List of legal moves
            
        Returns:
            Ordered list of moves
        """
        if len(moves) <= 2:
            return moves
        
        captures = []
        quiet_moves = []
        
        for move in moves:
            if board.is_capture(move):
                # MVV-LVA scoring: Most Valuable Victim - Least Valuable Attacker
                victim = board.piece_at(move.to_square)
                victim_value = self.piece_values.get(victim.piece_type, 0) if victim else 0
                
                attacker = board.piece_at(move.from_square)
                attacker_value = self.piece_values.get(attacker.piece_type, 0) if attacker else 0
                
                # Higher score = better capture
                score = victim_value * 100 - attacker_value
                captures.append((score, move))
            else:
                quiet_moves.append(move)
        
        # Sort captures by MVV-LVA score
        captures.sort(key=lambda x: x[0], reverse=True)
        
        # Return captures first, then quiet moves
        return [move for _, move in captures] + quiet_moves
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """
        Simple evaluation: material + piece-square tables
        
        Args:
            board: Position to evaluate
            
        Returns:
            Evaluation score from current player's perspective
        """
        if board.is_checkmate():
            return -900000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Calculate material and positional value for both sides
        white_score = self._evaluate_side(board, chess.WHITE)
        black_score = self._evaluate_side(board, chess.BLACK)
        
        # Return from current player's perspective
        if board.turn == chess.WHITE:
            return white_score - black_score
        else:
            return black_score - white_score
    
    def _evaluate_side(self, board: chess.Board, color: chess.Color) -> float:
        """
        Evaluate material and position for one side
        
        Args:
            board: Current position
            color: Side to evaluate
            
        Returns:
            Total score for the side
        """
        score = 0.0
        
        # Iterate through all pieces of this color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                # Material value
                score += self.piece_values.get(piece.piece_type, 0)
                
                # Positional value from piece-square tables
                table_square = square if color == chess.WHITE else (63 - square)
                
                if piece.piece_type == chess.PAWN:
                    score += self.pawn_table[table_square]
                elif piece.piece_type == chess.KNIGHT:
                    score += self.knight_table[table_square]
                elif piece.piece_type == chess.BISHOP:
                    score += self.bishop_table[table_square]
                elif piece.piece_type == chess.ROOK:
                    score += self.rook_table[table_square]
                elif piece.piece_type == chess.QUEEN:
                    score += self.queen_table[table_square]
                elif piece.piece_type == chess.KING:
                    score += self.king_table[table_square]
        
        return score
    
    def new_game(self):
        """Reset engine state for a new game"""
        self.nodes_searched = 0
    
    def get_engine_info(self) -> dict:
        """Return engine information and statistics"""
        return {
            'name': 'VPR',
            'version': '1.0',
            'author': 'Pat Snyder',
            'description': 'Barebones maximum depth experimental engine',
            'default_depth': self.default_depth,
            'nodes_searched': self.nodes_searched
        }
