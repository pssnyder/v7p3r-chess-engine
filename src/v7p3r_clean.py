#!/usr/bin/env python3
"""
V7P3R Chess Engine v6.3 - Clean & Simple
A fast, clean chess engine inspired by C0BR4's simplicity
Author: Pat Snyder
"""

import chess
import chess.engine
import time
from typing import Optional, Tuple, Dict, Any
from v7p3r_scoring_clean import V7P3RScoringCalculationClean


class V7P3RCleanEngine:
    """Clean, simple chess engine following C0BR4's elegant design patterns"""
    
    def __init__(self):
        # Basic configuration
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300, 
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0  # King safety handled separately
        }
        
        # Search configuration
        self.default_depth = 6
        self.nodes_searched = 0
        
        # Evaluation
        self.scoring_calculator = V7P3RScoringCalculationClean(self.piece_values)
        
        # Transposition table for opening guidance only
        self.transposition_table: Dict[str, Dict[str, Any]] = {}
        self._inject_opening_knowledge()
    
    def search(self, board: chess.Board, time_limit: float = 3.0) -> chess.Move:
        """Main search entry point - like C0BR4's Think() method"""
        self.nodes_searched = 0
        start_time = time.time()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
            
        # Simple iterative deepening
        best_move = legal_moves[0]
        best_pv = [best_move]
        depth = 1
        
        while depth <= self.default_depth and (time.time() - start_time) < time_limit * 0.8:
            try:
                move, score, pv = self._search_best_move(board, depth)
                if move:
                    best_move = move
                    best_pv = pv
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    nps = int(self.nodes_searched / max(elapsed_ms / 1000, 0.001))
                    pv_str = " ".join(str(m) for m in pv[:depth])
                    print(f"info depth {depth} score cp {int(score * 100)} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {pv_str}")
                depth += 1
            except:
                break
                
        return best_move
    
    def _search_best_move(self, board: chess.Board, depth: int) -> Tuple[Optional[chess.Move], float, list]:
        """Root search - like C0BR4's SearchBestMove()"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0.0, []
            
        # Order moves for better pruning
        legal_moves = self._order_moves(board, legal_moves)
        
        best_move = legal_moves[0]
        best_score = -99999.0
        best_pv = [best_move]
        alpha = -99999.0
        beta = 99999.0
        
        for move in legal_moves:
            board.push(move)
            score, pv = self._negamax_with_pv(board, depth - 1, -beta, -alpha)
            score = -score
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
                best_pv = [move] + pv
                alpha = max(alpha, score)
        
        return best_move, best_score, best_pv
    
    def _negamax_with_pv(self, board: chess.Board, depth: int, alpha: float, beta: float) -> Tuple[float, list]:
        """Negamax with alpha-beta pruning that also returns principal variation"""
        self.nodes_searched += 1
        
        # Terminal depth
        if depth == 0:
            return self._evaluate_position(board, board.turn), []
            
        # Terminal positions
        if board.is_game_over():
            if board.is_checkmate():
                return -99999.0 + depth, []  # Prefer faster checkmates
            return 0.0, []  # Stalemate
            
        legal_moves = list(board.legal_moves)
        legal_moves = self._order_moves(board, legal_moves)
        
        best_score = -99999.0
        best_pv = []
        
        for move in legal_moves:
            board.push(move)
            score, pv = self._negamax_with_pv(board, depth - 1, -beta, -alpha)
            score = -score
            board.pop()
            
            if score > best_score:
                best_score = score
                best_pv = [move] + pv
            
            alpha = max(alpha, score)
            
            if alpha >= beta:
                break  # Alpha-beta cutoff
                
        return best_score, best_pv
    
    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        """Negamax with alpha-beta pruning - like C0BR4's AlphaBeta()"""
        self.nodes_searched += 1
        
        # Terminal depth
        if depth == 0:
            return self._evaluate_position(board, board.turn)
            
        # Terminal positions
        if board.is_game_over():
            if board.is_checkmate():
                return -99999.0 + depth  # Prefer faster checkmates
            return 0.0  # Stalemate
            
        legal_moves = list(board.legal_moves)
        legal_moves = self._order_moves(board, legal_moves)
        
        best_score = -99999.0
        
        for move in legal_moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            
            if alpha >= beta:
                break  # Alpha-beta cutoff
                
        return best_score
    
    def _evaluate_position(self, board: chess.Board, perspective: chess.Color) -> float:
        """Position evaluation from perspective of given color"""
        # Calculate scores for both sides
        white_score = self.scoring_calculator.calculate_score_optimized(board, chess.WHITE)
        black_score = self.scoring_calculator.calculate_score_optimized(board, chess.BLACK)
        
        # Return from perspective (positive = good for perspective)
        if perspective == chess.WHITE:
            return white_score - black_score
        else:
            return black_score - white_score
    
    def _order_moves(self, board: chess.Board, moves) -> list:
        """C0BR4-style move ordering"""
        if len(moves) <= 1:
            return moves
            
        scored_moves = []
        
        for move in moves:
            score = self._score_move(board, move)
            scored_moves.append((move, score))
            
        # Sort by score (highest first)
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in scored_moves]
    
    def _score_move(self, board: chess.Board, move: chess.Move) -> float:
        """C0BR4-style move scoring"""
        score = 0.0
        
        # 1. Captures (MVV-LVA)
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                victim_value = self.piece_values.get(victim.piece_type, 0)
                attacker_value = self.piece_values.get(attacker.piece_type, 0)
                score += 10000 + (victim_value - attacker_value)
        
        # 2. Promotions
        if move.promotion:
            score += 9000 + self.piece_values.get(move.promotion, 0)
        
        # 3. Checks
        board.push(move)
        if board.is_check():
            score += 500
        board.pop()
        
        # 4. Center control
        if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
            score += 10
            
        # 5. Development
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            # Development from back rank
            if (piece.color == chess.WHITE and chess.square_rank(move.from_square) == 0) or \
               (piece.color == chess.BLACK and chess.square_rank(move.from_square) == 7):
                score += 5
        
        return score
    
    def _inject_opening_knowledge(self):
        """Simple opening book injection"""
        opening_moves = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4"),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "d2d4"),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3"),
        ]
        
        for fen, move_uci in opening_moves:
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(move_uci)
                if board.is_legal(move):
                    self.transposition_table[fen] = {'best_move': move}
            except:
                continue


# Simple engine interface for UCI
class V7P3REngine:
    """UCI interface wrapper for the clean engine"""
    
    def __init__(self):
        self.engine = V7P3RCleanEngine()
        
    def search(self, board: chess.Board, player: chess.Color, ai_config: dict = {}, stop_callback=None) -> chess.Move:
        """Main search interface"""
        time_limit = ai_config.get('time_limit', 3.0)
        return self.engine.search(board, time_limit)
