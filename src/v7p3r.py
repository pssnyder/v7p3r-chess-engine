#!/usr/bin/env python3
"""
V7P3R Chess Engine v7.0 - Clean Slate Edition
A fast, clean chess engine inspired by C0BR4's simplicity
Author: Pat Snyder
"""

import chess
import chess.engine
import sys
import time
from typing import Optional, Tuple, Dict, Any
from v7p3r_scoring_calculation import V7P3RScoringCalculationClean


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
        self.root_color = chess.WHITE  # Initialize root color
        
        # Evaluation
        self.scoring_calculator = V7P3RScoringCalculationClean(self.piece_values)
        
        # Move ordering improvements
        self.killer_moves = {}  # killer_moves[ply] = [move1, move2]
        self.history_scores = {}  # history_scores[move_key] = score
        
        # Transposition table for opening guidance only
        self.transposition_table: Dict[str, Dict[str, Any]] = {}
        self._inject_opening_knowledge()
    
    def search(self, board: chess.Board, time_limit: float = 3.0) -> chess.Move:
        """Main search entry point - like C0BR4's Think() method"""
        self.nodes_searched = 0
        self.root_color = board.turn  # Store the root perspective
        start_time = time.time()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
            
        # Adaptive time management - be more aggressive
        target_time = min(time_limit * 0.6, 8.0)  # Cap at 8 seconds, use 60% of allocation
        max_time = min(time_limit * 0.8, 12.0)   # Hard cap at 12 seconds
        
        # Simple iterative deepening with aggressive time management
        best_move = legal_moves[0]
        best_pv = [best_move]
        depth = 1
        last_complete_depth = 0
        
        while depth <= self.default_depth:
            iteration_start = time.time()
            try:
                move, score, pv = self._search_best_move(board, depth)
                iteration_time = time.time() - iteration_start
                
                if move:
                    best_move = move
                    best_pv = pv
                    last_complete_depth = depth
                    elapsed_ms = int((time.time() - start_time) * 1000)
                    nps = int(self.nodes_searched / max(elapsed_ms / 1000, 0.001))
                    pv_str = " ".join(str(m) for m in pv[:depth])
                    
                    # CRITICAL FIX: UCI scores should always be from side-to-move perspective
                    # The score from _search_best_move is already from the side-to-move perspective
                    # (positive = good for side to move, negative = bad for side to move)
                    uci_score = score
                    
                    # Format score for UCI output
                    score_str = self._format_uci_score(uci_score, depth)
                    
                    print(f"info depth {depth} score {score_str} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {pv_str}")
                    sys.stdout.flush()  # Ensure UCI info appears immediately
                
                # Aggressive time management: stop if we're approaching time limits
                elapsed = time.time() - start_time
                
                # If this iteration took a long time, probably stop
                if iteration_time > target_time * 0.4:
                    break
                    
                # If we're approaching target time, stop
                if elapsed > target_time:
                    break
                    
                # If next iteration would likely exceed max time, stop
                if elapsed + (iteration_time * 2.5) > max_time:
                    break
                    
                depth += 1
            except:
                break
                
        return best_move
    
    def new_game(self):
        """Reset search tables for a new game"""
        self.killer_moves.clear()
        self.history_scores.clear()
        self.nodes_searched = 0
    
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
    
    def _negamax_with_pv(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0, allow_null: bool = True) -> Tuple[float, list]:
        """Negamax with alpha-beta pruning that also returns principal variation"""
        self.nodes_searched += 1
        
        # Terminal depth
        if depth == 0:
            # Always evaluate from the root player's perspective
            # Negamax handles the sign flipping through negation
            eval_score = self._evaluate_position(board, self.root_color)
            # If it's not the root player's turn, we need to negate for negamax
            if board.turn != self.root_color:
                eval_score = -eval_score
            return eval_score, []
            
        # Terminal positions
        if board.is_game_over():
            if board.is_checkmate():
                # Mate scores: negative for losing, prefer faster checkmates
                # Use a cleaner mate scoring system
                return -29000.0 + ply, []  # Negative for being mated, adjusted by ply
            return 0.0, []  # Stalemate
        
        # Null move pruning disabled for stability
        # TODO: Re-implement null move pruning more carefully
        
        legal_moves = list(board.legal_moves)
        legal_moves = self._order_moves(board, legal_moves, ply)
        
        best_score = -99999.0
        best_pv = []
        
        for i, move in enumerate(legal_moves):
            # Simple late move reduction - skip some moves in shallow search
            if depth <= 1 and i > 8 and not board.is_capture(move):
                continue  # Skip non-capture moves late in the list at low depth
            
            board.push(move)
            score, pv = self._negamax_with_pv(board, depth - 1, -beta, -alpha, ply + 1)
            score = -score
            board.pop()
            
            if score > best_score:
                best_score = score
                best_pv = [move] + pv
            
            alpha = max(alpha, score)
            
            if alpha >= beta:
                # Alpha-beta cutoff - this move caused a cutoff, so it's a "killer"
                self._store_killer_move(move, ply)
                self._update_history_score(move, depth)
                break
                
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
                # Mate scores: negative for losing, prefer faster checkmates  
                # Use consistent mate scoring - depth here represents plies from root
                return -29000.0 + depth  # Negative for being mated, adjusted by depth remaining
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
    
    def _order_moves(self, board: chess.Board, moves, ply: int = 0) -> list:
        """Enhanced move ordering with killer moves and history"""
        if len(moves) <= 1:
            return moves
            
        scored_moves = []
        
        for move in moves:
            score = self._score_move(board, move, ply)
            scored_moves.append((move, score))
            
        # Sort by score (highest first)
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in scored_moves]
    
    def _score_move(self, board: chess.Board, move: chess.Move, ply: int = 0) -> float:
        """Enhanced move scoring with killer moves and history heuristic"""
        score = 0.0
        
        # 1. Captures (MVV-LVA) - highest priority
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                victim_value = self.piece_values.get(victim.piece_type, 0)
                attacker_value = self.piece_values.get(attacker.piece_type, 0)
                # MVV-LVA: prioritize capturing valuable pieces with less valuable pieces
                score += 100000 + (victim_value * 10) - attacker_value
        
        # 2. Promotions - very high priority
        if move.promotion:
            score += 90000 + self.piece_values.get(move.promotion, 0)
        
        # 3. Killer moves - good moves that caused cutoffs at this ply
        if ply in self.killer_moves:
            if move in self.killer_moves[ply]:
                score += 80000 - self.killer_moves[ply].index(move) * 1000
        
        # 4. Checks - tactical priority
        board.push(move)
        if board.is_check():
            score += 5000
        board.pop()
        
        # 5. History heuristic - moves that were good in similar positions
        move_key = f"{move.from_square}_{move.to_square}"
        if move_key in self.history_scores:
            score += min(self.history_scores[move_key], 4000)  # Cap history bonus
            
        # 6. Center control - positional bonus
        if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
            score += 100
            
        # 7. Development bonus - piece activity
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            # Development from back rank
            if (piece.color == chess.WHITE and chess.square_rank(move.from_square) == 0) or \
               (piece.color == chess.BLACK and chess.square_rank(move.from_square) == 7):
                score += 50
        
        # 8. King safety penalty for moving king early
        if piece and piece.piece_type == chess.KING:
            if len(list(board.move_stack)) < 10:  # Early game
                score -= 200
        
        return score
    
    def _store_killer_move(self, move: chess.Move, ply: int):
        """Store a killer move that caused a cutoff"""
        if ply not in self.killer_moves:
            self.killer_moves[ply] = []
        
        # Remove if already present
        if move in self.killer_moves[ply]:
            self.killer_moves[ply].remove(move)
        
        # Add to front
        self.killer_moves[ply].insert(0, move)
        
        # Keep only top 2 killer moves per ply
        if len(self.killer_moves[ply]) > 2:
            self.killer_moves[ply] = self.killer_moves[ply][:2]
    
    def _update_history_score(self, move: chess.Move, depth: int):
        """Update history heuristic score for a move"""
        move_key = f"{move.from_square}_{move.to_square}"
        bonus = depth * depth  # Higher bonus for deeper cutoffs
        
        if move_key in self.history_scores:
            self.history_scores[move_key] += bonus
        else:
            self.history_scores[move_key] = bonus
        
        # Prevent overflow
        if self.history_scores[move_key] > 10000:
            self.history_scores[move_key] = 10000
    
    def _format_uci_score(self, score: float, search_depth: int) -> str:
        """Format score for UCI output - properly handle mate scores"""
        # Check if this is a mate score (lowered threshold to catch 29000 scores)
        if abs(score) >= 28500:
            # This is a mate score
            if score > 0:
                # We have mate - calculate moves to mate more accurately
                # Score is close to 29000, so calculate depth more precisely
                if score >= 29000:
                    # This is a forced mate
                    depth_to_mate = 29000 - score + search_depth
                    mate_in = max(1, int((depth_to_mate + 1) / 2))
                else:
                    # Lower scoring mate - likely deeper
                    mate_in = max(1, min(10, int((29000 - score + search_depth) / 2)))
                return f"mate {mate_in}"
            else:
                # We're getting mated
                if score <= -29000:
                    depth_to_mate = 29000 + score + search_depth  # score is negative
                    mate_in = max(1, int((depth_to_mate + 1) / 2))
                else:
                    mate_in = max(1, min(10, int((29000 + score + search_depth) / 2)))
                return f"mate -{mate_in}"
        else:
            # Regular positional score in centipawns
            return f"cp {int(score)}"
    
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
