#!/usr/bin/env python3
"""
VPR Chess Engine - Pure Piece Potential Implementation

Core Philosophy:
"We look at pieces with the most potential and pieces with the least potential.
We don't look at captures, threats, checks - we look at POTENTIAL."

Revolutionary Principles:
1. Piece value = attacks + mobility (NO material assumptions)
2. Focus ONLY on highest and lowest potential pieces  
3. Assume imperfect opponent play (not perfect responses)
4. Preserve chaotic positions through lenient pruning
5. Break from traditional chess engine assumptions

"If a knight attacks 8 squares and can move to 8 positions freely,
it has a score of 16. An undeveloped rook with 2 attacks has score of 2.
We prioritize the knight, not traditional material values."

Author: V7P3R Project
Version: VPR Pure Potential v1.0
"""

import chess
import time
from typing import Optional, List, Dict
import random

class VPREngine:
    """
    V7P3R Pure Potential Chess Engine
    """
    
    # Class attributes for type hinting
    chaos_move_threshold: int
    
    def __init__(self):
        self.nodes_searched = 0
        self.search_start_time = 0.0
        self.board = chess.Board()
        
        # Attack calculation cache for speed
        self.attack_cache = {}
        
        # UCI configurable options
        self.chaos_move_threshold = 100  # Default chaos threshold for move selection
        
    def search(self, board: chess.Board, time_limit: float = 3.0, 
               depth: Optional[int] = None) -> chess.Move:
        """
        VPR search: Focus on piece potential, not traditional metrics
        """
        self.nodes_searched = 0
        self.search_start_time = time.time()
        self.attack_cache.clear()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        target_time = time_limit * 0.9
        max_depth = depth if depth else 25
        
        best_move = legal_moves[0]
        
        # Get potential-focused moves (expensive, do once)
        focus_moves = self._get_potential_focus_moves(board)
        
        # Iterative deepening
        for current_depth in range(1, max_depth + 1):
            elapsed = time.time() - self.search_start_time
            if elapsed > target_time:
                break
            
            current_best = best_move
            current_score = -999999
            
            for move in focus_moves:
                elapsed = time.time() - self.search_start_time
                if elapsed > target_time:
                    break
                
                board.push(move)
                move_score = -self._potential_negamax(board, current_depth - 1, -999999, 999999, target_time)
                board.pop()
                
                if move_score > current_score:
                    current_score = move_score
                    current_best = move
            
            if elapsed <= target_time:
                best_move = current_best
                
                elapsed_ms = int(elapsed * 1000)
                nps = int(self.nodes_searched / elapsed) if elapsed > 0 else 0
                print(f"info depth {current_depth} score cp {int(current_score)} "
                      f"nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {best_move}")
        
        return best_move
    
    def _get_potential_focus_moves(self, board: chess.Board) -> List[chess.Move]:
        """
        Core VPR Innovation: Focus ONLY on highest and lowest potential pieces
        """
        piece_potentials = []
        
        # Calculate true potential for all our pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                potential = self._calculate_piece_potential(board, square)
                piece_potentials.append((potential, square))
        
        if not piece_potentials:
            return list(board.legal_moves)
        
        # Sort by potential (highest to lowest)
        piece_potentials.sort(reverse=True, key=lambda x: x[0])
        
        # Focus on TOP pieces (attacks) and BOTTOM pieces (activation)
        num_pieces = len(piece_potentials)
        focus_squares = set()
        
        if num_pieces <= 4:
            focus_squares = {sq for _, sq in piece_potentials}
        else:
            # Top 2 and bottom 2 pieces only - ignore the middle
            top_pieces = piece_potentials[:2]
            bottom_pieces = piece_potentials[-2:]
            focus_squares = {sq for _, sq in top_pieces + bottom_pieces}
        
        # Get moves from focus pieces
        focus_moves = [move for move in board.legal_moves if move.from_square in focus_squares]
        return focus_moves if focus_moves else list(board.legal_moves)
    
    def _calculate_piece_potential(self, board: chess.Board, square: chess.Square) -> int:
        """
        TRUE piece potential calculation with caching
        
        Potential = attacks + mobility (NO material assumptions)
        """
        # Cache for speed
        board_key = board.fen().split()[0]
        cache_key = (board_key, square)
        
        if cache_key in self.attack_cache:
            return self.attack_cache[cache_key]
        
        potential = 0
        
        # Count attacks this piece makes
        attacks = len(board.attacks(square))
        potential += attacks
        
        # Count mobility (legal moves from this piece)
        mobility = sum(1 for move in board.legal_moves if move.from_square == square)
        potential += mobility
        
        # Cache and return
        self.attack_cache[cache_key] = potential
        return potential
    
    def _potential_negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, time_limit: float) -> float:
        """
        VPR negamax with chaos preservation and imperfect play assumptions
        """
        self.nodes_searched += 1
        
        # Periodic time check for speed
        if self.nodes_searched % 100 == 0:
            elapsed = time.time() - self.search_start_time
            if elapsed > time_limit:
                return 0
        
        # Terminal conditions
        if depth <= 0:
            return self._evaluate_pure_potential(board)
        
        if board.is_checkmate():
            return -999999
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Fast chaos detection
        legal_count = len(list(board.legal_moves))
        is_chaotic = legal_count > 100  # Chaos threshold
        
        max_eval = -999999
        
        # Simple move ordering for speed
        legal_moves = list(board.legal_moves)
        captures = [m for m in legal_moves if board.is_capture(m)]
        others = [m for m in legal_moves if not board.is_capture(m)]
        ordered_moves = captures + others
        
        for move in ordered_moves:
            board.push(move)
            eval_score = -self._potential_negamax(board, depth - 1, -beta, -alpha, time_limit)
            board.pop()
            
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            
            # LENIENT PRUNING: Don't prune chaotic positions
            if not is_chaotic and beta <= alpha:
                break
        
        return max_eval
    
    def _evaluate_pure_potential(self, board: chess.Board) -> float:
        """
        Pure potential evaluation - NO material values, NO PST tables
        """
        our_potential = 0
        their_potential = 0
        
        # Fast potential calculation for both sides
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Fast potential: just attacks (mobility is expensive)
                attacks = len(board.attacks(square))
                
                if piece.color == board.turn:
                    our_potential += attacks
                else:
                    their_potential += attacks
        
        # Potential difference is the score
        score = our_potential - their_potential
        
        # Chaos bonus (we thrive in complexity)
        legal_count = len(list(board.legal_moves))
        if legal_count > 100:
            score += 20
        
        # Imperfection factor (opponent won't play perfectly)
        imperfection = random.randint(-3, 3)
        score += imperfection
        
        return score
    
    # UCI Protocol Implementation
    def uci(self):
        print("id name VPR Pure Potential v1.0")
        print("id author V7P3R")
        print("uciok")
    
    def isready(self):
        print("readyok")
    
    def position(self, fen: Optional[str] = None, moves: Optional[List[str]] = None):
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
        
        if moves:
            for move_str in moves:
                try:
                    move = chess.Move.from_uci(move_str)
                    self.board.push(move)
                except:
                    pass
    
    def go(self, **kwargs):
        # Parse time controls
        time_limit = 3.0  # Default
        
        if 'movetime' in kwargs:
            time_limit = kwargs['movetime'] / 1000.0
        elif 'wtime' in kwargs and 'btime' in kwargs:
            if self.board.turn == chess.WHITE:
                time_limit = min(kwargs['wtime'] / 1000.0 / 20, 10.0)
            else:
                time_limit = min(kwargs['btime'] / 1000.0 / 20, 10.0)
        
        depth = kwargs.get('depth', None)
        
        best_move = self.search(self.board, time_limit, depth)
        print(f"bestmove {best_move}")
    
    def new_game(self):
        """Reset engine state for a new game"""
        self.nodes_searched = 0
        self.attack_cache.clear()
    
    def get_engine_info(self) -> dict:
        """Return engine information"""
        return {
            'name': 'VPR Pure Potential',
            'version': '3.0', 
            'author': 'V7P3R Project',
            'description': 'Revolutionary piece potential based chess AI',
            'nodes_searched': self.nodes_searched
        }

def main():
    """
    UCI main loop for VPR engine
    """
    engine = VPREngine()
    
    while True:
        try:
            command = input().strip().split()
            if not command:
                continue
            
            cmd = command[0]
            
            if cmd == "uci":
                engine.uci()
            elif cmd == "isready":
                engine.isready()
            elif cmd == "position":
                # Parse position command
                if len(command) > 1:
                    if command[1] == "startpos":
                        engine.position()
                        if len(command) > 3 and command[2] == "moves":
                            engine.position(moves=command[3:])
                    elif command[1] == "fen":
                        fen_parts = []
                        i = 2
                        while i < len(command) and command[i] != "moves":
                            fen_parts.append(command[i])
                            i += 1
                        fen = " ".join(fen_parts)
                        moves = command[i+1:] if i < len(command) and command[i] == "moves" else None
                        engine.position(fen, moves)
            elif cmd == "go":
                # Parse go command
                go_params = {}
                i = 1
                while i < len(command):
                    if command[i] in ["movetime", "wtime", "btime", "depth"]:
                        if i + 1 < len(command):
                            try:
                                go_params[command[i]] = int(command[i + 1])
                            except:
                                pass
                            i += 2
                        else:
                            i += 1
                    else:
                        i += 1
                engine.go(**go_params)
            elif cmd == "quit":
                break
                
        except EOFError:
            break
        except Exception as e:
            print(f"info string Error: {e}")

if __name__ == "__main__":
    main()