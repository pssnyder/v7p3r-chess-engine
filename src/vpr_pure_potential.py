#!/usr/bin/env python3
"""
VPR Chess Engine - Pure Piece Potential Implementation

Core Philosophy from User Vision:
"We look at pieces with the most potential and pieces with the least potential.
We don't look at captures, threats, checks - we look at POTENTIAL."

Key Principles:
1. Piece value = attacks + safe mobility (NO material assumptions)
2. Focus ONLY on highest and lowest potential pieces
3. Assume imperfect opponent play (not perfect responses)
4. Preserve chaotic positions through lenient pruning
5. Break from traditional chess engine assumptions

"If a knight attacks 8 squares and can move to 8 positions freely, 
it has a score of 16. A undeveloped rook with 2 attacks has score of 2.
We prioritize the knight, not traditional material values."
"""

import chess
import time
from typing import Optional, List, Tuple
import random

class VPREngine:
    """
    V7P3R Engine - True Piece Potential Based Chess AI
    """
    
    def __init__(self):
        self.nodes_searched = 0
        self.search_start_time = 0.0
        self.board = chess.Board()
        
        # Chaos detection threshold
        self.chaos_move_threshold = 150
        
    def search(self, board: chess.Board, time_limit: float = 3.0, 
               depth: Optional[int] = None) -> chess.Move:
        """
        VPR search focusing on piece potential, not traditional metrics
        """
        self.nodes_searched = 0
        self.search_start_time = time.time()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Use 85% of time for search
        target_time = time_limit * 0.85
        max_depth = depth if depth else 20
        
        best_move = legal_moves[0]
        
        # Iterative deepening with potential-based move selection
        for current_depth in range(1, max_depth + 1):
            elapsed = time.time() - self.search_start_time
            if elapsed > target_time:
                break
            
            # Get moves from high/low potential pieces ONLY
            focus_moves = self._get_potential_focus_moves(board)
            
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
            
            # Update if we completed this depth
            if elapsed <= target_time:
                best_move = current_best
                
                # UCI output
                elapsed_ms = int(elapsed * 1000)
                nps = int(self.nodes_searched / elapsed) if elapsed > 0 else 0
                print(f"info depth {current_depth} score cp {int(current_score)} "
                      f"nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {best_move}")
        
        return best_move
    
    def _get_potential_focus_moves(self, board: chess.Board) -> List[chess.Move]:
        """
        Core VPR Innovation: Focus ONLY on highest and lowest potential pieces
        
        "We look at the pieces with the most potential and ensure they are protected
        and maybe can be used in an attack, then the pieces with the least potential
        and see if maybe we can make them more valuable in the game."
        """
        # Calculate true potential for all our pieces
        piece_potentials = []
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                potential = self._calculate_true_potential(board, square)
                piece_potentials.append((potential, square))
        
        if not piece_potentials:
            return list(board.legal_moves)
        
        # Sort by potential (highest to lowest)
        piece_potentials.sort(reverse=True, key=lambda x: x[0])
        
        # Focus on TOP pieces (devastating attacks) and BOTTOM pieces (activation)
        num_pieces = len(piece_potentials)
        focus_squares = set()
        
        if num_pieces <= 4:
            # Few pieces - consider all
            focus_squares = {sq for _, sq in piece_potentials}
        else:
            # Take top 2 (for attacks) and bottom 2 (for activation)
            # Ignore middle pieces - they're not interesting
            top_pieces = piece_potentials[:2]
            bottom_pieces = piece_potentials[-2:]
            focus_squares = {sq for _, sq in top_pieces + bottom_pieces}
        
        # Get moves from these focus pieces
        focus_moves = []
        for move in board.legal_moves:
            if move.from_square in focus_squares:
                focus_moves.append(move)
        
        # Fallback to all moves if no focus moves
        return focus_moves if focus_moves else list(board.legal_moves)
    
    def _calculate_true_potential(self, board: chess.Board, square: chess.Square) -> int:
        """
        TRUE piece potential calculation - NO material assumptions
        
        Potential = attacks + safe_mobility
        "The number of squares each piece attacks + the number of squares it can move"
        """
        potential = 0
        
        # Count attacks this piece makes
        attacks = len(board.attacks(square))
        potential += attacks
        
        # Count safe mobility (squares it can move to safely)
        safe_mobility = 0
        for move in board.legal_moves:
            if move.from_square == square:
                to_square = move.to_square
                
                # Check if move lands on safe square
                board.push(move)
                is_safe = not board.is_attacked_by(not board.turn, to_square)
                board.pop()
                
                if is_safe:
                    safe_mobility += 1
        
        potential += safe_mobility
        
        return potential
    
    def _potential_negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, time_limit: float) -> float:
        """
        VPR negamax with imperfect play assumptions and chaos preservation
        """
        self.nodes_searched += 1
        
        # Time check
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
        
        # Chaos detection for lenient pruning
        is_chaotic = self._detect_chaos(board)
        
        max_eval = -999999
        
        # Get opponent's potential-focused moves
        focus_moves = self._get_potential_focus_moves(board)
        
        for move in focus_moves:
            board.push(move)
            eval_score = -self._potential_negamax(board, depth - 1, -beta, -alpha, time_limit)
            board.pop()
            
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            
            # LENIENT PRUNING: Don't prune in chaotic positions
            if not is_chaotic and beta <= alpha:
                break
        
        return max_eval
    
    def _detect_chaos(self, board: chess.Board) -> bool:
        """
        Fast chaos detection using python-chess built-ins
        
        "A position where the legal move count is greater than 200 can be considered
        an astronomically difficult position to calculate"
        """
        legal_move_count = len(list(board.legal_moves))
        
        # Astronomical chaos threshold
        if legal_move_count > self.chaos_move_threshold:
            return True
        
        # Tactical chaos indicators
        capture_count = sum(1 for move in board.legal_moves if board.is_capture(move))
        check_available = any(board.gives_check(move) for move in board.legal_moves)
        
        # Chaotic if many captures or checks with captures
        return capture_count > 5 or (check_available and capture_count > 2)
    
    def _evaluate_pure_potential(self, board: chess.Board) -> float:
        """
        Pure potential evaluation - NO material values, NO PST tables
        
        "We evaluate based on a piece's true value, not its perceived value.
        We make no assumptions about a piece's potential."
        """
        our_total_potential = 0
        their_total_potential = 0
        
        # Calculate total potential for both sides
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                potential = self._calculate_true_potential(board, square)
                
                if piece.color == board.turn:
                    our_total_potential += potential
                else:
                    their_total_potential += potential
        
        # Potential difference is the evaluation
        score = our_total_potential - their_total_potential
        
        # Chaos bonus (we thrive in complex positions)
        if self._detect_chaos(board):
            score += 50
        
        # Imperfect play factor - opponent won't respond perfectly
        # Traditional engines assume perfect responses, we don't
        imperfection = random.randint(-5, 5)
        score += imperfection
        
        return score
    
    # UCI Protocol Implementation
    def uci(self):
        print("id name VPR v1.0")
        print("id author V7P3R Potential Engine")
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