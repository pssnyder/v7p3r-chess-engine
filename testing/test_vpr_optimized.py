#!/usr/bin/env python3
"""
VPR Optimized - Pure Potential with Speed Focus

Maintains the core philosophy but optimized for speed:
- Piece value = attacks + mobility (no material assumptions)
- Focus on highest/lowest potential pieces only
- Fast potential calculation using cached attacks
- Lenient pruning in chaotic positions
"""

import chess
import time
from typing import Optional, List, Dict, Set
import random

class VPROptimized:
    """
    VPR Optimized - Fast potential-based engine
    """
    
    def __init__(self):
        self.nodes_searched = 0
        self.search_start_time = 0.0
        self.board = chess.Board()
        
        # Cache for attack calculations
        self.attack_cache = {}
        
    def search(self, board: chess.Board, time_limit: float = 3.0, 
               depth: Optional[int] = None) -> chess.Move:
        """
        Fast potential-based search
        """
        self.nodes_searched = 0
        self.search_start_time = time.time()
        self.attack_cache.clear()  # Fresh cache for each search
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        target_time = time_limit * 0.9
        max_depth = depth if depth else 25  # Aim deeper
        
        best_move = legal_moves[0]
        
        # Get focus moves once (expensive calculation)
        focus_moves = self._get_fast_focus_moves(board)
        
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
                move_score = -self._fast_negamax(board, current_depth - 1, -999999, 999999, target_time)
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
    
    def _get_fast_focus_moves(self, board: chess.Board) -> List[chess.Move]:
        """
        Fast calculation of moves from high/low potential pieces
        """
        piece_potentials = []
        
        # Calculate potential for our pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                potential = self._fast_potential(board, square)
                piece_potentials.append((potential, square))
        
        if not piece_potentials:
            return list(board.legal_moves)
        
        # Sort by potential
        piece_potentials.sort(reverse=True, key=lambda x: x[0])
        
        # Focus squares (top 2 and bottom 2)
        num_pieces = len(piece_potentials)
        if num_pieces <= 4:
            focus_squares = {sq for _, sq in piece_potentials}
        else:
            top_2 = piece_potentials[:2]
            bottom_2 = piece_potentials[-2:]
            focus_squares = {sq for _, sq in top_2 + bottom_2}
        
        # Get moves from focus pieces
        focus_moves = [move for move in board.legal_moves if move.from_square in focus_squares]
        return focus_moves if focus_moves else list(board.legal_moves)
    
    def _fast_potential(self, board: chess.Board, square: chess.Square) -> int:
        """
        Fast potential calculation with caching
        """
        # Cache key
        board_key = board.fen().split()[0]  # Position only
        cache_key = (board_key, square)
        
        if cache_key in self.attack_cache:
            return self.attack_cache[cache_key]
        
        potential = 0
        
        # Fast attack count
        attacks = len(board.attacks(square))
        potential += attacks
        
        # Fast mobility count (don't check safety for speed)
        mobility = sum(1 for move in board.legal_moves if move.from_square == square)
        potential += mobility
        
        # Cache result
        self.attack_cache[cache_key] = potential
        return potential
    
    def _fast_negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, time_limit: float) -> float:
        """
        Fast negamax with minimal chaos detection
        """
        self.nodes_searched += 1
        
        # Quick time check
        if self.nodes_searched % 100 == 0:  # Check every 100 nodes
            elapsed = time.time() - self.search_start_time
            if elapsed > time_limit:
                return 0
        
        # Terminal conditions
        if depth <= 0:
            return self._fast_eval(board)
        
        if board.is_checkmate():
            return -999999
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Fast chaos check
        legal_count = len(list(board.legal_moves))
        is_chaotic = legal_count > 100  # Simple threshold
        
        max_eval = -999999
        
        # Simple move ordering: captures first, then others
        legal_moves = list(board.legal_moves)
        captures = [m for m in legal_moves if board.is_capture(m)]
        others = [m for m in legal_moves if not board.is_capture(m)]
        ordered_moves = captures + others
        
        for move in ordered_moves:
            board.push(move)
            eval_score = -self._fast_negamax(board, depth - 1, -beta, -alpha, time_limit)
            board.pop()
            
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            
            # Lenient pruning: don't prune in chaos
            if not is_chaotic and beta <= alpha:
                break
        
        return max_eval
    
    def _fast_eval(self, board: chess.Board) -> float:
        """
        Ultra-fast evaluation based on potential only
        """
        our_potential = 0
        their_potential = 0
        
        # Fast potential calculation
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Ultra-fast potential: just attacks (skip mobility for speed)
                attacks = len(board.attacks(square))
                
                if piece.color == board.turn:
                    our_potential += attacks
                else:
                    their_potential += attacks
        
        score = our_potential - their_potential
        
        # Small chaos bonus
        legal_count = len(list(board.legal_moves))
        if legal_count > 100:
            score += 20
        
        # Tiny randomness for imperfect play
        score += random.randint(-2, 2)
        
        return score

def test_optimized_vpr():
    """Test the optimized VPR for speed and depth"""
    print("⚡ VPR OPTIMIZED - Speed Test")
    print("=" * 40)
    
    engine = VPROptimized()
    
    test_positions = [
        ("Starting", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Complex", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
        ("Endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    ]
    
    total_nodes = 0
    total_time = 0
    
    for name, fen in test_positions:
        print(f"\n🎯 {name}:")
        board = chess.Board(fen)
        
        start_time = time.perf_counter()
        best_move = engine.search(board, time_limit=3.0)
        end_time = time.perf_counter()
        
        search_time = end_time - start_time
        nodes = engine.nodes_searched
        nps = int(nodes / search_time) if search_time > 0 else 0
        
        print(f"   Best: {best_move}")
        print(f"   Nodes: {nodes:,}")
        print(f"   Time: {search_time*1000:.1f}ms")
        print(f"   NPS: {nps:,}")
        
        total_nodes += nodes
        total_time += search_time
    
    avg_nps = int(total_nodes / total_time) if total_time > 0 else 0
    print(f"\n📊 OPTIMIZED VPR SUMMARY:")
    print(f"   Total nodes: {total_nodes:,}")
    print(f"   Average NPS: {avg_nps:,}")
    print(f"   Philosophy: Pure potential, optimized for speed")

if __name__ == "__main__":
    test_optimized_vpr()