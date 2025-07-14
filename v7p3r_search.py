# v7p3r_search.py

"""Search Controller for V7P3R Chess Engine
Coordinates different search algorithms and manages search parameters.
"""

import chess
import time
import random
from v7p3r_move_ordering import MoveOrdering
from v7p3r_scoring import ScoringSystem

class SearchController:
    def __init__(self, config):
        self.config = config
        self.move_ordering = MoveOrdering()
        self.scoring = ScoringSystem(config)
        
        # Search configuration
        self.search_algorithm = config.get_setting('engine_config', 'search_algorithm', 'negamax')
        self.max_depth = config.get_setting('engine_config', 'depth', 6)
        self.use_ab_pruning = config.is_enabled('engine_config', 'use_ab_pruning')
        self.use_move_ordering = config.is_enabled('engine_config', 'use_move_ordering')
        self.max_ordered_moves = config.get_setting('engine_config', 'max_ordered_moves', 10)
        
        # Search statistics
        self.nodes_searched = 0
        self.cutoffs = 0
        self.search_time = 0
    
    def find_best_move(self, board, our_color):
        """Find the best move using configured search algorithm"""
        start_time = time.time()
        self.nodes_searched = 0
        self.cutoffs = 0
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Choose search algorithm
        if self.search_algorithm in ['negamax', 'minimax']:
            best_move, score = self._negamax_root(board, our_color, legal_moves)
        elif self.search_algorithm == 'simple':
            best_move, score = self._simple_search(board, our_color, legal_moves)
        elif self.search_algorithm == 'random':
            best_move = random.choice(legal_moves)
            score = 0
        else:
            # Fallback to negamax search
            best_move, score = self._negamax_root(board, our_color, legal_moves)
        
        self.search_time = time.time() - start_time
        
        return best_move
    
    def _negamax_root(self, board, our_color, legal_moves):
        """Root level negamax search"""
        best_move = None
        best_score = float('-inf')
        
        # Order moves for better search
        if self.use_move_ordering:
            legal_moves = self.move_ordering.order_moves(board, legal_moves, self.max_ordered_moves)
        
        alpha = float('-inf')
        beta = float('inf')
        
        for move in legal_moves:
            board.push(move)
            
            if self.use_ab_pruning:
                score = -self._negamax(board, self.max_depth - 1, -beta, -alpha, not our_color)
            else:
                score = -self._negamax_no_pruning(board, self.max_depth - 1, not our_color)
            
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            if self.use_ab_pruning:
                alpha = max(alpha, score)
                if beta <= alpha:
                    self.cutoffs += 1
                    break
        
        return best_move, best_score
    
    def _negamax(self, board, depth, alpha, beta, our_color):
        """Negamax search with alpha-beta pruning"""
        self.nodes_searched += 1
        
        # Check for repetition (penalize heavily)
        if board.is_repetition(2):  # Check for threefold repetition
            return -5000  # Penalize repetition
        
        # Terminal conditions
        if depth == 0:
            return self.scoring.evaluate_position(board, our_color)

        if board.is_checkmate():
            return -999999 + (self.max_depth - depth)  # Prefer quicker mates
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Generate and order moves
        legal_moves = list(board.legal_moves)
        if self.use_move_ordering:
            legal_moves = self.move_ordering.order_moves(board, legal_moves, self.max_ordered_moves)
        
        best_score = float('-inf')
        
        for move in legal_moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, not our_color)
            board.pop()
            
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            
            if beta <= alpha:
                self.cutoffs += 1
                break
        
        return best_score
    
    def _negamax_no_pruning(self, board, depth, our_color):
        """Negamax search without alpha-beta pruning"""
        self.nodes_searched += 1
        
        # Check for repetition (penalize heavily)
        if board.is_repetition(2):  # Check for threefold repetition
            return -5000  # Penalize repetition
        
        # Terminal conditions
        if depth == 0:
            return self.scoring.evaluate_position(board, our_color)
        
        if board.is_checkmate():
            return -999999 + (self.max_depth - depth)
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Generate moves
        legal_moves = list(board.legal_moves)
        if self.use_move_ordering:
            legal_moves = self.move_ordering.order_moves(board, legal_moves, self.max_ordered_moves)
        
        best_score = float('-inf')
        
        for move in legal_moves:
            board.push(move)
            score = -self._negamax_no_pruning(board, depth - 1, not our_color)
            board.pop()
            
            best_score = max(best_score, score)
        
        return best_score
    
    def _simple_search(self, board, our_color, legal_moves):
        """Simple search - evaluate all moves at depth 1"""
        best_move = None
        best_score = float('-inf')
        
        for move in legal_moves:
            score, details, critical = self.scoring.evaluate_move(board, move, our_color, depth=1)
            
            if score > best_score:
                best_score = score
                best_move = move
                
                # Short circuit for critical moves
                if critical:
                    break
        
        return best_move, best_score
    
    def get_search_stats(self):
        """Get search statistics"""
        return {
            'nodes_searched': self.nodes_searched,
            'cutoffs': self.cutoffs,
            'search_time': self.search_time,
            'nps': self.nodes_searched / self.search_time if self.search_time > 0 else 0
        }
