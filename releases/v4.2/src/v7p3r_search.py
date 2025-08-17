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
        """Find the best move using the configured search algorithm (streamlined)."""
        start_time = time.time()
        self.nodes_searched = 0
        self.cutoffs = 0

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Priority 1: Free material captures
        hanging_captures = self.move_ordering.get_hanging_piece_captures(board)
        if hanging_captures:
            self.search_time = time.time() - start_time
            print(f"[DEBUG] Hanging capture selected: {hanging_captures[0]}")
            return hanging_captures[0]

        # Priority 2: Checkmate in one
        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                self.search_time = time.time() - start_time
                print(f"[DEBUG] Checkmate in one found: {move}")
                return move
            board.pop()

        # Priority 3: Search (negamax, simple, or random)
        move_scores = None
        if self.search_algorithm in ['negamax', 'minimax', 'simple']:
            best_move, move_scores = self._negamax_root(board, our_color, legal_moves, debug=True)
        elif self.search_algorithm == 'random':
            best_move = random.choice(legal_moves)
            move_scores = []
        else:
            best_move, move_scores = self._negamax_root(board, our_color, legal_moves, debug=True)

        self.search_time = time.time() - start_time
        if best_move is not None and isinstance(move_scores, list) and move_scores:
            print("[DEBUG] Move candidates and scores:")
            for move, score in move_scores:
                print(f"  {move}: {score}")
            print(f"[DEBUG] Best move selected: {best_move}")
        return best_move
    
    def _negamax_root(self, board, our_color, legal_moves, debug=False):
        """Root level negamax search (single streamlined version). Returns best move and all move scores."""
        best_move = None
        best_score = float('-inf')
        move_scores = []

        # Order moves for better search
        if self.use_move_ordering:
            legal_moves = self.move_ordering.order_moves(board, legal_moves, self.max_ordered_moves)

        alpha = float('-inf')
        beta = float('inf')

        for move in legal_moves:
            board.push(move)
            score = -self._negamax(board, self.max_depth - 1, -beta, -alpha, not our_color)
            board.pop()

            move_scores.append((move, score))

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if self.use_ab_pruning and beta <= alpha:
                self.cutoffs += 1
                break

        return best_move, move_scores
    
    def _negamax(self, board, depth, alpha, beta, our_color):
        """Negamax search (with or without alpha-beta pruning, controlled by self.use_ab_pruning)."""
        self.nodes_searched += 1

        # Terminal conditions
        if depth == 0:
            return self.scoring.evaluate_position(board, our_color)

        # Check for repetition only at deeper levels (optimization)
        if depth >= 2 and board.is_repetition(2):
            return -5000
        
        # Early termination for large material advantages (optimization)
        if depth > 1:
            quick_eval = self._quick_material_evaluation(board, our_color)
            if abs(quick_eval) > 1500:  # Major material advantage (>= Rook + minor piece)
                return quick_eval

        if board.is_checkmate():
            return -999999 + (self.max_depth - depth)

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        # Get and order legal moves
        legal_moves = list(board.legal_moves)
        if self.use_move_ordering:
            legal_moves = self.move_ordering.order_moves_with_material_priority(board, legal_moves)
            # Only limit moves if no hanging pieces are available
            if self.max_ordered_moves and len(legal_moves) > self.max_ordered_moves:
                hanging_captures = self.move_ordering.get_hanging_piece_captures(board)
                if not hanging_captures:  # Only limit if no hanging pieces
                    legal_moves = legal_moves[:self.max_ordered_moves]

        best_score = float('-inf')

        for move in legal_moves:
            try:
                board.push(move)
                score = -self._negamax(board, depth - 1, -beta, -alpha, not our_color)
                board.pop()

                best_score = max(best_score, score)
                alpha = max(alpha, score)

                if self.use_ab_pruning and beta <= alpha:
                    self.cutoffs += 1
                    break
            except Exception as e:
                # Handle any board state corruption gracefully
                try:
                    board.pop()
                except:
                    pass
                continue

        return best_score
    
    def _quick_material_evaluation(self, board, our_color):
        """Quick material-only evaluation for early termination"""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 325,
            chess.ROOK: 500,
            chess.QUEEN: 900
        }
        
        our_material = 0
        their_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                value = piece_values[piece.piece_type]
                if piece.color == our_color:
                    our_material += value
                else:
                    their_material += value
        
        return our_material - their_material

    

    
    def get_search_stats(self):
        """Get search statistics"""
        return {
            'nodes_searched': self.nodes_searched,
            'cutoffs': self.cutoffs,
            'search_time': self.search_time,
            'nps': self.nodes_searched / self.search_time if self.search_time > 0 else 0
        }
