# v7p3r_search.py

"""Search Controller for V7P3R Chess Engine - Simplified Version
Coordinates different search algorithms and manages search parameters.
Removes redundant hanging piece and mate-in-1 scans - let move ordering handle priorities.
"""

import chess
import time
import random
from v7p3r_move_ordering import MoveOrdering
from v7p3r_scoring import ScoringSystem
from v7p3r_transposition import get_transposition_table

class SearchController:
    def __init__(self, config):
        self.config = config
        self.move_ordering = MoveOrdering()
        self.scoring = ScoringSystem(config)
        self.tt = get_transposition_table()  # Transposition table
        
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
        self.time_limit = 30.0
        self.start_time = 0
        self.current_search_depth = self.max_depth  # Adaptive depth
    
    def _time_up(self):
        """Check if we've exceeded our time limit"""
        # If start_time is 0, time checking is disabled
        if self.start_time == 0:
            return False
        return (time.time() - self.start_time) >= self.time_limit
    
    def find_best_move(self, board, our_color, time_limit=30.0):
        """Find the best move using the configured search algorithm (simplified version)."""
        start_time = time.time()
        self.nodes_searched = 0
        self.cutoffs = 0
        self.time_limit = time_limit
        self.start_time = start_time

        # Adaptive depth based on time limit for better responsiveness
        if time_limit <= 1.5:
            self.current_search_depth = 3  # Quick moves
        elif time_limit <= 3.0:
            self.current_search_depth = 4  # Blitz moves
        elif time_limit <= 10.0:
            self.current_search_depth = 5  # Standard moves
        else:
            self.current_search_depth = self.max_depth  # Long time controls

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Direct search with move ordering handling all priorities
        move_scores = None
        if self.search_algorithm in ['negamax', 'minimax', 'simple']:
            best_move, move_scores = self._negamax_root(board, our_color, legal_moves, self.current_search_depth, debug=True)
        elif self.search_algorithm == 'random':
            best_move = random.choice(legal_moves)
            move_scores = []
        else:
            best_move, move_scores = self._negamax_root(board, our_color, legal_moves, self.current_search_depth, debug=True)

        # If time ran out and we don't have a move, just pick the first legal move
        if best_move is None and legal_moves:
            best_move = legal_moves[0]
            print(f"[DEBUG] Time ran out, selecting first legal move: {best_move}")

        self.search_time = time.time() - start_time
        if best_move is not None and isinstance(move_scores, list) and move_scores:
            print("[DEBUG] Move candidates and scores:")
            for move, score in move_scores[:5]:  # Show top 5 moves only
                print(f"  {move}: {score}")
            print(f"[DEBUG] Best move selected: {best_move} (depth: {self.current_search_depth})")
        return best_move
    
    def _negamax_root(self, board, our_color, legal_moves, search_depth=None, debug=False):
        """Root level negamax search (single streamlined version). Returns best move and all move scores."""
        if search_depth is None:
            search_depth = self.max_depth
            
        best_move = None
        best_score = float('-inf')
        move_scores = []

        # Order moves for better search
        if self.use_move_ordering:
            legal_moves = self.move_ordering.order_moves(board, legal_moves, self.max_ordered_moves)

        alpha = float('-inf')
        beta = float('inf')

        for move in legal_moves:
            # Check time limit before evaluating each move
            if self._time_up():
                print(f"[DEBUG] Time limit reached, stopping search")
                break
                
            board.push(move)
            score = -self._negamax(board, search_depth - 1, -beta, -alpha, not our_color)
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
        """Negamax search with transposition table and alpha-beta pruning."""
        self.nodes_searched += 1
        original_alpha = alpha

        # Check time limit periodically (every 50 nodes for more frequent checks)
        if self.nodes_searched % 50 == 0 and self._time_up():
            return 0  # Return neutral score if time is up

        # Transposition table lookup
        tt_result = self.tt.lookup(board, depth, alpha, beta)
        if tt_result is not None:
            score, best_move = tt_result
            if score is not None:  # We can use this score
                return score

        # Terminal conditions
        if depth == 0:
            score = self.scoring.evaluate_position(board, our_color)
            self.tt.store(board, depth, score, node_type="exact")
            return score

        # Check for repetition only at deeper levels (optimization)
        if depth >= 2 and board.is_repetition(2):
            return -5000
        
        # Early termination for large material advantages (optimization)
        if depth > 1:
            quick_eval = self._quick_material_evaluation(board, our_color)
            if abs(quick_eval) > 1500:  # Major material advantage (>= Rook + minor piece)
                return quick_eval

        if board.is_checkmate():
            score = -999999 + (self.current_search_depth - depth)
            self.tt.store(board, depth, score, node_type="exact")
            return score

        if board.is_stalemate() or board.is_insufficient_material():
            self.tt.store(board, depth, 0, node_type="exact")
            return 0

        # Get and order legal moves  
        legal_moves = list(board.legal_moves)
        if self.use_move_ordering:
            # Pass beta to move ordering for potential early cutoffs
            legal_moves = self.move_ordering.order_moves(board, legal_moves, 
                                                       max_moves=self.max_ordered_moves, 
                                                       beta=beta)

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

        # Store result in transposition table
        if best_score <= original_alpha:
            node_type = "alpha"  # Upper bound
        elif best_score >= beta:
            node_type = "beta"   # Lower bound
        else:
            node_type = "exact"  # Exact score

        self.tt.store(board, depth, best_score, node_type=node_type)
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
