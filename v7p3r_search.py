# v7p3r_search.py

"""v7p3r Search Module
This module handles the main search functionality for the v7p3r chess engine.
It contains the search handler and core search algorithms (minimax, negamax, simple, random).
"""

import sys
import os
import chess
import random
from typing import Optional, Dict, List, Tuple
from v7p3r_config import v7p3rConfig
from v7p3r_utilities import get_timestamp
from v7p3r_deepsearch import v7p3rDeepSearch
from v7p3r_quiescence import v7p3rQuiescence

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class v7p3rSearch:
    def __init__(self, scoring_calculator, move_organizer, time_manager, opening_book, rules_manager, engine_config=None):
        # Engine Configuration
        self.config_manager = v7p3rConfig()
        if engine_config is not None:
            self.engine_config = engine_config
        else:
            self.engine_config = self.config_manager.get_engine_config()

        # Required Search Modules
        self.scoring_calculator = scoring_calculator
        self.move_organizer = move_organizer
        self.time_manager = time_manager
        self.opening_book = opening_book
        self.rules_manager = rules_manager

        # Initialize Advanced Search Modules
        self.deep_search = v7p3rDeepSearch(time_manager, self.engine_config)
        self.quiescence = v7p3rQuiescence(scoring_calculator, move_organizer, rules_manager, self.engine_config)

        # Search Configuration
        self.search_algorithm = self.engine_config.get('search_algorithm', 'simple')
        self.depth = self.engine_config.get('depth', 3)
        self.max_depth = self.engine_config.get('max_depth', 5)
        
        # Search Control Flags
        self.ab_pruning_enabled = self.engine_config.get('use_ab_pruning', True)
        self.draw_prevention_enabled = self.engine_config.get('use_draw_prevention', False)
        self.quiescence_enabled = self.engine_config.get('use_quiescence', True)
        self.move_ordering_enabled = self.engine_config.get('use_move_ordering', True)
        self.max_ordered_moves = self.engine_config.get('max_ordered_moves', 5)

        # Initialize search state
        self.root_board = chess.Board()
        self.nodes_searched = 0
        self.stats = {'depth': 0, 'nodes': 0, 'score': 0.0, 'time': 0.0, 'pv': []}
        self.pv_move_stack = []

    def _update_pv_line(self, move: chess.Move, score: float, depth: int):
        """Update the principal variation line with a new move"""
        self.pv_move_stack.append({
            'move': move,
            'score': score,
            'depth': depth,
            'nodes': self.nodes_searched
        })

    def find_checkmate_in_n(self, board: chess.Board, n: int) -> chess.Move:
        """Find a checkmate sequence within n moves"""
        if n <= 0:
            return chess.Move.null()

        # Check if we're in checkmate already
        if board.is_checkmate():
            return chess.Move.null()

        legal_moves = list(board.legal_moves)
        
        # First check if any move directly leads to checkmate
        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()

        # Then look for forced checkmate sequences
        for move in legal_moves:
            board.push(move)
            opponent_moves = list(board.legal_moves)
            
            # If opponent has no legal moves and it's not checkmate, it's stalemate
            if not opponent_moves:
                if not board.is_checkmate():  # Avoid stalemate
                    board.pop()
                    continue
            
            all_replies_lead_to_mate = True
            for opponent_move in opponent_moves:
                board.push(opponent_move)
                # Recursively search for checkmate with reduced depth
                if self.find_checkmate_in_n(board, n - 2) == chess.Move.null():
                    all_replies_lead_to_mate = False
                board.pop()
                if not all_replies_lead_to_mate:
                    break
            
            board.pop()
            if all_replies_lead_to_mate:
                return move
                
        return chess.Move.null()

    def _should_stop_search(self, avg_score: float, depth: int, start_time: float) -> bool:
        """Enhanced search stopping conditions"""
        current_time = self.time_manager.get_current_time()
        time_elapsed = current_time - start_time
        allocated_time = self.time_manager.get_allocated_move_time()

        # Time-based stopping
        if time_elapsed >= allocated_time * 0.95:  # Use 95% of allocated time
            return True

        # Node-based stopping
        if self.nodes_searched >= 1_000_000:  # Hard node limit
            return True

        # Score-based stopping
        if depth >= 3:  # Only consider scores after reasonable depth
            if avg_score > 500 and self.nodes_searched > 50_000:  # Clear advantage
                return True
            if avg_score > 900 and self.nodes_searched > 10_000:  # Near checkmate
                return True

        # PV-based stopping
        if len(self.pv_move_stack) > 1:
            last_two_scores = [entry['score'] for entry in self.pv_move_stack[-2:]]
            if abs(last_two_scores[1] - last_two_scores[0]) < 0.01:  # Score stabilized
                if self.nodes_searched > 25_000:  # Minimum node count
                    return True

        return False

    def iterative_deepening_search(self, board: chess.Board, color: chess.Color) -> chess.Move:
        """Perform iterative deepening search with early exit conditions"""
        start_time = self.time_manager.get_current_time()
        best_move = chess.Move.null()
        best_score = float('-inf')
        previous_best_moves = []
        
        for current_depth in range(1, self.max_depth + 1):
            self.depth = current_depth
            moves = list(board.legal_moves)
            
            # Get and order moves
            moves = self._get_ordered_moves(board, moves, depth=current_depth)
            
            alpha = float('-inf')
            beta = float('inf')
            current_best_move = chess.Move.null()
            current_best_score = float('-inf')
            
            for move in moves:
                board.push(move)
                score = -self._negamax(board, current_depth - 1, -beta, -alpha, not (board.turn == color))
                board.pop()
                
                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move
                    self._update_pv_line(move, score, current_depth)
                
                alpha = max(alpha, score)
                
                # Check if we should stop searching
                if self._should_stop_search(current_best_score, current_depth, start_time):
                    break
            
            # Update best move if we completed the depth
            if current_best_move != chess.Move.null():
                best_move = current_best_move
                best_score = current_best_score
                # Keep track of best moves for move ordering
                previous_best_moves.insert(0, best_move)
                if len(previous_best_moves) > 3:  # Keep last 3 best moves
                    previous_best_moves.pop()
            
            # Early exit conditions
            if best_score > 5000:  # Found checkmate
                break
            if self.time_manager.should_stop():
                break
            
            # Update stats
            self.stats.update({
                'depth': current_depth,
                'nodes': self.nodes_searched,
                'score': best_score,
                'time': self.time_manager.get_current_time() - start_time,
                'pv': [str(m['move']) for m in self.pv_move_stack if 'move' in m]
            })
        
        return best_move

    def search(self, board: chess.Board, color: chess.Color) -> chess.Move:
        """Main search handler that routes to the appropriate search algorithm"""
        self.nodes_searched = 0
        self.root_board = board
        start_time = self.time_manager.get_current_time()

        # First check for book moves if available
        if self.opening_book:
            book_move = self.opening_book.get_move(board)
            if book_move:
                return book_move

        # Route to appropriate search algorithm
        if self.search_algorithm == 'minimax':
            best_move = self._minimax_root(board, color)
        elif self.search_algorithm == 'negamax':
            best_move = self._negamax_root(board, color)
        elif self.search_algorithm == 'random':
            best_move = self._random_search(board)
        else:  # 'simple' or fallback
            best_move = self._simple_search(board, color)

        # Check for draw prevention if enabled
        if self.draw_prevention_enabled and best_move != chess.Move.null():
            board.push(best_move)
            is_draw = board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw()
            board.pop()
            
            if is_draw:
                # Try to find alternative move
                legal_moves = list(board.legal_moves)
                legal_moves.remove(best_move)
                if legal_moves:
                    backup_move = self._simple_search(board, color, exclude_moves=[best_move])
                    if backup_move != chess.Move.null():
                        best_move = backup_move

        return best_move

    def _simple_search(self, board: chess.Board, color: chess.Color, exclude_moves: Optional[List[chess.Move]] = None) -> chess.Move:
        """Simple search that evaluates moves at depth 1"""
        best_move = chess.Move.null()
        best_score = float('-inf')
        legal_moves = list(board.legal_moves)
        
        # Remove excluded moves if any
        if exclude_moves:
            legal_moves = [move for move in legal_moves if move not in exclude_moves]

        # Order moves
        legal_moves = self._get_ordered_moves(board, legal_moves, depth=1)

        for move in legal_moves:
            board.push(move)
            
            # Quick checkmate detection
            if board.is_checkmate():
                score = 20000
            else:
                score = self.scoring_calculator.calculate_score(board, color)
                
                # Add quiescence evaluation if enabled
                if self.quiescence_enabled:
                    score = self.quiescence.quiescence_search(board, float('-inf'), float('inf'), color)
                    
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _random_search(self, board: chess.Board) -> chess.Move:
        """Random move selector that still respects move ordering and quiescence"""
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return chess.Move.null()
            
        # Order moves if enabled to prefer better moves in random selection
        if self.move_ordering_enabled:
            legal_moves = self.move_organizer.order_moves(board, legal_moves, depth=0, cutoff=self.max_ordered_moves)
                
        return random.choice(legal_moves)

    def _minimax_root(self, board: chess.Board, color: chess.Color) -> chess.Move:
        """Root minimax search with iterative deepening"""
        best_move = chess.Move.null()
        start_time = self.time_manager.get_current_time()
        current_depth = self.deep_search.get_starting_depth()

        while True:
            moves = list(board.legal_moves)
            moves = self._get_ordered_moves(board, moves, depth=current_depth)

            best_score = float('-inf')
            alpha = float('-inf')
            beta = float('inf')

            for move in moves:
                board.push(move)
                score = self._minimax(board, current_depth - 1, alpha, beta, False, color)
                board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move
                    
                if self.ab_pruning_enabled:
                    alpha = max(alpha, score)

            self.deep_search.update_history(current_depth, best_score, best_move)

            # Check if we should continue deeper
            if not self.deep_search.should_increase_depth(current_depth, best_score, start_time):
                break

            current_depth = self.deep_search.get_next_depth(current_depth)

        return best_move

    def _minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, is_maximizing: bool, color: chess.Color) -> float:
        """Minimax algorithm with alpha-beta pruning"""
        self.nodes_searched += 1

        # Base case: check terminal states
        if board.is_checkmate():
            return -20000 if is_maximizing else 20000
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0

        # Base case: depth reached
        if depth <= 0:
            score = self.scoring_calculator.calculate_score(board, color)
            # Add quiescence search if enabled
            if self.quiescence_enabled:
                score = self.quiescence.quiescence_search(board, alpha, beta, color)
            return score

        moves = list(board.legal_moves)
        moves = self._get_ordered_moves(board, moves, depth=depth)

        if is_maximizing:
            max_eval = float('-inf')
            for move in moves:
                board.push(move)
                eval = self._minimax(board, depth - 1, alpha, beta, False, color)
                board.pop()
                max_eval = max(max_eval, eval)
                if self.ab_pruning_enabled:
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                board.push(move)
                eval = self._minimax(board, depth - 1, alpha, beta, True, color)
                board.pop()
                min_eval = min(min_eval, eval)
                if self.ab_pruning_enabled:
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return min_eval

    def _negamax_root(self, board: chess.Board, color: chess.Color) -> chess.Move:
        """Root negamax search with iterative deepening"""
        best_move = chess.Move.null()
        start_time = self.time_manager.get_current_time()
        current_depth = self.deep_search.get_starting_depth()

        while True:
            moves = list(board.legal_moves)
            moves = self._get_ordered_moves(board, moves, depth=current_depth)

            best_score = float('-inf')
            alpha = float('-inf')
            beta = float('inf')

            for move in moves:
                board.push(move)
                score = -self._negamax(board, current_depth - 1, -beta, -alpha, not color)
                board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move
                    
                if self.ab_pruning_enabled:
                    alpha = max(alpha, score)

            self.deep_search.update_history(current_depth, best_score, best_move)

            # Check if we should continue deeper
            if not self.deep_search.should_increase_depth(current_depth, best_score, start_time):
                break

            current_depth = self.deep_search.get_next_depth(current_depth)

        return best_move

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, color: chess.Color) -> float:
        """Negamax algorithm with alpha-beta pruning"""
        self.nodes_searched += 1

        # Base case: check terminal states
        if board.is_checkmate():
            return -20000
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0

        # Base case: depth reached
        if depth <= 0:
            score = self.scoring_calculator.calculate_score(board, color)
            # Add quiescence search if enabled
            if self.quiescence_enabled:
                score = self.quiescence.quiescence_search(board, alpha, beta, color)
            return score

        moves = list(board.legal_moves)
        moves = self._get_ordered_moves(board, moves, depth=depth)

        max_score = float('-inf')
        for move in moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, not color)
            board.pop()
            
            max_score = max(max_score, score)
            if self.ab_pruning_enabled:
                alpha = max(alpha, score)
                if beta <= alpha:
                    break

        return max_score

    def _get_ordered_moves(self, board: chess.Board, moves, depth: int = 0) -> List[chess.Move]:
        """Helper method to consistently order moves with proper parameters"""
        moves = list(moves) if not isinstance(moves, list) else moves
        if not moves:
            return []
            
        if self.move_ordering_enabled:
            return self.move_organizer.order_moves(board, moves, depth=depth, cutoff=self.max_ordered_moves)
        return moves

    def _search_position(self, board: chess.Board, depth: int, alpha: float = float('-inf'), beta: float = float('inf'), maximize: bool = True) -> Tuple[float, Optional[chess.Move]]:
        """Enhanced search with short-circuit detection and PV tracking"""
        self.nodes_searched += 1

        # 1. Immediate position evaluation (checkmate, stalemate)
        if board.is_checkmate():
            score = float('inf') if not maximize else float('-inf')
            return score, None
            
        if board.is_stalemate():
            return 0.0, None

        # 2. Base case - at max depth or no moves available
        if depth <= 0:
            if self.quiescence_enabled:
                score = self.quiescence.search(board, alpha, beta)  # Use quiescence search's own depth configuration
            else:
                score = self.scoring_calculator.evaluate_position(board)
            return score, None

        # 3. Generate and order moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self.scoring_calculator.evaluate_position(board), None

        if self.move_ordering_enabled:
            ordered_moves = self.move_organizer.order_moves(board, legal_moves)
            if ordered_moves:  # Only use if we got valid ordered moves
                legal_moves = ordered_moves[:self.max_ordered_moves] + [m for m in legal_moves if m not in ordered_moves[:self.max_ordered_moves]]

        # 4. Principal Variation Move Search
        best_move = None
        best_score = float('-inf') if maximize else float('inf')

        for move in legal_moves:
            # Apply move
            board.push(move)
            
            # Recursively search
            score, _ = self._search_position(board, depth - 1, alpha, beta, not maximize)
            
            # Undo move
            board.pop()
            
            # Update best score/move
            if maximize:
                if score > best_score:
                    best_score = score
                    best_move = move
                    alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                    beta = min(beta, best_score)

            # Alpha-beta pruning
            if self.ab_pruning_enabled and beta <= alpha:
                break

        # 5. Update PV line if we found a good move
        if best_move:
            self._update_pv_line(best_move, best_score, depth)

        return best_score, best_move
