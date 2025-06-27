# v7p3r_engine/search_algorithms.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import chess
import random
import logging
import datetime
from typing import Callable, Optional
from v7p3r_engine.v7p3r_time import v7p3rTime
from v7p3r_engine.v7p3r_score import v7p3rScore
from v7p3r_engine.v7p3r_ordering import v7p3rOrdering
from v7p3r_engine.v7p3r_pst import v7p3rPST


# =====================================
# ========== LOGGING SETUP ============
v7p3r_engine_logger = logging.getLogger("v7p3r_search_logger")
v7p3r_engine_logger.setLevel(logging.DEBUG)
if not v7p3r_engine_logger.handlers:
    if not os.path.exists('logging'):
        os.makedirs('logging', exist_ok=True)
    from logging.handlers import RotatingFileHandler
    # Use a timestamped log file for each engine run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"logging/v7p3r_search_{timestamp}.log"
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10*1024*1024,
        backupCount=3,
        delay=True
    )
    formatter = logging.Formatter(
        '%(asctime)s | %(funcName)-15s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    v7p3r_engine_logger.addHandler(file_handler)
    v7p3r_engine_logger.propagate = False

# =======================================
# ======= MAIN SEARCH CLASS ========
class v7p3rSearch:
    def __init__(self, board: chess.Board, current_player: chess.Color, v7p3r_config: dict, chess_game_config: dict):
        self.v7p3r_config = v7p3r_config
        self.time_manager = v7p3rTime()
        self.scoring_calculator = v7p3rScore(v7p3r_config, chess_game_config)
        self.logger = v7p3r_engine_logger
        self.show_thoughts = chess_game_config.get('monitoring', {}).get('show_thinking', False)

    def _random_search(self, board: chess.Board, player: chess.Color) -> chess.Move:
        """Select a random legal move from the board."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            if self.show_thoughts and self.logger:
                self.logger.debug(f"No legal moves available | FEN: {board.fen()}")
            return chess.Move.null() # Return null move if no legal moves
        move = random.choice(legal_moves)
        if self.show_thoughts and self.logger:
            self.logger.debug(f"Randomly selected move: {move} | FEN: {board.fen()}")
        return move

    def _simple_search(self, board: chess.Board) -> chess.Move:
        """Simple search that evaluates all legal moves and picks the best one at 1 ply."""
        best_move = chess.Move.null()
        # Initialize best_score to negative infinity for white, positive infinity for black for proper min/max
        best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
        simple_search_board = board.copy()  # Work on a copy of the board
        legal_moves = list(simple_search_board.legal_moves)
        if not legal_moves:
            return chess.Move.null() # No legal moves, likely game over
        for move in legal_moves:
            if self.time_manager.should_stop(self.depth if self.depth is not None else 1, self.nodes_searched):
                break # Stop if time is up
            self.nodes_searched += 1 # Increment nodes searched
            temp_board = simple_search_board.copy()
            temp_board.push(move)
            score = self.evaluator.evaluate_position_from_perspective(temp_board, simple_search_board.turn)
            if self.show_thoughts and self.logger:
                self.logger.debug(f"Simple search evaluating move: {move} | Score: {score:.3f} | Best score: {best_score:.3f} | FEN: {temp_board.fen()}")
            if simple_search_board.turn == chess.WHITE: # If white's turn, maximize score
                if score > best_score:
                    best_score = score
                    best_move = move
            else: # If black's turn, minimize score
                if score < best_score:
                    best_score = score
                    best_move = move
            
        return best_move

    def _lookahead_search(self, board: chess.Board, depth: int, alpha: float, beta: float):
        """Lookahead search with static depth. Returns score (float)."""
        self.nodes_searched += 1 # Increment nodes searched

        if depth == 0 or board.is_game_over(claim_draw=self._is_draw_condition(board)):
            if self.quiescence_enabled:
                return self._quiescence_search(board, alpha, beta, True)
            else:
                return self.evaluate_position_from_perspective(board, board.turn)

        best_score = -float('inf') # Always maximizing from the current player's perspective

        legal_moves = list(board.legal_moves)
        if self.move_ordering_enabled:
            legal_moves = self.order_moves(board, legal_moves, depth=depth)

        for move in legal_moves:
            board.push(move)
            # Recursive call: _lookahead_search only returns score (float)
            score = -self._lookahead_search(board, depth - 1, -beta, -alpha)
            board.pop()

            if score > best_score:
                best_score = score
            alpha = max(alpha, best_score)
            if alpha >= beta:
                self.update_killer_move(move, depth)
                self.update_history_score(board, move, depth) # Update history for cutoff moves
                break # Prune remaining moves at this depth
        
        return best_score

    def _minimax_search(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool):
        """Minimax search with alpha-beta pruning. Returns score (float)."""
        self.nodes_searched += 1 # Increment nodes searched
        if depth == 0 or board.is_game_over(claim_draw=self._is_draw_condition(board)):
            if self.quiescence_enabled:
                return self._quiescence_search(board, alpha, beta, maximizing_player)
            else:
                return self.evaluate_position_from_perspective(board, board.turn)
        best_score = -float('inf') if maximizing_player else float('inf')
        legal_moves = list(board.legal_moves)
        if self.move_ordering_enabled:
            legal_moves = self.order_moves(board, legal_moves, depth=depth)
        for move in legal_moves:
            board.push(move)
            score = self._minimax_search(board, depth-1, alpha, beta, not maximizing_player)
            board.pop()
            if maximizing_player:
                if score > best_score:
                    best_score = score
            else: # Minimizing player
                if score < best_score:
                    best_score = score
            # Alpha-beta pruning update
            if maximizing_player:
                alpha = max(alpha, score)
                if alpha >= beta:
                    self.update_killer_move(move, depth)
                    self.update_history_score(board, move, depth) # Update history for cutoff moves
                    break # Alpha-beta cutoff
            else: # Minimizing player
                beta = min(beta, score)
                if alpha >= beta:
                    break # Alpha-beta cutoff
        
        return best_score

    def _negamax_search(self, board: chess.Board, depth: int, alpha: float, beta: float, stop_callback: Optional[Callable[[], bool]] = None) -> float:
        self.nodes_searched += 1
        if stop_callback and stop_callback():
            return self.evaluate_position_from_perspective(board, board.turn)

        if depth == 0 or board.is_game_over(claim_draw=self._is_draw_condition(board)):
            if self.quiescence_enabled:
                return self._quiescence_search(board, alpha, beta, True, stop_callback)
            else:
                return self.evaluate_position_from_perspective(board, board.turn)

        best_score = -float('inf')
        best_move_local = None
        
        legal_moves = list(board.legal_moves)
        if self.move_ordering_enabled:
            legal_moves = self.order_moves(board, legal_moves, depth=depth)

        for move in legal_moves:
            board.push(move)
            # Recursive call: _negamax_search now always returns a score (float)
            score = -self._negamax_search(board, depth-1, -beta, -alpha, stop_callback)
            board.pop()

            if score > best_score:
                best_score = score
                best_move_local = move
            
            alpha = max(alpha, score)
            if alpha >= beta:
                self.update_killer_move(move, depth)
                self.update_history_score(board, move, depth) # Update history for cutoff moves
                break # Alpha-beta cutoff

        return best_score

    
    def _deep_search(self, board: chess.Board, depth: int, time_control: dict, current_depth: int = 1, stop_callback: Optional[Callable[[], bool]] = None) -> chess.Move:
        """Iterative deepening search with time management."""
        best_move_root = chess.Move.null() # The best move found at the root of the search
        best_score_root = -float('inf')
        
        # Define legal_moves at the beginning of the method for the current board state
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null() # No legal moves to search

        # Use max_depth from the resolved AI config for this search
        iterative_max_depth = self.v7p3r_config.get('max_depth', 5)

        # The 'depth' parameter passed to _deep_search can be the initial target depth from engine_config
        # Iterative deepening will go from current_depth up to iterative_max_depth or target 'depth'
        # Ensure we respect the overall max_depth from config.
        search_depth_limit = min(depth, iterative_max_depth)


        for iterative_depth in range(current_depth, search_depth_limit + 1):
            if stop_callback and stop_callback(): # Call stop_callback without arguments
                if self.logging_enabled and self.logger:
                    self.logger.info(f"Deepsearch stopped due to time limit at depth {iterative_depth-1}.")
                break # Stop if time runs out

            if self.time_manager.should_stop(self.depth if self.depth is not None else 1):
                if self.logging_enabled and self.logger:
                    self.logger.info(f"Deepsearch stopped by time manager at depth {iterative_depth-1}.")
                break # Stop if time manager says so

            # Get legal moves for the current board state for this iteration
            # This was the source of the error, legal_moves should be defined before this loop.
            # It's already defined at the start of the function for the initial board state.
            # If the board state for ordering changes per iteration (it doesn't here), it would need re-evaluation.
            # For iterative deepening, we consider moves from the root board state.
            
            current_iter_legal_moves = list(board.legal_moves) # Ensure we use the root board's legal moves for each iteration.
            if not current_iter_legal_moves: # Should not happen if initial check passed, but good for safety.
                break

            # Re-order moves at each iteration if move ordering is enabled
            if self.move_ordering_enabled:
                ordered_moves = self.order_moves(board, current_iter_legal_moves, depth=iterative_depth)
            else:
                ordered_moves = current_iter_legal_moves

            local_best_move_at_depth = chess.Move.null()
            local_best_score_at_depth = -float('inf')

            # Alpha and Beta for the current iteration
            alpha = -float('inf')  # Ensure alpha is initialized as a float
            beta = float('inf')    # Ensure beta is initialized as a float

            for move in ordered_moves:
                if stop_callback and stop_callback():
                    break # Stop if time is up mid-iteration

                temp_board = board.copy()
                temp_board.push(move)
                
                # Recursive call to negamax (or negascout, or minimax)
                # _negamax_search now always returns a score (float)
                current_move_score = -self._negamax_search(temp_board, iterative_depth - 1, -beta, -alpha, stop_callback)

                if current_move_score > local_best_score_at_depth:
                    local_best_score_at_depth = current_move_score
                    local_best_move_at_depth = move
                
                alpha = max(alpha, current_move_score) # Update alpha for the next sibling move
                if alpha >= beta:
                    # Beta cutoff, update killer and history
                    self.update_killer_move(move, iterative_depth)
                    self.update_history_score(board, move, iterative_depth)
                    break # Prune remaining moves at this depth
            
            # After each full depth iteration, update the overall best move
            if local_best_move_at_depth != chess.Move.null():
                best_move_root = local_best_move_at_depth
                best_score_root = local_best_score_at_depth
            
            # If checkmate is found, stop early
            checkmate_bonus = self.scoring_calculator.rules.get('checkmate_bonus', 1000000.0)
            if abs(best_score_root) > checkmate_bonus / 2: # Checkmate score is very high
                if self.logging_enabled and self.logger:
                    self.logger.info(f"Deepsearch found a potential checkmate at depth {iterative_depth}. Stopping early.")
                break

            if self.show_thoughts and self.logger:
                self.logger.debug(f"Deepsearch finished depth {iterative_depth}: Best move {best_move_root} with score {best_score_root:.2f}")

        return best_move_root if best_move_root != chess.Move.null() else self._simple_search(board) # Fallback if no move found