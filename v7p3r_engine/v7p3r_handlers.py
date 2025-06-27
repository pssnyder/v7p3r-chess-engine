# v7p3r_engine/v7p3r_handlers.py
import chess
import random
import logging
from typing import Optional, Callable
import time
from v7p3r_engine.v7p3r_search import v7p3rSearch
from v7p3r_engine.v7p3r_score import v7p3rScore
from v7p3r_engine.v7p3r_book import v7p3rBook
from v7p3r_engine.v7p3r_time import v7p3rTime

logging.basicConfig(level=logging.INFO)
v7p3r_handlers_logger = logging.getLogger("v7p3r_handlers_logger")

class v7p3rHandlers:
    def __init__(self, board: chess.Board = chess.Board(), player: chess.Color = chess.WHITE, engine_config: dict = {},):
        self.board = board
        self.current_player = player
        self.logger = engine_config.logger

    def sync_with_game_board(self, game_board: chess.Board):
        if not isinstance(game_board, chess.Board) or not game_board.is_valid():
            if self.logger:
                self.logger.error(f"Invalid game board state detected during sync! | FEN: {getattr(game_board, 'fen', lambda: 'N/A')()}")
            return False
        self.board = game_board.copy()
        self.game_board = game_board.copy()
        return True

    def has_game_board_changed(self):
        if self.game_board is None:
            return False
        return self.board.fen() != self.game_board.fen()

    def search(self, board: chess.Board, player: chess.Color):
        self.nodes_searched = 0
        search_start_time = time.perf_counter()

        self.sync_with_game_board(board)
        self.current_player = player

        current_move_time_limit_ms = self.handler_config.get('move_time_limit', 0)
        if current_move_time_limit_ms is None:
            current_move_time_limit_ms = 0
        self.time_manager.start_timer(current_move_time_limit_ms / 1000.0 if current_move_time_limit_ms > 0 else 0)
        
        book_move = self.opening_book.get_book_move(self.board)
        if book_move and self.board.is_legal(book_move):
            if self.logger:
                self.logger.debug(f"Opening book move found: {book_move} | FEN: {board.fen()}")
            search_duration = time.perf_counter() - search_start_time
            if self.logger:
                self.logger.debug(f"Opening book search took {search_duration:.4f} seconds and searched {self.nodes_searched} nodes.")
            return book_move
        
        if self.logger:
            self.logger.debug(f"== EVALUATION (Player: {'White' if player == chess.WHITE else 'Black'}) == | AI Type: {self.v7p3r_config.get('search_algorithm')} | Depth: {self.v7p3r_config.get('depth')} | Max Depth: {self.max_depth} == ")

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
            

        ordered_moves = self.order_moves(board, legal_moves, depth=self.depth if self.depth is not None else 1)
        
        best_score_overall = -float('inf')
        # For perspective-based evaluation, both players maximize their score from their own perspective
        # so we always start with -inf and maximize

        best_move = ordered_moves[0] if ordered_moves else chess.Move.null()

        for move in ordered_moves:
            if self.time_manager.should_stop(self.depth if self.depth is not None else 1):
                if self.logging_enabled and self.logger:
                    self.logger.info(f"Search stopped due to time limit during move iteration at root. Best move so far: {best_move}")
                break

            temp_board = self.board.copy()
            temp_board.push(move)
            current_move_score = 0.0

            try:
                # self.search_algorithm is correctly set by configure_for_side
                if self.search_algorithm == 'deepsearch':
                    # Pass self.depth (from resolved config) to _deep_search
                    final_deepsearch_move_result = self._deep_search(self.board.copy(), self.depth if self.depth is not None else 1, self.time_control, stop_callback=self.time_manager.should_stop)
                    if final_deepsearch_move_result != chess.Move.null():
                        best_move = final_deepsearch_move_result
                        if self.board.is_legal(best_move): # Check legality on original board
                            temp_board_after_move = self.board.copy()
                            temp_board_after_move.push(best_move)
                            # Evaluate from current player's perspective
                            current_move_score = self.evaluate_position_from_perspective(temp_board_after_move, self.current_player)
                        search_duration = time.perf_counter() - search_start_time
                        if self.logging_enabled and self.logger:
                            self.logger.debug(f"Deepsearch final move selection took {search_duration:.4f} seconds and searched {self.nodes_searched} nodes.")
                        return best_move

                elif self.search_algorithm == 'minimax':
                    current_move_score = self.search_engine._minimax_search(temp_board, (self.depth - 1) if self.depth is not None else 0, -float('inf'), float('inf'), temp_board.turn != self.current_player)
                elif self.search_algorithm == 'negamax':
                    current_move_score = -self.search_engine._negamax_search(temp_board, (self.depth - 1) if self.depth is not None else 0, -float('inf'), float('inf'))
                elif self.search_algorithm == 'lookahead':
                    current_move_score = -self.search_engine._lookahead_search(temp_board, (self.depth - 1) if self.depth is not None else 0, -float('inf'), float('inf'))
                elif self.search_algorithm == 'simple_search': # simple_search itself handles perspective
                    current_move_score = self.evaluate_position_from_perspective(temp_board, self.current_player)
                elif self.search_algorithm == 'evaluation_only':
                    current_move_score = self.evaluate_position_from_perspective(temp_board, self.current_player)
                elif self.search_algorithm == 'random':
                    best_move = self.search_engine._random_search(self.board.copy(), self.current_player)
                    search_duration = time.perf_counter() - search_start_time
                    if self.logging_enabled and self.logger:
                        self.logger.debug(f"Random search took {search_duration:.4f} seconds and searched {self.nodes_searched} nodes.")
                    return best_move # Random search returns a move directly
                else:
                    if self.logger:
                        self.logger.warning(f"Unrecognized AI type '{self.engine_config.search_algorithm}'. Falling back to evaluation_only for score.")
                    current_move_score = self.engine_config.search_engine.evaluate_position_from_perspective(temp_board, self.current_player)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in search algorithm '{self.search_algorithm}' for move {move}: {e}. Using immediate evaluation. | FEN: {temp_board.fen()}")
                current_move_score = self.engine_config.search_engine.evaluate_position_from_perspective(temp_board, self.current_player)

            if current_move_score > best_score_overall:
                best_score_overall = current_move_score
                best_move = move
                print(f"DEBUG: New best move: {move}, score: {current_move_score:.3f}")
            
            if self.logger:
                self.logger.debug(f"Root search iteration: Move={move}, Score={current_move_score:.2f}, Best Move So Far={best_move}, Best Score={best_score_overall:.2f}")

        if best_move == chess.Move.null() and ordered_moves: # Check ordered_moves, not just legal_moves
            best_move = random.choice(ordered_moves) # Fallback to random from ordered if no best move found
        
        if not isinstance(best_move, chess.Move) or not self.board.is_legal(best_move):
            if self.logger:
                self.logger.error(f"Invalid or illegal move detected after search: {best_move}. Selecting a random legal move. | FEN: {self.board.fen()}")
            current_legal_moves = list(self.board.legal_moves) # Get current legal moves
            if current_legal_moves:
                best_move = random.choice(current_legal_moves)
            else:
                best_move = chess.Move.null()
        
        search_duration = time.perf_counter() - search_start_time
        if self.logger:
            player_name = "White" if self.current_player == chess.WHITE else "Black"
            self.logger.debug(f"Search for {player_name} took {search_duration:.4f} seconds and searched {self.nodes_searched} nodes.")

        return best_move
