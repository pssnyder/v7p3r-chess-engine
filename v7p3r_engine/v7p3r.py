# v7p3r_engine/v7p3r.py

""" v7p3r Evaluation Engine
This module implements the evaluation engine for the v7p3r chess AI.
It provides various search algorithms, evaluation functions, and move ordering
"""

import chess
import yaml
import random
import logging
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import datetime
from typing import Optional, Callable, Tuple
from engine_utilities.piece_square_tables import PieceSquareTables
from engine_utilities.time_manager import TimeManager
from engine_utilities.opening_book import OpeningBook
from engine_utilities.v7p3r_scoring_calculation import v7p3rScoringCalculation

# =====================================
# ========== LOGGING SETUP ============
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
v7p3r_engine_logger = logging.getLogger("v7p3r_evaluation_engine")
v7p3r_engine_logger.setLevel(logging.DEBUG)
if not v7p3r_engine_logger.handlers:
    if not os.path.exists('logging'):
        os.makedirs('logging', exist_ok=True)
    from logging.handlers import RotatingFileHandler
    # Use a timestamped log file for each engine run
    timestamp = get_timestamp()
    log_file_path = f"logging/v7p3r_evaluation_engine_{timestamp}.log"
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

class v7p3rEvaluationEngine:
    def __init__(self, board: chess.Board = chess.Board(), player: chess.Color = chess.WHITE):
        self.board = board
        self.current_player = player
        self.time_manager = TimeManager()
        self.opening_book = OpeningBook()
        self.pst = PieceSquareTables()
        self.time_control = {'infinite': True}  # Default to infinite time control

        self.nodes_searched = 0
        self.killer_moves = [[None, None] for _ in range(50)] 
        self.history_table = {}
        self.counter_moves = {}

        # Default Piece Values
        self.piece_values = {
            chess.KING: 0.0,
            chess.QUEEN: 9.0,
            chess.ROOK: 5.0,
            chess.BISHOP: 3.25,
            chess.KNIGHT: 3.0,
            chess.PAWN: 1.0
        }

        # Load Configuration Files
        try:
            with open("config/v7p3r_config.yaml") as f:
                self.v7p3r_config = yaml.safe_load(f) or {}
            with open("config/chess_game_config.yaml") as f:
                self.chess_game_config = yaml.safe_load(f) or {}
        except Exception as e:
            v7p3r_engine_logger.error(f"Error loading v7p3r or game settings YAML files: {e}")
            self.v7p3r_config = {}
            self.chess_game_config = {}

        # Engine Name
        self.engine_name = self.v7p3r_config.get('engine_name', 'v7p3r')
        self.engine_version = self.v7p3r_config.get('engine_version', '1.0.0')

        # Performance Config
        self.performance_config = self.chess_game_config.get('performance', {})
        self.max_moves_to_evaluate = self.performance_config.get('max_moves_evaluated', 50)
        self.async_enabled = self.performance_config.get('async_mode', False)
        self.thread_limit = self.performance_config.get('thread_limit', 2)
        
        # Monitoring Config
        self.monitoring_config = self.chess_game_config.get('monitoring', {})
        self.logging_enabled = self.monitoring_config.get('enable_logging', False)
        self.evaluation_display_enabled = self.monitoring_config.get('show_evaluation', False)
        self.show_thoughts = self.monitoring_config.get('show_thinking', False)
        self.logger = v7p3r_engine_logger
        if not self.logging_enabled:
            self.show_thoughts = False
        if self.logging_enabled:
            self.logger.debug(f"Logging enabled for {player} v7p3rEvaluationEngine")
        
        # Engine Config
        self.configure_engine(self.board, self.engine_name)

        self.reset()

    def configure_engine(self, board: chess.Board, engine_name: str):
        """ Configures the engine for the current side based on the engine name """
        # Dynamically fetch the config for this engine
        self.engine_config = self.v7p3r_config.get(engine_name, {})  # Dynamically load configuration based on engine_name

        # Set up config values for evaluator and scoring
        self.ruleset = self.engine_config.get('ruleset')
        self.search_algorithm = self.engine_config.get('search_algorithm', 'random')
        self.depth = self.engine_config.get('depth', 3)
        self.max_depth = self.engine_config.get('max_depth', 4)
        self.solutions_enabled = self.engine_config.get('use_solutions', False)
        self.opening_book_enabled = self.engine_config.get('use_opening_book', False)
        self.pst_enabled = self.engine_config.get('pst', False)
        self.pst_weight = self.engine_config.get('pst_weight', 1.0)
        self.move_ordering_enabled = self.engine_config.get('move_ordering', False)
        self.quiescence_enabled = self.engine_config.get('quiescence', False)
        self.move_time_limit = self.engine_config.get('time_limit', 0)
        self.scoring_modifier = self.engine_config.get('scoring_modifier', 1.0)
        self.game_phase_awareness = self.engine_config.get('game_phase_awareness', False)
        self.engine_color = 'white' if board.turn else 'black'

        # Initialize a scoring setup for this engine config
        if self.logging_enabled and self.logger:
            self.logger.debug(f"Configuring v7p3r AI for {self.engine_color} via: {self.engine_config}")
        self.scoring_calculator = v7p3rScoringCalculation(self.engine_config, self.v7p3r_config)
        if self.show_thoughts and self.logger:
            self.logger.debug(f"AI configured for {self.engine_color}: type={self.search_algorithm} depth={self.depth}, ruleset={self.ruleset}")

    def close(self):
        self.reset()
        if self.show_thoughts and self.logger:
            self.logger.debug("v7p3rEvaluationEngine closed and resources cleaned up.")

    def reset(self):
        if self.board is None:
            board = chess.Board()
        else:
            board = self.board
        self.current_player = chess.WHITE if board.turn else chess.BLACK
        self.nodes_searched = 0
        self.killer_moves = [[None, None] for _ in range(50)]
        self.history_table.clear()
        self.counter_moves.clear()
        if self.show_thoughts and self.logger:
            self.logger.debug(f"v7p3rEvaluationEngine for {self.engine_color} reset to initial state.")
        
        self.configure_engine(self.board, self.engine_name)

    def _is_draw_condition(self, board):
        if board.can_claim_threefold_repetition():
            return True
        if board.can_claim_fifty_moves():
            return True
        if board.is_seventyfive_moves():
            return True
        return False

    def _get_game_phase_factor(self, board: chess.Board) -> float:
        if not self.game_phase_awareness:
            return 0.0
        
        total_material = 0
        for piece_type, value in self.piece_values.items():
            if piece_type != chess.KING:
                total_material += len(board.pieces(piece_type, chess.WHITE)) * value
                total_material += len(board.pieces(piece_type, chess.BLACK)) * value

        QUEEN_ROOK_MATERIAL = self.piece_values[chess.QUEEN] + self.piece_values[chess.ROOK]
        TWO_ROOK_MATERIAL = self.piece_values[chess.ROOK] * 2
        KNIGHT_BISHOP_MATERIAL = self.piece_values[chess.KNIGHT] + self.piece_values[chess.BISHOP]

        if total_material >= (QUEEN_ROOK_MATERIAL * 2) + (KNIGHT_BISHOP_MATERIAL * 2):
            return 0.0
        if total_material < (TWO_ROOK_MATERIAL + KNIGHT_BISHOP_MATERIAL * 2) and total_material > (KNIGHT_BISHOP_MATERIAL * 2):
            return 0.5
        if total_material <= (KNIGHT_BISHOP_MATERIAL * 2):
            return 1.0
        
        return 0.0

    # =================================
    # ===== MOVE SEARCH HANDLER =======

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

    def search(self, board: chess.Board, player: chess.Color, engine_config: dict = {}, stop_callback: Optional[Callable[[], bool]] = None) -> chess.Move:
        print(f"DEBUG: Starting search with algorithm: {self.search_algorithm}")
        self.nodes_searched = 0
        search_start_time = time.perf_counter()

        self.sync_with_game_board(board)
        self.current_player = player

        current_move_time_limit_ms = self.move_time_limit
        if current_move_time_limit_ms is None:
            current_move_time_limit_ms = 0
        self.time_manager.start_timer(current_move_time_limit_ms / 1000.0 if current_move_time_limit_ms > 0 else 0)
        
        if self.opening_book_enabled:
            book_move = self.opening_book.get_book_move(self.board)
            if book_move and self.board.is_legal(book_move):
                if self.show_thoughts and self.logger:
                    self.logger.debug(f"Opening book move found: {book_move} | FEN: {board.fen()}")
                search_duration = time.perf_counter() - search_start_time
                if self.logging_enabled and self.logger:
                    self.logger.debug(f"Opening book search took {search_duration:.4f} seconds and searched {self.nodes_searched} nodes.")
                return book_move
        
        if self.show_thoughts:
            # self.engine_config['search_algorithm'] and self.engine_config['depth'] are now correct
            self.logger.debug(f"== EVALUATION (Player: {'White' if player == chess.WHITE else 'Black'}) == | AI Type: {self.engine_config.get('search_algorithm')} | Depth: {self.engine_config.get('depth')} | Max Depth: {self.max_depth} == ")

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
            

        if self.move_ordering_enabled: # move_ordering_enabled is set by configure_for_side
            ordered_moves = self.order_moves(board, legal_moves, depth=self.depth if self.depth is not None else 1)
        else:
            ordered_moves = legal_moves # No ordering if disabled
        
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
                    current_move_score = self._minimax_search(temp_board, (self.depth - 1) if self.depth is not None else 0, -float('inf'), float('inf'), temp_board.turn != self.current_player)
                elif self.search_algorithm == 'negamax':
                    current_move_score = -self._negamax_search(temp_board, (self.depth - 1) if self.depth is not None else 0, -float('inf'), float('inf'))
                elif self.search_algorithm == 'lookahead':
                    current_move_score = -self._lookahead_search(temp_board, (self.depth - 1) if self.depth is not None else 0, -float('inf'), float('inf'))
                elif self.search_algorithm == 'simple_search': # simple_search itself handles perspective
                    current_move_score = self.evaluate_position_from_perspective(temp_board, self.current_player)
                elif self.search_algorithm == 'evaluation_only':
                    current_move_score = self.evaluate_position_from_perspective(temp_board, self.current_player)
                elif self.search_algorithm == 'random':
                    best_move = self._random_search(self.board.copy(), self.current_player)
                    search_duration = time.perf_counter() - search_start_time
                    if self.logging_enabled and self.logger:
                        self.logger.debug(f"Random search took {search_duration:.4f} seconds and searched {self.nodes_searched} nodes.")
                    return best_move # Random search returns a move directly
                else:
                    if self.show_thoughts and self.logger:
                        self.logger.warning(f"Unrecognized AI type '{self.search_algorithm}'. Falling back to evaluation_only for score.")
                    current_move_score = self.evaluate_position_from_perspective(temp_board, self.current_player)
            except Exception as e:
                if self.logging_enabled and self.logger:
                    self.logger.error(f"Error in search algorithm '{self.search_algorithm}' for move {move}: {e}. Using immediate evaluation. | FEN: {temp_board.fen()}")
                current_move_score = self.evaluate_position_from_perspective(temp_board, self.current_player)

            if current_move_score > best_score_overall:
                best_score_overall = current_move_score
                best_move = move
                print(f"DEBUG: New best move: {move}, score: {current_move_score:.3f}")
            
            if self.show_thoughts and self.logger:
                self.logger.debug(f"Root search iteration: Move={move}, Score={current_move_score:.2f}, Best Move So Far={best_move}, Best Score={best_score_overall:.2f}")

        if best_move == chess.Move.null() and ordered_moves: # Check ordered_moves, not just legal_moves
            best_move = random.choice(ordered_moves) # Fallback to random from ordered if no best move found
        
        if not isinstance(best_move, chess.Move) or not self.board.is_legal(best_move):
            if self.logging_enabled and self.logger:
                self.logger.error(f"Invalid or illegal move detected after search: {best_move}. Selecting a random legal move. | FEN: {self.board.fen()}")
            current_legal_moves = list(self.board.legal_moves) # Get current legal moves
            if current_legal_moves:
                best_move = random.choice(current_legal_moves)
            else:
                best_move = chess.Move.null()
        
        search_duration = time.perf_counter() - search_start_time
        if self.logging_enabled and self.logger:
            player_name = "White" if self.current_player == chess.WHITE else "Black"
            self.logger.debug(f"Search for {player_name} took {search_duration:.4f} seconds and searched {self.nodes_searched} nodes.")

        return best_move

    # =================================
    # ===== EVALUATION FUNCTIONS ======

    def evaluate_position_from_perspective(self, board: chess.Board, player: chess.Color) -> float:
        """Calculate position evaluation from specified player's perspective by delegating to scoring_calculator."""
        perspective_evaluation_board = board.copy()
        if not isinstance(player, chess.Color) or not perspective_evaluation_board.is_valid():
            if self.logger:
                player_name = "White" if player == chess.WHITE else "Black" if isinstance(player, chess.Color) else str(player)
                self.logger.error(f"Invalid input for evaluation from perspective. Player: {player_name}, FEN: {perspective_evaluation_board.fen() if hasattr(perspective_evaluation_board, 'fen') else 'N/A'}")
            return 0.0
        
        endgame_factor = self._get_game_phase_factor(perspective_evaluation_board)

        white_score = self.scoring_calculator.calculate_score(
            board=perspective_evaluation_board,
            color=chess.WHITE,
            endgame_factor=endgame_factor
        )
        black_score = self.scoring_calculator.calculate_score(
            board=perspective_evaluation_board,
            color=chess.BLACK,
            endgame_factor=endgame_factor
        )
        
        score = (white_score - black_score) if player == chess.WHITE else (black_score - white_score)
        
        if self.logging_enabled and self.logger:
            player_name = "White" if player == chess.WHITE else "Black"
            self.logger.debug(f"Position evaluation from {player_name} perspective (delegated): {score:.3f} | FEN: {perspective_evaluation_board.fen()} | Endgame Factor: {endgame_factor:.2f}")
        return score

    def evaluate_move(self, board: chess.Board, move: chess.Move = chess.Move.null()) -> float:
        """Quick evaluation of individual move on overall eval"""
        score = 0.0
        move_evaluation_board = board.copy()
        if not move_evaluation_board.is_legal(move):
            if self.logging_enabled and self.logger:
                self.logger.error(f"Attempted evaluation of an illegal move: {move} | FEN: {board.fen()}")
            return -9999999999
        
        move_evaluation_board.push(move)
        score = self.evaluate_position_from_perspective(move_evaluation_board, self.current_player)
        
        if self.show_thoughts and self.logger:
            self.logger.debug("Exploring the move: %s | Evaluation: %.3f | FEN: %s", move, score, board.fen())
        move_evaluation_board.pop()
        return score

    # ===================================
    # ======= HELPER FUNCTIONS ==========
    
    def order_moves(self, board: chess.Board, moves, depth: int = 0):
        """Order moves for better alpha-beta pruning efficiency"""
        if isinstance(moves, chess.Move):
            moves = [moves]
        
        if not moves or not isinstance(board, chess.Board) or not board.is_valid():
            if self.logger:
                self.logger.error(f"Invalid input to order_moves: board type {type(board)} | FEN: {board.fen() if hasattr(board, 'fen') else 'N/A'}")
            return []

        move_scores = []
        
        for move in moves:
            if not board.is_legal(move):
                if self.logger:
                    self.logger.warning(f"Illegal move passed to order_moves: {move} | FEN: {board.fen()}")
                continue
            
            score = self._order_move_score(board, move, depth)
            move_scores.append((move, score))

        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # max_moves_to_evaluate can come from v7p3r_config_data or game_settings_config_data (performance section)
        # self.engine_config should have this resolved value
        max_moves_to_evaluate = self.engine_config.get('max_moves_evaluated', None)
        
        if max_moves_to_evaluate is not None and max_moves_to_evaluate > 0:
            if self.logging_enabled and self.logger:
                self.logger.debug(f"Limiting ordered moves from {len(move_scores)} to {max_moves_to_evaluate}")
            move_scores = move_scores[:max_moves_to_evaluate]

        if self.logging_enabled and self.logger:
            self.logger.debug(f"Ordered moves at depth {depth}: {[f'{move} ({score:.2f})' for move, score in move_scores]} | FEN: {board.fen()}")
        
        return [move for move, _ in move_scores]

    def _order_move_score(self, board: chess.Board, move: chess.Move, depth: int = 0) -> float:
        """Calculate a score for a move for ordering purposes."""
        score = 0.0

        temp_board = board.copy()
        temp_board.push(move)
        if temp_board.is_checkmate():
            temp_board.pop()
            # Get checkmate move bonus from the current ruleset
            return self.scoring_calculator.get_rule_value('checkmate_move_bonus', 1000000.0)
        
        if temp_board.is_check(): # Check after move is made
            score += self.scoring_calculator.get_rule_value('check_move_bonus', 10000.0)
        temp_board.pop() # Pop before is_capture check on original board state

        if board.is_capture(move):
            score += self.scoring_calculator.get_rule_value('capture_move_bonus', 4000.0)
            victim_type = board.piece_type_at(move.to_square)
            aggressor_type = board.piece_type_at(move.from_square)
            if victim_type and aggressor_type:
                score += (self.piece_values.get(victim_type, 0) * 10) - self.piece_values.get(aggressor_type, 0)

        if depth < len(self.killer_moves) and move in self.killer_moves[depth]:
            score += self.scoring_calculator.get_rule_value('killer_move_bonus', 2000.0)

        # Countermove heuristic (if applicable)
        # if board.move_stack and self.counter_moves.get(board.move_stack[-1], None) == move:
        # score += self.scoring_calculator.get_rule_value('counter_move_bonus', 1000.0)

        score += self.history_table.get((board.turn, move.from_square, move.to_square), 0)
        
        if move.promotion:
            score += self.scoring_calculator.get_rule_value('promotion_move_bonus', 3000.0)
            if move.promotion == chess.QUEEN:
                score += self.piece_values.get(chess.QUEEN, 9.0) * 100 # Ensure piece_values is used

        return score
    
    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, maximizing_player: bool, stop_callback: Optional[Callable[[], bool]] = None, current_ply: int = 0) -> float:
        self.nodes_searched += 1

        if stop_callback and stop_callback(): # Pass current search depth and nodes
            return 0 # Or appropriate alpha/beta

        # Use self.current_player for perspective
        stand_pat_score = self.evaluate_position_from_perspective(board, self.current_player if maximizing_player else not self.current_player)

        if not maximizing_player: # Minimizing node (opponent's turn from perspective of self.current_player)
            stand_pat_score = -stand_pat_score # Adjust score if eval is always from white's POV

        if maximizing_player: # Current player (the one who initiated the search) is maximizing
            if stand_pat_score >= beta:
                return beta 
            alpha = max(alpha, stand_pat_score)
        else: # Opponent is minimizing (from current player's perspective)
            if stand_pat_score <= alpha:
                return alpha
            beta = min(beta, stand_pat_score)

        # Consider only captures and promotions (and maybe checks)
        # Use a fixed quiescence max depth (quiescence is a boolean in config, not a dict)
        max_q_depth = 5  # Fixed reasonable quiescence depth
        if current_ply >= self.depth + max_q_depth: # self.depth is the main search depth
             return stand_pat_score


        capture_moves = [move for move in board.legal_moves if board.is_capture(move) or move.promotion]
        if not capture_moves and not board.is_check(): # If no captures and not in check, return stand_pat
             return stand_pat_score
        
        # If in check, all legal moves should be considered to escape check.
        if board.is_check():
            capture_moves = list(board.legal_moves)


        # Order capture moves (e.g., MVV-LVA or simple capture value)
        # For simplicity, not re-ordering here, but could be beneficial.
        # ordered_q_moves = self.order_moves(board, capture_moves, depth=current_ply) # Can reuse order_moves logic

        for move in capture_moves: # Potentially use ordered_q_moves
            board.push(move)
            score = self._quiescence_search(board, alpha, beta, not maximizing_player, stop_callback, current_ply + 1)
            board.pop()

            if maximizing_player:
                alpha = max(alpha, score)
                if alpha >= beta:
                    break 
            else:
                beta = min(beta, score)
                if alpha >= beta:
                    break
        
        return alpha if maximizing_player else beta
    
    def update_killer_move(self, move, ply): # Renamed depth to ply for clarity, as it's depth in current search tree
        """Update killer move table with a move that caused a beta cutoff"""
        if ply >= len(self.killer_moves): # Ensure ply is within bounds
            return
        
        if move not in self.killer_moves[ply]:
            self.killer_moves[ply].insert(0, move)
            self.killer_moves[ply] = self.killer_moves[ply][:2]

    def update_history_score(self, board, move, depth):
        """Update history heuristic score for a move that caused a beta cutoff"""
        piece = board.piece_at(move.from_square)
        if piece is None:
            return
        history_key = (piece.piece_type, move.from_square, move.to_square)

        # Update history score using depth-squared bonus
        self.history_table[history_key] = self.history_table.get(history_key, 0) + depth * depth

    # =======================================
    # ======= MAIN SEARCH ALGORITHMS ========
    
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
            score = self.evaluate_position_from_perspective(temp_board, simple_search_board.turn)
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
            checkmate_bonus = self.scoring_calculator.get_rule_value('checkmate_bonus', 1000000.0)
            if abs(best_score_root) > checkmate_bonus / 2: # Checkmate score is very high
                if self.logging_enabled and self.logger:
                    self.logger.info(f"Deepsearch found a potential checkmate at depth {iterative_depth}. Stopping early.")
                break

            if self.show_thoughts and self.logger:
                self.logger.debug(f"Deepsearch finished depth {iterative_depth}: Best move {best_move_root} with score {best_score_root:.2f}")

        return best_move_root if best_move_root != chess.Move.null() else self._simple_search(board) # Fallback if no move found


# Example usage and Testing:
def main():
    # Example usage of the v7p3rEvaluationEngine
    engine = v7p3rEvaluationEngine()
    board = chess.Board()
    starting_positions = engine.v7p3r_config.get('starting_positions', {})
    player = chess.WHITE if board.turn else chess.BLACK
    move = engine.search(board, player)
    print(f"Best move for {('White' if player == chess.WHITE else 'Black')}: {move}")
    for position in starting_positions:
        print(f"Testing position: {position}")
        board.reset()  # Reset the board to the starting position
        board.set_fen(starting_positions[position])
        player = chess.WHITE if board.turn else chess.BLACK
        move = engine.search(board, player)
        print(f"Best move for {('White' if player == chess.WHITE else 'Black')} in position '{position}': {move}")
if __name__ == "__main__":
    main()