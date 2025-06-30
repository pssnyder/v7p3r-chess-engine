# v7p3r_engine/search_algorithms.py

import sys
import os
import chess
import random
import logging
import datetime

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base = getattr(sys, '_MEIPASS', None)
    if base:
        return os.path.join(base, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def get_log_file_path():
    # Optional timestamp for log file name
    timestamp = get_timestamp()
    log_dir = "logging"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"v7p3r_evaluation_engine.log")
v7p3r_engine_logger = logging.getLogger("v7p3r_evaluation_engine")
v7p3r_engine_logger.setLevel(logging.DEBUG)
_init_status = globals().get("_init_status", {})
if not _init_status.get("initialized", False):
    log_file_path = get_log_file_path()
    from logging.handlers import RotatingFileHandler
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
    _init_status["initialized"] = True
    # Store the log file path for later use (e.g., to match with PGN/config)
    _init_status["log_file_path"] = log_file_path

class v7p3rSearch:
    def __init__(self, engine_config: dict, scoring_calculator, move_organizer, time_manager, opening_book, logger: logging.Logger):
        # Configuration
        self.engine_config = engine_config
        self.search_algorithm = engine_config.get('search_algorithm', 'simple_search')  # Default to simple_search if not set
        self.depth = engine_config.get('depth', 2)  # Placeholder for engine reference
        self.max_depth = engine_config.get('max_depth', 5)  # Max depth of search for AI, default to 8 if not set
        self.logger = logger if logger else logging.getLogger("v7p3r_engine_logger")
        self.nodes_searched = 0  # Initialize nodes searched counter
        self.monitoring_enabled = engine_config.get('monitoring_enabled', False)  # Enable/disable monitoring

        # Required Search Modules
        self.scoring_calculator = scoring_calculator
        self.move_organizer = move_organizer
        self.time_manager = time_manager
        self.opening_book = opening_book

        # Engine Search Types (search algorithms used by v7p3r)
        self.search_algorithms = [
            'deepsearch',                    # Dynamic deep search via negamax with time control and iterative deepening
            'minimax',                       # Minimax search with alpha-beta pruning
            'negamax',                       # Negamax search with alpha-beta pruning
            'random',                        # Random move selection
        ]
    
    def search(self, board: chess.Board, player: chess.Color):
        """Perform a search for the best move for the given player."""
        self.current_player = player
        root_board = board.copy()
        self.board = root_board
        self.nodes_searched = 0  # Reset nodes searched for this search

        # Check for book moves
        book_move = self.opening_book.get_book_move(root_board)
        if book_move and root_board.is_legal(book_move):
            if self.monitoring_enabled and self.logger:
                self.logger.debug(f"Opening book move found: {book_move} | FEN: {root_board.fen()}")
            return book_move

        if self.monitoring_enabled and self.logger:
            self.logger.debug(f"== EVALUATION (Player: {'White' if player == chess.WHITE else 'Black'}) == | Search Type: {self.search_algorithm} | Depth: {self.depth} | Max Depth: {self.max_depth} == ")

        legal_moves = list(root_board.legal_moves)
        legal_moves = self.move_organizer.order_moves(root_board, legal_moves, depth=self.depth)
        if not legal_moves:
            if self.monitoring_enabled and self.logger:
                self.logger.error(f"[Error] No legal moves found for player: {'White' if player == chess.WHITE else 'Black'} | FEN: {root_board.fen()}")
            return chess.Move.null()

        best_score_overall = -float('inf')
        best_move = chess.Move.null()
        for move in legal_moves:
            temp_board = root_board.copy()
            temp_board.push(move)
            current_move_score = 0.0

            # Check for immediate checkmate or stalemate
            if temp_board.is_checkmate():
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Checkmate move found: {move} | FEN: {temp_board.fen()}")
                return move
            elif self._draw_search(temp_board, depth=1):
                continue

            try:
                if self.search_algorithm == 'deepsearch':
                    current_move_score = self._deep_search(root_board, self.depth, self.max_depth)[1]
                elif self.search_algorithm == 'minimax':
                    current_move_score = self._minimax_search(temp_board, self.depth, -float('inf'), float('inf'), False)
                elif self.search_algorithm == 'negamax':
                    current_move_score = self._negamax_search(temp_board, self.depth, -float('inf'), float('inf'))
                elif self.search_algorithm == 'simple':
                    current_move_score = self.scoring_calculator.evaluate_position(temp_board)
                elif self.search_algorithm == 'quiescence':
                    current_move_score = self._quiescence_search(temp_board, -float('inf'), float('inf'), True)
                elif self.search_algorithm == 'random':
                    current_move_score = random.uniform(-1.0, 1.0)
                else:
                    raise ValueError(f"Unknown search algorithm: {self.search_algorithm}")
            except Exception as e:
                if self.monitoring_enabled and self.logger:
                    self.logger.error(f"Error in search algorithm '{self.search_algorithm}' for move {move}: {e}. | FEN: {temp_board.fen()}")
                    continue
            if current_move_score > best_score_overall:
                best_score_overall = current_move_score
                best_move = move
                if self.monitoring_enabled and self.logger:
                    self.logger.debug(f"New best move: {move}, score: {current_move_score:.3f}")

            if self.monitoring_enabled and self.logger:
                self.logger.debug(f"Root search iteration: Move={move}, Score={current_move_score:.2f}, Best Move So Far={best_move}, Best Score={best_score_overall:.2f}")

        return best_move

    def _minimax_search(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool):
        """Minimax search with alpha-beta pruning. Returns score (float)."""
        if depth <= 0 or board.is_game_over():
            return self.scoring_calculator.evaluate_position(board)
        best_score = -float('inf') if maximizing_player else float('inf')
        legal_moves = list(board.legal_moves)
        #legal_moves = self.move_organizer.order_moves(board, legal_moves, depth-1)
        for move in legal_moves:
            temp_board = board.copy()
            temp_board.push(move)

            # Check for immediate checkmate or draw
            if temp_board.is_checkmate():
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Checkmate move found: {move} | FEN: {temp_board.fen()}")
                return 999999999 if maximizing_player else -999999999
            elif self._draw_search(temp_board, depth=1):
                continue

            # Recursive minimax call
            score = self._minimax_search(temp_board, depth - 1, alpha, beta, not maximizing_player)

            if maximizing_player:
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
            else:
                best_score = min(best_score, score)
                beta = min(beta, best_score)

            # Alpha-beta pruning
            if alpha >= beta:
                break

        return best_score

    def _negamax_search(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        """Negamax search with alpha-beta pruning and basic tactical extensions."""
        if depth <= 0 or board.is_game_over():
            return self.scoring_calculator.evaluate_position(board)
        # Internal node: explore moves
        best_score = -float('inf')
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            temp_board = board.copy()
            temp_board.push(move)
            checkmate_move = self._checkmate_search(temp_board, depth=1)
            if checkmate_move != chess.Move.null() and temp_board.is_legal(checkmate_move):
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Checkmate move found: {checkmate_move} | FEN: {temp_board.fen()}")
                return 999999999
            if self._draw_search(temp_board, depth=1):
                continue
            # Recursive negamax call with flipped perspectives
            score = -self._negamax_search(temp_board, depth-1, -beta, -alpha)
            self.nodes_searched += 1
            # Update best score
            if score > best_score:
                best_score = score
            # Alpha-beta pruning
            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Beta cutoff
        return best_score

    
    def _deep_search(self, board: chess.Board, depth: int, max_depth: int, current_depth: int = 1):
        """Iterative deepening search with time management and more aggressive tactics."""
        best_move_root = chess.Move.null() 
        best_score_root = -float('inf')
        # Define legal_moves at the beginning of the method for the current board state
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null(), -float('inf')  # No legal moves to search

        search_depth_limit = min(depth, max_depth)
        for iterative_depth in range(current_depth, search_depth_limit + 1):
            if self.time_manager.should_stop(self.depth if self.depth is not None else 1):
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Deepsearch stopped by time manager at depth {iterative_depth-1}.")
                break # Stop if time manager says so
            
            current_iter_legal_moves = list(board.legal_moves) 
            if not current_iter_legal_moves:
                break

            # Re-order moves at each iteration - prioritize captures, checks, and central pawn moves
            ordered_moves = self.move_organizer.order_moves(board, current_iter_legal_moves, depth=iterative_depth)

            local_best_move_at_depth = chess.Move.null()
            local_best_score_at_depth = -float('inf')

            # Use aspiration windows for deeper searches
            if iterative_depth >= 4 and best_score_root != -float('inf'):
                # Narrow alpha-beta window for more effective pruning
                window = 50.0 # Pawn = 100, so half pawn window
                alpha = best_score_root - window
                beta = best_score_root + window
            else:
                alpha = -float('inf')
                beta = float('inf')

            # Search with aspiration windows
            for move in ordered_moves:
                temp_board = board.copy()
                temp_board.push(move)
                checkmate_move = self._checkmate_search(temp_board, depth=1)
                if checkmate_move != chess.Move.null() and temp_board.is_legal(checkmate_move):
                    if self.monitoring_enabled and self.logger:
                        self.logger.info(f"Checkmate move found: {checkmate_move} | FEN: {temp_board.fen()}")
                    return checkmate_move, 999999999
                if self._draw_search(temp_board, depth=1):
                    continue
                # For potential checkmates, search deeper
                if temp_board.is_check():
                    # Search deeper when giving check to find potential checkmates
                    check_extension = 1
                    current_move_score = -self._negamax_search(temp_board, iterative_depth - 1 + check_extension, -beta, -alpha)
                else:
                    current_move_score = -self._negamax_search(temp_board, iterative_depth - 1, -beta, -alpha)
                
                self.nodes_searched += 1
                
                if current_move_score > local_best_score_at_depth:
                    local_best_score_at_depth = current_move_score
                    local_best_move_at_depth = move
                
                # If outside aspiration window, research with full window
                if current_move_score <= alpha or current_move_score >= beta:
                    alpha = -float('inf')
                    beta = float('inf')
                    current_move_score = -self._negamax_search(temp_board, iterative_depth - 1, -beta, -alpha)
                    
                    if current_move_score > local_best_score_at_depth:
                        local_best_score_at_depth = current_move_score
                        local_best_move_at_depth = move
                
                alpha = max(alpha, current_move_score)
                if alpha >= beta:
                    break
            
            # After each full depth iteration, update the overall best move
            if local_best_move_at_depth != chess.Move.null():
                best_move_root = local_best_move_at_depth
                best_score_root = local_best_score_at_depth
            
            # If checkmate or mate-in-N is found, stop early
            checkmate_bonus = self.scoring_calculator.rules.get('checkmate_bonus', 1000000.0)
            if abs(best_score_root) > checkmate_bonus / 2:
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Deepsearch found a potential checkmate at depth {iterative_depth}. Stopping early.")
                break

            if self.monitoring_enabled and self.logger:
                self.logger.debug(f"Deepsearch finished depth {iterative_depth}: Best move {best_move_root} with score {best_score_root:.2f}")

        return best_move_root, best_score_root
    
    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, maximizing_player: bool, current_ply: int = 0) -> float:
        """Quiescence search to handle tactical positions."""
        # Get a static evaluation first
        temp_board = board.copy()
        stand_pat_score = self.scoring_calculator.evaluate_position(temp_board)
        
        # Check for immediate game-ending conditions with huge scores
        if temp_board.is_checkmate():
            return -999999999 if maximizing_player else 999999999
        
        # Handle standing pat
        if maximizing_player:
            if stand_pat_score >= beta:
                return beta
            alpha = max(alpha, stand_pat_score)
        else:
            if stand_pat_score <= alpha:
                return alpha
            beta = min(beta, stand_pat_score)
            
        # Depth control
        max_q_depth = 5  # Increased from 3 for better tactical vision
        if current_ply >= self.depth + max_q_depth:
            return stand_pat_score
            
        # Consider check evasions and good captures only
        if temp_board.is_check():
            # Must consider all legal moves to escape check
            moves_to_consider = list(temp_board.legal_moves)
        else:
            # Only consider captures and promotions
            captures = []
            for move in temp_board.legal_moves:
                if temp_board.is_capture(move) or move.promotion or temp_board.gives_check(move):
                    captures.append(move)
                    
            moves_to_consider = captures
            
        # If no good tactical moves, return stand pat
        if not moves_to_consider:
            return stand_pat_score
            
        for move in moves_to_consider:
            temp_board.push(move)
            score = -self._quiescence_search(temp_board, -beta, -alpha, not maximizing_player, current_ply + 1)
            temp_board.pop()
            self.nodes_searched += 1
            
            if maximizing_player:
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
            else:
                if score < beta:
                    beta = score
                    if alpha >= beta:
                        break
                    
        return alpha if maximizing_player else beta
    
    def _checkmate_search(self, board: chess.Board, depth: int = 3, first_move: chess.Move = chess.Move.null()) -> chess.Move:
        """Identify checkmate patterns within the given depth (interpreted as ply)."""
        legal_moves = list(board.legal_moves)  # Use legal moves to avoid illegal checks
        if depth <= 0 or board.is_game_over() or not legal_moves:
            return chess.Move.null()  # No legal moves left or depth is zero

        for move in legal_moves:
            first_move = move if first_move == chess.Move.null() else first_move
            temp_board = board.copy()
            temp_board.push(move)
            if temp_board.is_checkmate():
                # Checkmate found
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Checkmate move found: {move} | FEN: {board.fen()}")
                return first_move  # Return the first move in the checkmate chain
            else:
                self._checkmate_search(temp_board,depth-1,first_move)
        return chess.Move.null()  # Return null move if no checkmate is found
    
    def _draw_search(self, board: chess.Board, depth: int = 3, first_move: chess.Move = chess.Move.null()) -> bool:
        """Identify draw patterns within the given depth (interpreted as ply)."""
        legal_moves = list(board.legal_moves)  # Use legal moves to avoid illegal checks
        if depth <= 0 or board.is_game_over() or not legal_moves:
            return True  # No pseudolegal moves left or depth is zero

        for move in legal_moves:
            first_move = move if first_move == chess.Move.null() else first_move
            temp_board = board.copy()
            temp_board.push(move)
            if (temp_board.is_stalemate()
                or temp_board.is_insufficient_material()
                or temp_board.can_claim_fifty_moves()
                or temp_board.can_claim_threefold_repetition()
                or temp_board.is_seventyfive_moves()
                or temp_board.is_fivefold_repetition()
                or temp_board.is_variant_draw()):
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Stalemate or draw condition met for move: {move} | FEN: {temp_board.fen()}")
                return True  # Return true if a draw condition is found
        return False  # Return false if no drawing moves are found
