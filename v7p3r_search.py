# v7p3r_search.py

import sys
import os
import chess
import random
from v7p3r_config import v7p3rConfig
from v7p3r_debug import v7p3rLogger, v7p3rUtilities

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup centralized logging for this module
v7p3r_search_logger = v7p3rLogger.setup_logger("v7p3r_search")

class v7p3rSearch:
    def __init__(self, scoring_calculator, move_organizer, time_manager, opening_book):
        # Engine Configuration
        self.config_manager = v7p3rConfig()
        self.engine_config = self.config_manager.get_engine_config()  # Ensure it's always a dictionary

        # Logging Setup
        self.logger = v7p3r_search_logger
        self.monitoring_enabled = self.engine_config.get('monitoring_enabled', True)
        self.verbose_output_enabled = self.engine_config.get('verbose_output', False)

        # Required Search Modules
        self.scoring_calculator = scoring_calculator
        self.move_organizer = move_organizer
        self.time_manager = time_manager
        self.opening_book = opening_book

        # Search Setup
        self.search_algorithm = self.engine_config.get('search_algorithm', 'simple')
        self.depth = self.engine_config.get('depth', 3)
        self.max_depth = self.engine_config.get('max_depth', 5)
        self.strict_draw_prevention = self.engine_config.get('strict_draw_prevention', False)
        self.quiescence_enabled = self.engine_config.get('use_quiescence', True)
        self.move_ordering_enabled = self.engine_config.get('use_move_ordering', True)
        self.max_moves = self.engine_config.get('max_moves', 5)

        # Initialize search state
        self.root_board = chess.Board()
        self.nodes_searched = 0
        self.root_move = chess.Move.null()
        self.current_move = chess.Move.null()
        self.pv_move_stack = [{}]
        self.color = chess.WHITE
        self.color_name = 'White' if self.color == chess.WHITE else 'Black'
        self.current_perspective = self.color
        self.current_turn = self.root_board.turn
        self.evaluation = 0.0
        self.best_move = chess.Move.null()
        self.best_score = -float('inf')
        self.fen = self.root_board.fen()
        self.search_id_counter = 0  # Counter for generating unique search IDs
        self.search_id = f"search[{self.search_id_counter}]_{v7p3rUtilities.get_timestamp()}"  # Unique ID for each search instance

        # Initialize search dataset
        self.search_dataset = {
            'search_id': self.search_id,
            'search_algorithm': self.search_algorithm,
            'depth': self.depth,
            'max_depth': self.max_depth,
            'nodes_searched': 0,
            'best_move': chess.Move.null(),
            'best_score': -float('inf'),
            'pv_move_stack': [],
            'color_name': self.color_name,
            'fen': self.fen,
            'evaluation': 0.0,
            'root_board_fen': self.root_board.fen(),
        }

    def search(self, board: chess.Board, color: chess.Color):
        """Perform a search for the best move for the given player."""
        try:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"SEARCH START: search_algorithm={self.search_algorithm}, depth={self.depth}, "
                  f"scoring_calculator type={type(self.scoring_calculator)}, "
                  f"move_organizer type={type(self.move_organizer)}, "
                  f"time_manager type={type(self.time_manager)}, "
                  f"opening_book type={type(self.opening_book)}")

            self.root_board = board.copy()
            self.current_turn = board.turn
            self.current_perspective = color
            self.nodes_searched = 0  # Reset nodes searched for this search
            self.color_name = 'White' if color == chess.WHITE else 'Black'
            self.best_move = chess.Move.null()
            self.best_score = -float('inf')
            self.pv_move_stack = [{}]
            self.root_move = chess.Move.null()  # Reset root move for this search


            # Check for checkmates
            checkmate_move = self._checkmate_search(self.root_board, self.depth if self.depth >= 5 else 5)
            if checkmate_move != chess.Move.null() and board.is_legal(checkmate_move):
                self.pv_move_stack = [{ # Initialize principal variation with the checkmate move
                    'move_number': 1,
                    'move': checkmate_move,
                    'color': self.current_perspective,  # FIXED: Use consistent perspective
                    'evaluation': self.scoring_calculator.evaluate_position_from_perspective(self.root_board, self.current_perspective)
                    }]
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"SEARCH: Checkmate move found: {checkmate_move} | FEN: {board.fen()}")
                return checkmate_move
            
            # Check for book moves
            try:
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"SEARCH: About to call opening_book.get_book_move")
                book_move = self.opening_book.get_book_move(self.root_board)
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"SEARCH: get_book_move returned: {book_move}")
            except Exception as e:
                if self.monitoring_enabled and self.logger:
                    self.logger.error(f"SEARCH ERROR in get_book_move: {e}")
                book_move = None
                
            if book_move and self.root_board.is_legal(book_move):
                self.root_move = book_move  # Set the root move to the book move
                self.evaluation = self.scoring_calculator.evaluate_position_from_perspective(self.root_board, color)
                self.pv_move_stack = [{ # Initialize principal variation with the book move
                    'move_number': 1,
                    'move': book_move,
                    'color': self.current_perspective,  # FIXED: Use consistent perspective
                    'evaluation': self.scoring_calculator.evaluate_position_from_perspective(self.root_board, self.current_perspective)
                    }]
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Opening book move found: {book_move} | FEN: {self.root_board.fen()}")
                return book_move

            if self.monitoring_enabled and self.logger:
                self.logger.info(f"== EVALUATION (Player: {self.color_name}) == | Search Type: {self.search_algorithm} | Depth: {self.depth} | Max Depth: {self.max_depth} == ")

            legal_moves = list(self.root_board.legal_moves)
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"SEARCH: Found {len(legal_moves)} legal moves")
            
            if self.move_ordering_enabled:
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"SEARCH: Ordering moves for depth {self.depth}")
                legal_moves = self.move_organizer.order_moves(self.root_board, legal_moves, depth=self.depth, cutoff=0)
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"SEARCH: Ordered moves, first few: {legal_moves[:3] if len(legal_moves) >= 3 else legal_moves}")
            else:
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"SEARCH: Using unordered moves for depth {self.depth}")

            # if only one move returned,then instantly send through
            if len(legal_moves) == 1: 
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Only one legal move found: {legal_moves[0]} | FEN: {self.root_board.fen()}")
                return legal_moves[0]
                
            if not legal_moves:
                if self.monitoring_enabled and self.logger:
                    self.logger.error(f"[Error] No legal moves found for player: {self.color_name} | FEN: {self.root_board.fen()}")
                return chess.Move.null()

            best_score_overall = -float('inf')
            best_move = chess.Move.null()
            for move in legal_moves:
                self.root_move = move  # Set the root move for this search
                self.pv_move_stack = [{ # Initialize principal variation with the move
                    'move_number': 1,
                    'move': move,
                    'color': self.current_perspective,  # FIXED: Use consistent perspective
                    'evaluation': self.scoring_calculator.evaluate_position_from_perspective(self.root_board, self.current_perspective)
                    }]
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"SEARCH: Examining move {move} | FEN: {self.root_board.fen()}")
                    
                temp_board = self.root_board.copy()
                temp_board.push(move)
                current_move_score = 0.0

                # Check for immediate checkmate
                if temp_board.is_checkmate():
                    if self.monitoring_enabled and self.logger:
                        self.logger.info(f"Checkmate move found: {move} | FEN: {temp_board.fen()}")
                    return move

                try:
                    if self.monitoring_enabled and self.logger:
                        self.logger.info(f"SEARCH: About to run {self.search_algorithm} for move {move}")
                    
                    if self.search_algorithm == 'minimax':
                        if self.monitoring_enabled and self.logger:
                            self.logger.info(f"SEARCH: Calling _minimax_search")
                            self.logger.info(f"SEARCH: temp_board type={type(temp_board)}, depth={self.depth}")
                        current_move_score = self._minimax_search(temp_board, self.depth, -float('inf'), float('inf'), False)
                        if self.monitoring_enabled and self.logger:
                            self.logger.info(f"SEARCH: _minimax_search returned: {current_move_score}, type={type(current_move_score)}")
                    elif self.search_algorithm == 'negamax':
                        if self.monitoring_enabled and self.logger:
                            self.logger.info(f"SEARCH: Calling _negamax_search")
                        current_move_score = self._negamax_search(temp_board, self.depth, -float('inf'), float('inf'))
                    elif self.search_algorithm == 'simple':
                        if self.monitoring_enabled and self.logger:
                            self.logger.info(f"SEARCH: Calling scoring_calculator.evaluate_position_from_position")
                            self.logger.info(f"SEARCH: scoring_calculator={self.scoring_calculator}, type={type(self.scoring_calculator)}")
                        current_move_score = self.scoring_calculator.evaluate_position_from_perspective(temp_board, color)
                    elif self.search_algorithm == 'quiescence':
                        if self.monitoring_enabled and self.logger:
                            self.logger.info(f"SEARCH: Calling _quiescence_search")
                        current_move_score = self._quiescence_search(temp_board, color, -float('inf'), float('inf'), True)
                    elif self.search_algorithm == 'random':
                        if self.monitoring_enabled and self.logger:
                            self.logger.info(f"SEARCH: Calling random.uniform")
                        current_move_score = random.uniform(-1.0, 1.0)
                    else:
                        raise ValueError(f"Unknown search algorithm: {self.search_algorithm}")
                        
                    if self.monitoring_enabled and self.logger:
                        self.logger.info(f"SEARCH: {self.search_algorithm} returned score: {current_move_score}")
                except Exception as e:
                    if self.monitoring_enabled and self.logger:
                        self.logger.error(f"Error in search algorithm '{self.search_algorithm}' for move {move}: {e}. | FEN: {temp_board.fen()}")
                        self.logger.error(f"Exception type: {type(e)}")
                        import traceback
                        self.logger.error(f"Traceback: {traceback.format_exc()}")
                if current_move_score > best_score_overall:
                    best_score_overall = current_move_score
                    best_move = move
                    if self.monitoring_enabled and self.logger:
                        self.logger.info(f"New best move: {move}, score: {current_move_score:.3f}")

                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"Root search iteration: Move={move}, Score={current_move_score:.2f}, Best Move So Far={best_move}, Best Score={best_score_overall:.2f}")

            return best_move
            
        except Exception as e:
            if self.monitoring_enabled and self.logger:
                self.logger.error(f"CRITICAL ERROR in search method: {e}")
                self.logger.error(f"Exception type: {type(e)}")
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                self.logger.error(f"Board FEN: {board.fen()}")
                self.logger.error(f"Player: {self.color_name}")
                self.logger.error(f"Search algorithm: {self.search_algorithm}")
            
            # Return a random legal move as fallback
            legal_moves = list(board.legal_moves)
            if legal_moves:
                fallback_move = random.choice(legal_moves)
                if self.monitoring_enabled and self.logger:
                    self.logger.error(f"Returning fallback move: {fallback_move}")
                return fallback_move
            return chess.Move.null()

    def _minimax_search(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool):
        """Minimax search with alpha-beta pruning. Returns score (float)."""
        if depth <= 0 or board.is_game_over():
            # Always evaluate from the current perspective
            eval_result = self.scoring_calculator.evaluate_position_from_perspective(board, self.current_perspective)
            
            # For minimax, if we're minimizing (opponent's turn), we need to negate the score
            # because minimax alternates between maximizing and minimizing players
            # The evaluation is already from our perspective, so we only negate for minimizing player
            if not maximizing_player:
                eval_result = -eval_result
                
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"MINIMAX: Evaluation from {self.color_name}'s perspective: {eval_result} | FEN: {board.fen()}")
            
            return eval_result
        
        # Internal node: explore moves
        best_score = -float('inf') if maximizing_player else float('inf')
        
        legal_moves = list(board.legal_moves)
        if self.monitoring_enabled and self.logger:
            self.logger.info(f"MINIMAX: Found {len(legal_moves)} legal moves")
        
        if self.move_ordering_enabled:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"MINIMAX: Ordering moves for depth {depth}")
            # Order moves for better alpha-beta pruning efficiency
            legal_moves = self.move_organizer.order_moves(board, legal_moves, depth=self.depth, cutoff=self.max_moves)
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"MINIMAX: Received {len(legal_moves)} ordered moves, first few: {legal_moves[:3] if len(legal_moves) >= 3 else legal_moves}")
        else:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"MINIMAX: Using unordered moves for depth {depth}")
        
        for move in legal_moves:
            temp_board = board.copy()
            temp_board.push(move)

            # Recursive minimax call
            score = self._minimax_search(temp_board, depth - 1, alpha, beta, not maximizing_player)
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"MINIMAX: Recursive call returned: {score} (type: {type(score)})")
            if not isinstance(score, (int, float)):
                if self.monitoring_enabled and self.logger:
                    self.logger.error(f"ERROR: Recursive minimax returned non-numeric value: {score} (type: {type(score)})")
                score = 0.0

            if maximizing_player:
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
            else:
                best_score = min(best_score, score)
                beta = min(beta, best_score)

            # Alpha-beta pruning
            if alpha >= beta:
                break

        if self.monitoring_enabled and self.logger:
            self.logger.info(f"MINIMAX: Returning best_score: {best_score} (type: {type(best_score)})")
        return best_score

    def _negamax_search(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        """Negamax search with alpha-beta pruning and basic tactical extensions."""
        if self.monitoring_enabled and self.logger:
            self.logger.info(f"NEGAMAX: Starting search at depth {depth}, alpha={alpha:.2f}, beta={beta:.2f}")
        
        if depth <= 0 or board.is_game_over():
            # Always evaluate from the current perspective
            eval_result = self.scoring_calculator.evaluate_position_from_perspective(board, self.current_perspective)
            
            # In negamax, we need to flip the sign if it's not the current player's turn
            # This ensures scores are always from the perspective of the player to move
            if board.turn != self.current_perspective:
                eval_result = -eval_result
                
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"NEGAMAX: Leaf node evaluation from {self.color_name}'s perspective: {eval_result:.2f} | FEN: {board.fen()}")
            
            return eval_result
            
        # Internal node: explore moves
        best_score = -float('inf')
        best_move = chess.Move.null()
        legal_moves = list(board.legal_moves)
        if self.monitoring_enabled and self.logger:
            self.logger.info(f"NEGAMAX: Found {len(legal_moves)} legal moves at depth {depth}")
        
        if self.move_ordering_enabled:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"NEGAMAX: Ordering moves for depth {depth}")
            # Order moves for better alpha-beta pruning efficiency   
            legal_moves = self.move_organizer.order_moves(board, legal_moves, depth=self.depth, cutoff=self.max_moves)
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"NEGAMAX: Received {len(legal_moves)} ordered moves, first few: {legal_moves[:3] if len(legal_moves) >= 3 else legal_moves}")
        else:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"NEGAMAX: Using unordered moves for depth {depth}")
            legal_moves = list(board.legal_moves)
            
        moves_evaluated = 0
        for move in legal_moves:
            moves_evaluated += 1
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"NEGAMAX: Evaluating move {moves_evaluated}/{len(legal_moves)}: {move}")
                
            temp_board = board.copy()
            temp_board.push(move)
                
            # Recursive negamax call with flipped perspectives
            score = -self._negamax_search(temp_board, depth-1, -beta, -alpha)
            self.nodes_searched += 1
            
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"NEGAMAX: Move {move} returned score: {score:.2f}")
            
            # Update best score
            if score > best_score:
                best_score = score
                best_move = move
                self.pv_move_stack.append({ # Update principal variation
                    'move_number': depth - self.depth + 1,
                    'move': best_move,
                    'score': best_score,
                    'color': self.current_perspective,  # FIXED: Use consistent perspective
                    'evaluation': self.scoring_calculator.evaluate_position_from_perspective(self.root_board, self.current_perspective)
                    })
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"NEGAMAX: New best score: {best_score:.2f}")
                    
            # Alpha-beta pruning
            alpha = max(alpha, score)
            if alpha >= beta:
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"NEGAMAX: Beta cutoff (alpha={alpha:.2f} >= beta={beta:.2f}) after {moves_evaluated} moves")
                break  # Beta cutoff
            if beta < best_score:
                beta = best_score
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"NEGAMAX: Updated beta to {beta:.2f} after move {move}")

        if self.monitoring_enabled and self.logger:
            self.logger.info(f"NEGAMAX: Returning best score: {best_score:.2f} after evaluating {moves_evaluated} moves at depth {depth}")
        return best_score

    def _quiescence_search(self, board: chess.Board, color: chess.Color, alpha: float, beta: float, maximizing_player: bool, current_ply: int = 0) -> float:
        """Quiescence search to handle tactical positions."""
        if self.monitoring_enabled and self.logger:
            self.logger.info(f"QUIESCENCE: Starting search at ply {current_ply}, alpha={alpha:.2f}, beta={beta:.2f}, maximizing={maximizing_player}")
        
        # Get a static evaluation first, always from the consistent perspective
        temp_board = board.copy()
        stand_pat_score = self.scoring_calculator.evaluate_position_from_perspective(temp_board, self.current_perspective)
        
        if self.monitoring_enabled and self.logger:
            self.logger.info(f"QUIESCENCE: Stand pat score: {stand_pat_score:.2f} (type: {type(stand_pat_score)})")
        
        # Check for immediate game-ending conditions with huge scores
        if temp_board.is_checkmate():
            # Determine if the current perspective is being checkmated
            if temp_board.turn == self.current_perspective:
                # Current perspective is checkmated - very bad
                checkmate_score = -999999999
            else:
                # Opponent is checkmated - very good
                checkmate_score = 999999999
                
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"QUIESCENCE: Checkmate detected, returning: {checkmate_score} from {self.color_name}'s perspective")
            return checkmate_score
        
        # Handle standing pat
        if maximizing_player:
            if stand_pat_score >= beta:
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"QUIESCENCE: Beta cutoff on stand pat ({stand_pat_score:.2f} >= {beta:.2f})")
                return beta
            alpha = max(alpha, stand_pat_score)
        else:
            if stand_pat_score <= alpha:
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"QUIESCENCE: Alpha cutoff on stand pat ({stand_pat_score:.2f} <= {alpha:.2f})")
                return alpha
            beta = min(beta, stand_pat_score)
            
        # Depth control - strict limit to prevent infinite recursion
        max_q_depth = 2  # Hard limit of 2 plies for quiescence search
        if current_ply >= max_q_depth:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"QUIESCENCE: Max depth reached (ply {current_ply} >= {max_q_depth}), returning stand pat: {stand_pat_score:.2f}")
            return stand_pat_score
            
        # Consider check evasions and good captures only
        if temp_board.is_check():
            # Must consider all legal moves to escape check
            moves_to_consider = list(temp_board.legal_moves)
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"QUIESCENCE: In check, considering all {len(moves_to_consider)} legal moves")
        else:
            # Only consider captures and promotions
            captures = []
            for move in temp_board.legal_moves:
                if temp_board.is_capture(move) or move.promotion or temp_board.gives_check(move):
                    captures.append(move)
                    
            moves_to_consider = captures
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"QUIESCENCE: Not in check, considering {len(moves_to_consider)} tactical moves (captures/promotions/checks)")
            
        # If no good tactical moves, return stand pat
        if not moves_to_consider:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"QUIESCENCE: No tactical moves available, returning stand pat: {stand_pat_score:.2f}")
            return stand_pat_score
        
        best_score = alpha if maximizing_player else beta
        moves_evaluated = 0
        
        for move in moves_to_consider:
            moves_evaluated += 1
            if self.monitoring_enabled and self.logger:
                move_type = "capture" if temp_board.is_capture(move) else "promotion" if move.promotion else "check"
                self.logger.info(f"QUIESCENCE: Evaluating move {moves_evaluated}/{len(moves_to_consider)}: {move} ({move_type})")
            
            temp_board.push(move)
            score = -self._quiescence_search(temp_board, color, -beta, -alpha, not maximizing_player, current_ply + 1)
            temp_board.pop()
            self.nodes_searched += 1
            
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"QUIESCENCE: Move {move} returned score: {score:.2f}")
            
            if maximizing_player:
                if score > best_score:
                    best_score = score
                    if self.monitoring_enabled and self.logger:
                        self.logger.info(f"QUIESCENCE: New best score for maximizing: {best_score:.2f}")
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        if self.monitoring_enabled and self.logger:
                            self.logger.info(f"QUIESCENCE: Beta cutoff (alpha={alpha:.2f} >= beta={beta:.2f})")
                        break
            else:
                if score < best_score:
                    best_score = score
                    if self.monitoring_enabled and self.logger:
                        self.logger.info(f"QUIESCENCE: New best score for minimizing: {best_score:.2f}")
                if score < beta:
                    beta = score
                    if alpha >= beta:
                        if self.monitoring_enabled and self.logger:
                            self.logger.info(f"QUIESCENCE: Alpha cutoff (alpha={alpha:.2f} >= beta={beta:.2f})")
                        break
        
        final_score = alpha if maximizing_player else beta
        if self.monitoring_enabled and self.logger:
            self.logger.info(f"QUIESCENCE: Returning final score: {final_score:.2f} after evaluating {moves_evaluated} moves at ply {current_ply}")
        return final_score

    def _checkmate_search(self, board: chess.Board, ply: int = 5, first_move: chess.Move = chess.Move.null(), potential_checkmate_moves: list[chess.Move] = []) -> chess.Move:
        """Identify forced checkmate patterns within the given depth if greater than 5 otherwise (interpreted as 5 ply search)."""
        # Base cases
        if ply <= 0:
            return chess.Move.null()
        
        if board.is_game_over():
            if board.is_checkmate():
                # Found checkmate, return the first move that started this sequence
                return first_move if first_move != chess.Move.null() else chess.Move.null()
            else:
                # Game over but not checkmate (stalemate, etc.)
                return chess.Move.null()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        # Determine if we're looking for attacking moves (odd ply) or defensive moves (even ply)
        # In mate search, we want: attacker move -> defender move -> attacker move -> etc.
        is_attacking_turn = (ply % 2 == 1)  # Odd ply = attacker's turn, even ply = defender's turn
        
        if is_attacking_turn:
            # Attacker's turn: look for forcing moves (checks, captures, threats)
            # Prioritize checking moves and captures
            checking_moves = []
            other_moves = []
            
            for move in legal_moves:
                temp_board = board.copy()
                temp_board.push(move)
                if temp_board.is_check():
                    checking_moves.append(move)
                else:
                    other_moves.append(move)
            
            # Try checking moves first, then other moves
            moves_to_try = checking_moves + other_moves
            
            for move in moves_to_try:
                # Set first_move if this is the start of the sequence
                current_first_move = move if first_move == chess.Move.null() else first_move
                
                temp_board = board.copy()
                temp_board.push(move)
                
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"MATE SEARCH: Attacker trying move {move} at ply {ply} | FEN: {temp_board.fen()}")
                
                # Check if this move immediately gives checkmate
                if temp_board.is_checkmate():
                    if self.monitoring_enabled and self.logger:
                        self.logger.info(f"MATE SEARCH: Immediate checkmate found with move {move} | First move: {current_first_move}")
                    return current_first_move
                
                # If this move gives check, continue searching for forced mate
                if temp_board.is_check():
                    # Recursively search from defender's perspective
                    mate_move = self._checkmate_search(temp_board, ply - 1, current_first_move, potential_checkmate_moves)
                    if mate_move != chess.Move.null():
                        if self.monitoring_enabled and self.logger:
                            self.logger.info(f"MATE SEARCH: Forced mate found starting with {current_first_move} after move {move}")
                        return current_first_move
                
                # For non-checking moves, only continue if we're at high depth (looking for quiet setups)
                elif ply > 3:
                    mate_move = self._checkmate_search(temp_board, ply - 1, current_first_move, potential_checkmate_moves)
                    if mate_move != chess.Move.null():
                        return current_first_move
        
        else:
            # Defender's turn: try all legal moves to see if any can escape mate
            # If ALL moves lead to mate, then we have a forced mate
            defender_can_escape = False
            
            for move in legal_moves:
                temp_board = board.copy()
                temp_board.push(move)
                
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"MATE SEARCH: Defender trying escape move {move} at ply {ply} | FEN: {temp_board.fen()}")
                
                # Check if this defensive move escapes mate
                mate_move = self._checkmate_search(temp_board, ply - 1, first_move, potential_checkmate_moves)
                
                if mate_move == chess.Move.null():
                    # Defender found an escape
                    defender_can_escape = True
                    if self.monitoring_enabled and self.logger:
                        self.logger.info(f"MATE SEARCH: Defender can escape with move {move}")
                    break
            
            # If defender cannot escape any move, mate is forced
            if not defender_can_escape:
                if self.monitoring_enabled and self.logger:
                    self.logger.info(f"MATE SEARCH: Defender has no escape, mate is forced with first move {first_move}")
                return first_move
        
        return chess.Move.null()  # No forced mate found

    def _draw_search(self, board: chess.Board, first_move: chess.Move = chess.Move.null()) -> bool:
        """Identify draw patterns within the given depth (interpreted as ply)."""
        legal_moves = list(board.legal_moves)  # Use legal moves to avoid illegal checks
        if board.is_game_over() or not legal_moves:
            return True  # Game over or no legal moves

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
                    self.logger.info(f"Stalemate or draw condition met for move: {first_move} | FEN: {temp_board.fen()}")
                return True  # Return true if a draw condition is found
        return False  # Return false if no drawing moves are found
