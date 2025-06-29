# v7p3r_engine/search_algorithms.py
from math import e
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import chess
import random
import logging

# =======================================
# ======= MAIN SEARCH CLASS ========
class v7p3rSearch:
    def __init__(self, engine_config: dict, scoring_calculator, move_organizer, time_manager, opening_book, logger: logging.Logger):
        # Configuration
        self.engine_config = engine_config
        self.engine_search_algorithm = engine_config.get('engine_search_algorithm', 'simple_search')  # Default to simple_search if not set
        self.depth = engine_config.get('depth', 2)  # Placeholder for engine reference
        self.max_depth = engine_config.get('max_depth', 5)  # Max depth of search for AI, default to 8 if not set
        self.logger = logger if logger else logging.getLogger("v7p3r_engine_logger")
        self.nodes_searched = 0  # Initialize nodes searched counter
        
        # Required Search Modules
        self.scoring_calculator = scoring_calculator
        self.move_organizer = move_organizer
        self.time_manager = time_manager
        self.opening_book = opening_book

        # Engine Search Types (search algorithms used by v7p3r)
        self.search_algorithms = [
            'deepsearch',                    # Dynamic deep search via negamax with time control and iterative deepening
            'lookahead',                     # Lookahead search with simple max value comparison
            'minimax',                       # Minimax search with alpha-beta pruning
            'negamax',                       # Negamax search with alpha-beta pruning
            'simple_search',                 # Simple 1 ply search
            'random',                        # Random move selection
        ]
    
    def search(self, board: chess.Board, player: chess.Color):
        """Perform a search for the best move for the given player."""
        self.current_player = player
        self.board = board.copy()
        self.nodes_searched = 0  # Reset nodes searched for this search

        # Check for checkmate patterns
        # checkmate_move = self._checkmate_search(board.copy(), depth=self.depth, first_move=chess.Move.null())
        checkmate_move = chess.Move.null() # TODO fix or remove - overriding this function for now
        if checkmate_move != chess.Move.null() and self.board.is_legal(checkmate_move):
            if self.logger:
                self.logger.debug(f"Checkmate move found: {checkmate_move} | FEN: {board.fen()}")
            return checkmate_move
        
        # Check for book moves
        book_move = self.opening_book.get_book_move(board.copy())
        if book_move and self.board.is_legal(book_move):
            if self.logger:
                self.logger.debug(f"Opening book move found: {book_move} | FEN: {board.fen()}")
            return book_move
        
        # Start the search evaluation
        if self.logger:
            self.logger.debug(f"== EVALUATION (Player: {'White' if player == chess.WHITE else 'Black'}) == | Search Type: {self.engine_search_algorithm} | Depth: {self.depth} | Max Depth: {self.max_depth} == ")

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return chess.Move.null()

        ordered_moves = self.move_organizer.order_moves(board, legal_moves, depth=self.depth if self.depth is not None else 1)
        best_score_overall = -float('inf')
        best_move = ordered_moves[0] if ordered_moves else chess.Move.null()
        for move in ordered_moves:
            temp_board = self.board.copy()
            temp_board.push(move)
            current_move_score = 0.0

            # Check for immediate checkmate or stalemate
            if temp_board.is_checkmate():
                if self.logger:
                    self.logger.info(f"Checkmate move found: {move} | FEN: {temp_board.fen()}")
                return move
            elif (temp_board.is_stalemate()
                or temp_board.is_insufficient_material()
                or temp_board.can_claim_fifty_moves()
                or temp_board.can_claim_threefold_repetition()
                or temp_board.is_seventyfive_moves()
                or temp_board.is_fivefold_repetition()
                or temp_board.is_variant_draw()):
                if self.logger:
                    self.logger.info(f"Stalemate or draw condition met for move: {move} | FEN: {temp_board.fen()}")
                continue # Skip this move if it leads to stalemate or draw

            try:
                if self.engine_search_algorithm == 'deepsearch':
                    # Pass self.depth (from resolved config) to _deep_search
                    final_deepsearch_move_result = self._deep_search(self.board.copy(), self.depth if self.depth is not None else 1, self.max_depth)
                    if final_deepsearch_move_result != chess.Move.null():
                        best_move = final_deepsearch_move_result
                        if self.board.is_legal(best_move): # Check legality on original board
                            temp_board_after_move = self.board.copy()
                            temp_board_after_move.push(best_move)
                            # Evaluate from current player's perspective
                            current_move_score = self.scoring_calculator.evaluate_position(temp_board_after_move)
                        return best_move
                elif self.engine_search_algorithm == 'minimax':
                    current_move_score = self._minimax_search(temp_board, (self.depth - 1) if self.depth is not None else 0, -float('inf'), float('inf'), temp_board.turn != self.current_player)
                elif self.engine_search_algorithm == 'negamax':
                    current_move_score = -self._negamax_search(temp_board, (self.depth - 1) if self.depth is not None else 0, -float('inf'), float('inf'))
                elif self.engine_search_algorithm == 'lookahead':
                    current_move_score = -self._lookahead_search(temp_board, (self.depth - 1) if self.depth is not None else 0, -float('inf'), float('inf'))
                elif self.engine_search_algorithm == 'simple_search': # simple_search itself handles perspective
                    current_move_score = self.scoring_calculator.evaluate_position(temp_board)
                elif self.engine_search_algorithm == 'random':
                    best_move = self._random_search(self.board.copy())
                    return best_move # Random search returns a move directly
                else:
                    if self.logger:
                        self.logger.warning(f"Unrecognized AI type '{self.engine_search_algorithm}'. Falling back to evaluation_only for score.")
                    current_move_score = self.scoring_calculator.evaluate_position(temp_board)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in search algorithm '{self.engine_search_algorithm}' for move {move}: {e}. Using immediate evaluation. | FEN: {temp_board.fen()}")
                current_move_score = self.scoring_calculator.evaluate_position(temp_board)

            if current_move_score > best_score_overall:
                best_score_overall = current_move_score
                best_move = move
                if self.logger:
                    self.logger.debug(f"New best move: {move}, score: {current_move_score:.3f}")
            
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
        return best_move

    def _random_search(self, board: chess.Board) -> chess.Move:
        """Select a random legal move from the board."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            if self.logger:
                self.logger.debug(f"No legal moves available | FEN: {board.fen()}")
            return chess.Move.null() # Return null move if no legal moves
        move = random.choice(legal_moves)

        # Check for immediate checkmate or stalemate
        for mating_move in legal_moves:
            temp_board = board.copy()
            temp_board.push(mating_move)
            if temp_board.is_checkmate():
                if self.logger:
                    self.logger.info(f"Checkmate move found: {mating_move} | FEN: {temp_board.fen()}")
                return mating_move

        if self.logger:
            self.logger.debug(f"Randomly selected move: {move} | FEN: {board.fen()}")
        return move

    def _simple_search(self, board: chess.Board) -> chess.Move:
        """Simple search that evaluates all legal moves and picks the best one at 1 ply."""
        best_move = chess.Move.null()
        # Initialize best_score to negative infinity for white, positive infinity for black for proper min/max
        best_score = -float('inf') if board.turn else float('inf')
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null() # No legal moves, likely game over
        for move in legal_moves:
            temp_board = board.copy()
            temp_board.push(move)

            # Check for immediate checkmate or stalemate
            if temp_board.is_checkmate():
                if self.logger:
                    self.logger.info(f"Checkmate move found: {move} | FEN: {temp_board.fen()}")
                return move
            elif (temp_board.is_stalemate()
                or temp_board.is_insufficient_material()
                or temp_board.can_claim_fifty_moves()
                or temp_board.can_claim_threefold_repetition()
                or temp_board.is_seventyfive_moves()
                or temp_board.is_fivefold_repetition()
                or temp_board.is_variant_draw()):
                if self.logger:
                    self.logger.info(f"Stalemate or draw condition met for move: {move} | FEN: {temp_board.fen()}")
                continue # Skip this move if it leads to stalemate or draw

            score = self.scoring_calculator.evaluate_position(temp_board)
            self.nodes_searched += 1  # Increment nodes searched
            if self.logger:
                self.logger.debug(f"Simple search evaluating move: {move} | Score: {score:.3f} | Best score: {best_score:.3f} | FEN: {temp_board.fen()}")
            if temp_board.turn == chess.WHITE: # If white's turn, maximize score
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
        if depth == 0 or board.is_game_over():
            return self._quiescence_search(board, alpha, beta, True)
        best_score = -float('inf') if board.turn else float('inf')
        legal_moves = self.move_organizer.order_moves(board, list(board.legal_moves), depth=depth)
        for move in legal_moves:
            temp_board = board.copy()
            temp_board.push(move)
            
            # Check for immediate checkmate or stalemate
            if temp_board.is_checkmate():
                if self.logger:
                    self.logger.info(f"Checkmate move found: {move} | FEN: {temp_board.fen()}")
                return 999999999 # Return a large score for checkmate
            elif (temp_board.is_stalemate()
                or temp_board.is_insufficient_material()
                or temp_board.can_claim_fifty_moves()
                or temp_board.can_claim_threefold_repetition()
                or temp_board.is_seventyfive_moves()
                or temp_board.is_fivefold_repetition()
                or temp_board.is_variant_draw()):
                if self.logger:
                    self.logger.info(f"Stalemate or draw condition met for move: {move} | FEN: {temp_board.fen()}")
                continue # Skip this move if it leads to stalemate or draw

            score = -self._lookahead_search(temp_board, depth - 1, -beta, -alpha)
            self.nodes_searched += 1  # Increment nodes searched
            if score > best_score:
                best_score = score
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break # Prune remaining moves at this depth
        return best_score

    def _minimax_search(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool):
        """Minimax search with alpha-beta pruning. Returns score (float)."""
        if depth == 0 or board.is_game_over():
            return self._quiescence_search(board, alpha, beta, maximizing_player)
        best_score = -float('inf') if maximizing_player else float('inf')
        legal_moves = self.move_organizer.order_moves(board, list(board.legal_moves), depth=depth)
        for move in legal_moves:
            temp_board = board.copy()
            temp_board.push(move)

            # Check for immediate checkmate or stalemate
            if temp_board.is_checkmate():
                if self.logger:
                    self.logger.info(f"Checkmate move found: {move} | FEN: {temp_board.fen()}")
                return 999999999 # Return a large score for checkmate
            elif (temp_board.is_stalemate()
                or temp_board.is_insufficient_material()
                or temp_board.can_claim_fifty_moves()
                or temp_board.can_claim_threefold_repetition()
                or temp_board.is_seventyfive_moves()
                or temp_board.is_fivefold_repetition()
                or temp_board.is_variant_draw()):
                if self.logger:
                    self.logger.info(f"Stalemate or draw condition met for move: {move} | FEN: {temp_board.fen()}")
                continue # Skip this move if it leads to stalemate or draw

            score = self._minimax_search(temp_board, depth-1, alpha, beta, not maximizing_player)
            self.nodes_searched += 1  # Increment nodes searched
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
                    break # Alpha-beta cutoff
            else: # Minimizing player
                beta = min(beta, score)
                if alpha >= beta:
                    break # Alpha-beta cutoff
        return best_score

    def _negamax_search(self, board: chess.Board, depth: int, alpha: float, beta: float, ) -> float:
        """Negamax search with alpha-beta pruning and basic tactical extensions."""
        # Check for immediate game ending
        if board.is_checkmate():
            return -float('inf')  # Lost
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0  # Draw
            
        # Leaf node: go to quiescence search
        if depth <= 0:
            return self._quiescence_search(board, alpha, beta, True)
            
        # Internal node: explore moves
        best_score = -float('inf')
        
        # Tactical extension: search deeper if in check (to avoid horizon effect)
        if board.is_check():
            depth += 1
            
        # Move ordering is critical for alpha-beta efficiency
        legal_moves = self.move_organizer.order_moves(board, list(board.legal_moves), depth=depth)
        
        for move in legal_moves:
            temp_board = board.copy()
            temp_board.push(move)
            
            # Check for immediate checkmate or stalemate
            if temp_board.is_checkmate():
                if self.logger:
                    self.logger.info(f"Checkmate move found: {move} | FEN: {temp_board.fen()}")
                return 999999999 # Return a large score for checkmate
            elif (temp_board.is_stalemate()
                or temp_board.is_insufficient_material()
                or temp_board.can_claim_fifty_moves()
                or temp_board.can_claim_threefold_repetition()
                or temp_board.is_seventyfive_moves()
                or temp_board.is_fivefold_repetition()
                or temp_board.is_variant_draw()):
                if self.logger:
                    self.logger.info(f"Stalemate or draw condition met for move: {move} | FEN: {temp_board.fen()}")
                continue # Skip this move if it leads to stalemate or draw

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

    
    def _deep_search(self, board: chess.Board, depth: int, max_depth: int, current_depth: int = 1) -> chess.Move:
        """Iterative deepening search with time management and more aggressive tactics."""
        best_move_root = chess.Move.null() 
        best_score_root = -float('inf')
        # Define legal_moves at the beginning of the method for the current board state
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null() # No legal moves to search

        search_depth_limit = min(depth, max_depth)
        for iterative_depth in range(current_depth, search_depth_limit + 1):
            if self.time_manager.should_stop(self.depth if self.depth is not None else 1):
                if self.logger:
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
                
                # Check for immediate checkmate or stalemate
                if temp_board.is_checkmate():
                    if self.logger:
                        self.logger.info(f"Checkmate move found: {move} | FEN: {temp_board.fen()}")
                    return move
                elif (temp_board.is_stalemate()
                    or temp_board.is_insufficient_material()
                    or temp_board.can_claim_fifty_moves()
                    or temp_board.can_claim_threefold_repetition()
                    or temp_board.is_seventyfive_moves()
                    or temp_board.is_fivefold_repetition()
                    or temp_board.is_variant_draw()):
                    if self.logger:
                        self.logger.info(f"Stalemate or draw condition met for move: {move} | FEN: {temp_board.fen()}")
                    continue # Skip this move if it leads to stalemate or draw
                
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
                if self.logger:
                    self.logger.info(f"Deepsearch found a potential checkmate at depth {iterative_depth}. Stopping early.")
                break

            if self.logger:
                self.logger.debug(f"Deepsearch finished depth {iterative_depth}: Best move {best_move_root} with score {best_score_root:.2f}")

        return best_move_root if best_move_root != chess.Move.null() else self._simple_search(board)
       
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
        # legal_moves = list(board.pseudo_legal_moves) # Complete list of pseudolegal moves
        legal_moves = list(board.legal_moves)  # Use legal moves to avoid illegal checks
        if depth <= 0 or board.is_game_over() or not legal_moves:
            return chess.Move.null()  # No pseudolegal moves left or depth is zero

        for move in legal_moves:
            first_move = move if first_move == chess.Move.null() else first_move
            temp_board = board.copy()
            temp_board.push(move)
            if temp_board.is_checkmate():
                # Checkmate found
                if self.logger:
                    self.logger.info(f"Checkmate move found: {move} | FEN: {board.fen()}")
                return first_move  # Return the first move in the checkmate chain
            else:
                # Recursive check for further moves
                next_move = self._checkmate_search(temp_board, depth - 1, first_move)
                if self.logger:
                    self.logger.info(f"Checking next move: {next_move} | Depth: {depth - 1} | FEN: {temp_board.fen()}")

        if self.logger:
            self.logger.debug(f"No checkmate move found within depth {depth}.")
        return chess.Move.null()  # Return null move if no checkmate is found
