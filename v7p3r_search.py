# v7p3r_search.py

import sys
import os
import chess
import random
from v7p3r_config import v7p3rConfig

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class v7p3rSearch:
    def __init__(self, scoring_calculator, move_organizer, time_manager, opening_book, engine_config=None):
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

        # Search Setup
        self.search_algorithm = self.engine_config.get('search_algorithm', 'simple')
        self.depth = self.engine_config.get('depth', 3)
        self.max_depth = self.engine_config.get('max_depth', 5)
        self.strict_draw_prevention = self.engine_config.get('strict_draw_prevention', False)
        self.quiescence_enabled = self.engine_config.get('use_quiescence', True)
        self.move_ordering_enabled = self.engine_config.get('use_move_ordering', True)
        self.max_ordered_moves = self.engine_config.get('max_ordered_moves', 5)

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
        """Search handler: delegates to the selected search algorithm, which is responsible for root move selection."""
        try:
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
                self.pv_move_stack = [{
                    'move_number': 1,
                    'move': checkmate_move,
                    'color': self.current_perspective,
                    'evaluation': self.scoring_calculator.evaluate_position_from_perspective(self.root_board, self.current_perspective)
                }]
                return checkmate_move

            # Check for book moves
            try:
                book_move = self.opening_book.get_book_move(self.root_board)
            except Exception as e:
                book_move = None

            if book_move and self.root_board.is_legal(book_move):
                self.root_move = book_move
                self.evaluation = self.scoring_calculator.evaluate_position_from_perspective(self.root_board, color)
                self.pv_move_stack = [{
                    'move_number': 1,
                    'move': book_move,
                    'color': self.current_perspective,
                    'evaluation': self.scoring_calculator.evaluate_position_from_perspective(self.root_board, self.current_perspective)
                }]
                return book_move

            # Delegate to the selected search algorithm, which is responsible for root move selection
            if self.search_algorithm == 'minimax':
                move, score = self._minimax_root(self.root_board, self.depth, color)
            elif self.search_algorithm == 'negamax':
                move, score = self._negamax_root(self.root_board, self.depth, color)
            elif self.search_algorithm == 'simple':
                move = self._simple_search(self.root_board)
                score = self.scoring_calculator.evaluate_position_from_perspective(self.root_board, color)
            elif self.search_algorithm == 'quiescence':
                move = self._quiescence_root(self.root_board, color)
                score = self.scoring_calculator.evaluate_position_from_perspective(self.root_board, color)
            elif self.search_algorithm == 'random':
                move = self._random_search(self.root_board)
                score = self.scoring_calculator.evaluate_position_from_perspective(self.root_board, color)
            else:
                move = chess.Move.null()
                score = 0.0

            # Update search state
            self.search_dataset['score'] = score
            self.search_dataset['best_move'] = move

            return move

        except Exception as e:
            raise

    def _minimax_root(self, board: chess.Board, depth: int, color: chess.Color):
        """Root node for minimax: iterates over legal moves and calls minimax recursively."""
        best_score = -float('inf')
        best_move = chess.Move.null()
        maximizing = True if color == chess.WHITE else False
        legal_moves = list(board.legal_moves)
        if self.move_ordering_enabled:
            legal_moves = self.move_organizer.order_moves(board, legal_moves, depth=depth, cutoff=0)
        for move in legal_moves:
            temp_board = board.copy()
            temp_board.push(move)
            score = self._minimax_search(temp_board, depth - 1, -float('inf'), float('inf'), not maximizing)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, best_score

    def _negamax_root(self, board: chess.Board, depth: int, color: chess.Color):
        """Root node for negamax: iterates over legal moves and calls negamax recursively."""
        best_score = -float('inf')
        best_move = chess.Move.null()
        legal_moves = list(board.legal_moves)
        if self.move_ordering_enabled:
            legal_moves = self.move_organizer.order_moves(board, legal_moves, depth=depth, cutoff=0)
        for move in legal_moves:
            temp_board = board.copy()
            temp_board.push(move)
            score = -self._negamax_search(temp_board, depth - 1, -float('inf'), float('inf'))
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, best_score

    def _simple_search(self, board: chess.Board):
        """Simple search: returns the first legal move."""
        legal_moves = list(board.legal_moves)
        return legal_moves[0] if legal_moves else chess.Move.null()

    def _random_search(self, board: chess.Board):
        """Random search: returns a random legal move."""
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves) if legal_moves else chess.Move.null()

    def _quiescence_root(self, board: chess.Board, color: chess.Color):
        """Stub for quiescence root (implement as needed)."""
        # For now, just return a random move
        return self._random_search(board)

    def _minimax_search(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool):
        """Minimax search with alpha-beta pruning. Returns score (float)."""
        self.nodes_searched += 1
        
        # Terminal condition - reached max depth or game over
        if depth <= 0 or board.is_game_over():
            # Always evaluate from the current perspective
            eval_result = self.scoring_calculator.evaluate_position_from_perspective(board, self.current_perspective)
            
            # For minimax, if we're minimizing (opponent's turn), we need to negate the score
            # because minimax alternates between maximizing and minimizing players
            # The evaluation is already from our perspective, so we only negate for minimizing player
            if not maximizing_player:
                eval_result = -eval_result
                
            # For positions that aren't quiet, use quiescence search if enabled
            if depth <= 0 and self.quiescence_enabled and (board.is_check() or self._position_is_tactical(board)):
                q_score = self._quiescence_search(board, self.current_perspective, alpha, beta, maximizing_player)
                return q_score
                
            return eval_result
        
        # Internal node: explore moves
        best_score = -float('inf') if maximizing_player else float('inf')
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            # No legal moves, must be checkmate or stalemate
            if board.is_check():
                return -999999999 if maximizing_player else 999999999  # Checkmate
            else:
                return 0  # Stalemate
                
        # Determine appropriate depth cutoff based on current depth
        cutoff = min(max(2, self.max_ordered_moves // (self.depth - depth + 1)), self.max_ordered_moves) if self.move_ordering_enabled else 0
        
        if self.move_ordering_enabled:
            # Order moves for better alpha-beta pruning efficiency
            ordered_moves = self.move_organizer.order_moves(board, legal_moves, depth=depth, cutoff=cutoff)
        else:
            ordered_moves = legal_moves
            
        for move in ordered_moves:
            board.push(move)
            
            # Recursive minimax call
            score = self._minimax_search(board, depth - 1, alpha, beta, not maximizing_player)
            board.pop()
            
            if not isinstance(score, (int, float)):
                score = 0.0

            if maximizing_player:
                if score > best_score:
                    best_score = score
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                beta = min(beta, best_score)

            # Alpha-beta pruning
            if alpha >= beta:
                break

        return best_score
        
    def _position_is_tactical(self, board: chess.Board) -> bool:
        """Determine if the position is tactical (captures available)"""
        for move in board.legal_moves:
            if board.is_capture(move) or move.promotion:
                return True
        return False

    def _negamax_search(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        """Negamax search with alpha-beta pruning and basic tactical extensions."""
        self.nodes_searched += 1
        
        # Terminal condition - reached max depth or game over
        if depth <= 0 or board.is_game_over():
            # Always evaluate from the current perspective
            eval_result = self.scoring_calculator.evaluate_position_from_perspective(board, self.current_perspective)
            
            # In negamax, we need to flip the sign if it's not the current player's turn
            # This ensures scores are always from the perspective of the player to move
            if board.turn != self.current_perspective:
                eval_result = -eval_result

            # For positions that aren't quiet, use quiescence search if enabled
            if depth <= 0 and self.quiescence_enabled and (board.is_check() or self._position_is_tactical(board)):
                q_score = self._quiescence_search(board, self.current_perspective, alpha, beta, board.turn == self.current_perspective)
                return q_score
            
            return eval_result
            
        # Internal node: explore moves
        best_score = -float('inf')
        best_move = chess.Move.null()
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            # No legal moves, must be checkmate or stalemate
            if board.is_check():
                return -999999999  # Checkmate
            else:
                return 0  # Stalemate
                
        # Determine appropriate depth cutoff based on current depth
        cutoff = min(max(2, self.max_ordered_moves // (self.depth - depth + 1)), self.max_ordered_moves) if self.move_ordering_enabled else 0
        
        if self.move_ordering_enabled:
            # Order moves for better alpha-beta pruning efficiency
            ordered_moves = self.move_organizer.order_moves(board, legal_moves, depth=depth, cutoff=cutoff)
        else:
            ordered_moves = legal_moves
            
        moves_evaluated = 0
        for move in ordered_moves:
            moves_evaluated += 1
                
            board.push(move)
            # Recursive negamax call with flipped perspectives
            score = -self._negamax_search(board, depth-1, -beta, -alpha)
            board.pop()
            
            # Update best score
            if score > best_score:
                best_score = score
                best_move = move
                if depth == self.depth:  # At root, update principal variation
                    self.pv_move_stack.append({
                        'move_number': depth,
                        'move': best_move,
                        'score': best_score,
                        'color': self.current_perspective
                    })
                    
            # Alpha-beta pruning
            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Beta cutoff

        return best_score

    def _quiescence_search(self, board: chess.Board, color: chess.Color, alpha: float, beta: float, maximizing_player: bool, current_ply: int = 0) -> float:
        """Quiescence search to handle tactical positions."""
        # Safety check to prevent infinite recursion
        # HARD LIMIT to prevent infinite recursion - early exit if we're too deep
        max_q_depth = 2  # Reduced hard limit for quiescence search
        if current_ply >= max_q_depth or self.nodes_searched > 10000:
            eval_result = self.scoring_calculator.evaluate_position_from_perspective(board, self.current_perspective)
            # Adjust evaluation for the player to move
            if not maximizing_player:
                eval_result = -eval_result
            return eval_result
            
        # Get a static evaluation first, always from the consistent perspective
        stand_pat_score = self.scoring_calculator.evaluate_position_from_perspective(board, self.current_perspective)
        # Adjust evaluation for the player to move
        if not maximizing_player:
            stand_pat_score = -stand_pat_score
            
        # Check for immediate game-ending conditions with huge scores
        if board.is_checkmate():
            # Determine if the current perspective is being checkmated
            if board.turn == self.current_perspective:
                # Current perspective is checkmated - very bad
                checkmate_score = -999999999
            else:
                # Opponent is checkmated - very good
                checkmate_score = 999999999
                
            return checkmate_score
        
        # Handle standing pat
        if maximizing_player:
            if stand_pat_score >= beta:
                return beta
            alpha = max(alpha, stand_pat_score)
        else:
            if stand_pat_score <= alpha:
                return alpha
            beta = min(beta, stand_pat_score)
            
        # Consider check evasions and good captures only
        if board.is_check():
            # Must consider all legal moves to escape check
            moves_to_consider = list(board.legal_moves)
        else:
            # Only consider captures and promotions
            captures = []
            for move in board.legal_moves:
                if board.is_capture(move) or move.promotion:
                    captures.append(move)
                    
            moves_to_consider = captures
            
        # If no good tactical moves, return stand pat
        if not moves_to_consider:
            return stand_pat_score
        
        # Limit number of moves to examine in quiescence to prevent explosion
        if len(moves_to_consider) > 5:
            moves_to_consider = self.move_organizer.order_moves(board, moves_to_consider, depth=0, cutoff=5)
        
        # For very limited quiescence search to prevent infinite loops
        # Only explore the top 3 captures maximum
        if current_ply > 0 and len(moves_to_consider) > 3:
            moves_to_consider = moves_to_consider[:3]
            
        for move in moves_to_consider:
            board.push(move)
            # For negamax-style quiescence search, we swap the perspective and negate the score
            score = -self._quiescence_search(board, color, -beta, -alpha, not maximizing_player, current_ply + 1)
            board.pop()
            self.nodes_searched += 1
            
            # In quiescence search when maximizing, we want higher scores
            if maximizing_player:
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
            # When minimizing, we want lower scores
            else:
                if score < beta:
                    beta = score
                    if alpha >= beta:
                        break
        
        return alpha if maximizing_player else beta

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
                
                # Check if this move immediately gives checkmate
                if temp_board.is_checkmate():
                    return current_first_move
                
                # If this move gives check, continue searching for forced mate
                if temp_board.is_check():
                    # Recursively search from defender's perspective
                    mate_move = self._checkmate_search(temp_board, ply - 1, current_first_move, potential_checkmate_moves)
                    if mate_move != chess.Move.null():
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
                
                # Check if this defensive move escapes mate
                mate_move = self._checkmate_search(temp_board, ply - 1, first_move, potential_checkmate_moves)
                
                if mate_move == chess.Move.null():
                    # Defender found an escape
                    defender_can_escape = True
                    break
            
            # If defender cannot escape any move, mate is forced
            if not defender_can_escape:
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
                return True  # Return true if a draw condition is found
        return False  # Return false if no drawing moves are found
