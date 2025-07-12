# v7p3r_search.py

"""v7p3r Search Module
This module handles the main search functionality for the v7p3r chess engine.
It contains the search handler and core search algorithms."""

import sys
import os
import chess
from typing import List, Optional, Tuple
from v7p3r_config import v7p3rConfig
from v7p3r_ordering import v7p3rOrdering
from v7p3r_mvv_lva import v7p3rMVVLVA
from v7p3r_book import v7p3rBook

# Ensure the parent directory is in sys.path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class v7p3rSearch:
    """The V7P3R chess engine search module."""
    def __init__(self, scoring_calculator=None, time_manager=None):
        # Engine Configuration
        self.config_manager = v7p3rConfig()
        self.engine_config = self.config_manager.get_engine_config()
        
        # Required Engine Modules
        self.scoring_calculator = scoring_calculator
        self.time_manager = time_manager
        self.opening_book = v7p3rBook()
        
        # Move ordering and tempo settings
        self.ordering = v7p3rOrdering(self.scoring_calculator)  # Changed from move_ordering to ordering
        self.move_ordering_enabled = True  # Enable move ordering by default
        self.tempo = self.scoring_calculator.tempo if self.scoring_calculator else None
        
        # Search Settings
        self.max_depth = self.engine_config.get('max_depth', 4)
        self.quiescence_depth = self.engine_config.get('quiescence_depth', 4)
        self.max_quiescence_depth = self.engine_config.get('max_quiescence_depth', 8)
        self.search_time_limit = self.engine_config.get('search_time_limit', 5.0)
        self.max_ordered_moves = self.engine_config.get('max_ordered_moves', 10)
        
        # Initialize killer moves and history tables
        self.killer_moves: List[Optional[chess.Move]] = [None] * 10  # Store killer moves at each depth
        self.history_table = {}  # Store history heuristic scores

        # Search Configuration
        self.depth = self.engine_config.get('depth', 3)  # Set default depth
        self.max_depth = self.engine_config.get('max_depth', 5)  # Set default max depth
        self.use_iterative_deepening = self.engine_config.get('use_iterative_deepening', True)
        self.use_quiescence = self.engine_config.get('use_quiescence', True)
        
        # PV tracking
        self.pv_move_stack: List[chess.Move] = []  # Principal variation moves
        
        # Score tracking
        self.best_move = None
        self.best_score = float('-inf')

    def search(self, board: chess.Board, color: chess.Color) -> Optional[chess.Move]:
        """Main search function that combines all search strategies with tempo awareness."""
        # 1. Check book moves first
        if not self.engine_config.get('skip_book', False):
            book_move = self.opening_book.get_book_move(board)
            if book_move:
                return book_move

        # 2. Get position assessment
        position = self.scoring_calculator.tempo.assess_position(board, color)
        phase = position['game_phase']
        endgame_factor = position['endgame_factor']
        tempo_score = position['tempo_score']
        risk_score = position['risk_score']
        zugzwang_risk = position['zugzwang_risk']

        # 3. Get all legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        # 4. Check for immediate wins and critical moves
        for move in legal_moves:
            board_copy = board.copy()
            board_copy.push(move)
            
            # Immediate win
            if board_copy.is_checkmate():
                return move
                
            # Avoid immediate draws if we have advantage
            if (board_copy.is_stalemate() or board_copy.is_repetition(3)) and \
               self.scoring_calculator._evaluate_material(board) > 0:
                continue
                
        # 5. Handle promotions with high priority but consider position
        has_promotion = any(m for m in legal_moves if m.promotion)
        if has_promotion:
            queen_promotions = [m for m in legal_moves if m.promotion == chess.QUEEN]
            if queen_promotions:
                # Consider position safety for each queen promotion
                scored_promotions = []
                for move in queen_promotions:
                    board_copy = board.copy()
                    board_copy.push(move)
                    assessment = self.scoring_calculator.tempo.assess_position(board_copy, color)
                    score = 1000.0  # Base score for queen promotion
                    if board.is_capture(move):
                        score += 100.0  # Extra for capture
                    score += assessment['tempo_score'] * 10.0  # Include tempo consideration
                    if assessment['risk_score'] < 0:
                        score *= 0.8  # Penalize risky promotions
                    scored_promotions.append((move, score))
                
                # Take the highest scoring queen promotion
                if scored_promotions:
                    scored_promotions.sort(key=lambda x: x[1], reverse=True)
                    return scored_promotions[0][0]
                
            # Consider any promotion if no good queen promotion
            promotion_moves = [m for m in legal_moves if m.promotion]
            if promotion_moves:
                return promotion_moves[0]
                
        # 6. Handle captures with tempo consideration
        if phase != 'opening' or endgame_factor > 0.3:  # More aggressive outside opening
            scored_captures = []
            for move in legal_moves:
                if board.is_capture(move):
                    captured = board.piece_at(move.to_square)
                    attacker = board.piece_at(move.from_square)
                    if captured and attacker:
                        # Calculate base capture value
                        cap_value = self.scoring_calculator.pst.piece_values.get(captured.piece_type, 0)
                        att_value = self.scoring_calculator.pst.piece_values.get(attacker.piece_type, 0)
                        if cap_value > att_value:
                            # Evaluate position after capture
                            board_copy = board.copy()
                            board_copy.push(move)
                            new_assessment = self.scoring_calculator.tempo.assess_position(board_copy, color)
                            
                            # Score the capture
                            score = cap_value - att_value
                            score += new_assessment['tempo_score'] * 5.0
                            if new_assessment['risk_score'] > risk_score:
                                score *= 1.2  # Bonus for improving position safety
                            
                            scored_captures.append((move, score))
            
            # Return the best capture if significantly good
            if scored_captures:
                scored_captures.sort(key=lambda x: x[1], reverse=True)
                if scored_captures[0][1] > 200:  # Threshold for "clearly winning" capture
                    return scored_captures[0][0]
                    
        # 7. Calculate appropriate search depth
        base_depth = self.depth
        if endgame_factor > 0.7:
            base_depth += 1  # Search deeper in endgame
        if abs(tempo_score) > 0.5:
            base_depth += 1  # Search deeper in critical positions
        if zugzwang_risk < -0.5:
            base_depth += 1  # Search deeper when zugzwang is possible
            
        # 8. Get moves ordered by tempo-aware scoring
        ordered_moves = self.ordering.order_moves(
            board, 
            max_moves=None,
            tempo_bonus=abs(tempo_score)  # Use tempo score to influence move ordering
        )
        
        # 9. Perform full search with iterative deepening
        if self.use_iterative_deepening:
            best_move = self.iterative_deepening_search(board, color)
            if best_move:
                return best_move
                
        # 10. Fallback to regular search
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in ordered_moves:
            board_copy = board.copy()
            board_copy.push(move)
            
            # Get score from child position
            score = -self._negamax(
                board_copy, 
                base_depth - 1,
                -beta,
                -alpha,
                not color
            )
            
            if score > best_score:
                best_score = score
                best_move = move
                alpha = max(alpha, score)
                
            if alpha >= beta:
                break
                
        return best_move if best_move else (ordered_moves[0] if ordered_moves else None)

    def iterative_deepening_search(self, board: chess.Board, color: chess.Color) -> Optional[chess.Move]:
        """Perform iterative deepening search with time management."""
        best_move = None
        
        # Try opening book first
        if not self.engine_config.get('skip_book', False):
            book_move = self.opening_book.get_book_move(board)
            if book_move:
                return book_move
                
        # Get time allocation for this move
        allocated_time = self.time_manager.get_allocated_move_time(board) if self.time_manager else self.search_time_limit
        self.search_time_limit = allocated_time
        
        # Clear PV tracking
        self.pv_move_stack = []
        
        # Iteratively increase depth
        for current_depth in range(1, self.max_depth + 1):
            # Perform search at current depth
            move, _ = self._alpha_beta_root(board, current_depth, color)
            
            if move:
                best_move = move
                # Store in PV stack
                self.pv_move_stack = [best_move]
                
            # Check if we've used our allocated time
            if self.time_manager and self.time_manager.is_time_up():
                break
                
        return best_move

    def _negamax_root(self, board: chess.Board, depth: int, color: chess.Color) -> Tuple[chess.Move, float]:
        """Root negamax search with alpha-beta pruning."""
        alpha = float('-inf')
        beta = float('inf')
        best_move = chess.Move.null()
        best_value = float('-inf')

        # Get ordered moves
        if self.move_ordering_enabled:
            tempo_bonus = self.scoring_calculator.tempo.assess_tempo(board, color)
            moves = self.ordering.order_moves(board, max_moves=self.max_ordered_moves, tempo_bonus=tempo_bonus)
        else:
            moves = list(board.legal_moves)

        # Search each move
        for move in moves:
            new_board = board.copy()
            new_board.push(move)

            value = -self._negamax(new_board, depth - 1, -beta, -alpha, not color)

            if value > best_value:
                best_value = value
                best_move = move
                alpha = max(alpha, value)

        # Update move ordering statistics
        if best_move != chess.Move.null():
            self._update_history_and_killer_moves(board, best_move, depth)

        return best_move, best_value

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, color: chess.Color) -> float:
        """
        Negamax search with alpha-beta pruning and tempo considerations.
        Returns the evaluation score from the position.
        """
        # Position hash for repetition detection
        position_hash = board.fen().split(' ')[0]  # Just the position part
        
        # Check for draws first
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
            
        if board.is_repetition(3):
            # Penalize repetition based on position evaluation
            base_eval = self.scoring_calculator.evaluate_position(board, color)
            return base_eval * 0.5  # Reduce score for repetitions
            
        # Check if we're at a leaf node
        if depth <= 0:
            if self.use_quiescence and not board.is_check():
                return self._quiescence_search(board, alpha, beta, color)
            else:
                return self.scoring_calculator.evaluate_position(board, color)
                
        # Get moves ordered by tempo-aware scoring
        assessment = self.scoring_calculator.tempo.assess_position(board, color)
        tempo_bonus = abs(assessment['tempo_score'])
        ordered_moves = self.ordering.order_moves(board, None, tempo_bonus)  # Fixed ordering reference and signature
        
        # Track best move at this node
        best_score = float('-inf')
        best_move = None
        
        # Search all moves
        for move in ordered_moves:
            board.push(move)
            
            # Recursive search with negation
            score = -self._negamax(board, depth - 1, -beta, -alpha, not color)
            
            board.pop()
            
            # Update best score
            if score > best_score:
                best_score = score
                best_move = move
                
            # Alpha-beta pruning
            alpha = max(alpha, score)
            if alpha >= beta:
                # Store killer move
                if not board.is_capture(move):
                    self.killer_moves[depth] = move
                break
                
        # Store best move in history table if it improved alpha
        if best_move and best_score > alpha:
            key = (best_move.from_square, best_move.to_square, color)
            self.history_table[key] = self.history_table.get(key, 0) + depth * depth
            
        return best_score
        
    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, color: chess.Color) -> float:
        """
        Quiescence search to handle tactical positions.
        Only looks at captures and checks to reach a quiet position.
        """
        # Get current stand-pat score
        stand_pat = self.scoring_calculator.evaluate_position(board, color)
        
        # Base case - return evaluated score
        if stand_pat >= beta:
            return beta
            
        # Update alpha if standing pat is better
        alpha = max(alpha, stand_pat)
        
        # Get captures and checks ordered by MVV-LVA
        capture_moves = []
        for move in board.legal_moves:
            if board.is_capture(move) or board.gives_check(move):
                capture_moves.append(move)
                
        # Exit if no captures/checks
        if not capture_moves:
            return stand_pat
            
        # Order moves by MVV-LVA and tempo
        assessment = self.scoring_calculator.tempo.assess_position(board, color)
        tempo_bonus = abs(assessment['tempo_score'])
        ordered_moves = self.ordering.order_moves(board, None, tempo_bonus)  # Fixed ordering signature
        
        # Filter to only captures/checks in MVV-LVA order
        ordered_captures = [move for move in ordered_moves if move in capture_moves]
        
        # Search captures/checks
        for move in ordered_captures:
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha, not color)
            board.pop()
            
            if score >= beta:
                return beta
                
            alpha = max(alpha, score)
            
        return alpha

    def find_checkmate_in_n(self, board: chess.Board, n: int) -> Optional[chess.Move]:
        """Find a checkmate sequence within n moves."""
        if n <= 0:
            return None

        # Check if we're in checkmate already
        if board.is_checkmate():
            return None

        legal_moves = list(board.legal_moves)
        if not legal_moves:  # No legal moves
            return None

        # First check if any move directly leads to checkmate
        for move in legal_moves:
            new_board = board.copy()
            new_board.push(move)
            if new_board.is_checkmate():
                return move

        # Then look for forced checkmate sequences
        for move in legal_moves:
            new_board = board.copy()
            new_board.push(move)
            
            # If opponent has no good response to prevent mate
            all_responses_lose = True
            for response in new_board.legal_moves:
                response_board = new_board.copy()
                response_board.push(response)
                if not self._has_mate_in_n(response_board, n - 2):
                    all_responses_lose = False
                    break
                    
            if all_responses_lose:
                return move

        return None

    def _has_mate_in_n(self, board: chess.Board, n: int) -> bool:
        """Helper function to check if position has mate in n moves."""
        if n <= 0:
            return False
            
        if board.is_checkmate():
            return True
            
        if board.is_stalemate() or board.is_insufficient_material():
            return False
            
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            new_board = board.copy()
            new_board.push(move)
            
            # If at least one response leads to mate
            all_responses_lose = True
            for response in new_board.legal_moves:
                response_board = new_board.copy()
                response_board.push(response)
                if not self._has_mate_in_n(response_board, n - 2):
                    all_responses_lose = False
                    break
                    
            if all_responses_lose:
                return True
                
        return False

    def _get_search_depth(self, board: chess.Board, legal_moves, phase: str, tempo_score: float) -> int:
        """Determine appropriate search depth based on position characteristics."""
        # Start with base depth
        depth = self.depth
        
        # Promotion possible - search deeper
        if any(m.promotion for m in legal_moves):
            return self.max_depth
            
        # Endgame position - search deeper
        if phase == 'endgame':
            return self.max_depth
            
        # Critical position based on tempo - search deeper
        if abs(tempo_score) > 0.5:
            return self.max_depth
            
        # Captures available - increase depth but respect max
        if any(board.is_capture(m) for m in legal_moves):
            return min(self.depth + 2, self.max_depth)
            
        # Use base depth for normal positions
        return depth

    def _order_moves(self, moves: List[chess.Move], board: chess.Board, tempo_bonus: float = 0.0) -> List[chess.Move]:
        """Order moves using the move ordering module."""
        return self.ordering.order_moves(board, max_moves=None, tempo_bonus=tempo_bonus)
        
    def _alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float, 
                   maximizing_player: bool, current_depth: int = 0) -> Tuple[Optional[chess.Move], float]:
        """Alpha-beta search with enhanced move ordering and tactical awareness."""
        # Get tempo bonus for critical positions
        tempo_bonus = self.tempo.calculate_tempo_bonus(board) if depth > 2 else 0.0
        
        # Get ordered moves
        moves = list(board.legal_moves)
        if not moves:
            if board.is_checkmate():
                return None, -9999999.0 if maximizing_player else 9999999.0
            return None, 0.0  # Draw
            
        ordered_moves = self._order_moves(moves, board, tempo_bonus)
        best_move = None
        best_score = float('-inf') if maximizing_player else float('inf')

        # Iterate over moves with enhanced ordering
        for move in ordered_moves:
            new_board = board.copy()
            new_board.push(move)

            # Quiescence search for tactical positions
            if depth <= self.quiescence_depth:
                score = -self.scoring_calculator.evaluate_position(new_board, not maximizing_player)
                if maximizing_player:
                    best_score = max(best_score, score)
                else:
                    best_score = min(best_score, score)
            else:
                # Regular alpha-beta search
                _, score = self._alpha_beta(new_board, depth - 1, -beta, -alpha, not maximizing_player, current_depth + 1)
                score = -score
                
                if maximizing_player:
                    best_score = max(best_score, score)
                    alpha = max(alpha, score)
                else:
                    best_score = min(best_score, score)
                    beta = min(beta, score)

            # Move ordering heuristic: prefer moves that are part of the principal variation
            if current_depth == 0 and move in self.pv_move_stack:
                best_score += 10.0  # Boost score for principal variation moves

            # Late-game tactical awareness: favor moves that create threats
            if depth <= 3 and not maximizing_player:
                for response in new_board.legal_moves:
                    response_board = new_board.copy()
                    response_board.push(response)
                    if response_board.is_check():
                        best_score -= 5.0  # Favor moves that create threats

            # Checkmate and stalemate detection
            if new_board.is_checkmate():
                return move, -9999999.0
            if new_board.is_stalemate():
                return move, 0.0  # Consider stalemate as draw

        return best_move, best_score

    def _update_history_and_killer_moves(self, board: chess.Board, move: chess.Move, depth: int):
        """Update history and killer moves after a good move is found."""
        # Update history table
        piece = board.piece_at(move.from_square)
        if piece:
            key = (piece.piece_type, move.from_square, move.to_square)
            self.history_table[key] = self.history_table.get(key, 0) + depth * depth
            
        # Update killer moves
        if depth < len(self.killer_moves):
            self.killer_moves[depth] = move

    def _alpha_beta_root(self, board: chess.Board, depth: int, color: chess.Color) -> tuple[chess.Move, float]:
        """
        Alpha-beta search at the root level with move ordering and tempo considerations.
        Returns the best move and its score.
        """
        alpha = float('-inf')
        beta = float('inf')
        best_score = float('-inf')
        
        # Get initial ordered moves with tempo consideration
        assessment = self.scoring_calculator.tempo.assess_position(board, color)
        tempo_bonus = abs(assessment['tempo_score'])
        ordered_moves = list(self.ordering.order_moves(board, None, tempo_bonus))
        
        # Ensure we have at least one move
        if not ordered_moves:
            # Return first legal move if no ordered moves available
            moves = list(board.legal_moves)
            if not moves:
                raise ValueError("No legal moves available")
            return moves[0], self.scoring_calculator.evaluate_position(board, color)
            
        # Track the principal variation
        self.pv_move_stack = []
        best_move = ordered_moves[0]  # Initialize with first move
        
        # Search each move
        for move in ordered_moves:
            board.push(move)
            
            # Regular negamax search with updated bounds
            score = -self._negamax(board, depth - 1, -beta, -alpha, not color)
            
            board.pop()
            
            # Update best move if we found a better one
            if score > best_score:
                best_score = score
                best_move = move
                self.pv_move_stack = [move]  # Start new PV with this move
                
            # Update alpha bound
            alpha = max(alpha, score)
            
        return best_move, best_score
