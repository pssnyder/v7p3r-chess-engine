# v7p3r_search.py

"""V7P3R Search Algorithms
Lightweight search implementation for V7P3R Chess Engine v1.2.
Implements alpha-beta search with move ordering and transposition tables.
"""

import chess
import random
from typing import Optional, Dict, List, Any

from v7p3r_scoring import V7P3RScoring
from v7p3r_time import V7P3RTime
from v7p3r_book import V7P3RBook
from v7p3r_config import V7P3RConfig


class V7P3RSearch:
    """Search algorithm implementation with alpha-beta pruning."""
    
    def __init__(self, scoring: V7P3RScoring, time_manager: V7P3RTime, book: V7P3RBook, config: V7P3RConfig):
        """Initialize search system.
        
        Args:
            scoring: Position evaluation system
            time_manager: Time control manager
            book: Opening book
            config: Engine configuration
        """
        self.scoring = scoring
        self.time_manager = time_manager
        self.book = book
        self.config = config
        
        # Load search configuration using the config's get_setting method
        self.use_transposition_table = config.get_setting('search_config', 'use_transposition_table', True)
        self.use_killer_moves = config.get_setting('search_config', 'use_killer_moves', True)
        self.use_history_heuristic = config.get_setting('search_config', 'use_history_heuristic', True)
        self.use_quiescence_search = config.get_setting('search_config', 'use_quiescence_search', True)
        self.quiescence_depth = config.get_setting('search_config', 'quiescence_depth', 4)
        
        # Search tracking
        self.nodes_searched = 0
        self.current_depth = 0
        
        # Move ordering data structures
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(50)]  # Two killer moves per depth
        self.history_table = {}  # History heuristic scores
    
    def find_best_move(self, board: chess.Board, transposition_table: Dict, 
                      killer_moves: List, history_table: Dict, 
                      time_limit: Optional[float] = None) -> Optional[chess.Move]:
        """Find the best move for the current position.
        
        Args:
            board: Current board position
            transposition_table: Transposition table from engine
            killer_moves: Killer moves from engine
            history_table: History table from engine
            time_limit: Time limit for search
            
        Returns:
            Best move found or None if no legal moves
        """
        if board.is_game_over():
            return None
        
        # Check opening book first
        book_move = self.book.get_book_move(board)
        if book_move:
            return book_move
        
        # Initialize search
        self.nodes_searched = 0
        self.killer_moves = killer_moves
        self.history_table = history_table
        
        # Determine search depth based on time
        depth = self.time_manager.get_search_depth(time_limit)
        
        # Start time tracking
        self.time_manager.start_search(time_limit)
        
        # Check transposition table
        if self.use_transposition_table:
            tt_entry = transposition_table.get(board.fen())
            if tt_entry and tt_entry.get('depth', 0) >= depth:
                move_str = tt_entry.get('best_move')
                if move_str:
                    try:
                        move = chess.Move.from_uci(move_str)
                        if move in board.legal_moves:
                            return move
                    except ValueError:
                        pass
        
        # Iterative deepening search
        best_move = None
        best_score = -float('inf')
        
        for current_depth in range(1, depth + 1):
            self.current_depth = current_depth
            
            if self.time_manager.should_stop_search():
                break
            
            # Alpha-beta search at current depth
            score, move = self._alpha_beta_root(board, current_depth, -float('inf'), float('inf'))
            
            if move and not self.time_manager.should_stop_search():
                best_move = move
                best_score = score
                
                # Store in transposition table
                if self.use_transposition_table and best_move:
                    transposition_table[board.fen()] = {
                        'depth': current_depth,
                        'score': best_score,
                        'best_move': best_move.uci(),
                        'flag': 'exact'
                    }
        
        return best_move
    
    def _alpha_beta_root(self, board: chess.Board, depth: int, alpha: float, beta: float) -> tuple:
        """Alpha-beta search at root level.
        
        Args:
            board: Current position
            depth: Search depth
            alpha: Alpha value
            beta: Beta value
            
        Returns:
            Tuple of (best_score, best_move)
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0, None
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(board, legal_moves, depth)
        
        best_score = -float('inf')
        best_move = ordered_moves[0]
        
        for move in ordered_moves:
            if self.time_manager.should_stop_search():
                break
            
            board.push(move)
            score = -self._alpha_beta(board, depth - 1, -beta, -alpha, False)
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
                alpha = max(alpha, score)
                
                if alpha >= beta:
                    # Update killer moves
                    if self.use_killer_moves and depth < len(self.killer_moves):
                        if self.killer_moves[depth][0] != move:
                            self.killer_moves[depth][1] = self.killer_moves[depth][0]
                            self.killer_moves[depth][0] = move
                    break
        
        return best_score, best_move
    
    def _alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        """Alpha-beta search algorithm.
        
        Args:
            board: Current position
            depth: Remaining search depth
            alpha: Alpha value
            beta: Beta value
            is_maximizing: True if maximizing player
            
        Returns:
            Position evaluation score
        """
        self.nodes_searched += 1
        
        # Terminal conditions
        if depth <= 0:
            if self.use_quiescence_search:
                return self._quiescence_search(board, self.quiescence_depth, alpha, beta, is_maximizing)
            else:
                return self.scoring.evaluate_board(board, 0.0)  # Use default endgame factor
        
        if board.is_game_over():
            if board.is_checkmate():
                return -9999.0 + (self.current_depth - depth)  # Prefer faster mates
            else:
                return 0.0  # Draw
        
        if self.time_manager.should_stop_search():
            return self.scoring.evaluate_board(board, 0.0)  # Use default endgame factor
        
        # Generate and order moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0
        
        ordered_moves = self._order_moves(board, legal_moves, depth)
        
        if is_maximizing:
            max_eval = -float('inf')
            for move in ordered_moves:
                board.push(move)
                eval_score = self._alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    # Update killer moves and history
                    self._update_move_ordering(move, depth)
                    break
            
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                board.push(move)
                eval_score = self._alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    # Update killer moves and history
                    self._update_move_ordering(move, depth)
                    break
            
            return min_eval
    
    def _quiescence_search(self, board: chess.Board, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        """Quiescence search for tactical positions.
        
        Args:
            board: Current position
            depth: Remaining quiescence depth
            alpha: Alpha value
            beta: Beta value
            is_maximizing: True if maximizing player
            
        Returns:
            Position evaluation score
        """
        self.nodes_searched += 1
        
        # Stand pat evaluation
        stand_pat = self.scoring.evaluate_board(board, 0.0)  # Use default endgame factor
        
        if depth <= 0:
            return stand_pat
        
        if is_maximizing:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)
        
        # Only consider captures and checks in quiescence
        tactical_moves = [move for move in board.legal_moves 
                         if board.is_capture(move) or board.gives_check(move)]
        
        if not tactical_moves:
            return stand_pat
        
        # Order tactical moves (captures first)
        tactical_moves.sort(key=lambda move: self._get_mvv_lva_score(board, move), reverse=True)
        
        if is_maximizing:
            for move in tactical_moves:
                board.push(move)
                score = self._quiescence_search(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if score >= beta:
                    return beta
                alpha = max(alpha, score)
            
            return alpha
        else:
            for move in tactical_moves:
                board.push(move)
                score = self._quiescence_search(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if score <= alpha:
                    return alpha
                beta = min(beta, score)
            
            return beta
    
    def _order_moves(self, board: chess.Board, moves: List[chess.Move], depth: int) -> List[chess.Move]:
        """Order moves for better alpha-beta pruning.
        
        Args:
            board: Current position
            moves: List of legal moves
            depth: Current search depth
            
        Returns:
            Ordered list of moves
        """
        move_scores = []
        
        for move in moves:
            score = 0
            
            # MVV-LVA for captures
            if board.is_capture(move):
                score += self._get_mvv_lva_score(board, move)
            
            # Checks
            if board.gives_check(move):
                score += 50
            
            # Killer moves
            if self.use_killer_moves and depth < len(self.killer_moves):
                if move == self.killer_moves[depth][0]:
                    score += 100
                elif move == self.killer_moves[depth][1]:
                    score += 80
            
            # History heuristic
            if self.use_history_heuristic:
                move_key = move.uci()
                score += self.history_table.get(move_key, 0)
            
            move_scores.append((score, move))
        
        # Sort by score (highest first)
        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in move_scores]
    
    def _get_mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        """Get Most Valuable Victim - Least Valuable Attacker score.
        
        Args:
            board: Current position
            move: Move to evaluate
            
        Returns:
            MVV-LVA score
        """
        if not board.is_capture(move):
            return 0
        
        attacker = board.piece_at(move.from_square)
        victim = board.piece_at(move.to_square)
        
        if not attacker or not victim:
            return 0
        
        # Piece values for MVV-LVA (in centipawns)
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 325,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 10000
        }
        
        victim_value = piece_values.get(victim.piece_type, 0)
        attacker_value = piece_values.get(attacker.piece_type, 0)
        
        # Most valuable victim first, then least valuable attacker
        return victim_value * 10 - attacker_value
    
    def _update_move_ordering(self, move: chess.Move, depth: int):
        """Update killer moves and history heuristic.
        
        Args:
            move: Move that caused cutoff
            depth: Search depth
        """
        # Update killer moves
        if self.use_killer_moves and depth < len(self.killer_moves):
            if self.killer_moves[depth][0] != move:
                self.killer_moves[depth][1] = self.killer_moves[depth][0]
                self.killer_moves[depth][0] = move
        
        # Update history heuristic
        if self.use_history_heuristic:
            move_key = move.uci()
            self.history_table[move_key] = self.history_table.get(move_key, 0) + depth * depth
    
    def get_search_info(self) -> Dict[str, Any]:
        """Get search statistics."""
        return {
            'nodes_searched': self.nodes_searched,
            'current_depth': self.current_depth,
            'nps': self.time_manager.get_nodes_per_second(),
            'time_info': self.time_manager.get_search_info()
        }
