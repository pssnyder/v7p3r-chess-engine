# v7p3r.py

""" V7P3R Evaluation Engine
This module implements the evaluation engine for the V7P3R chess AI.
It provides various search algorithms, evaluation functions, and move ordering
"""

import chess
import random
import time
from typing import Optional, Callable, Tuple
from time_manager import TimeManager
from v7p3r_scoring_calculation import V7P3RScoringCalculation # Import the new scoring module
from collections import OrderedDict

class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, maxlen=100000, **kwargs):
        self.maxlen = maxlen
        super().__init__(*args, **kwargs)
    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxlen:
            oldest = next(iter(self))
            del self[oldest]

class V7P3REvaluationEngine:
    def __init__(self, board: chess.Board = chess.Board(), player: chess.Color = chess.WHITE, ai_config=None):
        self.board = board
        self.current_player = player
        self.time_manager = TimeManager()

        self.nodes_searched = 0
        self.transposition_table = LimitedSizeDict(maxlen=1000000) 
        self.killer_moves = [[None, None] for _ in range(50)] 
        self.history_table = {}
        self.counter_moves = {}

        self.depth = 6
        self.max_depth = 10
        self.piece_values = {
            chess.KING: 0.0,
            chess.QUEEN: 9.0,
            chess.ROOK: 5.0,
            chess.BISHOP: 3.25,
            chess.KNIGHT: 3.0,
            chess.PAWN: 1.0
        }

        self.hash_size = 1024*1024
        self.transposition_table.maxlen = self.hash_size // 100 # Approximate entry size

        self.scoring_calculator = V7P3RScoringCalculation(self.piece_values)
        self.endgame_factor = 0.0
        self.reset()

    def reset(self):
        self.nodes_searched = 0
        self.transposition_table.clear()
        self.killer_moves = [[None, None] for _ in range(50)]
        self.history_table.clear()
        self.counter_moves.clear()

    def _is_draw_condition(self, board):
        if board.can_claim_threefold_repetition():
            return True
        if board.can_claim_fifty_moves():
            return True
        if board.is_seventyfive_moves():
            return True
        return False

    def _get_game_phase_factor(self, board: chess.Board) -> float:
        """Evaluate the game phase based on material balance"""
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
            return False
        self.board = game_board.copy()
        self.game_board = game_board.copy()
        return True

    def has_game_board_changed(self):
        if self.game_board is None:
            return False
        return self.board.fen() != self.game_board.fen()

    def search(self, board: chess.Board, player: chess.Color, ai_config: dict = {}, stop_callback: Optional[Callable[[], bool]] = None) -> chess.Move:
        """Main search function - sets up and calls minimax search"""
        self.nodes_searched = 0
        self.sync_with_game_board(board)
        self.current_player = player

        # Check for immediate transposition table hit
        search_depth = self.depth if self.depth is not None else 6
        trans_move, trans_score = self.get_transposition_move(board, search_depth)
        if trans_move and board.is_legal(trans_move):
            return trans_move

        # Call minimax search which handles all move iteration
        best_move, best_score = self._minimax_search_root(board, search_depth, -float('inf'), float('inf'), True, stop_callback)
        
        # Update transposition table with result
        if best_move:
            self.update_transposition_table(board, search_depth, best_move, best_score)
        
        # Fallback handling
        if best_move is None or best_move == chess.Move.null():
            legal_moves = list(board.legal_moves)
            if legal_moves:
                # Try move ordering for fallback
                hash_move, _ = self.get_transposition_move(board, 1)
                ordered_moves = self.order_moves(board, legal_moves, hash_move=hash_move, depth=1)
                best_move = ordered_moves[0] if ordered_moves else random.choice(legal_moves)
            else:
                return chess.Move.null()

        # Apply draw prevention
        best_move = self._enforce_strict_draw_prevention(board, best_move)
        
        # Final safety check
        if not isinstance(best_move, chess.Move) or not board.is_legal(best_move):
            legal_moves = list(board.legal_moves)
            if legal_moves:
                best_move = random.choice(legal_moves)
            else:
                best_move = chess.Move.null()

        return best_move

    def _minimax_search_root(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool, stop_callback: Optional[Callable[[], bool]] = None) -> Tuple[Optional[chess.Move], float]:
        """Root level minimax search that returns both move and score"""
        self.nodes_searched += 1
        
        if stop_callback and stop_callback():
            return None, self.evaluate_position_from_perspective(board, self.current_player)

        # Terminal conditions
        if depth <= 0 or board.is_game_over(claim_draw=self._is_draw_condition(board)):
            quies_score = self._quiescence_search(board, alpha, beta, maximizing_player, stop_callback)
            return None, quies_score

        # Get and order moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            if board.is_check():
                return None, -9999999999.0 if maximizing_player else 9999999999.0
            else:
                return None, 0.0  # Stalemate

        # Check transposition table for move ordering
        tt_move, _ = self.get_transposition_move(board, depth)
        ordered_moves = self.order_moves(board, legal_moves, hash_move=tt_move, depth=depth)

        best_move = None
        best_score = -float('inf') if maximizing_player else float('inf')

        for move in ordered_moves:
            if stop_callback and stop_callback():
                break

            board.push(move)
            score = self._minimax_search(board, depth - 1, alpha, beta, not maximizing_player, stop_callback)
            board.pop()

            # Update best move and score
            if maximizing_player:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
                if beta <= alpha:
                    self.update_killer_move(move, depth)
                    self.update_history_score(board, move, depth)
                    break
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha:
                    self.update_killer_move(move, depth)
                    self.update_history_score(board, move, depth)
                    break

        return best_move, best_score

    def find_best_move(self, board: chess.Board, time_limit: float = 5.0) -> chess.Move:
        """Main entry point for finding best move with proper time management and iterative deepening"""
        # Build time control dictionary
        time_control = {
            'movetime': int(time_limit * 1000)  # Convert to milliseconds
        }
        
        # Allocate time using time manager
        allocated_time = self.time_manager.allocate_time(time_control, board)
        self.time_manager.start_timer(allocated_time)
        
        # Set current player to whoever's turn it is
        current_player = board.turn
        
        # Iterative deepening search
        best_move = None
        max_depth = min(self.max_depth if self.max_depth else 8, 10)  # Cap at reasonable depth
        
        for current_depth in range(1, max_depth + 1):
            if self.time_manager.should_stop():
                break
                
            # Temporarily set depth for this iteration
            old_depth = self.depth
            self.depth = current_depth
            
            try:
                # Search at current depth
                move = self.search(board, current_player, stop_callback=self.time_manager.should_stop)
                
                # If we got a valid move and didn't run out of time, use it
                if move and move != chess.Move.null() and not self.time_manager.should_stop():
                    best_move = move
                
            except Exception as e:
                # If search fails, break and use last good result
                break
            finally:
                # Restore original depth
                self.depth = old_depth
        
        # Return best move found, or fallback if none
        if not best_move:
            best_move = self.search(board, current_player, stop_callback=self.time_manager.should_stop)
            
        return best_move if best_move else chess.Move.null()

    # =================================
    # ===== EVALUATION FUNCTIONS ======

    def evaluate_position(self, board: chess.Board) -> float:
        """Calculate base position evaluation by delegating to scoring_calculator."""
        positional_evaluation_board = board.copy()
        if not isinstance(positional_evaluation_board, chess.Board) or not positional_evaluation_board.is_valid():
            return 0.0
        
        endgame_factor = self._get_game_phase_factor(positional_evaluation_board)
        
        score = self.scoring_calculator.calculate_score(
            board=positional_evaluation_board,
            color=chess.WHITE,
            endgame_factor=endgame_factor
        ) - self.scoring_calculator.calculate_score(
            board=positional_evaluation_board,
            color=chess.BLACK,
            endgame_factor=endgame_factor
        )
        
        return score

    def evaluate_position_from_perspective(self, board: chess.Board, player: chess.Color) -> float:
        """Calculate position evaluation from specified player's perspective by delegating to scoring_calculator."""
        perspective_evaluation_board = board.copy()
        if not isinstance(player, chess.Color) or not perspective_evaluation_board.is_valid():
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
        
        return score

    def evaluate_move(self, board: chess.Board, move: chess.Move = chess.Move.null()) -> float:
        """Quick evaluation of individual move on overall eval"""
        score = 0.0
        move_evaluation_board = board.copy()
        if not move_evaluation_board.is_legal(move):
            return -9999999999
        
        move_evaluation_board.push(move)
        score = self.evaluate_position(move_evaluation_board)
        
        move_evaluation_board.pop()
        return score

    # ===================================
    # ======= HELPER FUNCTIONS ==========
    
    def order_moves(self, board: chess.Board, moves, hash_move: Optional[chess.Move] = None, depth: int = 0):
        """Order moves for better alpha-beta pruning efficiency"""
        if isinstance(moves, chess.Move):
            moves = [moves]
        
        if not moves or not isinstance(board, chess.Board) or not board.is_valid():
            return []

        move_scores = []
        
        if hash_move and hash_move in moves:
            # Use a very high bonus for hash move, potentially from config if defined
            hash_move_bonus = 2000000.0
            move_scores.append((hash_move, hash_move_bonus))
            moves = [m for m in moves if m != hash_move]

        for move in moves:
            if not board.is_legal(move):
                continue
            
            score = self._order_move_score(board, move, depth)
            move_scores.append((move, score))

        move_scores.sort(key=lambda x: x[1], reverse=True)

        return [move for move, _ in move_scores]

    def _order_move_score(self, board: chess.Board, move: chess.Move, depth: int = 6) -> float:
        """Enhanced move scoring with threat detection and better capture evaluation."""
        score = 0.0

        # Check for checkmate first (highest priority)
        temp_board = board.copy()
        temp_board.push(move)
        if temp_board.is_checkmate():
            temp_board.pop()
            return 9999999999.0
        
        # Check moves get high priority
        if temp_board.is_check():
            score += 10000.0
        temp_board.pop()

        # Enhanced capture evaluation using SEE
        if board.is_capture(move):
            see_score = self._static_exchange_evaluation(board, move)
            score += 1000000.0 + (see_score * 10000)  # Base capture bonus + SEE result
        
        # Threat detection - prioritize moves that save threatened pieces
        piece_threat_score = self._evaluate_piece_threats(board, move)
        score += piece_threat_score
        
        # Killer moves
        if depth < len(self.killer_moves) and move in self.killer_moves:
            score += 900000.0

        # History heuristic
        score += self.history_table.get((board.turn, move.from_square, move.to_square), 0)
        
        # Promotion bonus
        if move.promotion:
            score += 700000.0
            if move.promotion == chess.QUEEN:
                score += self.piece_values.get(chess.QUEEN, 9.0) * 100

        return score
    
    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, maximizing_player: bool, stop_callback: Optional[Callable[[], bool]] = None, current_ply: int = 0) -> float:
        """Enhanced quiescence search with better tactical awareness"""
        self.nodes_searched += 1

        if stop_callback and stop_callback():
            return self.evaluate_position_from_perspective(board, self.current_player)

        # Get stand-pat score (evaluation if we make no more moves)
        stand_pat_score = self.evaluate_position_from_perspective(board, self.current_player)

        # Depth limit for quiescence search
        max_q_depth = 10  # Increased for better tactical analysis
        if current_ply >= max_q_depth:
            return stand_pat_score

        # Alpha-beta pruning with stand-pat score
        if maximizing_player:
            if stand_pat_score >= beta:
                return beta 
            alpha = max(alpha, stand_pat_score)
        else:
            if stand_pat_score <= alpha:
                return alpha
            beta = min(beta, stand_pat_score)

        # Generate moves based on position
        moves = []
        
        if board.is_check():
            # If in check, consider all legal moves to escape check
            moves = list(board.legal_moves)
        else:
            # Enhanced tactical move generation
            for move in board.legal_moves:
                include_move = False
                
                # Always include captures
                if board.is_capture(move):
                    # Use SEE to filter bad captures only in deeper plies
                    if current_ply < 3:  # Include all captures in shallow quiescence
                        include_move = True
                    else:
                        # Filter obviously bad captures in deeper search
                        see_score = self._static_exchange_evaluation(board, move)
                        if see_score >= -0.5:  # Allow small sacrifices for tactics
                            include_move = True
                
                # Always include promotions
                elif move.promotion:
                    include_move = True
                
                # Include checks that might lead to tactics
                elif current_ply < 2:  # Only in shallow quiescence to avoid explosion
                    temp_board = board.copy()
                    temp_board.push(move)
                    if temp_board.is_check():
                        include_move = True
                    temp_board.pop()
                
                # Include moves that save threatened pieces (defensive moves)
                elif current_ply < 2:
                    moving_piece = board.piece_at(move.from_square)
                    if moving_piece and self._is_piece_threatened(board, move.from_square, moving_piece.color):
                        # Check if destination is safer
                        temp_board = board.copy()
                        temp_board.push(move)
                        if not self._is_piece_threatened(temp_board, move.to_square, moving_piece.color):
                            piece_value = self.piece_values.get(moving_piece.piece_type, 0)
                            if piece_value >= 3.0:  # Only for valuable pieces (not pawns)
                                include_move = True
                        temp_board.pop()
                
                if include_move:
                    moves.append(move)
            
        # If no tactical moves and not in check, return stand-pat
        if not moves:
            return stand_pat_score

        # Order moves for better pruning
        moves = self.order_moves(board, moves, depth=current_ply)

        # Search tactical moves
        for move in moves:
            board.push(move)
            score = self._quiescence_search(board, alpha, beta, not maximizing_player, stop_callback, current_ply + 1)
            board.pop()

            if maximizing_player:
                alpha = max(alpha, score)
                if alpha >= beta:
                    break  # Beta cutoff
            else:
                beta = min(beta, score)
                if alpha >= beta:
                    break  # Alpha cutoff
        
        return alpha if maximizing_player else beta
    
    def _static_exchange_evaluation(self, board: chess.Board, move: chess.Move) -> float:
        """
        Static Exchange Evaluation - Calculate the net material gain/loss from a capture sequence.
        Returns positive value if the exchange favors the moving side.
        """
        if not board.is_capture(move):
            return 0.0
        
        # Get piece values
        target_square = move.to_square
        victim_piece = board.piece_at(target_square)
        attacker_piece = board.piece_at(move.from_square)
        
        if not victim_piece or not attacker_piece:
            return 0.0
            
        victim_value = self.piece_values.get(victim_piece.piece_type, 0)
        attacker_value = self.piece_values.get(attacker_piece.piece_type, 0)
        
        # Start with capturing the victim
        gain = [victim_value]
        
        # Make the capture
        board.push(move)
        
        # Find all attackers to this square
        attackers = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and board.is_attacked_by(not board.turn, target_square):
                if piece.color != board.turn:  # Opponent's piece
                    attackers.append((square, piece))
        
        # Sort attackers by piece value (cheapest first)
        attackers.sort(key=lambda x: self.piece_values.get(x[1].piece_type, 0))
        
        # Simulate the exchange
        current_attacker_value = attacker_value
        side_to_move = not board.turn  # Opponent's turn to recapture
        
        for attacker_square, attacker_piece in attackers:
            # Check if this piece can actually attack the target
            temp_move = None
            for legal_move in board.legal_moves:
                if legal_move.from_square == attacker_square and legal_move.to_square == target_square:
                    temp_move = legal_move
                    break
            
            if temp_move:
                gain.append(current_attacker_value)
                current_attacker_value = self.piece_values.get(attacker_piece.piece_type, 0)
                side_to_move = not side_to_move
            
            # Limit depth to prevent infinite analysis
            if len(gain) > 8:
                break
        
        board.pop()  # Undo the original move
        
        # Calculate net gain using minimax on the gain array
        if len(gain) == 1:
            return gain[0]
        
        # Work backwards through the exchange
        for i in range(len(gain) - 2, -1, -1):
            gain[i] = max(0, gain[i] - gain[i + 1])
        
        return gain[0]
    
    def _evaluate_piece_threats(self, board: chess.Board, move: chess.Move) -> float:
        """
        Evaluate threats to pieces and prioritize moves that address them.
        Returns a bonus for moves that save threatened pieces or create threats.
        """
        score = 0.0
        
        # Check if the moving piece was under threat
        moving_piece = board.piece_at(move.from_square)
        if moving_piece and self._is_piece_threatened(board, move.from_square, moving_piece.color):
            # Bonus for moving a threatened piece to safety
            piece_value = self.piece_values.get(moving_piece.piece_type, 0)
            
            # Check if destination is safer
            temp_board = board.copy()
            temp_board.push(move)
            if not self._is_piece_threatened(temp_board, move.to_square, moving_piece.color):
                score += piece_value * 50000  # Large bonus for saving threatened piece
            temp_board.pop()
        
        # Check if move creates threats to opponent pieces
        temp_board = board.copy()
        temp_board.push(move)
        
        # Look for newly threatened opponent pieces
        for square in chess.SQUARES:
            piece = temp_board.piece_at(square)
            if piece and piece.color != board.turn:
                if self._is_piece_threatened(temp_board, square, piece.color):
                    # Check if this threat is new (wasn't there before)
                    if not self._is_piece_threatened(board, square, piece.color):
                        threat_value = self.piece_values.get(piece.piece_type, 0)
                        score += threat_value * 10000  # Bonus for creating threats
        
        temp_board.pop()
        
        # Penalty for leaving valuable pieces hanging
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn and square != move.from_square:
                if self._is_piece_threatened(board, square, piece.color):
                    # Check if we're not defending this piece with our move
                    temp_board = board.copy()
                    temp_board.push(move)
                    if self._is_piece_threatened(temp_board, square, piece.color):
                        # Still threatened after our move - penalty
                        piece_value = self.piece_values.get(piece.piece_type, 0)
                        score -= piece_value * 5000
                    temp_board.pop()
        
        return score
    
    def _is_piece_threatened(self, board: chess.Board, square: chess.Square, piece_color: chess.Color) -> bool:
        """Check if a piece at the given square is under attack by the opponent."""
        return board.is_attacked_by(not piece_color, square)

    def get_transposition_move(self, board: chess.Board, depth: int) -> Tuple[Optional[chess.Move], Optional[float]]:
        key = board.fen()
        if key in self.transposition_table:
            entry = self.transposition_table[key]
            if entry['depth'] >= depth:
                return entry['best_move'], entry['score']
        return None, None
    
    def update_transposition_table(self, board: chess.Board, depth: int, best_move: Optional[chess.Move], score: float):
        key = board.fen()
        if key in self.transposition_table:
            existing_entry = self.transposition_table[key]
            if depth < existing_entry['depth'] and score <= existing_entry['score']:
                return

        if best_move is not None:
            self.transposition_table[key] = {
                'best_move': best_move,
                'depth': depth,
                'score': score
            }

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
            return self._random_search(board)  # Fallback if no piece at from_square
        history_key = (piece.piece_type, move.from_square, move.to_square)

        # Update history score using depth-squared bonus
        self.history_table[history_key] = self.history_table.get(history_key, 0) + depth * depth

    def _enforce_strict_draw_prevention(self, board: chess.Board, move: Optional[chess.Move]) -> Optional[chess.Move]:
        """Enforce strict draw prevention rules to block moves that would lead to stalemate, insufficient material, or threefold repetition."""
        if move is None:
            return None
        
        temp_board = board.copy()
        try:
            temp_board.push(move)
            
            # IMPORTANT: Don't prevent checkmate moves! Checkmate is a win, not a draw.
            if temp_board.is_checkmate():
                temp_board.pop()
                return move  # Checkmate is good, return the move
            
            # Only prevent actual draw conditions
            if temp_board.is_stalemate() or temp_board.is_insufficient_material() or \
               temp_board.is_fivefold_repetition() or temp_board.is_repetition(count=3):
                
                temp_board.pop()
                legal_moves_from_current_board = list(board.legal_moves)
                non_draw_moves = []
                for m in legal_moves_from_current_board:
                    if m == move:
                        continue
                    test_board_for_draw = board.copy()
                    test_board_for_draw.push(m)
                    # Check for draws, but allow checkmate
                    if not (test_board_for_draw.is_stalemate() or test_board_for_draw.is_insufficient_material() or \
                            test_board_for_draw.is_fivefold_repetition() or test_board_for_draw.is_repetition(count=3)) or \
                       test_board_for_draw.is_checkmate():  # Allow checkmate moves
                        non_draw_moves.append(m)
                
                if non_draw_moves:
                    chosen_move = random.choice(non_draw_moves)
                    return chosen_move
                else:
                    return move  # If no alternatives, return original move
            else:
                temp_board.pop()
                return move  # No draw condition, return original move
                
        except ValueError:
            return self._random_search(board)

    # =======================================
    # ======= MAIN SEARCH ALGORITHMS ========
    
    def _random_search(self, board: chess.Board) -> chess.Move:
        """Select a random legal move from the board."""
        legal_moves = list(board.legal_moves)
        legal_moves = self.order_moves(board, legal_moves)
        if not legal_moves:
            return chess.Move.null() # Return null move if no legal moves
        legal_moves = legal_moves[:5]
        move = random.choice(legal_moves)
        return move

    def _minimax_search(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool, stop_callback: Optional[Callable[[], bool]] = None) -> float:
        """Recursive minimax search with alpha-beta pruning. Returns only the score."""
        self.nodes_searched += 1
        
        if stop_callback and stop_callback():
            return self.evaluate_position_from_perspective(board, self.current_player)

        # Check transposition table first
        tt_move, tt_score = self.get_transposition_move(board, depth)
        if tt_score is not None:
            return tt_score

        # Terminal node - call quiescence search
        if depth <= 0 or board.is_game_over(claim_draw=self._is_draw_condition(board)):
            return self._quiescence_search(board, alpha, beta, maximizing_player, stop_callback)

        # Generate and order moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            # No legal moves - checkmate or stalemate
            if board.is_check():
                return -9999999999.0 if maximizing_player else 9999999999.0
            else:
                return 0.0  # Stalemate

        ordered_moves = self.order_moves(board, legal_moves, hash_move=tt_move, depth=depth)
        best_move_for_tt = None

        if maximizing_player:
            max_eval = -float('inf')
            for move in ordered_moves:
                board.push(move)
                eval_score = self._minimax_search(board, depth - 1, alpha, beta, False, stop_callback)
                board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move_for_tt = move
                    
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    # Beta cutoff - update heuristics
                    self.update_killer_move(move, depth)
                    self.update_history_score(board, move, depth)
                    break
            
            # Store in transposition table
            if best_move_for_tt:
                self.update_transposition_table(board, depth, best_move_for_tt, max_eval)
            return max_eval
            
        else:  # Minimizing player
            min_eval = float('inf')
            for move in ordered_moves:
                board.push(move)
                eval_score = self._minimax_search(board, depth - 1, alpha, beta, True, stop_callback)
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move_for_tt = move
                    
                beta = min(beta, eval_score)
                if beta <= alpha:
                    # Alpha cutoff - update heuristics
                    self.update_killer_move(move, depth)
                    self.update_history_score(board, move, depth)
                    break
                    
            # Store in transposition table
            if best_move_for_tt:
                self.update_transposition_table(board, depth, best_move_for_tt, min_eval)
            return min_eval
