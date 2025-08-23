# v7p3r.py

""" V7P3R Evaluation Engine
This module implements the evaluation engine for the V7P3R chess AI.
It provides various search algorithms, evaluation functions, and move ordering
"""

import chess
import random
import time
from typing import Optional, Callable, Tuple
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

        self.nodes_searched = 0
        self.transposition_table = LimitedSizeDict(maxlen=1000000) 
        self.killer_moves = [[None, None] for _ in range(50)] 
        self.history_table = {}
        self.counter_moves = {}

        self.depth = 6  # Default depth - always use even numbers to include opponent response
        self.max_depth = 10
        self.piece_values = {
            chess.KING: 0.0,
            chess.QUEEN: 9.0,
            chess.ROOK: 5.0,
            chess.BISHOP: 3.25,
            chess.KNIGHT: 3.0,
            chess.PAWN: 1.0
        }
        
        # V6.2 Optimization flags
        self.use_fast_search = True  # Enable optimized search path
        self.fast_move_limit = 12    # Limit moves considered in fast mode
        self.use_optimized_scoring = True  # Use threshold-based early exits in scoring
        
        # V6.2 Opening knowledge injection
        self.use_opening_tt_injection = True  # Inject opening moves into TT
        if self.use_opening_tt_injection:
            self._inject_opening_knowledge()  # Pre-populate TT with good opening moves

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

    def set_search_mode(self, use_fast_search: bool, fast_move_limit: int = 12):
        """Configure search optimization settings"""
        self.use_fast_search = use_fast_search
        self.fast_move_limit = fast_move_limit
        if use_fast_search:
            print(f"V7P3R: Using fast search mode (limit: {fast_move_limit} moves)")
        else:
            print("V7P3R: Using traditional search mode")

    def set_optimization_mode(self, use_optimized_scoring: bool = True, use_opening_injection: bool = True):
        """Configure evaluation optimization settings"""
        self.use_optimized_scoring = use_optimized_scoring
        self.use_opening_tt_injection = use_opening_injection
        print(f"V7P3R: Optimized scoring: {use_optimized_scoring}, Opening injection: {use_opening_injection}")
        
        # Re-inject opening knowledge if enabled
        if use_opening_injection:
            self._inject_opening_knowledge()

    def _inject_opening_knowledge(self):
        """Inject good opening moves into transposition table for guidance"""
        if not self.use_opening_tt_injection:
            return
            
        # Always inject during initialization, but check ply when called during search
        # Only skip injection if we're in the middle of a game (ply >= 8)
        if hasattr(self, 'board') and self.board and self.board.ply() >= 8:
            return
            
        # Define opening positions and their preferred moves with evaluations
        opening_knowledge = [
            # Starting position - prioritize central pawn moves and knight development
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", 500),  # King's pawn - strong
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "d2d4", 450),  # Queen's pawn - strong
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3", 400),  # King's knight - good
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "b1c3", 350),  # Queen's knight - decent
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e3", -200), # Passive - discourage
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "d2d3", -150), # Passive - discourage
            
            # After e4 - good responses
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "e7e5", 500),   # Symmetric - strong
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "c7c5", 450),   # Sicilian - strong
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "e7e6", 400),   # French - good
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "d7d6", 350),   # Pirc - decent
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "g8f6", 400),   # Alekhine - good
            
            # After d4 - good responses  
            ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "d7d5", 500),   # Symmetric - strong
            ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "g8f6", 450),   # Indian systems - strong
            ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "e7e6", 400),   # Queen's Gambit Declined prep - good
            ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", "c7c5", 400),   # Benoni - good
            
            # After e4 e5 - knight development
            ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "g1f3", 500),  # King's knight - strong
            ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "b1c3", 400),  # Queen's knight - good
            ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "f1c4", 450),  # Italian setup - strong
            
            # Black's responses to Nf3
            ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "b8c6", 500), # Knight development - strong
            ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "g8f6", 450), # Knight development - strong
            ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "f8e7", 300), # Passive but solid
            
            # Sicilian Defense guidance
            ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", "g1f3", 500),  # Standard development - strong
            ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", "b1c3", 400),  # Alternative development - good
            ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", "f2f4", 250),  # King's Indian Attack - ok
            
            # Queen's Gambit guidance
            ("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2", "c2c4", 500),  # Queen's Gambit - strong
            ("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2", "g1f3", 450),  # King's Indian Attack - strong
            ("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2", "b1c3", 400),  # Development - good
            
            # Early middlegame - piece development priorities
            ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f1b5", 500), # Spanish/Ruy Lopez - strong
            ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f1c4", 450), # Italian Game - strong
            ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "b1c3", 400), # Four Knights - good
            ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "d2d3", 300), # Solid setup - ok
        ]
        
        # Inject knowledge into transposition table
        injected_count = 0
        position_moves = {}  # Group moves by position
        
        # First, group all moves by position to find the best one for each
        for fen, move_uci, score_cp in opening_knowledge:
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(move_uci)
                
                if board.is_legal(move):
                    if fen not in position_moves or score_cp > position_moves[fen][1]:
                        position_moves[fen] = (move, score_cp)
                        
            except (ValueError, chess.InvalidMoveError):
                # Skip invalid positions/moves
                continue
        
        # Now inject only the best move for each position
        for fen, (move, score_cp) in position_moves.items():
            board = chess.Board(fen)
            # Use very high depth so these moves take absolute priority over evaluation
            self.update_transposition_table(board, 15, move, score_cp / 100.0)
            injected_count += 1
        
        if injected_count > 0:
            print(f"V7P3R: Injected {injected_count} opening moves into transposition table")

    def clear_opening_knowledge(self):
        """Clear injected opening knowledge - useful for testing pure engine evaluation"""
        self.transposition_table.clear()
        print("V7P3R: Cleared opening knowledge from transposition table")

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
    # ===== V6.2 FAST SEARCH ==========
    
    def _fast_negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        """Ultra-lean negamax search optimized for speed - proper negamax implementation"""
        self.nodes_searched += 1
        
        # Terminal depth - use full evaluation with optimized scoring
        if depth == 0:
            return self.evaluate_position_from_perspective(board, board.turn)
        
        # Terminal positions
        if board.is_game_over():
            if board.is_checkmate():
                # Prefer faster checkmates - negative for current player if in checkmate
                return -999999 + depth
            return 0  # Stalemate or draw
        
        best_score = -999999
        
        # Get moves - use fast ordering for deeper searches
        if depth > 3:
            moves = self._fast_move_ordering(board)
        else:
            moves = list(board.legal_moves)[:self.fast_move_limit]
        
        # Search moves with proper negamax
        for move in moves:
            board.push(move)
            score = -self._fast_negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            
            best_score = max(best_score, score)
            alpha = max(alpha, score)
                
            # Alpha-beta cutoff
            if alpha >= beta:
                break
        
        return best_score
    
    def _fast_search_root(self, board: chess.Board, depth: int) -> Tuple[Optional[chess.Move], float]:
        """Fast root search that returns best move and score"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0
        
        # Fast move ordering for root
        ordered_moves = self._fast_move_ordering(board, legal_moves)
        
        best_move = ordered_moves[0]
        best_score = -999999
        alpha = -999999
        beta = 999999
        
        for move in ordered_moves:
            board.push(move)
            score = -self._fast_negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
                alpha = max(alpha, score)
        
        return best_move, best_score
    
    def _fast_material_balance(self, board: chess.Board) -> float:
        """Ultra-fast material calculation from current side's perspective"""
        material = 0
        
        for piece_type, value in self.piece_values.items():
            if piece_type == chess.KING:
                continue
            
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            
            # Always evaluate from the perspective of the side to move
            if board.turn == chess.WHITE:
                material += (white_count - black_count) * value
            else:
                material += (black_count - white_count) * value
        
        return material
    
    def _fast_move_ordering(self, board: chess.Board, moves=None) -> list:
        """C0BR4-style lightweight move ordering - simple and efficient"""
        if moves is None:
            moves = list(board.legal_moves)
        
        if len(moves) <= self.fast_move_limit:
            return moves
        
        scored_moves = []
        
        for move in moves:
            score = 0
            
            # 1. Captures (MVV-LVA style - like C0BR4)
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    # C0BR4 style: capture value - attacker value
                    capture_value = self.piece_values.get(victim.piece_type, 0)
                    attacker_value = self.piece_values.get(attacker.piece_type, 0)
                    score += 10000 + (capture_value - attacker_value)
            
            # 2. Promotions (like C0BR4)
            if move.promotion:
                score += 9000
                if move.promotion == chess.QUEEN:
                    score += self.piece_values.get(chess.QUEEN, 900)
            
            # 3. Checks (like C0BR4)
            if not board.is_capture(move):  # Only check if not capture to save time
                board.push(move)
                if board.is_check():
                    score += 500
                board.pop()
            
            # 4. Center control (like C0BR4)
            if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
                score += 10
                
            # 5. Development bonus (like C0BR4)
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                # Development from back rank
                if (piece.color == chess.WHITE and chess.square_rank(move.from_square) == 0) or \
                   (piece.color == chess.BLACK and chess.square_rank(move.from_square) == 7):
                    score += 5
            
            scored_moves.append((move, score))
        
        # Sort and return top moves only
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in scored_moves[:self.fast_move_limit]]

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
        """Main search function - uses fast or full search based on configuration"""
        self.nodes_searched = 0
        self.sync_with_game_board(board)
        self.current_player = player

        # Check for immediate transposition table hit (DISABLED FOR DEBUGGING)
        search_depth = self.depth if self.depth is not None else 6
        # Ensure depth is even to always include opponent response
        if search_depth % 2 == 1:
            search_depth += 1
        # trans_move, trans_score = self.get_transposition_move(board, search_depth)
        # if trans_move and board.is_legal(trans_move):
        #     return trans_move

        # V6.2: Choose search algorithm based on configuration
        if self.use_fast_search:
            best_move, best_score = self._fast_search_root(board, search_depth)
        else:
            # Call traditional minimax search
            best_move, best_score = self._minimax_search_root(board, search_depth, -float('inf'), float('inf'), True, stop_callback)
        
        # Update transposition table with result
        if best_move:
            self.update_transposition_table(board, search_depth, best_move, best_score)
        
        # Fallback handling
        if best_move is None or best_move == chess.Move.null():
            legal_moves = list(board.legal_moves)
            if legal_moves:
                # Try fast move ordering for fallback
                ordered_moves = self._fast_move_ordering(board, legal_moves)
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

    def _calculate_time_allocation(self, time_control: dict, board: chess.Board) -> float:
        """
        V6.2 Aggressive time allocation - matches C0BR4's approach for better blitz performance
        """
        # Handle fixed time per move
        if 'movetime' in time_control:
            return time_control['movetime'] / 1000.0
            
        # Handle infinite time or depth-based
        if time_control.get('infinite') or time_control.get('depth'):
            return float('inf')
            
        # Get time remaining for current side
        if board.turn:  # White to move
            remaining_time = time_control.get('wtime', 120000) / 1000.0  # Convert to seconds
            increment = time_control.get('winc', 0) / 1000.0
        else:  # Black to move
            remaining_time = time_control.get('btime', 120000) / 1000.0
            increment = time_control.get('binc', 0) / 1000.0
            
        # Get moves to go (estimate if not provided)
        moves_to_go = time_control.get('movestogo', None)
        
        if moves_to_go:
            # Tournament time control with moves to go
            base_time = remaining_time / max(moves_to_go, 1)
            allocated_time = base_time + increment * 0.95  # Use almost all increment
        else:
            # Sudden death or increment-based - be much more aggressive than v6.1
            if remaining_time <= 60:  # Under 1 minute - very aggressive
                estimated_moves_left = max(10, 20 - len(board.move_stack) // 2)
                base_time = remaining_time / estimated_moves_left
                allocated_time = base_time + increment * 0.95  # Use almost all increment
            elif remaining_time <= 300:  # Under 5 minutes - aggressive
                estimated_moves_left = max(15, 30 - len(board.move_stack) // 2)
                base_time = remaining_time / estimated_moves_left
                allocated_time = base_time + increment * 0.9
            else:
                estimated_moves_left = max(20, 40 - len(board.move_stack) // 2)
                base_time = remaining_time / estimated_moves_left
                allocated_time = base_time + increment * 0.8
            
        # Position complexity modifier - use more time for complex positions
        legal_moves_count = len(list(board.legal_moves))
        if legal_moves_count > 35:
            allocated_time *= 1.2  # Reduced from 1.4 - don't overthink
        elif legal_moves_count < 10:
            allocated_time *= 0.8  # Simple position
            
        # Game phase modifier - less conservative than v6.1
        piece_count = len(board.piece_map())
        if piece_count > 20:  # Opening/early middlegame
            allocated_time *= 0.8  # Don't overthink opening
        elif piece_count < 10:  # Endgame
            allocated_time *= 1.1  # Precision matters but don't overdo it
            
        # Safety limits - much more aggressive than v6.1
        min_time = 0.05  # Reduced minimum time
        if remaining_time <= 180:  # Blitz games (3 minutes or less)
            max_time_ratio = 0.35  # Use up to 35% for blitz (increased from 25%)
        elif remaining_time <= 600:  # Rapid games (10 minutes or less)
            max_time_ratio = 0.25  # Use up to 25% for rapid (increased from 20%)
        else:
            max_time_ratio = 0.20  # Use up to 20% for longer games (increased from 15%)
            
        max_time = min(remaining_time * max_time_ratio, remaining_time - 1.0)
        
        allocated_time = max(min_time, min(allocated_time, max_time))
        
        # Emergency time handling - be much less conservative
        if remaining_time < 15.0:  # Under 15 seconds - very aggressive
            allocated_time = min(allocated_time, remaining_time * 0.25)  # Use 25% of remaining
        elif remaining_time < 30.0:  # Under 30 seconds
            allocated_time = min(allocated_time, remaining_time * 0.20)  # Use 20% of remaining
        elif remaining_time < 60.0:  # Under 1 minute
            allocated_time = min(allocated_time, remaining_time * 0.15)  # Use 15% of remaining
            
        return allocated_time

    def search_with_time_management(self, board: chess.Board, time_control: dict) -> Tuple[chess.Move, int, int, int]:
        """
        V6.1 Enhanced search with integrated time management and iterative deepening
        Returns: (best_move, final_depth, total_nodes, search_time_ms)
        """
        search_start = time.time()
        self.sync_with_game_board(board)
        self.current_player = board.turn
        
        # Calculate time allocation
        allocated_time = self._calculate_time_allocation(time_control, board)
        max_time = allocated_time * 1.1  # Allow 10% buffer
        
        # Initialize search variables
        best_move = None
        best_score = 0
        total_nodes = 0
        final_depth = 0
        previous_best = None
        
        # Quick fallback - get any legal move immediately
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null(), 0, 0, 0
        
        # Get initial move from move ordering
        tt_move, _ = self.get_transposition_move(board, 1)
        if self.use_fast_search:
            ordered_moves = self._fast_move_ordering(board, legal_moves)
        else:
            ordered_moves = self.order_moves(board, legal_moves, hash_move=tt_move, depth=1)
        fallback_move = ordered_moves[0] if ordered_moves else legal_moves[0]
        best_move = fallback_move
        
        # Iterative deepening with time management
        max_depth = 7 if self.use_fast_search else 8  # Limit depth for fast search
        
        for depth in range(1, max_depth + 1):
            iteration_start = time.time()
            elapsed = iteration_start - search_start
            
            # Hard cutoff - don't start new iteration if 70% time used (more aggressive)
            if elapsed >= allocated_time * 0.70 and depth > 1:
                break  
                
            # Check depth-based stopping
            if 'depth' in time_control and depth > time_control['depth']:
                break
                
            # Reset nodes for this iteration
            self.nodes_searched = 0
            self.depth = depth
            
            # V6.2: Use fast or traditional search based on configuration
            if self.use_fast_search:
                move, score = self._fast_search_root(board, depth)
            else:
                # Search with aspiration window if we have previous score
                if depth > 2 and previous_best:
                    # Try narrow window first
                    window = 50  # centipawns
                    alpha = best_score - window
                    beta = best_score + window
                    
                    move, score = self._minimax_search_root(
                        board, depth, alpha, beta, True, 
                        lambda: time.time() - search_start >= max_time
                    )
                    
                    # If search failed (fell outside window), search with full window
                    if score <= alpha or score >= beta:
                        move, score = self._minimax_search_root(
                            board, depth, -float('inf'), float('inf'), True,
                            lambda: time.time() - search_start >= max_time
                        )
                else:
                    # Full window search for early depths
                    move, score = self._minimax_search_root(
                        board, depth, -float('inf'), float('inf'), True,
                        lambda: time.time() - search_start >= max_time
                    )
            
            # Update results if we got a valid move
            if move and move != chess.Move.null():
                best_move = move
                best_score = score
                previous_best = move
                final_depth = depth
                total_nodes += self.nodes_searched
                
                # Update transposition table
                self.update_transposition_table(board, depth, move, score)
            
            # Check time limits - more aggressive cutoff
            elapsed = time.time() - search_start
            if elapsed >= allocated_time * 0.85:  # Hard cutoff at 85% instead of 100%
                break
                
            # For movetime, be stricter about time limits
            if 'movetime' in time_control and elapsed >= (time_control['movetime'] / 1000.0) * 0.85:
                break
        
        # Ensure we have a valid move
        if not best_move or best_move == chess.Move.null():
            best_move = fallback_move
            
        # Apply final safety checks
        best_move = self._enforce_strict_draw_prevention(board, best_move)
        if best_move and not board.is_legal(best_move):
            best_move = fallback_move
            
        # Final fallback to ensure we always return a valid move
        if not best_move or not board.is_legal(best_move):
            best_move = legal_moves[0]
            
        search_time_ms = int((time.time() - search_start) * 1000)
        return best_move, final_depth, total_nodes, search_time_ms

    def _minimax_search_root(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool, stop_callback: Optional[Callable[[], bool]] = None) -> Tuple[Optional[chess.Move], float]:
        """Root level minimax search that returns both move and score"""
        self.nodes_searched += 1
        
        if stop_callback and stop_callback():
            return None, self.evaluate_position_from_perspective(board, self.current_player)

        # Terminal conditions
        if depth <= 0 or board.is_game_over(claim_draw=self._is_draw_condition(board)):
            position_score = self.evaluate_position_from_perspective(board, self.current_player)
            return None, position_score

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

    # =================================
    # ===== EVALUATION FUNCTIONS ======

    def evaluate_position(self, board: chess.Board) -> float:
        """Calculate base position evaluation by delegating to scoring_calculator."""
        positional_evaluation_board = board.copy()
        if not isinstance(positional_evaluation_board, chess.Board) or not positional_evaluation_board.is_valid():
            return 0.0

        endgame_factor = self._get_game_phase_factor(positional_evaluation_board)
        
        # Use optimized scoring if enabled
        if self.use_optimized_scoring:
            score = self.scoring_calculator.calculate_score_optimized(
                board=positional_evaluation_board,
                color=chess.WHITE,
                endgame_factor=endgame_factor
            ) - self.scoring_calculator.calculate_score_optimized(
                board=positional_evaluation_board,
                color=chess.BLACK,
                endgame_factor=endgame_factor
            )
        else:
            score = self.scoring_calculator.calculate_score_optimized(
                board=positional_evaluation_board,
                color=chess.WHITE,
                endgame_factor=endgame_factor
            ) - self.scoring_calculator.calculate_score_optimized(
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

        # Use optimized scoring if enabled
        if self.use_optimized_scoring:
            white_score = self.scoring_calculator.calculate_score_optimized(
                board=perspective_evaluation_board,
                color=chess.WHITE,
                endgame_factor=endgame_factor
            )
            black_score = self.scoring_calculator.calculate_score_optimized(
                board=perspective_evaluation_board,
                color=chess.BLACK,
                endgame_factor=endgame_factor
            )
        else:
            white_score = self.scoring_calculator.calculate_score_optimized(
                board=perspective_evaluation_board,
                color=chess.WHITE,
                endgame_factor=endgame_factor
            )
            black_score = self.scoring_calculator.calculate_score_optimized(
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
        """C0BR4-style move scoring - simple and efficient."""
        score = 0.0

        # 1. Captures (MVV-LVA - like C0BR4)
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                victim_value = self.piece_values.get(victim.piece_type, 0)
                attacker_value = self.piece_values.get(attacker.piece_type, 0)
                score += 10000 + (victim_value - attacker_value)

        # 2. Promotions (like C0BR4)
        if move.promotion:
            score += 9000 + self.piece_values.get(move.promotion, 0)

        # 3. Checks (like C0BR4)
        board.push(move)
        if board.is_check():
            score += 500
        board.pop()

        # 4. Center control (like C0BR4)
        if move.to_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
            score += 10

        # 5. Development bonus (like C0BR4)
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            # Development from back rank
            if (piece.color == chess.WHITE and chess.square_rank(move.from_square) == 0) or \
               (piece.color == chess.BLACK and chess.square_rank(move.from_square) == 7):
                score += 5

        return score
    

    
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

    def _calculate_queen_attack_bonus(self, board: chess.Board, move: chess.Move) -> float:
        """
        Calculate bonus for moves that attack the enemy queen with defended pieces.
        High priority for creating queen traps and tactical pressure.
        """
        bonus = 0.0
        
        # Find the enemy queen
        enemy_color = not board.turn
        enemy_queen_square = None
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.QUEEN and piece.color == enemy_color:
                enemy_queen_square = square
                break
        
        if enemy_queen_square is None:
            return 0.0  # No enemy queen on board
        
        # Check if this move attacks the enemy queen
        temp_board = board.copy()
        temp_board.push(move)
        
        # Check if our piece now attacks the enemy queen square
        if temp_board.is_attacked_by(board.turn, enemy_queen_square):
            # Verify the attacking piece is the one we just moved
            attacking_piece = temp_board.piece_at(move.to_square)
            if attacking_piece:
                # High bonus for attacking queen with defended piece
                base_bonus = 50000.0  # High priority but less than captures/checks
                
                # Check if our attacking piece is defended
                if temp_board.is_attacked_by(board.turn, move.to_square):
                    bonus += base_bonus
                    
                    # Extra bonus for attacking with less valuable pieces
                    piece_value = self.piece_values.get(attacking_piece.piece_type, 0)
                    if piece_value <= 3.0:  # Knights, Bishops, pawns
                        bonus += 10000.0  # Extra bonus for attacking with minor pieces
                    
                    # Check if enemy queen has limited escape squares (potential trap)
                    escape_squares = 0
                    for escape_square in chess.SQUARES:
                        if abs(chess.square_rank(escape_square) - chess.square_rank(enemy_queen_square)) <= 1 and \
                           abs(chess.square_file(escape_square) - chess.square_file(enemy_queen_square)) <= 1:
                            if escape_square != enemy_queen_square:
                                temp_test_board = temp_board.copy()
                                # Simulate queen moving to escape square
                                try:
                                    escape_move = chess.Move(enemy_queen_square, escape_square)
                                    if escape_move in temp_test_board.legal_moves:
                                        temp_test_board.push(escape_move)
                                        if not temp_test_board.is_attacked_by(board.turn, escape_square):
                                            escape_squares += 1
                                        temp_test_board.pop()
                                except:
                                    pass
                    
                    # Extra bonus if queen has few escape squares (potential trap)
                    if escape_squares <= 2:
                        bonus += 15000.0  # Trap bonus
                else:
                    # Smaller bonus for undefended attacks (risky)
                    bonus += base_bonus * 0.3
        
        temp_board.pop()
        return bonus

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

        # Terminal node - evaluate position directly
        if depth <= 0 or board.is_game_over(claim_draw=self._is_draw_condition(board)):
            return self.evaluate_position_from_perspective(board, self.current_player)

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
