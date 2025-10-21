#!/usr/bin/env python3
"""
VPR Chess Engine v5.0 - Calculated Outcomes Bitboard Implementation

Core Philosophy:
"We look at pieces with the most potential and pieces with the least potential. This isn't about assumptions and moves but about calculations and pieces."

Revolutionary v5.0 Features:
- Static Exchange Evaluation (SEE) for precise tactical analysis ✓
- Tactical move priority, with compounding attacks, adding to chaos effect ✓
- Improved speed performance with stricter SEE based move ordering ✓
- Lightning fast caching enhancements for search speed ✓
- Bitboard flash-layer comparisons for ultra-fast piece potential calculations ✓

"Chaos requires control, SEE gives us that control, overwhelming the opponent with tactical threats."

Author: Pat Snyder 
Version: VPR Calculated Outcomes v5.0
"""

import chess
import time
from typing import Optional, List, Dict, Tuple
import random

class VPREngine:
    """
    VPR v5.0 Calculated Outcomes Bitboard Engine
    """
    
    # Class attributes for type hinting
    chaos_move_threshold: int
    
    def __init__(self):
        self.nodes_searched = 0
        self.search_start_time = 0.0
        self.board = chess.Board()
        
        # Bitboard calculation cache for lightning speed
        self.bitboard_cache = {}
        self.potential_cache = {}
        
        # UCI configurable options
        self.chaos_move_threshold = 100  # Default chaos threshold for move selection
        
        # Pre-computed attack patterns (bitboards) - Calculated Outcomes lookups
        self._init_attack_patterns()
    
    def _init_attack_patterns(self):
        """Initialize pre-computed attack patterns for ultra-fast potential calculation"""
        # Knight attack patterns for each square (pre-computed bitboards)
        self.knight_attacks = {}
        for square in chess.SQUARES:
            attacks = 0
            row, col = divmod(square, 8)
            
            # All 8 possible knight moves
            knight_moves = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
            for dr, dc in knight_moves:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target_square = new_row * 8 + new_col
                    attacks |= (1 << target_square)
            
            self.knight_attacks[square] = attacks
        
        # King attack patterns 
        self.king_attacks = {}
        for square in chess.SQUARES:
            attacks = 0
            row, col = divmod(square, 8)
            
            # All 8 possible king moves
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 8 and 0 <= new_col < 8:
                        target_square = new_row * 8 + new_col
                        attacks |= (1 << target_square)
            
            self.king_attacks[square] = attacks
    
    def _static_exchange_evaluation(self, board: chess.Board, move: chess.Move) -> int:
        """
        VPR v5.0 Revolutionary SEE - Using CALCULATED piece potential, not static values!
        
        True to VPR philosophy: "A piece is as valuable as the attacking potential it has"
        Calculates exchange outcomes using dynamic piece potential rather than material.
        """
        if not board.is_capture(move):
            return 0
        
        target_square = move.to_square
        captured_piece = board.piece_at(target_square)
        
        if not captured_piece:
            return 0
        
        # VPR Innovation: Use calculated piece potential as "value"
        captured_potential = self._calculate_piece_potential_bitboard(board, target_square)
        
        # Get all attackers of the target square for both sides
        attackers = self._get_square_attackers(board, target_square)
        
        # Simulate the exchange sequence
        board_copy = board.copy()
        board_copy.push(move)
        
        # Get the piece that just moved to the target square
        attacking_piece = board_copy.piece_at(target_square)
        if not attacking_piece:
            return captured_potential
        
        # Recursively calculate the best response using potential-based values
        best_response = 0
        opponent_attackers = []
        
        for sq in attackers:
            piece_at_sq = board_copy.piece_at(sq)
            if piece_at_sq and piece_at_sq.color != attacking_piece.color:
                opponent_attackers.append(sq)
        
        if opponent_attackers:
            # Find the least valuable attacker using VPR potential calculation
            def get_piece_potential_value(sq):
                return self._calculate_piece_potential_bitboard(board_copy, sq)
            
            # Use LOWEST potential piece for counter-attack (most expendable)
            least_valuable_attacker = min(opponent_attackers, key=get_piece_potential_value)
            
            # Create the counter-capture move
            counter_move = chess.Move(least_valuable_attacker, target_square)
            if counter_move in board_copy.legal_moves:
                # Calculate attacking piece potential in new position
                attacking_piece_potential = self._calculate_piece_potential_bitboard(board_copy, target_square)
                
                # Recursively evaluate the counter-capture using potential values
                best_response = attacking_piece_potential - \
                               self._static_exchange_evaluation(board_copy, counter_move)
        
        return captured_potential - max(0, best_response)
    
    def _get_square_attackers(self, board: chess.Board, square: chess.Square) -> List[chess.Square]:
        """Get all pieces attacking a specific square using bitboard magic"""
        attackers = []
        
        # Check all squares for pieces that can attack the target
        for from_square in chess.SQUARES:
            piece = board.piece_at(from_square)
            if piece and chess.Move(from_square, square) in board.legal_moves:
                attackers.append(from_square)
        
        return attackers
        
    def search(self, board: chess.Board, time_limit: float = 3.0, 
               depth: Optional[int] = None) -> chess.Move:
        """
        VPR v5.0 Lightning search: Bitboard-based piece potential
        """
        self.nodes_searched = 0
        self.search_start_time = time.time()
        self.bitboard_cache.clear()
        self.potential_cache.clear()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        target_time = time_limit * 0.9
        max_depth = depth if depth else 25
        
        best_move = legal_moves[0]
        
        # Get SEE-ordered tactical moves with potential focus (VPR v5.0 innovation)
        focus_moves = self._get_see_ordered_tactical_moves(board)
        
        # Iterative deepening
        for current_depth in range(1, max_depth + 1):
            elapsed = time.time() - self.search_start_time
            if elapsed > target_time:
                break
            
            current_best = best_move
            current_score = -999999
            
            for move in focus_moves:
                elapsed = time.time() - self.search_start_time
                if elapsed > target_time:
                    break
                
                board.push(move)
                move_score = -self._potential_negamax(board, current_depth - 1, -999999, 999999, target_time)
                board.pop()
                
                if move_score > current_score:
                    current_score = move_score
                    current_best = move
            
            if elapsed <= target_time:
                best_move = current_best
                
                elapsed_ms = int(elapsed * 1000)
                nps = int(self.nodes_searched / elapsed) if elapsed > 0 else 0
                print(f"info depth {current_depth} score cp {int(current_score)} "
                      f"nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {best_move}")
        
        return best_move
    
    def _get_see_ordered_tactical_moves(self, board: chess.Board) -> List[chess.Move]:
        """
        VPR v5.0 Revolutionary Enhancement: SEE-based tactical move ordering
        
        Combines piece potential focus with Static Exchange Evaluation for 
        precise tactical control. "Chaos requires control, SEE gives us that control."
        """
        legal_moves = list(board.legal_moves)
        
        # Separate moves by type for tactical prioritization
        captures_with_see = []
        potential_focus_moves = []
        other_moves = []
        
        # Get potential-focused pieces first
        piece_potentials = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                potential = self._calculate_piece_potential_bitboard(board, square)
                piece_potentials.append((potential, square))
        
        if not piece_potentials:
            return legal_moves
        
        # Sort by potential (highest to lowest)
        piece_potentials.sort(reverse=True, key=lambda x: x[0])
        
        # Focus on TOP pieces (attacks) and BOTTOM pieces (activation)
        num_pieces = len(piece_potentials)
        focus_squares = set()
        
        if num_pieces <= 4:
            focus_squares = {sq for _, sq in piece_potentials}
        else:
            # Top 2 and bottom 2 pieces only - ignore the middle
            top_pieces = piece_potentials[:2]
            bottom_pieces = piece_potentials[-2:]
            focus_squares = {sq for _, sq in top_pieces + bottom_pieces}
        
        # Categorize and evaluate moves
        for move in legal_moves:
            if board.is_capture(move):
                # Captures: Evaluate with SEE for tactical precision
                see_value = self._static_exchange_evaluation(board, move)
                captures_with_see.append((see_value, move))
            elif move.from_square in focus_squares:
                # Potential-focused moves from high/low potential pieces
                potential_focus_moves.append(move)
            else:
                # Other moves
                other_moves.append(move)
        
        # Sort captures by SEE value (best first) - tactical priority
        captures_with_see.sort(reverse=True, key=lambda x: x[0])
        winning_captures = [move for see_val, move in captures_with_see if see_val > 0]
        neutral_captures = [move for see_val, move in captures_with_see if see_val == 0]
        losing_captures = [move for see_val, move in captures_with_see if see_val < 0]
        
        # VPR v5.0 Tactical Move Priority:
        # 1. Winning captures (positive SEE) - immediate tactical advantage
        # 2. Potential-focused moves - piece activation and positioning
        # 3. Neutral captures (SEE = 0) - maintaining material balance
        # 4. Other moves - general development
        # 5. Losing captures (negative SEE) - desperation/chaos moves
        ordered_moves = (winning_captures + potential_focus_moves + 
                        neutral_captures + other_moves + losing_captures)
        
        return ordered_moves if ordered_moves else legal_moves
    
    def _get_potential_focus_moves(self, board: chess.Board) -> List[chess.Move]:
        """
        Core VPR Innovation: Focus ONLY on highest and lowest potential pieces
        """
        piece_potentials = []
        
        # Calculate true potential for all our pieces using LIGHTNING-FAST bitboards
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                potential = self._calculate_piece_potential_bitboard(board, square)
                piece_potentials.append((potential, square))
        
        if not piece_potentials:
            return list(board.legal_moves)
        
        # Sort by potential (highest to lowest)
        piece_potentials.sort(reverse=True, key=lambda x: x[0])
        
        # Focus on TOP pieces (attacks) and BOTTOM pieces (activation)
        num_pieces = len(piece_potentials)
        focus_squares = set()
        
        if num_pieces <= 4:
            focus_squares = {sq for _, sq in piece_potentials}
        else:
            # Top 2 and bottom 2 pieces only - ignore the middle
            top_pieces = piece_potentials[:2]
            bottom_pieces = piece_potentials[-2:]
            focus_squares = {sq for _, sq in top_pieces + bottom_pieces}
        
        # Get moves from focus pieces
        focus_moves = [move for move in board.legal_moves if move.from_square in focus_squares]
        return focus_moves if focus_moves else list(board.legal_moves)
    
    def _calculate_piece_potential_bitboard(self, board: chess.Board, square: chess.Square) -> int:
        """
        VPR v5.0 Calculated Outcomes bitboard-based piece potential calculation
        
        Uses bitboard flash-layer comparisons for instant analysis:
        - Attack bitboards for instant attack counting
        - Occupancy masking for blocked path detection  
        - Piece-by-piece approach with blazing speed
        """
        piece = board.piece_at(square)
        if not piece:
            return 0
        
        # Cache key for this position
        board_hash = hash(board.fen().split()[0])  # Just piece positions
        cache_key = (board_hash, square)
        
        if cache_key in self.potential_cache:
            return self.potential_cache[cache_key]
        
        potential = 0
        piece_type = piece.piece_type
        
        # Get bitboard representations for lightning-fast operations
        occupied = board.occupied
        our_pieces = board.occupied_co[piece.color]
        enemy_pieces = board.occupied_co[not piece.color]
        
        if piece_type == chess.PAWN:
            potential = self._calculate_pawn_potential_bitboard(board, square, piece.color, occupied, enemy_pieces)
        
        elif piece_type == chess.KNIGHT:
            # Lightning-fast pre-computed knight attacks
            attack_bitboard = self.knight_attacks[square]
            
            # Attacks = squares we attack (not blocked by our pieces)
            attacks = attack_bitboard & ~our_pieces
            potential += bin(attacks).count('1')
            
            # Mobility = squares we can move to (not occupied by our pieces)
            mobility = attack_bitboard & ~our_pieces
            potential += bin(mobility).count('1')
        
        elif piece_type == chess.BISHOP:
            potential = self._calculate_sliding_potential_bitboard(board, square, occupied, our_pieces, enemy_pieces, 'bishop')
        
        elif piece_type == chess.ROOK:
            potential = self._calculate_sliding_potential_bitboard(board, square, occupied, our_pieces, enemy_pieces, 'rook')
        
        elif piece_type == chess.QUEEN:
            # Queen = rook + bishop potential
            rook_potential = self._calculate_sliding_potential_bitboard(board, square, occupied, our_pieces, enemy_pieces, 'rook')
            bishop_potential = self._calculate_sliding_potential_bitboard(board, square, occupied, our_pieces, enemy_pieces, 'bishop')
            potential = rook_potential + bishop_potential
        
        elif piece_type == chess.KING:
            # Lightning-fast pre-computed king attacks
            attack_bitboard = self.king_attacks[square]
            
            # King potential = safe squares it can move to
            safe_squares = attack_bitboard & ~our_pieces
            potential += bin(safe_squares).count('1')
        
        # Cache for speed and return
        self.potential_cache[cache_key] = potential
        return potential
    
    def _calculate_pawn_potential_bitboard(self, board: chess.Board, square: chess.Square, 
                                           color: chess.Color, occupied: int, enemy_pieces: int) -> int:
        """Lightning-fast pawn potential using bitboards"""
        potential = 0
        row, col = divmod(square, 8)
        
        if color == chess.WHITE:
            # White pawn attacks (diagonal)
            if row < 7:  # Not on 8th rank
                if col > 0:  # Can attack left
                    target = (row + 1) * 8 + (col - 1)
                    if enemy_pieces & (1 << target):
                        potential += 2  # Can capture
                    else:
                        potential += 1  # Controls square
                
                if col < 7:  # Can attack right
                    target = (row + 1) * 8 + (col + 1)
                    if enemy_pieces & (1 << target):
                        potential += 2  # Can capture
                    else:
                        potential += 1  # Controls square
                
                # Pawn push potential
                target = (row + 1) * 8 + col
                if not (occupied & (1 << target)):
                    potential += 1  # Can move forward
                    
                    # Double push from starting position
                    if row == 1 and not (occupied & (1 << ((row + 2) * 8 + col))):
                        potential += 1
        
        else:  # Black pawn
            if row > 0:  # Not on 1st rank
                if col > 0:  # Can attack left
                    target = (row - 1) * 8 + (col - 1)
                    if enemy_pieces & (1 << target):
                        potential += 2  # Can capture
                    else:
                        potential += 1  # Controls square
                
                if col < 7:  # Can attack right
                    target = (row - 1) * 8 + (col + 1)
                    if enemy_pieces & (1 << target):
                        potential += 2  # Can capture
                    else:
                        potential += 1  # Controls square
                
                # Pawn push potential
                target = (row - 1) * 8 + col
                if not (occupied & (1 << target)):
                    potential += 1  # Can move forward
                    
                    # Double push from starting position
                    if row == 6 and not (occupied & (1 << ((row - 2) * 8 + col))):
                        potential += 1
        
        return potential
    
    def _calculate_sliding_potential_bitboard(self, board: chess.Board, square: chess.Square,
                                              occupied: int, our_pieces: int, enemy_pieces: int,
                                              piece_type: str) -> int:
        """Lightning-fast sliding piece potential (rook/bishop) using bitboards"""
        potential = 0
        row, col = divmod(square, 8)
        
        if piece_type == 'rook':
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Horizontal/vertical
        else:  # bishop
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # Diagonal
        
        for dr, dc in directions:
            # Slide in this direction until blocked
            current_row, current_col = row + dr, col + dc
            
            while 0 <= current_row < 8 and 0 <= current_col < 8:
                target_square = current_row * 8 + current_col
                target_bit = 1 << target_square
                
                if our_pieces & target_bit:
                    # Blocked by our own piece - stop here
                    break
                
                if enemy_pieces & target_bit:
                    # Can attack enemy piece
                    potential += 2  # Attack + potential capture
                    break  # Stop after enemy piece
                
                # Empty square - can move here and attack through
                potential += 1
                current_row += dr
                current_col += dc
        
        return potential
    
    def _potential_negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, time_limit: float) -> float:
        """
        VPR negamax with chaos preservation and imperfect play assumptions
        """
        self.nodes_searched += 1
        
        # Periodic time check for speed
        if self.nodes_searched % 100 == 0:
            elapsed = time.time() - self.search_start_time
            if elapsed > time_limit:
                return 0
        
        # Terminal conditions
        if depth <= 0:
            return self._evaluate_pure_potential_bitboard(board)
        
        if board.is_checkmate():
            return -999999
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Fast chaos detection
        legal_count = len(list(board.legal_moves))
        is_chaotic = legal_count > 100  # Chaos threshold
        
        max_eval = -999999
        
        # VPR v5.0 SEE-enhanced move ordering for tactical precision
        legal_moves = list(board.legal_moves)
        
        # Quick tactical move ordering with SEE
        captures_with_see = []
        non_captures = []
        
        for move in legal_moves:
            if board.is_capture(move):
                see_value = self._static_exchange_evaluation(board, move)
                captures_with_see.append((see_value, move))
            else:
                non_captures.append(move)
        
        # Sort captures by SEE value (winning captures first)
        captures_with_see.sort(reverse=True, key=lambda x: x[0])
        winning_captures = [move for see_val, move in captures_with_see if see_val > 0]
        other_captures = [move for see_val, move in captures_with_see if see_val <= 0]
        
        # Move ordering: winning captures, non-captures, other captures
        ordered_moves = winning_captures + non_captures + other_captures
        
        for move in ordered_moves:
            board.push(move)
            eval_score = -self._potential_negamax(board, depth - 1, -beta, -alpha, time_limit)
            board.pop()
            
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            
            # LENIENT PRUNING: Don't prune chaotic positions
            if not is_chaotic and beta <= alpha:
                break
        
        return max_eval
    
    def _evaluate_pure_potential_bitboard(self, board: chess.Board) -> float:
        """
        VPR v5.0 LIGHTNING evaluation using bitboards
        
        Flash-layer bitboard comparisons for instant position analysis
        """
        our_potential = 0
        their_potential = 0
        
        # LIGHTNING-FAST bitboard-based potential calculation
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                potential = self._calculate_piece_potential_bitboard(board, square)
                
                if piece.color == board.turn:
                    our_potential += potential
                else:
                    their_potential += potential
        
        # Potential difference is the score
        score = our_potential - their_potential
        
        # VPR v5.0 Tactical Enhancement: Potential-based tactical bonus
        tactical_bonus = 0
        for move in board.legal_moves:
            if board.is_capture(move):
                see_value = self._static_exchange_evaluation(board, move)
                if see_value > 0:
                    # Bonus based on potential gained, not static material
                    tactical_bonus += min(see_value // 10, 15)  # Scale potential bonus appropriately
        
        score += tactical_bonus
        
        # Chaos bonus (we thrive in complexity) - fast legal move count
        legal_moves = list(board.legal_moves)
        legal_count = len(legal_moves)
        
        if legal_count > self.chaos_move_threshold:
            score += 25  # Chaos bonus
        
        # Imperfection factor (opponent won't play perfectly)
        imperfection = random.randint(-2, 2)
        score += imperfection
        
        return score
    
    def new_game(self):
        """Reset engine state for a new game"""
        self.nodes_searched = 0
        self.bitboard_cache.clear()
        self.potential_cache.clear()
    
    def get_engine_info(self) -> dict:
        """Return engine information"""
        return {
            'name': 'VPR Calculated Outcomes',
            'version': '5.0', 
            'author': 'Pat Snyder',
            'description': 'Bitboard-based Calculated Outcomes piece potential engine',
            'nodes_searched': self.nodes_searched
        }

# Engine can be imported and used by UCI interface
# For direct UCI communication, use vpr_uci.py instead