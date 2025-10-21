#!/usr/bin/env python3
"""
TAL-BOT Chess Engine - Revolutionary Entropy-Driven Chess AI

An experimental anti-engine based on Mikhail Tal's sacrificial attacking style.
Uses dynamic piece values and chaos-driven search to destroy traditional engines.

THE 2+2=5 PRINCIPLE: Position >> Material
- True piece values based on attacks + legal moves (not static assumptions)
- Chaos factor preservation over safe evaluation
- Priority-based move ordering (best/worst pieces only)
- Entropy warfare: force opponents into complex positions

REVOLUTIONARY FEATURES:
- Dynamic piece value system (true positional worth)
- Chaos factor calculation and preservation
- Priority piece selection (ignore mediocre pieces)
- Principal Variation following for bullet speed
- Anti-engine logic that breaks traditional patterns

TAL-BOT MOTTO: "Into the dark forest where only we know the way out"

Author: Pat Snyder
Version: TAL-BOT v1.0 (The Entropy Engine)
"""

import time
import chess
from typing import Optional, List


class VPREngine:
    """VPR - Barebones chess engine optimized for maximum search depth"""
    
    def __init__(self):
        # Basic piece values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        # Simple piece-square tables for positional awareness
        # Values are from white's perspective, will be flipped for black
        self._init_piece_square_tables()
        
        # Search configuration
        self.default_depth = 8  # Target deeper search
        self.nodes_searched = 0
        self.search_start_time = 0
        
        # TAL-BOT PV Following System
        self.principal_variation = []  # Current PV line
        self.pv_board_states = []      # Board states for PV following
        self.last_search_pv = []       # Last complete PV for instant moves
        self.pv_move_times = []        # Time taken for each PV move
        
    def _init_piece_square_tables(self):
        """Initialize basic piece-square tables for positional evaluation"""
        
        # Pawn table - encourage central pawns and advancement
        self.pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        # Knight table - prefer center, avoid edges
        self.knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
        
        # Bishop table - prefer long diagonals
        self.bishop_table = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]
        
        # Rook table - prefer open files and 7th rank
        self.rook_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ]
        
        # Queen table - slight center preference
        self.queen_table = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]
        
        # King middlegame table - stay safe, prefer castled position
        self.king_table = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]
    
    def search(self, board: chess.Board, time_limit: float = 3.0, 
               depth: Optional[int] = None) -> chess.Move:
        """
        TAL-BOT main search with PV collection and following
        
        Args:
            board: Current position
            time_limit: Time limit in seconds
            depth: Optional fixed depth (otherwise uses default)
            
        Returns:
            Best move found
        """
        self.nodes_searched = 0
        self.search_start_time = time.time()
        
        # Check for PV following opportunity (instant move)
        instant_move = self._check_pv_following(board)
        if instant_move:
            print(f"info string PV following: instant move {instant_move}")
            return instant_move
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Calculate time allocation (use 80% of available time)
        target_time = time_limit * 0.8
        
        # Iterative deepening with PV collection
        max_depth = depth if depth else self.default_depth
        best_move = legal_moves[0]
        best_score = -999999
        self.principal_variation = []
        
        for current_depth in range(1, max_depth + 1):
            iteration_start = time.time()
            elapsed = time.time() - self.search_start_time
            
            # Stop if we're running out of time
            if elapsed > target_time:
                break
            
            # Search at current depth with PV collection
            current_pv = []
            score = -999999
            current_best = best_move
            time_exceeded = False
            
            for move in legal_moves:
                # Check time before starting move search
                elapsed = time.time() - self.search_start_time
                if elapsed > target_time:
                    time_exceeded = True
                    break
                
                board.push(move)
                
                # Negamax search with PV collection
                move_pv = []
                move_score = -self._negamax_with_pv(board, current_depth - 1, -999999, 999999, target_time, move_pv)
                
                board.pop()
                
                if move_score > score:
                    score = move_score
                    current_best = move
                    current_pv = [move] + move_pv  # Build full PV
            
            # Update best move and PV only if we completed this depth
            if not time_exceeded:
                best_move = current_best
                best_score = score
                self.principal_variation = current_pv[:]  # Store current PV
                
                # UCI info output with full PV
                elapsed = time.time() - self.search_start_time
                nps = int(self.nodes_searched / elapsed) if elapsed > 0 else 0
                
                # Format PV for UCI
                pv_string = " ".join(str(move) for move in current_pv[:10])  # Show first 10 moves
                
                print(f"info depth {current_depth} score cp {int(score)} "
                      f"nodes {self.nodes_searched} time {int(elapsed * 1000)} "
                      f"nps {nps} pv {pv_string}")
        
        # Store final PV for potential following
        self.last_search_pv = self.principal_variation[:]
        self._store_pv_board_states(board)
        
        return best_move
    
    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, target_time: float) -> float:
        """
        TAL-BOT negamax: chaos-driven search with conditional pruning
        
        Revolutionary change: Preserve chaotic positions even if they look "bad"
        
        Args:
            board: Current position
            depth: Remaining depth to search
            alpha: Alpha bound
            beta: Beta bound
            target_time: Target time limit for search
            
        Returns:
            Evaluation score from current player's perspective
        """
        self.nodes_searched += 1
        
        # Check time every 1000 nodes
        if self.nodes_searched % 1000 == 0:
            elapsed = time.time() - self.search_start_time
            if elapsed > target_time:
                return 0  # Return neutral score if time exceeded
        
        # Terminal conditions
        if depth <= 0:
            return self._evaluate_position(board)
        
        if board.is_game_over():
            if board.is_checkmate():
                return -900000 + (self.default_depth - depth)  # Prefer faster mates
            return 0  # Draw
        
        # Calculate chaos factor for this position
        chaos_factor = self._calculate_chaos_factor(board)
        
        # Move generation and ordering (using TAL-BOT priority system)
        legal_moves = list(board.legal_moves)
        ordered_moves = self._order_moves_simple(board, legal_moves)
        
        best_score = -999999
        moves_searched = 0
        
        for move in ordered_moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, target_time)
            board.pop()
            
            moves_searched += 1
            
            # TAL-BOT conditional pruning logic
            if score > best_score:
                best_score = score
            
            if score > alpha:
                alpha = score
            
            # Revolutionary chaos-driven pruning
            if alpha >= beta:
                # Standard alpha-beta cutoff - but check chaos compensation first
                material_balance = self._quick_material_balance(board)
                
                # If position is chaotic enough, delay pruning to explore more
                if chaos_factor >= 50 and moves_searched < 5:
                    continue  # Keep searching in chaotic positions
                
                break  # Standard beta cutoff
        
        # Chaos bonus: reward keeping chaotic lines alive
        if chaos_factor >= 75:
            best_score += int(chaos_factor * 0.5)  # Chaos compensation
        
        return best_score
    
    def _quick_material_balance(self, board: chess.Board) -> int:
        """
        Quick material balance calculation for pruning decisions
        
        Args:
            board: Current position
            
        Returns:
            Material balance from current player's perspective
        """
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        if board.turn == chess.WHITE:
            return white_material - black_material
        else:
            return black_material - white_material
    
    def _order_moves_simple(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """
        TAL-BOT move ordering: priority pieces first, then tactical considerations
        
        Revolutionary approach:
        1. Moves from highest true-value pieces (tactical opportunities)
        2. Moves from lowest true-value pieces (activation potential)  
        3. Captures (MVV-LVA)
        4. Other moves
        
        Args:
            board: Current position
            moves: List of legal moves
            
        Returns:
            Ordered list of moves (chaos-optimized)
        """
        if len(moves) <= 2:
            return moves
        
        # Get priority pieces for current player
        high_priority, low_priority = self._get_priority_pieces(board, board.turn)
        
        priority_moves = []
        captures = []
        quiet_moves = []
        
        for move in moves:
            from_square = move.from_square
            
            # Priority 1: Moves from high-value pieces (tactical opportunities)
            if from_square in high_priority:
                # Calculate move value based on piece's true value
                true_value = self._calculate_piece_true_value(board, from_square, board.turn)
                
                if board.is_capture(move):
                    # High-priority capture - maximum priority
                    victim = board.piece_at(move.to_square)
                    victim_value = self.piece_values.get(victim.piece_type, 0) if victim else 0
                    score = (true_value * 100) + victim_value
                    priority_moves.append((score + 1000, move))  # Massive priority boost
                else:
                    # High-priority quiet move
                    score = true_value * 100
                    priority_moves.append((score + 500, move))  # High priority boost
            
            # Priority 2: Moves from low-value pieces (activation potential)
            elif from_square in low_priority:
                true_value = self._calculate_piece_true_value(board, from_square, board.turn)
                if true_value <= 2:  # Truly inactive pieces
                    score = 100 - true_value  # Lower true value = higher activation priority
                    priority_moves.append((score + 300, move))  # Activation priority boost
                else:
                    quiet_moves.append(move)
            
            # Standard tactical moves
            elif board.is_capture(move):
                # MVV-LVA for non-priority captures
                victim = board.piece_at(move.to_square)
                victim_value = self.piece_values.get(victim.piece_type, 0) if victim else 0
                
                attacker = board.piece_at(move.from_square)
                attacker_value = self.piece_values.get(attacker.piece_type, 0) if attacker else 0
                
                score = victim_value * 100 - attacker_value
                captures.append((score, move))
            
            else:
                quiet_moves.append(move)
        
        # Sort priority moves by score (highest first)
        priority_moves.sort(key=lambda x: x[0], reverse=True)
        
        # Sort captures by MVV-LVA score
        captures.sort(key=lambda x: x[0], reverse=True)
        
        # Return ordered moves: Priority → Captures → Quiet
        result = [move for _, move in priority_moves]
        result += [move for _, move in captures]
        result += quiet_moves
        
        return result
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """
        TAL-BOT evaluation: dynamic piece values + chaos factor
        
        Args:
            board: Position to evaluate
            
        Returns:
            Evaluation score from current player's perspective
        """
        if board.is_checkmate():
            return -900000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Calculate dynamic evaluation for both sides
        white_score = self._evaluate_side_dynamic(board, chess.WHITE)
        black_score = self._evaluate_side_dynamic(board, chess.BLACK)
        
        # Calculate chaos factor for position complexity
        chaos_factor = self._calculate_chaos_factor(board)
        
        # Apply chaos bonus to current player (encourages complex positions)
        if board.turn == chess.WHITE:
            base_score = white_score - black_score
            return base_score + (chaos_factor * 3)  # 3cp per chaos point
        else:
            base_score = black_score - white_score
            return base_score + (chaos_factor * 3)
    
    def _calculate_piece_true_value(self, board: chess.Board, square: chess.Square, color: chess.Color) -> int:
        """
        Calculate TRUE piece value based on attacks + legal moves (not static assumptions)
        
        This is the core of TAL-BOT: pieces are valued by what they can DO, not what
        traditional chess says they're worth.
        
        Args:
            board: Current position
            square: Square containing the piece
            color: Color of the piece
            
        Returns:
            True value = attacks_count + legal_moves_count
        """
        piece = board.piece_at(square)
        if not piece or piece.color != color:
            return 0
        
        # Count squares this piece attacks
        attacks = board.attacks(square)
        attack_count = len(attacks)
        
        # Count legal moves this piece can make
        legal_moves_count = 0
        for move in board.legal_moves:
            if move.from_square == square:
                # Only count moves to squares not occupied by our own pieces
                target_piece = board.piece_at(move.to_square)
                if not target_piece or target_piece.color != color:
                    legal_moves_count += 1
        
        true_value = attack_count + legal_moves_count
        
        # Bonus for pieces in immediate tactical situations
        if board.is_attacked_by(not color, square):
            true_value += 5  # Piece under attack needs attention
        
        return true_value
    
    def _get_priority_pieces(self, board: chess.Board, color: chess.Color) -> tuple:
        """
        Get priority pieces: highest and lowest true value pieces
        
        This implements TAL-BOT's revolutionary approach:
        - Focus on best pieces (tactical opportunities)  
        - Focus on worst pieces (activation potential)
        - Ignore middle pieces (adequately placed)
        
        Args:
            board: Current position
            color: Side to analyze
            
        Returns:
            (high_priority_squares, low_priority_squares)
        """
        piece_values = []
        
        # Calculate true value for all pieces of this color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                true_value = self._calculate_piece_true_value(board, square, color)
                piece_values.append((true_value, square, piece.piece_type))
        
        # Sort by true value (highest first)
        piece_values.sort(key=lambda x: x[0], reverse=True)
        
        if len(piece_values) < 3:
            return [sq for _, sq, _ in piece_values], []
        
        # Take top 2-3 pieces (highest true value)
        high_priority = [sq for _, sq, _ in piece_values[:3]]
        
        # Take bottom 2-3 pieces with value 0-2 (needs activation)
        low_priority = [sq for val, sq, _ in piece_values if val <= 2][-3:]
        
        return high_priority, low_priority
    
    def _check_pv_following(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Check if we can instantly play a move from stored PV (PV following)
        
        This is crucial for bullet/blitz where TAL-BOT should excel.
        If opponent plays expected moves, we blitz out our response.
        
        Args:
            board: Current position
            
        Returns:
            Instant move if PV following applies, None otherwise
        """
        if not self.last_search_pv or len(self.last_search_pv) < 2:
            return None
        
        # Check if current position matches expected PV position
        # We need to verify the opponent played our predicted move
        
        # For now, simple implementation: if we have stored PV and 
        # it's our turn, try to play the next move in the PV
        if len(self.last_search_pv) >= 1:
            candidate_move = self.last_search_pv[0]
            
            # Verify the move is still legal in current position
            if candidate_move in board.legal_moves:
                # Remove this move from PV and advance
                self.last_search_pv = self.last_search_pv[1:]
                return candidate_move
        
        return None
    
    def _store_pv_board_states(self, board: chess.Board):
        """
        Store board states for PV following verification
        
        Args:
            board: Current position
        """
        # Store current board state and PV for next move
        self.pv_board_states = [board.copy()]
        
        # Simulate PV moves to store expected positions
        temp_board = board.copy()
        for i, move in enumerate(self.principal_variation[:5]):  # Store first 5 moves
            if move in temp_board.legal_moves:
                temp_board.push(move)
                self.pv_board_states.append(temp_board.copy())
            else:
                break
    
    def _negamax_with_pv(self, board: chess.Board, depth: int, alpha: float, beta: float, 
                        target_time: float, pv_line: List[chess.Move]) -> float:
        """
        TAL-BOT negamax with PV collection for full line display
        
        Args:
            board: Current position
            depth: Remaining depth to search
            alpha: Alpha bound
            beta: Beta bound
            target_time: Target time limit for search
            pv_line: List to collect PV moves
            
        Returns:
            Evaluation score from current player's perspective
        """
        self.nodes_searched += 1
        
        # Check time every 1000 nodes
        if self.nodes_searched % 1000 == 0:
            elapsed = time.time() - self.search_start_time
            if elapsed > target_time:
                return 0  # Return neutral score if time exceeded
        
        # Terminal conditions
        if depth <= 0:
            return self._evaluate_position(board)
        
        if board.is_game_over():
            if board.is_checkmate():
                return -900000 + (self.default_depth - depth)  # Prefer faster mates
            return 0  # Draw
        
        # Calculate chaos factor for this position
        chaos_factor = self._calculate_chaos_factor(board)
        
        # Move generation and ordering (using TAL-BOT priority system)
        legal_moves = list(board.legal_moves)
        ordered_moves = self._order_moves_simple(board, legal_moves)
        
        best_score = -999999
        best_pv = []
        moves_searched = 0
        
        for move in ordered_moves:
            board.push(move)
            
            # Recursive search with PV collection
            current_pv = []
            score = -self._negamax_with_pv(board, depth - 1, -beta, -alpha, target_time, current_pv)
            
            board.pop()
            
            moves_searched += 1
            
            # TAL-BOT conditional pruning with PV tracking
            if score > best_score:
                best_score = score
                best_pv = [move] + current_pv  # Build PV line
            
            if score > alpha:
                alpha = score
            
            # Revolutionary chaos-driven pruning with PV consideration
            if alpha >= beta:
                # Standard alpha-beta cutoff - but check chaos compensation first
                material_balance = self._quick_material_balance(board)
                
                # If position is chaotic enough, delay pruning to explore more
                if chaos_factor >= 50 and moves_searched < 5:
                    continue  # Keep searching in chaotic positions
                
                break  # Standard beta cutoff
        
        # Copy best PV to output parameter
        pv_line[:] = best_pv
        
        # Chaos bonus: reward keeping chaotic lines alive
        if chaos_factor >= 75:
            best_score += int(chaos_factor * 0.5)  # Chaos compensation
        
        return best_score
    
    def _calculate_chaos_factor(self, board: chess.Board) -> int:
        """
        Calculate position chaos factor for entropy-driven pruning
        
        Uses python-chess built-ins for maximum efficiency. High chaos = complex
        position that traditional engines struggle with = TAL-BOT advantage
        
        Args:
            board: Current position
            
        Returns:
            Chaos score (higher = more complex/chaotic)
        """
        # Use built-in python-chess methods for speed
        legal_moves = list(board.legal_moves)
        legal_count = len(legal_moves)
        
        # Count tactical elements quickly
        checks_count = sum(1 for move in legal_moves if board.gives_check(move))
        captures_count = sum(1 for move in legal_moves if board.is_capture(move))
        
        # Count attacks on high-value pieces (king, queen)
        king_attacks = 0
        queen_attacks = 0
        
        # Find enemy king and queen
        enemy_color = not board.turn
        enemy_king_square = board.king(enemy_color)
        
        if enemy_king_square:
            king_attacks = len(board.attackers(board.turn, enemy_king_square))
        
        # Count queen attacks (if queen exists)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.QUEEN and piece.color == enemy_color:
                queen_attacks += len(board.attackers(board.turn, square))
        
        # Weighted chaos calculation
        chaos_score = (
            legal_count * 0.5 +      # Breadth of possibilities
            captures_count * 1.0 +   # Forced exchanges
            checks_count * 1.5 +     # Forced responses  
            king_attacks * 2.0 +     # King danger
            queen_attacks * 1.5      # Queen pressure
        )
        
        # Astronomical complexity bonus (>200 legal moves)
        if legal_count > 200:
            chaos_score += 100  # Massive entropy bonus!
        elif legal_count > 100:
            chaos_score += 50   # High complexity bonus
        elif legal_count > 60:
            chaos_score += 25   # Moderate complexity bonus
        
        return int(chaos_score)
    
    def _evaluate_side_dynamic(self, board: chess.Board, color: chess.Color) -> float:
        """
        Calculate position chaos factor for entropy-driven pruning
        
        Uses python-chess built-ins for maximum efficiency. High chaos = complex
        position that traditional engines struggle with = TAL-BOT advantage
        
        Args:
            board: Current position
            
        Returns:
            Chaos score (higher = more complex/chaotic)
        """
        # Use built-in python-chess methods for speed
        legal_moves = list(board.legal_moves)
        legal_count = len(legal_moves)
        
        # Count tactical elements quickly
        checks_count = sum(1 for move in legal_moves if board.gives_check(move))
        captures_count = sum(1 for move in legal_moves if board.is_capture(move))
        
        # Count attacks on high-value pieces (king, queen)
        king_attacks = 0
        queen_attacks = 0
        
        # Find enemy king and queen
        enemy_color = not board.turn
        enemy_king_square = board.king(enemy_color)
        
        if enemy_king_square:
            king_attacks = len(board.attackers(board.turn, enemy_king_square))
        
        # Count queen attacks (if queen exists)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.QUEEN and piece.color == enemy_color:
                queen_attacks += len(board.attackers(board.turn, square))
        
        # Weighted chaos calculation
        chaos_score = (
            legal_count * 0.5 +      # Breadth of possibilities
            captures_count * 1.0 +   # Forced exchanges
            checks_count * 1.5 +     # Forced responses  
            king_attacks * 2.0 +     # King danger
            queen_attacks * 1.5      # Queen pressure
        )
        
        # Astronomical complexity bonus (>200 legal moves)
        if legal_count > 200:
            chaos_score += 100  # Massive entropy bonus!
        elif legal_count > 100:
            chaos_score += 50   # High complexity bonus
        elif legal_count > 60:
            chaos_score += 25   # Moderate complexity bonus
        
        return int(chaos_score)
    
    def _evaluate_side_dynamic(self, board: chess.Board, color: chess.Color) -> float:
        """
        TAL-BOT dynamic evaluation: true piece values + chaos awareness
        
        Args:
            board: Current position
            color: Side to evaluate
            
        Returns:
            Dynamic score combining true piece values with positional factors
        """
        total_score = 0.0
        
        # Get priority pieces for focused evaluation
        high_priority, low_priority = self._get_priority_pieces(board, color)
        
        # Evaluate all pieces but weight by priority
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                # Get true piece value
                true_value = self._calculate_piece_true_value(board, square, color)
                
                # Base material value (reduced weight - 50% vs traditional 100%)
                material_value = self.piece_values.get(piece.piece_type, 0) * 0.5
                
                # Positional value from piece-square tables (traditional)
                table_square = square if color == chess.WHITE else (63 - square)
                positional_value = 0
                
                if piece.piece_type == chess.PAWN:
                    positional_value = self.pawn_table[table_square]
                elif piece.piece_type == chess.KNIGHT:
                    positional_value = self.knight_table[table_square]
                elif piece.piece_type == chess.BISHOP:
                    positional_value = self.bishop_table[table_square]
                elif piece.piece_type == chess.ROOK:
                    positional_value = self.rook_table[table_square]
                elif piece.piece_type == chess.QUEEN:
                    positional_value = self.queen_table[table_square]
                elif piece.piece_type == chess.KING:
                    positional_value = self.king_table[table_square]
                
                # Activity score (true value converted to centipawns)
                activity_score = true_value * 10  # 10cp per attack/move point
                
                # Priority bonus/penalty
                priority_bonus = 0
                if square in high_priority:
                    priority_bonus = 20  # High-value pieces get bonus attention
                elif square in low_priority:
                    priority_bonus = 10  # Low-value pieces get activation bonus
                
                # Combine all factors: 50% material + 25% positional + 25% activity
                piece_score = (material_value + 
                              positional_value * 0.5 + 
                              activity_score * 0.5 + 
                              priority_bonus)
                
                total_score += piece_score
        
        return total_score
    
    def _evaluate_side(self, board: chess.Board, color: chess.Color) -> float:
        """
        Legacy method - kept for compatibility during transition
        
        Args:
            board: Current position
            color: Side to evaluate
            
        Returns:
            Total score for the side
        """
        score = 0.0
        
        # Iterate through all pieces of this color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                # Material value
                score += self.piece_values.get(piece.piece_type, 0)
                
                # Positional value from piece-square tables
                table_square = square if color == chess.WHITE else (63 - square)
                
                if piece.piece_type == chess.PAWN:
                    score += self.pawn_table[table_square]
                elif piece.piece_type == chess.KNIGHT:
                    score += self.knight_table[table_square]
                elif piece.piece_type == chess.BISHOP:
                    score += self.bishop_table[table_square]
                elif piece.piece_type == chess.ROOK:
                    score += self.rook_table[table_square]
                elif piece.piece_type == chess.QUEEN:
                    score += self.queen_table[table_square]
                elif piece.piece_type == chess.KING:
                    score += self.king_table[table_square]
        
        return score
    
    def new_game(self):
        """Reset engine state for a new game"""
        self.nodes_searched = 0
    
    def get_engine_info(self) -> dict:
        """Return engine information and statistics"""
        return {
            'name': 'VPR',
            'version': '1.0',
            'author': 'Pat Snyder',
            'description': 'Barebones maximum depth experimental engine',
            'default_depth': self.default_depth,
            'nodes_searched': self.nodes_searched
        }
