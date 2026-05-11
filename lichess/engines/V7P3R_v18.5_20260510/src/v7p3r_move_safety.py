#!/usr/bin/env python3
"""
V7P3R Move Safety Checker - v18.0.0
Lightweight defensive tactical awareness to prevent hanging pieces

Focuses on:
- Detecting moves that leave pieces undefended
- Identifying opponent's forcing moves (captures, checks)
- Preventing middlegame material losses

Design principle: Speed-first - use minimal board copies and fast checks
"""

import chess
from typing import Optional, Tuple


class MoveSafetyChecker:
    """
    Ultra-lightweight move safety checker for defensive tactics
    Prevents hanging pieces without expensive deep search
    """
    
    def __init__(self, piece_values: dict):
        self.piece_values = piece_values
        
    def evaluate_move_safety(self, board: chess.Board, move: chess.Move) -> float:
        """
        Evaluate if a move creates tactical vulnerability
        Returns penalty (negative score) if move hangs material
        
        Speed: ~1000 checks per second (negligible impact on search)
        """
        penalty = 0.0
        
        # Make move temporarily
        board.push(move)
        
        try:
            # Check 1: Did we leave a piece hanging?
            hanging_penalty = self._check_hanging_pieces(board)
            penalty += hanging_penalty
            
            # Check 2: Did we expose our king to checks?
            if board.is_check():
                # Opponent can give check - mild penalty (checks aren't always bad)
                penalty -= 20.0
            
            # Check 3: Can opponent capture valuable material?
            capture_threat = self._check_immediate_captures(board)
            penalty += capture_threat
            
        finally:
            board.pop()
        
        return penalty
    
    def _check_hanging_pieces(self, board: chess.Board) -> float:
        """
        Check if our pieces are hanging (attacked and undefended)
        Returns negative penalty if material is hanging
        """
        penalty = 0.0
        our_color = not board.turn  # We just moved, so it's opponent's turn
        
        # Check each of our pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == our_color:
                # Skip pawns and king (too expensive to check everything)
                if piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KNIGHT, chess.BISHOP]:
                    if self._is_piece_hanging(board, square, piece):
                        # Piece is hanging - apply penalty based on value
                        piece_value = self.piece_values.get(piece.piece_type, 0)
                        penalty -= piece_value * 0.35  # 35% of piece value as penalty (Phase 2: reduced from 50%)
        
        return penalty
    
    def _is_piece_hanging(self, board: chess.Board, square: int, piece: chess.Piece) -> bool:
        """
        Check if piece is hanging (attacked by opponent, not defended by us)
        Fast check using python-chess built-in methods
        """
        our_color = piece.color
        enemy_color = not our_color
        
        # Check if opponent attacks this square
        is_attacked = board.is_attacked_by(enemy_color, square)
        if not is_attacked:
            return False  # Not attacked, can't be hanging
        
        # Check if we defend this square
        is_defended = board.is_attacked_by(our_color, square)
        if is_defended:
            # Defended - do SEE (Static Exchange Evaluation) to check if it's safe
            # Simple heuristic: if attacker is lower value than defender, it's safe
            attackers = self._get_attackers(board, square, enemy_color)
            defenders = self._get_attackers(board, square, our_color)
            
            if attackers and defenders:
                # Get lowest value attacker vs lowest value defender
                min_attacker = min(attackers)
                min_defender = min(defenders)
                
                # If they can trade favorably, piece is hanging
                piece_value = self.piece_values.get(piece.piece_type, 0)
                if min_attacker < piece_value:
                    return True  # They can capture with lower value piece
            
            return False  # Defended adequately
        
        # Attacked and undefended = hanging
        return True
    
    def _get_attackers(self, board: chess.Board, square: int, color: bool) -> list:
        """Get list of piece values attacking a square"""
        attackers = []
        
        # Check all pieces of this color
        for attacker_square in chess.SQUARES:
            piece = board.piece_at(attacker_square)
            if piece and piece.color == color:
                if board.is_attacked_by(color, square):
                    # This piece attacks the square
                    attackers.append(self.piece_values.get(piece.piece_type, 0))
        
        return attackers
    
    def _check_immediate_captures(self, board: chess.Board) -> float:
        """
        Check if opponent can capture valuable material on their next move
        Returns penalty if high-value captures are available
        """
        penalty = 0.0
        our_color = not board.turn
        
        # Check opponent's capturing moves
        for move in board.legal_moves:
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece and captured_piece.color == our_color:
                    # Opponent can capture our piece
                    capture_value = self.piece_values.get(captured_piece.piece_type, 0)
                    
                    # Only penalize if it's a high-value piece (Q, R)
                    if captured_piece.piece_type in [chess.QUEEN, chess.ROOK]:
                        penalty -= capture_value * 0.10  # 10% penalty (Phase 2: reduced from 15%)
        
        return penalty
    
    def get_safe_moves(self, board: chess.Board, moves: list, threshold: float = -50.0) -> list:
        """
        Filter moves to only safe ones (penalty above threshold)
        Use this to avoid obviously bad moves
        
        threshold: minimum safety score (-50 = allow small penalties)
        """
        safe_moves = []
        
        for move in moves:
            safety_score = self.evaluate_move_safety(board, move)
            if safety_score >= threshold:
                safe_moves.append((move, safety_score))
        
        # Sort by safety (most safe first)
        safe_moves.sort(key=lambda x: x[1], reverse=True)
        
        return [move for move, score in safe_moves]
