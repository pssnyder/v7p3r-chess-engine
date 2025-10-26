#!/usr/bin/env python3

"""
V14.3 Blunder-Proof Firewall Implementation

The ADHD/Bullet Chess Solution: Three critical safety checks that prevent
obvious blunders in time pressure situations.
"""

import chess
from typing import Tuple, Optional

class BlunderProofFirewall:
    """
    Blunder-Proofing Firewall to prevent obvious mistakes in time pressure
    
    Three-part firewall:
    1. Safety Check: Protect King and Queen from direct attacks
    2. Control Check: Ensure moves improve piece mobility/control
    3. Threat Check: Don't ignore immediate tactical threats
    """
    
    def __init__(self, piece_values: dict):
        self.piece_values = piece_values
    
    def is_move_safe(self, board: chess.Board, move: chess.Move) -> Tuple[bool, str]:
        """
        Apply the three-part firewall to a move
        
        Returns:
            tuple: (is_safe, reason_if_rejected)
        """
        
        # 1. SAFETY CHECK: King and Queen protection
        safety_ok, safety_reason = self._safety_check(board, move)
        if not safety_ok:
            return False, f"SAFETY: {safety_reason}"
        
        # 2. CONTROL CHECK: Piece mobility and positional gain
        control_ok, control_reason = self._control_check(board, move)
        if not control_ok:
            return False, f"CONTROL: {control_reason}"
        
        # 3. THREAT CHECK: Don't ignore immediate threats
        threat_ok, threat_reason = self._threat_check(board, move)
        if not threat_ok:
            return False, f"THREAT: {threat_reason}"
        
        return True, "SAFE"
    
    def _safety_check(self, board: chess.Board, move: chess.Move) -> Tuple[bool, str]:
        """
        Safety Check: Does this move expose King or Queen to direct attack?
        """
        board.push(move)
        
        try:
            # Check if our king is in danger after the move
            our_king = board.king(not board.turn)  # Our color (move was pushed)
            if our_king and board.is_attacked_by(board.turn, our_king):
                # King is under attack - check if it's defended or can escape
                if not self._is_adequately_defended(board, our_king, not board.turn):
                    board.pop()
                    return False, "Exposes King to undefended attack"
            
            # Check if our queen is hanging after the move
            our_queen_squares = [sq for sq in chess.SQUARES 
                               if board.piece_at(sq) is not None and 
                               board.piece_at(sq).piece_type == chess.QUEEN and 
                               board.piece_at(sq).color == (not board.turn)]
            
            for queen_sq in our_queen_squares:
                if board.is_attacked_by(board.turn, queen_sq):
                    if not self._is_adequately_defended(board, queen_sq, not board.turn):
                        board.pop()
                        return False, "Exposes Queen to undefended attack"
            
            # Check if move creates retreat necessity (≥2 tempo moves)
            if self._forces_retreat_sequence(board, move):
                board.pop()
                return False, "Forces costly retreat sequence"
                
        finally:
            board.pop()
        
        return True, ""
    
    def _control_check(self, board: chess.Board, move: chess.Move) -> Tuple[bool, str]:
        """
        Control Check: Does this move increase piece mobility/control?
        """
        # Calculate mobility before move
        piece = board.piece_at(move.from_square)
        if not piece:
            return False, "No piece to move"
        
        mobility_before = len(list(board.attacks(move.from_square)))
        
        # Calculate mobility after move
        board.push(move)
        mobility_after = len(list(board.attacks(move.to_square)))
        board.pop()
        
        # Check for positional gains that justify reduced mobility
        positional_gains = self._evaluate_positional_gains(board, move)
        
        # If mobility decreases, check if positional gain justifies it
        if mobility_after < mobility_before:
            if positional_gains < 25:  # Threshold for acceptable trade-off
                return False, f"Reduces mobility ({mobility_before} → {mobility_after}) without sufficient positional gain"
        
        return True, ""
    
    def _threat_check(self, board: chess.Board, move: chess.Move) -> Tuple[bool, str]:
        """
        Threat Check: Does this move ignore immediate tactical threats?
        """
        # Check for immediate threats against us
        immediate_threats = self._find_immediate_threats(board)
        
        # If there are threats, check if our move addresses them
        if immediate_threats:
            material_threat = sum(threat['value'] for threat in immediate_threats)
            
            # Major material threat (>3 points) or king exposure must be addressed
            if material_threat > 300 or any(threat['type'] == 'king_threat' for threat in immediate_threats):
                if not self._move_addresses_threats(board, move, immediate_threats):
                    threat_desc = f"{material_threat/100:.1f} points" if material_threat > 0 else "King exposure"
                    return False, f"Ignores immediate threat worth {threat_desc}"
        
        return True, ""
    
    def _is_adequately_defended(self, board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
        """Check if a square is adequately defended by pieces of given color"""
        attackers = len(list(board.attackers(not color, square)))
        defenders = len(list(board.attackers(color, square)))
        
        # Simple heuristic: need at least as many defenders as attackers
        return defenders >= attackers
    
    def _forces_retreat_sequence(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move forces a costly retreat sequence"""
        # Simplified check: if move puts piece under attack and piece is valuable
        piece = board.piece_at(move.to_square)
        if piece and piece.color == (not board.turn):  # Our piece after move
            if board.is_attacked_by(board.turn, move.to_square):
                piece_value = self.piece_values.get(piece.piece_type, 0)
                # If valuable piece is attacked and not defended, likely retreat needed
                if piece_value >= 300 and not self._is_adequately_defended(board, move.to_square, piece.color):
                    return True
        return False
    
    def _evaluate_positional_gains(self, board: chess.Board, move: chess.Move) -> int:
        """Evaluate positional gains from a move"""
        gains = 0
        
        # Capture value
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                gains += self.piece_values.get(captured_piece.piece_type, 0)
        
        # Check gives tempo
        if board.gives_check(move):
            gains += 25
        
        # Central squares bonus
        if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
            gains += 20
        
        # Development bonus
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            if move.from_square in [chess.B1, chess.G1, chess.C1, chess.F1,  # White
                                  chess.B8, chess.G8, chess.C8, chess.F8]:  # Black
                gains += 30
        
        return gains
    
    def _find_immediate_threats(self, board: chess.Board) -> list:
        """Find immediate tactical threats against our pieces"""
        threats = []
        our_color = board.turn
        
        # Check each of our pieces for attacks
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == our_color:
                if board.is_attacked_by(not our_color, square):
                    # Check if piece is adequately defended
                    if not self._is_adequately_defended(board, square, our_color):
                        threat_value = self.piece_values.get(piece.piece_type, 0)
                        threat_type = 'king_threat' if piece.piece_type == chess.KING else 'material_threat'
                        threats.append({
                            'square': square,
                            'piece': piece,
                            'value': threat_value,
                            'type': threat_type
                        })
        
        return threats
    
    def _move_addresses_threats(self, board: chess.Board, move: chess.Move, threats: list) -> bool:
        """Check if a move addresses any of the given threats"""
        board.push(move)
        
        try:
            # Check if threats are resolved after the move
            remaining_threats = self._find_immediate_threats(board)
            remaining_value = sum(t['value'] for t in remaining_threats)
            original_value = sum(t['value'] for t in threats)
            
            # Move addresses threats if it significantly reduces threat value
            return remaining_value < original_value * 0.5
            
        finally:
            board.pop()
    
    def filter_safe_moves(self, board: chess.Board, moves: list) -> list:
        """
        Filter a list of moves to only include safe ones
        
        Returns:
            list: Safe moves with their safety scores
        """
        safe_moves = []
        
        for move in moves:
            is_safe, reason = self.is_move_safe(board, move)
            if is_safe:
                safe_moves.append((move, "SAFE"))
            else:
                # For debugging - can be removed in production
                safe_moves.append((move, f"REJECTED: {reason}"))
        
        return safe_moves