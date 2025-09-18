#!/usr/bin/env python3
"""
V7P3R v11 Phase 3B: Adaptive Move Ordering System
Intelligent move ordering based on position posture assessment
Author: Pat Snyder
"""

import chess
from typing import List, Tuple, Dict
from v7p3r_posture_assessment import V7P3RPostureAssessment, PositionVolatility, GamePosture


class V7P3RAdaptiveMoveOrdering:
    """Adaptive move ordering system that prioritizes moves based on position posture"""
    
    def __init__(self, posture_assessor: V7P3RPostureAssessment):
        self.posture_assessor = posture_assessor
        
        # Move type priorities for different postures
        self.posture_priorities = {
            GamePosture.EMERGENCY: {
                'escape_moves': 1000,      # King moves, piece retreats from attack
                'blocking_moves': 900,     # Block checks, interpose pieces
                'defensive_captures': 800, # Capture attacking pieces
                'defensive_moves': 700,    # Defend attacked pieces
                'other_captures': 200,     # Other captures (lower priority)
                'other_moves': 100         # All other moves
            },
            GamePosture.DEFENSIVE: {
                'defensive_captures': 900, # Capture attacking pieces
                'defensive_moves': 800,    # Defend attacked pieces
                'escape_moves': 700,       # Move pieces to safety
                'development': 600,        # Improve piece coordination
                'good_captures': 500,      # Profitable captures
                'other_captures': 300,     # Other captures
                'other_moves': 200         # Quiet moves
            },
            GamePosture.BALANCED: {
                'good_captures': 900,      # Profitable captures
                'tactical_moves': 800,     # Tactics and threats
                'development': 700,        # Piece development
                'defensive_moves': 600,    # Defend when needed
                'other_captures': 500,     # Equal captures
                'castling': 400,           # King safety
                'other_moves': 300         # Quiet positional moves
            },
            GamePosture.OFFENSIVE: {
                'tactical_moves': 1000,    # Attacks, tactics, threats
                'good_captures': 900,      # Profitable captures
                'attacking_moves': 800,    # Moves that create threats
                'development': 700,        # Aggressive development
                'other_captures': 600,     # Any captures
                'defensive_moves': 400,    # Defense (lower priority)
                'other_moves': 300         # Quiet moves
            }
        }
        
        # Piece values for capture evaluation
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 10000
        }
    
    def order_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[Tuple[chess.Move, int]]:
        """
        Order moves based on current position posture
        Returns: List of (move, priority_score) tuples, sorted by priority
        """
        if not moves:
            return []
        
        # Assess current position posture
        volatility, posture = self.posture_assessor.assess_position_posture(board)
        
        # Get priority scheme for this posture
        priorities = self.posture_priorities[posture]
        
        # Score each move
        scored_moves = []
        for move in moves:
            move_type, base_score = self._classify_move(board, move, posture, volatility)
            priority = priorities.get(move_type, 100)  # Default priority
            
            # Final score combines priority and base score
            final_score = priority + base_score
            scored_moves.append((move, final_score))
        
        # Sort by score (highest first)
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        return scored_moves
    
    def _classify_move(self, board: chess.Board, move: chess.Move, 
                      posture: GamePosture, volatility: PositionVolatility) -> Tuple[str, int]:
        """
        Classify a move and assign a base score
        Returns: (move_type, base_score)
        """
        base_score = 0
        
        # Check basic move properties
        is_capture = board.is_capture(move)
        is_check = board.gives_check(move)
        piece = board.piece_at(move.from_square)
        
        if piece is None:
            return ('other_moves', 0)
        
        # Handle captures
        if is_capture:
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                capture_value = self.piece_values[captured_piece.piece_type] - self.piece_values[piece.piece_type]
                
                # Check if this capture defends us
                if self._is_defensive_capture(board, move):
                    base_score = capture_value + 50  # Bonus for defensive captures
                    return ('defensive_captures', base_score)
                elif capture_value > 0:
                    base_score = capture_value
                    return ('good_captures', base_score)
                else:
                    base_score = capture_value
                    return ('other_captures', base_score)
        
        # Check for special move types based on posture
        if posture in [GamePosture.EMERGENCY, GamePosture.DEFENSIVE]:
            # Prioritize defensive moves
            if self._is_escape_move(board, move):
                return ('escape_moves', base_score + 100)
            elif self._is_blocking_move(board, move):
                return ('blocking_moves', base_score + 80)
            elif self._is_defensive_move(board, move):
                return ('defensive_moves', base_score + 60)
        
        elif posture == GamePosture.OFFENSIVE:
            # Prioritize aggressive moves
            if self._is_tactical_move(board, move):
                return ('tactical_moves', base_score + 100)
            elif self._creates_threats(board, move):
                return ('attacking_moves', base_score + 80)
        
        # Check for development moves
        if self._is_development_move(board, move):
            return ('development', base_score + 40)
        
        # Check for castling
        if board.is_castling(move):
            return ('castling', base_score + 30)
        
        # Default classification
        return ('other_moves', base_score)
    
    def _is_defensive_capture(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if capture removes a piece that was attacking us"""
        target_square = move.to_square
        
        # Check if the captured piece was attacking any of our pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                if board.is_attacked_by(not board.turn, square):
                    # Check if the piece being captured was one of the attackers
                    board.push(move)
                    still_attacked = board.is_attacked_by(not board.turn, square)
                    board.pop()
                    
                    if not still_attacked:  # This capture removed a threat
                        return True
        
        return False
    
    def _is_escape_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move escapes a piece from attack"""
        from_square = move.from_square
        
        # Check if the piece was under attack and moves to safety
        if board.is_attacked_by(not board.turn, from_square):
            board.push(move)
            safe = not board.is_attacked_by(not board.turn, move.to_square)
            board.pop()
            return safe
        
        return False
    
    def _is_blocking_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move blocks an attack on our pieces"""
        # Simple implementation: check if move interposes on a line of attack
        # This is a simplified version - full implementation would need ray analysis
        
        board.push(move)
        
        # Count how many of our pieces are under attack after the move
        our_attacked_after = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                if board.is_attacked_by(not board.turn, square):
                    our_attacked_after += 1
        
        board.pop()
        
        # Count how many were under attack before
        our_attacked_before = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                if board.is_attacked_by(not board.turn, square):
                    our_attacked_before += 1
        
        # If fewer pieces are attacked after the move, it might be blocking
        return our_attacked_after < our_attacked_before
    
    def _is_defensive_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move defends one of our pieces"""
        to_square = move.to_square
        
        # Check if the destination square defends any of our pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                # See if this move would defend that piece
                board.push(move)
                defended = board.is_attacked_by(board.turn, square)
                board.pop()
                
                if defended and board.is_attacked_by(not board.turn, square):
                    return True
        
        return False
    
    def _is_tactical_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move creates tactical threats"""
        board.push(move)
        
        # Count enemy pieces under attack after the move
        enemy_attacked = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                if board.is_attacked_by(board.turn, square):
                    enemy_attacked += 1
        
        board.pop()
        
        # If we attack 2+ enemy pieces, it's tactical
        return enemy_attacked >= 2
    
    def _creates_threats(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move creates new threats"""
        # Count threats before move
        threats_before = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                if board.is_attacked_by(board.turn, square):
                    threats_before += 1
        
        # Count threats after move
        board.push(move)
        threats_after = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                if board.is_attacked_by(board.turn, square):
                    threats_after += 1
        board.pop()
        
        return threats_after > threats_before
    
    def _is_development_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move develops a piece"""
        piece = board.piece_at(move.from_square)
        if piece is None:
            return False
        
        # Check if piece is moving from back rank (development)
        back_rank = 0 if piece.color else 7
        from_rank = chess.square_rank(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        
        # Simple heuristic: moving from back rank towards center
        if from_rank == back_rank and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            return abs(to_rank - 3.5) < abs(from_rank - 3.5)  # Moving towards center
        
        return False
    
    def get_best_moves(self, board: chess.Board, count: int = 10) -> List[chess.Move]:
        """Get the best moves according to current posture assessment"""
        legal_moves = list(board.legal_moves)
        scored_moves = self.order_moves(board, legal_moves)
        
        # Return top 'count' moves
        return [move for move, score in scored_moves[:count]]
    
    def get_move_classification(self, board: chess.Board, move: chess.Move) -> str:
        """Get the classification of a specific move for debugging"""
        volatility, posture = self.posture_assessor.assess_position_posture(board)
        move_type, base_score = self._classify_move(board, move, posture, volatility)
        return f"{move_type} (posture: {posture.value}, score: {base_score})"