# v7p3r_secondary_scoring.py

"""Secondary Scoring for V7P3R Chess Engine
Handles castling evaluation and tactical scoring.
"""

import chess
from v7p3r_utils import (
    is_capture_that_escapes_check, 
    evaluate_exchange,
    PIECE_VALUES,
    CHECKMATE_SCORE,
    DRAW_PENALTY
)

class SecondaryScoring:
    def __init__(self, config=None):
        self.config = config
        # Set default values if config is None
        self.use_castling = True
        self.use_tactics = True
        self.use_captures_to_escape_check = True
        
        # Load config if provided
        if config:
            self.use_castling = config.is_enabled('engine_config', 'use_castling')
            self.use_tactics = config.is_enabled('engine_config', 'use_tactics')
            self.use_captures_to_escape_check = config.is_enabled('engine_config', 'use_captures_to_escape_check')
    
    def evaluate_secondary_score(self, board, move, our_color, material_score):
        """Calculate secondary scoring components"""
        castling_score = self._evaluate_castling(board, move, our_color, material_score) if self.use_castling else 0
        tactical_score = self._evaluate_tactics(board, move, our_color) if self.use_tactics else 0
        escape_check_score = self._evaluate_escape_check(board, move, our_color) if self.use_captures_to_escape_check else 0
        
        return {
            'castling_score': castling_score,
            'tactical_score': tactical_score,
            'escape_check_score': escape_check_score,
            'total': castling_score + tactical_score + escape_check_score
        }
    
    def _evaluate_castling(self, board, move, our_color, material_score):
        """Evaluate castling moves and king/rook moves that affect castling rights"""
        score = 0
        
        # Check if this move is castling
        if board.is_castling(move):
            # Castling is good - add material score as bonus
            score += material_score
            return score
        
        # Check if move affects castling rights
        moving_piece = board.piece_at(move.from_square)
        if not moving_piece:
            return score
        
        # If we have castling rights and move king or rook without castling, penalty
        if moving_piece.color == our_color:
            if moving_piece.piece_type == chess.KING:
                if board.has_kingside_castling_rights(our_color) or board.has_queenside_castling_rights(our_color):
                    # Moving king without castling when we still have rights - penalty
                    score -= material_score // 4
            
            elif moving_piece.piece_type == chess.ROOK:
                # Check if this rook move affects castling
                if our_color == chess.WHITE:
                    if (move.from_square == chess.H1 and board.has_kingside_castling_rights(chess.WHITE)) or \
                       (move.from_square == chess.A1 and board.has_queenside_castling_rights(chess.WHITE)):
                        score -= material_score // 8
                else:
                    if (move.from_square == chess.H8 and board.has_kingside_castling_rights(chess.BLACK)) or \
                       (move.from_square == chess.A8 and board.has_queenside_castling_rights(chess.BLACK)):
                        score -= material_score // 8
        
        return score
    
    def _evaluate_tactics(self, board, move, our_color):
        """Basic tactical evaluation - pins, skewers, hanging pieces"""
        score = 0
        
        # Make the move to evaluate the resulting position
        board_copy = board.copy()
        board_copy.push(move)
        
        # Check for discovered attacks
        score += self._check_discovered_attacks(board, board_copy, move, our_color)
        
        # Check for pins and skewers
        score += self._check_pins_and_skewers(board_copy, our_color)
        
        # Check for hanging pieces (basic version)
        score += self._check_hanging_pieces(board_copy, our_color)
        
        return score
    
    def _check_discovered_attacks(self, original_board, new_board, move, our_color):
        """Check for discovered attacks created by the move"""
        score = 0
        
        # Simple discovered attack detection
        # If we moved a piece and now attack more squares than before
        moving_piece = original_board.piece_at(move.from_square)
        if not moving_piece or moving_piece.color != our_color:
            return score
        
        # Count attacks before and after the move
        original_attacks = len(list(original_board.attacks(move.from_square)))
        
        # Check if moving piece reveals attacks from pieces behind it
        # This is a simplified version - full implementation would be more complex
        return score
    
    def _check_pins_and_skewers(self, board, our_color):
        """Check for pins and skewers in the position"""
        score = 0
        
        # Basic pin detection - look for pieces that can't move without exposing king
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != our_color:  # Check opponent pieces
                # Simplified pin detection
                if self._is_pinned(board, square, our_color):
                    score += 50  # Small bonus for pinning opponent pieces
        
        return score
    
    def _is_pinned(self, board, square, our_color):
        """Check if a piece is pinned (simplified version)"""
        piece = board.piece_at(square)
        if not piece:
            return False
        
        # This is a placeholder for more sophisticated pin detection
        # Full implementation would check if moving the piece exposes the king
        return False
    
    def _check_hanging_pieces(self, board, our_color):
        """Check for hanging (undefended) pieces"""
        score = 0
        
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 325,
            chess.ROOK: 500,
            chess.QUEEN: 900
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != our_color and piece.piece_type != chess.KING:
                # Check if this opponent piece is attacked and undefended
                if board.is_attacked_by(our_color, square):
                    # Check if it's defended
                    if not board.is_attacked_by(not our_color, square):
                        # Hanging piece - we can capture it
                        score += piece_values.get(piece.piece_type, 0) // 10
        
        return score
    
    def _evaluate_escape_check(self, board, move, our_color):
        """Evaluate moves that escape check, especially captures that escape check"""
        score = 0
        
        # If we're in check, prioritize moves that escape it
        if board.is_check():
            # Base bonus for escaping check
            score += 50
            
            # SPECIAL CASE: Capturing to escape check is highly valuable
            if is_capture_that_escapes_check(board, move):
                # Get the value of the capture
                exchange_value = evaluate_exchange(board, move)
                
                # Huge bonus for capturing to escape check, especially if it's profitable
                if exchange_value > 0:
                    # Free material AND escapes check - extremely valuable
                    score += 500 + exchange_value * 2
                else:
                    # Even if not profitable, capturing to escape check is good
                    score += 300
                
                # Check if this resolves the position favorably
                board_copy = board.copy()
                board_copy.push(move)
                
                # If after this we're no longer in check and have a good position
                if not board_copy.is_check():
                    # Further bonus for completely resolving the check situation
                    score += 100
                    
                    # Check if we're attacking anything after this move
                    for target_square in chess.SQUARES:
                        target_piece = board_copy.piece_at(target_square)
                        if target_piece and target_piece.color != our_color:
                            if board_copy.is_attacked_by(our_color, target_square):
                                # We're attacking their pieces after escaping check - great!
                                score += PIECE_VALUES[target_piece.piece_type] // 10
        
        return score
