# v7p3r_secondary_scoring.py

"""Secondary Scoring for V7P3R Chess Engine
Handles castling evaluation and basic tactical scoring.
"""

import chess

class SecondaryScoring:
    def __init__(self):
        pass
    
    def evaluate_secondary_score(self, board, move, our_color, material_score):
        """Calculate secondary scoring components"""
        castling_score = self._evaluate_castling(board, move, our_color, material_score)
        tactical_score = self._evaluate_tactics(board, move, our_color)
        
        return {
            'castling_score': castling_score,
            'tactical_score': tactical_score,
            'total': castling_score + tactical_score
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
