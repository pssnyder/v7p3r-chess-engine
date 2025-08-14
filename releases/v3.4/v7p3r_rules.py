# v7p3r_rules.py

"""Rules and Position Analysis for V7P3R Chess Engine
Handles game rules, position validation, and decision guidelines.
"""

import chess
from v7p3r_utils import get_game_phase, is_draw_position

class GameRules:
    def __init__(self, config):
        self.config = config
        self.use_stalemate_awareness = config.is_enabled('engine_config', 'use_stalemate_awarness')
        self.use_draw_prevention = config.is_enabled('engine_config', 'use_draw_prevention')
        self.use_checkmate_detection = config.is_enabled('engine_config', 'use_checkmate_detection')
    
    def validate_move(self, board, move):
        """Validate that a move is legal and acceptable"""
        # Basic legality check
        if move not in board.legal_moves:
            return False, "Illegal move"
        
        # Check for stalemate creation
        if self.use_stalemate_awareness:
            board_copy = board.copy()
            board_copy.push(move)
            if board_copy.is_stalemate():
                return False, "Move causes stalemate"
        
        return True, "Move is valid"
    
    def should_avoid_draw(self, board, move, our_color, material_balance):
        """Check if we should avoid a draw in this position"""
        if not self.use_draw_prevention:
            return False
        
        # If we're ahead in material, avoid draws
        if material_balance > 200:  # Ahead by more than 2 pawns
            board_copy = board.copy()
            board_copy.push(move)
            
            # Check for various draw conditions using utility function
            if is_draw_position(board_copy):
                return True
        
        return False
    
    def get_game_phase(self, board):
        """Determine the current game phase using standardized utility function"""
        return get_game_phase(board)
    
    def is_critical_position(self, board):
        """Check if position requires special attention"""
        # Check for immediate threats
        if board.is_check():
            return True
        
        # Check for mate threats
        if self._has_mate_threat(board):
            return True
        
        # Check for major material imbalances
        if self._has_major_imbalance(board):
            return True
        
        return False
    
    def _has_mate_threat(self, board):
        """Check for immediate mate threats"""
        if not self.use_checkmate_detection:
            return False
        
        # Simple mate threat detection
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return True
            board.pop()
        
        return False
    
    def _has_major_imbalance(self, board):
        """Check for major material imbalances"""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 325,
            chess.ROOK: 500,
            chess.QUEEN: 900
        }
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        # Major imbalance if difference > 500 centipawns (roughly a rook)
        return abs(white_material - black_material) > 500
    
    def get_position_guidelines(self, board, our_color):
        """Get position-specific guidelines for move selection"""
        guidelines = {
            'phase': self.get_game_phase(board),
            'critical': self.is_critical_position(board),
            'in_check': board.is_check(),
            'can_castle': (board.has_kingside_castling_rights(our_color) or 
                          board.has_queenside_castling_rights(our_color)),
            'material_balance': self._get_material_balance(board, our_color)
        }
        
        return guidelines
    
    def _get_material_balance(self, board, our_color):
        """Get material balance from our perspective"""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 325,
            chess.ROOK: 500,
            chess.QUEEN: 900
        }
        
        our_material = 0
        their_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                value = piece_values[piece.piece_type]
                if piece.color == our_color:
                    our_material += value
                else:
                    their_material += value
        
        return our_material - their_material
