# v7p3r_pst.py

import os
import sys
import chess

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class v7p3rPST:
    """ Piece-Square Tables for chess position evaluation. """
    
    def __init__(self):
        # Base piece values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Initialize piece-square tables
        self.pst_tables = {
            'P': [  # Pawn
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [50, 50, 50, 50, 50, 50, 50, 50],
                [10, 10, 20, 30, 30, 20, 10, 10],
                [ 5,  5, 10, 25, 25, 10,  5,  5],
                [ 0,  0,  0, 20, 20,  0,  0,  0],
                [ 5, -5,-10,  0,  0,-10, -5,  5],
                [ 5, 10, 10,-20,-20, 10, 10,  5],
                [ 0,  0,  0,  0,  0,  0,  0,  0]
            ],
            'N': [  # Knight
                [-30,-20,-10,-10,-10,-10,-20,-30],  # Less penalty in corners
                [-20,  0, 10, 10, 10, 10,  0,-20],
                [-10, 10, 20, 25, 25, 20, 10,-10],  # Higher central values
                [-10, 15, 25, 30, 30, 25, 15,-10],
                [-10, 10, 25, 30, 30, 25, 10,-10],
                [-10, 15, 20, 25, 25, 20, 15,-10],
                [-20,  0, 10, 15, 15, 10,  0,-20],
                [-30,-20,-10,-10,-10,-10,-20,-30]
            ],
            'B': [  # Bishop
                [-20,-10,-10,-10,-10,-10,-10,-20],
                [-10,  5,  0,  0,  0,  0,  5,-10],
                [-10, 10, 10, 10, 10, 10, 10,-10],  # Higher values on diagonals
                [-10,  0, 15, 20, 20, 15,  0,-10],
                [-10,  5, 15, 20, 20, 15,  5,-10],
                [-10,  0, 10, 10, 10, 10,  0,-10],
                [-10,  5,  0,  0,  0,  0,  5,-10],
                [-20,-10,-10,-10,-10,-10,-10,-20]
            ],
            'R': [  # Rook
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [ 5, 10, 10, 10, 10, 10, 10,  5],
                [-5,  0,  0,  0,  0,  0,  0, -5],
                [-5,  0,  0,  0,  0,  0,  0, -5],
                [-5,  0,  0,  0,  0,  0,  0, -5],
                [-5,  0,  0,  0,  0,  0,  0, -5],
                [20, 20, 20, 20, 20, 20, 20, 20],  # 7th rank bonus
                [ 0,  0,  0,  5,  5,  0,  0,  0]
            ],
            'Q': [  # Queen
                [-20,-10,-10, -5, -5,-10,-10,-20],
                [-10,  0,  0,  0,  0,  0,  0,-10],
                [-10,  0,  5,  5,  5,  5,  0,-10],
                [ -5,  0,  5,  5,  5,  5,  0, -5],
                [  0,  0,  5,  5,  5,  5,  0, -5],
                [-10,  5,  5,  5,  5,  5,  0,-10],
                [-10,  0,  5,  0,  0,  0,  0,-10],
                [-20,-10,-10, -5, -5,-10,-10,-20]
            ],
            'K': [  # King middlegame
                [-30,-40,-40,-50,-50,-40,-40,-30],
                [-30,-40,-40,-50,-50,-40,-40,-30],
                [-30,-40,-40,-50,-50,-40,-40,-30],
                [-30,-40,-40,-50,-50,-40,-40,-30],
                [-20,-30,-30,-40,-40,-30,-30,-20],
                [-10,-20,-20,-20,-20,-20,-20,-10],
                [ 20, 20,  0,  0,  0,  0, 20, 20],
                [ 20, 30, 10,  0,  0, 10, 30, 20]
            ],
            'K_endgame': [  # King endgame
                [-50,-40,-30,-20,-20,-30,-40,-50],
                [-30,-20,-10,  0,  0,-10,-20,-30],
                [-30,-10, 20, 30, 30, 20,-10,-30],
                [-30,-10, 30, 40, 40, 30,-10,-30],
                [-30,-10, 30, 40, 40, 30,-10,-30],
                [-30,-10, 20, 30, 30, 20,-10,-30],
                [-30,-30,  0,  0,  0,  0,-30,-30],
                [-50,-30,-30,-30,-30,-30,-30,-50]
            ]
        }

    def get_piece_square_value(self, piece_type: chess.PieceType, square: chess.Square, is_white: bool, endgame_factor: float = 0.0) -> float:
        """Get the piece-square table value for a piece at a given square"""
        piece_symbol = chess.piece_name(piece_type).upper()[0]
        
        # Handle king separately for endgame
        if piece_type == chess.KING and endgame_factor > 0.5:
            table = self.pst_tables['K_endgame']
        else:
            table = self.pst_tables[piece_symbol]
            
        # Convert square to rank/file
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        # Flip rank for black pieces
        if not is_white:
            rank = 7 - rank
            
        return float(table[rank][file])
    
    def get_piece_value(self, piece: chess.Piece) -> float:
        """Get the base value of a piece"""
        return float(self.piece_values[piece.piece_type])
    
    def evaluate_board_position(self, board: chess.Board) -> float:
        """Evaluate the entire board position using piece-square tables"""
        score = 0.0
        endgame_factor = self._calculate_endgame_factor(board)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Get base piece value
                value = self.get_piece_value(piece)
                # Add positional value from PST
                value += self.get_piece_square_value(piece.piece_type, square, piece.color == chess.WHITE, endgame_factor)
                # Apply color factor
                score += value if piece.color == chess.WHITE else -value
                
        return score
        
    def _calculate_endgame_factor(self, board: chess.Board) -> float:
        """Calculate endgame factor between 0.0 (middlegame) and 1.0 (endgame)"""
        # Count material excluding kings
        total_material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                total_material += self.piece_values[piece.piece_type]
                
        # Max material = 2 queens + 4 rooks + 4 bishops + 4 knights + 16 pawns
        max_material = 2 * 900 + 4 * 500 + 4 * 330 + 4 * 320 + 16 * 100
        
        # Convert to factor 0.0 to 1.0
        return 1.0 - (total_material / max_material)