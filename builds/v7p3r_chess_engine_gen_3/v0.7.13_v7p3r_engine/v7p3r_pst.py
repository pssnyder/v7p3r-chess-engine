# v7p3r_pst.py

"""V7P3R Piece Square Tables Module.
This module provides piece square table evaluation with game phase awareness.
"""

import chess
from v7p3r_mvv_lva import v7p3rMVVLVA

class v7p3rPST:
    """Class for piece square table evaluation in the V7P3R chess engine."""

    def __init__(self):
        """Initialize piece square tables."""
        # Use MVV-LVA for piece values to maintain single source of truth
        self.mvv_lva = v7p3rMVVLVA()

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
                [-30,-20,-10,-10,-10,-10,-20,-30],
                [-20,  0, 10, 10, 10, 10,  0,-20],
                [-10, 10, 20, 25, 25, 20, 10,-10],
                [-10, 15, 25, 30, 30, 25, 15,-10],
                [-10, 10, 25, 30, 30, 25, 10,-10],
                [-10, 15, 20, 25, 25, 20, 15,-10],
                [-20,  0, 10, 15, 15, 10,  0,-20],
                [-30,-20,-10,-10,-10,-10,-20,-30]
            ],
            'B': [  # Bishop
                [-20,-10,-10,-10,-10,-10,-10,-20],
                [-10,  5,  0,  0,  0,  0,  5,-10],
                [-10, 10, 10, 10, 10, 10, 10,-10],
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
                [20, 20, 20, 20, 20, 20, 20, 20],
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

    def get_piece_square_value(self, piece_type: chess.PieceType, square: chess.Square,
                              is_white: bool, game_phase: float) -> float:
        """Get piece square table value for a piece.

        Args:
            piece_type: Type of piece
            square: Square the piece is on
            is_white: Whether piece is white
            game_phase: 0.0 for opening/middlegame, 1.0 for endgame

        Returns:
            float: Piece square table value
        """
        piece_symbol = chess.piece_name(piece_type).upper()[0]

        # Use endgame table for king in endgame
        if piece_type == chess.KING and game_phase > 0.5:
            table = self.pst_tables['K_endgame']
        else:
            table = self.pst_tables[piece_symbol]

        # Get square coordinates
        rank = chess.square_rank(square)
        file = chess.square_file(square)

        # Flip coordinates for black
        if not is_white:
            rank = 7 - rank

        return float(table[rank][file])

    def calculate_game_phase(self, board: chess.Board) -> float:
        """Calculate game phase factor (0.0 = opening/middlegame, 1.0 = endgame).

        Args:
            board: Current board position

        Returns:
            float: Game phase factor
        """
        material = 0

        # Count material excluding kings and using MVV-LVA values
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                material += self.mvv_lva.get_piece_value(piece.piece_type)

        # Max material (2Q + 4R + 4B + 4N + 16P)
        max_material = (2 * 900 + 4 * 500 + 4 * 330 + 4 * 320 + 16 * 100)

        return 1.0 - (material / max_material)

    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate position using piece square tables.

        Args:
            board: Current board position

        Returns:
            float: Position evaluation (positive favors white)
        """
        score = 0.0
        game_phase = self.calculate_game_phase(board)

        # Evaluate each piece
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Base value from MVV-LVA
                value = self.mvv_lva.get_piece_value(piece.piece_type)
                # Add position value from PST
                value += self.get_piece_square_value(
                    piece.piece_type,
                    square,
                    piece.color == chess.WHITE,
                    game_phase
                )
                # Apply color factor
                score += value if piece.color == chess.WHITE else -value

        return score