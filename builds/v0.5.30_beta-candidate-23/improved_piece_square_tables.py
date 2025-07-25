
# improved_piece_square_tables.py
# Enhanced piece-square tables for better positional play

import chess

class PieceSquareTables:
    def __init__(self):
        # Pawn piece-square table (encourages central control and advancement)
        self.PAWN_TABLE = [
            [  0,   0,   0,   0,   0,   0,   0,   0],
            [ 50,  50,  50,  50,  50,  50,  50,  50],
            [ 10,  10,  20,  30,  30,  20,  10,  10],
            [  5,   5,  10,  25,  25,  10,   5,   5],
            [  0,   0,   0,  20,  20,   0,   0,   0],
            [  5,  -5, -10,   0,   0, -10,  -5,   5],
            [  5,  10,  10, -20, -20,  10,  10,   5],
            [  0,   0,   0,   0,   0,   0,   0,   0]
        ]

        # Knight piece-square table (strongly encourages centralization)
        self.KNIGHT_TABLE = [
            [-50, -40, -30, -30, -30, -30, -40, -50],
            [-40, -20,   0,   0,   0,   0, -20, -40],
            [-30,   0,  10,  15,  15,  10,   0, -30],
            [-30,   5,  15,  20,  20,  15,   5, -30],
            [-30,   0,  15,  20,  20,  15,   0, -30],
            [-30,   5,  10,  15,  15,  10,   5, -30],
            [-40, -20,   0,   5,   5,   0, -20, -40],
            [-50, -40, -30, -30, -30, -30, -40, -50]
        ]

        # Bishop piece-square table (encourages diagonals and center)
        self.BISHOP_TABLE = [
            [-20, -10, -10, -10, -10, -10, -10, -20],
            [-10,   0,   0,   0,   0,   0,   0, -10],
            [-10,   0,   5,  10,  10,   5,   0, -10],
            [-10,   5,   5,  10,  10,   5,   5, -10],
            [-10,   0,  10,  10,  10,  10,   0, -10],
            [-10,  10,  10,  10,  10,  10,  10, -10],
            [-10,   5,   0,   0,   0,   0,   5, -10],
            [-20, -10, -10, -10, -10, -10, -10, -20]
        ]

        # Rook piece-square table (encourages 7th rank and open files)
        self.ROOK_TABLE = [
            [  0,   0,   0,   0,   0,   0,   0,   0],
            [  5,  10,  10,  10,  10,  10,  10,   5],
            [ -5,   0,   0,   0,   0,   0,   0,  -5],
            [ -5,   0,   0,   0,   0,   0,   0,  -5],
            [ -5,   0,   0,   0,   0,   0,   0,  -5],
            [ -5,   0,   0,   0,   0,   0,   0,  -5],
            [ -5,   0,   0,   0,   0,   0,   0,  -5],
            [  0,   0,   0,   5,   5,   0,   0,   0]
        ]

        # King middle game table (encourages castling and safety)
        self.KING_MG_TABLE = [
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-20, -30, -30, -40, -40, -30, -30, -20],
            [-10, -20, -20, -20, -20, -20, -20, -10],
            [ 20,  20,   0,   0,   0,   0,  20,  20],
            [ 20,  30,  10,   0,   0,  10,  30,  20]
        ]

        # King endgame table (encourages activity)
        self.KING_EG_TABLE = [
            [-50, -40, -30, -20, -20, -30, -40, -50],
            [-30, -20, -10,   0,   0, -10, -20, -30],
            [-30, -10,  20,  30,  30,  20, -10, -30],
            [-30, -10,  30,  40,  40,  30, -10, -30],
            [-30, -10,  30,  40,  40,  30, -10, -30],
            [-30, -10,  20,  30,  30,  20, -10, -30],
            [-30, -30,   0,   0,   0,   0, -30, -30],
            [-50, -30, -30, -30, -30, -30, -30, -50]
        ]

    def get_piece_value(self, piece, square, endgame_factor=0.0):
        """Get piece-square table value for a piece on a square"""
        if piece.color == chess.BLACK:
            # Flip the table for black pieces
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            square = chess.square(file, 7 - rank)

        rank = chess.square_rank(square)
        file = chess.square_file(square)

        if piece.piece_type == chess.PAWN:
            return self.PAWN_TABLE[rank][file]
        elif piece.piece_type == chess.KNIGHT:
            return self.KNIGHT_TABLE[rank][file]
        elif piece.piece_type == chess.BISHOP:
            return self.BISHOP_TABLE[rank][file]
        elif piece.piece_type == chess.ROOK:
            return self.ROOK_TABLE[rank][file]
        elif piece.piece_type == chess.KING:
            # Interpolate between middle game and endgame tables
            mg_value = self.KING_MG_TABLE[rank][file]
            eg_value = self.KING_EG_TABLE[rank][file]
            return mg_value * (1 - endgame_factor) + eg_value * endgame_factor

        return 0  # Queen doesn't need piece-square table
