# v7p3r_pst.py

import os
import sys
import chess
import logging
from typing import Optional
from v7p3r_debug import v7p3rLogger, v7p3rUtilities

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup centralized logging for this module
v7p3r_pst_logger = v7p3rLogger.setup_logger("v7p3r_pst")

class v7p3rPST:
    """ Piece-Square Tables for chess position evaluation. """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger if logger else v7p3r_pst_logger
        # Initialize piece-square tables
        self.tables = self._create_tables()
        # Default Piece Values in centipawns
        self.piece_values = {
            chess.KING: 20000,  # Not used for material calculation, but for MVV-LVA
            chess.QUEEN: 900,   # Standard centipawn value
            chess.ROOK: 500,    # Standard centipawn value
            chess.BISHOP: 330,  # Standard centipawn value
            chess.KNIGHT: 320,  # Standard centipawn value
            chess.PAWN: 100     # Standard centipawn value
        }
    def _create_tables(self):
        """Create all piece-square tables"""
        # Generic table - encourages center control
        self.PST_TABLE = [
            [  0,  0,  0,  0,  0,  0,  0,  0],  # 8th rank (promotion)
            [ -5, 10, 10, 10, 10, 10, 10, -5],  # 7th rank 
            [ -5, 10, 20, 20, 20, 20, 10, -5],  # 6th rank
            [ -5, 10, 20, 30, 30, 20, 10, -5],  # 5th rank
            [ -5, 10, 20, 30, 30, 20, 10, -5],  # 4th rank
            [ -5, 10, 20, 20, 20, 20, 10, -5],  # 3rd rank
            [ 10, 10, 10, 10, 10, 10, 10, 10],  # 2nd rank
            [ 10, 10, 10, 10, 10, 10, 10, 10]   # 1st rank
        ]

        # Pawn table - encourages advancement and center control
        self.PAWN_TABLE = [
            [  0,  0,  0,  0,  0,  0,  0,  0],  # 8th rank (promotion)
            [  0,  0,  0,  0,  0,  0,  0,  0],  # 7th rank 
            [  0,  0,  0,  0,  0,  0,  0,  0],  # 6th rank
            [  0,  0,  0,  0,  0,  0,  0,  0],  # 5th rank
            [  5,  5, 10, 20, 20, 10,  5,  0],  # 4th rank
            [  5,  5,-10, 10, 10,-10, -5,  5],  # 3rd rank
            [  5, 10, 10,-20,-20, 10, 10,  5],  # 2nd rank
            [  0,  0,  0,  0,  0,  0,  0,  0]   # 1st rank
        ]

        # Pawn middlegame table - encourages advancement and center control
        self.PAWN_MG_TABLE = [
            [  0,  0,  0,  0,  0,  0,  0,  0],  # 8th rank (promotion)
            [ 50, 50, 50, 50, 50, 50, 50, 50],  # 7th rank 
            [ 10, 10, 20, 30, 30, 20, 10, 10],  # 6th rank
            [  5,  5, 10, 25, 25, 10,  5,  5],  # 5th rank
            [  0,  0,  0, 20, 20,  0,  0,  0],  # 4th rank
            [  5,  5,-10, 10, 10,-10, -5,  5],  # 3rd rank
            [  5, 10, 10,-20,-20, 10, 10,  5],  # 2nd rank
            [  0,  0,  0,  0,  0,  0,  0,  0]   # 1st rank
        ]

        # Pawn endgame table - encourages advancement and king safety in the endgame
        self.PAWN_EG_TABLE = [
            [  0,  0,  0,  0,  0,  0,  0,  0],  # 8th rank (promotion)
            [ 90, 90, 90, 90, 90, 90, 90, 90],  # 7th rank (promotion square)
            [ 70, 70, 70, 70, 70, 70, 70, 70],  # 6th rank
            [ 50, 50, 50, 50, 50, 50, 50, 50],  # 5th rank
            [ 30, 30, 30, 30, 30, 30, 30, 30],  # 4th rank
            [ 10, 10, 10, 10, 10, 10, 10, 10],  # 3rd rank
            [  0,  0,  0,  0,  0,  0,  0,  0],  # 2nd rank
            [  0,  0,  0,  0,  0,  0,  0,  0]   # 1st rank
        ]
        
        # Knight table - heavily penalizes rim placement
        self.KNIGHT_TABLE = [
            [-50,-40,-30,-30,-30,-30,-40,-50],
            [-40,-20,  0,  0,  0,  0,-20,-40],
            [-30,  0, 10, 15, 15, 10,  0,-30],
            [-30,  5, 15, 20, 20, 15,  5,-30],
            [-30,  0, 15, 20, 20, 15,  0,-30],
            [-30,  5, 10, 15, 15, 10,  5,-30],
            [-40,-20,  0,  5,  5,  0,-20,-40],
            [-50,-20,-30,-30,-30,-30,-20,-50]
        ]
        
        # Bishop table - encourages long diagonals
        self.BISHOP_TABLE = [
            [-20,-10,-10,-10,-10,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5, 10, 10,  5,  0,-10],
            [-10,  5,  5, 10, 10,  5,  5,-10],
            [-10,  0, 10, 10, 10, 10,  0,-10],
            [-10, 10, 10, 10, 10, 10, 10,-10],
            [-10,  5,  0,  0,  0,  0,  5,-10],
            [-20,-10,-20,-10,-10,-20,-10,-20]
        ]
        
        # Rook table - encourages 7th rank and center files
        self.ROOK_TABLE = [
            [  0,  0,  0,  0,  0,  0,  0,  0],
            [  5, 10, 10, 10, 10, 10, 10,  5],
            [ -5,  0,  0,  0,  0,  0,  0, -5],
            [ -5,  0,  0,  0,  0,  0,  0, -5],
            [ -5,  0,  0,  0,  0,  0,  0, -5],
            [ -5,  0,  0,  0,  0,  0,  0, -5],
            [ -5,  0,  0,  0,  0,  0,  0, -5],
            [  0,  0,  0,  5,  5,  0,  0,  0]
        ]
        
        # Queen table - discourages early development
        self.QUEEN_TABLE = [
            [-20,-10,-10, -5, -5,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5,  5,  5,  5,  0,-10],
            [ -5,  0,  5,  5,  5,  5,  0, -5],
            [  0,  0,  5,  5,  5,  5,  0, -5],
            [-10,  5,  5,  5,  5,  5,  0,-10],
            [-10,  0,  5,  0,  0,  0,  0,-10],
            [-20,-10,-10, -5, -5,-10,-10,-20]
        ]

        # King table - encourages stability and castling protection
        self.KING_TABLE = [ # Renamed from KING_TABLE for clarity
            [-50,-50,-50,-50,-50,-50,-50,-50],
            [-50,-50,-50,-50,-50,-50,-50,-50],
            [-50,-50,-50,-50,-50,-50,-50,-50],
            [-50,-50,-50,-50,-50,-50,-50,-50],
            [-50,-50,-50,-50,-50,-50,-50,-50],
            [-50,-50,-50,-50,-50,-50,-50,-50],
            [-50,-50,-50,-50,-50,-50,-50,-50],
            [  0,  0,  0, 50, 50,  0,  0,  0] # Adjusted slightly for castling bonus
        ]

        # King middlegame table - encourages castling and safety (middlegame)
        self.KING_MG_TABLE = [ # Renamed from KING_TABLE for clarity
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-20,-30,-30,-40,-40,-30,-30,-20],
            [-10,-20,-20,-20,-20,-20,-20,-10],
            [  5,  5,-10,-10,-10,-10,  5,  5],
            [ 20, 30, 10, -5, -5, 10, 30, 20] # Adjusted slightly for castling bonus
        ]

        # King endgame table (encourages activity and centralization)
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
    
    def get_piece_value(self, piece):
        """
        Get the piece value based on its type without using piece-square tables.
        
        Args:
            piece: chess.Piece object
        Returns:
            Value in centipawns (positive is good for the piece's color)
        """
        if piece is None:
            return 0
        
        # Get the piece value directly in centipawns
        return self.piece_values.get(piece.piece_type, 0)
    
    def get_pst_value(self, piece, square, color, endgame_factor=0.0):
        """
        Get the piece-square table value for a piece on a specific square.
        
        Args:
            piece: chess.Piece object
            square: chess square (0-63)
            color: chess.WHITE or chess.BLACK
            endgame_factor: float between 0.0 (opening) and 1.0 (endgame)
            
        Returns:
            Value in centipawns (positive is good for the piece's color)
        """

        file = chess.square_file(square)  # 0-7 (a-h)
        rank = chess.square_rank(square)  # 0-7 (1-8)
        
        # For black pieces, flip the rank (black's perspective) to use the same table
        if color == chess.BLACK:
            rank = 7 - rank
        
        table_value = 0
        if piece.piece_type == chess.PAWN:
            # Interpolate between game phase pawn tables
            op_value = self.PAWN_TABLE[rank][file] if endgame_factor <= 0.25 else 0
            mg_value = self.PAWN_MG_TABLE[rank][file] if endgame_factor <= 0.75 else 0
            eg_value = self.PAWN_EG_TABLE[rank][file] if endgame_factor > 0.75 else 0
            table_value = (op_value * (1 - endgame_factor)) + (mg_value * (1 - endgame_factor)) + (eg_value * (1 - endgame_factor)) 
        elif piece.piece_type == chess.KNIGHT:
            table_value = self.KNIGHT_TABLE[rank][file]
        elif piece.piece_type == chess.BISHOP:
            table_value = self.BISHOP_TABLE[rank][file]
        elif piece.piece_type == chess.ROOK:
            table_value = self.ROOK_TABLE[rank][file]
        elif piece.piece_type == chess.QUEEN:
            table_value = self.QUEEN_TABLE[rank][file]
        elif piece.piece_type == chess.KING:
            # Interpolate between game phase king tables
            op_value = self.KING_TABLE[rank][file] if endgame_factor <= 0.25 else 0
            mg_value = self.KING_MG_TABLE[rank][file] if endgame_factor <= 0.75 else 0
            eg_value = self.KING_EG_TABLE[rank][file] if endgame_factor > 0.75 else 0
            table_value = (op_value * (1 - endgame_factor)) + (mg_value * (1 - endgame_factor)) + (eg_value * (1 - endgame_factor))
        else:
            table_value = 0
        
        return table_value
    
    def evaluate_board_position(self, board, endgame_factor=0.0):
        """
        Evaluate the entire board using piece-square tables.
        
        Args:
            board: chess.Board object
            endgame_factor: float between 0.0 (middlegame) and 1.0 (endgame)
            
        Returns:
            Total piece-square table score (positive favors white)
        """
        total_score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.get_piece_value(piece)
                if piece.color == chess.WHITE:
                    total_score += value
                else:
                    total_score -= value
                    
        return total_score / 100.0  # Convert centipawns to pawn units

# Example usage and testing
if __name__ == "__main__":
    # Test the piece-square tables
    pst = v7p3rPST(logging.getLogger("v7p3rPST"))
    board = chess.Board()
    
    print("Initial position PST evaluation:", pst.evaluate_board_position(board))
    
    # Test specific squares
    print(f"Knight on h3 value (White): {pst.get_pst_value(chess.Piece(chess.KNIGHT, chess.WHITE), chess.H3, chess.WHITE)}")
    print(f"Knight on e4 value (White): {pst.get_pst_value(chess.Piece(chess.KNIGHT, chess.WHITE), chess.E4, chess.WHITE)}")
    print(f"Difference (White): {pst.get_pst_value(chess.Piece(chess.KNIGHT, chess.WHITE), chess.E4, chess.WHITE) - pst.get_pst_value(chess.Piece(chess.KNIGHT, chess.WHITE), chess.H3, chess.WHITE)} centipawns")

    # Test black perspective
    print(f"Knight on a6 value (Black): {pst.get_pst_value(chess.Piece(chess.KNIGHT, chess.BLACK), chess.A6, chess.BLACK)}")
    print(f"Knight on d5 value (Black): {pst.get_pst_value(chess.Piece(chess.KNIGHT, chess.BLACK), chess.D5, chess.BLACK)}")

    # Test endgame factor
    endgame_board = chess.Board("8/8/8/8/8/8/P7/K7 w - - 0 1") # White pawn on 7th, King on 1st
    print("\nEndgame board (PST with endgame_factor=0.0):", pst.evaluate_board_position(endgame_board, endgame_factor=0.0))
    print("Endgame board (PST with endgame_factor=1.0):", pst.evaluate_board_position(endgame_board, endgame_factor=1.0))
    
    # Test King endgame positions
    king_central_endgame_board = chess.Board("8/8/8/4k3/4K3/8/8/8 w - - 0 1")
    print(f"\nKing central (endgame_factor=1.0) for White: {pst.get_pst_value(chess.Piece(chess.KING, chess.WHITE), chess.E4, chess.WHITE, endgame_factor=1.0)}")
    print(f"King corner (endgame_factor=1.0) for White: {pst.get_pst_value(chess.Piece(chess.KING, chess.WHITE), chess.A1, chess.WHITE, endgame_factor=1.0)}")
