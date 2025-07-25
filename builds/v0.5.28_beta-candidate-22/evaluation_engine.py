import chess
import numpy as np

class EvaluationEngine:
    def __init__(self, board):
        self.board = board
        self.piece_values = {
            chess.KING: 10,
            chess.QUEEN: 9,
            chess.ROOK: 5,
            chess.BISHOP: 3,
            chess.KNIGHT: 3,
            chess.PAWN: 1
        }
        
    def evaluate_position(self):
        score = 0
        
        # Check for immediate wins
        if self.board.is_checkmate():
            return 1000 if self.board.turn else -1000
            
        # Check for forced mates
        score += self._checkmate_threats()
        
        # Material and positional evaluation
        score += self._material_evaluation()
        score += self._positional_evaluation()
        score += self._king_safety()
        score += self._special_moves()
        
        return score

    def _checkmate_threats(self):
        score = 0
        # Check for mate in 1
        for move in self.board.legal_moves:
            self.board.push(move)
            if self.board.is_checkmate():
                score += 100 if self.board.turn else -100
            self.board.pop()
        return score

    def _material_evaluation(self):
        score = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color == self.board.turn:
                    score += value
                else:
                    score -= value
        return score

    def _positional_evaluation(self):
        score = 0
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                # Center control
                if square in center_squares:
                    score += 1
                
                # Development
                if piece.piece_type in [chess.BISHOP, chess.KNIGHT]:
                    if square not in [chess.B1, chess.G1, chess.B8, chess.G8]:
                        score += 0.5
        return score

    def _king_safety(self):
        score = 0
        king_square = self.board.king(self.board.turn)
        
        # Check threats
        if self.board.is_check():
            score -= 10
            
        # Castling rights
        if self.board.has_castling_rights(self.board.turn):
            score += 3
            
        return score

    def _special_moves(self):
        score = 0
        # En passant
        if self.board.ep_square:
            score += 1
            
        # Promotion opportunities
        for move in self.board.legal_moves:
            if move.promotion:
                score += 5
        return score
