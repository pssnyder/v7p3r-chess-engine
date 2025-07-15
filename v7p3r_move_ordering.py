# v7p3r_move_ordering.py

"""Move Ordering for V7P3R Chess Engine
Prioritizes moves for better alpha-beta pruning and search efficiency.
"""

import chess
from v7p3r_mvv_lva import MVVLVA

class MoveOrdering:
    def __init__(self):
        self.mvv_lva = MVVLVA()
    
    def order_moves(self, board, moves, max_moves=None):
        """Order moves for optimal search (best moves first)"""
        if not moves:
            return []
        
        scored_moves = []
        
        for move in moves:
            score = self._score_move(board, move)
            scored_moves.append((move, score))
        
        # Sort by score (highest first)
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Limit number of moves if specified
        if max_moves:
            scored_moves = scored_moves[:max_moves]
        
        return [move for move, score in scored_moves]
    
    def _captures_checking_piece(self, board, move):
        """Check if this move captures a piece that is giving check"""
        if not board.is_capture(move) or not board.is_check():
            return False
        
        # Get the square being captured
        to_square = move.to_square
        
        # Get all attackers of the king
        king_square = board.king(board.turn)
        attackers = board.attackers(not board.turn, king_square)
        
        # Check if the capture target is one of the checking pieces
        return to_square in attackers
    
    def _score_move(self, board, move):
        """Score a move for ordering purposes"""
        score = 0
        
        # 1. Checkmate gets highest priority
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_checkmate():
            return 1000000
        
        # 2. Avoid repetition (heavy penalty)
        if board_copy.is_repetition(2):  # Check for threefold repetition
            return -500000
        
        # 3. FREE MATERIAL CAPTURES - Extremely high priority!
        if board.is_capture(move):
            is_free, material_gain = self.mvv_lva.is_free_capture(board, move)
            if is_free:
                # SHORT CIRCUIT for free material - highest priority after checkmate
                if material_gain >= 500:  # Rook or higher
                    return 900000 + material_gain
                elif material_gain >= 300:  # Knight/Bishop
                    return 800000 + material_gain
                else:  # Pawn or small material
                    return 700000 + material_gain
        
        # 4. If in check, prioritize safe captures of checking piece
        if board.is_check():
            if board.is_capture(move):
                # Check if this capture targets the checking piece
                if self._captures_checking_piece(board, move):
                    # Verify it's a safe capture using MVV-LVA
                    mvv_lva_score = self.mvv_lva.get_capture_score(board, move)
                    if mvv_lva_score > 0:  # Positive score means safe/profitable
                        return 100000 + mvv_lva_score  # Very high priority
        
        # 5. All other captures (MVV-LVA enhanced evaluation)
        if board.is_capture(move):
            mvv_lva_score = self.mvv_lva.get_capture_score(board, move)
            score += 10000 + mvv_lva_score
        
        # 6. Checks get medium priority
        if board_copy.is_check():
            score += 5000
        
        # 4. Promotions get high priority
        if move.promotion:
            promotion_values = {
                chess.QUEEN: 900,
                chess.ROOK: 500,
                chess.KNIGHT: 300,
                chess.BISHOP: 325
            }
            score += 8000 + promotion_values.get(move.promotion, 0)
        
        # 5. Castling gets medium priority
        if board.is_castling(move):
            score += 3000
        
        # 6. Center control
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        
        # Bonus for center squares (e4, e5, d4, d5)
        if 3 <= to_file <= 4 and 3 <= to_rank <= 4:
            score += 100
        # Smaller bonus for extended center
        elif 2 <= to_file <= 5 and 2 <= to_rank <= 5:
            score += 50
        
        # 7. Piece development (early game)
        moving_piece = board.piece_at(move.from_square)
        if moving_piece:
            if moving_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                # Bonus for developing pieces from back rank
                from_rank = chess.square_rank(move.from_square)
                if (moving_piece.color == chess.WHITE and from_rank == 0) or \
                   (moving_piece.color == chess.BLACK and from_rank == 7):
                    score += 200
        
        # 8. Avoid moving same piece twice in opening
        # This would require move history - simplified version
        
        return score
    
    def get_killer_moves(self, depth):
        """Get killer moves for this depth (placeholder for future implementation)"""
        # Killer moves are non-capture moves that caused beta cutoffs
        # This would require maintaining a killer move table
        return []
    
    def get_history_score(self, move):
        """Get history heuristic score (placeholder for future implementation)"""
        # History heuristic tracks how often moves cause cutoffs
        return 0
    
    def get_hanging_piece_captures(self, board):
        """Get moves that capture hanging (undefended) pieces"""
        hanging_captures = []
        
        # Use the enhanced MVV-LVA to find hanging pieces
        hanging_pieces = self.mvv_lva.find_hanging_pieces(board, board.turn)
        
        # Find moves that capture these hanging pieces
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            if board.is_capture(move):
                target_square = move.to_square
                # Check if this move captures a hanging piece
                for hanging_square, hanging_piece, value in hanging_pieces:
                    if target_square == hanging_square:
                        hanging_captures.append((move, value))
                        break
        
        # Sort by value (highest first)
        hanging_captures.sort(key=lambda x: x[1], reverse=True)
        return [move for move, value in hanging_captures]
    
    def order_moves_with_material_priority(self, board, moves):
        """Enhanced move ordering that prioritizes free material captures"""
        if not moves:
            return []
        
        # First, get any hanging piece captures
        hanging_captures = self.get_hanging_piece_captures(board)
        
        # Score all moves
        scored_moves = []
        for move in moves:
            # Skip hanging captures (we'll add them first)
            if move in hanging_captures:
                continue
            score = self._score_move(board, move)
            scored_moves.append((move, score))
        
        # Sort non-hanging moves by score
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Return hanging captures first, then other moves by score
        ordered_moves = hanging_captures + [move for move, score in scored_moves]
        
        return ordered_moves
