# v7p3r_mvv_lva.py

"""Most Valuable Victim - Least Valuable Attacker (MVV-LVA) for V7P3R Chess Engine
Evaluates capture moves for move ordering and scoring.
"""

import chess

class MVVLVA:
    def __init__(self):
        # Piece values for MVV-LVA (slightly different ordering than PST)
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 320,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 10000
        }
        
        # MVV-LVA table [victim][attacker]
        self.mvv_lva_table = {}
        self._build_mvv_lva_table()
    
    def _build_mvv_lva_table(self):
        """Build the MVV-LVA scoring table"""
        pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        
        for victim in pieces:
            self.mvv_lva_table[victim] = {}
            for attacker in pieces:
                # Higher score for capturing valuable pieces with less valuable pieces
                victim_value = self.piece_values[victim]
                attacker_value = self.piece_values[attacker]
                
                # MVV-LVA score: victim value * 10 - attacker value
                score = victim_value * 10 - attacker_value
                self.mvv_lva_table[victim][attacker] = score
    
    def get_capture_score(self, board, move):
        """Get MVV-LVA score for a capture move with free material detection"""
        if not board.is_capture(move):
            return 0
        
        # Get the capturing piece
        capturing_piece = board.piece_at(move.from_square)
        if not capturing_piece:
            return 0
        
        # Get the captured piece
        captured_piece = board.piece_at(move.to_square)
        if not captured_piece:
            # En passant capture
            if board.is_en_passant(move):
                return self.mvv_lva_table[chess.PAWN][capturing_piece.piece_type]
            return 0
        
        # Basic MVV-LVA score
        base_score = self.mvv_lva_table[captured_piece.piece_type][capturing_piece.piece_type]
        
        # Check if this is a FREE MATERIAL capture
        material_gain = self._evaluate_capture_sequence(board, move)
        
        # SHORT CIRCUIT: If we win significant free material, prioritize massively
        if material_gain >= 100:  # At least a pawn worth of material
            if material_gain >= 300:  # Knight/Bishop or higher
                return 50000 + material_gain  # Extremely high priority
            else:  # Pawn-level material
                return 30000 + material_gain  # Very high priority
        
        # If it's a losing capture, heavily penalize
        if material_gain < -50:
            return base_score - 5000  # Low priority for losing material
        
        return base_score + material_gain
    
    def _evaluate_capture_sequence(self, board, move):
        """Evaluate the material outcome of a capture sequence"""
        # Make a copy of the board to simulate the capture
        test_board = board.copy()
        test_board.push(move)
        
        # Get the capturing piece and target square
        capturing_piece = board.piece_at(move.from_square)
        target_square = move.to_square
        captured_piece = board.piece_at(target_square)
        
        if not capturing_piece or not captured_piece:
            return 0
        
        # Initial material gain (what we capture)
        material_gain = self.piece_values[captured_piece.piece_type]
        
        # Check if the opponent can recapture
        opponent_attackers = test_board.attackers(not test_board.turn, target_square)
        
        if not opponent_attackers:
            # NO RECAPTURE POSSIBLE - FREE MATERIAL!
            return material_gain
        
        # Find the least valuable attacker that can recapture
        least_valuable_attacker = None
        min_attacker_value = float('inf')
        
        for attacker_square in opponent_attackers:
            attacker_piece = test_board.piece_at(attacker_square)
            if attacker_piece:
                attacker_value = self.piece_values[attacker_piece.piece_type]
                if attacker_value < min_attacker_value:
                    min_attacker_value = attacker_value
                    least_valuable_attacker = attacker_piece.piece_type
        
        if least_valuable_attacker:
            # They can recapture - calculate net material exchange
            our_loss = self.piece_values[capturing_piece.piece_type]
            their_loss = self.piece_values[captured_piece.piece_type]
            net_gain = their_loss - our_loss
            
            # If we're trading up significantly, still prioritize
            if net_gain >= 200:  # Trading up by 2+ pawns worth
                return material_gain * 0.8  # High but not maximum priority
            elif net_gain > 0:
                return material_gain * 0.5  # Moderate priority for profitable trades
            else:
                return -abs(net_gain)  # Negative for unprofitable trades
        
        return 0  # Neutral if unclear
    
    def get_threat_score(self, board, square, attacking_piece_type):
        """Get threat score for attacking a piece on a square"""
        threatened_piece = board.piece_at(square)
        if not threatened_piece:
            return 0
        
        # Use MVV-LVA logic for threats
        return self.mvv_lva_table[threatened_piece.piece_type][attacking_piece_type]
    
    def sort_captures(self, board, moves):
        """Sort capture moves by MVV-LVA score (highest first)"""
        capture_moves = []
        other_moves = []
        
        for move in moves:
            if board.is_capture(move):
                score = self.get_capture_score(board, move)
                capture_moves.append((move, score))
            else:
                other_moves.append(move)
        
        # Sort captures by score (highest first)
        capture_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Return sorted captures followed by other moves
        return [move for move, score in capture_moves] + other_moves
    
    def find_hanging_pieces(self, board, color):
        """Find undefended pieces that can be captured for free"""
        hanging_pieces = []
        
        # Look at all opponent pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != color:  # Opponent's piece
                # Check if this piece is attacked by us
                our_attackers = board.attackers(color, square)
                if our_attackers:
                    # Check if it's defended by opponent
                    their_defenders = board.attackers(piece.color, square)
                    
                    if not their_defenders:
                        # HANGING PIECE! No defenders
                        hanging_pieces.append((square, piece, self.piece_values[piece.piece_type]))
                    else:
                        # Check if we can win the exchange
                        min_attacker_value = min(
                            self.piece_values[board.piece_at(att_sq).piece_type] 
                            for att_sq in our_attackers 
                            if board.piece_at(att_sq)
                        )
                        min_defender_value = min(
                            self.piece_values[board.piece_at(def_sq).piece_type] 
                            for def_sq in their_defenders 
                            if board.piece_at(def_sq)
                        )
                        
                        # If we can capture with a less valuable piece than their defender
                        if min_attacker_value < min_defender_value:
                            net_gain = self.piece_values[piece.piece_type] - min_attacker_value
                            if net_gain > 0:
                                hanging_pieces.append((square, piece, net_gain))
        
        # Sort by value (highest first)
        hanging_pieces.sort(key=lambda x: x[2], reverse=True)
        return hanging_pieces
    
    def is_free_capture(self, board, move):
        """Check if a capture move wins free material"""
        if not board.is_capture(move):
            return False, 0
        
        material_gain = self._evaluate_capture_sequence(board, move)
        return material_gain >= 100, material_gain
