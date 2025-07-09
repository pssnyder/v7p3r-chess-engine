# v7p3r_rules.py

""" v7p3r Rules Management Module
This module is responsible for managing chess rules, validating moves, and handling special game rules.
It is designed to be used by the v7p3r chess engine.
"""

import chess
import sys
import os
from v7p3r_debug import v7p3rLogger, v7p3rUtilities

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup centralized logging for this module
v7p3r_rules_logger = v7p3rLogger.setup_logger("v7p3r_rules")

# ==========================================
# ========= RULE SCORING CLASS =========
class v7p3rRules:
    def __init__(self, ruleset, pst):
        self.logger = v7p3r_rules_logger

        # Initialize scoring parameters
        self.root_board = chess.Board()
        self.game_phase = 'opening'  # Default game phase
        self.endgame_factor = 0.0  # Default endgame factor for endgame awareness
        self.fallback_modifier = 1.0  # Default modifier for scoring components (was 100, reduced to avoid score inflation)
        self.score_counter = 0
        self.score_id = f"score[{self.score_counter}]_{v7p3rUtilities.get_timestamp()}"
        self.fen = self.root_board.fen()
        self.root_move = chess.Move.null()
        self.score = 0.0

        # Set up required modules
        self.ruleset = ruleset
        self.pst = pst
        
        # Standard centipawn piece values - centralizing to avoid duplication
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000  # Not used for material calculation, but for MVV-LVA
        }

    def checkmate_threats(self, board: chess.Board, color: chess.Color) -> float:
        """
        Assess if 'color' can deliver a checkmate on their next move.
        Only consider legal moves for 'color' without mutating the original board's turn.
        """
        score = 0.0
        checkmate_threats_modifier = self.ruleset.get('checkmate_threats_modifier', self.fallback_modifier)
        if checkmate_threats_modifier == 0.0:
            return score

        if board.is_checkmate() and board.turn == color:
            score += 9999.0 * checkmate_threats_modifier
        elif board.is_checkmate() and board.turn != color:
            score -= 9999.0 * checkmate_threats_modifier
        return score

    def material_count(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate the material count based purely on piece numbers per side, showing current side's material advantage."""
        score = 0.0
        material_count_modifier = self.ruleset.get('material_count_modifier', self.fallback_modifier)
        if material_count_modifier == 0.0:
            return score

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                score += 1.0 * material_count_modifier
            elif piece and piece.color != color:
                score -= 1.0 * material_count_modifier

        return score

    def material_score(self, board: chess.Board, color: chess.Color) -> float:
        """Simple material count for given color"""
        score = 0.0
        material_score_modifier = self.ruleset.get('material_score_modifier', 1.0)
        if material_score_modifier == 0.0:
            return score

        # Iterate over all pieces of the given color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                # Add the material value of this piece
                score += self.pst.get_piece_value(piece) * material_score_modifier
            elif piece and piece.color != color:
                # Subtract the opponent's material value
                score -= self.pst.get_piece_value(piece) * material_score_modifier
        return score
    
    def pst_score(self, board: chess.Board, color: chess.Color, endgame_factor: float) -> float:
        """Calculate the score based on Piece-Square Tables (PST) for the given color."""
        score = 0.0
        pst_score_modifier = self.ruleset.get('pst_score_modifier', self.fallback_modifier)
        if pst_score_modifier == 0.0:
            return score

        # Iterate over all pieces of the given color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                # Add the PST value for this piece at its current square
                score += self.pst.get_pst_value(piece, square, color, endgame_factor) * pst_score_modifier
            elif piece and piece.color != color:
                # Subtract the opponent's PST value
                score -= self.pst.get_pst_value(piece, square, piece.color, endgame_factor) * pst_score_modifier

        return score
    
    def piece_captures(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate the score based on modern MVV-LVA (Most Valuable Victim - Least Valuable Attacker) evaluation."""
        score = 0.0
        piece_captures_modifier = self.ruleset.get('piece_captures_modifier', self.fallback_modifier)
        if piece_captures_modifier == 0.0:
            return score

        # Use centralized piece values for consistency
        mvv_lva_values = self.piece_values

        # Get all capture moves
        captures = []
        for move in board.legal_moves:
            if board.is_capture(move):
                captures.append(move)
                
        # Limit number of captures to analyze (max 8 captures)
        max_captures = 8
        capture_count = min(len(captures), max_captures)
        
        # Only analyze the most important captures if we have too many
        if capture_count > 0:
            # Sort captures by rough MVV-LVA (prioritize high-value victims)
            capture_priorities = []
            for move in captures:
                victim_piece = board.piece_at(move.to_square)
                attacker_piece = board.piece_at(move.from_square)
                
                if victim_piece and attacker_piece:
                    victim_value = mvv_lva_values.get(victim_piece.piece_type, 0)
                    attacker_value = mvv_lva_values.get(attacker_piece.piece_type, 0)
                    priority = victim_value - (attacker_value * 0.1)
                    capture_priorities.append((move, priority))
                    
            # Sort by priority (highest first) and take top N captures
            capture_priorities.sort(key=lambda x: x[1], reverse=True)
            top_captures = [cp[0] for cp in capture_priorities[:max_captures]]
            
            # Evaluate these top captures
            for move in top_captures:
                victim_piece = board.piece_at(move.to_square)
                attacker_piece = board.piece_at(move.from_square)
                
                if victim_piece and attacker_piece and victim_piece.color != color:
                    victim_value = mvv_lva_values.get(victim_piece.piece_type, 0)
                    attacker_value = mvv_lva_values.get(attacker_piece.piece_type, 0)
                    
                    # Basic MVV-LVA score: prioritize high-value victims with low-value attackers
                    mvv_lva_score = victim_value - (attacker_value * 0.1)
                    
                    # Check if the capture is safe using our simplified SEE
                    capture_safety = self._evaluate_capture_safety(board, move, color)
                    
                    # Apply bonuses/penalties based on capture quality
                    if capture_safety > 0:
                        # Winning capture
                        score += (mvv_lva_score + capture_safety) * piece_captures_modifier * 0.01
                    elif capture_safety == 0:
                        # Equal trade
                        score += mvv_lva_score * piece_captures_modifier * 0.005
                    else:
                        # Losing capture - penalize but less severely for high-value victims
                        penalty_factor = 0.002 if victim_value >= 500 else 0.001
                        score += mvv_lva_score * piece_captures_modifier * penalty_factor

        # Simplify defensive consideration - only check for direct threats to valuable pieces
        valuable_pieces = [chess.QUEEN, chess.ROOK]
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color and piece.piece_type in valuable_pieces:
                if board.is_attacked_by(not color, square):
                    victim_value = mvv_lva_values.get(piece.piece_type, 0)
                    score -= victim_value * piece_captures_modifier * 0.005

        return score

    def _evaluate_capture_safety(self, board: chess.Board, capture_move: chess.Move, color: chess.Color) -> float:
        """
        Helper function to evaluate the safety of a capture move.
        Returns the material balance (positive if good for the player, negative if bad).
        Simplified SEE (Static Exchange Evaluation) to avoid infinite loops.
        """
        # Make the capture move temporarily
        board_copy = board.copy()
        
        # Get the value of the initially captured piece before making the move
        captured_piece = board_copy.piece_at(capture_move.to_square)
        if not captured_piece:
            return 0  # No piece to capture
            
        captured_piece_value = self.piece_values.get(captured_piece.piece_type, 0)
        
        # Get the value of the attacking piece
        attacking_piece = board_copy.piece_at(capture_move.from_square)
        if not attacking_piece:
            return 0  # No attacker (shouldn't happen)
            
        attacking_piece_value = self.piece_values.get(attacking_piece.piece_type, 0)
        
        # Make the move
        board_copy.push(capture_move)
        target_square = capture_move.to_square
        
        # Check if the square is now defended by the opponent
        is_defended = board_copy.is_attacked_by(not color, target_square)
        
        if not is_defended:
            # Simple case: we capture a piece and it's not defended
            return captured_piece_value
            
        # If the square is defended, we have a simple trade
        # For simplicity, we'll just do a basic evaluation
        # Positive if we're trading a lower value piece for a higher value one
        return captured_piece_value - attacking_piece_value

    def castling(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate castling rights and opportunities"""
        score = 0.0
        castling_modifier = self.ruleset.get('castling_modifier', self.fallback_modifier)
        if castling_modifier == 0.0:
            return score
        
        # Standard values for castling in centipawns
        castled_value = 50.0  # Value for having castled
        lost_castling_rights_penalty = -25.0  # Penalty for losing castling rights without castling
        
        # Check if castled - more robust check considering king's final position
        king_sq = board.king(color)
        white_castling_score = 0.0
        black_castling_score = 0.0
        
        if king_sq: # Ensure king exists
            if color == chess.WHITE:
                if king_sq == chess.G1: # Kingside castled
                    white_castling_score += castled_value
                elif king_sq == chess.C1: # Queenside castled
                    white_castling_score += castled_value
                # Check if castling rights are lost without castling
                elif not board.has_kingside_castling_rights(chess.WHITE) and not board.has_queenside_castling_rights(chess.WHITE):
                    white_castling_score += lost_castling_rights_penalty
            else: # Black
                if king_sq == chess.G8: # Kingside castled
                    black_castling_score += castled_value
                elif king_sq == chess.C8: # Queenside castled
                    black_castling_score += castled_value
                # Check if castling rights are lost without castling
                elif not board.has_kingside_castling_rights(chess.BLACK) and not board.has_queenside_castling_rights(chess.BLACK):
                    black_castling_score += lost_castling_rights_penalty
            
            # Additional modification for opponents castling opportunities
            if color == chess.WHITE:
                score = white_castling_score - black_castling_score
            else:
                score = black_castling_score - white_castling_score
                
        return score * castling_modifier