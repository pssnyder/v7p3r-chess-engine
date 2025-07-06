# v7p3r_score.py

""" v7p3r Scoring Calculation Module (Original Full Version)
This module is responsible for calculating the score of a chess position based on various factors,
including material balance, piece-square tables, king safety, and other positional features.
It is designed to be used by the v7p3r chess engine.
"""

import chess
import sys
import os
import logging
import datetime

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base = getattr(sys, '_MEIPASS', None)
    if base:
        return os.path.join(base, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# =====================================
# ========== LOGGING SETUP ============
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create logging directory relative to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
log_dir = os.path.join(project_root, 'logging')
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# Setup individual logger for this file
timestamp = get_timestamp()
log_filename = f"v7p3r_rules_{timestamp}.log"
log_file_path = os.path.join(log_dir, log_filename)

v7p3r_rules_logger = logging.getLogger(f"v7p3r_rules_{timestamp}")
v7p3r_rules_logger.setLevel(logging.DEBUG)

if not v7p3r_rules_logger.handlers:
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10*1024*1024,
        backupCount=3,
        delay=True
    )
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(funcName)-15s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    v7p3r_rules_logger.addHandler(file_handler)
    v7p3r_rules_logger.propagate = False

# ==========================================
# ========= RULE SCORING CLASS =========
class v7p3rRules:
    def __init__(self, ruleset, pst):
        self.logger = v7p3r_rules_logger

        # Initialize scoring parameters
        self.root_board = chess.Board()
        self.game_phase = 'opening'  # Default game phase
        self.endgame_factor = 0.0  # Default endgame factor for endgame awareness
        self.fallback_modifier = 100
        self.score_counter = 0
        self.score_id = f"score[{self.score_counter}]_{timestamp}"
        self.fen = self.root_board.fen()
        self.root_move = chess.Move.null()
        self.score = 0.0

        # Set up required modules
        self.ruleset = ruleset
        self.pst = pst
        

    def _checkmate_threats(self, board: chess.Board, color: chess.Color) -> float:
        """
        Assess if 'color' can deliver a checkmate on their next move.
        Only consider legal moves for 'color' without mutating the original board's turn.
        """
        score = 0.0
        checkmate_threats_modifier = self.ruleset.get('checkmate_threats_modifier', self.fallback_modifier*9999)
        if checkmate_threats_modifier == 0.0:
            return score

        if board.is_checkmate() and board.turn == color:
            score += checkmate_threats_modifier
        elif board.is_checkmate() and board.turn != color:
            score -= checkmate_threats_modifier
        return score

    def _material_score(self, board: chess.Board, color: chess.Color) -> float:
        """Simple material count for given color"""
        score = 0.0
        material_score_modifier = self.ruleset.get('material_score_modifier', 1.0)
        if material_score_modifier == 0.0:
            return score

        for piece_type, value in self.pst.piece_values.items():
            score += len(board.pieces(piece_type, color)) * value * material_score_modifier
        # Apply material weight from ruleset
        return score

    def _piece_captures(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate the score based on modern MVV-LVA (Most Valuable Victim - Least Valuable Attacker) evaluation."""
        score = 0.0
        piece_captures_modifier = self.ruleset.get('piece_captures_modifier', self.fallback_modifier)
        if piece_captures_modifier == 0.0:
            return score

        # MVV-LVA piece values (in centipawns for better granularity)
        mvv_lva_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000  # King captures are checkmate
        }

        # Evaluate all capture moves for the current player
        for move in board.legal_moves:
            if board.is_capture(move):
                victim_piece = board.piece_at(move.to_square)
                attacker_piece = board.piece_at(move.from_square)
                
                if victim_piece and attacker_piece and victim_piece.color != color:
                    victim_value = mvv_lva_values.get(victim_piece.piece_type, 0)
                    attacker_value = mvv_lva_values.get(attacker_piece.piece_type, 0)
                    
                    # Basic MVV-LVA score: prioritize high-value victims with low-value attackers
                    mvv_lva_score = victim_value - (attacker_value * 0.1)
                    
                    # Check if the capture is safe (Static Exchange Evaluation approximation)
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

        # Evaluate opponent's capture threats against us (defensive consideration)
        for move in board.legal_moves:
            if board.is_capture(move):
                victim_piece = board.piece_at(move.to_square)
                attacker_piece = board.piece_at(move.from_square)
                
                if victim_piece and attacker_piece and victim_piece.color == color:
                    # Our piece is being threatened
                    victim_value = mvv_lva_values.get(victim_piece.piece_type, 0)
                    
                    # Apply defensive penalty
                    score -= victim_value * piece_captures_modifier * 0.005

        return score

    def _evaluate_capture_safety(self, board: chess.Board, capture_move: chess.Move, color: chess.Color) -> float:
        """
        Helper function to evaluate the safety of a capture move.
        """
        # Make the capture move temporarily
        board_copy = board.copy()
        board_copy.push(capture_move)
        
        target_square = capture_move.to_square
        attackers_and_defenders = []
        
        # Find all pieces that can attack the target square
        for square in chess.SQUARES:
            piece = board_copy.piece_at(square)
            if piece and board_copy.is_attacked_by(piece.color, target_square):
                piece_value = {
                    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
                }.get(piece.piece_type, 0)
                attackers_and_defenders.append((piece_value, piece.color))
        
        # Sort by piece value (least valuable first for each side)
        attackers_and_defenders.sort(key=lambda x: x[0])
        
        # Simulate the exchange sequence
        material_balance = 0
        current_turn = not color  # Opponent responds to our capture
        
        # Get the value of the initially captured piece
        captured_piece = board.piece_at(capture_move.to_square)
        if captured_piece:
            captured_piece_value = {
                chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
            }.get(captured_piece.piece_type, 0)
            material_balance += captured_piece_value  # We gain the initial capture
        else:
            return 0  # No piece to capture
        
        # Process recaptures alternately
        for piece_value, piece_color in attackers_and_defenders:
            if piece_color == current_turn:
                if current_turn == color:
                    material_balance += piece_value  # We recapture
                else:
                    material_balance -= piece_value  # Opponent recaptures
                current_turn = not current_turn
        
        return material_balance

    def _center_control(self, board: chess.Board, color: chess.Color) -> float:
        """Simple center control"""
        score = 0.0
        center_control_modifier = self.ruleset.get('center_control_modifier', self.fallback_modifier)
        if center_control_modifier == 0.0:
            return score

        center = [chess.D4, chess.D5, chess.E4, chess.E5]
        for square in center:
            # Check if current player controls (has a piece on) center square
            piece = board.piece_at(square)
            if piece and piece.color == color:
                score += center_control_modifier
            elif piece and piece.color != color:
                score -= center_control_modifier
        return score

    def _piece_development(self, board: chess.Board, color: chess.Color) -> float:
        score = 0.0
        piece_development_modifier = self.ruleset.get('piece_development_modifier', self.fallback_modifier)
        if piece_development_modifier == 0.0:
            return score
        
        undeveloped_count = 0.0
        starting_squares = {
            chess.WHITE: {chess.KNIGHT: [chess.B1, chess.G1], chess.BISHOP: [chess.C1, chess.F1]},
            chess.BLACK: {chess.KNIGHT: [chess.B8, chess.G8], chess.BISHOP: [chess.C8, chess.F8]}
        }

        for piece_type_key, squares in starting_squares[color].items(): # Renamed piece_type to piece_type_key
            for square in squares:
                piece = board.piece_at(square)
                # Ensure piece exists before accessing attributes
                if piece and piece.color == color and piece.piece_type == piece_type_key:
                    # Piece is still on its starting square
                    undeveloped_count += 1

        # Apply penalty only if castling rights exist (implies early/middlegame and not yet developed)
        if undeveloped_count > 0 and (board.has_kingside_castling_rights(color) or board.has_queenside_castling_rights(color)):
            score -= undeveloped_count * piece_development_modifier
        elif undeveloped_count == 0 and self.game_phase == 'opening':
            score += piece_development_modifier

        return score

    def _board_coverage(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate attacking coverage of pieces"""
        score = 0.0
        board_coverage_modifier = self.ruleset.get('board_coverage_modifier', self.fallback_modifier)
        if board_coverage_modifier == 0.0:
            return score
        # Iterate over all pieces of the given color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color and piece.piece_type != chess.KING: # Exclude king from general mobility
                score += len(list(board.attacks(square))) * board_coverage_modifier
            elif piece and piece.color != color:
                score -= len(list(board.attacks(square))) * board_coverage_modifier
        return score
    
    def _castling(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate castling rights and opportunities"""
        score = 0.0
        castling_modifier = self.ruleset.get('castling_modifier', self.fallback_modifier)
        if castling_modifier == 0.0:
            return score
        
        # Check if castled - more robust check considering king's final position
        king_sq = board.king(color)
        white_castling_score = 0.0
        black_castling_score = 0.0
        if king_sq: # Ensure king exists
            if color == chess.WHITE:
                if king_sq == chess.G1: # Kingside castled
                    white_castling_score += castling_modifier
                elif king_sq == chess.C1: # Queenside castled
                    white_castling_score += castling_modifier
                else:
                    white_castling_score -= castling_modifier
            else: # Black
                if king_sq == chess.G8: # Kingside castled
                    black_castling_score += castling_modifier
                elif king_sq == chess.C8: # Queenside castled
                    black_castling_score += castling_modifier
                else: 
                    black_castling_score -= castling_modifier
            
            # Additional modification for opponents castling opportunities
            score = white_castling_score - black_castling_score if color == chess.WHITE else black_castling_score - white_castling_score
        return score
    
    def _castling_protection(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate castling protection for the king"""
        score = 0.0
        castling_protection_modifier = self.ruleset.get('castling_protection_modifier', self.fallback_modifier)
        if castling_protection_modifier == 0.0:
            return score

         # Bonus if still has kingside or queenside castling rights
        if board.has_kingside_castling_rights(color) and board.has_queenside_castling_rights(color):
            score += castling_protection_modifier
        elif board.has_kingside_castling_rights(color) or board.has_queenside_castling_rights(color):
            score += castling_protection_modifier / 2
        else:
            score -= castling_protection_modifier / 2
        
        return score
    
    def _piece_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate piece defense coordination for all pieces of the given color."""
        score = 0.0
        piece_coordination_modifier = self.ruleset.get('piece_coordination_modifier', self.fallback_modifier)
        if piece_coordination_modifier == 0.0:
            return score
        
        # For each piece of the given color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                # If the piece is defended by another friendly piece (i.e., the square it's on is attacked by its own color)
                if board.is_attacked_by(color, square): 
                    score += piece_coordination_modifier
                else:
                    score -= piece_coordination_modifier
        return score
    
    def _rook_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate bonus for rook pairs on same file/rank and 7th rank - FIXED DOUBLE COUNTING"""
        score = 0.0
        rook_coordination_modifier = self.ruleset.get('rook_coordination_modifier', self.fallback_modifier)
        if rook_coordination_modifier == 0.0:
            return score
        
        # Get all rooks of the given color
        rooks = list(board.pieces(chess.ROOK, color))

        # Check individual rook positions first (7th rank bonus)
        for rook_square in rooks:
            # Rook on 7th rank bonus (critical for attacking pawns)
            # White on rank 7 (index 6) or black on rank 2 (index 1)
            if (color == chess.WHITE and chess.square_rank(rook_square) == 6) or \
               (color == chess.BLACK and chess.square_rank(rook_square) == 1):
                score += rook_coordination_modifier

        # Check rook coordination (pairs)
        for i in range(len(rooks)):
            for j in range(i+1, len(rooks)):
                sq1, sq2 = rooks[i], rooks[j]
                # Same file (stacked rooks)
                if chess.square_file(sq1) == chess.square_file(sq2):
                    score += rook_coordination_modifier
                # Same rank (coordinated rooks)
                if chess.square_rank(sq1) == chess.square_rank(sq2):
                    score += rook_coordination_modifier
                elif (color == chess.WHITE and chess.square_rank(sq1) == 6) or \
                    (color == chess.BLACK and chess.square_rank(sq1) == 1):
                    score += rook_coordination_modifier / 2
                else:
                    score -= rook_coordination_modifier
        return score
