
# v7p3r_scoring.py
"""
Main Scoring Module for V7P3R Chess Engine
Coordinates all scoring components and provides unified evaluation interface.
"""

import chess
from v7p3r_quiescence import QuiescenceSearch
from v7p3r_pst import PieceSquareTables
from v7p3r_mvv_lva import MVVLVA
from v7p3r_utils import (
    is_draw_position,
    is_capture_that_escapes_check,
    get_material_balance, 
    evaluate_exchange,
    find_hanging_pieces,
    PIECE_VALUES,
    CHECKMATE_SCORE,
    STALEMATE_PENALTY,
    DRAW_PENALTY
)

class ScoringSystem:
    def __init__(self, config):
        self.config = config
        self.tempo = TempoCalculation()
        self.primary = PrimaryScoring()
        self.secondary = SecondaryScoring(config)
        self.quiescence = QuiescenceSearch()
        
        # Configuration flags
        self.use_tempo = config.is_enabled('engine_config', 'use_tempo_scoring')
        self.use_primary = config.is_enabled('engine_config', 'use_primary_scoring')
        self.use_secondary = config.is_enabled('engine_config', 'use_secondary_scoring')
        self.use_quiescence = config.is_enabled('engine_config', 'use_quiescence')
    
    def evaluate_move(self, board, move, our_color, depth=0, alpha=None, beta=None):
        """
        Comprehensive move evaluation. Returns (score, details, critical_move).
        Applies tempo, primary, secondary, and (optionally) quiescence scoring.
        """
        total_score = 0
        evaluation_details = {}
        critical_move = False

        board_copy = board.copy()
        board_copy.push(move)

        # 1. Tempo Calculation (critical moves)
        if self.use_tempo:
            tempo_score, is_critical = self.tempo.evaluate_tempo(board, move, depth)
            total_score += tempo_score
            evaluation_details['tempo'] = tempo_score
            critical_move = is_critical
            if self.tempo.should_short_circuit(tempo_score):
                evaluation_details['short_circuit'] = True
                return total_score, evaluation_details, critical_move

        # 2. Primary Scoring
        if self.use_primary:
            primary_eval = self.primary.evaluate_primary_score(board_copy, our_color)
            total_score += primary_eval['total']
            evaluation_details['primary'] = primary_eval

        # 3. Secondary Scoring
        if self.use_secondary:
            material_score = evaluation_details.get('primary', {}).get('material_score', 0)
            secondary_eval = self.secondary.evaluate_secondary_score(board, move, our_color, material_score)
            total_score += secondary_eval['total']
            evaluation_details['secondary'] = secondary_eval

        # 4. Quiescence Search (if position is not quiet)
        if self.use_quiescence and not self.quiescence.is_quiet_position(board_copy):
            if alpha is not None and beta is not None:
                quies_score = self.quiescence.quiescence_search(
                    board_copy, alpha, beta, our_color, self.primary
                )
                # Adjust total score based on quiescence result
                total_score += (quies_score - evaluation_details.get('primary', {}).get('total', 0)) // 2
                evaluation_details['quiescence'] = quies_score

        return total_score, evaluation_details, critical_move
    
    def evaluate_position(self, board, our_color):
        """Static position evaluation without move"""
        if self.use_primary:
            primary_eval = self.primary.evaluate_primary_score(board, our_color)
            return primary_eval['total']
        return 0
    
    def get_material_balance(self, board, our_color):
        """Get current material balance"""
        return self.primary.get_material_balance(board, our_color)

class TempoCalculation:
    def __init__(self):
        self.checkmate_score = CHECKMATE_SCORE
        self.stalemate_penalty = STALEMATE_PENALTY
        self.mate_threat_bonus = 50000
    
    def evaluate_tempo(self, board, move, depth):
        """Evaluate tempo factors for a move"""
        # Make the move on a copy to evaluate the resulting position
        board_copy = board.copy()
        board_copy.push(move)
        
        tempo_score = 0
        critical_move = False
        
        # Check for immediate checkmate
        if board_copy.is_checkmate():
            return self.checkmate_score, True
        
        # Check for stalemate (avoid this)
        if board_copy.is_stalemate():
            return self.stalemate_penalty, True
        
        # Check for draw conditions (avoid when ahead)
        if is_draw_position(board_copy):
            material_balance = get_material_balance(board_copy, board.turn)
            if material_balance > 0:  # We're ahead, avoid draw
                tempo_score += DRAW_PENALTY
        
        # Check for checkmate threats within mate horizon
        mate_threat = self._find_mate_threat(board_copy, depth)
        if mate_threat:
            if mate_threat > 0:  # We have mate threat
                tempo_score += self.mate_threat_bonus
                critical_move = True
            else:  # Opponent has mate threat
                tempo_score += mate_threat
        
        # Check if move gives check (small bonus)
        if board_copy.is_check():
            tempo_score += 100
        
        # Special bonus: If in check and capturing the checking piece safely
        if board.is_check() and board.is_capture(move):
            if is_capture_that_escapes_check(board, move):
                # Additional tempo bonus for resolving check via capture
                tempo_score += 500
                critical_move = True
        
        return tempo_score, critical_move
    
    # Using standardized utility functions from v7p3r_utils.py
    
    def _find_mate_threat(self, board, max_depth):
        """Look for mate threats within specified depth"""
        if max_depth <= 0:
            return None
        
        # Simple mate threat detection - look for forced sequences
        if board.is_check():
            legal_moves = list(board.legal_moves)
            if len(legal_moves) == 1:
                # Only one legal move - might be forced
                board.push(legal_moves[0])
                if board.is_checkmate():
                    board.pop()
                    return -self.checkmate_score + max_depth  # Opponent mates us
                
                # Look deeper
                deeper_threat = self._find_mate_threat(board, max_depth - 1)
                board.pop()
                if deeper_threat:
                    return deeper_threat
        
        # Look for our mate threats
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return self.checkmate_score - max_depth  # We mate opponent
            board.pop()
        
        return None
    
    def should_short_circuit(self, score):
        """Determine if we should short circuit based on tempo score"""
        return abs(score) >= self.mate_threat_bonus
    

class PrimaryScoring:
    def __init__(self):
        self.pst = PieceSquareTables()
        self.mvv_lva = MVVLVA()
    
    def evaluate_primary_score(self, board, our_color):
        """
        Calculate primary scoring components: material, PST, and capture potential.
        Returns a dict with all components and total.
        """
        material_count = self._get_material_count(board, our_color)
        material_score = self._get_material_score(board, our_color)
        pst_score = self._get_pst_score(board, our_color)
        capture_score = self._get_capture_potential(board, our_color)
        return {
            'material_count': material_count,
            'material_score': material_score,
            'pst_score': pst_score,
            'capture_score': capture_score,
            'total': material_score + pst_score + capture_score
        }
    
    def _get_material_count(self, board, our_color):
        """Get raw piece count difference"""
        our_pieces = 0
        their_pieces = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                if piece.color == our_color:
                    our_pieces += 1
                else:
                    their_pieces += 1
        
        return our_pieces - their_pieces
    
    def _get_material_score(self, board, our_color):
        """Get material value difference using standard utility function"""
        return get_material_balance(board, our_color)
    
    def _get_pst_score(self, board, our_color):
        """Get piece square table evaluation"""
        our_pst = 0
        their_pst = 0
        is_endgame = self.pst.is_endgame(board)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Mirror squares for Black pieces so they're evaluated from their perspective
                if piece.color == chess.WHITE:
                    pst_square = square
                else:
                    pst_square = chess.square_mirror(square)
                    
                pst_value = self.pst.get_pst_value(piece.piece_type, pst_square, is_endgame)
                
                if piece.color == our_color:
                    our_pst += pst_value
                else:
                    their_pst += pst_value
        
        return our_pst - their_pst
    
    def _get_capture_potential(self, board, our_color):
        """Evaluate immediate capture opportunities using enhanced logic"""
        # Look for immediate favorable exchanges
        exchange_score = 0
        for move in board.legal_moves:
            if board.is_capture(move):
                # Only consider captures by our pieces
                moving_piece = board.piece_at(move.from_square)
                if moving_piece and moving_piece.color == our_color:
                    # Calculate exchange value
                    exchange_value = evaluate_exchange(board, move)
                    if exchange_value > 0:
                        exchange_score += exchange_value
        
        # Find hanging pieces (undefended or underdefended pieces)
        hanging_pieces = find_hanging_pieces(board, our_color)
        hanging_score = sum(value for _, _, value in hanging_pieces)
        
        # MVV-LVA score for additional insight
        mvv_lva_score = 0
        for move in board.legal_moves:
            if board.is_capture(move):
                moving_piece = board.piece_at(move.from_square)
                if moving_piece and moving_piece.color == our_color:
                    mvv_lva_score += self.mvv_lva.get_capture_score(board, move) // 100  # Scale down
        
        # Total capture potential
        return exchange_score + hanging_score + mvv_lva_score
    
    def get_material_balance(self, board, our_color):
        """Get current material balance for external use"""
        return self._get_material_score(board, our_color)


class SecondaryScoring:
    def __init__(self, config=None):
        self.config = config
        # Set default values if config is None
        self.use_castling = True
        self.use_tactics = True
        self.use_captures_to_escape_check = True
        
        # Load config if provided
        if config:
            self.use_castling = config.is_enabled('engine_config', 'use_castling')
            self.use_tactics = config.is_enabled('engine_config', 'use_tactics')
            self.use_captures_to_escape_check = config.is_enabled('engine_config', 'use_captures_to_escape_check')
    
    def evaluate_secondary_score(self, board, move, our_color, material_score):
        """Calculate secondary scoring components"""
        castling_score = self._evaluate_castling(board, move, our_color, material_score) if self.use_castling else 0
        tactical_score = self._evaluate_tactics(board, move, our_color) if self.use_tactics else 0
        escape_check_score = self._evaluate_escape_check(board, move, our_color) if self.use_captures_to_escape_check else 0
        
        return {
            'castling_score': castling_score,
            'tactical_score': tactical_score,
            'escape_check_score': escape_check_score,
            'total': castling_score + tactical_score + escape_check_score
        }
    
    def _evaluate_castling(self, board, move, our_color, material_score):
        """Evaluate castling moves and king/rook moves that affect castling rights"""
        score = 0
        
        # Check if this move is castling
        if board.is_castling(move):
            # Castling is good - add material score as bonus
            score += material_score
            return score
        
        # Check if move affects castling rights
        moving_piece = board.piece_at(move.from_square)
        if not moving_piece:
            return score
        
        # If we have castling rights and move king or rook without castling, penalty
        if moving_piece.color == our_color:
            if moving_piece.piece_type == chess.KING:
                if board.has_kingside_castling_rights(our_color) or board.has_queenside_castling_rights(our_color):
                    # Moving king without castling when we still have rights - penalty
                    score -= material_score // 4
            
            elif moving_piece.piece_type == chess.ROOK:
                # Check if this rook move affects castling
                if our_color == chess.WHITE:
                    if (move.from_square == chess.H1 and board.has_kingside_castling_rights(chess.WHITE)) or \
                       (move.from_square == chess.A1 and board.has_queenside_castling_rights(chess.WHITE)):
                        score -= material_score // 8
                else:
                    if (move.from_square == chess.H8 and board.has_kingside_castling_rights(chess.BLACK)) or \
                       (move.from_square == chess.A8 and board.has_queenside_castling_rights(chess.BLACK)):
                        score -= material_score // 8
        
        return score
    
    def _evaluate_tactics(self, board, move, our_color):
        """Basic tactical evaluation - pins, skewers, hanging pieces"""
        score = 0
        
        # Make the move to evaluate the resulting position
        board_copy = board.copy()
        board_copy.push(move)
        
        # Check for discovered attacks
        score += self._check_discovered_attacks(board, board_copy, move, our_color)
        
        # Check for pins and skewers
        score += self._check_pins_and_skewers(board_copy, our_color)
        
        # Check for hanging pieces (basic version)
        score += self._check_hanging_pieces(board_copy, our_color)
        
        return score
    
    def _check_discovered_attacks(self, original_board, new_board, move, our_color):
        """Check for discovered attacks created by the move"""
        score = 0
        
        # Simple discovered attack detection
        # If we moved a piece and now attack more squares than before
        moving_piece = original_board.piece_at(move.from_square)
        if not moving_piece or moving_piece.color != our_color:
            return score
        
        # Count attacks before and after the move
        original_attacks = len(list(original_board.attacks(move.from_square)))
        
        # Check if moving piece reveals attacks from pieces behind it
        # This is a simplified version - full implementation would be more complex
        return score
    
    def _check_pins_and_skewers(self, board, our_color):
        """Check for pins and skewers in the position"""
        score = 0
        
        # Basic pin detection - look for pieces that can't move without exposing king
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != our_color:  # Check opponent pieces
                # Simplified pin detection
                if self._is_pinned(board, square, our_color):
                    score += 50  # Small bonus for pinning opponent pieces
        
        return score
    
    def _is_pinned(self, board, square, our_color):
        """Check if a piece is pinned (simplified version)"""
        piece = board.piece_at(square)
        if not piece:
            return False
        
        # This is a placeholder for more sophisticated pin detection
        # Full implementation would check if moving the piece exposes the king
        return False
    
    def _check_hanging_pieces(self, board, our_color):
        """Check for hanging (undefended) pieces"""
        score = 0
        
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 325,
            chess.ROOK: 500,
            chess.QUEEN: 900
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != our_color and piece.piece_type != chess.KING:
                # Check if this opponent piece is attacked and undefended
                if board.is_attacked_by(our_color, square):
                    # Check if it's defended
                    if not board.is_attacked_by(not our_color, square):
                        # Hanging piece - we can capture it
                        score += piece_values.get(piece.piece_type, 0) // 10
        
        return score
    
    def _evaluate_escape_check(self, board, move, our_color):
        """Evaluate moves that escape check, especially captures that escape check"""
        score = 0
        
        # If we're in check, prioritize moves that escape it
        if board.is_check():
            # Base bonus for escaping check
            score += 50
            
            # SPECIAL CASE: Capturing to escape check is highly valuable
            if is_capture_that_escapes_check(board, move):
                # Get the value of the capture
                exchange_value = evaluate_exchange(board, move)
                
                # Huge bonus for capturing to escape check, especially if it's profitable
                if exchange_value > 0:
                    # Free material AND escapes check - extremely valuable
                    score += 500 + exchange_value * 2
                else:
                    # Even if not profitable, capturing to escape check is good
                    score += 300
                
                # Check if this resolves the position favorably
                board_copy = board.copy()
                board_copy.push(move)
                
                # If after this we're no longer in check and have a good position
                if not board_copy.is_check():
                    # Further bonus for completely resolving the check situation
                    score += 100
                    
                    # Check if we're attacking anything after this move
                    for target_square in chess.SQUARES:
                        target_piece = board_copy.piece_at(target_square)
                        if target_piece and target_piece.color != our_color:
                            if board_copy.is_attacked_by(our_color, target_square):
                                # We're attacking their pieces after escaping check - great!
                                score += PIECE_VALUES[target_piece.piece_type] // 10
        
        return score
