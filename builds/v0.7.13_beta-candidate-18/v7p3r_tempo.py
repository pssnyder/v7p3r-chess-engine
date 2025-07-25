# v7p3r_tempo.py
"""V7P3R Tempo and Risk Management Module
This module manages the tempo and risk assessment for the V7P3R chess engine.
Tempo in this context is defined as the positions that keep play moving vs stop it,
such as drawing move prevention, drawish position avoidance, and waiting moves.
Risk in this context is defined as critical game ending scenarios,
such as being checkmated, force stalemated, and stalemating the opponent."""

import chess
from v7p3r_config import v7p3rConfig
from typing import Dict, Tuple, Optional, List

class v7p3rTempo:
    def __init__(self, config=None, pst=None, rules=None):
        # Engine Configuration
        self.config_manager = v7p3rConfig() if config is None else config
        self.engine_config = self.config_manager.get_engine_config()
        self.game_config = self.config_manager.get_game_config()
        
        # Shared components
        self.pst = pst
        self.rules = rules
        
        # Configuration modifiers
        self.fallback_modifier = 100
        self.tempo_modifier = self.game_config.get('tempo_modifier', 0.1)
        self.zugzwang_modifier = self.game_config.get('zugzwang_modifier', -0.2)
        self.draw_modifier = self.game_config.get('draw_modifier', -500000.0)
        self.stalemate_modifier = self.game_config.get('stalemate_modifier', -1000000.0)
        
        # Game phase tracking
        self.game_phase = 'opening'
        self.endgame_factor = 0.0
        self.use_game_phase = True
        self.position_history = {}  # {FEN: count}
        
        # Internal state tracking
        self.position_scores = {}  # {FEN: score}
        self.waiting_moves = {}    # {FEN: List[chess.Move]}

    def get_game_phase_factor(self, board: chess.Board) -> float:
        """
        Get the current game phase factor (0.0 for opening to 1.0 for endgame).
        This method is used by move ordering and evaluation to properly weight moves.
        """
        # Calculate only if needed
        if not hasattr(self, '_cached_phase_factor') or board.fen() not in self._cached_phase_factor:
            _, factor = self.calculate_game_phase(board)
            if not hasattr(self, '_cached_phase_factor'):
                self._cached_phase_factor = {}
            self._cached_phase_factor[board.fen()] = factor
        return self._cached_phase_factor.get(board.fen(), 0.0)

    def calculate_game_phase(self, board: chess.Board) -> Tuple[str, float]:
        """
        Determine the current game phase and endgame factor.
        Uses piece count and development to assess game phase.
        
        Returns:
            Tuple[str, float]: (phase, endgame_factor)
            phase: 'opening', 'middlegame', or 'endgame'
            endgame_factor: 0.0 (pure opening) to 1.0 (pure endgame)
        """
        if not self.use_game_phase:
            return 'opening', 0.0
            
        # Count material (in centipawns for consistency with PST values)
        total_material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                total_material += self.pst.piece_values.get(piece.piece_type, 0)
                
        # Max material (excluding kings) = 2 * (900 + 2*500 + 2*330 + 2*320 + 8*100)
        max_material = 2 * (900 + 2*500 + 2*330 + 2*320 + 8*100)  # 7780
        material_factor = 1.0 - (total_material / max_material)
        
        # Development factor (penalize undeveloped pieces in opening)
        development_penalty = 0
        if total_material > max_material * 0.8:  # Still in opening/early middlegame
            for color in [chess.WHITE, chess.BLACK]:
                development_penalty += (
                    self._count_undeveloped_pieces(board, color) * 
                    (0.1 if material_factor < 0.3 else 0.05)
                )
        
        # Combine factors
        endgame_factor = min(1.0, max(0.0, material_factor + development_penalty))
        
        # Determine phase
        if endgame_factor < 0.3:
            phase = 'opening'
        elif endgame_factor < 0.7:
            phase = 'middlegame'
        else:
            phase = 'endgame'
            
        # Cache results
        self.game_phase = phase
        self.endgame_factor = endgame_factor
        
        return phase, endgame_factor

    def _count_undeveloped_pieces(self, board: chess.Board, color: bool) -> int:
        """Count pieces that haven't moved from their starting squares."""
        undeveloped = 0
        back_rank = 0 if color == chess.WHITE else 7
        
        # Check knights and bishops still on back rank
        piece_types = [chess.KNIGHT, chess.BISHOP]
        for file in range(8):
            square = chess.square(file, back_rank)
            piece = board.piece_at(square)
            if piece and piece.color == color and piece.piece_type in piece_types:
                undeveloped += 1
                
        return undeveloped

    def assess_tempo(self, board: chess.Board, color: chess.Color) -> float:
        """
        Assess the tempo advantage in the position.
        A positive score means the position favors active play for color.
        """
        score = 0.0
        phase, endgame_factor = self.calculate_game_phase(board)
        
        # Initial position and early moves handling
        if len(board.move_stack) == 0:
            if color == chess.WHITE:
                return self.tempo_modifier  # White's opening advantage
            else:
                return -self.tempo_modifier  # Black's opening disadvantage
                
        # Development tempo (opening/early middlegame)
        if phase == 'opening':
            development_score = self._assess_development_tempo(board, color)
            score += development_score * 2.0  # Double importance in opening
            
            # Extra importance to center control in first moves
            if len(board.move_stack) < 4:
                central_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
                our_center = sum(1 for sq in central_squares if board.is_attacked_by(color, sq))
                their_center = sum(1 for sq in central_squares if board.is_attacked_by(not color, sq))
                score += (our_center - their_center) * self.tempo_modifier
            
        # Right to move bonus
        if board.turn == color:
            score += self.tempo_modifier * (1.0 if phase == 'opening' else 0.5)
        
        # Piece mobility
        mobility_score = self._assess_piece_mobility(board, color)
        score += mobility_score * (2.0 if phase == 'opening' else 1.0)  # More important in opening
        
        return score

    def _assess_development_tempo(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate development speed and piece coordination"""
        score = 0.0
        
        # Development tracking
        our_developed = 0
        their_developed = 0
        
        # Count developed pieces and penalize early queen moves
        back_rank = chess.A1 if color == chess.WHITE else chess.A8
        queen_start = chess.D1 if color == chess.WHITE else chess.D8
        
        # Check minor piece development
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for piece in board.pieces(piece_type, color):
                if piece not in [back_rank + 1, back_rank + 2, back_rank + 5, back_rank + 6]:
                    our_developed += 1
            for piece in board.pieces(piece_type, not color):
                opponent_back = chess.A8 if color == chess.WHITE else chess.A1
                if piece not in [opponent_back + 1, opponent_back + 2, opponent_back + 5, opponent_back + 6]:
                    their_developed += 1
        
        # Early queen development penalty (first 6 moves)
        if len(board.move_stack) < 12:  # 6 moves per side
            queen_squares = board.pieces(chess.QUEEN, color)
            for square in queen_squares:
                if square != queen_start:
                    score -= 0.2  # Penalty for early queen development
        
        # Check central pawn control
        central_files = ['d', 'e']
        central_ranks = ['4', '5'] if color == chess.WHITE else ['4', '5']
        
        our_center_pawns = 0
        their_center_pawns = 0
        
        for file in central_files:
            for rank in central_ranks:
                square = chess.parse_square(file + rank)
                piece = board.piece_at(square)
                if piece:
                    if piece.piece_type == chess.PAWN:
                        if piece.color == color:
                            our_center_pawns += 1
                        else:
                            their_center_pawns += 1
                            
        # Center control bonus
        score += (our_center_pawns - their_center_pawns) * 0.1
                    
        # Development advantage
        score += (our_developed - their_developed) * 0.15
        
        return score * self.tempo_modifier

    def _assess_piece_mobility(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate piece mobility and control of key squares"""
        score = 0.0
        
        # Store original turn
        original_turn = board.turn
        
        # Count legal moves for both sides
        board.turn = color
        our_moves = len(list(board.legal_moves))
        # Count central square control
        central_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        our_center_control = sum(1 for sq in central_squares if board.is_attacked_by(color, sq))
        
        # Now check opponent's mobility
        board.turn = not color
        their_moves = len(list(board.legal_moves))
        their_center_control = sum(1 for sq in central_squares if board.is_attacked_by(not color, sq))
        
        # Restore original turn
        board.turn = original_turn
        
        # Scale factors
        mobility_value = 0.01  # Small bonus per legal move
        center_control_value = 0.05  # Bonus for each central square controlled
        
        # Mobility differential (normalize by total moves)
        total_moves = max(our_moves + their_moves, 1)
        normalized_mobility = (our_moves - their_moves) / total_moves
        score += normalized_mobility * mobility_value
        
        # Center control differential
        score += (our_center_control - their_center_control) * center_control_value
        
        return score

    def assess_zugzwang(self, board: chess.Board, color: chess.Color) -> float:
        """
        Detect and evaluate potential zugzwang positions.
        Returns a negative score if the position is likely to force disadvantageous moves.
        """
        score = 0.0
        
        # Get current position key
        fen_key = board.fen().split(' ')[0]  # Position only, ignore move counts
        current_score = self.position_scores.get(fen_key, None)
        
        # Even without history, check for immediate zugzwang signs
        if board.turn == color:
            legal_moves = list(board.legal_moves)
            useful_moves = 0
            forced_moves = 0
            
            if len(legal_moves) < 5:  # Limited options, possible zugzwang
                for move in legal_moves:
                    board_copy = board.copy()
                    board_copy.push(move)
                    
                    # Check if move is forced
                    if board.is_check() or len(list(board.attackers(not color, move.from_square))) > 0:
                        forced_moves += 1
                    # Check if move is useful (improves position)
                    elif not board_copy.is_check() and not board.is_capture(move):
                        useful_moves += 1
            
                # Penalize positions with many forced moves or few useful moves
                if forced_moves > 0:
                    score += self.zugzwang_modifier * forced_moves
                if useful_moves < 2:  # Need some freedom of choice
                    score += self.zugzwang_modifier * (2 - useful_moves)
                    
        # If we have position history, consider score trend
        if current_score is not None:
            waiting_moves = self.waiting_moves.get(fen_key, [])
            
            # More sophisticated historical analysis
            if len(waiting_moves) < 3 and board.turn == color:
                score += self.zugzwang_modifier * (3 - len(waiting_moves))
                
                # Check if any previous waiting moves remain useful
                useful_waiting_moves = 0
                for move in waiting_moves:
                    board_copy = board.copy()
                    try:
                        board_copy.push(move)
                        if not board_copy.is_check() and not board.is_capture(move):
                            useful_waiting_moves += 1
                    except:
                        continue  # Move might be invalid in current position
                
                # Extra penalty if previously useful moves are no longer available
                if useful_waiting_moves < len(waiting_moves):
                    score += self.zugzwang_modifier * (len(waiting_moves) - useful_waiting_moves)
                
        return score

    def assess_checkmate_threats(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate checkmate threats both for and against"""
        score = 0.0
        
        # First check if we can checkmate
        if mate_move := self.rules.find_checkmate_in_n(board, 5, color):
            return 9999.0  # Maximum score for winning
            
        # Then check if we're in danger of being mated
        opponent_board = board.copy()
        opponent_board.turn = not color
        if self.rules.find_checkmate_in_n(opponent_board, 4, not color):
            score -= 9999.0  # Maximum penalty for losing
            
        return score

    def assess_stalemate_threats(self, board: chess.Board, color: chess.Color) -> float:
        """
        Evaluate stalemate possibilities and find moves that avoid or create them 
        based on whether we're winning or losing.
        """
        score = 0.0
        
        # Check material balance to determine if we want to encourage/discourage draws
        if self.rules.material_count(board, color) > 200:  # Winning by 2 pawns or more
            # Penalize moves that allow stalemate chances
            if self.rules.is_stalemate_threatened(board):
                score += self.stalemate_modifier
        else:
            # In losing positions, don't penalize drawing chances as heavily
            score += self.stalemate_modifier * 0.5
            
        return score

    def assess_drawish_positions(self, board: chess.Board) -> float:
        """
        Evaluate how likely the position is to be drawn and apply appropriate
        penalties/bonuses based on the game situation.
        """
        score = 0.0
        
        # Check for basic drawn positions
        if self.rules.is_likely_draw(board):
            return self.draw_modifier
            
        # Track position repetitions
        fen_key = board.fen().split(' ')[0]  # Position only, ignore move counts
        self.position_history[fen_key] = self.position_history.get(fen_key, 0) + 1
        
        # Penalize repetitions
        if self.position_history[fen_key] > 1:
            score += self.draw_modifier * (self.position_history[fen_key] - 1) * 0.3
            
        return score

    def update_position_score(self, board: chess.Board, score: float):
        """Update the historical score for a position"""
        fen_key = board.fen().split(' ')[0]  # Position only, ignore move counts
        self.position_scores[fen_key] = score

    def update_position_history(self, fen: str) -> None:
        """Update the position history with a new position."""
        if fen in self.position_history:
            self.position_history[fen] += 1
        else:
            self.position_history[fen] = 1

    def get_repetition_count(self, fen: str) -> int:
        """Get how many times a position has occurred."""
        return self.position_history.get(fen, 0)

    def assess_position(self, board: chess.Board, color: chess.Color) -> dict:
        """Assess a position's tempo and risk characteristics.
        Returns dict with:
            - game_phase: str
            - endgame_factor: float
            - tempo_score: float
            - risk_score: float
            - zugzwang_risk: float
        """
        # Get game phase
        phase, endgame_factor = self.calculate_game_phase(board)
        
        # Base tempo assessment
        tempo_score = self.assess_tempo(board, color)
        
        # Calculate risk factors
        risk_score = 0.0
        zugzwang_risk = 0.0
        
        # 1. Check immediate threats
        if board.is_check():
            risk_score -= 0.5  # Being in check is risky
            
        # 2. Assess material tension
        for move in board.legal_moves:
            if board.is_capture(move):
                risk_score += 0.1  # More tension means more risk
                
        # 3. Check repetition risk
        rep_count = self.get_repetition_count(board.fen())
        if rep_count > 1:
            risk_score -= 0.2 * rep_count  # Penalize repetitions
            
        # 4. Assess zugzwang risk in endgame
        if endgame_factor > 0.7:  # Deep in endgame
            moves = list(board.legal_moves)
            if len(moves) < 5:  # Few moves available
                # Look for forced moves
                all_moves_lose_material = True
                for move in moves:
                    new_board = board.copy()
                    new_board.push(move)
                    if not self._move_loses_material(move, board):
                        all_moves_lose_material = False
                        break
                        
                if all_moves_lose_material:
                    zugzwang_risk = -1.0  # High risk of zugzwang
                    
        return {
            'game_phase': phase,
            'endgame_factor': endgame_factor,
            'tempo_score': tempo_score,
            'risk_score': risk_score,
            'zugzwang_risk': zugzwang_risk
        }

    def _move_loses_material(self, move: chess.Move, board: chess.Board) -> bool:
        """Check if a move immediately loses material."""
        # Simple material counting
        material_before = self._count_material(board)
        
        # Make move
        new_board = board.copy()
        new_board.push(move)
        material_after = self._count_material(new_board)
        
        return material_after < material_before
        
    def _count_material(self, board: chess.Board) -> int:
        """Count total material on board."""
        total = 0
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        for piece_type in piece_values:
            total += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
            total += len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
            
        return total