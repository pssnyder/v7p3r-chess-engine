"""
V18.2: Modular Position Evaluation
Executes only the modules selected by the current profile.

Philosophy:
- DESPERATE mode: Skip 22 strategic modules, run only 10 tactical modules
- EMERGENCY mode: Minimal 5-module evaluation for time pressure
- FAST mode: Balanced 12-18 modules for speed
- TACTICAL mode: Full tactical suite (18-22 modules)
- ENDGAME mode: Endgame-specific subset (10-15 modules)
- COMPREHENSIVE mode: All relevant modules (20-28 modules)

Author: Pat Snyder
Created: 2025-12-27
"""

import chess
from typing import Dict, Set
from v7p3r_position_context import PositionContext
from v7p3r_eval_selector import EvaluationProfile


class ModularEvaluator:
    """Executes only selected evaluation modules"""
    
    def __init__(self, fast_evaluator):
        """
        Initialize with reference to existing fast evaluator for module implementations.
        
        Args:
            fast_evaluator: The v7p3r_fast_evaluator.FastEvaluator instance
        """
        self.fast_eval = fast_evaluator
        
        # Piece values for material counting
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
    
    def evaluate_with_profile(self, board: chess.Board, profile: EvaluationProfile, 
                               context: PositionContext) -> float:
        """
        Evaluate position using selected profile's modules.
        
        **V18.3 ACTUAL MODULAR EXECUTION**:
        Execute only the evaluation components in the profile.
        
        KEY OPTIMIZATION:
        - DESPERATE/EMERGENCY/FAST: Material + PST only (skip strategic) → 2-3x faster
        - Full profiles: All components for complete evaluation
        
        Expected: DESPERATE depth 8-9 (vs baseline 6.0 with monolithic)
        
        Args:
            board: Chess position to evaluate
            profile: Selected evaluation profile with active modules
            context: Position context
        
        Returns:
            Evaluation score (centipawns, current player perspective)
        """
        # Get active modules for O(1) lookup
        active_modules = set(profile.active_modules)
        
        # Check if we need strategic evaluation (the expensive part)
        needs_strategic = any(module in active_modules for module in [
            'king_safety_basic', 'king_safety_complex', 'pawn_structure',
            'rook_open_files', 'bishop_pair', 'knight_outposts',
            'center_control', 'space_advantage', 'piece_mobility',
            'passed_pawns', 'pawn_chains', 'isolated_pawns',
            'doubled_pawns', 'backward_pawns'
        ])
        
        # FAST PATH: Material + PST only (DESPERATE, EMERGENCY, FAST modes)
        if not needs_strategic:
            material = self.fast_eval.evaluate_material(board)
            pst = self.fast_eval.evaluate_pst(board)
            # Combine with standard weights: 60% PST + 40% Material
            score = int(pst * 0.6 + material * 0.4)
            return score if board.turn == chess.WHITE else -score
        
        # FULL PATH: All components (TACTICAL, ENDGAME, COMPREHENSIVE modes)
        score = self.fast_eval.evaluate(board)
        return score
    
    # ==================== MATERIAL & PST ====================
    
    def _evaluate_material(self, board: chess.Board) -> float:
        """Material count - most critical module"""
        white_material = 0
        black_material = 0
        
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            white_material += len(board.pieces(piece_type, chess.WHITE)) * self.piece_values[piece_type]
            black_material += len(board.pieces(piece_type, chess.BLACK)) * self.piece_values[piece_type]
        
        diff = white_material - black_material
        return diff if board.turn == chess.WHITE else -diff
    
    def _evaluate_pst(self, board: chess.Board) -> float:
        """Piece-square table evaluation"""
        # Delegate to fast evaluator's PST logic
        white_score = self.fast_eval._evaluate_piece_placement(board, chess.WHITE)
        black_score = self.fast_eval._evaluate_piece_placement(board, chess.BLACK)
        diff = white_score - black_score
        return diff if board.turn == chess.WHITE else -diff
    
    # ==================== TACTICAL MODULES ====================
    
    def _evaluate_hanging_pieces(self, board: chess.Board) -> float:
        """Detect undefended pieces (critical for DESPERATE mode)"""
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Check if piece is attacked and not defended
                attackers = len(board.attackers(not piece.color, square))
                defenders = len(board.attackers(piece.color, square))
                
                if attackers > 0 and defenders == 0:
                    # Hanging piece penalty
                    penalty = self.piece_values[piece.piece_type] * 0.5
                    if piece.color == board.turn:
                        score -= penalty
                    else:
                        score += penalty
        
        return score
    
    def _evaluate_captures(self, board: chess.Board) -> float:
        """Evaluate available captures"""
        score = 0
        for move in board.legal_moves:
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured:
                    score += self.piece_values[captured.piece_type] * 0.1
        return score
    
    def _evaluate_checks(self, board: chess.Board) -> float:
        """Bonus for checking moves"""
        score = 0
        for move in board.legal_moves:
            if board.gives_check(move):
                score += 15  # Small bonus for check availability
        return score
    
    def _evaluate_tactical_patterns(self, board: chess.Board) -> float:
        """Detect pins, forks, skewers (simplified)"""
        # Placeholder - full implementation would detect these patterns
        return 0
    
    def _evaluate_exchanges(self, board: chess.Board) -> float:
        """Static Exchange Evaluation (simplified)"""
        # Placeholder - full SEE implementation
        return 0
    
    def _evaluate_trapped_pieces(self, board: chess.Board) -> float:
        """Detect trapped pieces"""
        # Placeholder
        return 0
    
    def _evaluate_back_rank(self, board: chess.Board) -> float:
        """Back rank weakness detection"""
        # Placeholder
        return 0
    
    # ==================== STRATEGIC MODULES ====================
    
    def _evaluate_king_safety_basic(self, board: chess.Board) -> float:
        """Basic king safety (castling rights, pawn shield)"""
        score = 0
        
        # Castling rights bonus
        if board.has_kingside_castling_rights(board.turn):
            score += 15
        if board.has_queenside_castling_rights(board.turn):
            score += 10
        
        if board.has_kingside_castling_rights(not board.turn):
            score -= 15
        if board.has_queenside_castling_rights(not board.turn):
            score -= 10
        
        return score
    
    def _evaluate_king_safety_complex(self, board: chess.Board) -> float:
        """Advanced king safety (attacks near king)"""
        # Placeholder
        return 0
    
    def _evaluate_move_safety(self, board: chess.Board) -> float:
        """Basic move safety check"""
        # Simplified - just check if we're leaving pieces hanging
        return 0  # Fast evaluator handles this internally
    
    def _evaluate_pawn_structure(self, board: chess.Board) -> float:
        """Overall pawn structure quality"""
        # Delegate to fast evaluator
        white_score = self.fast_eval._evaluate_pawn_structure(board, chess.WHITE)
        black_score = self.fast_eval._evaluate_pawn_structure(board, chess.BLACK)
        diff = white_score - black_score
        return diff if board.turn == chess.WHITE else -diff
    
    def _evaluate_passed_pawns(self, board: chess.Board) -> float:
        """Passed pawn bonus"""
        # Placeholder
        return 0
    
    def _evaluate_pawn_chains(self, board: chess.Board) -> float:
        """Connected pawn bonus"""
        # Placeholder
        return 0
    
    def _evaluate_isolated_pawns(self, board: chess.Board) -> float:
        """Isolated pawn penalty"""
        # Placeholder
        return 0
    
    def _evaluate_backward_pawns(self, board: chess.Board) -> float:
        """Backward pawn penalty"""
        return 0
    
    def _evaluate_doubled_pawns(self, board: chess.Board) -> float:
        """Doubled pawn penalty"""
        # Placeholder
        return 0
    
    def _evaluate_bishop_pair(self, board: chess.Board) -> float:
        """Bishop pair bonus"""
        white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
        
        score = 0
        if white_bishops >= 2:
            score += 30
        if black_bishops >= 2:
            score -= 30
        
        return score if board.turn == chess.WHITE else -score
    
    def _evaluate_knight_outposts(self, board: chess.Board) -> float:
        """Knight outpost bonus"""
        # Placeholder
        return 0
    
    def _evaluate_rook_files(self, board: chess.Board) -> float:
        """Rook on open/semi-open file"""
        # Placeholder
        return 0
    
    def _evaluate_rook_seventh(self, board: chess.Board) -> float:
        """Rook on 7th rank bonus"""
        # Placeholder
        return 0
    
    def _evaluate_connected_rooks(self, board: chess.Board) -> float:
        """Connected rooks bonus"""
        # Placeholder
        return 0
    
    def _evaluate_queen_activity(self, board: chess.Board) -> float:
        """Queen mobility/centralization"""
        # Placeholder
        return 0
    
    def _evaluate_mobility(self, board: chess.Board) -> float:
        """Piece mobility"""
        our_moves = len(list(board.legal_moves))
        
        board.push(chess.Move.null())
        their_moves = len(list(board.legal_moves)) if board.legal_moves else 0
        board.pop()
        
        return (our_moves - their_moves) * 2
    
    def _evaluate_center_control(self, board: chess.Board) -> float:
        """Central square control"""
        # Placeholder
        return 0
    
    def _evaluate_space(self, board: chess.Board) -> float:
        """Space advantage"""
        # Placeholder
        return 0
    
    def _evaluate_development(self, board: chess.Board) -> float:
        """Piece development in opening"""
        # Placeholder
        return 0
    
    def _evaluate_tempo(self, board: chess.Board) -> float:
        """Time/tempo evaluation"""
        # Placeholder
        return 0
    
    # ==================== ENDGAME MODULES ====================
    
    def _evaluate_endgame_patterns(self, board: chess.Board, context: PositionContext) -> float:
        """Known endgame patterns (KPK, etc.)"""
        # Placeholder
        return 0
    
    def _evaluate_king_activity_endgame(self, board: chess.Board) -> float:
        """King centralization in endgame"""
        # Placeholder
        return 0
    
    def _evaluate_pawn_races(self, board: chess.Board) -> float:
        """Pawn race evaluation"""
        # Placeholder
        return 0
    
    def _evaluate_opposition(self, board: chess.Board) -> float:
        """King opposition in endgame"""
        return 0
    
    def _evaluate_zugzwang(self, board: chess.Board) -> float:
        """Zugzwang detection"""
        return 0
    
    def _evaluate_repetition(self, board: chess.Board) -> float:
        """Repetition penalty"""
        return 0
