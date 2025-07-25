# v7p3r_score.py

""" v7p3r Scoring Calculation Module (Original Full Version)
This module is responsible for calculating the score of a chess position based on various factors,
including material balance, piece-square tables, king safety, and other positional features.
It is designed to be used by the v7p3r chess engine.
"""

import chess
import sys
import os
from typing import Optional
from v7p3r_config import v7p3rConfig
from v7p3r_debug import v7p3rLogger, v7p3rUtilities

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup centralized logging for this module
v7p3r_score_logger = v7p3rLogger.setup_logger("v7p3r_score")

class v7p3rScore:
    def __init__(self, rules_manager, pst):
        """ Initialize the scoring calculation engine with configuration settings.  """
        # Engine Configuration
        self.config_manager = v7p3rConfig()
        self.engine_config = self.config_manager.get_engine_config()  # Ensure it's always a dictionary
        self.game_config = self.config_manager.get_game_config()

        # Logging Setup
        self.logger = v7p3r_score_logger
        self.monitoring_enabled = self.engine_config.get('monitoring_enabled', True)
        self.verbose_output_enabled = self.engine_config.get('verbose_output', False)

        # Required Scoring Modules
        self.pst = pst
        self.rules_manager = rules_manager

        # Scoring Setup
        self.ruleset_name = self.engine_config.get('ruleset', 'default_ruleset')
        self.ruleset = self.config_manager.get_ruleset()
        self.use_game_phase = self.engine_config.get('use_game_phase', True)
        self.white_player = self.game_config.get('white_player', 'v7p3r')
        self.black_player = self.game_config.get('black_player', 'v7p3r')
        self.fallback_modifier = 100

        # Initialize scoring parameters
        self.root_board = chess.Board()
        self.game_phase = 'opening'  # Default game phase
        self.endgame_factor = 0.0  # Default endgame factor for endgame awareness
        self.score_counter = 0
        self.score_id = f"score[{self.score_counter}]_{v7p3rUtilities.get_timestamp()}"
        self.fen = self.root_board.fen()
        self.root_move = chess.Move.null()
        self.score = 0.0
        
        # Initialize score dataset
        self.score_dataset = {
            'fen': self.fen,
            'move': self.root_move,
            'piece': None,
            'color': None,
            'current_player': None,
            'self.v7p3r_thinking': False,
            'evaluation': 0.0,
            'game_phase': self.game_phase,
            'endgame_factor': self.endgame_factor,
            'checkmate_threats': 0.0,
            'material_score': 0.0,
            'pst_score': 0.0,
            'piece_captures': 0.0,
            'center_control': 0.0,
            'piece_development': 0.0,
            'board_coverage': 0.0,
            'castling': 0.0,
            'castling_protection': 0.0,
            'piece_coordination': 0.0,
            'rook_coordination': 0.0
        }

    # =================================
    # ===== EVALUATION FUNCTIONS ======

    def evaluate_position(self, board: chess.Board) -> float:
        """Calculate position evaluation from general perspective by delegating to scoring_calculator."""
        if not isinstance(board, chess.Board) or not board.is_valid():
            if self.monitoring_enabled and self.logger:
                self.logger.error(f"[Error] Invalid input for evaluation. Board: {board.fen() if hasattr(board, 'fen') else 'N/A'}")
            return 0.0
        
        white_score = self.calculate_score(
            board=board,
            color=chess.WHITE,
        )
        black_score = self.calculate_score(
            board=board,
            color=chess.BLACK,
        )

        # Return the score from whites perspective
        score = white_score - black_score
        if self.monitoring_enabled and self.logger:
            self.logger.info(f"Evaluation: {score:.3f} | FEN: {board.fen()}")
        self.score_dataset['evaluation'] = score
        return score
    
    def evaluate_position_from_perspective(self, board: chess.Board, color: Optional[chess.Color] = chess.WHITE) -> float:
        """Calculate position evaluation from specified player's perspective by delegating to scoring_calculator."""
        self.color_name = "White" if color == chess.WHITE else "Black"
        
        if not isinstance(color, chess.Color) or not board.is_valid():
            if self.monitoring_enabled and self.logger:
                self.logger.error(f"[Error] Invalid input for evaluation from perspective. Player: {self.color_name}, FEN: {board.fen() if hasattr(board, 'fen') else 'N/A'}")
            return 0.0

        # Game over checks for quick returns
        checkmate_threats_modifier = self.ruleset.get('checkmate_threats_modifier', self.fallback_modifier)
        if board.is_checkmate():
            return -1 * checkmate_threats_modifier if board.turn == color else checkmate_threats_modifier
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        # Calculate direct scores for both sides
        # Use a simplified score calculation for quiescence and deep searches
        # to improve performance
        white_score = self.calculate_score(board=board, color=chess.WHITE)
        black_score = self.calculate_score(board=board, color=chess.BLACK)
        
        # Convert to score from the requested perspective
        if color == chess.WHITE:
            score = white_score - black_score
        else:
            score = black_score - white_score

        if self.monitoring_enabled and self.verbose_output_enabled and self.logger:
            self.logger.info(f"{self.color_name}'s perspective: {score:.3f} | FEN: {board.fen()}")
            
        self.score_dataset['evaluation'] = score
        return score
    
    # ==========================================
    # ========== GAME PHASE CALCULATION ========
    def _calculate_game_phase(self, board: chess.Board):
        """
        Determines the current phase of the game: 'opening', 'middlegame', or 'endgame'.
        Uses material and castling rights as heuristics.
        Sets self.game_phase for use in other scoring functions.
        """
        phase = 'opening'
        endgame_factor = 0.0

        if not self.use_game_phase:
            self.game_phase = 'opening'
            self.endgame_factor = 0.0
            return  # If game phase is not used, default to opening

        # Count total material (excluding kings)
        material = sum([
            len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK))
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        ])
        # Heuristic: opening if all queens/rooks/bishops/knights are present, endgame if queens are gone or little material
        if material <= 8:
            # Endgame Phase
            phase = "endgame"
            # Heuristic: if less than 8 pieces are on the board, endgame is likely
            endgame_factor = 1.0
        elif material <= 20:
            # Middlegame Phase
            phase = "middlegame"
            # Heuristic: if less than 20 pieces are on the board, middlegame is likely
            endgame_factor = 0.5
            if material < 18 and not board.has_castling_rights(chess.WHITE) and not board.has_castling_rights(chess.BLACK):
                # Heuristic: if less than 18 pieces are on the board and both sides no longer have castling rights, unstable position, collapse of opening preparation
                endgame_factor = 0.75
            elif (not board.has_castling_rights(chess.WHITE) or not board.has_castling_rights(chess.BLACK)):
                # Heuristic: some piece exchanges, at least one player has castled, entering middlegame
                endgame_factor = 0.6
        elif material > 20:
            # Opening Phase
            phase = 'opening'
            # Heuristic: if more than 20 pieces are on the board, game is still in opening phase
            endgame_factor = 0.1
            if material < 24 and ((board.has_castling_rights(chess.WHITE) and not board.has_castling_rights(chess.BLACK)) or (not board.has_castling_rights(chess.WHITE) and board.has_castling_rights(chess.BLACK))):
                # Heuristic: multiple pieces exchanged, one side has castled, position is destabilizing
                endgame_factor = 0.5
            elif material <= 28 and (board.has_castling_rights(chess.WHITE) and board.has_castling_rights(chess.BLACK)):
                # Heuristic: piece exchanges but both sides remain un-castled
                endgame_factor = 0.35
            elif material < 32 and (board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK)):
                # Heuristic: at least one exchange, game just beginning
                endgame_factor = 0.25
            elif material == 32:
                # Heuristic: all material remains on the board, fully stable/closed position, game beginning, early play
                endgame_factor = 0.0
            
        self.game_phase = phase
        self.endgame_factor = endgame_factor
        self.score_dataset['game_phase'] = phase
        self.score_dataset['endgame_factor'] = endgame_factor
        self.score_dataset['material'] = material
        if self.monitoring_enabled and self.logger:
            self.logger.info(f"[Game Phase] Current phase: {self.game_phase} | Endgame factor: {self.endgame_factor:.2f} | FEN: {board.fen()}")
        return
    
    # ==========================================
    # ========= CALCULATION FUNCTION ===========
    def calculate_score(self, board: chess.Board, color: chess.Color, endgame_factor: float = 0.0) -> float:
        """
        Calculates the position evaluation score for a given board and color,
        applying dynamic ruleset settings and endgame awareness.
        This is the main public method for this class.
        """
        score = 0.0
        
        # Update scoring dictionary with minimal position information
        self.score_dataset['fen'] = board.fen()
        self.color_name = "White" if color == chess.WHITE else "Black"
        self.score_dataset['color'] = self.color_name
        
        # Only log detailed info when monitoring is enabled and verbose is true
        detailed_logging = self.monitoring_enabled and self.verbose_output_enabled and self.logger
        
        # Optimize computation: Start with most important factors first
        
        # CHECKMATE THREATS - Most critical factor
        checkmate_threats_score = self.rules_manager._checkmate_threats(board, color) or 0.0
        score += checkmate_threats_score
        
        # MATERIAL SCORE - Second most important factor
        material_score = self.rules_manager._material_score(board, color) or 0.0
        score += material_score
        
        # Store these critical values for logging
        self.score_dataset['checkmate_threats'] = checkmate_threats_score
        self.score_dataset['material_score'] = material_score
        
        # Decide if we need more detailed evaluation
        # Early return if we're in a very clear position (queen + multiple pieces advantage)
        if abs(material_score) > 1500 and not board.is_check():  # Big material advantage and not in check
            if detailed_logging:
                self.logger.info(f"[Scoring Calc] Early return due to large material advantage: {material_score:.3f}")
            
            # Ensure the score is within reasonable bounds
            material_score = max(-2000, min(2000, material_score))
            return material_score
        
        # GAME PHASE
        if self.use_game_phase:
            self._calculate_game_phase(board)
        
        # PST SCORE
        pst_score = self.rules_manager._pst_score(board, color, self.endgame_factor) or 0.0
        score += pst_score
        self.score_dataset['pst_score'] = pst_score
        
        # PIECE CAPTURES
        piece_captures_score = self.rules_manager._piece_captures(board, color) or 0.0
        score += piece_captures_score
        self.score_dataset['piece_captures'] = piece_captures_score
        
        # CENTER CONTROL
        center_control_score = self.rules_manager._center_control(board, color) or 0.0
        score += center_control_score
        self.score_dataset['center_control'] = center_control_score
        
        # PIECE DEVELOPMENT
        piece_development_score = self.rules_manager._piece_development(board, color) or 0.0
        score += piece_development_score
        self.score_dataset['piece_development'] = piece_development_score
        
        # Normalize score to reasonable centipawn range if needed
        # Only normalize if it's not a checkmate score
        if abs(score) < 9000:  # Not a checkmate or near-checkmate evaluation
            # Clip extreme scores that aren't mate scores
            score = max(-2000, min(2000, score))
        
        # Log only the final score unless detailed logging is enabled
        if self.monitoring_enabled and self.logger:
            self.logger.info(f"[Scoring Calc] Final score for {self.color_name}: {score:.3f}")
            
        self.score_dataset['evaluation'] = score
        return score
