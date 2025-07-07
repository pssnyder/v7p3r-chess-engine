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
        current_turn = board.turn
        self.color_name = "White" if color == chess.WHITE else "Black"
        
        if not isinstance(color, chess.Color) or color != current_turn or not board.is_valid():
            if self.monitoring_enabled and self.logger:
                self.logger.error(f"[Error] Invalid input for evaluation from perspective. Player: {self.color_name}, Turn: {board.turn}, FEN: {board.fen() if hasattr(board, 'fen') else 'N/A'}")
            return 0.0

        score = self.calculate_score(board=board,color=color)

        if self.monitoring_enabled and self.logger:
            self.logger.info(f"{self.color_name}'s perspective: {score:.3f} | FEN: {board.fen()}")
        self.score_dataset['evaluation'] = score
        return score
    
    # ==========================================
    # ========== GAME PHASE CALCULATION ========
    def calculate_game_phase(self, board: chess.Board):
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
        
        # Update scoring dictionary with current position information
        self.score_dataset['fen'] = board.fen()
        last_move = board.peek() if board.move_stack else chess.Move.null()
        self.score_dataset['move'] = last_move
        self.score_dataset['piece'] = board.piece_type_at(last_move.to_square) if last_move else None

        # Helper for consistent color display in logs
        self.color_name = "White" if color == chess.WHITE else "Black"
        self.score_dataset['color'] = self.color_name
        self.current_player_name = self.game_config.get('white_player','') if color == chess.WHITE else self.game_config.get('black_player','')
        self.score_dataset['current_player'] = self.current_player_name
        self.v7p3r_thinking = self.current_player_name == 'v7p3r'
        self.score_dataset['v7p3r_thinking'] = self.v7p3r_thinking
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Starting score calculation for {self.current_player_name} engine as {self.color_name} | Ruleset: {self.ruleset_name} | FEN: {board.fen()}")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Starting score calculation for {self.current_player_name} engine as {self.color_name} | Ruleset: {self.ruleset_name} | FEN: {board.fen()}")
        
        # GAME PHASE
        if self.use_game_phase:
            self.calculate_game_phase(board)
        else:
            self.game_phase = 'opening'
            self.endgame_factor = 0.0

        # CHECKMATE THREATS
        checkmate_threats_score = 1.0 * (self.rules_manager._checkmate_threats(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Checkmate threats score for {self.color_name}: {checkmate_threats_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Checkmate threats score for {self.color_name}: {checkmate_threats_score:.3f} (Ruleset: {self.ruleset_name})")
        score += checkmate_threats_score
        self.score_dataset['checkmate_threats'] = checkmate_threats_score

        # MATERIAL SCORE
        material_score = self.rules_manager._material_score(board, color) or 0.0
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Material score for {self.color_name}: {material_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Material score for {self.color_name}: {material_score:.3f} (Ruleset: {self.ruleset_name})")
        score += 1.0 * material_score
        self.score_dataset['material_score'] = 1.0 * self.rules_manager._material_score(board, color)
        
        # MATERIAL COUNT
        material_count_score = 1.0 * (self.rules_manager._material_count(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Material count score for {self.color_name}: {material_count_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Material count score for {self.color_name}: {material_count_score:.3f} (Ruleset: {self.ruleset_name})")
        score += 1.0 * material_count_score
        self.score_dataset['material_count'] = 1.0 * self.rules_manager._material_count(board, color)

        # PIECE-SQUARE TABLE SCORE
        pst_board_score = self.pst.evaluate_board_position(board, color) or 0.0
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Piece-square table score for {self.color_name}: {pst_board_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Piece-square table score for {self.color_name}: {pst_board_score:.3f} (Ruleset: {self.ruleset_name})")
        score += 1.0 * pst_board_score
        self.score_dataset['pst_score'] = 1.0 * pst_board_score

        # PIECE CAPTURES
        piece_captures_score = 1.0 * (self.rules_manager._piece_captures(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Piece captures score for {self.color_name}: {piece_captures_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Piece captures score for {self.color_name}: {piece_captures_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_captures_score
        self.score_dataset['piece_captures'] = piece_captures_score

        # CENTER CONTROL
        center_control_score = 1.0 * (self.rules_manager._center_control(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Center control score for {self.color_name}: {center_control_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Center control score for {self.color_name}: {center_control_score:.3f} (Ruleset: {self.ruleset_name})")
        score += center_control_score
        self.score_dataset['center_control'] = center_control_score

        # PIECE DEVELOPMENT
        piece_development_score = 1.0 * (self.rules_manager._piece_development(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Piece development score for {self.color_name}: {piece_development_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Piece development score for {self.color_name}: {piece_development_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_development_score
        self.score_dataset['piece_development'] = piece_development_score

        # BOARD COVERAGE
        board_coverage_score = 1.0 * (self.rules_manager._board_coverage(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Board coverage score for {self.color_name}: {board_coverage_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Board coverage score for {self.color_name}: {board_coverage_score:.3f} (Ruleset: {self.ruleset_name})")
        score += board_coverage_score
        self.score_dataset['board_coverage'] = board_coverage_score

        # CASTLING
        castling_score = 1.0 * (self.rules_manager._castling(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Castling score for {self.color_name}: {castling_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Castling score for {self.color_name}: {castling_score:.3f} (Ruleset: {self.ruleset_name})")
        score += castling_score
        self.score_dataset['castling'] = castling_score

        # CASTLING PROTECTION
        castling_protection_score = 1.0 * (self.rules_manager._castling_protection(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Castling protection score for {self.color_name}: {castling_protection_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Castling protection score for {self.color_name}: {castling_protection_score:.3f} (Ruleset: {self.ruleset_name})")
        score += castling_protection_score
        self.score_dataset['castling_protection'] = castling_protection_score

        # PIECE COORDINATION
        piece_coordination_score = 1.0 * (self.rules_manager._piece_coordination(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Piece coordination score for {self.color_name}: {piece_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Piece coordination score for {self.color_name}: {piece_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_coordination_score
        self.score_dataset['piece_coordination'] = piece_coordination_score

        # ROOK COORDINATION
        rook_coordination_score = 1.0 * (self.rules_manager._rook_coordination(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Rook coordination score for {self.color_name}: {rook_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Rook coordination score for {self.color_name}: {rook_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        score += rook_coordination_score
        self.score_dataset['rook_coordination'] = rook_coordination_score
        
        # FINAL SCORE
        self.score_dataset['score'] = score  # Update final score in scoring dictionary
        
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Final score for {self.current_player_name} as {self.color_name}: {score:.3f} | Ruleset: {self.ruleset_name} | FEN: {board.fen()}")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Final score for {self.current_player_name} as {self.color_name}: {score:.3f} | Ruleset: {self.ruleset_name} | FEN: {board.fen()}")
        return score