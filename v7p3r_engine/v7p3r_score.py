# v7p3r_score.py

""" v7p3r Scoring Calculation Module (Original Full Version)
This module is responsible for calculating the score of a chess position based on various factors,
including material balance, piece-square tables, king safety, and other positional features.
It is designed to be used by the v7p3r chess engine.
"""

import chess
import yaml
import sys
import os
import logging
import datetime
from typing import Optional

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
log_filename = f"v7p3r_score_{timestamp}.log"
log_file_path = os.path.join(log_dir, log_filename)

v7p3r_score_logger = logging.getLogger(f"v7p3r_score_{timestamp}")
v7p3r_score_logger.setLevel(logging.DEBUG)

if not v7p3r_score_logger.handlers:
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10*1024*1024,
        backupCount=3,
        delay=True
    )
    formatter = logging.Formatter(
        '%(asctime)s | %(funcName)-15s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    v7p3r_score_logger.addHandler(file_handler)
    v7p3r_score_logger.propagate = False

class v7p3rScore:
    def __init__(self, engine_config: dict, pst, logger: logging.Logger):
        """ Initialize the scoring calculation engine with configuration settings.  """
        # Engine Configuration
        self.engine_config = engine_config

        # Scoring Config
        self.logger = v7p3r_score_logger
        self.print_scoring = engine_config.get('verbose_output', False)
        self.ruleset_name = engine_config.get('engine_ruleset', 'default_evaluation')
        self.use_game_phase = engine_config.get('use_game_phase', False)
        self.monitoring_enabled = engine_config.get('monitoring_enabled', False)
        self.verbose_output_enabled = engine_config.get('verbose_output', False)
        self.game_phase = 'opening'  # Default game phase
        self.game_factor = 0.0  # Default game factor for endgame awareness
        self.fallback_modifier = 100
        self.scoring = {
            'move': chess.Move.null(),
            'score': 0.0,
            'game_phase': 'opening',
            'endgame_factor': 0.0,
        }
        
        # Required Scoring Modules
        self.pst = pst

        # Ruleset Loading
        self.rulesets = {}
        self.rules = {}
        try:
            with open("v7p3r_engine/rulesets/rulesets.yaml") as f:
                self.rulesets = yaml.safe_load(f) or {}
                self.rules = self.rulesets.get(f"{self.ruleset_name}", {})
                if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                    self.logger.debug(f"v7p3rScoringCalculation initialized with ruleset: {self.ruleset_name}")
                    self.logger.debug(f"Current ruleset parameters: {self.rules}")
        except Exception as e:
            if self.logger and self.monitoring_enabled:
                self.logger.error(f"[Error] Could not load rulesets: {e}")

    # =================================
    # ===== EVALUATION FUNCTIONS ======

    def evaluate_position(self, board: chess.Board) -> float:
        """Calculate position evaluation from general perspective by delegating to scoring_calculator."""
        if not isinstance(board, chess.Board) or not board.is_valid():
            if self.logger and self.monitoring_enabled:
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
        
        # Return the score from whites persective, inverted if blacks turn
        score = white_score - black_score if board.turn else -1 * (white_score - black_score)
        if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
            self.logger.debug(f"Evaluation: {score:.3f} | FEN: {board.fen()}")
        return score
    
    def evaluate_position_from_perspective(self, board: chess.Board, color: Optional[chess.Color] = chess.WHITE) -> float:
        """Calculate position evaluation from specified player's perspective by delegating to scoring_calculator."""
        current_turn = board.turn
        color_name = "White" if color == chess.WHITE else "Black"
        
        if not isinstance(color, chess.Color) or color != current_turn or not board.is_valid():
            if self.logger and self.monitoring_enabled:
                self.logger.error(f"[Error] Invalid input for evaluation from perspective. Player: {color_name}, Turn: {board.turn}, FEN: {board.fen() if hasattr(board, 'fen') else 'N/A'}")
            return 0.0

        score = self.calculate_score(board=board,color=color)

        if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
            self.logger.debug(f"{color_name}'s perspective: {score:.3f} | FEN: {board.fen()}")
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
            self.game_factor = 0.0
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
        elif material < 20 and (not board.has_castling_rights(chess.WHITE) or not board.has_castling_rights(chess.BLACK)):
            # Middlegame Phase
            phase = "middlegame"
            # Heuristic: if less than 20 pieces are on the board and at least one player has castled
            endgame_factor = 0.5
            if material < 18 and not board.has_castling_rights(chess.WHITE) and not board.has_castling_rights(chess.BLACK):
                # Heuristic: if less than 20 pieces are on the board and both sides have still not castled, unstable position, collapse of opening preparation
                endgame_factor = 0.75
        elif material <= 32 and (board.has_castling_rights(chess.WHITE) and board.has_castling_rights(chess.BLACK)):
            # Opening Phase
            phase = 'opening'
            if material < 24 and ((board.has_castling_rights(chess.WHITE) and not board.has_castling_rights(chess.BLACK)) or (not board.has_castling_rights(chess.WHITE) and board.has_castling_rights(chess.BLACK))):
                # Heuristic: multiple pieces exchanged, one side has castled, position is destabilizing
                endgame_factor = 0.5
            elif material <= 28 and (board.has_castling_rights(chess.WHITE) and board.has_castling_rights(chess.BLACK)):
                # Heuristic: piece exchanges but both sides remain un-castled
                endgame_factor = 0.35
            elif material < 32:
                # Heuristic: at least one exchange, game just beginning
                endgame_factor = 0.25
            elif material == 32:
                # Heuristic: all material remains on the board, fully stable/closed position
                endgame_factor = 0.1
            else:
                endgame_factor = 0.0
            
        self.game_phase = phase
        self.game_factor = endgame_factor
        self.scoring['game_phase'] = phase
        self.scoring['endgame_factor'] = endgame_factor
        if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
            self.logger.debug(f"[Game Phase] Current phase: {self.game_phase} | Endgame factor: {self.game_factor:.2f} | FEN: {board.fen()}")
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
        
        # Helper for consistent color display in logs
        color_name = "White" if color == chess.WHITE else "Black"
        current_player_name = self.engine_config.get('white_player','') if color == chess.WHITE else self.engine_config.get('black_player','')
        v7p3r_thinking = (color == chess.WHITE and self.engine_config.get('white_player','') == 'v7p3r') or (color == chess.BLACK and self.engine_config.get('black_player', '') == 'v7p3r')
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Starting score calculation for {current_player_name} engine as {color_name} | Ruleset: {self.ruleset_name} | FEN: {board.fen()}")
            if self.print_scoring:
                print(f"[Scoring Calc] Starting score calculation for {current_player_name} engine as {color_name} | Ruleset: {self.ruleset_name} | FEN: {board.fen()}")
        
        # CHECKMATE THREATS (BONUS)
        checkmate_threats_score = 1.0 * (self._checkmate_threats(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Checkmate threats score for {color_name}: {checkmate_threats_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Checkmate threats score for {color_name}: {checkmate_threats_score:.3f} (Ruleset: {self.ruleset_name})")
        score += checkmate_threats_score
        self.scoring['checkmate_threats'] = checkmate_threats_score

        # KING SAFETY
        king_safety_score = 1.0 * (self._king_safety(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] King safety score for {color_name}: {king_safety_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] King safety score for {color_name}: {king_safety_score:.3f} (Ruleset: {self.ruleset_name})")
        score += king_safety_score
        self.scoring['king_safety'] = king_safety_score

        # KING ATTACK
        king_attack_score = 1.0 * (self._king_attack(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] King attack score for {color_name}: {king_attack_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] King attack score for {color_name}: {king_attack_score:.3f} (Ruleset: {self.ruleset_name})")
        score += king_attack_score
        self.scoring['king_attack'] = king_attack_score

        # DRAW SCENARIOS
        draw_scenarios_score = 1.0 * (self._draw_scenarios(board) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Draw scenarios score for {color_name}: {draw_scenarios_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Draw scenarios score for {color_name}: {draw_scenarios_score:.3f} (Ruleset: {self.ruleset_name})")
        score += draw_scenarios_score
        self.scoring['draw_scenarios'] = draw_scenarios_score

        # MATERIAL SCORE AND PST
        score += 1.0 * self._material_score(board, color)
        pst_board_score = self.pst.evaluate_board_position(board, endgame_factor)
        if color == chess.BLACK:
            pst_board_score = -pst_board_score
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Piece-square table score for {color_name}: {pst_board_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Piece-square table score for {color_name}: {pst_board_score:.3f} (Ruleset: {self.ruleset_name})")
        score += 1.0 * pst_board_score
        self.scoring['material_score'] = 1.0 * self._material_score(board, color)

        # PIECE COORDINATION
        piece_coordination_score = 1.0 * (self._piece_coordination(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Piece coordination score for {color_name}: {piece_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Piece coordination score for {color_name}: {piece_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_coordination_score
        self.scoring['piece_coordination'] = piece_coordination_score
        
        # CENTER CONTROL
        center_control_score = 1.0 * (self._center_control(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Center control score for {color_name}: {center_control_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Center control score for {color_name}: {center_control_score:.3f} (Ruleset: {self.ruleset_name})")
        score += center_control_score
        self.scoring['center_control'] = center_control_score
        
        # PAWN STRUCTURE
        pawn_structure_score = 1.0 * (self._pawn_structure(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Pawn structure score for {color_name}: {pawn_structure_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Pawn structure score for {color_name}: {pawn_structure_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_structure_score
        self.scoring['pawn_structure'] = pawn_structure_score
        
        # PAWN WEAKNESSES
        pawn_weaknesses_score = 1.0 * (self._pawn_weaknesses(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Pawn weaknesses score for {color_name}: {pawn_weaknesses_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Pawn weaknesses score for {color_name}: {pawn_weaknesses_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_weaknesses_score
        self.scoring['pawn_weaknesses'] = pawn_weaknesses_score
        
        # PASSED PAWNS
        passed_pawns_score = 1.0 * (self._passed_pawns(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Passed pawns score for {color_name}: {passed_pawns_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Passed pawns score for {color_name}: {passed_pawns_score:.3f} (Ruleset: {self.ruleset_name})")
        score += passed_pawns_score
        self.scoring['passed_pawns'] = passed_pawns_score
        
        # PAWN COUNT
        pawn_count_score = 1.0 * (self._pawn_count(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Pawn count score for {color_name}: {pawn_count_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Pawn count score for {color_name}: {pawn_count_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_count_score
        self.scoring['pawn_count'] = pawn_count_score

        # PAWN PROMOTION
        pawn_promotion_score = 1.0 * (self._pawn_promotion(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Pawn promotion score for {color_name}: {pawn_promotion_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Pawn promotion score for {color_name}: {pawn_promotion_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_promotion_score
        self.scoring['pawn_promotion'] = pawn_promotion_score

        # BISHOP COUNT
        bishop_count_score = 1.0 * (self._bishop_count(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Bishop count score for {color_name}: {bishop_count_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Bishop count score for {color_name}: {bishop_count_score:.3f} (Ruleset: {self.ruleset_name})")
        score += bishop_count_score
        self.scoring['bishop_count'] = bishop_count_score

        # KNIGHT COUNT
        knight_count_score = 1.0 * (self._knight_count(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Knight count score for {color_name}: {knight_count_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Knight count score for {color_name}: {knight_count_score:.3f} (Ruleset: {self.ruleset_name})")
        score += knight_count_score
        self.scoring['knight_count'] = knight_count_score

        # BISHOP VISION
        bishop_vision_score = 1.0 * (self._bishop_vision(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Bishop vision score for {color_name}: {bishop_vision_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Bishop vision score for {color_name}: {bishop_vision_score:.3f} (Ruleset: {self.ruleset_name})")
        score += bishop_vision_score
        self.scoring['bishop_vision'] = bishop_vision_score

        # ROOK COORDINATION
        rook_coordination_score = 1.0 * (self._rook_coordination(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Rook coordination score for {color_name}: {rook_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Rook coordination score for {color_name}: {rook_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        score += rook_coordination_score
        self.scoring['rook_coordination'] = rook_coordination_score

        # CASTLING
        castling_score = 1.0 * (self._castling(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Castling score for {color_name}: {castling_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Castling score for {color_name}: {castling_score:.3f} (Ruleset: {self.ruleset_name})")
        score += castling_score
        self.scoring['castling'] = castling_score

        # CASTLING PROTECTION
        castling_protection_score = 1.0 * (self._castling_protection(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Castling protection score for {color_name}: {castling_protection_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Castling protection score for {color_name}: {castling_protection_score:.3f} (Ruleset: {self.ruleset_name})")
        score += castling_protection_score
        self.scoring['castling_protection'] = castling_protection_score

        # PIECE ACTIVITY
        piece_activity_score = 1.0 * (self._piece_activity(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Piece activity score for {color_name}: {piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Piece activity score for {color_name}: {piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_activity_score
        self.scoring['piece_activity'] = piece_activity_score
        
        # KNIGHT ACTIVITY
        knight_activity_score = 1.0 * (self._knight_activity(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Knight activity score for {color_name}: {knight_activity_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Knight activity score for {color_name}: {knight_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        score += knight_activity_score
        self.scoring['knight_activity'] = knight_activity_score

        # BISHOP ACTIVITY
        bishop_activity_score = 1.0 * (self._bishop_activity(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Bishop activity score for {color_name}: {bishop_activity_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Bishop activity score for {color_name}: {bishop_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        score += bishop_activity_score
        self.scoring['bishop_activity'] = bishop_activity_score

        # PIECE MOBILITY
        mobility_score = 1.0 * (self._board_coverage(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Mobility score for {color_name}: {mobility_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Mobility score for {color_name}: {mobility_score:.3f} (Ruleset: {self.ruleset_name})")
        score += mobility_score
        self.scoring['mobility'] = mobility_score
        
        # PIECE DEVELOPMENT
        piece_development_score = 1.0 * (self._piece_development(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Piece development score for {color_name}: {piece_development_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Piece development score for {color_name}: {piece_development_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_development_score
        self.scoring['piece_development'] = piece_development_score

        # PIECE ATTACKS
        piece_attacks_score = 1.0 * (self._piece_attacks(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Piece attacks score for {color_name}: {self._piece_attacks(board, color):.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Piece attacks score for {color_name}: {piece_attacks_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_attacks_score
        self.scoring['piece_attacks'] = piece_attacks_score

        # PIECE PROTECTION
        piece_protection_score = 1.0 * (self._piece_protection(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Piece protection score for {color_name}: {piece_protection_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Piece protection score for {color_name}: {piece_protection_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_protection_score
        self.scoring['piece_protection'] = piece_protection_score

        # QUEEN ATTACK
        queen_attack_score = 1.0 * (self._queen_attack(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Queen attack score for {color_name}: {queen_attack_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Queen attack score for {color_name}: {queen_attack_score:.3f} (Ruleset: {self.ruleset_name})")
        score += queen_attack_score
        self.scoring['queen_attack'] = queen_attack_score

        # PIECE CAPTURES
        piece_captures_score = 1.0 * (self._piece_captures(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Piece captures score for {color_name}: {piece_captures_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Piece captures score for {color_name}: {piece_captures_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_captures_score
        self.scoring['piece_captures'] = piece_captures_score

        # TEMPO
        tempo_score = 1.0 * (self._tempo(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Tempo score for {color_name}: {self._tempo(board, color):.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Tempo score for {color_name}: {tempo_score:.3f} (Ruleset: {self.ruleset_name})")
        score += tempo_score
        self.scoring['tempo'] = tempo_score

        # EN PASSANT
        en_passant_score = 1.0 * (self._en_passant(board, color) or 0.0) # Pass color
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] En passant score for {color_name}: {en_passant_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] En passant score for {color_name}: {en_passant_score:.3f} (Ruleset: {self.ruleset_name})")
        score += en_passant_score
        self.scoring['en_passant'] = en_passant_score

        # OPEN FILES
        open_files_score = 1.0 * (self._open_files(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Open files score for {color_name}: {open_files_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Open files score for {color_name}: {open_files_score:.3f} (Ruleset: {self.ruleset_name})")
        score += open_files_score
        self.scoring['open_files'] = open_files_score
        
        # STALEMATE
        stalemate_score = 1.0 * (self._stalemate(board) or 0.0)
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Stalemate score for {color_name}: {stalemate_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Stalemate score for {color_name}: {stalemate_score:.3f} (Ruleset: {self.ruleset_name})")
        score += stalemate_score
        self.scoring['stalemate'] = stalemate_score

        # GAME PHASE
        if self.use_game_phase:
            self.calculate_game_phase(board)
        else:
            self.game_phase = 'opening'
            self.game_factor = 0.0
        
        # FINAL SCORE
        if v7p3r_thinking:
            if self.logger and self.monitoring_enabled and self.verbose_output_enabled:
                self.logger.info(f"[Scoring Calc] Final score for {current_player_name} as {color_name}: {score:.3f} | Ruleset: {self.ruleset_name} | FEN: {board.fen()}")
            if self.print_scoring:
                print(f"[Scoring Calc] Final score for {current_player_name} as {color_name}: {score:.3f} | Ruleset: {self.ruleset_name} | FEN: {board.fen()}")
        return score

    # ==========================================
    # ========= RULE SCORING FUNCTIONS =========

    def _checkmate_threats(self, board: chess.Board, color: chess.Color) -> float:
        """
        Assess if 'color' can deliver a checkmate on their next move.
        Only consider legal moves for 'color' without mutating the original board's turn.
        """
        score = 0.0
        checkmate_threats_modifier = self.rules.get('checkmate_threats_modifier', 0.0)
        if checkmate_threats_modifier == 0.0:
            return score

        if board.is_checkmate() and board.turn == color:
            score += checkmate_threats_modifier
        elif board.is_checkmate() and board.turn != color:
            score -= checkmate_threats_modifier
        return score

    def _draw_scenarios(self, board: chess.Board) -> float:
        """ Assess if the game is in a draw scenario."""
        score = 0.0
        draw_modifier = self.rules.get('draw_modifier', 0.0)
        if draw_modifier == 0.0:
            return score

        if board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition() or board.is_repetition(count=2) or board.is_seventyfive_moves() or board.is_variant_draw():
            score -= draw_modifier
        # Do not apply the symetric draw modifier if the game is not a draw
        return score

    def _material_score(self, board: chess.Board, color: chess.Color) -> float:
        """Simple material count for given color"""
        score = 0.0
        material_weight = self.rules.get('material_weight', 1.0)
        if material_weight == 0.0:
            return score

        for piece_type, value in self.pst.piece_values.items():
            score += len(board.pieces(piece_type, color)) * value
        # Apply material weight from ruleset
        return score * material_weight

    def _piece_captures(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate the score based on modern MVV-LVA (Most Valuable Victim - Least Valuable Attacker) evaluation."""
        score = 0.0
        piece_captures_modifier = self.rules.get('piece_captures_modifier', self.fallback_modifier)
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

    def _knight_activity(self, board: chess.Board, color: chess.Color) -> float:
        """
        Mobility calculation with safe squares for Knights.
        """
        score = 0.0
        knight_activity_modifier = self.rules.get('knight_activity_modifier', self.fallback_modifier)
        if knight_activity_modifier == 0.0:
            return score

        for square in board.pieces(chess.KNIGHT, color):
            safe_moves = 0
            unsafe_moves = 0
            # Iterate through squares attacked by the knight
            for target in board.attacks(square):
                # Check if the target square is not attacked by enemy pawns
                if not self._is_attacked_by_pawn(board, target, not color):
                    safe_moves += 1
                else:
                    unsafe_moves += 1
            if safe_moves >= unsafe_moves:
                score += safe_moves * knight_activity_modifier
            else:
                score -= unsafe_moves * knight_activity_modifier

        return score

    def _bishop_activity(self, board: chess.Board, color: chess.Color) -> float:
        """
        Mobility calculation with safe squares for Bishops.
        """
        score = 0.0
        bishop_activity_modifier = self.rules.get('bishop_activity_modifier', self.fallback_modifier)
        if bishop_activity_modifier == 0.0:
            return score

        for square in board.pieces(chess.BISHOP, color):
            safe_moves = 0
            unsafe_moves = 0
            # Iterate through squares attacked by the bishop
            for target in board.attacks(square):
                # Check if the target square is not attacked by enemy pawns
                if not self._is_attacked_by_pawn(board, target, not color):
                    safe_moves += 1
                else:
                    unsafe_moves += 1
            if safe_moves >= unsafe_moves:
                score += safe_moves * bishop_activity_modifier
            else:
                score -= unsafe_moves * bishop_activity_modifier

        return score

    def _is_attacked_by_pawn(self, board: chess.Board, square: chess.Square, by_color: chess.Color) -> bool:
        """Helper function to check if a square is attacked by enemy pawns"""
        # Check if any of the attackers of 'square' from 'by_color' are pawns.
        for attacker_square in board.attackers(by_color, square):
            piece = board.piece_at(attacker_square)
            if piece and piece.piece_type == chess.PAWN:
                return True
        return False
    
    def _tempo(self, board: chess.Board, color: chess.Color) -> float:
        """If it's the player's turn and the game is still ongoing, give a small tempo bonus"""
        score = 0.0
        tempo_modifier = self.rules.get('tempo_modifier', self.fallback_modifier)
        if tempo_modifier == 0.0:
            return score

        if board.turn == color and not board.is_game_over() and board.is_valid():
            score += tempo_modifier
        else:
            score -= tempo_modifier
        return score

    def _center_control(self, board: chess.Board, color: chess.Color) -> float:
        """Simple center control"""
        score = 0.0
        center_control_modifier = self.rules.get('center_control_modifier', self.fallback_modifier)
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

    def _piece_activity(self, board: chess.Board, color: chess.Color) -> float:
        """Mobility and attack patterns"""
        score = 0.0
        piece_activity_modifier = self.rules.get('piece_activity_modifier', self.fallback_modifier)
        if piece_activity_modifier == 0.0:
            return score

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                score += piece_activity_modifier
            elif piece and piece.color != color:
                score -= piece_activity_modifier

        return score

    def _king_safety(self, board: chess.Board, color: chess.Color) -> float:
        score = 0.0
        king_safety_modifier = self.rules.get('king_safety_modifier', self.fallback_modifier)
        if king_safety_modifier == 0.0:
            return score

        king_square = board.king(color)
        if king_square is None:
            return score

        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)

        # Define squares for pawn shield relative to king's current rank
        # Consider pawns on the two ranks in front of the king for a shield
        shield_ranks = []
        if color == chess.WHITE:
            if king_rank < 7: shield_ranks.append(king_rank + 1)
            if king_rank < 6: shield_ranks.append(king_rank + 2) # For king on 1st rank
        else: # Black
            if king_rank > 0: shield_ranks.append(king_rank - 1)
            if king_rank > 1: shield_ranks.append(king_rank - 2) # For king on 8th rank

        for rank_offset in shield_ranks:
            for file_offset in [-1, 0, 1]: # Check adjacent files too
                target_file = king_file + file_offset
                if 0 <= target_file <= 7 and 0 <= rank_offset <= 7:
                    shield_square = chess.square(target_file, rank_offset)
                    piece = board.piece_at(shield_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        score += king_safety_modifier
                    elif not piece or (piece and piece.color != color):
                        score -= king_safety_modifier
        return score

    def _king_attack(self, board: chess.Board, color: chess.Color) -> float:
        """
        Evaluate if the opponent's king is under threat (in check) from 'color'.
        Adds a penalty/bonus if the specified 'color' is giving or receiving check.
        """
        score = 0.0
        check_modifier = self.rules.get('check_modifier', self.fallback_modifier)
        if check_modifier == 0.0:
            return score

        # Check if the board is in check.
        if board.is_check():
            if board.turn != color: # If it's *not* 'color's turn, and board is in check, 'color' is in check
                score -= check_modifier
            else: # If it *is* 'color's turn, and board is in check, then 'color' just gave check
                score += check_modifier
        return score

    def _piece_development(self, board: chess.Board, color: chess.Color) -> float:
        score = 0.0
        piece_development_modifier = self.rules.get('piece_development_modifier', self.fallback_modifier)
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
        board_coverage_modifier = self.rules.get('board_coverage_modifier', self.fallback_modifier)
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
    
    def _en_passant(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate en passant opportunities for the given color - STATIC EVALUATION"""
        score = 0.0
        en_passant_modifier = self.rules.get('en_passant_modifier', self.fallback_modifier)
        if en_passant_modifier == 0.0:
            return score
        
        # En passant opportunity for 'color' - check if any of our pawns can capture en passant
        if board.ep_square:
            ep_file = chess.square_file(board.ep_square)
            
            # Check if any of our pawns can capture the en passant square
            for file_offset in [-1, 1]:
                potential_pawn_file = ep_file + file_offset
                if 0 <= potential_pawn_file <= 7:
                    # For white, pawn would be on rank 4 (index 4), for black on rank 3 (index 3)
                    pawn_rank = 4 if color == chess.WHITE else 3
                    potential_pawn_square = chess.square(potential_pawn_file, pawn_rank)
                    piece = board.piece_at(potential_pawn_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        score += en_passant_modifier
                        break  # Found one en passant opportunity
        return score

    def _pawn_promotion(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate promotion opportunities for pawns of the given color"""
        # Promotion opportunities for 'color' - check pawns on 7th/2nd rank
        score = 0.0
        promotion_modifier = self.rules.get('pawn_promotion_modifier', self.fallback_modifier)
        if promotion_modifier == 0.0:
            return score

        promotion_rank = 6 if color == chess.WHITE else 1  # 7th rank for white, 2nd rank for black
        for pawn_square in board.pieces(chess.PAWN, color):
            if chess.square_rank(pawn_square) == promotion_rank:
                score += promotion_modifier
        
        # check the board for recent promotions
        last_move = board.peek()
        if last_move.promotion is not None:
            # If the last move was a promotion, give an increased bonus to prevent stalling
            if board.turn == color:
                score += promotion_modifier * 10 # ensure the bonus for actually promoting is always higher than the bonus for being able to promote
            else:
                score -= promotion_modifier
        return score
    
    def _piece_attacks(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate opponents hanging pieces for tactical opportunities"""
        score = 0.0
        piece_attacks_modifier = self.rules.get('piece_attacks_modifier', self.fallback_modifier)
        if piece_attacks_modifier == 0.0:
            return score
        
        # Determine opponent color
        opponent_color = not color

        # Check all squares for hanging pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == opponent_color:
                # Check if opponent piece is attacked by us and not defended by opponent
                if board.is_attacked_by(color, square) and not board.is_attacked_by(opponent_color, square):
                    score += piece_attacks_modifier
            elif piece and piece.color == color:
                # Check if our piece is attacked by opponent and not defended by us
                if board.is_attacked_by(opponent_color, square) and not board.is_attacked_by(color, square):
                    score -= piece_attacks_modifier
        return score

    def _piece_protection(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate whether pieces are defended"""
        score = 0.0
        piece_protection_modifier = self.rules.get('piece_protection_modifier', self.fallback_modifier)
        if piece_protection_modifier == 0.0:
            return score
        
        # Determine opponent color
        opponent_color = not color

        # Check all squares for hanging pieces and piece safety
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                # Penalty for our pieces being attacked by opponent and not defended by us
                if board.is_attacked_by(opponent_color, square) and not board.is_attacked_by(color, square):
                    score -= piece_protection_modifier
        
        return score
    
    def _castling(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate castling rights and opportunities"""
        score = 0.0
        castling_modifier = self.rules.get('castling_modifier', self.fallback_modifier)
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
        castling_protection_modifier = self.rules.get('castling_protection_modifier', self.fallback_modifier)
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
        piece_coordination_modifier = self.rules.get('piece_coordination_modifier', self.fallback_modifier)
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
    
    def _pawn_structure(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate pawn structure (doubled, isolated pawns)"""
        score = 0.0
        pawn_structure_modifier = self.rules.get('pawn_structure_modifier', self.fallback_modifier)
        if pawn_structure_modifier == 0.0:
            return score
        
        # Count doubled pawns
        for file in range(8):
            pawns_on_file = [s for s in board.pieces(chess.PAWN, color) if chess.square_file(s) == file]
            if len(pawns_on_file) > 1:
                score -= (len(pawns_on_file) - 1) * pawn_structure_modifier

        # Count isolated pawns
        for square in board.pieces(chess.PAWN, color):
            file = chess.square_file(square)
            is_isolated = True
            # Check left file
            if file > 0:
                for r in range(8):
                    p = board.piece_at(chess.square(file - 1, r))
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        is_isolated = False
                        score += pawn_structure_modifier
                        break
            # Check right file
            if is_isolated and file < 7:
                for r in range(8):
                    p = board.piece_at(chess.square(file + 1, r))
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        is_isolated = False
                        score += pawn_structure_modifier
                        break
            if is_isolated:
                score -= pawn_structure_modifier
        return score

    def _pawn_weaknesses(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate pawn weaknesses (e.g., backward pawns)."""
        score = 0.0
        pawn_weaknesses_modifier = self.rules.get('pawn_weaknesses_modifier', self.fallback_modifier)
        if pawn_weaknesses_modifier == 0.0:
            return score

        direction = 1 if color == chess.WHITE else -1
        for square in board.pieces(chess.PAWN, color):
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            is_backward = True

            # Check if the pawn is defended by another pawn
            for offset in [-1, 1]:  # Check adjacent files
                adjacent_file = file + offset
                if 0 <= adjacent_file <= 7:  # Ensure file is valid
                    defender_square = chess.square(adjacent_file, rank - direction)
                    defender_piece = board.piece_at(defender_square)
                    if defender_piece and defender_piece.piece_type == chess.PAWN and defender_piece.color == color:
                        is_backward = False
                        score += pawn_weaknesses_modifier  # Pawn is defended, not backward
                        break

            # Check if the file is open or semi-open
            if is_backward:
                score -= pawn_weaknesses_modifier
                has_opponent_pawn = any(
                    (piece := board.piece_at(chess.square(file, r))) and piece.piece_type == chess.PAWN and piece.color != color
                    for r in range(8)
                )
                if not has_opponent_pawn:  # File is open
                    score += pawn_weaknesses_modifier
                else:
                    score -= pawn_weaknesses_modifier

        return score

    def _pawn_count(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate pawn majority on the queenside or kingside"""
        score = 0.0
        pawn_count_modifier = self.rules.get('pawn_count_modifier', self.fallback_modifier)
        if pawn_count_modifier == 0.0:
            return score
        
        # Count pawns on each side of the board for both colors
        # Files a-d are queenside, e-h are kingside
        white_pawns_kingside = len([p for p in board.pieces(chess.PAWN, chess.WHITE) if chess.square_file(p) >= 4])
        white_pawns_queenside = len([p for p in board.pieces(chess.PAWN, chess.WHITE) if chess.square_file(p) < 4])
        black_pawns_kingside = len([p for p in board.pieces(chess.PAWN, chess.BLACK) if chess.square_file(p) >= 4])
        black_pawns_queenside = len([p for p in board.pieces(chess.PAWN, chess.BLACK) if chess.square_file(p) < 4])
        
        # Compare pawn counts on each wing
        if color == chess.WHITE:
            if white_pawns_kingside > black_pawns_kingside:
                score += pawn_count_modifier / 2 # Half bonus for kingside
            else:
                score -= pawn_count_modifier / 2
            if white_pawns_queenside > black_pawns_queenside:
                score += pawn_count_modifier / 2 # Half bonus for queenside
            else:
                score -= pawn_count_modifier / 2
        else: # Black
            if black_pawns_kingside > white_pawns_kingside:
                score += pawn_count_modifier / 2
            else:
                score -= pawn_count_modifier / 2
            if black_pawns_queenside > white_pawns_queenside:
                score += pawn_count_modifier / 2
            else:
                score -= pawn_count_modifier / 2
        
        return score

    def _passed_pawns(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate passed pawns for the given color."""
        score = 0.0
        passed_pawn_bonus = self.rules.get('passed_pawn_bonus', self.fallback_modifier)
        if passed_pawn_bonus == 0.0:
            return score

        opponent_color = not color
        for square in board.pieces(chess.PAWN, color):
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            is_passed = True
            # Check all pawns of the opponent
            for opp_square in board.pieces(chess.PAWN, opponent_color):
                opp_file = chess.square_file(opp_square)
                opp_rank = chess.square_rank(opp_square)
                if abs(opp_file - file) <= 1:
                    if (color == chess.WHITE and opp_rank > rank) or (color == chess.BLACK and opp_rank < rank):
                        is_passed = False
                        break
            if is_passed:
                score += passed_pawn_bonus
        return score

    def _knight_count(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate knight pair bonus"""
        score = 0.0
        knight_count_modifier = self.rules.get('knight_count_modifier', self.fallback_modifier)
        if knight_count_modifier == 0.0:
            return score
        
        # Count knights of the given color
        knights = list(board.pieces(chess.KNIGHT, color))
        if len(knights) >= 2:
            score += knight_count_modifier * 2

        # Check if opponent has more knights than us
        opponent_knights = list(board.pieces(chess.KNIGHT, not color))
        if len(knights) < len(opponent_knights):
            score -= knight_count_modifier

        return score

    def _bishop_count(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate bishop pair bonus"""
        score = 0.0
        bishop_count_modifier = self.rules.get('bishop_count_modifier', self.fallback_modifier)
        if bishop_count_modifier == 0.0:
            return score
        
        # Count bishops of the given color
        bishops = list(board.pieces(chess.BISHOP, color))
        if len(bishops) >= 2:
            score += bishop_count_modifier
        
        # Check if opponent has more bishops than us
        opponent_bishops = list(board.pieces(chess.KNIGHT, not color))
        if len(bishops) < len(opponent_bishops):
            score -= bishop_count_modifier

        return score

    def _bishop_vision(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate bishop vision bonus based on squares attacked."""
        score = 0.0
        bishop_vision_modifier = self.rules.get('bishop_vision_modifier', self.fallback_modifier)
        if bishop_vision_modifier == 0.0:
            return score
        
        for sq in board.pieces(chess.BISHOP, color):
            attacks = board.attacks(sq)
            # Bonus for having more attacked squares (i.e., good vision)
            if len(list(attacks)) > 5: # Bishops generally attack 7-13 squares, adjust threshold as needed
                score += bishop_vision_modifier
            else:
                score -= bishop_vision_modifier
        return score

    def _rook_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate bonus for rook pairs on same file/rank and 7th rank - FIXED DOUBLE COUNTING"""
        score = 0.0
        rook_coordination_modifier = self.rules.get('rook_coordination_modifier', self.fallback_modifier)
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

    def _open_files(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate open files for rooks and king safety."""
        score = 0.0
        open_files_modifier = self.rules.get('open_files_modifier', self.fallback_modifier)
        if open_files_modifier == 0.0:
            return score
        
        for file in range(8):
            is_file_open = True
            has_own_pawn_on_file = False
            has_opponent_pawn_on_file = False
            for rank in range(8):
                sq = chess.square(file, rank)
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.PAWN:
                    is_file_open = False # File is not open if it has any pawns
                    if piece.color == color:
                        has_own_pawn_on_file = True
                    else:
                        has_opponent_pawn_on_file = True
            
            # Bonus for controlling an open or semi-open file
            # An open file has no pawns. A semi-open file has only opponent pawns.
            if is_file_open: # Truly open file
                score += open_files_modifier # Full bonus for open file
            elif not is_file_open and not has_own_pawn_on_file and has_opponent_pawn_on_file: # Semi-open file for 'color'
                score += open_files_modifier / 2 # Half bonus for semi-open

            # Bonus if a rook is on an open or semi-open file
            if any(board.piece_at(chess.square(file, r)) == chess.Piece(chess.ROOK, color) for r in range(8)):
                if is_file_open or (not is_file_open and not has_own_pawn_on_file): # If open or semi-open
                    score += open_files_modifier / 2 # Half bonus for rook on open/semi-open file
                elif not is_file_open and has_own_pawn_on_file and has_opponent_pawn_on_file:
                    score -= open_files_modifier / 2 # Limited Rook vision behind pawns
        return score
    
    def _stalemate(self, board: chess.Board) -> float:
        """Check if the position is a stalemate"""
        stalemate_modifier = self.rules.get('stalemate_modifier', self.fallback_modifier)
        if stalemate_modifier == 0.0:
            return 0.0
        
        # If the board is in stalemate, return the modifier
        if board.is_stalemate():
            return stalemate_modifier
        return 0.0
    
    def _queen_attack(self, board: chess.Board, color: chess.Color) -> float:
        """
        Evaluate if the opponent's queen is under attack by a lesser piece - STATIC EVALUATION
        """
        score = 0.0
        queen_attack_modifier = self.rules.get('queen_attack_modifier', self.fallback_modifier)
        if queen_attack_modifier == 0.0:
            return score
        
        opponent_color = not color
        
        # Find opponent's queen(s)
        for queen_square in board.pieces(chess.QUEEN, opponent_color):
            # Check if our pieces are attacking the opponent's queen
            if board.is_attacked_by(color, queen_square):
                # Find what's attacking the queen
                for attacker_square in board.attackers(color, queen_square):
                    attacker_piece = board.piece_at(attacker_square)
                    if attacker_piece and attacker_piece.piece_type != chess.QUEEN:
                        # Lesser piece attacking queen - bonus for tactical opportunity
                        score += queen_attack_modifier
                        break  # Only count once per queen under attack
        for queen_square in board.pieces(chess.QUEEN, opponent_color):
            # Check if our queen is under attack by a lesser piece
            if board.is_attacked_by(opponent_color, queen_square):
                # Find what's attacking our queen
                for attacker_square in board.attackers(opponent_color, queen_square):
                    attacker_piece = board.piece_at(attacker_square)
                    if attacker_piece and attacker_piece.piece_type != chess.QUEEN:
                        # Lesser piece attacking our queen - penalty for tactical vulnerability
                        score -= queen_attack_modifier * 2  # Double penalty for our queen being attacked
                        break
        return score    

# Testing
if __name__ == "__main__":
    # Print the current default config values and run a test scoring calculation
    
    # Initialize required arguments
    engine_config = {
        'verbose_output': True,
        'engine_ruleset': 'default_evaluation',
        'use_game_phase': False,
        'piece_values': {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0
        },
    }
    from v7p3r_pst import v7p3rPST
    pst = v7p3rPST(logging.getLogger("v7p3r_pst_logger"))
    logger = logging.getLogger("v7p3r_engine_logger")
    
    scoring_calculator = v7p3rScore(engine_config=engine_config, pst=pst, logger=logger)
    print("Current default config values:")
    for key, value in scoring_calculator.rules.items():
        print(f"  {key}: {value}")
    
    # Run a test scoring calculation
    board = chess.Board()
    score = scoring_calculator.calculate_score(board=board, color=chess.WHITE)
    print(f"Test scoring calculation result: {score}")