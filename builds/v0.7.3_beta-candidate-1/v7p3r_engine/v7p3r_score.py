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
from typing import Optional
from v7p3r_config import v7p3rConfig

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
        '%(asctime)s | %(levelname)s | %(funcName)-15s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    v7p3r_score_logger.addHandler(file_handler)
    v7p3r_score_logger.propagate = False

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
        self.score_id = f"score[{self.score_counter}]_{timestamp}"
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
            'material': 0,
            'checkmate_threats': 0.0,
            'king_safety': 0.0,
            'king_attack': 0.0,
            'draw_scenarios': 0.0,
            'material_score': 0.0,
            'piece_square_table_score': 0.0,
            'piece_coordination': 0.0,
            'center_control': 0.0,
            'pawn_structure': 0.0,
            'pawn_weaknesses': 0.0,
            'passed_pawns': 0.0,
            'pawn_count': 0.0,
            'pawn_promotion': 0.0,
            'bishop_count': 0.0,
            'knight_count': 0.0
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
        
        # Return the score from whites persective, inverted if blacks turn
        score = white_score - black_score if board.turn else -1 * (white_score - black_score)
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
        self.score_dataset['evaluation'] = self.evaluate_position(board)
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
        
        # CHECKMATE THREATS
        checkmate_threats_score = 1.0 * (self.rules_manager._checkmate_threats(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Checkmate threats score for {self.color_name}: {checkmate_threats_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Checkmate threats score for {self.color_name}: {checkmate_threats_score:.3f} (Ruleset: {self.ruleset_name})")
        score += checkmate_threats_score
        self.score_dataset['checkmate_threats'] = checkmate_threats_score

        # KING SAFETY
        king_safety_score = 1.0 * (self.rules_manager._king_safety(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] King safety score for {self.color_name}: {king_safety_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] King safety score for {self.color_name}: {king_safety_score:.3f} (Ruleset: {self.ruleset_name})")
        score += king_safety_score
        self.score_dataset['king_safety'] = king_safety_score

        # KING ATTACK
        king_attack_score = 1.0 * (self.rules_manager._king_attack(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] King attack score for {self.color_name}: {king_attack_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] King attack score for {self.color_name}: {king_attack_score:.3f} (Ruleset: {self.ruleset_name})")
        score += king_attack_score
        self.score_dataset['king_attack'] = king_attack_score

        # DRAW SCENARIOS
        draw_scenarios_score = 1.0 * (self.rules_manager._draw_scenarios(board) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Draw scenarios score for {self.color_name}: {draw_scenarios_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Draw scenarios score for {self.color_name}: {draw_scenarios_score:.3f} (Ruleset: {self.ruleset_name})")
        score += draw_scenarios_score
        self.score_dataset['draw_scenarios'] = draw_scenarios_score

        # MATERIAL SCORE AND PST
        score += 1.0 * self.rules_manager._material_score(board, color)
        pst_board_score = self.pst.evaluate_board_position(board, endgame_factor)
        if color == chess.BLACK:
            pst_board_score = -pst_board_score
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Piece-square table score for {self.color_name}: {pst_board_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Piece-square table score for {self.color_name}: {pst_board_score:.3f} (Ruleset: {self.ruleset_name})")
        score += 1.0 * pst_board_score
        self.score_dataset['material_score'] = 1.0 * self.rules_manager._material_score(board, color)

        # PIECE COORDINATION
        piece_coordination_score = 1.0 * (self.rules_manager._piece_coordination(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Piece coordination score for {self.color_name}: {piece_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Piece coordination score for {self.color_name}: {piece_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_coordination_score
        self.score_dataset['piece_coordination'] = piece_coordination_score
        
        # CENTER CONTROL
        center_control_score = 1.0 * (self.rules_manager._center_control(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Center control score for {self.color_name}: {center_control_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Center control score for {self.color_name}: {center_control_score:.3f} (Ruleset: {self.ruleset_name})")
        score += center_control_score
        self.score_dataset['center_control'] = center_control_score
        
        # PAWN STRUCTURE
        pawn_structure_score = 1.0 * (self.rules_manager._pawn_structure(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Pawn structure score for {self.color_name}: {pawn_structure_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Pawn structure score for {self.color_name}: {pawn_structure_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_structure_score
        self.score_dataset['pawn_structure'] = pawn_structure_score
        
        # PAWN WEAKNESSES
        pawn_weaknesses_score = 1.0 * (self.rules_manager._pawn_weaknesses(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Pawn weaknesses score for {self.color_name}: {pawn_weaknesses_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Pawn weaknesses score for {self.color_name}: {pawn_weaknesses_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_weaknesses_score
        self.score_dataset['pawn_weaknesses'] = pawn_weaknesses_score
        
        # PASSED PAWNS
        passed_pawns_score = 1.0 * (self.rules_manager._passed_pawns(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Passed pawns score for {self.color_name}: {passed_pawns_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Passed pawns score for {self.color_name}: {passed_pawns_score:.3f} (Ruleset: {self.ruleset_name})")
        score += passed_pawns_score
        self.score_dataset['passed_pawns'] = passed_pawns_score
        
        # PAWN COUNT
        pawn_count_score = 1.0 * (self.rules_manager._pawn_count(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Pawn count score for {self.color_name}: {pawn_count_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Pawn count score for {self.color_name}: {pawn_count_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_count_score
        self.score_dataset['pawn_count'] = pawn_count_score

        # PAWN PROMOTION
        pawn_promotion_score = 1.0 * (self.rules_manager._pawn_promotion(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Pawn promotion score for {self.color_name}: {pawn_promotion_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Pawn promotion score for {self.color_name}: {pawn_promotion_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_promotion_score
        self.score_dataset['pawn_promotion'] = pawn_promotion_score

        # BISHOP COUNT
        bishop_count_score = 1.0 * (self.rules_manager._bishop_count(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Bishop count score for {self.color_name}: {bishop_count_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Bishop count score for {self.color_name}: {bishop_count_score:.3f} (Ruleset: {self.ruleset_name})")
        score += bishop_count_score
        self.score_dataset['bishop_count'] = bishop_count_score

        # KNIGHT COUNT
        knight_count_score = 1.0 * (self.rules_manager._knight_count(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Knight count score for {self.color_name}: {knight_count_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Knight count score for {self.color_name}: {knight_count_score:.3f} (Ruleset: {self.ruleset_name})")
        score += knight_count_score
        self.score_dataset['knight_count'] = knight_count_score

        # BISHOP VISION
        bishop_vision_score = 1.0 * (self.rules_manager._bishop_vision(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Bishop vision score for {self.color_name}: {bishop_vision_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Bishop vision score for {self.color_name}: {bishop_vision_score:.3f} (Ruleset: {self.ruleset_name})")
        score += bishop_vision_score
        self.score_dataset['bishop_vision'] = bishop_vision_score

        # ROOK COORDINATION
        rook_coordination_score = 1.0 * (self.rules_manager._rook_coordination(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Rook coordination score for {self.color_name}: {rook_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Rook coordination score for {self.color_name}: {rook_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        score += rook_coordination_score
        self.score_dataset['rook_coordination'] = rook_coordination_score

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

        # PIECE ACTIVITY
        piece_activity_score = 1.0 * (self.rules_manager._piece_activity(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Piece activity score for {self.color_name}: {piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Piece activity score for {self.color_name}: {piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_activity_score
        self.score_dataset['piece_activity'] = piece_activity_score
        
        # KNIGHT ACTIVITY
        knight_activity_score = 1.0 * (self.rules_manager._knight_activity(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Knight activity score for {self.color_name}: {knight_activity_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Knight activity score for {self.color_name}: {knight_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        score += knight_activity_score
        self.score_dataset['knight_activity'] = knight_activity_score

        # BISHOP ACTIVITY
        bishop_activity_score = 1.0 * (self.rules_manager._bishop_activity(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Bishop activity score for {self.color_name}: {bishop_activity_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Bishop activity score for {self.color_name}: {bishop_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        score += bishop_activity_score
        self.score_dataset['bishop_activity'] = bishop_activity_score

        # PIECE MOBILITY
        mobility_score = 1.0 * (self.rules_manager._board_coverage(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Mobility score for {self.color_name}: {mobility_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Mobility score for {self.color_name}: {mobility_score:.3f} (Ruleset: {self.ruleset_name})")
        score += mobility_score
        self.score_dataset['mobility'] = mobility_score
        
        # PIECE DEVELOPMENT
        piece_development_score = 1.0 * (self.rules_manager._piece_development(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Piece development score for {self.color_name}: {piece_development_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Piece development score for {self.color_name}: {piece_development_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_development_score
        self.score_dataset['piece_development'] = piece_development_score

        # PIECE ATTACKS
        piece_attacks_score = 1.0 * (self.rules_manager._piece_attacks(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Piece attacks score for {self.color_name}: {piece_attacks_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Piece attacks score for {self.color_name}: {piece_attacks_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_attacks_score
        self.score_dataset['piece_attacks'] = piece_attacks_score

        # PIECE PROTECTION
        piece_protection_score = 1.0 * (self.rules_manager._piece_protection(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Piece protection score for {self.color_name}: {piece_protection_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Piece protection score for {self.color_name}: {piece_protection_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_protection_score
        self.score_dataset['piece_protection'] = piece_protection_score

        # QUEEN ATTACK
        queen_attack_score = 1.0 * (self.rules_manager._queen_attack(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Queen attack score for {self.color_name}: {queen_attack_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Queen attack score for {self.color_name}: {queen_attack_score:.3f} (Ruleset: {self.ruleset_name})")
        score += queen_attack_score
        self.score_dataset['queen_attack'] = queen_attack_score

        # PIECE CAPTURES
        piece_captures_score = 1.0 * (self.rules_manager._piece_captures(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Piece captures score for {self.color_name}: {piece_captures_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Piece captures score for {self.color_name}: {piece_captures_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_captures_score
        self.score_dataset['piece_captures'] = piece_captures_score

        # TEMPO
        tempo_score = 1.0 * (self.rules_manager._tempo(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Tempo score for {self.color_name}: {tempo_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Tempo score for {self.color_name}: {tempo_score:.3f} (Ruleset: {self.ruleset_name})")
        score += tempo_score
        self.score_dataset['tempo'] = tempo_score

        # EN PASSANT
        en_passant_score = 1.0 * (self.rules_manager._en_passant(board, color) or 0.0) # Pass color
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] En passant score for {self.color_name}: {en_passant_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] En passant score for {self.color_name}: {en_passant_score:.3f} (Ruleset: {self.ruleset_name})")
        score += en_passant_score
        self.score_dataset['en_passant'] = en_passant_score

        # OPEN FILES
        open_files_score = 1.0 * (self.rules_manager._open_files(board, color) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Open files score for {self.color_name}: {open_files_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Open files score for {self.color_name}: {open_files_score:.3f} (Ruleset: {self.ruleset_name})")
        score += open_files_score
        self.score_dataset['open_files'] = open_files_score
        
        # STALEMATE
        stalemate_score = 1.0 * (self.rules_manager._stalemate(board) or 0.0)
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Stalemate score for {self.color_name}: {stalemate_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Stalemate score for {self.color_name}: {stalemate_score:.3f} (Ruleset: {self.ruleset_name})")
        score += stalemate_score
        self.score_dataset['stalemate'] = stalemate_score

        # GAME PHASE
        if self.use_game_phase:
            self.calculate_game_phase(board)
        else:
            self.game_phase = 'opening'
            self.endgame_factor = 0.0
        
        # FINAL SCORE
        self.score_dataset['score'] = score  # Update final score in scoring dictionary
        
        if self.v7p3r_thinking:
            if self.monitoring_enabled and self.logger:
                self.logger.info(f"[Scoring Calc] Final score for {self.current_player_name} as {self.color_name}: {score:.3f} | Ruleset: {self.ruleset_name} | FEN: {board.fen()}")
            if self.verbose_output_enabled:
                print(f"[Scoring Calc] Final score for {self.current_player_name} as {self.color_name}: {score:.3f} | Ruleset: {self.ruleset_name} | FEN: {board.fen()}")
        return score
