# v7p3r_scoring_calculation.py

""" v7p3r Scoring Calculation Module
This module is responsible for calculating the score of a chess position based on various factors,
including material balance, piece-square tables, king safety, and other positional features.
It is designed to be used by the v7p3r chess engine.
"""

import chess
import logging
import datetime
import sys
import os

from cycler import V
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from v7p3r_engine.v7p3r_pst import v7p3rPST # Need this for PST evaluation

# At module level, define a single logger for this file
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
v7p3r_scoring_logger = logging.getLogger("v7p3r_scoring_calculation")
v7p3r_scoring_logger.setLevel(logging.DEBUG)
if not v7p3r_scoring_logger.handlers:
    if not os.path.exists('logging'):
        os.makedirs('logging', exist_ok=True)
    from logging.handlers import RotatingFileHandler
    # Use a timestamped log file for each engine run
    timestamp = get_timestamp()
    log_file_path = f"logging/v7p3r_scoring_calculation_{timestamp}.log"
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
v7p3r_scoring_logger.addHandler(file_handler)
v7p3r_scoring_logger.propagate = False

class v7p3rScore:
    def __init__(self, engine_config: dict, v7p3r_config: dict):
        self.engine_config = engine_config # Only the engine configuration for this AI
        self.v7p3r_config = v7p3r_config # This is the full v7p3r_config.yaml content

        # Initialize logger first
        self.logger = v7p3r_scoring_logger

        self.print_scoring = self.engine_config.get('print_scoring', False)

        # Ruleset and scoring modifier are determined by the resolved engine_config
        self.ruleset_name = self.engine_config.get('ruleset', 'default_evaluation')
        
        # Load all rulesets from v7p3r_yaml_config into self.rulesets
        self.rules = self.v7p3r_config.get(self.ruleset_name, {}) # Get the specific ruleset based on name

        if v7p3r_scoring_logger:
            v7p3r_scoring_logger.debug(f"v7p3rScoringCalculation initialized with ruleset: {self.ruleset_name}")
            v7p3r_scoring_logger.debug(f"Current ruleset parameters: {self.rules}")

        # Set up additional scoring tools
        self.pst = v7p3rPST() # Load piece-square tables from config

    # Renamed from _calculate_score to calculate_score to be the public API
    def calculate_score(self, board: chess.Board, color: chess.Color, endgame_factor: float = 0.0) -> float:
        """
        Calculates the position evaluation score for a given board and color,
        applying dynamic ruleset settings and endgame awareness.
        This is the main public method for this class.
        """
        score = 0.0
        
        # Helper for consistent color display in logs
        color_name = "White" if color == chess.WHITE else "Black"

       # Set up config values for scoring
        self.search_algorithm = self.engine_config.get('search_algorithm', 'random')
        self.depth = self.engine_config.get('depth', 3)
        self.max_depth = self.engine_config.get('max_depth', 4)
        self.solutions_enabled = self.engine_config.get('use_solutions', False)
        self.pst_enabled = self.engine_config.get('pst', False)
        self.pst_weight = self.engine_config.get('weight', 1.0)
        self.move_ordering_enabled = self.engine_config.get('move_ordering', False)
        self.quiescence_enabled = self.engine_config.get('quiescence', False)
        self.move_time_limit = self.engine_config.get('time_limit', 0)
        self.scoring_modifier = self.engine_config.get('scoring_modifier', 1.0)
        self.game_phase_awareness = self.engine_config.get('game_phase_awareness', False)
        self.engine_color = 'white' if board.turn else 'black'
        
        # Critical scoring components
        checkmate_threats_score = self.scoring_modifier * (self._checkmate_threats(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Checkmate threats score for {color_name}: {checkmate_threats_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Checkmate threats score for {color_name}: {checkmate_threats_score:.3f} (Ruleset: {self.ruleset_name})")
        score += checkmate_threats_score

        king_safety_score = self.scoring_modifier * (self._king_safety(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"King safety score for {color_name}: {king_safety_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"King safety score for {color_name}: {king_safety_score:.3f} (Ruleset: {self.ruleset_name})")
        score += king_safety_score
        
        king_threat_score = self.scoring_modifier * (self._king_threat(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"King threat score for {color_name}: {king_threat_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"King threat score for {color_name}: {king_threat_score:.3f} (Ruleset: {self.ruleset_name})")
        score += king_threat_score
        
        king_endangerment_score = self.scoring_modifier * (self._king_endangerment(board) or 0.0)
        if self.logger:
            self.logger.debug(f"King endangerment score for {color_name}: {king_endangerment_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"King endangerment score for {color_name}: {king_endangerment_score:.3f} (Ruleset: {self.ruleset_name})")
        score += king_endangerment_score

        draw_scenarios_score = self.scoring_modifier * (self._draw_scenarios(board) or 0.0)
        if self.logger:
            self.logger.debug(f"Draw scenarios score for {color_name}: {draw_scenarios_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Draw scenarios score for {color_name}: {draw_scenarios_score:.3f} (Ruleset: {self.ruleset_name})")
        score += draw_scenarios_score
        
        # Material and piece-square table evaluation
        score += self.scoring_modifier * self._material_score(board, color)
        if self.pst_enabled:
            pst_board_score = self.pst.evaluate_board_position(board, endgame_factor)
            if color == chess.BLACK:
                pst_board_score = -pst_board_score
            if self.logger:
                self.logger.debug(f"Piece-square table score for {color_name}: {pst_board_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"Piece-square table score for {color_name}: {pst_board_score:.3f} (Ruleset: {self.ruleset_name})")
            score += self.scoring_modifier * self.pst_weight * pst_board_score

        # Piece coordination and control
        piece_coordination_score = self.scoring_modifier * (self._piece_coordination(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Piece coordination score for {color_name}: {piece_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Piece coordination score for {color_name}: {piece_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_coordination_score
        
        center_control_score = self.scoring_modifier * (self._center_control(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Center control score for {color_name}: {center_control_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Center control score for {color_name}: {center_control_score:.3f} (Ruleset: {self.ruleset_name})")
        score += center_control_score
        
        pawn_structure_score = self.scoring_modifier * (self._pawn_structure(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Pawn structure score for {color_name}: {pawn_structure_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Pawn structure score for {color_name}: {pawn_structure_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_structure_score
        
        pawn_weaknesses_score = self.scoring_modifier * (self._pawn_weaknesses(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Pawn weaknesses score for {color_name}: {pawn_weaknesses_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Pawn weaknesses score for {color_name}: {pawn_weaknesses_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_weaknesses_score
        
        passed_pawns_score = self.scoring_modifier * (self._passed_pawns(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Passed pawns score for {color_name}: {passed_pawns_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Passed pawns score for {color_name}: {passed_pawns_score:.3f} (Ruleset: {self.ruleset_name})")
        score += passed_pawns_score
        
        pawn_majority_score = self.scoring_modifier * (self._pawn_majority(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Pawn majority score for {color_name}: {pawn_majority_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Pawn majority score for {color_name}: {pawn_majority_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_majority_score
        
        bishop_pair_score = self.scoring_modifier * (self._bishop_pair(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Bishop pair score for {color_name}: {bishop_pair_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Bishop pair score for {color_name}: {bishop_pair_score:.3f} (Ruleset: {self.ruleset_name})")
        score += bishop_pair_score
        
        knight_pair_score = self.scoring_modifier * (self._knight_pair(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Knight pair score for {color_name}: {knight_pair_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Knight pair score for {color_name}: {knight_pair_score:.3f} (Ruleset: {self.ruleset_name})")
        score += knight_pair_score
        
        bishop_vision_score = self.scoring_modifier * (self._bishop_vision(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Bishop vision score for {color_name}: {bishop_vision_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Bishop vision score for {color_name}: {bishop_vision_score:.3f} (Ruleset: {self.ruleset_name})")
        score += bishop_vision_score
        
        rook_coordination_score = self.scoring_modifier * (self._rook_coordination(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Rook coordination score for {color_name}: {rook_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Rook coordination score for {color_name}: {rook_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        score += rook_coordination_score
        
        castling_evaluation_score = self.scoring_modifier * (self._castling_evaluation(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Castling evaluation score for {color_name}: {castling_evaluation_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Castling evaluation score for {color_name}: {castling_evaluation_score:.3f} (Ruleset: {self.ruleset_name})")
        score += castling_evaluation_score
        
        # Piece development and mobility
        piece_activity_score = self.scoring_modifier * (self._piece_activity(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Piece activity score for {color_name}: {piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Piece activity score for {color_name}: {piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_activity_score
        
        improved_minor_piece_activity_score = self.scoring_modifier * (self._improved_minor_piece_activity(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Improved minor piece activity score for {color_name}: {improved_minor_piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Improved minor piece activity score for {color_name}: {improved_minor_piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        score += improved_minor_piece_activity_score
        
        mobility_score = self.scoring_modifier * (self._mobility_score(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Mobility score for {color_name}: {mobility_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Mobility score for {color_name}: {mobility_score:.3f} (Ruleset: {self.ruleset_name})")
        score += mobility_score
        
        undeveloped_pieces_score = self.scoring_modifier * (self._undeveloped_pieces(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Undeveloped pieces score for {color_name}: {undeveloped_pieces_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Undeveloped pieces score for {color_name}: {undeveloped_pieces_score:.3f} (Ruleset: {self.ruleset_name})")
        score += undeveloped_pieces_score
        
        # Tactical and strategic considerations
        tactical_evaluation_score = self.scoring_modifier * (self._tactical_evaluation(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Tactical evaluation score for {color_name}: {self._tactical_evaluation(board, color):.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Tactical evaluation score for {color_name}: {tactical_evaluation_score:.3f} (Ruleset: {self.ruleset_name})")
        score += tactical_evaluation_score
        
        queen_capture_score = self.scoring_modifier * (self._queen_capture(board) or 0.0)
        if self.logger:
            self.logger.debug(f"Queen capture score for {color_name}: {queen_capture_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Queen capture score for {color_name}: {queen_capture_score:.3f} (Ruleset: {self.ruleset_name})")
        score += queen_capture_score

        tempo_bonus_score = self.scoring_modifier * (self._tempo_bonus(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Tempo bonus score for {color_name}: {self._tempo_bonus(board, color):.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Tempo bonus score for {color_name}: {tempo_bonus_score:.3f} (Ruleset: {self.ruleset_name})")
        score += tempo_bonus_score
        
        special_moves_score = self.scoring_modifier * (self._special_moves(board, color) or 0.0) # Pass color
        if self.logger:
            self.logger.debug(f"Special moves score for {color_name}: {special_moves_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Special moves score for {color_name}: {special_moves_score:.3f} (Ruleset: {self.ruleset_name})")
        score += special_moves_score
        
        open_files_score = self.scoring_modifier * (self._open_files(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Open files score for {color_name}: {open_files_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Open files score for {color_name}: {open_files_score:.3f} (Ruleset: {self.ruleset_name})")
        score += open_files_score
        
        stalemate_score = self.scoring_modifier * (self._stalemate(board) or 0.0)
        if self.logger:
            self.logger.debug(f"Stalemate score for {color_name}: {stalemate_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Stalemate score for {color_name}: {stalemate_score:.3f} (Ruleset: {self.ruleset_name})")
        score += stalemate_score

        self.calculate_game_phase(board)

        if self.logger:
            self.logger.debug(f"Final score for {color_name}: {score:.3f} (Ruleset: {self.ruleset_name}, Modifier: {self.scoring_modifier}) | FEN: {board.fen()}")

        return score

    def calculate_game_phase(self, board: chess.Board) -> str:
        """
        Determines the current phase of the game: 'opening', 'middlegame', or 'endgame'.
        Uses material and castling rights as heuristics.
        Sets self.game_phase for use in other scoring functions.
        """
        # Count total material (excluding kings)
        material = sum([
            len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK))
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        ])
        # Heuristic: opening if all queens/rooks/bishops/knights are present, endgame if queens are gone or little material
        if material >= 28:
            phase = 'opening'
        elif material <= 12 or (not board.pieces(chess.QUEEN, chess.WHITE) and not board.pieces(chess.QUEEN, chess.BLACK)):
            phase = 'endgame'
        else:
            phase = 'middlegame'
        self.game_phase = phase
        return phase

    # ==========================================
    # ========= RULE SCORING FUNCTIONS =========
    # These functions are now methods of ViperScoringCalculation
    # and access their rule values via self.rules.get()

    def _checkmate_threats(self, board: chess.Board, color: chess.Color) -> float:
        """
        Assess if 'color' can deliver a checkmate on their next move.
        Only consider legal moves for 'color' without mutating the original board's turn.
        """
        score = 0.0
        # Only check threats for the current player's turn to avoid double counting
        # or checking irrelevant checks for the given 'color' if not their turn.
        # This function should assess if 'color' can deliver a checkmate on their *next* move.
        original_turn = board.turn
        # Only consider pseudo-legal moves for the correct color
        try:
            # Use a copy so we don't mutate the original board
            board_copy = board.copy()
            board_copy.turn = color
            for move in board_copy.pseudo_legal_moves:
                # Only consider moves that are legal (pseudo-legal may include illegal under check)
                if not board_copy.is_legal(move):
                    continue
                board_copy.push(move)
                if board_copy.is_checkmate():
                    score += self.rules.get('checkmate_bonus', 0)
                    board_copy.pop()
                    # If a checkmate is found, we can break and return the bonus.
                    # However, a common heuristic might be to give the bonus to the side *delivering* mate.
                    # This function implies it's for `color` to *threaten* mate to opponent.
                    # So the `board.turn` in `board.is_checkmate()` should be the *opponent's* turn after `move` is pushed.
                    if board_copy.turn != color: # If the checkmate is on the *opponent's* king
                        return score # Return immediately
                else:
                    board_copy.pop()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in _checkmate_threats: {e} | FEN: {board.fen()}")
        return score

    def _draw_scenarios(self, board: chess.Board) -> float:
        score = 0.0
        if board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition() or board.is_repetition(count=2):
            score += self.rules.get('draw_penalty', -9999999999.0)
        return score

    def _material_score(self, board: chess.Board, color: chess.Color) -> float:
        """Simple material count for given color"""
        score = 0.0
        for piece_type, value in self.pst.piece_values.items():
            score += len(board.pieces(piece_type, color)) * value
        # Apply material weight from ruleset
        return score * self.rules.get('material_weight', 1.0)
    
    # This method is correctly called directly from self.pst.evaluate_board_position in calculate_score

    def _improved_minor_piece_activity(self, board: chess.Board, color: chess.Color) -> float:
        """
        Mobility calculation with safe squares for Knights and Bishops.
        """
        score = 0.0

        for square in board.pieces(chess.KNIGHT, color):
            safe_moves = 0
            # Iterate through squares attacked by the knight
            for target in board.attacks(square):
                # Check if the target square is not attacked by enemy pawns
                if not self._is_attacked_by_pawn(board, target, not color):
                    safe_moves += 1
            score += safe_moves * self.rules.get('knight_activity_bonus', 0.0)

        for square in board.pieces(chess.BISHOP, color):
            safe_moves = 0
            for target in board.attacks(square):
                if not self._is_attacked_by_pawn(board, target, not color):
                    safe_moves += 1
            score += safe_moves * self.rules.get('bishop_activity_bonus', 0.0)

        return score

    def _tempo_bonus(self, board: chess.Board, color: chess.Color) -> float:
        """If it's the player's turn and the game is still ongoing, give a small tempo bonus"""
        # The 'current_player' attribute is from EvaluationEngine, need to pass it or infer.
        # This method is part of scoring specific 'color'. So, if it's 'color's turn.
        if board.turn == color and not board.is_game_over() and board.is_valid():
            return self.rules.get('tempo_bonus', 0.0)
        return 0.0

    def _is_attacked_by_pawn(self, board: chess.Board, square: chess.Square, by_color: chess.Color) -> bool:
        """Helper function to check if a square is attacked by enemy pawns"""
        # Check if any of the attackers of 'square' from 'by_color' are pawns.
        for attacker_square in board.attackers(by_color, square):
            piece = board.piece_at(attacker_square)
            if piece and piece.piece_type == chess.PAWN:
                return True
        return False

    def _center_control(self, board: chess.Board, color: chess.Color) -> float:
        """Simple center control"""
        score = 0.0
        center = [chess.D4, chess.D5, chess.E4, chess.E5]
        for square in center:
            # Check if current player controls (has a piece on) center square
            piece = board.piece_at(square)
            if piece and piece.color == color:
                score += self.rules.get('center_control_bonus', 0.0)
        return score

    def _piece_activity(self, board: chess.Board, color: chess.Color) -> float:
        """Mobility and attack patterns"""
        score = 0.0

        for square in board.pieces(chess.KNIGHT, color):
            score += len(list(board.attacks(square))) * self.rules.get('knight_activity_bonus', 0.0)

        for square in board.pieces(chess.BISHOP, color):
            score += len(list(board.attacks(square))) * self.rules.get('bishop_activity_bonus', 0.0)

        return score

    def _king_safety(self, board: chess.Board, color: chess.Color) -> float:
        score = 0.0
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
                        score += self.rules.get('king_safety_bonus', 0.0)
        
        return score

    def _king_threat(self, board: chess.Board, color: chess.Color) -> float:
        """
        Evaluate if the opponent's king is under threat (in check) from 'color'.
        Adds a penalty/bonus if the specified 'color' is giving or receiving check.
        """
        score = 0.0
        # Check if the board is in check.
        if board.is_check():
            # If it's 'color's turn AND 'color' just caused check (i.e., opponent is in check)
            # This is hard to tell from `board.turn` directly without knowing the previous move.
            # Simpler: if `color` is the one whose turn it is AND the board IS in check, it means
            # the *opponent* of `color` is in check (from previous move).
            # If `color` is the one whose turn it is NOT AND the board IS in check, it means
            # `color` itself is in check.
            
            # This method calculates score from the perspective of 'color'
            if board.turn != color: # If it's *not* 'color's turn, and board is in check, 'color' is in check
                score += self.rules.get('in_check_penalty', 0.0)
            else: # If it *is* 'color's turn, and board is in check, then 'color' just gave check
                score += self.rules.get('check_bonus', 0.0)
        return score

    def _undeveloped_pieces(self, board: chess.Board, color: chess.Color) -> float:
        score = 0.0
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
            score += undeveloped_count * self.rules.get('undeveloped_penalty', 0.0)

        return score

    def _mobility_score(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate mobility of pieces"""
        score = 0.0
        
        # Iterate over all pieces of the given color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color and piece.piece_type != chess.KING: # Exclude king from general mobility
                score += len(list(board.attacks(square))) * self.rules.get('piece_mobility_bonus', 0.0)

        return score
    
    def _special_moves(self, board: chess.Board, color: chess.Color) -> float: # Added color parameter
        """Evaluate special moves and opportunities for the given color"""
        score = 0.0
        
        # En passant opportunity for 'color'
        if board.ep_square:
            for move in board.legal_moves:
                # Check if the move is an en passant capture by 'color'
                if move.to_square == board.ep_square:
                    piece_on_from_square = board.piece_at(move.from_square)
                    # Ensure piece_on_from_square is not None before accessing attributes
                    if piece_on_from_square and piece_on_from_square.piece_type == chess.PAWN and piece_on_from_square.color == color:
                        if board.is_en_passant(move):
                            score += self.rules.get('en_passant_bonus', 0.0)
                            break # Found one en passant, bonus applied
        
        # Promotion opportunities for 'color'
        # Iterate legal moves for the current player on the board. 
        # If board.turn is not 'color', these are not 'color's opportunities right now.
        # This function should evaluate based on 'color's potential.
        # A simple way: check pawns of 'color' that are one step from promotion.
        promotion_rank = 7 if color == chess.WHITE else 0
        for pawn_square in board.pieces(chess.PAWN, color):
            if chess.square_rank(pawn_square) == (promotion_rank -1 if color == chess.WHITE else promotion_rank + 1):
                # Check if pawn can advance to promotion rank
                # This is a simplified check; a full check involves move generation.
                # For now, just having a pawn on the 7th/2nd is a strong indicator.
                score += self.rules.get('pawn_promotion_bonus', 0.0) 
                # A more accurate way would be to check board.generate_legal_moves() for promotions for 'color'
                # but that might be too slow for an eval term. The current approach is a heuristic.
        return score

    def _tactical_evaluation(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate tactical elements related to captures and hanging pieces."""
        score = 0.0
        
        for move in board.legal_moves:
            if board.is_capture(move):
                piece_making_capture = board.piece_at(move.from_square)
                # Ensure piece_making_capture is not None
                if piece_making_capture and piece_making_capture.color == color:
                    score += self.rules.get('capture_bonus', 0.0)
        
        opponent_color = not color

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == opponent_color:
                # Check if it's attacked by 'color' and not defended by 'opponent_color' (i.e., not attacked by opponent_color)
                if board.is_attacked_by(color, square) and not board.is_attacked_by(opponent_color, square):
                    score += self.rules.get('hanging_piece_bonus', 0.0)
            elif piece and piece.color == color:
                # Penalty for 'color' having pieces attacked by opponent_color and not defended by 'color'
                if board.is_attacked_by(opponent_color, square) and not board.is_attacked_by(color, square):
                    score += self.rules.get('undefended_piece_penalty', 0.0)
        
        return score

    def _castling_evaluation(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate castling rights and opportunities"""
        score = 0.0

        # Check if castled - more robust check considering king's final position
        king_sq = board.king(color)
        if king_sq: # Ensure king exists
            if color == chess.WHITE:
                if king_sq == chess.G1: # Kingside castled
                    score += self.rules.get('castling_bonus', 0.0)
                elif king_sq == chess.C1: # Queenside castled
                    score += self.rules.get('castling_bonus', 0.0)
            else: # Black
                if king_sq == chess.G8: # Kingside castled
                    score += self.rules.get('castling_bonus', 0.0)
                elif king_sq == chess.C8: # Queenside castled
                    score += self.rules.get('castling_bonus', 0.0)

        # Penalty if castling rights lost and not yet castled
        initial_king_square = chess.E1 if color == chess.WHITE else chess.E8
        if not board.has_castling_rights(color) and king_sq == initial_king_square:
            score += self.rules.get('castling_protection_penalty', 0.0)
        
        # Bonus if still has kingside or queenside castling rights
        if board.has_kingside_castling_rights(color) and board.has_queenside_castling_rights(color):
            score += self.rules.get('castling_protection_bonus', 0.0)
        elif board.has_kingside_castling_rights(color) or board.has_queenside_castling_rights(color):
            score += self.rules.get('castling_protection_bonus', 0.0) / 2
        
        return score

    def _piece_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate piece defense coordination for all pieces of the given color."""
        score = 0.0
        # For each piece of the given color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                # If the piece is defended by another friendly piece (i.e., the square it's on is attacked by its own color)
                if board.is_attacked_by(color, square): 
                    score += self.rules.get('piece_coordination_bonus', 0.0) 
        return score
    
    def _pawn_structure(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate pawn structure (doubled, isolated pawns)"""
        score = 0.0
        
        # Count doubled pawns
        for file in range(8):
            pawns_on_file = [s for s in board.pieces(chess.PAWN, color) if chess.square_file(s) == file]
            if len(pawns_on_file) > 1:
                score += (len(pawns_on_file) - 1) * self.rules.get('doubled_pawn_penalty', 0.0)
        
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
                        break

            # Check right file
            if is_isolated and file < 7:
                for r in range(8):
                    p = board.piece_at(chess.square(file + 1, r))
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        is_isolated = False
                        break

            if is_isolated:
                score += self.rules.get('isolated_pawn_penalty', 0.0)
        
        # No general pawn_structure_bonus here, as it's typically derived from good structure
        # (absence of penalties, presence of passed pawns, etc.)
        # If score is positive from penalties, it implies bad structure, so no bonus.

        return score

    def _pawn_weaknesses(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate pawn weaknesses (e.g., backward pawns)."""
        score = 0.0

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
                        break

            # Check if the file is open or semi-open
            if is_backward:
                has_opponent_pawn = any(
                    (piece := board.piece_at(chess.square(file, r))) and piece.piece_type == chess.PAWN and piece.color != color
                    for r in range(8)
                )
                if not has_opponent_pawn:  # File is open
                    score += self.rules.get('backward_pawn_penalty', 0.0)

        return score

    def _pawn_majority(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate pawn majority on the queenside or kingside"""
        score = 0.0
        
        # Count pawns on each side of the board for both colors
        # Files a-d are queenside, e-h are kingside
        white_pawns_kingside = len([p for p in board.pieces(chess.PAWN, chess.WHITE) if chess.square_file(p) >= 4])
        white_pawns_queenside = len([p for p in board.pieces(chess.PAWN, chess.WHITE) if chess.square_file(p) < 4])
        black_pawns_kingside = len([p for p in board.pieces(chess.PAWN, chess.BLACK) if chess.square_file(p) >= 4])
        black_pawns_queenside = len([p for p in board.pieces(chess.PAWN, chess.BLACK) if chess.square_file(p) < 4])
        
        # Compare pawn counts on each wing
        if color == chess.WHITE:
            if white_pawns_kingside > black_pawns_kingside:
                score += self.rules.get('pawn_majority_bonus', 0.0) / 2 # Half bonus for kingside
            if white_pawns_queenside > black_pawns_queenside:
                score += self.rules.get('pawn_majority_bonus', 0.0) / 2 # Half bonus for queenside
            # Optionally add penalty for minority
            # if white_pawns_kingside < black_pawns_kingside:
            #     score += self.rules.get('pawn_minority_penalty', 0.0) / 2
            # if white_pawns_queenside < black_pawns_queenside:
            #     score += self.rules.get('pawn_minority_penalty', 0.0) / 2
        else: # Black
            if black_pawns_kingside > white_pawns_kingside:
                score += self.rules.get('pawn_majority_bonus', 0.0) / 2
            if black_pawns_queenside > white_pawns_queenside:
                score += self.rules.get('pawn_majority_bonus', 0.0) / 2
            # Optionally add penalty for minority
            # if black_pawns_kingside < white_pawns_kingside:
            #     score += self.rules.get('pawn_minority_penalty', 0.0) / 2
            # if black_pawns_queenside < white_pawns_queenside:
            #     score += self.rules.get('pawn_minority_penalty', 0.0) / 2
        
        return score

    def _passed_pawns(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate passed pawns for the given color."""
        score = 0.0
        opponent_color = not color
        for square in board.pieces(chess.PAWN, color):
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            is_passed = True
            # Check all pawns of the opponent
            for opp_square in board.pieces(chess.PAWN, opponent_color):
                opp_file = chess.square_file(opp_square)
                opp_rank = chess.square_rank(opp_square)
                # For white, passed if no black pawn is on same/adjacent file and ahead
                # For black, passed if no white pawn is on same/adjacent file and ahead
                if abs(opp_file - file) <= 1:
                    if (color == chess.WHITE and opp_rank > rank) or (color == chess.BLACK and opp_rank < rank):
                        is_passed = False
                        break
            if is_passed:
                score += self.rules.get('passed_pawn_bonus', 0.0)
        return score

    def _knight_pair(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate knight pair bonus"""
        score = 0.0
        knights = list(board.pieces(chess.KNIGHT, color))
        if len(knights) >= 2:
            score += self.rules.get('knight_pair_bonus', 0.0) # Bonus for having *a* knight pair
            # If the bonus is per knight in a pair, it would be len(knights) * bonus / 2 (or similar)
        return score

    def _bishop_pair(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate bishop pair bonus"""
        score = 0.0
        bishops = list(board.pieces(chess.BISHOP, color))
        if len(bishops) >= 2:
            score += self.rules.get('bishop_pair_bonus', 0.0)
        return score

    def _bishop_vision(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate bishop vision bonus based on squares attacked."""
        score = 0.0
        for sq in board.pieces(chess.BISHOP, color):
            attacks = board.attacks(sq)
            # Bonus for having more attacked squares (i.e., good vision)
            if len(list(attacks)) > 5: # Bishops generally attack 7-13 squares, adjust threshold as needed
                score += self.rules.get('bishop_vision_bonus', 0.0)
        return score

    def _rook_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate bonus for rook pairs on same file/rank and 7th rank."""
        score = 0.0
        rooks = list(board.pieces(chess.ROOK, color))

        for i in range(len(rooks)):
            for j in range(i+1, len(rooks)):
                sq1, sq2 = rooks[i], rooks[j]
                if chess.square_file(sq1) == chess.square_file(sq2):
                    score += self.rules.get('stacked_rooks_bonus', 0.0)
                if chess.square_rank(sq1) == chess.square_rank(sq2):
                    score += self.rules.get('coordinated_rooks_bonus', 0.0)
                
                # Rook on 7th rank bonus (critical for attacking pawns)
                # Check for white on rank 7 (index 6) or black on rank 2 (index 1)
                if (color == chess.WHITE and (chess.square_rank(sq1) == 6 or chess.square_rank(sq2) == 6)) or \
                   (color == chess.BLACK and (chess.square_rank(sq1) == 1 or chess.square_rank(sq2) == 1)):
                    score += self.rules.get('rook_position_bonus', 0.0)
        return score

    def _open_files(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate open files for rooks and king safety."""
        score = 0.0
        
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
                score += self.rules.get('open_file_bonus', 0.0)
            elif not is_file_open and not has_own_pawn_on_file and has_opponent_pawn_on_file: # Semi-open file for 'color'
                score += self.rules.get('open_file_bonus', 0.0) / 2 # Half bonus for semi-open (tuneable)

            # Bonus if a rook is on an open or semi-open file
            if any(board.piece_at(chess.square(file, r)) == chess.Piece(chess.ROOK, color) for r in range(8)):
                if is_file_open or (not is_file_open and not has_own_pawn_on_file): # If open or semi-open
                    score += self.rules.get('file_control_bonus', 0.0)
            
            # Exposed king penalty if king is on an open/semi-open file
            king_sq = board.king(color)
            if king_sq is not None and chess.square_file(king_sq) == file:
                if is_file_open or (not is_file_open and not has_own_pawn_on_file): # If king is on an open/semi-open file
                    score += self.rules.get('exposed_king_penalty', 0.0)

        return score
    
    def _stalemate(self, board: chess.Board) -> float:
        """Check if the position is a stalemate"""
        if board.is_stalemate():
            return self.rules.get('stalemate_penalty', 0.0)
        return 0.0
    
    def _queen_capture(self, board: chess.Board):
        """
        Awards a large bonus for capturing the opponent's queen with a lesser-valued piece.
        """
        move = board.pop() # Get the last move played
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece and captured_piece.piece_type == chess.QUEEN:
                mover_piece = board.piece_at(move.from_square)
                if mover_piece and mover_piece.piece_type != chess.QUEEN:
                    return self.rules.get('queen_capture_bonus', 1000.0)
        return 0.0

    def _king_endangerment(self, board: chess.Board):
        """
        Penalizes non-castling king moves outside the endgame.
        Uses self.game_phase (should be set by calculate_game_phase).
        """
        move = board.pop() # Get the last move played
        mover_piece = board.piece_at(move.from_square)
        if mover_piece and mover_piece.piece_type == chess.KING:
            # Ignore castling
            if board.is_castling(move):
                return 0.0
            # Penalize king moves in opening/middlegame
            if getattr(self, 'game_phase', None) != 'endgame':
                return self.rules.get('king_safety_penalty', -100.0)
        return 0.0