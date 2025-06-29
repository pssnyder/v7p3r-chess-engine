# v7p3r_score.py

""" v7p3r Scoring Calculation Module (New Simplified Version)
This module is responsible for calculating the score of a chess position based on various factors,
including material balance, piece-square tables, king safety, and other positional features.
It is designed to be used by the v7p3r chess engine.
"""

import chess
import yaml
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import logging

class v7p3rScore:
    def __init__(self, engine_config: dict, pst, logger: logging.Logger):
        """ Initialize the scoring calculation engine with configuration settings.  """
        # Engine Configuration
        self.engine_config = engine_config

        # Scoring Config
        self.logger = logger if logger else logging.getLogger("v7p3r_engine_logger")
        self.print_scoring = engine_config.get('verbose_output', False)
        self.ruleset_name = engine_config.get('engine_ruleset', 'default_evaluation')
        
        self.use_game_phase = engine_config.get('use_game_phase', False)
        self.bonus_fallback = 100
        self.penalty_fallback = -100

        # Required Scoring Modules
        self.pst = pst

        # Ruleset Loading
        self.rulesets = {}
        self.rules = {}
        try:
            with open("v7p3r_engine/rulesets/rulesets.yaml") as f:
                self.rulesets = yaml.safe_load(f) or {}
                self.rules = self.rulesets.get(f"{self.ruleset_name}", {})
        except Exception as e:
            self.logger.error(f"Error loading rulesets file: {e}")

        if self.logger:
            self.logger.debug(f"v7p3rScoringCalculation initialized with ruleset: {self.ruleset_name}")
            self.logger.debug(f"Current ruleset parameters: {self.rules}")

    # =================================
    # ===== EVALUATION FUNCTIONS ======

    def evaluate_position(self, board: chess.Board) -> float:
        """Calculate position evaluation from specified player's perspective by delegating to scoring_calculator."""
        perspective_evaluation_board = board.copy()
        player = perspective_evaluation_board.turn
        if self.use_game_phase:
            self.calculate_game_phase(perspective_evaluation_board)
        if not isinstance(player, chess.Color) or not perspective_evaluation_board.is_valid():
            if self.logger:
                player_name = "White" if player == chess.WHITE else "Black" if isinstance(player, chess.Color) else str(player)
                self.logger.error(f"Invalid input for evaluation from perspective. Player: {player_name}, FEN: {perspective_evaluation_board.fen() if hasattr(perspective_evaluation_board, 'fen') else 'N/A'}")
            return 0.0

        white_score = self.calculate_score(
            board=perspective_evaluation_board,
            color=chess.WHITE,
        )
        black_score = self.calculate_score(
            board=perspective_evaluation_board,
            color=chess.BLACK,
        )
        
        score = (white_score - black_score) if player == chess.WHITE else (black_score - white_score)
        
        if self.logger:
            player_name = "White" if player == chess.WHITE else "Black"
            self.logger.debug(f"Position evaluation from {player_name} perspective ({self.game_phase}: {self.game_factor}): {score:.3f} | FEN: {perspective_evaluation_board.fen()}")
        return score
    
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
            if self.logger:
                self.logger.debug(f"[Scoring Calc] Starting score calculation for {current_player_name} engine as {color_name} | Ruleset: {self.ruleset_name} | FEN: {board.fen()}")
            if self.print_scoring:
                print(f"[Scoring Calc] Starting score calculation for {current_player_name} engine as {color_name} | Ruleset: {self.ruleset_name} | FEN: {board.fen()}")
        
        # Critical scoring components
        checkmate_threats_score = 1.0 * (self._checkmate_threats(board) or 0.0)
        if v7p3r_thinking:
            if self.logger:
                self.logger.debug(f"[Scoring Calc] Checkmate threats score for {color_name}: {checkmate_threats_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Checkmate threats score for {color_name}: {checkmate_threats_score:.3f} (Ruleset: {self.ruleset_name})")
        score += checkmate_threats_score

        draw_scenarios_score = 1.0 * (self._draw_scenarios(board) or 0.0)
        if v7p3r_thinking:
            if self.logger:
                self.logger.debug(f"[Scoring Calc] Draw scenarios score for {color_name}: {draw_scenarios_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Draw scenarios score for {color_name}: {draw_scenarios_score:.3f} (Ruleset: {self.ruleset_name})")
        score += draw_scenarios_score
        
        # Material and piece-square table evaluation
        score += 1.0 * self._material_score(board, color)
        pst_board_score = self.pst.evaluate_board_position(board, endgame_factor)
        if color == chess.BLACK:
            pst_board_score = -pst_board_score
        if v7p3r_thinking:
            if self.logger:
                self.logger.debug(f"[Scoring Calc] Piece-square table score for {color_name}: {pst_board_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Piece-square table score for {color_name}: {pst_board_score:.3f} (Ruleset: {self.ruleset_name})")
        score += 1.0 * pst_board_score

        # Piece coordination and control
        piece_coordination_score = 1.0 * (self._piece_coordination(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger:
                self.logger.debug(f"[Scoring Calc] Piece coordination score for {color_name}: {piece_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Piece coordination score for {color_name}: {piece_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_coordination_score
        
        center_control_score = 1.0 * (self._center_control(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger:
                self.logger.debug(f"[Scoring Calc] Center control score for {color_name}: {center_control_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Center control score for {color_name}: {center_control_score:.3f} (Ruleset: {self.ruleset_name})")
        score += center_control_score
        
        pawn_majority_score = 1.0 * (self._pawn_majority(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger:
                self.logger.debug(f"[Scoring Calc] Pawn majority score for {color_name}: {pawn_majority_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Pawn majority score for {color_name}: {pawn_majority_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_majority_score
        
        bishop_pair_score = 1.0 * (self._bishop_pair(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger:
                self.logger.debug(f"[Scoring Calc] Bishop pair score for {color_name}: {bishop_pair_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Bishop pair score for {color_name}: {bishop_pair_score:.3f} (Ruleset: {self.ruleset_name})")
        score += bishop_pair_score
        
        knight_pair_score = 1.0 * (self._knight_pair(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger:
                self.logger.debug(f"[Scoring Calc] Knight pair score for {color_name}: {knight_pair_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Knight pair score for {color_name}: {knight_pair_score:.3f} (Ruleset: {self.ruleset_name})")
        score += knight_pair_score
        
        castling_evaluation_score = 1.0 * (self._castling_evaluation(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger:
                self.logger.debug(f"[Scoring Calc] Castling evaluation score for {color_name}: {castling_evaluation_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Castling evaluation score for {color_name}: {castling_evaluation_score:.3f} (Ruleset: {self.ruleset_name})")
        score += castling_evaluation_score
        
        mobility_score = 1.0 * (self._mobility_score(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger:
                self.logger.debug(f"[Scoring Calc] Mobility score for {color_name}: {mobility_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Mobility score for {color_name}: {mobility_score:.3f} (Ruleset: {self.ruleset_name})")
        score += mobility_score
        
        undeveloped_pieces_score = 1.0 * (self._undeveloped_pieces(board, color) or 0.0)
        if v7p3r_thinking:
            if self.logger:
                self.logger.debug(f"[Scoring Calc] Undeveloped pieces score for {color_name}: {undeveloped_pieces_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Undeveloped pieces score for {color_name}: {undeveloped_pieces_score:.3f} (Ruleset: {self.ruleset_name})")
        score += undeveloped_pieces_score
        
        stalemate_score = 1.0 * (self._stalemate(board) or 0.0)
        if v7p3r_thinking:
            if self.logger:
                self.logger.debug(f"[Scoring Calc] Stalemate score for {color_name}: {stalemate_score:.3f} (Ruleset: {self.ruleset_name})")
            if self.print_scoring:
                print(f"[Scoring Calc] Stalemate score for {color_name}: {stalemate_score:.3f} (Ruleset: {self.ruleset_name})")
        score += stalemate_score

        # Log final score
        if v7p3r_thinking:
            if self.logger:
                self.logger.debug(f"[Scoring Calc] Final score for {current_player_name} as {color_name}: {score:.3f} | Ruleset: {self.ruleset_name} | FEN: {board.fen()}")
            if self.print_scoring:
                print(f"[Scoring Calc] Final score for {current_player_name} as {color_name}: {score:.3f} | Ruleset: {self.ruleset_name} | FEN: {board.fen()}")
        return score

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
        if material <= 10:
            # Endgame Phase
            phase = "endgame"
            endgame_factor = 1.0
        elif material < 25 and (not board.has_castling_rights(chess.WHITE) or not board.has_castling_rights(chess.BLACK)):
            # Middlegame Phase
            phase = "middlegame"
            # Heuristic: if less than 25 pieces are on the board and one player has castled
            endgame_factor = 0.5
            if material < 20 and not board.has_castling_rights(chess.WHITE) and not board.has_castling_rights(chess.BLACK):
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
        return

    # ==========================================
    # ========= RULE SCORING FUNCTIONS =========

    def _checkmate_threats(self, board: chess.Board) -> float:
        """
        Assess if the move is checkmate.
        """
        score = 0.0

        if board.is_checkmate():
            score += self.rules.get('checkmate_bonus', self.bonus_fallback)
        return score

    def _draw_scenarios(self, board: chess.Board) -> float:
        score = 0.0
        if board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition() or board.is_repetition(count=2) or board.is_seventyfive_moves() or board.is_variant_draw():
            score += self.rules.get('draw_penalty', self.penalty_fallback)
        return score

    def _material_score(self, board: chess.Board, color: chess.Color) -> float:
        """Simple material count for given color"""
        score = 0.0
        for piece_type, value in self.pst.piece_values.items():
            score += len(board.pieces(piece_type, color)) * value
        # Apply material weight from ruleset
        return score * self.rules.get('material_weight', 1.0)
    
    def _center_control(self, board: chess.Board, color: chess.Color) -> float:
        """Simple center control"""
        score = 0.0
        center = [chess.D4, chess.D5, chess.E4, chess.E5]
        for square in center:
            # Check if current player controls (has a piece on) center square
            piece = board.piece_at(square)
            if piece and piece.color == color:
                score += self.rules.get('center_control_bonus', self.bonus_fallback)
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
            score += undeveloped_count * self.rules.get('undeveloped_penalty', self.penalty_fallback)

        return score

    def _mobility_score(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate mobility of pieces"""
        score = 0.0
        
        # Iterate over all pieces of the given color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color and piece.piece_type != chess.KING: # Exclude king from general mobility
                score += len(list(board.attacks(square))) * self.rules.get('piece_mobility_bonus', self.bonus_fallback)

        return score
    
    def _castling_evaluation(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate castling rights and opportunities"""
        score = 0.0

        # Check if castled - more robust check considering king's final position
        king_sq = board.king(color)
        if king_sq: # Ensure king exists
            if color == chess.WHITE:
                if king_sq == chess.G1: # Kingside castled
                    score += self.rules.get('castling_bonus', self.bonus_fallback)
                elif king_sq == chess.C1: # Queenside castled
                    score += self.rules.get('castling_bonus', self.bonus_fallback)
            else: # Black
                if king_sq == chess.G8: # Kingside castled
                    score += self.rules.get('castling_bonus', self.bonus_fallback)
                elif king_sq == chess.C8: # Queenside castled
                    score += self.rules.get('castling_bonus', self.bonus_fallback)

        # Penalty if castling rights lost and not yet castled
        initial_king_square = chess.E1 if color == chess.WHITE else chess.E8
        if not board.has_castling_rights(color) and king_sq == initial_king_square:
            score += self.rules.get('castling_protection_penalty', self.penalty_fallback)
        
        # Bonus if still has kingside or queenside castling rights
        if board.has_kingside_castling_rights(color) and board.has_queenside_castling_rights(color):
            score += self.rules.get('castling_protection_bonus', self.bonus_fallback)
        elif board.has_kingside_castling_rights(color) or board.has_queenside_castling_rights(color):
            score += self.rules.get('castling_protection_bonus', self.bonus_fallback) / 2

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
                    score += self.rules.get('piece_coordination_bonus', self.bonus_fallback) 
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
                score += self.rules.get('pawn_majority_bonus', self.bonus_fallback) / 2 # Half bonus for kingside
            if white_pawns_queenside > black_pawns_queenside:
                score += self.rules.get('pawn_majority_bonus', self.bonus_fallback) / 2 # Half bonus for queenside
            # Optionally add penalty for minority
            if white_pawns_kingside < black_pawns_kingside:
                score -= self.rules.get('pawn_majority_penalty', self.penalty_fallback) / 2
            if white_pawns_queenside < black_pawns_queenside:
                score -= self.rules.get('pawn_majority_penalty', self.penalty_fallback) / 2
        else: # Black
            if black_pawns_kingside > white_pawns_kingside:
                score += self.rules.get('pawn_majority_bonus', self.bonus_fallback) / 2
            if black_pawns_queenside > white_pawns_queenside:
                score += self.rules.get('pawn_majority_bonus', self.bonus_fallback) / 2
            # Optionally add penalty for minority
            if black_pawns_kingside < white_pawns_kingside:
                score -= self.rules.get('pawn_majority_penalty', self.penalty_fallback) / 2
            if black_pawns_queenside < white_pawns_queenside:
                score -= self.rules.get('pawn_majority_penalty', self.penalty_fallback) / 2

        return score

    def _knight_pair(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate knight pair bonus"""
        score = 0.0
        knights = list(board.pieces(chess.KNIGHT, color))
        if len(knights) >= 2:
            score += self.rules.get('knight_pair_bonus', self.bonus_fallback) # Bonus for having *a* knight pair
            # If the bonus is per knight in a pair, it would be len(knights) * bonus / 2 (or similar)
        return score

    def _bishop_pair(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate bishop pair bonus"""
        score = 0.0
        bishops = list(board.pieces(chess.BISHOP, color))
        if len(bishops) >= 2:
            score += self.rules.get('bishop_pair_bonus', self.bonus_fallback)
        return score

    def _stalemate(self, board: chess.Board) -> float:
        """Check if the position is a stalemate"""
        if board.is_stalemate():
            return self.rules.get('stalemate_penalty', self.penalty_fallback)
        return 0.0
    

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
    pst = v7p3rPST(engine_config['piece_values'], logging.getLogger("v7p3r_pst_logger"))
    logger = logging.getLogger("v7p3r_engine_logger")
    
    scoring_calculator = v7p3rScore(engine_config=engine_config, pst=pst, logger=logger)
    print("Current default config values:")
    for key, value in scoring_calculator.rules.items():
        print(f"  {key}: {value}")
    
    # Run a test scoring calculation
    board = chess.Board()
    score = scoring_calculator.calculate_score(board=board, color=chess.WHITE)
    print(f"Test scoring calculation result: {score}")