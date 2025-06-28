# v7p3r_score.py

""" v7p3r Scoring Calculation Module
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

        # Scoring Config
        self.logger = logger if logger else logging.getLogger("v7p3r_engine_logger")
        self.print_scoring = engine_config.get('verbose_output', False)
        self.ruleset_name = engine_config.get('engine_ruleset', 'default_evaluation')
        # Required Scoring Modules
        self.pst = pst

        # Ruleset Loading
        self.rulesets = {}
        self.rules = {}
        try:
            with open("v7p3r_engine/rulesets.yaml") as f:
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
            self.logger.debug(f"Position evaluation from {player_name} perspective (delegated): {score:.3f} | FEN: {perspective_evaluation_board.fen()}")
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
        
        # Critical scoring components
        checkmate_threats_score = 1.0 * (self._checkmate_threats(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Checkmate threats score for {color_name}: {checkmate_threats_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Checkmate threats score for {color_name}: {checkmate_threats_score:.3f} (Ruleset: {self.ruleset_name})")
        score += checkmate_threats_score

        king_safety_score = 1.0 * (self._king_safety(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"King safety score for {color_name}: {king_safety_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"King safety score for {color_name}: {king_safety_score:.3f} (Ruleset: {self.ruleset_name})")
        score += king_safety_score
        
        king_threat_score = 1.0 * (self._king_threat(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"King threat score for {color_name}: {king_threat_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"King threat score for {color_name}: {king_threat_score:.3f} (Ruleset: {self.ruleset_name})")
        score += king_threat_score
        
        king_endangerment_score = 1.0 * (self._king_endangerment(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"King endangerment score for {color_name}: {king_endangerment_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"King endangerment score for {color_name}: {king_endangerment_score:.3f} (Ruleset: {self.ruleset_name})")
        score += king_endangerment_score

        draw_scenarios_score = 1.0 * (self._draw_scenarios(board) or 0.0)
        if self.logger:
            self.logger.debug(f"Draw scenarios score for {color_name}: {draw_scenarios_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Draw scenarios score for {color_name}: {draw_scenarios_score:.3f} (Ruleset: {self.ruleset_name})")
        score += draw_scenarios_score
        
        # Material and piece-square table evaluation
        score += 1.0 * self._material_score(board, color)
        pst_board_score = self.pst.evaluate_board_position(board, endgame_factor)
        if color == chess.BLACK:
            pst_board_score = -pst_board_score
        if self.logger:
            self.logger.debug(f"Piece-square table score for {color_name}: {pst_board_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Piece-square table score for {color_name}: {pst_board_score:.3f} (Ruleset: {self.ruleset_name})")
        score += 1.0 * pst_board_score

        # Piece coordination and control
        piece_coordination_score = 1.0 * (self._piece_coordination(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Piece coordination score for {color_name}: {piece_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Piece coordination score for {color_name}: {piece_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_coordination_score
        
        center_control_score = 1.0 * (self._center_control(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Center control score for {color_name}: {center_control_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Center control score for {color_name}: {center_control_score:.3f} (Ruleset: {self.ruleset_name})")
        score += center_control_score
        
        pawn_structure_score = 1.0 * (self._pawn_structure(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Pawn structure score for {color_name}: {pawn_structure_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Pawn structure score for {color_name}: {pawn_structure_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_structure_score
        
        pawn_weaknesses_score = 1.0 * (self._pawn_weaknesses(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Pawn weaknesses score for {color_name}: {pawn_weaknesses_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Pawn weaknesses score for {color_name}: {pawn_weaknesses_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_weaknesses_score
        
        passed_pawns_score = 1.0 * (self._passed_pawns(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Passed pawns score for {color_name}: {passed_pawns_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Passed pawns score for {color_name}: {passed_pawns_score:.3f} (Ruleset: {self.ruleset_name})")
        score += passed_pawns_score
        
        pawn_majority_score = 1.0 * (self._pawn_majority(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Pawn majority score for {color_name}: {pawn_majority_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Pawn majority score for {color_name}: {pawn_majority_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_majority_score
        
        bishop_pair_score = 1.0 * (self._bishop_pair(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Bishop pair score for {color_name}: {bishop_pair_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Bishop pair score for {color_name}: {bishop_pair_score:.3f} (Ruleset: {self.ruleset_name})")
        score += bishop_pair_score
        
        knight_pair_score = 1.0 * (self._knight_pair(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Knight pair score for {color_name}: {knight_pair_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Knight pair score for {color_name}: {knight_pair_score:.3f} (Ruleset: {self.ruleset_name})")
        score += knight_pair_score
        
        bishop_vision_score = 1.0 * (self._bishop_vision(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Bishop vision score for {color_name}: {bishop_vision_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Bishop vision score for {color_name}: {bishop_vision_score:.3f} (Ruleset: {self.ruleset_name})")
        score += bishop_vision_score
        
        rook_coordination_score = 1.0 * (self._rook_coordination(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Rook coordination score for {color_name}: {rook_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Rook coordination score for {color_name}: {rook_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        score += rook_coordination_score
        
        castling_evaluation_score = 1.0 * (self._castling_evaluation(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Castling evaluation score for {color_name}: {castling_evaluation_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Castling evaluation score for {color_name}: {castling_evaluation_score:.3f} (Ruleset: {self.ruleset_name})")
        score += castling_evaluation_score
        
        # Piece development and mobility
        piece_activity_score = 1.0 * (self._piece_activity(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Piece activity score for {color_name}: {piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Piece activity score for {color_name}: {piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_activity_score
        
        improved_minor_piece_activity_score = 1.0 * (self._improved_minor_piece_activity(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Improved minor piece activity score for {color_name}: {improved_minor_piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Improved minor piece activity score for {color_name}: {improved_minor_piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        score += improved_minor_piece_activity_score
        
        mobility_score = 1.0 * (self._mobility_score(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Mobility score for {color_name}: {mobility_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Mobility score for {color_name}: {mobility_score:.3f} (Ruleset: {self.ruleset_name})")
        score += mobility_score
        
        undeveloped_pieces_score = 1.0 * (self._undeveloped_pieces(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Undeveloped pieces score for {color_name}: {undeveloped_pieces_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Undeveloped pieces score for {color_name}: {undeveloped_pieces_score:.3f} (Ruleset: {self.ruleset_name})")
        score += undeveloped_pieces_score
        
        # Tactical and strategic considerations
        tactical_evaluation_score = 1.0 * (self._tactical_evaluation(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Tactical evaluation score for {color_name}: {self._tactical_evaluation(board, color):.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Tactical evaluation score for {color_name}: {tactical_evaluation_score:.3f} (Ruleset: {self.ruleset_name})")
        score += tactical_evaluation_score
        
        queen_capture_score = 1.0 * (self._queen_capture(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Queen capture score for {color_name}: {queen_capture_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Queen capture score for {color_name}: {queen_capture_score:.3f} (Ruleset: {self.ruleset_name})")
        score += queen_capture_score

        tempo_bonus_score = 1.0 * (self._tempo_bonus(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Tempo bonus score for {color_name}: {self._tempo_bonus(board, color):.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Tempo bonus score for {color_name}: {tempo_bonus_score:.3f} (Ruleset: {self.ruleset_name})")
        score += tempo_bonus_score
        
        special_moves_score = 1.0 * (self._special_moves(board, color) or 0.0) # Pass color
        if self.logger:
            self.logger.debug(f"Special moves score for {color_name}: {special_moves_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Special moves score for {color_name}: {special_moves_score:.3f} (Ruleset: {self.ruleset_name})")
        score += special_moves_score
        
        open_files_score = 1.0 * (self._open_files(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Open files score for {color_name}: {open_files_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Open files score for {color_name}: {open_files_score:.3f} (Ruleset: {self.ruleset_name})")
        score += open_files_score
        
        stalemate_score = 1.0 * (self._stalemate(board) or 0.0)
        if self.logger:
            self.logger.debug(f"Stalemate score for {color_name}: {stalemate_score:.3f} (Ruleset: {self.ruleset_name})")
        if self.print_scoring:
            print(f"Stalemate score for {color_name}: {stalemate_score:.3f} (Ruleset: {self.ruleset_name})")
        score += stalemate_score

        self.calculate_game_phase(board)

        if self.logger:
            self.logger.debug(f"Final score for {color_name}: {score:.3f} (Ruleset: {self.ruleset_name}, Modifier: {1.0}) | FEN: {board.fen()}")

        return score

    def calculate_game_phase(self, board: chess.Board) -> str:
        """
        Determines the current phase of the game: 'opening', 'middlegame', or 'endgame'.
        Uses material and castling rights as heuristics.
        Sets self.game_phase for use in other scoring functions.
        """
        phase = 'opening'
        endgame_factor = 0.0
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
            # Heuristic: if less than 24 pieces are on the board and one player has castled
            endgame_factor = 0.5
            if material < 20 and not board.has_castling_rights(chess.WHITE) and not board.has_castling_rights(chess.BLACK):
                # Heuristic: if less than 20 pieces are on the board and 
                endgame_factor = 0.75
        elif material <= 32 and (board.has_castling_rights(chess.WHITE) and board.has_castling_rights(chess.BLACK)):
            # Opening Phase
            phase = 'opening'
            if material == 32:
                # Heuristic: all material remains on the board, fully stable/closed position
                endgame_factor = 0.0
            elif material < 32:    
                endgame_factor = 0.1
            elif material <= 28 and (board.has_castling_rights(chess.WHITE) and board.has_castling_rights(chess.BLACK)):
                # Progressing into opening position remains closed and stable
                endgame_factor = 0.2
            elif material <= 20 and (board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK)):
                # Traisitioning into 
                endgame_factor = 0.5
            
        self.game_phase = phase
        self.game_factor = endgame_factor
        return phase

    # ==========================================
    # ========= RULE SCORING FUNCTIONS =========

    def _checkmate_threats(self, board: chess.Board, color: chess.Color) -> float:
        """
        Assess if 'color' can deliver a checkmate on their next move.
        Only consider legal moves for 'color' without mutating the original board's turn.
        """
        score = 0.0

        if board.is_checkmate() and board.turn == (color == chess.WHITE):
            score += self.rules.get('checkmate_bonus', 0)
        return score

    def _draw_scenarios(self, board: chess.Board) -> float:
        score = 0.0
        if board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition() or board.is_repetition(count=2) or board.is_seventyfive_moves() or board.is_variant_draw():
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
    
    def _special_moves(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate special moves and opportunities for the given color - STATIC EVALUATION"""
        score = 0.0
        
        # En passant opportunity for 'color' - check if any of our pawns can capture en passant
        if board.ep_square:
            ep_file = chess.square_file(board.ep_square)
            ep_rank = chess.square_rank(board.ep_square)
            
            # Check if any of our pawns can capture the en passant square
            for file_offset in [-1, 1]:
                potential_pawn_file = ep_file + file_offset
                if 0 <= potential_pawn_file <= 7:
                    # For white, pawn would be on rank 4 (index 4), for black on rank 3 (index 3)
                    pawn_rank = 4 if color == chess.WHITE else 3
                    potential_pawn_square = chess.square(potential_pawn_file, pawn_rank)
                    piece = board.piece_at(potential_pawn_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        score += self.rules.get('en_passant_bonus', 0.0)
                        break  # Found one en passant opportunity
        
        # Promotion opportunities for 'color' - check pawns on 7th/2nd rank
        promotion_rank = 6 if color == chess.WHITE else 1  # 7th rank for white, 2nd rank for black
        for pawn_square in board.pieces(chess.PAWN, color):
            if chess.square_rank(pawn_square) == promotion_rank:
                score += self.rules.get('pawn_promotion_bonus', 0.0)
        
        return score

    def _tactical_evaluation(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate tactical elements related to hanging pieces - STATIC EVALUATION"""
        score = 0.0
        opponent_color = not color

        # Check all squares for hanging pieces and piece safety
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == opponent_color:
                # Check if opponent piece is attacked by us and not defended by opponent
                if board.is_attacked_by(color, square) and not board.is_attacked_by(opponent_color, square):
                    score += self.rules.get('hanging_piece_bonus', 0.0)
            elif piece and piece.color == color:
                # Penalty for our pieces being attacked by opponent and not defended by us
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
        """Calculate bonus for rook pairs on same file/rank and 7th rank - FIXED DOUBLE COUNTING"""
        score = 0.0
        rooks = list(board.pieces(chess.ROOK, color))

        # Check individual rook positions first (7th rank bonus)
        for rook_square in rooks:
            # Rook on 7th rank bonus (critical for attacking pawns)
            # White on rank 7 (index 6) or black on rank 2 (index 1)
            if (color == chess.WHITE and chess.square_rank(rook_square) == 6) or \
               (color == chess.BLACK and chess.square_rank(rook_square) == 1):
                score += self.rules.get('rook_position_bonus', 0.0)

        # Check rook coordination (pairs)
        for i in range(len(rooks)):
            for j in range(i+1, len(rooks)):
                sq1, sq2 = rooks[i], rooks[j]
                # Same file (stacked rooks)
                if chess.square_file(sq1) == chess.square_file(sq2):
                    score += self.rules.get('stacked_rooks_bonus', 0.0)
                # Same rank (coordinated rooks)
                if chess.square_rank(sq1) == chess.square_rank(sq2):
                    score += self.rules.get('coordinated_rooks_bonus', 0.0)
        
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
    
    def _queen_capture(self, board: chess.Board, color: chess.Color) -> float:
        """
        Evaluate if the opponent's queen is under attack by a lesser piece - STATIC EVALUATION
        """
        score = 0.0
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
                        score += self.rules.get('queen_capture_bonus', 100.0)
                        break  # Only count once per queen under attack
        
        return score

    def _king_endangerment(self, board: chess.Board, color: chess.Color) -> float:
        """
        Evaluate king safety based on current position - STATIC EVALUATION
        Penalizes exposed king position outside the endgame.
        """
        score = 0.0
        king_square = board.king(color)
        if king_square is None:
            return score
        
        # Count pieces to determine game phase (simplified)
        total_pieces = len(board.piece_map())
        is_endgame = total_pieces <= 10  # Rough endgame threshold
        
        if not is_endgame:
            # In opening/middlegame, penalize king away from starting position if not castled
            initial_king_square = chess.E1 if color == chess.WHITE else chess.E8
            castled_squares = {
                chess.WHITE: [chess.G1, chess.C1],  # Kingside, queenside
                chess.BLACK: [chess.G8, chess.C8]
            }
            
            # If king is not on initial square and not castled, apply penalty
            if king_square != initial_king_square and king_square not in castled_squares[color]:
                score += self.rules.get('king_safety_penalty', -50.0)
                
            # Additional penalty if king is in center files during opening/middlegame
            king_file = chess.square_file(king_square)
            if 2 <= king_file <= 5:  # Files c-f (danger zone)
                score += self.rules.get('king_safety_penalty', -50.0) / 2
        
        return score
    

# Testing
if __name__ == "__main__":
    # Print the current default config values and run a test scoring calculation
    
    # Initialize required arguments
    engine_config = {
        'verbose_output': True,
        'engine_ruleset': 'default_evaluation'
    }
    from v7p3r_pst import v7p3rPST
    pst = v7p3rPST()  # Replace with an actual piece-square table object
    logger = logging.getLogger("v7p3r_engine_logger")
    
    scoring_calculator = v7p3rScore(engine_config=engine_config, pst=pst, logger=logger)
    print("Current default config values:")
    for key, value in scoring_calculator.rules.items():
        print(f"  {key}: {value}")
    
    # Run a test scoring calculation
    board = chess.Board()
    score = scoring_calculator.calculate_score(board=board, color=chess.WHITE)
    print(f"Test scoring calculation result: {score}")