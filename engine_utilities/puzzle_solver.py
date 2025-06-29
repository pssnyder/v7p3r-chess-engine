# engine_utilities/puzzle_solver.py
"""
Puzzle Solver
"""
import chess
import yaml
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
from v7p3r_engine.v7p3r_score import v7p3rScore
from v7p3r_engine.v7p3r_pst import v7p3rPST

class v7p3rSolver:
    def __init__(self, engine_config, logger: logging.Logger):
        self.engine_config = engine_config
        self.logger = logger
        self.pst = v7p3rPST()
        self.ruleset_name = engine_config.get('engine_ruleset', 'default_evaluation')
        # Required Scoring Modules
        self.pst = v7p3rPST()
        self.scoring_calculator = v7p3rScore(engine_config=engine_config, pst=self.pst, logger=self.logger)
        self.endgame_factor = 0.0  # Default endgame factor, adjusted during gameplay

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
    
    def solve_puzzle(self, fen: str):
        """Solve a chess puzzle by finding the best move."""
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError("Invalid chess board state")
        
        best_move = None
        best_score = float('-inf')
        
        for move in board.legal_moves:
            board.push(move)
            score = self._evaluate_position(board)
            board.pop()  # Undo the move
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move, best_score
    
    def _calculate_score(self, board: chess.Board, color: chess.Color, endgame_factor: float = 0.0) -> float:
        """
        Local override for scoring calculation function and engine thought deep dive.
        """
        score = 0.0
        
        # Helper for consistent color display in logs
        color_name = "White" if color == chess.WHITE else "Black"
        
        # Critical scoring components
        checkmate_threats_score = 1.0 * (self.scoring_calculator._checkmate_threats(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Checkmate threats score for {color_name}: {checkmate_threats_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Checkmate threats score for {color_name}: {checkmate_threats_score:.3f} (Ruleset: {self.ruleset_name})")
        score += checkmate_threats_score

        king_safety_score = 1.0 * (self.scoring_calculator._king_safety(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"King safety score for {color_name}: {king_safety_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"King safety score for {color_name}: {king_safety_score:.3f} (Ruleset: {self.ruleset_name})")
        score += king_safety_score
        
        king_threat_score = 1.0 * (self.scoring_calculator._king_threat(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"King threat score for {color_name}: {king_threat_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"King threat score for {color_name}: {king_threat_score:.3f} (Ruleset: {self.ruleset_name})")
        score += king_threat_score
        
        king_endangerment_score = 1.0 * (self.scoring_calculator._king_endangerment(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"King endangerment score for {color_name}: {king_endangerment_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"King endangerment score for {color_name}: {king_endangerment_score:.3f} (Ruleset: {self.ruleset_name})")
        score += king_endangerment_score

        draw_scenarios_score = 1.0 * (self.scoring_calculator._draw_scenarios(board) or 0.0)
        if self.logger:
            self.logger.debug(f"Draw scenarios score for {color_name}: {draw_scenarios_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Draw scenarios score for {color_name}: {draw_scenarios_score:.3f} (Ruleset: {self.ruleset_name})")
        score += draw_scenarios_score
        
        # Material and piece-square table evaluation
        score += 1.0 * self.scoring_calculator._material_score(board, color)
        pst_board_score = self.pst.evaluate_board_position(board, endgame_factor)
        if color == chess.BLACK:
            pst_board_score = -pst_board_score
        if self.logger:
            self.logger.debug(f"Piece-square table score for {color_name}: {pst_board_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Piece-square table score for {color_name}: {pst_board_score:.3f} (Ruleset: {self.ruleset_name})")
        score += 1.0 * pst_board_score

        # Piece coordination and control
        piece_coordination_score = 1.0 * (self.scoring_calculator._piece_coordination(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Piece coordination score for {color_name}: {piece_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Piece coordination score for {color_name}: {piece_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_coordination_score
        
        center_control_score = 1.0 * (self.scoring_calculator._center_control(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Center control score for {color_name}: {center_control_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Center control score for {color_name}: {center_control_score:.3f} (Ruleset: {self.ruleset_name})")
        score += center_control_score
        
        pawn_structure_score = 1.0 * (self.scoring_calculator._pawn_structure(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Pawn structure score for {color_name}: {pawn_structure_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Pawn structure score for {color_name}: {pawn_structure_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_structure_score
        
        pawn_weaknesses_score = 1.0 * (self.scoring_calculator._pawn_weaknesses(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Pawn weaknesses score for {color_name}: {pawn_weaknesses_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Pawn weaknesses score for {color_name}: {pawn_weaknesses_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_weaknesses_score
        
        passed_pawns_score = 1.0 * (self.scoring_calculator._passed_pawns(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Passed pawns score for {color_name}: {passed_pawns_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Passed pawns score for {color_name}: {passed_pawns_score:.3f} (Ruleset: {self.ruleset_name})")
        score += passed_pawns_score
        
        pawn_majority_score = 1.0 * (self.scoring_calculator._pawn_majority(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Pawn majority score for {color_name}: {pawn_majority_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Pawn majority score for {color_name}: {pawn_majority_score:.3f} (Ruleset: {self.ruleset_name})")
        score += pawn_majority_score
        
        bishop_pair_score = 1.0 * (self.scoring_calculator._bishop_pair(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Bishop pair score for {color_name}: {bishop_pair_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Bishop pair score for {color_name}: {bishop_pair_score:.3f} (Ruleset: {self.ruleset_name})")
        score += bishop_pair_score
        
        knight_pair_score = 1.0 * (self.scoring_calculator._knight_pair(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Knight pair score for {color_name}: {knight_pair_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Knight pair score for {color_name}: {knight_pair_score:.3f} (Ruleset: {self.ruleset_name})")
        score += knight_pair_score
        
        bishop_vision_score = 1.0 * (self.scoring_calculator._bishop_vision(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Bishop vision score for {color_name}: {bishop_vision_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Bishop vision score for {color_name}: {bishop_vision_score:.3f} (Ruleset: {self.ruleset_name})")
        score += bishop_vision_score
        
        rook_coordination_score = 1.0 * (self.scoring_calculator._rook_coordination(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Rook coordination score for {color_name}: {rook_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Rook coordination score for {color_name}: {rook_coordination_score:.3f} (Ruleset: {self.ruleset_name})")
        score += rook_coordination_score
        
        castling_evaluation_score = 1.0 * (self.scoring_calculator._castling_evaluation(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Castling evaluation score for {color_name}: {castling_evaluation_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Castling evaluation score for {color_name}: {castling_evaluation_score:.3f} (Ruleset: {self.ruleset_name})")
        score += castling_evaluation_score
        
        # Piece development and mobility
        piece_activity_score = 1.0 * (self.scoring_calculator._piece_activity(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Piece activity score for {color_name}: {piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Piece activity score for {color_name}: {piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        score += piece_activity_score
        
        improved_minor_piece_activity_score = 1.0 * (self.scoring_calculator._improved_minor_piece_activity(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Improved minor piece activity score for {color_name}: {improved_minor_piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Improved minor piece activity score for {color_name}: {improved_minor_piece_activity_score:.3f} (Ruleset: {self.ruleset_name})")
        score += improved_minor_piece_activity_score
        
        mobility_score = 1.0 * (self.scoring_calculator._mobility_score(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Mobility score for {color_name}: {mobility_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Mobility score for {color_name}: {mobility_score:.3f} (Ruleset: {self.ruleset_name})")
        score += mobility_score
        
        undeveloped_pieces_score = 1.0 * (self.scoring_calculator._undeveloped_pieces(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Undeveloped pieces score for {color_name}: {undeveloped_pieces_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Undeveloped pieces score for {color_name}: {undeveloped_pieces_score:.3f} (Ruleset: {self.ruleset_name})")
        score += undeveloped_pieces_score
        
        # Tactical and strategic considerations
        tactical_evaluation_score = 1.0 * (self.scoring_calculator._tactical_evaluation(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Tactical evaluation score for {color_name}: {self.scoring_calculator._tactical_evaluation(board, color):.3f} (Ruleset: {self.ruleset_name})")
        print(f"Tactical evaluation score for {color_name}: {tactical_evaluation_score:.3f} (Ruleset: {self.ruleset_name})")
        score += tactical_evaluation_score
        
        queen_capture_score = 1.0 * (self.scoring_calculator._queen_capture(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Queen capture score for {color_name}: {queen_capture_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Queen capture score for {color_name}: {queen_capture_score:.3f} (Ruleset: {self.ruleset_name})")
        score += queen_capture_score

        tempo_bonus_score = 1.0 * (self.scoring_calculator._tempo_bonus(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Tempo bonus score for {color_name}: {self.scoring_calculator._tempo_bonus(board, color):.3f} (Ruleset: {self.ruleset_name})")
        print(f"Tempo bonus score for {color_name}: {tempo_bonus_score:.3f} (Ruleset: {self.ruleset_name})")
        score += tempo_bonus_score
        
        special_moves_score = 1.0 * (self.scoring_calculator._special_moves(board, color) or 0.0) # Pass color
        if self.logger:
            self.logger.debug(f"Special moves score for {color_name}: {special_moves_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Special moves score for {color_name}: {special_moves_score:.3f} (Ruleset: {self.ruleset_name})")
        score += special_moves_score
        
        open_files_score = 1.0 * (self.scoring_calculator._open_files(board, color) or 0.0)
        if self.logger:
            self.logger.debug(f"Open files score for {color_name}: {open_files_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Open files score for {color_name}: {open_files_score:.3f} (Ruleset: {self.ruleset_name})")
        score += open_files_score
        
        stalemate_score = 1.0 * (self.scoring_calculator._stalemate(board) or 0.0)
        if self.logger:
            self.logger.debug(f"Stalemate score for {color_name}: {stalemate_score:.3f} (Ruleset: {self.ruleset_name})")
        print(f"Stalemate score for {color_name}: {stalemate_score:.3f} (Ruleset: {self.ruleset_name})")
        score += stalemate_score

        if self.logger:
            self.logger.debug(f"Final score for {color_name}: {score:.3f} (Ruleset: {self.ruleset_name}, Modifier: {1.0}) | FEN: {board.fen()}")

        return score

    def _evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate the position of the board using the scoring calculator.
        This is a local override to ensure we use the correct ruleset and scoring logic.
        """
        color = board.turn
        score = self._calculate_score(board, color, self.endgame_factor)
        
        if self.logger:
            self.logger.debug(f"Evaluated position score: {score:.3f} (FEN: {board.fen()})")
        
        return score
    
if __name__ == "__main__":
    # Example usage
    engine_config = {
        "print_scoring": True,
        "engine_ruleset": "default_evaluation"
    }
    
    logger = logging.getLogger("puzzle_solver_logger")
    logger.setLevel(logging.DEBUG)
    
    solver = v7p3rSolver(engine_config, logger)
    
    # Test with a sample FEN
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    best_move, best_score = solver.solve_puzzle(fen)
    
    print(f"Best move: {best_move}, Score: {best_score}")