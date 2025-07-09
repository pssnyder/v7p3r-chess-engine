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

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class v7p3rScore:
    def __init__(self, rules_manager, pst):
        """ Initialize the scoring calculation engine with configuration settings.  """
        # Engine Configuration
        self.config_manager = v7p3rConfig()
        self.engine_config = self.config_manager.get_engine_config()  # Ensure it's always a dictionary
        self.game_config = self.config_manager.get_game_config()

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
            'evaluation': 0.0,
            'game_phase': self.game_phase,
            'endgame_factor': self.endgame_factor,
            'checkmate_threats_score': 0.0,
            'material_count': 0.0,
            'material_score': 0.0,
            'pst_score': 0.0,
            'piece_captures_score': 0.0,
            'castling_score': 0.0,
            'total_score': 0.0,
        }

    # =================================
    # ===== EVALUATION FUNCTIONS ======

    def evaluate_position(self, board: chess.Board) -> float:
        """Calculate position evaluation from general perspective by delegating to scoring_calculator."""
        if not isinstance(board, chess.Board) or not board.is_valid():
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
        self.score_dataset['evaluation'] = score
        return score
    
    def evaluate_position_from_perspective(self, board: chess.Board, color: Optional[chess.Color] = chess.WHITE) -> float:
        """Calculate position evaluation from specified player's perspective by delegating to scoring_calculator."""
        self.color_name = "White" if color == chess.WHITE else "Black"
        
        if not isinstance(color, chess.Color) or not board.is_valid():
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
        
        # CHECKMATE THREATS - Most critical factor
        checkmate_threats_score = self.rules_manager.checkmate_threats(board, color) or 0.0
        score += checkmate_threats_score
        self.score_dataset['checkmate_threats_score'] = checkmate_threats_score

        # MATERIAL COUNT
        material_count = self.rules_manager.material_count(board, color) or 0.0
        score += material_count
        self.score_dataset['material_count'] = material_count

        # MATERIAL SCORE
        material_score = self.rules_manager.material_score(board, color) or 0.0
        score += material_score
        self.score_dataset['material_score'] = material_score
        
        # Stop Check - Return if we're in a very clear position (queen + multiple pieces advantage)
        if (material_count > 10 or material_score > 1500) and not board.is_check():  # Big material advantage and not in check
            return score
        
        # GAME PHASE
        if self.use_game_phase:
            self._calculate_game_phase(board)
        
        # PST SCORE
        pst_score = self.rules_manager.pst_score(board, color, self.endgame_factor) or 0.0
        score += pst_score
        self.score_dataset['pst_score'] = pst_score
        
        # PIECE CAPTURES
        piece_captures_score = self.rules_manager.piece_captures(board, color) or 0.0
        score += piece_captures_score
        self.score_dataset['piece_captures'] = piece_captures_score
        
        # CASTLING
        castling_score = self.rules_manager.castling(board, color) or 0.0
        score += castling_score
        self.score_dataset['castling_score'] = castling_score
        
        # Get a final evaluation score
        self.score_dataset['evaluation'] = score
        
        return score