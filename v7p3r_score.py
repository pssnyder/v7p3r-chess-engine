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
from v7p3r_utilities import get_timestamp
from v7p3r_mvv_lva import v7p3rMVVLVA
from v7p3r_tempo import v7p3rTempo  # Import tempo assessment

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
        self.mvv_lva = v7p3rMVVLVA()
        self.tempo = v7p3rTempo(self.config_manager, pst, rules_manager)  # Add tempo manager

        # Scoring Setup
        self.ruleset_name = self.engine_config.get('ruleset', 'default_ruleset')
        self.ruleset = self.config_manager.get_ruleset()
        
        # Ensure ruleset is a dictionary, not a string
        if isinstance(self.ruleset, str):
            self.ruleset = {}
            
        # Scoring control flags from config
        self.use_game_phase = self.engine_config.get('use_game_phase', True)
        self.use_primary_scoring = self.engine_config.get('use_primary_scoring', True)
        self.use_secondary_scoring = self.engine_config.get('use_secondary_scoring', True)
        self.use_mvv_lva = self.engine_config.get('use_mvv_lva', True)

        self.white_player = self.game_config.get('white_player', 'v7p3r')
        self.black_player = self.game_config.get('black_player', 'v7p3r')
        self.fallback_modifier = 100

        # Initialize scoring parameters
        self.root_board = chess.Board()
        self.game_phase = 'opening'  # Default game phase
        self.endgame_factor = 0.0  # Default endgame factor for endgame awareness
        self.score_counter = 0
        self.score_id = f"score[{self.score_counter}]_{get_timestamp()}"
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
            'tempo_score': 0.0,  # Add tempo score
            'total_score': 0.0,
        }

    # =================================
    # ===== EVALUATION FUNCTIONS ======

    def _evaluate_material(self, board: chess.Board) -> float:
        """Evaluate material balance between the two players."""
        material_score = 0.0
        
        # Material values (pawn = 1, knight = 3, bishop = 3, rook = 5, queen = 9)
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
        }
        
        # Calculate material score based on piece values
        for piece_type, value in piece_values.items():
            white_material = len(board.pieces(piece_type, chess.WHITE))
            black_material = len(board.pieces(piece_type, chess.BLACK))
            material_score += (white_material - black_material) * value
        
        return material_score

    def _evaluate_piece_development(self, board: chess.Board) -> float:
        """Evaluate piece development and control of the center"""
        score = 0.0
        
        # Center squares
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        near_center_squares = [chess.C3, chess.D3, chess.E3, chess.F3,
                             chess.C4, chess.F4,
                             chess.C5, chess.F5,
                             chess.C6, chess.D6, chess.E6, chess.F6]
        
        # Evaluate piece development and center control
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Center control bonus
                if square in center_squares:
                    score += 30 if piece.color == chess.WHITE else -30
                elif square in near_center_squares:
                    score += 15 if piece.color == chess.WHITE else -15
                
                # Development bonus for minor pieces
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    if piece.color == chess.WHITE and chess.square_rank(square) > 1:
                        score += 20
                    elif piece.color == chess.BLACK and chess.square_rank(square) < 6:
                        score -= 20
                
                # Penalty for early queen development
                if piece.piece_type == chess.QUEEN:
                    if piece.color == chess.WHITE and chess.square_rank(square) > 1:
                        score -= 30
                    elif piece.color == chess.BLACK and chess.square_rank(square) < 6:
                        score += 30
                
                # Severe penalty for early king movement
                if piece.piece_type == chess.KING:
                    if len(list(board.move_stack)) < 20:  # Early game
                        if piece.color == chess.WHITE and square != chess.E1:
                            score -= 100
                        elif piece.color == chess.BLACK and square != chess.E8:
                            score += 100
        
        return score

    def _calculate_mvv_lva_score(self, move: chess.Move, board: chess.Board) -> float:
        """Enhanced MVV-LVA with piece values from config and safety evaluation"""
        if not board.is_capture(move):
            return 0.0
            
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        
        if victim is None or attacker is None:
            return 0.0
            
        # Use piece values from config or fallback to defaults
        piece_values = self.rules_manager.piece_values
        
        victim_value = piece_values.get(victim.piece_type, 100)
        attacker_value = piece_values.get(attacker.piece_type, 100)
        
        # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
        # Prioritize capturing valuable pieces with less valuable pieces
        mvv_lva_score = victim_value * 10 - attacker_value
        
        # Add position context - capturing towards center is slightly better
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        if move.to_square in center_squares:
            mvv_lva_score += 10
            
        return mvv_lva_score

    def _evaluate_castling_state(self, board: chess.Board) -> float:
        """Evaluate castling opportunities and king safety"""
        score = 0.0
        
        # If we've castled, big bonus
        if board.has_castling_rights(chess.WHITE):
            if not board.king(chess.WHITE) == chess.E1:  # King has moved
                score -= self.pst.piece_values[chess.KING] * 0.1
        else:
            # Lost castling rights without castling
            if board.king(chess.WHITE) == chess.E1:  # Still on starting square
                score -= self.pst.piece_values[chess.KING] * 0.2
                
        if board.has_castling_rights(chess.BLACK):
            if not board.king(chess.BLACK) == chess.E8:  # King has moved
                score += self.pst.piece_values[chess.KING] * 0.1
        else:
            # Lost castling rights without castling
            if board.king(chess.BLACK) == chess.E8:  # Still on starting square
                score += self.pst.piece_values[chess.KING] * 0.2
                
        return score

    def evaluate_position(self, board: chess.Board, color: chess.Color = chess.WHITE) -> float:
        """Evaluate a chess position for the given color."""
        # Initialize score accumulator
        score = 0.0
        
        # 1. Check for checkmate/stalemate (highest priority)
        if board.is_checkmate():
            return -1000000.0 if board.turn == color else 1000000.0
        if board.is_stalemate():
            return self.tempo.stalemate_modifier
            
        # 2. Check for draws (second priority)
        if board.is_repetition(3) or board.is_fifty_moves() or board.is_insufficient_material():
            return self.tempo.draw_modifier

        # 3. Get comprehensive position assessment (new)
        assessment = self.tempo.assess_position(board, color)
        phase = assessment['game_phase']
        endgame_factor = assessment['endgame_factor']
        tempo_score = assessment['tempo_score']
        risk_score = assessment['risk_score']
        zugzwang_risk = assessment['zugzwang_risk']
        
        # 4. Primary scoring components
        if self.use_primary_scoring:
            material_score = self._evaluate_material(board)
            pst_score = self._evaluate_piece_square_tables(board)
            mvv_lva_score = 0.0
            if self.use_mvv_lva:
                for move in board.legal_moves:
                    mvv_lva_score += self.mvv_lva.calculate_mvv_lva_score(move, board) * (1 if board.turn == color else -1)
            
            # Apply phase-based weights
            score += material_score * (1.0 + 0.2 * endgame_factor)  # Material more important in endgame
            score += pst_score * (1.2 - 0.4 * endgame_factor)      # PST less important in endgame
            score += mvv_lva_score * (1.0 - 0.3 * endgame_factor)  # Tactics less important in endgame

        # 5. Secondary scoring components
        if self.use_secondary_scoring:
            castling_score = self._evaluate_castling(board)
            captures_score = self._evaluate_piece_captures(board)
            
            # Apply phase-based weights
            score += castling_score * (1.0 - endgame_factor)        # Castling irrelevant in endgame
            score += captures_score * (1.0 + 0.5 * endgame_factor)  # Captures more important in endgame
            
        # 6. Apply tempo and risk adjustments
        score += tempo_score * (1.0 + 0.5 * endgame_factor)  # Tempo more important in endgame
        score += risk_score * (-1.0 if material_score > 0 else 0.5)  # Avoid risks when ahead
        
        # 7. Zugzwang handling
        if zugzwang_risk < -0.5 and endgame_factor > 0.6:
            score *= 0.8  # Significant penalty for zugzwang-prone positions in endgame
        
        # Update position history
        self.tempo.update_position_score(board, score)
        
        # Update score dataset
        self.score_dataset.update({
            'fen': board.fen(),
            'game_phase': phase,
            'endgame_factor': endgame_factor,
            'tempo_score': tempo_score,
            'risk_score': risk_score,
            'zugzwang_risk': zugzwang_risk,
            'total_score': score
        })
        
        return score if board.turn == color else -score
    
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
        applying dynamic ruleset settings and endgame awareness based on configuration.
        This is the main public method for this class.
        """
        score = 0.0
        
        # Update scoring dictionary with minimal position information
        self.score_dataset['fen'] = board.fen()
        self.color_name = "White" if color == chess.WHITE else "Black"
        self.score_dataset['color'] = self.color_name
        
        # Primary Scoring Components (always enabled for basic functionality)
        if self.use_primary_scoring:
            # CHECKMATE THREATS - Most critical factor
            checkmate_threats_score = self.rules_manager.checkmate_threats(board, color) or 0.0
            score += checkmate_threats_score
            self.score_dataset['checkmate_threats_score'] = checkmate_threats_score

            # MATERIAL COUNT (Basic piece counting)
            material_count = self.rules_manager.material_count(board, color) or 0.0
            score += material_count
            self.score_dataset['material_count'] = material_count

            # MATERIAL SCORE (Weighted piece values)
            material_score = self.rules_manager.material_score(board, color) or 0.0
            score += material_score
            self.score_dataset['material_score'] = material_score
        else:
            # Fallback to basic material count if primary scoring is disabled
            basic_material = self.rules_manager.material_count(board, color) or 0.0
            score += basic_material * self.fallback_modifier  # Apply modifier to make scores comparable
            self.score_dataset['material_count'] = basic_material
        
        # Stop Check - Return if we're in a very clear position (queen + multiple pieces advantage)
        if (score > 1500) and not board.is_check():  # Big material advantage and not in check
            return score
        
        # GAME PHASE (Affects both primary and secondary scoring)
        if self.use_game_phase:
            self._calculate_game_phase(board)
        
        # Secondary Scoring Components (optional advanced evaluation)
        if self.use_secondary_scoring:
            # PST SCORE
            pst_score = self.rules_manager.pst_score(board, color, self.endgame_factor) or 0.0
            score += pst_score
            self.score_dataset['pst_score'] = pst_score
            
            # MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
            if self.use_mvv_lva:
                # Calculate MVV-LVA based capture scores
                piece_captures_score = 0.0
                for move in board.legal_moves:
                    if board.is_capture(move):
                        mvv_lva_score = self.mvv_lva.calculate_mvv_lva_score(move, board)
                        piece_captures_score += mvv_lva_score * (1.0 if board.turn == color else -1.0)
                score += piece_captures_score
                self.score_dataset['piece_captures'] = piece_captures_score

            # PIECE DEVELOPMENT AND CENTER CONTROL
            development_score = self._evaluate_piece_development(board) if color == chess.WHITE else -self._evaluate_piece_development(board)
            score += development_score
            self.score_dataset['development_score'] = development_score

            # PAWN STRUCTURE AND ADVANCEMENT
            pawn_structure_score = self.rules_manager.evaluate_pawn_structure(board, color) if hasattr(self.rules_manager, 'evaluate_pawn_structure') else 0.0
            score += pawn_structure_score
            self.score_dataset['pawn_structure_score'] = pawn_structure_score

            # MOBILITY (piece movement possibilities)
            mobility_score = self.rules_manager.evaluate_mobility(board, color) if hasattr(self.rules_manager, 'evaluate_mobility') else 0.0
            score += mobility_score
            self.score_dataset['mobility_score'] = mobility_score

            # KING SAFETY
            king_safety_score = self.rules_manager.evaluate_king_safety(board, color) if hasattr(self.rules_manager, 'evaluate_king_safety') else 0.0
            score += king_safety_score
            self.score_dataset['king_safety_score'] = king_safety_score

        return score
    
    def _calculate_material_score(self, board: chess.Board) -> float:
        """Calculate material balance score using piece values"""
        score = 0.0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.pst.get_piece_value(piece)
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
                    
        return score
        
    def _calculate_pst_score(self, board: chess.Board) -> float:
        """Calculate piece-square table score for positional evaluation"""
        score = 0.0
        endgame_factor = self._calculate_endgame_factor(board)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                pst_value = self.pst.get_pst_value(piece, square, piece.color, endgame_factor)
                if piece.color == chess.WHITE:
                    score += pst_value
                else:
                    score -= pst_value
                    
        return score
        
    def _calculate_endgame_factor(self, board: chess.Board) -> float:
        """Calculate endgame factor between 0.0 (middlegame) and 1.0 (endgame)
        Based on material remaining on the board."""
        # Count material excluding kings
        total_material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                total_material += self.pst.get_piece_value(piece)
                
        # Max material = 2 queens + 4 rooks + 4 bishops + 4 knights + 16 pawns
        max_material = 2 * 900 + 4 * 500 + 4 * 330 + 4 * 320 + 16 * 100
        
        # Convert to factor 0.0 to 1.0
        return 1.0 - (total_material / max_material)
        
    def _evaluate_castling_status(self, board: chess.Board) -> float:
        """Evaluate castling rights and king safety"""
        score = 0.0
        castling_value = 50  # Base value for having castling rights
        
        # Evaluate white's castling status
        if board.has_kingside_castling_rights(chess.WHITE):
            score += castling_value
        if board.has_queenside_castling_rights(chess.WHITE):
            score += castling_value
            
        # Evaluate black's castling status
        if board.has_kingside_castling_rights(chess.BLACK):
            score -= castling_value
        if board.has_queenside_castling_rights(chess.BLACK):
            score -= castling_value
            
        return score
        
    def _evaluate_mobility(self, board: chess.Board) -> float:
        """Evaluate piece mobility (number of legal moves)"""
        original_turn = board.turn
        mobility_value = 10  # Value per legal move
        
        # Count white's mobility
        board.turn = chess.WHITE
        white_mobility = len(list(board.legal_moves))
        
        # Count black's mobility
        board.turn = chess.BLACK
        black_mobility = len(list(board.legal_moves))
        
        # Restore original turn
        board.turn = original_turn
        
        return (white_mobility - black_mobility) * mobility_value

    def _calculate_primary_score(self, board: chess.Board) -> float:
        """Calculate primary score including material, PST, and MVV-LVA."""
        if not self.use_primary_scoring:
            return 0.0
            
        # Get material score
        material_score = self._evaluate_material(board) * self.ruleset.get('material_weight', 1.0)
        
        # Get piece-square table score
        pst_score = self.pst.evaluate_board_position(board) * self.ruleset.get('pst_weight', 0.5)
        
        # Get MVV-LVA based tactical score
        tactical_score = 0.0
        if self.use_mvv_lva:
            for move in board.legal_moves:
                # Get MVV-LVA capture score
                capture_score = self.mvv_lva.calculate_mvv_lva_score(move, board)
                tactical_score += capture_score
                
                # Add tactical pattern evaluation
                if capture_score > 0 or board.gives_check(move):
                    pattern_score = self.mvv_lva.evaluate_tactical_pattern(board, move)
                    tactical_score += pattern_score
            
            tactical_score *= self.ruleset.get('tactical_weight', 0.3)
        
        # Update score dataset
        self.score_dataset.update({
            'material_score': material_score,
            'pst_score': pst_score,
            'piece_captures_score': tactical_score
        })
        
        return material_score + pst_score + tactical_score

    def _evaluate_piece_square_tables(self, board: chess.Board) -> float:
        """Evaluate piece positions using piece-square tables."""
        score = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
                
            # Get PST value for the piece at its current position
            score += self.pst.get_piece_square_value(piece.piece_type, square, piece.color == chess.WHITE)
            
        return score
            
    def _evaluate_castling(self, board: chess.Board) -> float:
        """Evaluate castling status and king safety."""
        score = 0.0
        castling_bonus = 0.3  # Base bonus for having castling rights
        
        # Check white's castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            score += castling_bonus
        if board.has_queenside_castling_rights(chess.WHITE):
            score += castling_bonus
            
        # Check black's castling rights
        if board.has_kingside_castling_rights(chess.BLACK):
            score -= castling_bonus
        if board.has_queenside_castling_rights(chess.BLACK):
            score -= castling_bonus
            
        # Additional bonus if already castled
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        if white_king_square is not None and white_king_square in (chess.G1, chess.C1):
            score += 0.5  # Bonus for completed castling
        if black_king_square is not None and black_king_square in (chess.G8, chess.C8):
            score -= 0.5  # Bonus for completed castling
            
        return score
        
    def _evaluate_piece_captures(self, board: chess.Board) -> float:
        """Evaluate piece capture potential using MVV-LVA."""
        score = 0.0
        
        # Calculate capture scores
        for move in board.legal_moves:
            if board.is_capture(move):
                # Get MVV-LVA tactical score
                mvv_lva_score = self.mvv_lva.calculate_mvv_lva_score(move, board)
                score += mvv_lva_score
                
                # Add tactical pattern evaluation
                pattern_score = self.mvv_lva.evaluate_tactical_pattern(board, move)
                score += pattern_score
        
        self.score_dataset['piece_captures_score'] = score
        return score
        
    def _evaluate_tactical_patterns(self, board: chess.Board) -> float:
        """Evaluate tactical patterns and threats."""
        score = 0.0
        
        # Include MVV-LVA based tactical evaluation
        capture_score = self._evaluate_piece_captures(board)
        score += capture_score * 0.5  # Weight for tactical evaluation
        
        # Store scores
        self.score_dataset['tactical_pattern_score'] = score
        return score

    def score_move(self, move: chess.Move, board: chess.Board) -> int:
        """Score a move based on various scoring factors.
        This method is used by the move ordering module to prioritize moves.
        
        Args:
            move: The move to score
            board: The current board position
            
        Returns:
            int: A score representing the estimated value of the move
        """
        # Use MVV-LVA for captures
        if board.is_capture(move):
            return self.mvv_lva.score_move(move, board)
        
        # Check for check
        board.push(move)
        gives_check = board.is_check()
        board.pop()
        
        if gives_check:
            return 50  # Bonus for moves that give check
        
        # Evaluate positional improvement
        score = 0
        
        # Piece development in opening
        if self.game_phase == 'opening':
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                # Bonus for developing pieces in opening
                if (piece.color == chess.WHITE and chess.square_rank(move.from_square) == 1 and 
                    chess.square_rank(move.to_square) > 1):
                    score += 30
                elif (piece.color == chess.BLACK and chess.square_rank(move.from_square) == 6 and 
                      chess.square_rank(move.to_square) < 6):
                    score += 30
        
        # Bonus for moves to the center
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        if move.to_square in center_squares:
            score += 20
        
        return score