# v7p3r_scoring_calculation.py

""" V7P3R Scoring Calculation Module
This module implements the scoring calculation for the V7P3R Chess Engine.
It provides various scoring functions for evaluation, quiescence, and move ordering
"""

import chess
import logging
import os
from piece_square_tables import PieceSquareTables # Need this for PST evaluation

class V7P3RScoringCalculation:
    """
    Encapsulates all evaluation scoring functions for the V7P3R Chess Engine.
    Allows for dynamic selection of evaluation rulesets.
    """
    def __init__(self, piece_values):
        self.piece_values = piece_values
        self.pst = PieceSquareTables()

    # Renamed from _calculate_score to calculate_score to be the public API
    def calculate_score(self, board: chess.Board, color: chess.Color, endgame_factor: float = 0.0) -> float:
        """
        Calculates the position evaluation score for a given board and color,
        applying dynamic ruleset settings and endgame awareness.
        This is the main public method for this class.
        """
        score = 0.0

        # Critical scoring components
        score += (self._checkmate_threats(board, color) or 0.0)
        score += (self._king_safety(board, color) or 0.0)
        score += (self._king_threat(board, color) or 0.0)
        score += (self._queen_safety(board, color) or 0.0)  # Prevent queen blunders
        score += (self._draw_scenarios(board) or 0.0)

        # Material and piece-square table evaluation
        score += self._material_score(board, color)
        pst_board_score = self.pst.evaluate_board_position(board, endgame_factor) # This is likely from White's perspective
        if color == chess.BLACK:
            pst_board_score = -pst_board_score # Adjust if PST is always White-centric
        score += pst_board_score

        # Piece coordination and control
        score += (self._piece_coordination(board, color) or 0.0)
        score += (self._center_control(board, color) or 0.0)
        score += (self._pawn_structure(board, color) or 0.0)
        score += (self._passed_pawns(board, color) or 0.0)
        score += (self._pawn_majority(board, color) or 0.0)
        score += (self._bishop_pair(board, color) or 0.0)
        score += (self._knight_pair(board, color) or 0.0)
        score += (self._bishop_vision(board, color) or 0.0)
        score += (self._rook_coordination(board, color) or 0.0)
        score += (self._castling_evaluation(board, color) or 0.0)

        # Piece development and mobility
        score += (self._piece_activity(board, color) or 0.0)
        score += (self._improved_minor_piece_activity(board, color) or 0.0)
        score += (self._mobility_score(board, color) or 0.0)
        score += (self._undeveloped_pieces(board, color) or 0.0)

        # Tactical and strategic considerations
        score += (self._tactical_evaluation(board, color) or 0.0)
        score += (self._tempo_bonus(board, color) or 0.0)
        score += (self._special_moves(board, color) or 0.0) # Pass color
        score += (self._open_files(board, color) or 0.0)
        score += (self._stalemate(board) or 0.0)

        return score

    # ==========================================
    # ========= RULE SCORING FUNCTIONS =========
    # These functions are now methods of V7P3RScoringCalculation
    # and access their rule values via self._get_rule_value()

    def _checkmate_threats(self, board: chess.Board, color: chess.Color) -> float:
        """
        Assess if 'color' can deliver a checkmate on their next move.
        Only consider legal moves for 'color' without mutating the original board's turn.
        """
        score = 0.0

        # Use a copy so we don't mutate the original board
        board_copy = board.copy()
        board_copy.turn = color
        for move in board_copy.pseudo_legal_moves:
            # Only consider moves that are legal (pseudo-legal may include illegal under check)
            if not board_copy.is_legal(move):
                continue
            board_copy.push(move)
            if board_copy.is_checkmate():
                score += 9999999999.0
                board_copy.pop()
                if board_copy.turn != color: # If the checkmate is on the *opponent's* king
                    return score # Return immediately
            else:
                board_copy.pop()
        return score

    def _draw_scenarios(self, board: chess.Board) -> float:
        score = 0.0
        if board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition() or board.is_repetition(count=2):
            score += -9999999999.0
        return score

    def _stalemate(self, board: chess.Board) -> float:
        """Check if the position is a stalemate"""
        if board.is_stalemate():
            return -9999999999
        return 0.0

    def _material_score(self, board: chess.Board, color: chess.Color) -> float:
        """Simple material count for given color"""
        score = 0.0
        for piece_type, value in self.piece_values.items():
            score += len(board.pieces(piece_type, color)) * value
        # Apply material weight from ruleset
        return score
    
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
            score += safe_moves

        for square in board.pieces(chess.BISHOP, color):
            safe_moves = 0
            for target in board.attacks(square):
                if not self._is_attacked_by_pawn(board, target, not color):
                    safe_moves += 1
            score += safe_moves

        return score

    def _tempo_bonus(self, board: chess.Board, color: chess.Color) -> float:
        """If it's the player's turn and the game is still ongoing, give a small tempo bonus"""
        # The 'current_player' attribute is from EvaluationEngine, need to pass it or infer.
        # This method is part of scoring specific 'color'. So, if it's 'color's turn.
        if board.turn == color and not board.is_game_over() and board.is_valid():
            return 0.1
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
                score += 0.5
        return score

    def _piece_activity(self, board: chess.Board, color: chess.Color) -> float:
        """Mobility and attack patterns"""
        score = 0.0

        for square in board.pieces(chess.KNIGHT, color):
            score += len(list(board.attacks(square))) * 1.0

        for square in board.pieces(chess.BISHOP, color):
            score += len(list(board.attacks(square))) * 1.0

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
                        score += 0.1

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
                score += -9.0
            else: # If it *is* 'color's turn, and board is in check, then 'color' just gave check
                score += 9.0
        return score

    def _queen_safety(self, board: chess.Board, color: chess.Color) -> float:
        """
        Evaluate queen safety to prevent queen blunders.
        Applies heavy penalty if our queen is under attack and can be captured.
        This overrides other tactical considerations to prevent losing the queen.
        """
        score = 0.0
        
        # Find our queen
        queen_square = None
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.QUEEN and piece.color == color:
                queen_square = square
                break
        
        if queen_square is None:
            return 0.0  # No queen on board
        
        # Check if our queen is under attack
        if self._is_square_attacked_by_opponent(board, queen_square, color):
            # Heavy penalty for exposed queen - this should override other considerations
            penalty = -900.0  # Almost queen value to strongly discourage queen exposure
            
            # Check if queen can move to safety
            queen_moves = 0
            safe_moves = 0
            
            # Generate queen moves to see if any lead to safety
            temp_board = board.copy()
            for move in temp_board.legal_moves:
                if move.from_square == queen_square:
                    queen_moves += 1
                    
                    # Test if destination is safe
                    temp_board.push(move)
                    if not self._is_square_attacked_by_opponent(temp_board, move.to_square, color):
                        safe_moves += 1
                    temp_board.pop()
            
            # If queen has no safe moves, apply maximum penalty
            if queen_moves > 0 and safe_moves == 0:
                penalty = -1200.0  # Even heavier penalty for trapped queen
            
            score += penalty
            
        return score

    def _is_square_attacked_by_opponent(self, board: chess.Board, square: chess.Square, defending_color: chess.Color) -> bool:
        """
        Check if the given square is attacked by the opponent of defending_color.
        """
        opponent_color = not defending_color
        
        # Check all opponent pieces for attacks on this square
        for attacker_square in chess.SQUARES:
            attacker_piece = board.piece_at(attacker_square)
            if attacker_piece and attacker_piece.color == opponent_color:
                # Check if this piece attacks the target square
                if self._piece_attacks_square(board, attacker_square, square, attacker_piece):
                    return True
        
        return False
    
    def _piece_attacks_square(self, board: chess.Board, from_square: chess.Square, to_square: chess.Square, piece: chess.Piece) -> bool:
        """
        Check if a piece at from_square can attack to_square.
        """
        # Create a hypothetical capture move to test if it's legal
        try:
            # Check if there's a legal move from from_square to to_square
            for move in board.legal_moves:
                if move.from_square == from_square and move.to_square == to_square:
                    return True
            
            # Also check pseudo-legal moves for pieces that might be pinned
            temp_board = board.copy()
            try:
                move = chess.Move(from_square, to_square)
                if move in temp_board.pseudo_legal_moves:
                    # For pawns, check diagonal attacks specifically
                    if piece.piece_type == chess.PAWN:
                        file_diff = abs(chess.square_file(to_square) - chess.square_file(from_square))
                        rank_diff = chess.square_rank(to_square) - chess.square_rank(from_square)
                        
                        if piece.color == chess.WHITE:
                            return file_diff == 1 and rank_diff == 1
                        else:
                            return file_diff == 1 and rank_diff == -1
                    
                    return True
            except:
                pass
                
        except:
            pass
            
        return False

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
            score += undeveloped_count

        return score

    def _mobility_score(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate mobility of pieces"""
        score = 0.0
        
        # Iterate over all pieces of the given color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color and piece.piece_type != chess.KING: # Exclude king from general mobility
                score += len(list(board.attacks(square)))

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
                            score += 1.0
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
                score += 1.0
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
                    score += 1.0

        opponent_color = not color

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == opponent_color:
                # Check if it's attacked by 'color' and not defended by 'opponent_color' (i.e., not attacked by opponent_color)
                if board.is_attacked_by(color, square) and not board.is_attacked_by(opponent_color, square):
                    score += 0.5
            elif piece and piece.color == color:
                # Penalty for 'color' having pieces attacked by opponent_color and not defended by 'color'
                if board.is_attacked_by(opponent_color, square) and not board.is_attacked_by(color, square):
                    score -= 0.5

        return score

    def _castling_evaluation(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate castling rights and opportunities"""
        score = 0.0

        # Check if castled - more robust check considering king's final position
        king_sq = board.king(color)
        if king_sq: # Ensure king exists
            if color == chess.WHITE:
                if king_sq == chess.G1: # Kingside castled
                    score += 1.0
                elif king_sq == chess.C1: # Queenside castled
                    score += 1.0
            else: # Black
                if king_sq == chess.G8: # Kingside castled
                    score += 1.0
                elif king_sq == chess.C8: # Queenside castled
                    score += 1.0

        # Penalty if castling rights lost and not yet castled
        initial_king_square = chess.E1 if color == chess.WHITE else chess.E8
        if not board.has_castling_rights(color) and king_sq == initial_king_square:
            score -= 1.0

        # Bonus if still has kingside or queenside castling rights
        if board.has_kingside_castling_rights(color) and board.has_queenside_castling_rights(color):
            score += 1.0
        elif board.has_kingside_castling_rights(color) or board.has_queenside_castling_rights(color):
            score += 0.5

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
                    score += 0.5
        return score
    
    def _pawn_structure(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate pawn structure (doubled, isolated pawns)"""
        score = 0.0
        
        # Count doubled pawns
        for file in range(8):
            pawns_on_file = [s for s in board.pieces(chess.PAWN, color) if chess.square_file(s) == file]
            if len(pawns_on_file) > 1:
                score += (len(pawns_on_file) - 1)
        
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
                score -= 0.5

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
                score += 0.25 # Half bonus for kingside
            if white_pawns_queenside > black_pawns_queenside:
                score += 0.25 # Half bonus for queenside

            if white_pawns_kingside < black_pawns_kingside:
                score -= 0.25
            if white_pawns_queenside < black_pawns_queenside:
                score -= 0.25
        else: # Black
            if black_pawns_kingside > white_pawns_kingside:
                score += 0.25
            if black_pawns_queenside > white_pawns_queenside:
                score += 0.25
            if black_pawns_kingside < white_pawns_kingside:
                score -= 0.25
            if black_pawns_queenside < white_pawns_queenside:
                score -= 0.25

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
                score += 0.5
        return score

    def _knight_pair(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate knight pair bonus"""
        score = 0.0
        knights = list(board.pieces(chess.KNIGHT, color))
        if len(knights) >= 2:
            score += 0.5
        return score

    def _bishop_pair(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate bishop pair bonus"""
        score = 0.0
        bishops = list(board.pieces(chess.BISHOP, color))
        if len(bishops) >= 2:
            score += 0.5
        return score

    def _bishop_vision(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate bishop vision bonus based on squares attacked."""
        score = 0.0
        for sq in board.pieces(chess.BISHOP, color):
            attacks = board.attacks(sq)
            # Bonus for having more attacked squares (i.e., good vision)
            if len(list(attacks)) > 5: # Bishops generally attack 7-13 squares, adjust threshold as needed
                score += 0.1
        return score

    def _rook_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate bonus for rook pairs on same file/rank and 7th rank."""
        score = 0.0
        rooks = list(board.pieces(chess.ROOK, color))

        for i in range(len(rooks)):
            for j in range(i+1, len(rooks)):
                sq1, sq2 = rooks[i], rooks[j]
                if chess.square_file(sq1) == chess.square_file(sq2):
                    score += 0.5
                if chess.square_rank(sq1) == chess.square_rank(sq2):
                    score += 0.5

                # Rook on 7th rank bonus (critical for attacking pawns)
                # Check for white on rank 7 (index 6) or black on rank 2 (index 1)
                if (color == chess.WHITE and (chess.square_rank(sq1) == 6 or chess.square_rank(sq2) == 6)) or \
                   (color == chess.BLACK and (chess.square_rank(sq1) == 1 or chess.square_rank(sq2) == 1)):
                    score += 0.5
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
                score += 0.5
            elif not is_file_open and not has_own_pawn_on_file and has_opponent_pawn_on_file: # Semi-open file for 'color'
                score += 0.25 # Half bonus for semi-open (tuneable)

            # Bonus if a rook is on an open or semi-open file
            if any(board.piece_at(chess.square(file, r)) == chess.Piece(chess.ROOK, color) for r in range(8)):
                if is_file_open or (not is_file_open and not has_own_pawn_on_file): # If open or semi-open
                    score += 0.5
            
            # Exposed king penalty if king is on an open/semi-open file
            king_sq = board.king(color)
            if king_sq is not None and chess.square_file(king_sq) == file:
                if is_file_open or (not is_file_open and not has_own_pawn_on_file): # If king is on an open/semi-open file
                    score -= 5.0

        return score