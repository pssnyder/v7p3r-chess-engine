# v7p3r_scoring_calculation.py

""" V7P3R Scoring Calculation Module
This module implements the scoring calculation for the V7P3R Chess Engine.
It provides various scoring functions for evaluation, quiescence, and move ordering
"""

import chess
import logging
import os

class V7P3RScoringCalculation:
    """
    Encapsulates all evaluation scoring functions for the V7P3R Chess Engine.
    Allows for dynamic selection of evaluation rulesets.
    """
    def __init__(self, piece_values):
        self.piece_values = piece_values

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
        score += (self._material_score(board, color) or 0.0)

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

    def _improved_minor_piece_activity(self, board: chess.Board, color: chess.Color) -> float:
        """
        Enhanced mobility calculation emphasizing centralization and tactical positioning.
        """
        score = 0.0

        # Enhanced Knight Activity
        for knight_square in board.pieces(chess.KNIGHT, color):
            knight_score = self._calculate_knight_activity(board, knight_square, color)
            score += knight_score

        # Enhanced Bishop Activity (handled by _bishop_vision now)
        # This function now focuses on knight-specific improvements
        
        return score

    def _calculate_knight_activity(self, board: chess.Board, knight_square: chess.Square, color: chess.Color) -> float:
        """Calculate comprehensive knight activity score."""
        score = 0.0
        
        # Centralization bonus - knights are most effective in center
        centralization_score = self._knight_centralization_bonus(knight_square)
        score += centralization_score
        
        # Edge penalty - knights on edges are less effective
        edge_penalty = self._knight_edge_penalty(knight_square)
        score += edge_penalty
        
        # Safe mobility - count safe squares the knight can reach
        safe_mobility = self._knight_safe_mobility(board, knight_square, color)
        score += safe_mobility
        
        # Outpost bonus - defended knights in enemy territory
        outpost_bonus = self._knight_outpost_bonus(board, knight_square, color)
        score += outpost_bonus
        
        # Attack quality - bonus for attacking high-value targets
        attack_quality = self._knight_attack_quality(board, knight_square, color)
        score += attack_quality
        
        return score
    
    def _knight_centralization_bonus(self, knight_square: chess.Square) -> float:
        """Heavy bonus for knights in central squares."""
        file = chess.square_file(knight_square)
        rank = chess.square_rank(knight_square)
        
        # Perfect central squares (d4, e4, d5, e5)
        if (file in [3, 4] and rank in [3, 4]):
            return 1.2  # Excellent central placement
            
        # Good central squares (c3-f3, c4-f4, c5-f5, c6-f6)
        elif (file in [2, 3, 4, 5] and rank in [2, 3, 4, 5]):
            return 0.6  # Good central placement
            
        # Decent central area
        elif (file in [1, 2, 3, 4, 5, 6] and rank in [1, 2, 3, 4, 5, 6]):
            return 0.2  # Decent placement
            
        return 0.0
    
    def _knight_edge_penalty(self, knight_square: chess.Square) -> float:
        """Penalty for knights on board edges."""
        file = chess.square_file(knight_square)
        rank = chess.square_rank(knight_square)
        
        penalty = 0.0
        
        # Heavy penalty for corner squares
        if (file in [0, 7] and rank in [0, 7]):
            penalty -= 1.0  # Corners are terrible for knights
            
        # Moderate penalty for edge squares
        elif (file in [0, 7] or rank in [0, 7]):
            penalty -= 0.5  # Edges limit knight mobility
            
        return penalty
    
    def _knight_safe_mobility(self, board: chess.Board, knight_square: chess.Square, color: chess.Color) -> float:
        """Count safe squares the knight can move to."""
        safe_moves = 0
        total_moves = 0
        
        for target in board.attacks(knight_square):
            total_moves += 1
            # Check if target square is safe (not attacked by enemy pawns/pieces)
            if not self._is_square_under_attack(board, target, color):
                safe_moves += 1
                
        # Bonus based on mobility ratio
        if total_moves > 0:
            mobility_ratio = safe_moves / total_moves
            return mobility_ratio * 0.8  # Up to 0.8 bonus for full safe mobility
            
        return 0.0
    
    def _knight_outpost_bonus(self, board: chess.Board, knight_square: chess.Square, color: chess.Color) -> float:
        """Bonus for knights in outpost positions (defended in enemy territory)."""
        rank = chess.square_rank(knight_square)
        
        # Define enemy territory
        enemy_territory = rank >= 4 if color == chess.WHITE else rank <= 3
        
        if not enemy_territory:
            return 0.0
            
        # Check if knight is defended by our pawns
        if self._is_knight_defended_by_pawn(board, knight_square, color):
            # Extra bonus for well-placed outposts
            if rank >= 5 if color == chess.WHITE else rank <= 2:
                return 1.0  # Strong outpost in deep enemy territory
            else:
                return 0.6  # Good outpost
                
        return 0.0
    
    def _is_knight_defended_by_pawn(self, board: chess.Board, knight_square: chess.Square, color: chess.Color) -> bool:
        """Check if knight is defended by friendly pawn."""
        # Check diagonal squares where defending pawns would be
        file = chess.square_file(knight_square)
        rank = chess.square_rank(knight_square)
        
        if color == chess.WHITE:
            # White pawns defend from rank below
            pawn_ranks = [rank - 1] if rank > 0 else []
        else:
            # Black pawns defend from rank above  
            pawn_ranks = [rank + 1] if rank < 7 else []
            
        for pawn_rank in pawn_ranks:
            for pawn_file in [file - 1, file + 1]:
                if 0 <= pawn_file <= 7:
                    pawn_square = chess.square(pawn_file, pawn_rank)
                    piece = board.piece_at(pawn_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        return True
                        
        return False
    
    def _knight_attack_quality(self, board: chess.Board, knight_square: chess.Square, color: chess.Color) -> float:
        """Bonus for knights attacking high-value targets."""
        score = 0.0
        
        for target in board.attacks(knight_square):
            target_piece = board.piece_at(target)
            if target_piece and target_piece.color != color:
                # Bonus based on target piece value
                piece_value = self.piece_values.get(target_piece.piece_type, 0)
                if piece_value >= 5.0:  # Rook or Queen
                    score += 0.4
                elif piece_value >= 3.0:  # Bishop or Knight
                    score += 0.2
                elif piece_value >= 1.0:  # Pawn
                    score += 0.1
                    
        return score
    
    def _is_square_under_attack(self, board: chess.Board, square: chess.Square, defending_color: chess.Color) -> bool:
        """Check if square is under attack by enemy."""
        return board.is_attacked_by(not defending_color, square)

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
        """Enhanced castling evaluation considering development and king safety."""
        score = 0.0

        king_square = board.king(color)
        if not king_square:
            return 0.0
            
        # Check if already castled
        castled_bonus = self._check_if_castled(king_square, color)
        score += castled_bonus
        
        if castled_bonus > 0:
            # Already castled - no need for further castling evaluation
            return score
            
        # Evaluate castling opportunities and urgency
        development_factor = self._calculate_development_stage(board, color)
        king_danger = self._assess_king_danger(board, color)
        
        # Castling rights evaluation
        castling_rights_bonus = self._evaluate_castling_rights(board, color, development_factor, king_danger)
        score += castling_rights_bonus
        
        # Penalty for delaying castling too long
        delay_penalty = self._castling_delay_penalty(board, color, development_factor)
        score += delay_penalty
        
        return score
    
    def _check_if_castled(self, king_square: chess.Square, color: chess.Color) -> float:
        """Check if king has castled and give appropriate bonus."""
        if color == chess.WHITE:
            if king_square == chess.G1:  # Kingside castled
                return 1.5  # Good king safety
            elif king_square == chess.C1:  # Queenside castled
                return 1.2  # Decent king safety, but less secure
        else:  # Black
            if king_square == chess.G8:  # Kingside castled
                return 1.5
            elif king_square == chess.C8:  # Queenside castled
                return 1.2
                
        return 0.0
    
    def _calculate_development_stage(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate how developed the pieces are (0.0 = early game, 1.0 = fully developed)."""
        developed_pieces = 0
        total_minor_pieces = 0
        
        # Count developed minor pieces
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for square in board.pieces(piece_type, color):
                total_minor_pieces += 1
                
                # Check if piece has moved from starting position
                starting_rank = 0 if color == chess.WHITE else 7
                if chess.square_rank(square) != starting_rank:
                    developed_pieces += 1
                    
        # Check queen development (should castle before major queen development)
        queen_squares = list(board.pieces(chess.QUEEN, color))
        queen_developed = False
        if queen_squares:
            queen_square = queen_squares[0]
            starting_square = chess.D1 if color == chess.WHITE else chess.D8
            if queen_square != starting_square:
                queen_developed = True
                
        if total_minor_pieces > 0:
            minor_development = developed_pieces / total_minor_pieces
        else:
            minor_development = 1.0
            
        # Factor in queen development (early queen development reduces castling urgency)
        if queen_developed:
            return min(1.0, minor_development + 0.3)
        else:
            return minor_development
    
    def _assess_king_danger(self, board: chess.Board, color: chess.Color) -> float:
        """Assess how much danger the king is in (0.0 = safe, 1.0 = high danger)."""
        danger_score = 0.0
        king_square = board.king(color)
        
        if not king_square:
            return 0.0
            
        # Check if king is in check
        if board.is_check():
            danger_score += 0.4
            
        # Count enemy pieces attacking squares near king
        king_area_attacks = 0
        for square in self._get_king_area_squares(king_square):
            if board.is_attacked_by(not color, square):
                king_area_attacks += 1
                
        # Normalize based on number of squares around king (max 9)
        danger_score += (king_area_attacks / 9.0) * 0.6
        
        return min(1.0, danger_score)
    
    def _get_king_area_squares(self, king_square: chess.Square) -> list:
        """Get squares in the king's immediate area."""
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        area_squares = []
        for file_offset in [-1, 0, 1]:
            for rank_offset in [-1, 0, 1]:
                new_file = king_file + file_offset
                new_rank = king_rank + rank_offset
                
                if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
                    area_squares.append(chess.square(new_file, new_rank))
                    
        return area_squares
    
    def _evaluate_castling_rights(self, board: chess.Board, color: chess.Color, development_factor: float, king_danger: float) -> float:
        """Evaluate castling rights based on development and danger."""
        score = 0.0
        
        has_kingside = board.has_kingside_castling_rights(color)
        has_queenside = board.has_queenside_castling_rights(color)
        
        if not has_kingside and not has_queenside:
            # Lost all castling rights - penalty
            return -1.5
            
        # Base bonus for having castling rights
        if has_kingside and has_queenside:
            score += 1.0  # Both rights preserved
        elif has_kingside or has_queenside:
            score += 0.6  # One right preserved
            
        # Urgency multiplier based on development and danger
        urgency_multiplier = development_factor + king_danger
        score *= (1.0 + urgency_multiplier)
        
        return score
    
    def _castling_delay_penalty(self, board: chess.Board, color: chess.Color, development_factor: float) -> float:
        """Penalty for not castling when development is advanced."""
        king_square = board.king(color)
        initial_square = chess.E1 if color == chess.WHITE else chess.E8
        
        # Only penalize if king is still on initial square
        if king_square != initial_square:
            return 0.0
            
        # Penalty increases with development
        if development_factor >= 0.8:
            return -1.0  # Heavy penalty for not castling when fully developed
        elif development_factor >= 0.6:
            return -0.6  # Moderate penalty
        elif development_factor >= 0.4:
            return -0.3  # Small penalty
            
        return 0.0

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
        """Enhanced bishop activity evaluation focusing on diagonal control and positioning."""
        score = 0.0
        
        for bishop_square in board.pieces(chess.BISHOP, color):
            # Calculate diagonal ray lengths for this bishop
            diagonal_score = self._calculate_bishop_diagonal_strength(board, bishop_square)
            score += diagonal_score
            
            # Fianchetto position bonus
            fianchetto_bonus = self._bishop_fianchetto_bonus(bishop_square, color)
            score += fianchetto_bonus
            
            # Central diagonal control bonus
            central_diagonal_bonus = self._bishop_central_diagonal_bonus(bishop_square)
            score += central_diagonal_bonus
            
            # Opposition territory control
            enemy_territory_bonus = self._bishop_enemy_territory_control(board, bishop_square, color)
            score += enemy_territory_bonus
            
        return score

    def _calculate_bishop_diagonal_strength(self, board: chess.Board, bishop_square: chess.Square) -> float:
        """Calculate bishop strength based on diagonal ray length and mobility."""
        score = 0.0
        
        # Get all squares the bishop attacks
        attacked_squares = board.attacks(bishop_square)
        attack_count = len(attacked_squares)
        
        # Bonus for high mobility (more attacked squares)
        if attack_count >= 10:
            score += 0.8  # Excellent mobility
        elif attack_count >= 7:
            score += 0.5  # Good mobility  
        elif attack_count >= 4:
            score += 0.2  # Decent mobility
        # No bonus for low mobility bishops
        
        # Additional bonus for long diagonal control
        # Count the longest unobstructed diagonal ray
        max_ray_length = self._get_longest_bishop_ray(board, bishop_square)
        if max_ray_length >= 6:
            score += 0.4  # Long-range bishop
        elif max_ray_length >= 4:
            score += 0.2  # Medium-range bishop
            
        return score
    
    def _get_longest_bishop_ray(self, board: chess.Board, bishop_square: chess.Square) -> int:
        """Get the length of the longest unobstructed diagonal ray."""
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # Bishop diagonal directions
        max_length = 0
        
        for dx, dy in directions:
            length = 0
            file = chess.square_file(bishop_square)
            rank = chess.square_rank(bishop_square)
            
            # Follow the diagonal until we hit a piece or board edge
            while True:
                file += dx
                rank += dy
                
                # Check bounds
                if file < 0 or file > 7 or rank < 0 or rank > 7:
                    break
                    
                length += 1
                target_square = chess.square(file, rank)
                
                # If there's a piece, stop counting (but include this square)
                if board.piece_at(target_square):
                    break
                    
            max_length = max(max_length, length)
            
        return max_length
    
    def _bishop_fianchetto_bonus(self, bishop_square: chess.Square, color: chess.Color) -> float:
        """Bonus for bishops in fianchetto positions."""
        fianchetto_squares = {
            chess.WHITE: [chess.B2, chess.G2],
            chess.BLACK: [chess.B7, chess.G7]
        }
        
        if bishop_square in fianchetto_squares[color]:
            return 0.6  # Strong bonus for fianchetto bishops
        return 0.0
    
    def _bishop_central_diagonal_bonus(self, bishop_square: chess.Square) -> float:
        """Bonus for bishops on central diagonals."""
        # Main diagonals: a1-h8 and h1-a8
        file = chess.square_file(bishop_square)
        rank = chess.square_rank(bishop_square)
        
        # Check if on main diagonal (a1-h8: file == rank)
        if file == rank:
            return 0.3
            
        # Check if on anti-diagonal (h1-a8: file + rank == 7)  
        if file + rank == 7:
            return 0.3
            
        return 0.0
    
    def _bishop_enemy_territory_control(self, board: chess.Board, bishop_square: chess.Square, color: chess.Color) -> float:
        """Bonus for bishops controlling squares in enemy territory."""
        score = 0.0
        attacked_squares = board.attacks(bishop_square)
        
        # Define enemy territory
        enemy_ranks = [5, 6, 7] if color == chess.WHITE else [0, 1, 2]
        
        enemy_territory_attacks = 0
        for square in attacked_squares:
            if chess.square_rank(square) in enemy_ranks:
                enemy_territory_attacks += 1
                
        # Bonus for attacking enemy territory
        score += enemy_territory_attacks * 0.1
        
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