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

        # Advanced tactical pattern recognition
        score += (self._tactical_pattern_recognition(board, color) or 0.0)
        score += (self._enhanced_pawn_structure(board, color) or 0.0)
        score += (self._endgame_logic(board, color) or 0.0)

        # Piece coordination and control
        score += (self._piece_coordination(board, color) or 0.0)
        score += (self._center_control(board, color) or 0.0)
        # Note: _pawn_structure removed - replaced by _enhanced_pawn_structure in v5.4
        # Note: _passed_pawns removed - included in _enhanced_pawn_structure in v5.4
        score += (self._pawn_majority(board, color) or 0.0)
        score += (self._bishop_pair(board, color) or 0.0)
        score += (self._knight_pair(board, color) or 0.0)
        score += (self._bishop_vision(board, color) or 0.0)
        score += (self._rook_coordination(board, color) or 0.0)

        return score

    def calculate_score_optimized(self, board: chess.Board, color: chess.Color, endgame_factor: float = 0.0) -> float:
        """
        Optimized scoring with early exit thresholds for performance.
        Skips expensive calculations when score differentials are already decisive.
        """
        score = 0.0
        
        # Phase 1: Critical threats and material (always calculate)
        checkmate_score = self._checkmate_threats(board, color) or 0.0
        score += checkmate_score
        
        # Early exit 1: Checkmate threats found
        if abs(checkmate_score) > 5000:  # Mate threats are decisive
            return score
            
        king_safety = self._king_safety(board, color) or 0.0
        score += king_safety
        king_threat = self._king_threat(board, color) or 0.0  
        score += king_threat
        queen_safety = self._queen_safety(board, color) or 0.0
        score += queen_safety
        
        # Early exit 2: King is in severe danger
        if (king_safety + king_threat) < -300:  # King under severe attack
            return score + (self._material_score(board, color) or 0.0)
            
        material_score = self._material_score(board, color) or 0.0
        score += material_score
        draw_score = self._draw_scenarios(board) or 0.0
        score += draw_score
        
        # Early exit 3: Massive material advantage
        if abs(material_score) > 1500:  # 15+ point material advantage  
            return score
            
        # Phase 2: Tactical patterns (skip if already decisive)
        if abs(score) < 300:  # Within 3 points - tactics matter (much more aggressive)
            tactical_score = self._tactical_pattern_recognition(board, color) or 0.0
            score += tactical_score
            
            # Early exit 4: Tactical advantage found
            if abs(tactical_score) > 200:  # Even smaller tactical advantage triggers exit
                return score
                
        # Phase 3: Basic positional factors (only in very balanced positions)  
        if abs(score) < 150:  # Within 1.5 points - basic positional evaluation needed
            score += (self._piece_coordination(board, color) or 0.0)
            score += (self._center_control(board, color) or 0.0)
            
            # Early exit 5: After basic positional, still gap
            if abs(score) > 200:  # 2+ point gap after basic positional
                return score
                
        # Phase 4: Expensive calculations (only in extremely close games)
        if abs(score) < 50:  # Within 0.5 points - detailed evaluation needed
            score += (self._enhanced_pawn_structure(board, color) or 0.0)
            score += (self._pawn_majority(board, color) or 0.0)
            score += (self._bishop_pair(board, color) or 0.0) 
            score += (self._knight_pair(board, color) or 0.0)
            score += (self._bishop_vision(board, color) or 0.0)
            score += (self._rook_coordination(board, color) or 0.0)
            score += (self._castling_evaluation(board, color) or 0.0)
            
        # Phase 5: Development and mobility (only if still very close)
        if abs(score) < 200:  # Within 2 points - micro-optimizations needed
            score += (self._improved_minor_piece_activity(board, color) or 0.0)
            score += (self._mobility_score(board, color) or 0.0)
            score += (self._undeveloped_pieces(board, color) or 0.0)
            score += (self._tempo_bonus(board, color) or 0.0)
            score += (self._special_moves(board, color) or 0.0)
            score += (self._open_files(board, color) or 0.0)
            
        # Always include endgame logic and stalemate checks
        score += (self._endgame_logic(board, color) or 0.0)
        score += (self._stalemate(board) or 0.0)
        
        return score

        score += (self._rook_coordination(board, color) or 0.0)
        score += (self._castling_evaluation(board, color) or 0.0)

        # Piece development and mobility
        score += (self._improved_minor_piece_activity(board, color) or 0.0)
        score += (self._mobility_score(board, color) or 0.0)
        score += (self._undeveloped_pieces(board, color) or 0.0)

        # Tactical and strategic considerations
        # Note: _tactical_evaluation removed - replaced by _tactical_pattern_recognition in v5.4
        score += (self._tempo_bonus(board, color) or 0.0)
        score += (self._special_moves(board, color) or 0.0) # Pass color
        score += (self._open_files(board, color) or 0.0)
        score += (self._stalemate(board) or 0.0)

        # V5.4 Chess theoretical principles
        score += (self._opening_principles(board, color) or 0.0)
        score += (self._capture_guidelines(board, color) or 0.0)

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
        # Map string keys to chess piece type constants
        piece_type_map = {
            'pawn': chess.PAWN,
            'knight': chess.KNIGHT,
            'bishop': chess.BISHOP,
            'rook': chess.ROOK,
            'queen': chess.QUEEN,
            'king': chess.KING
        }
        
        for piece_name, value in self.piece_values.items():
            if piece_name in piece_type_map:
                piece_type = piece_type_map[piece_name]
                score += len(board.pieces(piece_type, color)) * value
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
        Optimized to use chess library's built-in attack detection.
        """
        # Use chess library's built-in efficient attack detection
        return board.is_attacked_by(not defending_color, square)
    
    def _piece_attacks_square(self, board: chess.Board, from_square: chess.Square, to_square: chess.Square, piece: chess.Piece) -> bool:
        """
        Check if a piece at from_square can attack to_square.
        Optimized version using piece-specific attack patterns.
        """
        # Use chess library's built-in attack detection which is much faster
        if piece.piece_type == chess.PAWN:
            # Pawn attacks diagonally
            file_diff = chess.square_file(to_square) - chess.square_file(from_square)
            rank_diff = chess.square_rank(to_square) - chess.square_rank(from_square)
            
            if piece.color == chess.WHITE:
                return abs(file_diff) == 1 and rank_diff == 1
            else:
                return abs(file_diff) == 1 and rank_diff == -1
                
        elif piece.piece_type == chess.KNIGHT:
            # Knight moves in L-shape
            file_diff = abs(chess.square_file(to_square) - chess.square_file(from_square))
            rank_diff = abs(chess.square_rank(to_square) - chess.square_rank(from_square))
            return (file_diff == 2 and rank_diff == 1) or (file_diff == 1 and rank_diff == 2)
            
        elif piece.piece_type == chess.BISHOP:
            # Bishop moves diagonally
            file_diff = abs(chess.square_file(to_square) - chess.square_file(from_square))
            rank_diff = abs(chess.square_rank(to_square) - chess.square_rank(from_square))
            if file_diff != rank_diff:
                return False
            # Check if path is clear
            return not self._is_path_blocked(board, from_square, to_square)
            
        elif piece.piece_type == chess.ROOK:
            # Rook moves in straight lines
            from_file, from_rank = chess.square_file(from_square), chess.square_rank(from_square)
            to_file, to_rank = chess.square_file(to_square), chess.square_rank(to_square)
            if from_file != to_file and from_rank != to_rank:
                return False
            # Check if path is clear
            return not self._is_path_blocked(board, from_square, to_square)
            
        elif piece.piece_type == chess.QUEEN:
            # Queen combines rook and bishop
            return (self._piece_attacks_square(board, from_square, to_square, chess.Piece(chess.ROOK, piece.color)) or
                   self._piece_attacks_square(board, from_square, to_square, chess.Piece(chess.BISHOP, piece.color)))
                   
        elif piece.piece_type == chess.KING:
            # King moves one square in any direction
            file_diff = abs(chess.square_file(to_square) - chess.square_file(from_square))
            rank_diff = abs(chess.square_rank(to_square) - chess.square_rank(from_square))
            return file_diff <= 1 and rank_diff <= 1 and (file_diff + rank_diff > 0)
            
        return False

    def _is_path_blocked(self, board: chess.Board, from_square: chess.Square, to_square: chess.Square) -> bool:
        """Check if the path between two squares is blocked by pieces."""
        from_file, from_rank = chess.square_file(from_square), chess.square_rank(from_square)
        to_file, to_rank = chess.square_file(to_square), chess.square_rank(to_square)
        
        # Calculate direction
        file_dir = 0 if from_file == to_file else (1 if to_file > from_file else -1)
        rank_dir = 0 if from_rank == to_rank else (1 if to_rank > from_rank else -1)
        
        # Check each square in the path (excluding start and end)
        current_file, current_rank = from_file + file_dir, from_rank + rank_dir
        while current_file != to_file or current_rank != to_rank:
            square = chess.square(current_file, current_rank)
            if board.piece_at(square) is not None:
                return True
            current_file += file_dir
            current_rank += rank_dir
            
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
        """Evaluate mobility of pieces. Optimized version."""
        score = 0.0
        
        # Use chess library's efficient piece location instead of iterating all squares
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece_type, color):
                # Quick mobility count using chess library's attacks method
                score += len(list(board.attacks(square))) * 0.1  # Scale factor to prevent mobility explosion

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
        """Evaluate tactical elements related to captures and hanging pieces. Optimized version."""
        score = 0.0
        
        # Count captures efficiently
        for move in board.legal_moves:
            if board.is_capture(move):
                piece_making_capture = board.piece_at(move.from_square)
                if piece_making_capture and piece_making_capture.color == color:
                    score += 1.0

        # Efficiently check hanging pieces using chess library piece locations
        opponent_color = not color
        
        # Check opponent pieces for hanging status
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece_type, opponent_color):
                # Use chess library's efficient attack detection
                if board.is_attacked_by(color, square) and not board.is_attacked_by(opponent_color, square):
                    score += 0.5
                    
        # Check our pieces for hanging status
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece_type, color):
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
        """Evaluate piece defense coordination for all pieces of the given color. Optimized version."""
        score = 0.0
        
        # Use chess library's efficient piece location instead of iterating all squares
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            for square in board.pieces(piece_type, color):
                # Use chess library's efficient attack detection
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
        """Evaluate open files for rooks and king safety. Optimized version."""
        score = 0.0
        
        # Pre-calculate pawn positions by file for efficiency
        our_pawns_by_file = {}
        enemy_pawns_by_file = {}
        
        for pawn_square in board.pieces(chess.PAWN, color):
            file = chess.square_file(pawn_square)
            our_pawns_by_file[file] = True
            
        for pawn_square in board.pieces(chess.PAWN, not color):
            file = chess.square_file(pawn_square)
            enemy_pawns_by_file[file] = True
        
        for file in range(8):
            has_our_pawn = file in our_pawns_by_file
            has_enemy_pawn = file in enemy_pawns_by_file
            
            # Determine file type
            if not has_our_pawn and not has_enemy_pawn:
                # Truly open file
                score += 0.5
                file_bonus = 0.5
            elif not has_our_pawn and has_enemy_pawn:
                # Semi-open file for us
                score += 0.25
                file_bonus = 0.5
            else:
                file_bonus = 0.0

            # Check if we have a rook on this file
            if file_bonus > 0:
                for rook_square in board.pieces(chess.ROOK, color):
                    if chess.square_file(rook_square) == file:
                        score += file_bonus
                        break  # Only count once per file
            
            # King safety check
            king_square = board.king(color)
            if king_square and chess.square_file(king_square) == file:
                if not has_our_pawn:  # King on open or semi-open file
                    score -= 5.0

        return score

    # ===============================================
    # V5.4 TACTICAL PATTERN RECOGNITION SYSTEM
    # ===============================================

    def _tactical_pattern_recognition(self, board: chess.Board, color: chess.Color) -> float:
        """
        Advanced tactical pattern recognition system.
        Detects pins, forks, skewers, discoveries, and other tactical motifs.
        """
        score = 0.0
        
        # Pin detection and evaluation
        score += self._detect_pins(board, color)
        
        # Fork recognition (including queen right-triangle patterns)
        score += self._detect_forks(board, color)
        
        # Skewer detection (especially king-queen alignments)
        score += self._detect_skewers(board, color)
        
        # Discovered attack recognition
        score += self._detect_discovered_attacks(board, color)
        
        # Removing the guard tactics
        score += self._detect_guard_removal(board, color)
        
        # Deflection tactics
        score += self._detect_deflection_tactics(board, color)
        
        return score

    def _detect_pins(self, board: chess.Board, color: chess.Color) -> float:
        """Detect pinned pieces and pin opportunities."""
        score = 0.0
        
        # Check for our pieces that are creating pins
        for piece_square in board.pieces(chess.ROOK, color) | board.pieces(chess.BISHOP, color) | board.pieces(chess.QUEEN, color):
            pin_bonus = self._analyze_pin_opportunities(board, piece_square, color)
            score += pin_bonus
        
        # Penalty for our pieces being pinned
        for piece_square in chess.SQUARES:
            piece = board.piece_at(piece_square)
            if piece and piece.color == color:
                if self._is_piece_pinned(board, piece_square, color):
                    # Penalty based on piece value
                    piece_value = self.piece_values.get(piece.piece_type, 0)
                    score -= piece_value * 0.3  # 30% penalty for being pinned
        
        return score

    def _analyze_pin_opportunities(self, board: chess.Board, pinning_square: chess.Square, color: chess.Color) -> float:
        """Analyze if a piece is creating a pin."""
        pinning_piece = board.piece_at(pinning_square)
        if not pinning_piece:
            return 0.0
            
        bonus = 0.0
        
        # Get all squares this piece attacks
        for target_square in board.attacks(pinning_square):
            target_piece = board.piece_at(target_square)
            if target_piece and target_piece.color != color:
                # Check if there's a more valuable piece behind this target
                behind_piece = self._get_piece_behind_target(board, pinning_square, target_square)
                if behind_piece and behind_piece.color != color:
                    # We have a pin! Calculate bonus
                    target_value = self.piece_values.get(target_piece.piece_type, 0)
                    behind_value = self.piece_values.get(behind_piece.piece_type, 0)
                    
                    if behind_value > target_value:
                        # High-value pin (valuable piece behind less valuable one)
                        bonus += 1.5
                    elif behind_piece.piece_type == chess.KING:
                        # Absolute pin (piece pinned to king)
                        bonus += 2.0
                    else:
                        # Regular pin
                        bonus += 0.8
        
        return bonus

    def _get_piece_behind_target(self, board: chess.Board, attacker_square: chess.Square, target_square: chess.Square) -> chess.Piece | None:
        """Find piece directly behind target in line of attack."""
        # Calculate direction vector
        att_file, att_rank = chess.square_file(attacker_square), chess.square_rank(attacker_square)
        tgt_file, tgt_rank = chess.square_file(target_square), chess.square_rank(target_square)
        
        file_diff = tgt_file - att_file
        rank_diff = tgt_rank - att_rank
        
        # Normalize direction
        if file_diff != 0:
            file_step = 1 if file_diff > 0 else -1
        else:
            file_step = 0
            
        if rank_diff != 0:
            rank_step = 1 if rank_diff > 0 else -1
        else:
            rank_step = 0
        
        # Continue in same direction to find piece behind
        next_file = tgt_file + file_step
        next_rank = tgt_rank + rank_step
        
        if 0 <= next_file <= 7 and 0 <= next_rank <= 7:
            behind_square = chess.square(next_file, next_rank)
            return board.piece_at(behind_square)
            
        return None

    def _is_piece_pinned(self, board: chess.Board, piece_square: chess.Square, color: chess.Color) -> bool:
        """Check if our piece is pinned."""
        piece = board.piece_at(piece_square)
        if not piece or piece.color != color:
            return False
            
        # Temporarily remove the piece and see if king comes under attack
        temp_board = board.copy()
        temp_board.remove_piece_at(piece_square)
        
        king_square = temp_board.king(color)
        if king_square and temp_board.is_attacked_by(not color, king_square):
            return True
            
        return False

    def _detect_forks(self, board: chess.Board, color: chess.Color) -> float:
        """Detect fork opportunities and queen right-triangle patterns."""
        score = 0.0
        
        # Check each of our pieces for fork potential
        for piece_square in chess.SQUARES:
            piece = board.piece_at(piece_square)
            if piece and piece.color == color:
                fork_bonus = self._analyze_fork_potential(board, piece_square, piece, color)
                score += fork_bonus
        
        return score

    def _analyze_fork_potential(self, board: chess.Board, piece_square: chess.Square, piece: chess.Piece, color: chess.Color) -> float:
        """Analyze fork potential for a specific piece."""
        attacked_squares = board.attacks(piece_square)
        enemy_targets = []
        
        # Collect enemy pieces this piece attacks
        for target_square in attacked_squares:
            target_piece = board.piece_at(target_square)
            if target_piece and target_piece.color != color:
                enemy_targets.append((target_square, target_piece))
        
        # Fork bonus based on number and value of targets
        if len(enemy_targets) >= 2:
            fork_value = 0.0
            
            # Calculate total value of forked pieces
            total_target_value = sum(self.piece_values.get(target[1].piece_type, 0) for target in enemy_targets)
            
            # Special queen right-triangle pattern detection
            if piece.piece_type == chess.QUEEN:
                right_triangle_bonus = self._detect_queen_right_triangle(piece_square, enemy_targets)
                fork_value += right_triangle_bonus
            
            # General fork bonuses
            if len(enemy_targets) == 2:
                fork_value += 1.0 + (total_target_value * 0.1)
            elif len(enemy_targets) >= 3:
                fork_value += 2.0 + (total_target_value * 0.15)  # Multi-fork bonus
            
            # Extra bonus for forking king
            for target_square, target_piece in enemy_targets:
                if target_piece.piece_type == chess.KING:
                    fork_value += 1.5  # Royal fork bonus
            
            return fork_value
        
        return 0.0

    def _detect_queen_right_triangle(self, queen_square: chess.Square, targets: list) -> float:
        """Detect queen right-triangle fork patterns."""
        if len(targets) < 2:
            return 0.0
            
        queen_file, queen_rank = chess.square_file(queen_square), chess.square_rank(queen_square)
        bonus = 0.0
        
        # Check each pair of targets for right-triangle pattern
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                target1_square, target1_piece = targets[i]
                target2_square, target2_piece = targets[j]
                
                t1_file, t1_rank = chess.square_file(target1_square), chess.square_rank(target1_square)
                t2_file, t2_rank = chess.square_file(target2_square), chess.square_rank(target2_square)
                
                # Check for right-triangle pattern (one target on rank, one on file, or diagonal combinations)
                forms_right_triangle = (
                    (t1_file == queen_file and t2_rank == queen_rank) or  # L-shape: vertical + horizontal
                    (t1_rank == queen_rank and t2_file == queen_file) or  # L-shape: horizontal + vertical
                    (abs(t1_file - queen_file) == abs(t1_rank - queen_rank) and 
                     abs(t2_file - queen_file) == abs(t2_rank - queen_rank) and
                     (t1_file - queen_file) * (t2_file - queen_file) < 0)  # Diagonal right-angle
                )
                
                if forms_right_triangle:
                    # Right-triangle pattern bonus
                    target_values = self.piece_values.get(target1_piece.piece_type, 0) + self.piece_values.get(target2_piece.piece_type, 0)
                    bonus += 0.8 + (target_values * 0.05)
        
        return bonus

    def _detect_skewers(self, board: chess.Board, color: chess.Color) -> float:
        """Detect skewer opportunities, especially king-queen alignments."""
        score = 0.0
        
        # Check our long-range pieces for skewer opportunities
        long_range_pieces = board.pieces(chess.ROOK, color) | board.pieces(chess.BISHOP, color) | board.pieces(chess.QUEEN, color)
        
        for piece_square in long_range_pieces:
            skewer_bonus = self._analyze_skewer_opportunities(board, piece_square, color)
            score += skewer_bonus
        
        return score

    def _analyze_skewer_opportunities(self, board: chess.Board, attacking_square: chess.Square, color: chess.Color) -> float:
        """Analyze skewer opportunities for a long-range piece."""
        bonus = 0.0
        attacking_piece = board.piece_at(attacking_square)
        
        if not attacking_piece:
            return 0.0
        
        # Check all directions this piece can move
        directions = self._get_piece_directions(attacking_piece.piece_type)
        
        for direction in directions:
            skewer_value = self._check_direction_for_skewer(board, attacking_square, direction, color)
            bonus += skewer_value
        
        return bonus

    def _get_piece_directions(self, piece_type: chess.PieceType) -> list:
        """Get movement directions for different piece types."""
        if piece_type == chess.ROOK:
            return [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Horizontal and vertical
        elif piece_type == chess.BISHOP:
            return [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # Diagonals
        elif piece_type == chess.QUEEN:
            return [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # All directions
        else:
            return []

    def _check_direction_for_skewer(self, board: chess.Board, start_square: chess.Square, direction: tuple, color: chess.Color) -> float:
        """Check a specific direction for skewer patterns."""
        dx, dy = direction
        file = chess.square_file(start_square)
        rank = chess.square_rank(start_square)
        
        first_enemy_piece = None
        first_enemy_square = None
        
        # Move in the direction to find the first enemy piece
        step = 1
        while True:
            new_file = file + (dx * step)
            new_rank = rank + (dy * step)
            
            if new_file < 0 or new_file > 7 or new_rank < 0 or new_rank > 7:
                break
                
            square = chess.square(new_file, new_rank)
            piece = board.piece_at(square)
            
            if piece:
                if piece.color != color:
                    first_enemy_piece = piece
                    first_enemy_square = square
                    break
                else:
                    # Our own piece blocks the path
                    return 0.0
            
            step += 1
        
        if not first_enemy_piece:
            return 0.0
        
        # Continue in same direction to find second enemy piece
        step += 1
        while True:
            new_file = file + (dx * step)
            new_rank = rank + (dy * step)
            
            if new_file < 0 or new_file > 7 or new_rank < 0 or new_rank > 7:
                break
                
            square = chess.square(new_file, new_rank)
            piece = board.piece_at(square)
            
            if piece:
                if piece.color != color:
                    # Found second enemy piece - potential skewer!
                    first_value = self.piece_values.get(first_enemy_piece.piece_type, 0)
                    second_value = self.piece_values.get(piece.piece_type, 0)
                    
                    # Skewer is valuable if the front piece is less valuable than the back piece
                    if first_value < second_value:
                        # Calculate skewer bonus
                        value_difference = second_value - first_value
                        bonus = 1.0 + (value_difference * 0.1)
                        
                        # Extra bonus for king skewers
                        if piece.piece_type == chess.KING:
                            bonus += 2.0
                        
                        return bonus
                else:
                    # Our piece blocks, no skewer possible
                    break
            
            step += 1
        
        return 0.0

    def _detect_discovered_attacks(self, board: chess.Board, color: chess.Color) -> float:
        """Detect discovered attack opportunities."""
        score = 0.0
        
        # For each of our pieces, check if moving it would create a discovered attack
        for piece_square in chess.SQUARES:
            piece = board.piece_at(piece_square)
            if piece and piece.color == color:
                discovery_bonus = self._analyze_discovered_attack_potential(board, piece_square, color)
                score += discovery_bonus
        
        return score

    def _analyze_discovered_attack_potential(self, board: chess.Board, piece_square: chess.Square, color: chess.Color) -> float:
        """Analyze discovered attack potential when moving a specific piece."""
        bonus = 0.0
        
        # Get legal moves for this piece
        piece_moves = [move for move in board.legal_moves if move.from_square == piece_square]
        
        for move in piece_moves[:5]:  # Limit to first 5 moves for performance
            # Make the move temporarily
            temp_board = board.copy()
            temp_board.push(move)
            
            # Check if any of our remaining pieces now attack valuable enemy pieces
            # that weren't being attacked before
            discovery_value = self._calculate_discovery_value(board, temp_board, piece_square, color)
            bonus += discovery_value
            
        return bonus

    def _calculate_discovery_value(self, original_board: chess.Board, new_board: chess.Board, moved_piece_square: chess.Square, color: chess.Color) -> float:
        """Calculate the value of discovered attacks after a move."""
        value = 0.0
        
        # Find pieces that gained new attacks after the move
        for our_piece_square in chess.SQUARES:
            our_piece = new_board.piece_at(our_piece_square)
            if our_piece and our_piece.color == color and our_piece_square != moved_piece_square:
                
                # Get attacks before and after the move
                old_attacks = original_board.attacks(our_piece_square) if original_board.piece_at(our_piece_square) else set()
                new_attacks = new_board.attacks(our_piece_square)
                
                # Find newly attacked enemy pieces
                newly_attacked = new_attacks - old_attacks
                
                for target_square in newly_attacked:
                    target_piece = new_board.piece_at(target_square)
                    if target_piece and target_piece.color != color:
                        # Check if this target is defended
                        is_defended = new_board.is_attacked_by(not color, target_square)
                        
                        target_value = self.piece_values.get(target_piece.piece_type, 0)
                        our_piece_value = self.piece_values.get(our_piece.piece_type, 0)
                        
                        if not is_defended:
                            # Undefended piece discovered - full value
                            value += target_value * 0.8
                        elif target_value > our_piece_value:
                            # Defended but favorable trade
                            value += (target_value - our_piece_value) * 0.3
        
        return value

    def _detect_guard_removal(self, board: chess.Board, color: chess.Color) -> float:
        """Detect opportunities to remove defenders of valuable pieces."""
        score = 0.0
        
        # Find enemy pieces that are defending other valuable pieces
        for enemy_square in chess.SQUARES:
            enemy_piece = board.piece_at(enemy_square)
            if enemy_piece and enemy_piece.color != color:
                
                # Check if this piece is defending something valuable
                defended_value = self._calculate_defended_piece_value(board, enemy_square, not color)
                
                if defended_value > 0:
                    # Check if we can capture this defender
                    if board.is_attacked_by(color, enemy_square):
                        defender_value = self.piece_values.get(enemy_piece.piece_type, 0)
                        
                        # Bonus for removing guard, especially if defender is less valuable than defended
                        if defended_value > defender_value:
                            score += 1.0 + ((defended_value - defender_value) * 0.1)
                        else:
                            score += 0.5  # Still good to remove guards
        
        return score

    def _calculate_defended_piece_value(self, board: chess.Board, defender_square: chess.Square, defending_color: chess.Color) -> float:
        """Calculate total value of pieces defended by a specific piece."""
        total_value = 0.0
        
        # Get squares defended by this piece
        defended_squares = board.attacks(defender_square)
        
        for square in defended_squares:
            piece = board.piece_at(square)
            if piece and piece.color == defending_color:
                # Check if this piece is also attacked by opponent
                if board.is_attacked_by(not defending_color, square):
                    piece_value = self.piece_values.get(piece.piece_type, 0)
                    total_value += piece_value
        
        return total_value

    def _detect_deflection_tactics(self, board: chess.Board, color: chess.Color) -> float:
        """Detect deflection opportunities - forcing pieces away from important duties."""
        score = 0.0
        
        # Look for enemy pieces that are performing important defensive functions
        for enemy_square in chess.SQUARES:
            enemy_piece = board.piece_at(enemy_square)
            if enemy_piece and enemy_piece.color != color:
                
                # Check if this piece has important defensive duties
                defensive_importance = self._assess_defensive_importance(board, enemy_square, not color)
                
                if defensive_importance > 0:
                    # Check if we can force this piece to move (deflect it)
                    deflection_potential = self._assess_deflection_potential(board, enemy_square, color)
                    
                    if deflection_potential > 0:
                        # Bonus for deflection opportunity
                        score += defensive_importance * deflection_potential * 0.5
        
        return score

    def _assess_defensive_importance(self, board: chess.Board, piece_square: chess.Square, color: chess.Color) -> float:
        """Assess how important a piece's defensive role is."""
        importance = 0.0
        
        # Check what this piece is defending
        defended_squares = board.attacks(piece_square)
        
        for square in defended_squares:
            defended_piece = board.piece_at(square)
            if defended_piece and defended_piece.color == color:
                
                # Check if defended piece is also under attack
                if board.is_attacked_by(not color, square):
                    piece_value = self.piece_values.get(defended_piece.piece_type, 0)
                    
                    # Higher importance for defending more valuable pieces
                    if defended_piece.piece_type == chess.KING:
                        importance += 10.0  # Critical defense
                    elif defended_piece.piece_type == chess.QUEEN:
                        importance += 3.0   # Very important
                    else:
                        importance += piece_value * 0.1
        
        # Check if piece is defending key squares (like king area)
        king_square = board.king(color)
        if king_square:
            king_area = self._get_king_area_squares(king_square)
            defended_king_area_squares = len([sq for sq in defended_squares if sq in king_area])
            importance += defended_king_area_squares * 0.5
        
        return importance

    def _assess_deflection_potential(self, board: chess.Board, target_square: chess.Square, attacking_color: chess.Color) -> float:
        """Assess our ability to deflect an enemy piece."""
        potential = 0.0
        
        # Check if we can attack this piece
        if board.is_attacked_by(attacking_color, target_square):
            potential += 0.5
            
            # Higher potential if we can force a favorable trade
            target_piece = board.piece_at(target_square)
            if target_piece:
                target_value = self.piece_values.get(target_piece.piece_type, 0)
                
                # Find our cheapest attacker
                cheapest_attacker_value = float('inf')
                for attacker_square in board.attackers(attacking_color, target_square):
                    attacker_piece = board.piece_at(attacker_square)
                    if attacker_piece:
                        attacker_value = self.piece_values.get(attacker_piece.piece_type, 0)
                        cheapest_attacker_value = min(cheapest_attacker_value, attacker_value)
                
                if cheapest_attacker_value <= target_value:
                    potential += 0.5  # Favorable or equal trade possible
        
        return potential

    # ===============================================
    # V5.4 ENHANCED PAWN STRUCTURE ANALYSIS
    # ===============================================

    def _enhanced_pawn_structure(self, board: chess.Board, color: chess.Color) -> float:
        """
        Advanced pawn structure analysis including isolation, doubling,
        backward pawns, pawn chains, and pawn storms.
        """
        score = 0.0
        
        # Advanced isolated pawn detection
        score += self._detect_isolated_pawns(board, color)
        
        # Doubled and tripled pawn penalties
        score += self._analyze_doubled_pawns(board, color)
        
        # Backward pawn detection
        score += self._detect_backward_pawns(board, color)
        
        # Pawn chain evaluation
        score += self._evaluate_pawn_chains(board, color)
        
        # Pawn storm detection (attacking enemy king)
        score += self._evaluate_pawn_storms(board, color)
        
        # Passed pawn advanced evaluation
        score += self._advanced_passed_pawn_eval(board, color)
        
        # Pawn base protection (pyramid structure)
        score += self._evaluate_pawn_base_protection(board, color)
        
        # Weak pawn square detection
        score += self._detect_weak_pawn_squares(board, color)
        
        return score

    def _detect_isolated_pawns(self, board: chess.Board, color: chess.Color) -> float:
        """Advanced isolated pawn detection with severity assessment."""
        score = 0.0
        
        for pawn_square in board.pieces(chess.PAWN, color):
            pawn_file = chess.square_file(pawn_square)
            pawn_rank = chess.square_rank(pawn_square)
            
            # Check for supporting pawns on adjacent files
            has_left_support = self._has_pawn_support_on_file(board, pawn_file - 1, color)
            has_right_support = self._has_pawn_support_on_file(board, pawn_file + 1, color)
            
            if not has_left_support and not has_right_support:
                # Isolated pawn - assess severity
                isolation_penalty = -0.5
                
                # More severe penalty in endgame
                if self._is_endgame(board):
                    isolation_penalty *= 1.5
                
                # More severe if on open file
                if self._is_file_open_for_opponent(board, pawn_file, color):
                    isolation_penalty *= 1.3
                
                # Less severe if pawn is advanced and active
                if (color == chess.WHITE and pawn_rank >= 5) or (color == chess.BLACK and pawn_rank <= 2):
                    isolation_penalty *= 0.7  # Reduce penalty for advanced isolated pawns
                
                score += isolation_penalty
        
        return score

    def _has_pawn_support_on_file(self, board: chess.Board, file: int, color: chess.Color) -> bool:
        """Check if there's a pawn of the same color on the specified file. Optimized version."""
        if file < 0 or file > 7:
            return False
            
        # Use chess library's efficient piece location rather than iterating all squares
        for pawn_square in board.pieces(chess.PAWN, color):
            if chess.square_file(pawn_square) == file:
                return True
        return False

    def _analyze_doubled_pawns(self, board: chess.Board, color: chess.Color) -> float:
        """Analyze doubled and tripled pawns with context. Optimized version."""
        score = 0.0
        
        # Group pawns by file efficiently using chess library
        pawns_by_file = {}
        for pawn_square in board.pieces(chess.PAWN, color):
            file = chess.square_file(pawn_square)
            rank = chess.square_rank(pawn_square)
            if file not in pawns_by_file:
                pawns_by_file[file] = []
            pawns_by_file[file].append((pawn_square, rank))
        
        # Analyze files with multiple pawns
        for file, pawns_on_file in pawns_by_file.items():
            if len(pawns_on_file) > 1:
                # Base penalty for doubled pawns
                doubling_penalty = -0.5 * (len(pawns_on_file) - 1)
                
                # Assess if doubled pawns have compensating factors
                compensation = self._assess_doubled_pawn_compensation(board, file, pawns_on_file, color)
                doubling_penalty += compensation
                
                score += doubling_penalty
        
        return score

    def _assess_doubled_pawn_compensation(self, board: chess.Board, file: int, pawns: list, color: chess.Color) -> float:
        """Assess if doubled pawns have compensating factors."""
        compensation = 0.0
        
        # Check if file is semi-open (no enemy pawns)
        if not self._has_enemy_pawns_on_file(board, file, color):
            compensation += 0.2  # Semi-open file can be useful
        
        # Check if doubled pawns control important central squares
        central_control = 0
        for pawn_square, pawn_rank in pawns:
            pawn_attacks = board.attacks(pawn_square)
            for attacked_square in pawn_attacks:
                if attacked_square in [chess.D4, chess.D5, chess.E4, chess.E5]:
                    central_control += 1
        
        if central_control > 0:
            compensation += central_control * 0.1
        
        return compensation

    def _has_enemy_pawns_on_file(self, board: chess.Board, file: int, our_color: chess.Color) -> bool:
        """Check if enemy has pawns on specified file. Optimized version."""
        enemy_color = not our_color
        # Use chess library's efficient piece location
        for pawn_square in board.pieces(chess.PAWN, enemy_color):
            if chess.square_file(pawn_square) == file:
                return True
        return False

    def _detect_backward_pawns(self, board: chess.Board, color: chess.Color) -> float:
        """Detect backward pawns that cannot advance safely."""
        score = 0.0
        
        for pawn_square in board.pieces(chess.PAWN, color):
            if self._is_backward_pawn(board, pawn_square, color):
                # Backward pawn penalty
                penalty = -0.4
                
                # More severe in endgame
                if self._is_endgame(board):
                    penalty *= 1.2
                
                score += penalty
        
        return score

    def _is_backward_pawn(self, board: chess.Board, pawn_square: chess.Square, color: chess.Color) -> bool:
        """Check if a pawn is backward (cannot advance safely and no friendly pawn support)."""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        # Check if pawn can advance
        advance_rank = pawn_rank + 1 if color == chess.WHITE else pawn_rank - 1
        if advance_rank < 0 or advance_rank > 7:
            return False
            
        advance_square = chess.square(pawn_file, advance_rank)
        
        # If advance square is occupied or attacked by enemy, pawn may be backward
        if board.piece_at(advance_square) or board.is_attacked_by(not color, advance_square):
            # Check if there are supporting pawns that can protect the advance
            supporting_pawns = self._count_supporting_pawns(board, pawn_square, color)
            
            # Check if adjacent pawns are more advanced
            left_file_advanced = self._is_adjacent_file_more_advanced(board, pawn_file - 1, pawn_rank, color)
            right_file_advanced = self._is_adjacent_file_more_advanced(board, pawn_file + 1, pawn_rank, color)
            
            if supporting_pawns == 0 and (left_file_advanced or right_file_advanced):
                return True
        
        return False

    def _count_supporting_pawns(self, board: chess.Board, pawn_square: chess.Square, color: chess.Color) -> int:
        """Count pawns that can potentially support this pawn's advance."""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        support_count = 0
        
        # Check adjacent files for supporting pawns
        for file_offset in [-1, 1]:
            support_file = pawn_file + file_offset
            if 0 <= support_file <= 7:
                # Look for pawns that could advance to support
                for rank in range(8):
                    square = chess.square(support_file, rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        # Check if this pawn could potentially support
                        pawn_can_support = False
                        if color == chess.WHITE:
                            pawn_can_support = rank < pawn_rank
                        else:
                            pawn_can_support = rank > pawn_rank
                        
                        if pawn_can_support:
                            support_count += 1
                            break
        
        return support_count

    def _is_adjacent_file_more_advanced(self, board: chess.Board, file: int, our_rank: int, color: chess.Color) -> bool:
        """Check if pawns on adjacent file are more advanced."""
        if file < 0 or file > 7:
            return False
        
        for rank in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                if color == chess.WHITE:
                    return rank > our_rank
                else:
                    return rank < our_rank
        
        return False

    def _evaluate_pawn_chains(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate pawn chains and phalanxes."""
        score = 0.0
        
        for pawn_square in board.pieces(chess.PAWN, color):
            chain_bonus = self._calculate_pawn_chain_bonus(board, pawn_square, color)
            score += chain_bonus
        
        return score

    def _calculate_pawn_chain_bonus(self, board: chess.Board, pawn_square: chess.Square, color: chess.Color) -> float:
        """Calculate bonus for pawn being part of a chain."""
        bonus = 0.0
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        # Check for pawn protection (diagonal support from behind)
        protected_by_pawn = self._is_protected_by_pawn(board, pawn_square, color)
        if protected_by_pawn:
            bonus += 0.2
        
        # Check for pawn phalanx (side-by-side pawns)
        has_phalanx = self._has_pawn_phalanx(board, pawn_square, color)
        if has_phalanx:
            bonus += 0.3
        
        # Check for advanced pawn chain
        if protected_by_pawn:
            chain_length = self._calculate_chain_length(board, pawn_square, color)
            if chain_length >= 3:
                bonus += 0.2 * (chain_length - 2)  # Bonus for long chains
        
        # Central pawn chain bonus
        if pawn_file in [3, 4]:  # D and E files
            bonus *= 1.2
        
        return bonus

    def _is_protected_by_pawn(self, board: chess.Board, pawn_square: chess.Square, color: chess.Color) -> bool:
        """Check if pawn is protected by another pawn."""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        # Check diagonal squares behind the pawn
        protection_rank = pawn_rank - 1 if color == chess.WHITE else pawn_rank + 1
        
        if 0 <= protection_rank <= 7:
            for file_offset in [-1, 1]:
                protection_file = pawn_file + file_offset
                if 0 <= protection_file <= 7:
                    protection_square = chess.square(protection_file, protection_rank)
                    piece = board.piece_at(protection_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        return True
        
        return False

    def _has_pawn_phalanx(self, board: chess.Board, pawn_square: chess.Square, color: chess.Color) -> bool:
        """Check if pawn has a phalanx (side-by-side pawn)."""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        for file_offset in [-1, 1]:
            adjacent_file = pawn_file + file_offset
            if 0 <= adjacent_file <= 7:
                adjacent_square = chess.square(adjacent_file, pawn_rank)
                piece = board.piece_at(adjacent_square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    return True
        
        return False

    def _calculate_chain_length(self, board: chess.Board, pawn_square: chess.Square, color: chess.Color) -> int:
        """Calculate the length of a pawn chain starting from this pawn."""
        length = 1
        current_square = pawn_square
        
        # Follow the chain backwards
        while True:
            pawn_file = chess.square_file(current_square)
            pawn_rank = chess.square_rank(current_square)
            
            found_support = False
            support_rank = pawn_rank - 1 if color == chess.WHITE else pawn_rank + 1
            
            if 0 <= support_rank <= 7:
                for file_offset in [-1, 1]:
                    support_file = pawn_file + file_offset
                    if 0 <= support_file <= 7:
                        support_square = chess.square(support_file, support_rank)
                        piece = board.piece_at(support_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            length += 1
                            current_square = support_square
                            found_support = True
                            break
            
            if not found_support:
                break
        
        return length

    def _evaluate_pawn_storms(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate pawn storms against enemy king."""
        score = 0.0
        
        enemy_king_square = board.king(not color)
        if not enemy_king_square:
            return 0.0
        
        enemy_king_file = chess.square_file(enemy_king_square)
        
        # Check pawns near enemy king files
        storm_files = [enemy_king_file + offset for offset in [-1, 0, 1] if 0 <= enemy_king_file + offset <= 7]
        
        for storm_file in storm_files:
            storm_bonus = self._calculate_storm_potential(board, storm_file, color)
            score += storm_bonus
        
        return score

    def _calculate_storm_potential(self, board: chess.Board, file: int, color: chess.Color) -> float:
        """Calculate pawn storm potential on a specific file."""
        bonus = 0.0
        
        # Find our most advanced pawn on this file
        most_advanced_rank = None
        for rank in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                if color == chess.WHITE:
                    most_advanced_rank = rank if most_advanced_rank is None else max(most_advanced_rank, rank)
                else:
                    most_advanced_rank = rank if most_advanced_rank is None else min(most_advanced_rank, rank)
        
        if most_advanced_rank is not None:
            # Calculate advancement bonus
            if color == chess.WHITE:
                advancement = most_advanced_rank - 1  # Relative to starting rank
            else:
                advancement = 6 - most_advanced_rank  # Relative to starting rank
            
            if advancement >= 3:
                bonus += 0.3 + (advancement * 0.1)
        
        return bonus

    def _advanced_passed_pawn_eval(self, board: chess.Board, color: chess.Color) -> float:
        """Advanced passed pawn evaluation with king proximity and support."""
        score = 0.0
        
        for pawn_square in board.pieces(chess.PAWN, color):
            if self._is_passed_pawn_advanced(board, pawn_square, color):
                passed_bonus = self._calculate_passed_pawn_bonus(board, pawn_square, color)
                score += passed_bonus
        
        return score

    def _is_passed_pawn_advanced(self, board: chess.Board, pawn_square: chess.Square, color: chess.Color) -> bool:
        """Advanced passed pawn detection."""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        # Check files in front of pawn
        files_to_check = [pawn_file + offset for offset in [-1, 0, 1] if 0 <= pawn_file + offset <= 7]
        
        for check_file in files_to_check:
            for rank in range(8):
                if color == chess.WHITE and rank <= pawn_rank:
                    continue
                if color == chess.BLACK and rank >= pawn_rank:
                    continue
                
                square = chess.square(check_file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color != color:
                    return False  # Enemy pawn blocks path
        
        return True

    def _calculate_passed_pawn_bonus(self, board: chess.Board, pawn_square: chess.Square, color: chess.Color) -> float:
        """Calculate bonus for passed pawn with various factors."""
        bonus = 0.5  # Base passed pawn bonus
        
        pawn_rank = chess.square_rank(pawn_square)
        
        # Advancement bonus
        if color == chess.WHITE:
            advancement = pawn_rank - 1
        else:
            advancement = 6 - pawn_rank
        
        bonus += advancement * 0.2
        
        # King proximity factor
        our_king = board.king(color)
        enemy_king = board.king(not color)
        
        if our_king and enemy_king:
            our_king_distance = chess.square_distance(our_king, pawn_square)
            enemy_king_distance = chess.square_distance(enemy_king, pawn_square)
            
            # Bonus if our king is closer to support the pawn
            if our_king_distance < enemy_king_distance:
                bonus += 0.3
            
            # Extra bonus in endgame with close king support
            if self._is_endgame(board) and our_king_distance <= 2:
                bonus += 0.5
        
        # Protection bonus
        if board.is_attacked_by(color, pawn_square):
            bonus += 0.2
        
        return bonus

    def _evaluate_pawn_base_protection(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate pawn pyramid structure and base protection."""
        score = 0.0
        
        # Find pawn pyramids and check base protection
        for pawn_square in board.pieces(chess.PAWN, color):
            if self._is_pawn_base(board, pawn_square, color):
                base_protection = self._assess_base_protection(board, pawn_square, color)
                score += base_protection
        
        return score

    def _is_pawn_base(self, board: chess.Board, pawn_square: chess.Square, color: chess.Color) -> bool:
        """Check if pawn is a base of a pawn structure (has pawns in front)."""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        # Check if there are pawns in front on same or adjacent files
        front_pawns = 0
        for file_offset in [-1, 0, 1]:
            check_file = pawn_file + file_offset
            if 0 <= check_file <= 7:
                for rank_offset in range(1, 4):  # Check 3 ranks ahead
                    if color == chess.WHITE:
                        check_rank = pawn_rank + rank_offset
                    else:
                        check_rank = pawn_rank - rank_offset
                    
                    if 0 <= check_rank <= 7:
                        check_square = chess.square(check_file, check_rank)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            front_pawns += 1
        
        return front_pawns >= 2  # Consider it a base if it supports multiple pawns

    def _assess_base_protection(self, board: chess.Board, base_square: chess.Square, color: chess.Color) -> float:
        """Assess how well protected a pawn base is."""
        protection = 0.0
        
        # Check if base is defended
        if board.is_attacked_by(color, base_square):
            protection += 0.3
        
        # Penalty if base is undefended and under attack
        if board.is_attacked_by(not color, base_square) and not board.is_attacked_by(color, base_square):
            protection -= 0.8  # Heavy penalty for weak base
        
        # Check for piece support
        defenders = len(list(board.attackers(color, base_square)))
        if defenders >= 2:
            protection += 0.2  # Well-defended base
        
        return protection

    def _detect_weak_pawn_squares(self, board: chess.Board, color: chess.Color) -> float:
        """Detect weak squares in pawn structure that enemy can exploit. Optimized version."""
        score = 0.0
        
        # Focus only on critical squares rather than checking all 64 squares
        # Check central squares and advanced squares for weaknesses
        critical_squares = []
        
        # Add central squares
        critical_squares.extend([chess.D3, chess.D4, chess.D5, chess.D6, chess.E3, chess.E4, chess.E5, chess.E6])
        
        # Add squares in advanced territory
        if color == chess.WHITE:
            # Check 5th and 6th ranks for white
            for file in range(8):
                critical_squares.extend([chess.square(file, 4), chess.square(file, 5)])
        else:
            # Check 3rd and 4th ranks for black  
            for file in range(8):
                critical_squares.extend([chess.square(file, 2), chess.square(file, 3)])
        
        # Check only these critical squares for weakness
        for square in critical_squares:
            if not board.piece_at(square):  # Empty square
                weakness = self._assess_square_weakness(board, square, color)
                score += weakness
        
        return score

    def _assess_square_weakness(self, board: chess.Board, square: chess.Square, color: chess.Color) -> float:
        """Assess if an empty square is weak for our pawn structure."""
        weakness = 0.0
        
        # Check if square can be controlled by enemy pieces but not by our pawns
        enemy_can_control = board.is_attacked_by(not color, square)
        we_can_control_with_pawns = self._can_control_with_pawns(board, square, color)
        
        if enemy_can_control and not we_can_control_with_pawns:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            # More serious weakness in central files and advanced ranks
            if file in [3, 4]:  # D and E files
                weakness -= 0.2
            
            if (color == chess.WHITE and rank >= 5) or (color == chess.BLACK and rank <= 2):
                weakness -= 0.3  # Advanced weak squares are dangerous
        
        return weakness

    def _can_control_with_pawns(self, board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
        """Check if we can control a square with our pawns. Optimized version."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        # Direct calculation: check if any existing pawn could potentially attack this square
        # Much faster than nested loops through all squares
        attack_rank = rank - 1 if color == chess.WHITE else rank + 1
        
        if 0 <= attack_rank <= 7:
            # Check left and right files for pawns that could attack this square
            for attack_file in [file - 1, file + 1]:
                if 0 <= attack_file <= 7:
                    attack_square = chess.square(attack_file, attack_rank)
                    piece = board.piece_at(attack_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        return True
                    
                    # Also check if pawn behind could advance to attack position
                    behind_rank = attack_rank - 1 if color == chess.WHITE else attack_rank + 1
                    if 0 <= behind_rank <= 7:
                        behind_square = chess.square(attack_file, behind_rank)
                        behind_piece = board.piece_at(behind_square)
                        if behind_piece and behind_piece.piece_type == chess.PAWN and behind_piece.color == color:
                            # Check if path is clear for advancement
                            if not board.piece_at(attack_square):
                                return True
        
        return False

    def _is_endgame(self, board: chess.Board) -> bool:
        """Simple endgame detection based on material. Optimized version."""
        # Quick piece count using chess library's efficient methods
        total_pieces = 0
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            total_pieces += len(board.pieces(piece_type, chess.WHITE))
            total_pieces += len(board.pieces(piece_type, chess.BLACK))
        
        # Endgame if 6 or fewer pieces (excluding kings and pawns)
        return total_pieces <= 6

    def _is_file_open_for_opponent(self, board: chess.Board, file: int, our_color: chess.Color) -> bool:
        """Check if file is open from opponent's perspective."""
        enemy_color = not our_color
        
        # File is open for opponent if they have no pawns on it
        for rank in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == enemy_color:
                return False
        
        return True

    # ===============================================
    # V5.4 ENDGAME LOGIC SYSTEM
    # ===============================================

    def _endgame_logic(self, board: chess.Board, color: chess.Color) -> float:
        """
        Advanced endgame evaluation including king activity,
        pawn promotion, opposition, and endgame principles.
        """
        if not self._is_endgame(board):
            return 0.0
            
        score = 0.0
        
        # King activity and centralization
        score += self._king_activity_endgame(board, color)
        
        # King-pawn coordination
        score += self._king_pawn_coordination(board, color)
        
        # Opposition evaluation
        score += self._evaluate_opposition(board, color)
        
        # Pawn promotion evaluation
        score += self._pawn_promotion_endgame(board, color)
        
        # King safety in endgame (different from middlegame)
        score += self._endgame_king_safety(board, color)
        
        # Zugzwang detection
        score += self._detect_zugzwang(board, color)
        
        return score

    def _king_activity_endgame(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate king activity and centralization in endgame."""
        score = 0.0
        king_square = board.king(color)
        
        if not king_square:
            return 0.0
        
        # King centralization bonus
        centralization_bonus = self._calculate_king_centralization(king_square)
        score += centralization_bonus
        
        # King mobility bonus
        mobility_bonus = self._calculate_king_mobility(board, king_square, color)
        score += mobility_bonus
        
        # King attacking enemy pawns
        pawn_attack_bonus = self._king_attacking_pawns(board, king_square, color)
        score += pawn_attack_bonus
        
        # King supporting own pawns
        pawn_support_bonus = self._king_supporting_pawns(board, king_square, color)
        score += pawn_support_bonus
        
        return score

    def _calculate_king_centralization(self, king_square: chess.Square) -> float:
        """Calculate bonus for king centralization."""
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)
        
        # Distance from center (d4, d5, e4, e5)
        center_distance = min(
            abs(file - 3) + abs(rank - 3),  # Distance to d4
            abs(file - 3) + abs(rank - 4),  # Distance to d5
            abs(file - 4) + abs(rank - 3),  # Distance to e4
            abs(file - 4) + abs(rank - 4)   # Distance to e5
        )
        
        # Bonus decreases with distance from center
        centralization_bonus = max(0, 1.0 - (center_distance * 0.15))
        
        return centralization_bonus

    def _calculate_king_mobility(self, board: chess.Board, king_square: chess.Square, color: chess.Color) -> float:
        """Calculate king mobility bonus in endgame."""
        legal_moves = 0
        
        # Count legal king moves
        for move in board.legal_moves:
            if move.from_square == king_square:
                legal_moves += 1
        
        # Mobility bonus (active king is good in endgame)
        mobility_bonus = legal_moves * 0.05
        
        return mobility_bonus

    def _king_attacking_pawns(self, board: chess.Board, king_square: chess.Square, color: chess.Color) -> float:
        """Bonus for king attacking enemy pawns."""
        bonus = 0.0
        king_attacks = board.attacks(king_square)
        
        for attacked_square in king_attacks:
            piece = board.piece_at(attacked_square)
            if piece and piece.piece_type == chess.PAWN and piece.color != color:
                # Bonus for attacking enemy pawns with king
                bonus += 0.3
                
                # Extra bonus if pawn is undefended
                if not board.is_attacked_by(not color, attacked_square):
                    bonus += 0.5
        
        return bonus

    def _king_supporting_pawns(self, board: chess.Board, king_square: chess.Square, color: chess.Color) -> float:
        """Bonus for king supporting own pawns."""
        bonus = 0.0
        
        # Check proximity to own pawns
        for pawn_square in board.pieces(chess.PAWN, color):
            distance = chess.square_distance(king_square, pawn_square)
            
            if distance <= 2:
                # Close king support
                bonus += 0.2
                
                # Extra bonus for supporting passed pawns
                if self._is_passed_pawn_advanced(board, pawn_square, color):
                    bonus += 0.4
        
        return bonus

    def _king_pawn_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate king-pawn coordination in endgame."""
        score = 0.0
        king_square = board.king(color)
        
        if not king_square:
            return 0.0
        
        # King ahead of own pawns
        ahead_bonus = self._king_ahead_of_pawns(board, king_square, color)
        score += ahead_bonus
        
        # King blocking enemy pawns
        blocking_bonus = self._king_blocking_enemy_pawns(board, king_square, color)
        score += blocking_bonus
        
        # King-pawn vs king evaluation
        if self._is_king_pawn_endgame(board):
            kp_evaluation = self._evaluate_king_pawn_endgame(board, color)
            score += kp_evaluation
        
        return score

    def _king_ahead_of_pawns(self, board: chess.Board, king_square: chess.Square, color: chess.Color) -> float:
        """Bonus for king being ahead of own pawns (leading the charge)."""
        bonus = 0.0
        king_rank = chess.square_rank(king_square)
        
        for pawn_square in board.pieces(chess.PAWN, color):
            pawn_rank = chess.square_rank(pawn_square)
            
            # Check if king is ahead of pawn
            if color == chess.WHITE:
                king_ahead = king_rank > pawn_rank
            else:
                king_ahead = king_rank < pawn_rank
            
            if king_ahead:
                bonus += 0.1
                
                # Extra bonus if it's a passed pawn
                if self._is_passed_pawn_advanced(board, pawn_square, color):
                    bonus += 0.3
        
        return bonus

    def _king_blocking_enemy_pawns(self, board: chess.Board, king_square: chess.Square, color: chess.Color) -> float:
        """Bonus for king blocking enemy pawn advancement."""
        bonus = 0.0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        for enemy_pawn_square in board.pieces(chess.PAWN, not color):
            pawn_file = chess.square_file(enemy_pawn_square)
            pawn_rank = chess.square_rank(enemy_pawn_square)
            
            # Check if king is blocking pawn's advancement
            if pawn_file == king_file:
                if color == chess.WHITE:
                    blocking = king_rank < pawn_rank  # White king below black pawn
                else:
                    blocking = king_rank > pawn_rank  # Black king above white pawn
                
                if blocking:
                    bonus += 0.4
                    
                    # Extra bonus for blocking passed pawns
                    if self._is_passed_pawn_advanced(board, enemy_pawn_square, not color):
                        bonus += 0.6
        
        return bonus

    def _is_king_pawn_endgame(self, board: chess.Board) -> bool:
        """Check if it's a king and pawn endgame."""
        piece_count = 0
        pawn_count = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                if piece.piece_type == chess.PAWN:
                    pawn_count += 1
                elif piece.piece_type != chess.KING:
                    piece_count += 1
        
        return piece_count == 0 and pawn_count > 0

    def _evaluate_king_pawn_endgame(self, board: chess.Board, color: chess.Color) -> float:
        """Specialized evaluation for king and pawn endgames."""
        score = 0.0
        
        # Key square control for pawn promotion
        for pawn_square in board.pieces(chess.PAWN, color):
            key_square_bonus = self._evaluate_key_squares(board, pawn_square, color)
            score += key_square_bonus
        
        # Pawn race evaluation
        race_evaluation = self._evaluate_pawn_races(board, color)
        score += race_evaluation
        
        return score

    def _evaluate_key_squares(self, board: chess.Board, pawn_square: chess.Square, color: chess.Color) -> float:
        """Evaluate control of key squares for pawn promotion."""
        bonus = 0.0
        our_king = board.king(color)
        enemy_king = board.king(not color)
        
        if not our_king or not enemy_king:
            return 0.0
        
        # Calculate key squares (squares the king needs to control for pawn to promote)
        key_squares = self._get_key_squares_for_pawn(pawn_square, color)
        
        for key_square in key_squares:
            our_distance = chess.square_distance(our_king, key_square)
            enemy_distance = chess.square_distance(enemy_king, key_square)
            
            # Bonus if we control key squares
            if our_distance < enemy_distance:
                bonus += 0.5
            elif our_distance == enemy_distance:
                # Equal distance - check who moves first
                if board.turn == color:
                    bonus += 0.3  # We move first, slight advantage
                else:
                    bonus -= 0.3  # They move first, slight disadvantage
        
        return bonus

    def _get_key_squares_for_pawn(self, pawn_square: chess.Square, color: chess.Color) -> list:
        """Get key squares that king needs to control for pawn promotion."""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        key_squares = []
        
        # Basic key squares in front of pawn
        if color == chess.WHITE:
            target_ranks = [pawn_rank + 1, pawn_rank + 2, 7]  # Include promotion square
        else:
            target_ranks = [pawn_rank - 1, pawn_rank - 2, 0]  # Include promotion square
        
        for rank in target_ranks:
            if 0 <= rank <= 7:
                key_squares.append(chess.square(pawn_file, rank))
        
        return key_squares

    def _evaluate_pawn_races(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate pawn races to promotion."""
        score = 0.0
        
        our_closest_promotion = self._calculate_closest_promotion(board, color)
        enemy_closest_promotion = self._calculate_closest_promotion(board, not color)
        
        if our_closest_promotion is not None and enemy_closest_promotion is not None:
            # Compare promotion times
            if our_closest_promotion < enemy_closest_promotion:
                score += 2.0  # We promote first
            elif our_closest_promotion > enemy_closest_promotion:
                score -= 2.0  # They promote first
            else:
                # Same promotion time - check who moves first
                if board.turn == color:
                    score += 1.0
                else:
                    score -= 1.0
        
        return score

    def _calculate_closest_promotion(self, board: chess.Board, color: chess.Color) -> int | None:
        """Calculate minimum moves to promotion for fastest pawn."""
        min_moves = None
        
        for pawn_square in board.pieces(chess.PAWN, color):
            moves_to_promotion = self._moves_to_promotion(board, pawn_square, color)
            if moves_to_promotion is not None:
                if min_moves is None or moves_to_promotion < min_moves:
                    min_moves = moves_to_promotion
        
        return min_moves

    def _moves_to_promotion(self, board: chess.Board, pawn_square: chess.Square, color: chess.Color) -> int | None:
        """Calculate moves needed for pawn to promote."""
        pawn_rank = chess.square_rank(pawn_square)
        
        if color == chess.WHITE:
            moves_needed = 7 - pawn_rank
        else:
            moves_needed = pawn_rank
        
        # Check if path is clear
        if self._is_promotion_path_clear(board, pawn_square, color):
            return moves_needed
        else:
            return None  # Path blocked

    def _is_promotion_path_clear(self, board: chess.Board, pawn_square: chess.Square, color: chess.Color) -> bool:
        """Check if pawn has clear path to promotion."""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        if color == chess.WHITE:
            ranks_to_check = range(pawn_rank + 1, 8)
        else:
            ranks_to_check = range(pawn_rank - 1, -1, -1)
        
        for rank in ranks_to_check:
            square = chess.square(pawn_file, rank)
            if board.piece_at(square):
                return False  # Path blocked
        
        return True

    def _evaluate_opposition(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate opposition in king and pawn endgames."""
        score = 0.0
        our_king = board.king(color)
        enemy_king = board.king(not color)
        
        if not our_king or not enemy_king:
            return 0.0
        
        # Check for direct opposition
        direct_opposition = self._has_direct_opposition(our_king, enemy_king, color, board.turn)
        if direct_opposition:
            score += 0.5
        
        # Check for distant opposition
        distant_opposition = self._has_distant_opposition(our_king, enemy_king, color, board.turn)
        if distant_opposition:
            score += 0.3
        
        return score

    def _has_direct_opposition(self, our_king: chess.Square, enemy_king: chess.Square, color: chess.Color, turn: chess.Color) -> bool:
        """Check for direct opposition between kings."""
        our_file, our_rank = chess.square_file(our_king), chess.square_rank(our_king)
        enemy_file, enemy_rank = chess.square_file(enemy_king), chess.square_rank(enemy_king)
        
        # Direct opposition: kings on same rank/file with one square between, and it's enemy's turn
        same_file = our_file == enemy_file and abs(our_rank - enemy_rank) == 2
        same_rank = our_rank == enemy_rank and abs(our_file - enemy_file) == 2
        
        return (same_file or same_rank) and turn != color

    def _has_distant_opposition(self, our_king: chess.Square, enemy_king: chess.Square, color: chess.Color, turn: chess.Color) -> bool:
        """Check for distant opposition."""
        our_file, our_rank = chess.square_file(our_king), chess.square_rank(our_king)
        enemy_file, enemy_rank = chess.square_file(enemy_king), chess.square_rank(enemy_king)
        
        # Distant opposition: same rank/file, even number of squares between, enemy's turn
        same_file = our_file == enemy_file
        same_rank = our_rank == enemy_rank
        
        if same_file:
            distance = abs(our_rank - enemy_rank)
            return distance > 2 and distance % 2 == 0 and turn != color
        elif same_rank:
            distance = abs(our_file - enemy_file)
            return distance > 2 and distance % 2 == 0 and turn != color
        
        return False

    def _pawn_promotion_endgame(self, board: chess.Board, color: chess.Color) -> float:
        """Enhanced pawn promotion evaluation for endgame."""
        score = 0.0
        
        for pawn_square in board.pieces(chess.PAWN, color):
            promotion_score = self._calculate_promotion_urgency(board, pawn_square, color)
            score += promotion_score
        
        return score

    def _calculate_promotion_urgency(self, board: chess.Board, pawn_square: chess.Square, color: chess.Color) -> float:
        """Calculate urgency and value of pawn promotion."""
        urgency_score = 0.0
        pawn_rank = chess.square_rank(pawn_square)
        
        # Distance to promotion
        if color == chess.WHITE:
            distance_to_promotion = 7 - pawn_rank
        else:
            distance_to_promotion = pawn_rank
        
        # Higher bonus for pawns closer to promotion
        distance_bonus = max(0, 3.0 - distance_to_promotion * 0.5)
        urgency_score += distance_bonus
        
        # Check if pawn is passed and unblocked
        if self._is_passed_pawn_advanced(board, pawn_square, color):
            if self._is_promotion_path_clear(board, pawn_square, color):
                urgency_score += 2.0  # High priority for free promotion
            
            # King support for passed pawn
            our_king = board.king(color)
            if our_king:
                king_distance = chess.square_distance(our_king, pawn_square)
                if king_distance <= 2:
                    urgency_score += 1.0  # King support bonus
        
        return urgency_score

    def _endgame_king_safety(self, board: chess.Board, color: chess.Color) -> float:
        """King safety considerations specific to endgame."""
        score = 0.0
        our_king = board.king(color)
        enemy_king = board.king(not color)
        
        if not our_king or not enemy_king:
            return 0.0
        
        # In endgame, kings should be active, but avoid stalemate patterns
        stalemate_risk = self._assess_stalemate_risk(board, color)
        score += stalemate_risk
        
        # Avoid king being driven to edge
        edge_penalty = self._calculate_edge_penalty(our_king)
        score += edge_penalty
        
        return score

    def _assess_stalemate_risk(self, board: chess.Board, color: chess.Color) -> float:
        """Assess risk of stalemate."""
        # If it's our turn and we have very few legal moves, there might be stalemate risk
        if board.turn == color:
            legal_moves = list(board.legal_moves)
            if len(legal_moves) <= 2:
                return -0.5  # Risk of stalemate
        
        return 0.0

    def _calculate_edge_penalty(self, king_square: chess.Square) -> float:
        """Penalty for king being driven to edge in endgame."""
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)
        
        # Distance from edge
        edge_distance = min(file, 7 - file, rank, 7 - rank)
        
        # Penalty for being too close to edge
        if edge_distance == 0:
            return -1.0  # On the edge
        elif edge_distance == 1:
            return -0.5  # One square from edge
        
        return 0.0

    def _detect_zugzwang(self, board: chess.Board, color: chess.Color) -> float:
        """Detect potential zugzwang situations."""
        # This is a simplified zugzwang detection
        # In real games, this is very complex and position-specific
        
        if board.turn != color:
            return 0.0  # Only evaluate when it's our turn
        
        # Count legal moves
        legal_moves = list(board.legal_moves)
        
        # If we have very few moves and position is tense, might be zugzwang
        if len(legal_moves) <= 3:
            # Check if any move significantly worsens our position
            current_eval = self._quick_position_eval(board, color)
            
            worse_moves = 0
            for move in legal_moves:
                board.push(move)
                new_eval = self._quick_position_eval(board, color)
                board.pop()
                
                if new_eval < current_eval - 0.5:
                    worse_moves += 1
            
            # If most moves worsen position, we might be in zugzwang
            if worse_moves >= len(legal_moves) - 1:
                return -0.8  # Zugzwang penalty
        
        return 0.0

    def _quick_position_eval(self, board: chess.Board, color: chess.Color) -> float:
        """Quick position evaluation for zugzwang detection."""
        # Simplified evaluation - just material and king activity
        material = 0.0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values.get(piece.piece_type, 0)
                if piece.color == color:
                    material += value
                else:
                    material -= value
        
        # Add king centralization
        king_square = board.king(color)
        if king_square:
            material += self._calculate_king_centralization(king_square) * 0.5
        
        return material

    # ===============================================
    # V5.4 CHESS THEORETICAL PRINCIPLES
    # ===============================================

    def _opening_principles(self, board: chess.Board, color: chess.Color) -> float:
        """
        Evaluate adherence to chess opening principles.
        Includes development tracking, queen restraint, and piece development order.
        """
        score = 0.0
        
        # Early queen development penalty
        score += self._evaluate_queen_restraint(board, color)
        
        # Piece development order evaluation
        score += self._evaluate_development_order(board, color)
        
        # Castle timing evaluation
        score += self._evaluate_castle_timing(board, color)
        
        # Central pawn development
        score += self._evaluate_central_pawn_development(board, color)
        
        return score

    def _capture_guidelines(self, board: chess.Board, color: chess.Color) -> float:
        """
        Evaluate capture decision quality based on chess principles.
        Prefers captures toward center and with pieces over pawns.
        """
        score = 0.0
        
        # This is evaluated during move generation/ordering rather than static evaluation
        # But we can evaluate current position for capture quality indicators
        
        # Bonus for having pieces aimed at center after captures
        score += self._evaluate_center_oriented_captures(board, color)
        
        # Penalty for poor pawn structure from captures
        score += self._evaluate_capture_pawn_structure(board, color)
        
        return score

    def _evaluate_queen_restraint(self, board: chess.Board, color: chess.Color) -> float:
        """Penalty for early queen development beyond 3rd rank."""
        penalty = 0.0
        
        queen_squares = list(board.pieces(chess.QUEEN, color))
        if not queen_squares:
            return 0.0
            
        queen_square = queen_squares[0]
        queen_rank = chess.square_rank(queen_square)
        
        # Check if this is early game (based on development)
        development_stage = self._calculate_development_stage(board, color)
        
        if development_stage < 0.6:  # Still in opening/early middlegame
            # Define early development ranks for queen
            if color == chess.WHITE:
                # White queen shouldn't go beyond 4th rank early
                if queen_rank >= 4:
                    penalty -= 1.0 * (queen_rank - 3)  # Increasing penalty
            else:
                # Black queen shouldn't go beyond 5th rank early  
                if queen_rank <= 3:
                    penalty -= 1.0 * (4 - queen_rank)  # Increasing penalty
                    
        return penalty

    def _evaluate_development_order(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate proper piece development order."""
        score = 0.0
        
        # Prefer knights before bishops in opening
        knights_developed = 0
        bishops_developed = 0
        
        # Count developed minor pieces
        starting_squares = {
            chess.WHITE: {
                chess.KNIGHT: [chess.B1, chess.G1], 
                chess.BISHOP: [chess.C1, chess.F1]
            },
            chess.BLACK: {
                chess.KNIGHT: [chess.B8, chess.G8], 
                chess.BISHOP: [chess.C8, chess.F8]
            }
        }
        
        # Check knight development
        for start_square in starting_squares[color][chess.KNIGHT]:
            piece = board.piece_at(start_square)
            if not piece or piece.piece_type != chess.KNIGHT or piece.color != color:
                knights_developed += 1
                
        # Check bishop development  
        for start_square in starting_squares[color][chess.BISHOP]:
            piece = board.piece_at(start_square)
            if not piece or piece.piece_type != chess.BISHOP or piece.color != color:
                bishops_developed += 1
        
        # Bonus for developing knights first
        if knights_developed > 0 and bishops_developed == 0:
            score += 0.3  # Good development order
        elif bishops_developed > knights_developed:
            score -= 0.2  # Questionable development order
            
        return score

    def _evaluate_castle_timing(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate appropriate castle timing."""
        score = 0.0
        
        # Already handled in _castling_evaluation, but add specific timing bonuses
        development_factor = self._calculate_development_stage(board, color)
        
        # Check if castled
        king_square = board.king(color)
        if not king_square:
            return 0.0
            
        castled = self._check_if_castled(king_square, color) > 0
        
        if castled:
            # Bonus for castling at appropriate time
            if 0.4 <= development_factor <= 0.8:
                score += 0.5  # Good timing
            elif development_factor < 0.3:
                score += 0.2  # A bit early but okay
        else:
            # Penalty for not castling when appropriate
            if development_factor > 0.7:
                score -= 1.0  # Should have castled by now
                
        return score

    def _evaluate_central_pawn_development(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate central pawn development in opening."""
        score = 0.0
        
        # Check e4/e5 and d4/d5 pawn development
        center_files = [3, 4]  # D and E files (0-indexed: a=0, b=1, c=2, d=3, e=4, f=5, g=6, h=7)
        
        for file in center_files:
            # Check if central pawn has been moved forward
            starting_rank = 1 if color == chess.WHITE else 6
            advanced_rank = 3 if color == chess.WHITE else 4
            
            start_square = chess.square(file, starting_rank)
            advanced_square = chess.square(file, advanced_rank)
            
            start_piece = board.piece_at(start_square)
            advanced_piece = board.piece_at(advanced_square)
            
            if advanced_piece and advanced_piece.piece_type == chess.PAWN and advanced_piece.color == color:
                score += 0.4  # Good central pawn development
            elif not start_piece:
                # Pawn moved somewhere - check if it's still controlling center
                for rank in range(8):
                    square = chess.square(file, rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        # Found the pawn - check if it's controlling center
                        pawn_attacks = board.attacks(square)
                        center_control = any(sq in [chess.D4, chess.D5, chess.E4, chess.E5] for sq in pawn_attacks)
                        if center_control:
                            score += 0.2
                        break
                        
        return score

    def _evaluate_center_oriented_captures(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate if pieces are well-placed to capture toward center."""
        score = 0.0
        
        # Check if our pieces can recapture toward center on key squares
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        
        for center_square in center_squares:
            # Count how many of our pieces can recapture this square
            our_attackers = len(list(board.attackers(color, center_square)))
            if our_attackers > 0:
                score += our_attackers * 0.1  # Small bonus for center control
                
        return score

    def _evaluate_capture_pawn_structure(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate pawn structure quality, considering capture effects."""
        score = 0.0
        
        # This overlaps with existing pawn structure evaluation
        # Add specific bonuses for good capture-related pawn structure
        
        # Check for pawn majorities on each wing
        kingside_pawns = 0
        queenside_pawns = 0
        
        for file in range(8):
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    if file >= 4:  # Kingside
                        kingside_pawns += 1
                    else:  # Queenside
                        queenside_pawns += 1
        
        # Bonus for maintaining pawn structure integrity
        if kingside_pawns >= 3 and queenside_pawns >= 3:
            score += 0.3  # Balanced pawn structure
            
        return score