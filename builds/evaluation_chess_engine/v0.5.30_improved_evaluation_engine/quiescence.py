
# improved_quiescence.py
# Adding quiescence search to avoid horizon effect

import chess
from improved_piece_square_tables import PieceSquareTables

class ImprovedEvaluation:
    def __init__(self, board=None):
        self.board = board if board else chess.Board()

        # Material values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0  # King has no material value
        }

        # Initialize piece-square tables
        self.psqt = PieceSquareTables()

        # Cache for positions
        self._position_cache = {}

    def evaluate(self):
        """Evaluate the current position from white's perspective"""
        # Use cache for performance
        fen = self.board.fen()
        if fen in self._position_cache:
            return self._position_cache[fen]

        # Handle checkmate/stalemate
        if self.board.is_checkmate():
            # Return large negative/positive score for checkmate
            return -10000 if self.board.turn == chess.WHITE else 10000

        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0  # Draw

        # Calculate the main evaluation components
        material_score = self._evaluate_material()
        position_score = self._evaluate_position()
        development_score = self._evaluate_development()
        king_safety_score = self._evaluate_king_safety()
        pawn_structure_score = self._evaluate_pawn_structure()
        mobility_score = self._evaluate_mobility()

        # Final score is the sum of all components
        total_score = (
            material_score +
            position_score +
            development_score +
            king_safety_score +
            pawn_structure_score +
            mobility_score
        )

        # Store in cache
        self._position_cache[fen] = total_score

        # Return from white's perspective
        return total_score

    def _evaluate_material(self):
        """Calculate material balance"""
        white_material = sum(len(self.board.pieces(pt, chess.WHITE)) * self.piece_values[pt] 
                            for pt in self.piece_values)
        black_material = sum(len(self.board.pieces(pt, chess.BLACK)) * self.piece_values[pt] 
                            for pt in self.piece_values)

        return white_material - black_material

    def _evaluate_position(self):
        """Evaluate piece positions using piece-square tables"""
        # Calculate endgame factor (0 for opening, 1 for endgame)
        total_material = sum(len(self.board.pieces(pt, chess.WHITE)) * self.piece_values[pt] 
                            for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
        total_material += sum(len(self.board.pieces(pt, chess.BLACK)) * self.piece_values[pt] 
                            for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])

        # Max material (excluding kings/pawns): 2*(3+3+5+9)*100 = 4000
        max_material = 4000
        endgame_factor = 1.0 - min(1.0, total_material / max_material)

        score = 0
        # For each piece type and color
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            # White pieces
            for square in self.board.pieces(piece_type, chess.WHITE):
                piece = self.board.piece_at(square)
                score += self.psqt.get_piece_value(piece, square, endgame_factor)

            # Black pieces
            for square in self.board.pieces(piece_type, chess.BLACK):
                piece = self.board.piece_at(square)
                score -= self.psqt.get_piece_value(piece, square, endgame_factor)

        return score

    def _evaluate_development(self):
        """Evaluate piece development and opening principles"""
        score = 0

        # Opening phase detection
        is_opening = True
        move_count = len(self.board.move_stack)
        if move_count > 20:  # Roughly 10 moves for each side
            is_opening = False

        if is_opening:
            # Development bonus for minor pieces moved from initial position
            for color in [chess.WHITE, chess.BLACK]:
                sign = 1 if color == chess.WHITE else -1

                # Knight development
                knight_squares = [chess.B1, chess.G1] if color == chess.WHITE else [chess.B8, chess.G8]
                for square in knight_squares:
                    if not self.board.piece_at(square) or self.board.piece_at(square).piece_type != chess.KNIGHT:
                        score += 10 * sign  # Knight has moved from initial square

                # Bishop development
                bishop_squares = [chess.C1, chess.F1] if color == chess.WHITE else [chess.C8, chess.F8]
                for square in bishop_squares:
                    if not self.board.piece_at(square) or self.board.piece_at(square).piece_type != chess.BISHOP:
                        score += 10 * sign  # Bishop has moved from initial square

                # Check if castled
                if self._has_castled(color):
                    score += 40 * sign  # Big bonus for castling

                # Center control
                center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
                for square in center_squares:
                    if self.board.is_attacked_by(color, square):
                        score += 5 * sign  # Bonus for attacking center
                        # Extra bonus for occupying center with pawn
                        piece = self.board.piece_at(square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            score += 10 * sign

        return score

    def _has_castled(self, color):
        """Check if a side has castled"""
        # Simple heuristic - king not on starting square and rook moved
        king_square = self.board.king(color)
        if color == chess.WHITE:
            return king_square in [chess.G1, chess.C1]
        else:
            return king_square in [chess.G8, chess.C8]

    def _evaluate_king_safety(self):
        """Evaluate king safety"""
        score = 0

        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            king_square = self.board.king(color)

            # Pawn shield - check for pawns in front of king
            pawn_shield_count = 0
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)

            # Define the rank in front of the king based on color
            shield_rank = king_rank + (1 if color == chess.WHITE else -1)

            # Check if shield rank is valid
            if 0 <= shield_rank <= 7:
                # Check three files (king's file and adjacent files)
                for file_offset in [-1, 0, 1]:
                    shield_file = king_file + file_offset

                    # Check if shield file is valid
                    if 0 <= shield_file <= 7:
                        shield_square = chess.square(shield_file, shield_rank)
                        piece = self.board.piece_at(shield_square)

                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            pawn_shield_count += 1

            # Bonus for each pawn in shield
            score += 10 * sign * pawn_shield_count

            # Penalty for checks
            if self.board.is_check() and self.board.turn == color:
                score -= 50 * sign

            # Penalty for open lines to king
            open_lines = 0
            for direction in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                dx, dy = direction
                curr_x, curr_y = chess.square_file(king_square), chess.square_rank(king_square)

                # Check in each direction
                for distance in range(1, 8):
                    new_x, new_y = curr_x + dx * distance, curr_y + dy * distance

                    # Check if new square is on the board
                    if 0 <= new_x <= 7 and 0 <= new_y <= 7:
                        new_square = chess.square(new_x, new_y)
                        piece = self.board.piece_at(new_square)

                        if piece:
                            if piece.color != color and (
                                (piece.piece_type == chess.QUEEN) or
                                (piece.piece_type == chess.ROOK and dx * dy == 0) or
                                (piece.piece_type == chess.BISHOP and dx * dy != 0)
                            ):
                                open_lines += 1
                            break

            # Penalty for open lines to king
            score -= 15 * sign * open_lines

        return score

    def _evaluate_pawn_structure(self):
        """Evaluate pawn structure"""
        score = 0

        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1

            # Count isolated pawns
            isolated_count = 0
            # Count doubled pawns
            doubled_count = 0
            # Count passed pawns
            passed_count = 0

            for file in range(8):
                pawns_in_file = 0
                for rank in range(8):
                    square = chess.square(file, rank)
                    piece = self.board.piece_at(square)

                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        pawns_in_file += 1

                        # Check if isolated
                        is_isolated = True
                        for adj_file in [file - 1, file + 1]:
                            if 0 <= adj_file <= 7:
                                for r in range(8):
                                    adj_square = chess.square(adj_file, r)
                                    adj_piece = self.board.piece_at(adj_square)
                                    if adj_piece and adj_piece.piece_type == chess.PAWN and adj_piece.color == color:
                                        is_isolated = False
                                        break

                        if is_isolated:
                            isolated_count += 1

                        # Check if passed
                        if self._is_passed_pawn(square, color):
                            passed_count += 1

                # Count doubled pawns in file
                if pawns_in_file > 1:
                    doubled_count += pawns_in_file - 1

            # Apply penalties and bonuses
            score -= 20 * sign * isolated_count  # Penalty for isolated pawns
            score -= 15 * sign * doubled_count   # Penalty for doubled pawns
            score += 30 * sign * passed_count    # Bonus for passed pawns

        return score

    def _is_passed_pawn(self, square, color):
        """Check if a pawn is passed"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        # Direction of pawn movement
        direction = 1 if color == chess.WHITE else -1

        # Check for opposing pawns that can block or capture
        for check_file in [file - 1, file, file + 1]:
            if 0 <= check_file <= 7:
                # Check all ranks in front of the pawn
                current_rank = rank + direction
                while 0 <= current_rank <= 7:
                    check_square = chess.square(check_file, current_rank)
                    piece = self.board.piece_at(check_square)

                    if piece and piece.piece_type == chess.PAWN and piece.color != color:
                        return False  # Not passed

                    current_rank += direction

        return True  # Passed

    def _evaluate_mobility(self):
        """Evaluate piece mobility"""
        score = 0

        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1

            # Save current turn
            original_turn = self.board.turn

            # Set board turn to calculate mobility
            self.board.turn = color

            # Get legal moves
            legal_moves = list(self.board.legal_moves)

            # Restore original turn
            self.board.turn = original_turn

            # Basic mobility score
            score += 2 * sign * len(legal_moves)

            # More detailed mobility by piece type
            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                for square in self.board.pieces(piece_type, color):
                    # Count attacks for this piece
                    attacks = len(list(self.board.attacks(square)))

                    # Weight by piece type
                    if piece_type == chess.KNIGHT:
                        score += 4 * sign * attacks
                    elif piece_type == chess.BISHOP:
                        score += 5 * sign * attacks
                    elif piece_type == chess.ROOK:
                        score += 2 * sign * attacks
                    elif piece_type == chess.QUEEN:
                        score += 1 * sign * attacks

        return score


def quiescence_search(board, alpha, beta, depth, evaluation):
    """
    Quiescence search to avoid horizon effect

    Args:
        board: Current board position
        alpha: Alpha bound
        beta: Beta bound
        depth: Current depth (negative for quiescence search)
        evaluation: Evaluation function object

    Returns:
        Evaluation score from white's perspective
    """
    # Stand-pat score
    evaluation.board = board
    stand_pat = evaluation.evaluate()

    # Return immediately on checkmate/stalemate
    if board.is_checkmate():
        return -10000 if board.turn == chess.WHITE else 10000

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    # Limit quiescence search depth
    if depth <= -4:
        return stand_pat

    # Fail-hard beta cutoff
    if stand_pat >= beta:
        return beta

    # Update alpha if stand-pat is better
    if stand_pat > alpha:
        alpha = stand_pat

    # Consider only captures and promotions for quiescence
    for move in board.legal_moves:
        if board.is_capture(move) or move.promotion:
            # Make the move
            board.push(move)

            # Call recursively
            score = -quiescence_search(board, -beta, -alpha, depth - 1, evaluation)

            # Unmake the move
            board.pop()

            # Fail-hard beta cutoff
            if score >= beta:
                return beta

            # Update alpha if this move is better
            if score > alpha:
                alpha = score

    return alpha
