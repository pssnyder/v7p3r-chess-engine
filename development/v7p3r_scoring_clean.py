# v7p3r_scoring_clean.py

""" V7P3R Scoring Calculation Module - Clean Version
This module implements the scoring calculation for the V7P3R Chess Engine.
Streamlined to remove duplicates and improve performance.
"""

import chess

class V7P3RScoringCalculation:
    """Clean, optimized scoring calculation for V7P3R Chess Engine."""
    
    def __init__(self, piece_values):
        self.piece_values = piece_values

    def calculate_score_optimized(self, board: chess.Board, color: chess.Color, endgame_factor: float = 0.0) -> float:
        """
        Optimized scoring with early exit thresholds for performance.
        """
        score = 0.0
        
        # Phase 1: Critical threats and material (always calculate)
        checkmate_score = self._checkmate_threats(board, color)
        score += checkmate_score
        
        # Early exit 1: Checkmate threats found
        if abs(checkmate_score) > 5000:
            return score
            
        king_safety = self._king_safety(board, color)
        score += king_safety
        king_threat = self._king_threat(board, color)  
        score += king_threat
        queen_safety = self._queen_safety(board, color)
        score += queen_safety
        
        # Early exit 2: King is in severe danger
        if (king_safety + king_threat) < -300:
            return score + self._material_score(board, color)
            
        material_score = self._material_score(board, color)
        score += material_score
        draw_score = self._draw_scenarios(board)
        score += draw_score
        
        # Early exit 3: Massive material advantage
        if abs(material_score) > 1500:
            return score
            
        # Phase 2: Basic tactical patterns (only if game is close)
        if abs(score) < 300:
            # Keep only lightweight tactical evaluation
            score += self._lightweight_tactics(board, color)
                
        # Phase 3: Basic positional factors (only in balanced positions)  
        if abs(score) < 150:
            score += self._piece_coordination(board, color)
            score += self._center_control(board, color)
            
        # Phase 4: Detailed evaluation (only in very close games)
        if abs(score) < 50:
            score += self._basic_pawn_structure(board, color)
            score += self._pawn_majority(board, color)
            score += self._bishop_pair(board, color) 
            score += self._knight_pair(board, color)
            score += self._rook_coordination(board, color)
            score += self._castling_evaluation(board, color)
            
        # Phase 5: Fine-grained evaluation (only if still very close)
        if abs(score) < 25:
            score += self._mobility_score(board, color)
            score += self._open_files(board, color)
            
        # Always include endgame logic if it's endgame
        if self._is_endgame(board):
            score += self._endgame_logic(board, color)
            
        return score

    # ==========================================
    # ========= CORE SCORING FUNCTIONS ========
    # ==========================================

    def _checkmate_threats(self, board: chess.Board, color: chess.Color) -> float:
        """Assess if 'color' can deliver a checkmate on their next move."""
        score = 0.0
        board_copy = board.copy()
        board_copy.turn = color
        
        for move in list(board_copy.legal_moves)[:10]:  # Limit check for performance
            board_copy.push(move)
            if board_copy.is_checkmate():
                score += 9999.0
                board_copy.pop()
                return score
            board_copy.pop()
        return score

    def _draw_scenarios(self, board: chess.Board) -> float:
        """Check for draw scenarios."""
        if (board.is_stalemate() or board.is_insufficient_material() or 
            board.is_fivefold_repetition() or board.is_repetition(count=2)):
            return -9999.0
        return 0.0

    def _material_score(self, board: chess.Board, color: chess.Color) -> float:
        """Simple material count for given color."""
        score = 0.0
        piece_type_map = {
            'pawn': chess.PAWN, 'knight': chess.KNIGHT, 'bishop': chess.BISHOP,
            'rook': chess.ROOK, 'queen': chess.QUEEN, 'king': chess.KING
        }
        
        for piece_name, value in self.piece_values.items():
            if piece_name in piece_type_map:
                piece_type = piece_type_map[piece_name]
                score += len(board.pieces(piece_type, color)) * value
        return score

    def _king_safety(self, board: chess.Board, color: chess.Color) -> float:
        """Basic king safety evaluation."""
        score = 0.0
        king_square = board.king(color)
        if not king_square:
            return score

        # Check for pawn shield
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)

        shield_ranks = []
        if color == chess.WHITE:
            if king_rank < 7: shield_ranks.append(king_rank + 1)
            if king_rank < 6: shield_ranks.append(king_rank + 2)
        else:
            if king_rank > 0: shield_ranks.append(king_rank - 1)
            if king_rank > 1: shield_ranks.append(king_rank - 2)

        for rank_offset in shield_ranks:
            for file_offset in [-1, 0, 1]:
                target_file = king_file + file_offset
                if 0 <= target_file <= 7 and 0 <= rank_offset <= 7:
                    shield_square = chess.square(target_file, rank_offset)
                    piece = board.piece_at(shield_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        score += 0.1

        return score

    def _king_threat(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate if the opponent's king is under threat."""
        if board.is_check():
            if board.turn != color:
                return -9.0  # We are in check
            else:
                return 9.0   # We just gave check
        return 0.0

    def _queen_safety(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate queen safety to prevent blunders."""
        queen_squares = list(board.pieces(chess.QUEEN, color))
        if not queen_squares:
            return 0.0
        
        queen_square = queen_squares[0]
        if board.is_attacked_by(not color, queen_square):
            return -900.0  # Heavy penalty for exposed queen
        return 0.0

    def _lightweight_tactics(self, board: chess.Board, color: chess.Color) -> float:
        """Lightweight tactical evaluation - captures and hanging pieces only."""
        score = 0.0
        
        # Count captures
        for move in board.legal_moves:
            if board.is_capture(move):
                piece_making_capture = board.piece_at(move.from_square)
                if piece_making_capture and piece_making_capture.color == color:
                    score += 1.0

        # Check hanging pieces efficiently
        opponent_color = not color
        
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            for square in board.pieces(piece_type, opponent_color):
                if board.is_attacked_by(color, square) and not board.is_attacked_by(opponent_color, square):
                    score += self.piece_values.get(piece_type, 0) * 0.1
                    
            for square in board.pieces(piece_type, color):
                if board.is_attacked_by(opponent_color, square) and not board.is_attacked_by(color, square):
                    score -= self.piece_values.get(piece_type, 0) * 0.1

        return score

    def _piece_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate piece defense coordination."""
        score = 0.0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece_type, color):
                if board.is_attacked_by(color, square): 
                    score += 0.5
        return score

    def _center_control(self, board: chess.Board, color: chess.Color) -> float:
        """Simple center control."""
        score = 0.0
        center = [chess.D4, chess.D5, chess.E4, chess.E5]
        for square in center:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                score += 0.5
        return score

    def _basic_pawn_structure(self, board: chess.Board, color: chess.Color) -> float:
        """Basic pawn structure evaluation - doubled and isolated pawns."""
        score = 0.0
        
        # Group pawns by file
        pawns_by_file = {}
        for pawn_square in board.pieces(chess.PAWN, color):
            file = chess.square_file(pawn_square)
            pawns_by_file[file] = pawns_by_file.get(file, 0) + 1
        
        # Penalty for doubled pawns
        for file, count in pawns_by_file.items():
            if count > 1:
                score -= 0.5 * (count - 1)
        
        # Penalty for isolated pawns
        for file in pawns_by_file:
            has_support = False
            for adjacent_file in [file - 1, file + 1]:
                if adjacent_file in pawns_by_file:
                    has_support = True
                    break
            if not has_support:
                score -= 0.5
                
        return score

    def _pawn_majority(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate pawn majority on queenside/kingside."""
        white_kingside = len([p for p in board.pieces(chess.PAWN, chess.WHITE) if chess.square_file(p) >= 4])
        white_queenside = len([p for p in board.pieces(chess.PAWN, chess.WHITE) if chess.square_file(p) < 4])
        black_kingside = len([p for p in board.pieces(chess.PAWN, chess.BLACK) if chess.square_file(p) >= 4])
        black_queenside = len([p for p in board.pieces(chess.PAWN, chess.BLACK) if chess.square_file(p) < 4])
        
        score = 0.0
        if color == chess.WHITE:
            if white_kingside > black_kingside: score += 0.25
            if white_queenside > black_queenside: score += 0.25
            if white_kingside < black_kingside: score -= 0.25
            if white_queenside < black_queenside: score -= 0.25
        else:
            if black_kingside > white_kingside: score += 0.25
            if black_queenside > white_queenside: score += 0.25
            if black_kingside < white_kingside: score -= 0.25
            if black_queenside < white_queenside: score -= 0.25

        return score

    def _bishop_pair(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate bishop pair bonus."""
        bishops = list(board.pieces(chess.BISHOP, color))
        return 0.5 if len(bishops) >= 2 else 0.0

    def _knight_pair(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate knight pair bonus."""
        knights = list(board.pieces(chess.KNIGHT, color))
        return 0.5 if len(knights) >= 2 else 0.0

    def _rook_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate bonus for rook coordination."""
        score = 0.0
        rooks = list(board.pieces(chess.ROOK, color))

        for i in range(len(rooks)):
            for j in range(i+1, len(rooks)):
                sq1, sq2 = rooks[i], rooks[j]
                if chess.square_file(sq1) == chess.square_file(sq2):
                    score += 0.5
                if chess.square_rank(sq1) == chess.square_rank(sq2):
                    score += 0.5
        return score

    def _castling_evaluation(self, board: chess.Board, color: chess.Color) -> float:
        """Basic castling evaluation."""
        score = 0.0
        king_square = board.king(color)
        if not king_square:
            return 0.0
            
        # Check if already castled
        if color == chess.WHITE:
            if king_square in [chess.G1, chess.C1]:
                score += 1.5
        else:
            if king_square in [chess.G8, chess.C8]:
                score += 1.5
        
        # Bonus for having castling rights
        if board.has_kingside_castling_rights(color) or board.has_queenside_castling_rights(color):
            score += 0.5
            
        return score

    def _mobility_score(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate mobility of pieces."""
        score = 0.0
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece_type, color):
                score += len(list(board.attacks(square))) * 0.05
        return score

    def _open_files(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate open files for rooks."""
        score = 0.0
        
        # Pre-calculate pawn positions by file
        our_pawns_by_file = set()
        enemy_pawns_by_file = set()
        
        for pawn_square in board.pieces(chess.PAWN, color):
            our_pawns_by_file.add(chess.square_file(pawn_square))
            
        for pawn_square in board.pieces(chess.PAWN, not color):
            enemy_pawns_by_file.add(chess.square_file(pawn_square))
        
        for file in range(8):
            has_our_pawn = file in our_pawns_by_file
            has_enemy_pawn = file in enemy_pawns_by_file
            
            if not has_our_pawn and not has_enemy_pawn:
                score += 0.5  # Open file
            elif not has_our_pawn and has_enemy_pawn:
                score += 0.25  # Semi-open file

        return score

    def _is_endgame(self, board: chess.Board) -> bool:
        """Simple endgame detection."""
        total_pieces = 0
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            total_pieces += len(board.pieces(piece_type, chess.WHITE))
            total_pieces += len(board.pieces(piece_type, chess.BLACK))
        return total_pieces <= 6

    def _endgame_logic(self, board: chess.Board, color: chess.Color) -> float:
        """Basic endgame evaluation."""
        if not self._is_endgame(board):
            return 0.0
            
        score = 0.0
        king_square = board.king(color)
        
        if king_square:
            # King centralization bonus
            file, rank = chess.square_file(king_square), chess.square_rank(king_square)
            center_distance = min(
                abs(file - 3) + abs(rank - 3),
                abs(file - 3) + abs(rank - 4),
                abs(file - 4) + abs(rank - 3),
                abs(file - 4) + abs(rank - 4)
            )
            score += max(0, 1.0 - (center_distance * 0.15))
        
        return score
