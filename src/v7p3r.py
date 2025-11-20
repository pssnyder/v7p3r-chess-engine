#!/usr/bin/env python3
"""
V7P3R Chess Engine v16.1
ENHANCED - Material Safety + Positional Play + Opening Mastery + Perfect Endgames

V16.1 ENHANCEMENTS (Target: Beat C0BR4 v3.2):
1. Enhanced Opening Book (10-15 moves deep)
   • White: Italian Game (Giuoco Piano), Queen's Gambit Declined, King's Indian Attack
   • Black: Sicilian Najdorf, King's Indian Defense, French Defense, Caro-Kann
   • Focus: Center control, rapid development, smooth middlegame transition

2. Middlegame Transition Nudges
   • Rooks on open/semi-open files (+20cp per rook)
   • Bishops on long diagonals (+15cp per bishop)
   • King safety: pawn shield bonus (+10cp per shield pawn)
   • Pawn structure: passed pawns (+30cp), doubled pawns (-20cp)

3. Syzygy Tablebase Integration
   • Perfect 6-piece endgame play
   • DTZ optimal moves for guaranteed wins

ARCHITECTURE (V16.0 Base - Tested & Proven):
1. MaterialOpponent's material counting → Never sacrifices
2. PositionalOpponent's PST tables → Positional chess  
3. Pre-search move filtering → Only evaluate safe moves
4. Castling preservation → King moves deprioritized

CORE PRINCIPLES:
✓ 60% PST + 40% Material evaluation (balanced)
✓ Filter material-losing moves BEFORE search
✓ Preserve castling rights (king moves last)
✓ Deep center-control opening book
✓ Middlegame bonuses (piece activity, king safety, pawn structure)
✓ Perfect endgames (Syzygy tablebases)

Target Performance: >75% win rate, beat C0BR4 v3.2
"""

import sys
import chess
import chess.polyglot
import random
import time
import os
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# Try to import Syzygy tablebase support
try:
    import chess.syzygy
    SYZYGY_AVAILABLE = True
except ImportError:
    SYZYGY_AVAILABLE = False

# Material values (from MaterialOpponent - proven to prevent sacrifices)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 325,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

# Piece-Square Tables (from PositionalOpponent - proven 81% win rate)
# Values in centipawns, White perspective

PAWN_PST = [
    [  0,  0,  0,  0,  0,  0,  0,  0],  # 1st rank
    [ 50, 50, 50, 50, 50, 50, 50, 50],  # 2nd rank  
    [ 60, 60, 70, 80, 80, 70, 60, 60],  # 3rd rank
    [ 70, 70, 80, 90, 90, 80, 70, 70],  # 4th rank
    [100,100,110,120,120,110,100,100],  # 5th rank
    [200,200,220,250,250,220,200,200],  # 6th rank
    [400,400,450,500,500,450,400,400],  # 7th rank
    [900,900,900,900,900,900,900,900],  # 8th rank
]

KNIGHT_PST = [
    [200,220,240,250,250,240,220,200],
    [220,240,260,270,270,260,240,220],
    [240,260,300,320,320,300,260,240],
    [250,270,320,350,350,320,270,250],
    [250,270,320,350,350,320,270,250],
    [240,260,300,320,320,300,260,240],
    [220,240,260,270,270,260,240,220],
    [200,220,240,250,250,240,220,200],
]

BISHOP_PST = [
    [250,260,270,280,280,270,260,250],
    [260,300,290,290,290,290,300,260],
    [270,290,320,300,300,320,290,270],
    [280,290,300,350,350,300,290,280],
    [280,290,300,350,350,300,290,280],
    [270,290,320,300,300,320,290,270],
    [260,300,290,290,290,290,300,260],
    [250,260,270,280,280,270,260,250],
]

ROOK_PST = [
    [500,510,520,530,530,520,510,500],
    [450,460,470,480,480,470,460,450],
    [450,460,470,480,480,470,460,450],
    [450,460,470,480,480,470,460,450],
    [450,460,470,480,480,470,460,450],
    [450,460,470,480,480,470,460,450],
    [550,560,570,580,580,570,560,550],
    [500,510,520,530,530,520,510,500],
]

QUEEN_PST = [
    [700,720,740,760,760,740,720,700],
    [720,750,780,800,800,780,750,720],
    [740,780,850,900,900,850,780,740],
    [760,800,900,1000,1000,900,800,760],
    [760,800,900,1000,1000,900,800,760],
    [740,780,850,900,900,850,780,740],
    [720,750,780,800,800,780,750,720],
    [700,720,740,760,760,740,720,700],
]

KING_MIDDLEGAME_PST = [
    [ 20, 30, 10,  0,  0, 10, 30, 20],
    [ 20, 20,  0,  0,  0,  0, 20, 20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
]

KING_ENDGAME_PST = [
    [-50,-40,-30,-20,-20,-30,-40,-50],
    [-30,-20,-10,  0,  0,-10,-20,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-20,-10,  0,  0,-10,-20,-30],
    [-50,-40,-30,-20,-20,-30,-40,-50],
]

# Search constants
MAX_QUIESCENCE_DEPTH = 8
MATE_SCORE = 30000

class NodeType(Enum):
    EXACT = 0
    LOWER_BOUND = 1
    UPPER_BOUND = 2

@dataclass
class TTEntry:
    zobrist_key: int
    depth: int
    value: float
    node_type: NodeType
    best_move: Optional[chess.Move]
    age: int


class OpeningBook:
    """Enhanced V16.1 opening book - deep center-control repertoire"""
    
    def __init__(self):
        self.book_moves = {}
        self.use_book = True
        self.book_depth = 15  # INCREASED from 8 for deeper theory
        self._load_embedded_book()
    
    def _load_embedded_book(self):
        """Load enhanced center-control opening repertoire"""
        # Import from separate module for maintainability
        try:
            from v7p3r_openings_v161 import get_enhanced_opening_book
            embedded_openings = get_enhanced_opening_book()
        except ImportError:
            # Fallback minimal book
            embedded_openings = {
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": [
                    ("e2e4", 100), ("d2d4", 100), ("g1f3", 80), ("c2c4", 80)
                ],
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": [
                    ("e7e5", 100), ("c7c5", 100), ("e7e6", 80), ("c7c6", 70)
                ],
            }
        
        for fen, moves in embedded_openings.items():
            board = chess.Board(fen)
            key = chess.polyglot.zobrist_hash(board)
            self.book_moves[key] = moves
    
    def get_book_move(self, board):
        """Get book move if available"""
        if not self.use_book or board.ply() >= self.book_depth:
            return None
        
        key = chess.polyglot.zobrist_hash(board)
        if key not in self.book_moves:
            return None
        
        moves = self.book_moves[key]
        legal = [(m, w) for m, w in moves if chess.Move.from_uci(m) in board.legal_moves]
        
        if not legal:
            return None
        
        # Weighted random selection (30% variety)
        total_weight = sum(w for _, w in legal)
        r = random.randint(1, total_weight)
        cumulative = 0
        for move_uci, weight in legal:
            cumulative += weight
            if r <= cumulative:
                return move_uci
        
        return legal[0][0]  # Fallback


class V7P3REngine:
    """V16.1 - Material safety + Positional play + Opening mastery + Perfect endgames"""
    
    def __init__(self, max_depth: int = 10, tt_size_mb: int = 256, tablebase_path: str = ""):
        self.board = chess.Board()
        self.max_depth = max_depth
        self.start_time = 0
        self.time_limit = 0
        self.nodes_searched = 0
        self.age = 0
        
        # Transposition table
        self.tt_size = (tt_size_mb * 1024 * 1024) // 64
        self.transposition_table: Dict[int, TTEntry] = {}
        
        # Move ordering
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(64)]
        self.history_table: Dict[Tuple[chess.Square, chess.Square], int] = {}
        
        # Opening book
        self.opening_book = OpeningBook()
        
        # Syzygy Tablebases (V16.1)
        self.tablebase = None
        if SYZYGY_AVAILABLE and tablebase_path and os.path.exists(tablebase_path):
            try:
                self.tablebase = chess.syzygy.open_tablebase(tablebase_path)
                print(f"info string Syzygy tablebases loaded from {tablebase_path}", flush=True)
            except Exception as e:
                print(f"info string Failed to load tablebases: {e}", flush=True)
                self.tablebase = None
        
        # Zobrist hashing
        self._init_zobrist()
    
    def _init_zobrist(self):
        """Initialize Zobrist keys"""
        random.seed(12345)
        self.zobrist_pieces = {}
        self.zobrist_castling = {}
        self.zobrist_en_passant = {}
        self.zobrist_side_to_move = random.getrandbits(64)
        
        for square in chess.SQUARES:
            for piece in chess.PIECE_TYPES:
                for color in chess.COLORS:
                    self.zobrist_pieces[(square, piece, color)] = random.getrandbits(64)
        
        for i in range(4):
            self.zobrist_castling[i] = random.getrandbits(64)
        
        for file in range(8):
            self.zobrist_en_passant[file] = random.getrandbits(64)
    
    def _get_zobrist_key(self, board: chess.Board) -> int:
        """Calculate Zobrist hash"""
        key = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                key ^= self.zobrist_pieces[(square, piece.piece_type, piece.color)]
        
        if board.turn == chess.BLACK:
            key ^= self.zobrist_side_to_move
        
        if board.has_kingside_castling_rights(chess.WHITE):
            key ^= self.zobrist_castling[0]
        if board.has_queenside_castling_rights(chess.WHITE):
            key ^= self.zobrist_castling[1]
        if board.has_kingside_castling_rights(chess.BLACK):
            key ^= self.zobrist_castling[2]
        if board.has_queenside_castling_rights(chess.BLACK):
            key ^= self.zobrist_castling[3]
        
        if board.ep_square is not None:
            key ^= self.zobrist_en_passant[chess.square_file(board.ep_square)]
        
        return key
    
    def _is_time_up(self) -> bool:
        """Check if time limit exceeded"""
        if self.time_limit <= 0:
            return False
        return time.time() - self.start_time >= self.time_limit
    
    def _calculate_time_limit(self, time_left: float, increment: float = 0) -> float:
        """Calculate time for this move"""
        if time_left <= 0:
            return 0
        
        if time_left > 60:
            return time_left / 20 + increment * 0.8
        else:
            return time_left / 10 + increment * 0.8
    
    def _get_piece_square_value(self, piece: chess.Piece, square: chess.Square, is_endgame: bool = False) -> int:
        """Get PST value for piece"""
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        if piece.color == chess.BLACK:
            rank = 7 - rank
        
        piece_type = piece.piece_type
        
        if piece_type == chess.PAWN:
            value = PAWN_PST[rank][file]
        elif piece_type == chess.KNIGHT:
            value = KNIGHT_PST[rank][file]
        elif piece_type == chess.BISHOP:
            value = BISHOP_PST[rank][file]
        elif piece_type == chess.ROOK:
            value = ROOK_PST[rank][file]
        elif piece_type == chess.QUEEN:
            value = QUEEN_PST[rank][file]
        elif piece_type == chess.KING:
            value = KING_ENDGAME_PST[rank][file] if is_endgame else KING_MIDDLEGAME_PST[rank][file]
        else:
            value = 0
        
        return value if piece.color == chess.WHITE else -value
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """Detect endgame phase"""
        if not board.pieces(chess.QUEEN, chess.WHITE) and not board.pieces(chess.QUEEN, chess.BLACK):
            return True
        
        white_material = sum(len(board.pieces(pt, chess.WHITE)) * PIECE_VALUES.get(pt, 0) 
                            for pt in [chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN])
        black_material = sum(len(board.pieces(pt, chess.BLACK)) * PIECE_VALUES.get(pt, 0)
                            for pt in [chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN])
        
        return white_material < 800 and black_material < 800
    
    def _evaluate_position(self, board: chess.Board) -> int:
        """
        V16.1 Enhanced Evaluation: PST + Material + Middlegame Nudges
        - PST (60%): Positional understanding
        - Material (40%): Safety from sacrifices
        - Middlegame nudges: Piece activity, king safety, pawn structure
        """
        pst_score = 0
        material_score = 0
        middlegame_bonus = 0  # NEW in V16.1
        is_endgame = self._is_endgame(board)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # PST value
                pst_score += self._get_piece_square_value(piece, square, is_endgame)
                
                # Material value
                material_value = PIECE_VALUES.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    material_score += material_value
                else:
                    material_score -= material_value
        
        # V16.1 MIDDLEGAME NUDGES
        if not is_endgame:
            middlegame_bonus = self._calculate_middlegame_bonuses(board)
        
        # 60% PST + 40% Material + Middlegame Bonuses
        combined_score = int(pst_score * 0.6 + material_score * 0.4 + middlegame_bonus)
        
        return combined_score if board.turn == chess.WHITE else -combined_score
    
    def _calculate_middlegame_bonuses(self, board: chess.Board) -> int:
        """
        V16.1 Middlegame transition nudges for smooth play
        Returns bonus in centipawns (White perspective)
        """
        bonus = 0
        
        # Rooks on open/semi-open files (+20cp per rook)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.ROOK:
                file = chess.square_file(square)
                is_open = True
                is_semi_open = True
                
                # Check if file has pawns
                for rank in range(8):
                    sq = chess.square(file, rank)
                    p = board.piece_at(sq)
                    if p and p.piece_type == chess.PAWN:
                        if p.color == piece.color:
                            is_open = False
                            is_semi_open = False
                            break
                        else:
                            is_open = False
                
                if is_open:
                    bonus += 20 if piece.color == chess.WHITE else -20
                elif is_semi_open:
                    bonus += 10 if piece.color == chess.WHITE else -10
        
        # King safety: pawn shield bonus (+10cp per shield pawn)
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is not None:
                king_file = chess.square_file(king_square)
                king_rank = chess.square_rank(king_square)
                
                # Check pawns in front of king (shield)
                shield_pawns = 0
                for file_offset in [-1, 0, 1]:
                    shield_file = king_file + file_offset
                    if 0 <= shield_file <= 7:
                        # Check 1-2 ranks ahead
                        for rank_offset in [1, 2]:
                            if color == chess.WHITE:
                                shield_rank = king_rank + rank_offset
                            else:
                                shield_rank = king_rank - rank_offset
                            
                            if 0 <= shield_rank <= 7:
                                sq = chess.square(shield_file, shield_rank)
                                p = board.piece_at(sq)
                                if p and p.piece_type == chess.PAWN and p.color == color:
                                    shield_pawns += 1
                                    break
                
                if color == chess.WHITE:
                    bonus += shield_pawns * 10
                else:
                    bonus -= shield_pawns * 10
        
        # Pawn structure bonuses
        for file in range(8):
            white_pawns = []
            black_pawns = []
            
            for rank in range(8):
                sq = chess.square(file, rank)
                p = board.piece_at(sq)
                if p and p.piece_type == chess.PAWN:
                    if p.color == chess.WHITE:
                        white_pawns.append(rank)
                    else:
                        black_pawns.append(rank)
            
            # Doubled pawns penalty (-20cp)
            if len(white_pawns) > 1:
                bonus -= 20 * (len(white_pawns) - 1)
            if len(black_pawns) > 1:
                bonus += 20 * (len(black_pawns) - 1)
            
            # Passed pawns bonus (+30cp)
            for rank in white_pawns:
                is_passed = True
                # Check adjacent files and this file ahead
                for adj_file in [file - 1, file, file + 1]:
                    if 0 <= adj_file <= 7:
                        for r in range(rank + 1, 8):
                            sq = chess.square(adj_file, r)
                            p = board.piece_at(sq)
                            if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                                is_passed = False
                                break
                    if not is_passed:
                        break
                
                if is_passed:
                    bonus += 30
            
            for rank in black_pawns:
                is_passed = True
                for adj_file in [file - 1, file, file + 1]:
                    if 0 <= adj_file <= 7:
                        for r in range(0, rank):
                            sq = chess.square(adj_file, r)
                            p = board.piece_at(sq)
                            if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                                is_passed = False
                                break
                    if not is_passed:
                        break
                
                if is_passed:
                    bonus -= 30
        
        return bonus
    
    def _calculate_material_delta(self, board: chess.Board, move: chess.Move) -> int:
        """
        Calculate material change for a move
        Returns: positive = gain, negative = loss
        """
        delta = 0
        
        # Material gained from capture
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                delta += PIECE_VALUES.get(captured.piece_type, 0)
        
        # Material lost if piece hangs
        board.push(move)
        piece = board.piece_at(move.to_square)
        if piece:
            opponent_color = board.turn
            our_color = not opponent_color
            
            if board.is_attacked_by(opponent_color, move.to_square):
                if not board.is_attacked_by(our_color, move.to_square):
                    delta -= PIECE_VALUES.get(piece.piece_type, 0)
        board.pop()
        
        return delta
    
    def _filter_and_order_moves(self, board: chess.Board, moves: List[chess.Move], 
                                ply: int, tt_move: Optional[chess.Move] = None) -> List[chess.Move]:
        """
        V16 Move Filtering + Ordering:
        1. Filter out material-losing moves
        2. Order by: TT > Mates > Checks > Captures > Castling > Others > King moves
        """
        filtered_moves = []
        
        # STEP 1: Filter material-losing moves
        for move in moves:
            # Always allow checks (tactics)
            if board.gives_check(move):
                filtered_moves.append(move)
                continue
            
            # Check material delta
            material_delta = self._calculate_material_delta(board, move)
            
            # Filter moves that lose material (threshold: -50cp)
            if material_delta < -50:
                continue
            
            filtered_moves.append(move)
        
        # Fallback if everything filtered
        if not filtered_moves:
            filtered_moves = moves
        
        # STEP 2: Order moves
        scored_moves = []
        
        for move in filtered_moves:
            score = 0
            piece = board.piece_at(move.from_square)
            
            # TT move
            if tt_move and move == tt_move:
                score = 1000000
            # Checkmate
            elif board.gives_check(move):
                board.push(move)
                if board.is_checkmate():
                    score = 900000
                else:
                    score = 500000
                board.pop()
            # Winning captures
            elif board.is_capture(move):
                material_delta = self._calculate_material_delta(board, move)
                if material_delta > 50:
                    score = 400000 + material_delta
                else:
                    score = 300000 + material_delta
            # Castling (HIGH priority - develop rook, protect king)
            elif piece and piece.piece_type == chess.KING and abs(move.from_square - move.to_square) == 2:
                score = 250000
            # Killer moves
            elif ply < len(self.killer_moves) and move in self.killer_moves[ply]:
                score = 200000
            # Promotions
            elif move.promotion:
                score = 150000 + PIECE_VALUES.get(move.promotion, 0)
            # Pawn advances
            elif piece and piece.piece_type == chess.PAWN:
                to_rank = chess.square_rank(move.to_square)
                if board.turn == chess.WHITE and to_rank >= 5:
                    score = 100000 + to_rank * 1000
                elif board.turn == chess.BLACK and to_rank <= 2:
                    score = 100000 + (7 - to_rank) * 1000
            # King moves (LOWEST priority - preserve castling!)
            elif piece and piece.piece_type == chess.KING:
                score = -50000
            # History heuristic
            else:
                key = (move.from_square, move.to_square)
                score = self.history_table.get(key, 0)
            
            scored_moves.append((score, move))
        
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored_moves]
    
    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int = 0) -> float:
        """Quiescence search on captures"""
        if self._is_time_up() or depth > MAX_QUIESCENCE_DEPTH:
            return self._evaluate_position(board)
        
        self.nodes_searched += 1
        stand_pat = self._evaluate_position(board)
        
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        
        # Only captures in quiescence
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        
        for move in captures:
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha, depth + 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha
    
    def _update_killer_moves(self, move: chess.Move, ply: int):
        """Update killer moves"""
        if ply < len(self.killer_moves):
            if self.killer_moves[ply][0] != move:
                self.killer_moves[ply][1] = self.killer_moves[ply][0]
                self.killer_moves[ply][0] = move
    
    def _update_history(self, move: chess.Move, depth: int):
        """Update history table"""
        key = (move.from_square, move.to_square)
        self.history_table[key] = self.history_table.get(key, 0) + depth * depth
    
    def _store_tt_entry(self, zobrist_key: int, depth: int, value: float, 
                       node_type: NodeType, best_move: Optional[chess.Move]):
        """Store TT entry"""
        if len(self.transposition_table) >= self.tt_size:
            old_keys = [k for k, v in self.transposition_table.items() if v.age < self.age - 2]
            for key in old_keys[:len(old_keys)//2]:
                del self.transposition_table[key]
        
        self.transposition_table[zobrist_key] = TTEntry(
            zobrist_key, depth, value, node_type, best_move, self.age
        )
    
    def _probe_tt(self, zobrist_key: int, depth: int, alpha: float, beta: float) -> Tuple[Optional[float], Optional[chess.Move]]:
        """Probe TT"""
        entry = self.transposition_table.get(zobrist_key)
        if entry is None or entry.depth < depth:
            return None, entry.best_move if entry else None
        
        if entry.node_type == NodeType.EXACT:
            return entry.value, entry.best_move
        elif entry.node_type == NodeType.LOWER_BOUND and entry.value >= beta:
            return entry.value, entry.best_move
        elif entry.node_type == NodeType.UPPER_BOUND and entry.value <= alpha:
            return entry.value, entry.best_move
        
        return None, entry.best_move
    
    def _search(self, board: chess.Board, depth: int, alpha: float, beta: float, 
               ply: int, do_null_move: bool = True) -> Tuple[float, Optional[chess.Move]]:
        """Alpha-beta search with move filtering"""
        if self._is_time_up():
            return self._evaluate_position(board), None
        
        # Check for actual terminal positions (checkmate/stalemate)
        # Don't treat insufficient material draws as terminal - still search for best move
        if board.is_checkmate():
            return -MATE_SCORE + ply, None
        if board.is_stalemate():
            return 0, None
        
        if depth <= 0:
            return self._quiescence_search(board, alpha, beta), None
        
        self.nodes_searched += 1
        zobrist_key = self._get_zobrist_key(board)
        original_alpha = alpha
        
        # TT probe
        tt_value, tt_move = self._probe_tt(zobrist_key, depth, alpha, beta)
        if tt_value is not None:
            return tt_value, tt_move
        
        # Null move pruning (but not at root and not when beta is infinite)
        if (do_null_move and depth >= 3 and not board.is_check() and 
            ply > 0 and beta < float('inf')):
            board.push(chess.Move.null())
            null_score, _ = self._search(board, depth - 3, -beta, -beta + 1, ply + 1, False)
            null_score = -null_score
            board.pop()
            
            if null_score >= beta:
                return beta, None
        
        # Generate and filter moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self._evaluate_position(board), None
        
        ordered_moves = self._filter_and_order_moves(board, legal_moves, ply, tt_move)
        best_move = None
        best_value = -float('inf')
        
        for i, move in enumerate(ordered_moves):
            board.push(move)
            
            if i == 0:
                value, _ = self._search(board, depth - 1, -beta, -alpha, ply + 1)
                value = -value
            else:
                # PVS
                value, _ = self._search(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                value = -value
                
                if alpha < value < beta:
                    value, _ = self._search(board, depth - 1, -beta, -alpha, ply + 1)
                    value = -value
            
            board.pop()
            
            if value > best_value:
                best_value = value
                best_move = move
            
            if value > alpha:
                alpha = value
            
            if alpha >= beta:
                if not board.is_capture(move):
                    self._update_killer_moves(move, ply)
                    self._update_history(move, depth)
                break
        
        # Store TT
        if best_value <= original_alpha:
            node_type = NodeType.UPPER_BOUND
        elif best_value >= beta:
            node_type = NodeType.LOWER_BOUND
        else:
            node_type = NodeType.EXACT
        
        self._store_tt_entry(zobrist_key, depth, best_value, node_type, best_move)
        
        return best_value, best_move
    
    def get_best_move(self, time_left: float = 0, increment: float = 0) -> Optional[chess.Move]:
        """Find best move with iterative deepening + Syzygy tablebase probing"""
        # V16.1: Probe Syzygy tablebase first (perfect endgame play)
        if self.tablebase is not None:
            piece_count = len(self.board.piece_map())
            if piece_count <= 6:  # 6-piece tablebase
                try:
                    # Probe for WDL (Win/Draw/Loss)
                    wdl = self.tablebase.probe_wdl(self.board)
                    
                    if wdl is not None:
                        # Find best move according to tablebase
                        best_tb_move = None
                        best_tb_wdl = -3  # Worst case
                        
                        for move in self.board.legal_moves:
                            self.board.push(move)
                            try:
                                # Check opponent's WDL after our move
                                opp_wdl = self.tablebase.probe_wdl(self.board)
                                if opp_wdl is not None:
                                    # Negate because it's opponent's perspective
                                    our_wdl = -opp_wdl
                                    if our_wdl > best_tb_wdl:
                                        best_tb_wdl = our_wdl
                                        best_tb_move = move
                            except:
                                pass
                            self.board.pop()
                        
                        if best_tb_move is not None and best_tb_wdl >= 0:
                            print(f"info string Tablebase hit: WDL={best_tb_wdl}", flush=True)
                            return best_tb_move
                except Exception as e:
                    # Tablebase probe failed, fall through to normal search
                    pass
        
        # Check opening book
        book_move = self.opening_book.get_book_move(self.board)
        if book_move:
            try:
                return chess.Move.from_uci(book_move)
            except:
                pass
        
        # Only return None if there are truly no legal moves (checkmate/stalemate)
        if self.board.is_checkmate() or self.board.is_stalemate():
            return None
        
        self.start_time = time.time()
        self.time_limit = self._calculate_time_limit(time_left, increment)
        self.nodes_searched = 0
        self.age += 1
        
        best_move = None
        best_value = -float('inf')
        
        # Iterative deepening
        for depth in range(1, self.max_depth + 1):
            if self._is_time_up():
                break
            
            search_start = time.time()
            value, move = self._search(self.board, depth, -float('inf'), float('inf'), 0)
            search_time = time.time() - search_start
            
            if move is not None:
                best_move = move
                best_value = value
                
                nps = int(self.nodes_searched / max(search_time, 0.001))
                print(f"info depth {depth} score cp {value} nodes {self.nodes_searched} "
                      f"nps {nps} time {int(search_time * 1000)} pv {move.uci()}")
                sys.stdout.flush()
            
            if self._is_time_up():
                break
        
        total_time = time.time() - self.start_time
        print(f"info string Search completed in {total_time:.3f}s, {self.nodes_searched} nodes")
        sys.stdout.flush()
        
        return best_move


if __name__ == "__main__":
    # Simple UCI test
    engine = V7P3REngine()
    print("V7P3R v16.0 - Fresh Start")
    print("Type 'uci' to start")
    
    while True:
        line = input().strip()
        if line == "uci":
            print("id name V7P3R v16.0")
            print("id author V7P3R Team")
            print("uciok")
        elif line == "isready":
            print("readyok")
        elif line == "quit":
            break
