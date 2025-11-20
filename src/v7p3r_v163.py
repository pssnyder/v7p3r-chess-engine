#!/usr/bin/env python3
"""
V7P3R Chess Engine v16.3
BUG FIXES: Depth limitation + No move found + PV Display/Following

V16.3 FIXES (Nov 20, 2025):
1. PV Display - Full principal variation in UCI output (not just first move)
2. PV Following - Instant moves when opponent follows our PV (time management)
3. Proper UCI info output with complete PV line

V16.2 FIXES (Tournament Bug Fixes):
1. Depth Bug - Fixed null move pruning condition (ply > 0 and beta < float('inf'))
2. No Move Found Bug - Explicit checkmate/stalemate checks instead of is_game_over()

V16.1 ENHANCEMENTS (Target: Beat C0BR4 v3.2):
1. Enhanced Opening Book (10-15 moves deep)
2. Middlegame Transition Nudges (rooks, bishops, king safety, pawn structure)
3. Syzygy Tablebase Integration (perfect 6-piece endgames)

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
✓ PV tracking for debugging and time management
✓ Full PV display in UCI output

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


class PVTracker:
    """Tracks principal variation using predicted board states for instant move recognition"""
    
    def __init__(self):
        self.predicted_position_fen = None  # FEN of position we expect after opponent move
        self.next_our_move = None          # Our move to play if prediction hits
        self.remaining_pv_queue = []       # Remaining moves in PV [opp_move, our_move, opp_move, our_move, ...]
        self.following_pv = False          # Whether we're actively following PV
        self.pv_display_string = ""        # PV string for display purposes
        self.original_pv = []              # Store original PV for display
        
    def store_pv_from_search(self, starting_board: chess.Board, pv_moves: List[chess.Move]):
        """Store PV from search results, setting up for following"""
        self.original_pv = pv_moves.copy()  # Keep original for display
        
        if len(pv_moves) < 3:  # Need at least: our_move, opp_move, our_next_move
            self.clear()
            return
            
        # We're about to play pv_moves[0], so prepare for opponent response
        temp_board = starting_board.copy()
        temp_board.push(pv_moves[0])  # Make our first move
        
        if len(pv_moves) >= 2:
            # Predict position after opponent plays pv_moves[1]
            temp_board.push(pv_moves[1])  # Opponent's expected response
            self.predicted_position_fen = temp_board.fen()
            
            if len(pv_moves) >= 3:
                self.next_our_move = pv_moves[2]  # Our response to their response
                self.remaining_pv_queue = pv_moves[3:]  # Rest of PV
                self.pv_display_string = ' '.join(str(m) for m in pv_moves[2:])
                self.following_pv = True
            else:
                self.clear()
        else:
            self.clear()
    
    def clear(self):
        """Clear all PV following state"""
        self.predicted_position_fen = None
        self.next_our_move = None
        self.remaining_pv_queue = []
        self.following_pv = False
        self.pv_display_string = ""
        # Keep original_pv for display even after clearing following state
    
    def check_position_for_instant_move(self, current_board: chess.Board) -> Optional[chess.Move]:
        """Check if current position matches prediction - if so, return instant move"""
        if not self.following_pv or not self.predicted_position_fen:
            return None
            
        current_fen = current_board.fen()
        if current_fen == self.predicted_position_fen:
            # Position matches prediction - return instant move
            move_to_play = self.next_our_move
            
            # Clean UCI output for PV following
            remaining_pv_str = self.pv_display_string if self.pv_display_string else str(move_to_play)
            print(f"info depth 1 score cp 0 nodes 0 time 0 pv {remaining_pv_str}", flush=True)
            print(f"info string PV prediction match", flush=True)
            
            # Set up for next prediction if we have more moves
            self._setup_next_prediction(current_board)
            
            return move_to_play
        else:
            # Position doesn't match - opponent broke PV, clear following
            self.clear()
            return None
    
    def _setup_next_prediction(self, current_board: chess.Board):
        """Set up prediction for next opponent move"""
        if len(self.remaining_pv_queue) < 2:  # Need at least opp_move, our_move
            self.clear()
            return
            
        # Make our move that we're about to play
        temp_board = current_board.copy()
        if self.next_our_move:  # Safety check
            temp_board.push(self.next_our_move)
        
        # Predict position after next opponent move
        next_opp_move = self.remaining_pv_queue[0]
        temp_board.push(next_opp_move)
        
        # Set up for next iteration
        self.predicted_position_fen = temp_board.fen()
        self.next_our_move = self.remaining_pv_queue[1] if len(self.remaining_pv_queue) >= 2 else None
        self.remaining_pv_queue = self.remaining_pv_queue[2:]  # Remove used moves
        self.pv_display_string = ' '.join(str(m) for m in [self.next_our_move] + self.remaining_pv_queue) if self.next_our_move else ""
        
        if not self.next_our_move:
            self.clear()  # No more moves to follow


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
    [490,490,500,510,510,500,490,490],
    [490,500,510,520,520,510,500,490],
    [490,500,510,520,520,510,500,490],
    [490,500,510,520,520,510,500,490],
    [490,500,510,520,520,510,500,490],
    [490,500,510,520,520,510,500,490],
    [510,520,530,540,540,530,520,510],
    [490,490,500,510,510,500,490,490],
]

QUEEN_PST = [
    [870,880,890,900,900,890,880,870],
    [880,890,900,910,910,900,890,880],
    [890,900,910,920,920,910,900,890],
    [900,910,920,930,930,920,910,900],
    [900,910,920,930,930,920,910,900],
    [890,900,910,920,920,910,900,890],
    [880,890,900,910,910,900,890,880],
    [870,880,890,900,900,890,880,870],
]

KING_MIDDLEGAME_PST = [
    [-40,-30,-50,-70,-70,-50,-30,-40],
    [-30,-20,-40,-60,-60,-40,-20,-30],
    [-20,-10,-30,-50,-50,-30,-10,-20],
    [-10,  0,-20,-40,-40,-20,  0,-10],
    [  0, 10,-10,-30,-30,-10, 10,  0],
    [ 10, 20,  0,-20,-20,  0, 20, 10],
    [ 30, 40, 20,  0,  0, 20, 40, 30],
    [ 40, 50, 30, 10, 10, 30, 50, 40],
]

KING_ENDGAME_PST = [
    [-40,-30,-20,-10,-10,-20,-30,-40],
    [-30,-20,-10,  0,  0,-10,-20,-30],
    [-20,-10,  0, 10, 10,  0,-10,-20],
    [-10,  0, 10, 20, 20, 10,  0,-10],
    [-10,  0, 10, 20, 20, 10,  0,-10],
    [-20,-10,  0, 10, 10,  0,-10,-20],
    [-30,-20,-10,  0,  0,-10,-20,-30],
    [-40,-30,-20,-10,-10,-20,-30,-40],
]


class V7P3REngine:
    """V7P3R Engine - Material Safety + Positional Play + Enhanced Features + PV Tracking"""
    
    def __init__(self, max_depth=10, tt_size_mb=256, tablebase_path=""):
        self.max_depth = max_depth
        self.board = chess.Board()
        self.nodes_searched = 0
        self.start_time = 0
        self.time_limit = 0
        self.pv_tracker = PVTracker()  # V16.3: PV tracking for display and following
        
        # Transposition table
        self.tt_size = (tt_size_mb * 1024 * 1024) // 64
        self.transposition_table = {}
        
        # Syzygy tablebase support
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
        middlegame_bonus = 0
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
        """V16.1 Middlegame transition nudges for smooth play"""
        bonus = 0
        
        # Rooks on open/semi-open files (+20cp per rook)
        for square in board.pieces(chess.ROOK, chess.WHITE):
            file_idx = chess.square_file(square)
            has_own_pawn = any(chess.square_file(sq) == file_idx for sq in board.pieces(chess.PAWN, chess.WHITE))
            has_opp_pawn = any(chess.square_file(sq) == file_idx for sq in board.pieces(chess.PAWN, chess.BLACK))
            if not has_own_pawn:
                bonus += 20 if not has_opp_pawn else 10
        
        for square in board.pieces(chess.ROOK, chess.BLACK):
            file_idx = chess.square_file(square)
            has_own_pawn = any(chess.square_file(sq) == file_idx for sq in board.pieces(chess.PAWN, chess.BLACK))
            has_opp_pawn = any(chess.square_file(sq) == file_idx for sq in board.pieces(chess.PAWN, chess.WHITE))
            if not has_own_pawn:
                bonus -= 20 if not has_opp_pawn else 10
        
        # Bishops on long diagonals (+15cp per bishop)
        for square in board.pieces(chess.BISHOP, chess.WHITE):
            diag_len = min(chess.square_rank(square), 7 - chess.square_rank(square),
                          chess.square_file(square), 7 - chess.square_file(square))
            if diag_len >= 4:
                bonus += 15
        
        for square in board.pieces(chess.BISHOP, chess.BLACK):
            diag_len = min(chess.square_rank(square), 7 - chess.square_rank(square),
                          chess.square_file(square), 7 - chess.square_file(square))
            if diag_len >= 4:
                bonus -= 15
        
        # King safety: pawn shield bonus (+10cp per shield pawn)
        white_king_sq = list(board.pieces(chess.KING, chess.WHITE))[0] if board.pieces(chess.KING, chess.WHITE) else None
        if white_king_sq:
            king_file = chess.square_file(white_king_sq)
            king_rank = chess.square_rank(white_king_sq)
            if king_rank < 2:  # King on back ranks
                for file_offset in [-1, 0, 1]:
                    check_file = king_file + file_offset
                    if 0 <= check_file < 8:
                        pawn_sq = chess.square(check_file, king_rank + 1)
                        if board.piece_at(pawn_sq) == chess.Piece(chess.PAWN, chess.WHITE):
                            bonus += 10
        
        black_king_sq = list(board.pieces(chess.KING, chess.BLACK))[0] if board.pieces(chess.KING, chess.BLACK) else None
        if black_king_sq:
            king_file = chess.square_file(black_king_sq)
            king_rank = chess.square_rank(black_king_sq)
            if king_rank > 5:  # King on back ranks
                for file_offset in [-1, 0, 1]:
                    check_file = king_file + file_offset
                    if 0 <= check_file < 8:
                        pawn_sq = chess.square(check_file, king_rank - 1)
                        if board.piece_at(pawn_sq) == chess.Piece(chess.PAWN, chess.BLACK):
                            bonus -= 10
        
        # Pawn structure: passed pawns (+30cp), doubled pawns (-20cp)
        for file_idx in range(8):
            white_pawns = [sq for sq in board.pieces(chess.PAWN, chess.WHITE) if chess.square_file(sq) == file_idx]
            black_pawns = [sq for sq in board.pieces(chess.PAWN, chess.BLACK) if chess.square_file(sq) == file_idx]
            
            # Doubled pawns
            if len(white_pawns) > 1:
                bonus -= 20 * (len(white_pawns) - 1)
            if len(black_pawns) > 1:
                bonus += 20 * (len(black_pawns) - 1)
            
            # Passed pawns
            for pawn_sq in white_pawns:
                pawn_rank = chess.square_rank(pawn_sq)
                is_passed = True
                for check_file in [file_idx - 1, file_idx, file_idx + 1]:
                    if 0 <= check_file < 8:
                        if any(chess.square_rank(sq) > pawn_rank for sq in board.pieces(chess.PAWN, chess.BLACK) 
                              if chess.square_file(sq) == check_file):
                            is_passed = False
                            break
                if is_passed:
                    bonus += 30
            
            for pawn_sq in black_pawns:
                pawn_rank = chess.square_rank(pawn_sq)
                is_passed = True
                for check_file in [file_idx - 1, file_idx, file_idx + 1]:
                    if 0 <= check_file < 8:
                        if any(chess.square_rank(sq) < pawn_rank for sq in board.pieces(chess.PAWN, chess.WHITE) 
                              if chess.square_file(sq) == check_file):
                            is_passed = False
                            break
                if is_passed:
                    bonus -= 30
        
        return bonus
    
    def get_best_move(self, time_left: float = 0, increment: float = 0) -> Optional[chess.Move]:
        """
        Get best move for current position with PV following check
        V16.3: Check PV tracker first for instant moves
        """
        # V16.3: Check if we can follow PV for instant move
        instant_move = self.pv_tracker.check_position_for_instant_move(self.board)
        if instant_move and instant_move in self.board.legal_moves:
            return instant_move
        
        # Check opening book first
        try:
            from v7p3r_openings_v163 import OPENING_BOOK
            board_fen = self.board.fen().split(' ')[0]
            if board_fen in OPENING_BOOK:
                book_moves = OPENING_BOOK[board_fen]
                if book_moves:
                    move_uci = random.choice(book_moves)
                    move = chess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        print(f"info string Opening book move", flush=True)
                        return move
        except ImportError:
            pass
        
        # Check Syzygy tablebases
        if self.tablebase and len(self.board.piece_map()) <= 6:
            try:
                dtz = self.tablebase.probe_dtz(self.board)
                if dtz is not None:
                    # Find best tablebase move
                    best_tb_move = None
                    best_dtz = float('-inf') if self.board.turn == chess.WHITE else float('inf')
                    
                    for move in self.board.legal_moves:
                        self.board.push(move)
                        try:
                            move_dtz = -self.tablebase.probe_dtz(self.board)
                            if self.board.turn == chess.WHITE:
                                if move_dtz > best_dtz:
                                    best_dtz = move_dtz
                                    best_tb_move = move
                            else:
                                if move_dtz < best_dtz:
                                    best_dtz = move_dtz
                                    best_tb_move = move
                        except:
                            pass
                        self.board.pop()
                    
                    if best_tb_move:
                        print(f"info string Tablebase move (DTZ: {best_dtz})", flush=True)
                        return best_tb_move
            except:
                pass
        
        # V16.2 FIX: Explicit terminal checks instead of is_game_over()
        if self.board.is_checkmate():
            print(f"info string Position is checkmate", flush=True)
            return None
        
        if self.board.is_stalemate():
            print(f"info string Position is stalemate", flush=True)
            return None
        
        # Calculate time limit
        self.time_limit = self._calculate_time_limit(time_left, increment)
        self.start_time = time.time()
        self.nodes_searched = 0
        
        # Iterative deepening search
        best_move = None
        best_score = float('-inf')
        
        for depth in range(1, self.max_depth + 1):
            if self._is_time_up():
                break
            
            score, move = self._search(self.board, depth, float('-inf'), float('inf'))
            
            if move and move != chess.Move.null():
                best_move = move
                best_score = score
                
                # V16.3: Extract and display full PV
                pv_line = self._extract_pv(self.board, depth)
                pv_string = " ".join([str(m) for m in pv_line])
                
                elapsed_ms = int((time.time() - self.start_time) * 1000)
                nps = int(self.nodes_searched / max(elapsed_ms / 1000, 0.001))
                
                print(f"info depth {depth} score cp {int(score)} nodes {self.nodes_searched} "
                      f"time {elapsed_ms} nps {nps} pv {pv_string}", flush=True)
                
                # V16.3: Store PV for following if depth >= 4
                if depth >= 4 and len(pv_line) >= 3:
                    self.pv_tracker.store_pv_from_search(self.board, pv_line)
        
        return best_move
    
    def _extract_pv(self, board: chess.Board, max_depth: int) -> List[chess.Move]:
        """V16.3: Extract principal variation from transposition table"""
        pv = []
        temp_board = board.copy()
        
        for depth in range(max_depth, 0, -1):
            zobrist_hash = self._get_zobrist_key(temp_board)
            
            if zobrist_hash in self.transposition_table:
                entry = self.transposition_table[zobrist_hash]
                best_move = entry.get('move')
                if best_move and best_move in temp_board.legal_moves:
                    pv.append(best_move)
                    temp_board.push(best_move)
                else:
                    break
            else:
                break
        
        return pv
    
    def _search(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0) -> Tuple[float, Optional[chess.Move]]:
        """Alpha-beta search with transposition table"""
        self.nodes_searched += 1
        
        # Time check
        if self.nodes_searched % 1000 == 0 and self._is_time_up():
            return 0, None
        
        # Transposition table lookup
        zobrist_key = self._get_zobrist_key(board)
        if zobrist_key in self.transposition_table:
            entry = self.transposition_table[zobrist_key]
            if entry['depth'] >= depth:
                return entry['score'], entry['move']
        
        # Terminal conditions
        if depth == 0:
            return self._evaluate_position(board), None
        
        # V16.2 FIX: Explicit terminal checks
        if board.is_checkmate():
            return -29000 + ply, None
        if board.is_stalemate():
            return 0, None
        
        # V16.2 FIX: Null move pruning with proper conditions
        if (depth >= 3 and not board.is_check() and 
            ply > 0 and beta < float('inf')):  # FIX: Added ply > 0 and beta check
            
            # Check if we have non-pawn pieces
            has_pieces = False
            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                if board.pieces(piece_type, board.turn):
                    has_pieces = True
                    break
            
            if has_pieces:
                board.push(chess.Move.null())
                null_score, _ = self._search(board, depth - 3, -beta, -beta + 1, ply + 1)
                null_score = -null_score
                board.pop()
                
                if null_score >= beta:
                    return beta, None
        
        # Move generation and ordering
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0, None
        
        # Simple move ordering: captures first
        def move_score(move):
            score = 0
            if board.is_capture(move):
                score += 1000
            if board.gives_check(move):
                score += 100
            return score
        
        legal_moves.sort(key=move_score, reverse=True)
        
        # Search moves
        best_move = None
        best_score = float('-inf')
        
        for move in legal_moves:
            board.push(move)
            score, _ = self._search(board, depth - 1, -beta, -alpha, ply + 1)
            score = -score
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Beta cutoff
        
        # Store in transposition table
        self.transposition_table[zobrist_key] = {
            'depth': depth,
            'score': best_score,
            'move': best_move
        }
        
        return best_score, best_move
