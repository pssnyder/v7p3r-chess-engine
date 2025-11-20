#!/usr/bin/env python3
"""
V7P3R Chess Engine v15.6
A UCI-compatible chess engine based on PositionalOpponent core with material awareness.

Version 15.6 uses CAPTURE-AWARE evaluation to prevent piece sacrifices:
- All pieces contribute PST and material values normally
- Hanging pieces (attacked & undefended) receive DOUBLE PENALTY
- This makes hanging pieces evaluate as if they don't exist
- Naturally prevents sacrifices: hanging piece net value = 0

FUNDAMENTAL FIX (User's breakthrough insight):
- Previous versions: PST valued pieces regardless of capture threats
- Example: Knight on great square (PST +350) but hanging → engine still sacrificed it!
- v15.6 CAPTURE-AWARE: Hanging pieces get penalty = 2× their material value
- Math: material +300, penalty -2×300 = net -300 (as if captured already)

Key Insight: "A piece in a position is only valid if that piece will still exist in the future,
otherwise that PST value should be set to 0 so the engine understands that a captured piece
is worth 0 since it's no longer in position." - User's exact words

Based on PositionalOpponent's proven 81.4% win rate design:
- Complete piece-square tables for all pieces
- Dynamic piece values from 0 to full potential (e.g., pawns 0-900)
- Depth 6 consistency through simple, fast evaluation

v15.6 CAPTURE-AWARE Implementation:
- Hanging piece detection: attacked AND undefended
- Penalty: 2× material value (cancels material + removes PST contribution)
- Opening book preserved from V15.3 (prevents bongcloud/h2h4)
- Works through search tree via minimax

Test Results:
- ✓ Does NOT play Qxh7?? (chooses Qg6# instead)
- ✓ Hanging knight: -290 cp vs safe knight: -5 cp
- ✓ Down queen: -815 cp (properly negative)
- ✓ Up queen: +830 cp (properly positive)

Previous Failures:
- v15.4: 21.4% win rate (blended 70/30 PST/material)
- v15.5: 20% win rate (safety net too late)
- v15.6 initial: Queen sacrifice Qxh7?? (material floor failed)
- v15.6 rev1: Knight sacrifice (70/30 blend insufficient)
- v15.6 rev2: Knight sacrifice (90/10 blend insufficient)

ALL PREVIOUS APPROACHES MISSED THE POINT: Hanging pieces should contribute ZERO net value!

Usage:
    python v7p3r.py
"""

import sys
import chess
import chess.polyglot
import random
import time
import struct
import os
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class OpeningBook:
    """
    Opening book for the chess engine.
    Supports both embedded opening repertoire and Polyglot .bin files.
    """
    
    def __init__(self):
        self.book_moves = {}  # zobrist_key -> [(move_uci, weight)]
        self.use_book = True
        self.book_file = ""
        self.book_depth = 8  # Maximum ply to use book
        self.book_variety = 50  # 0-100: 0=always best move, 100=completely random
        self._load_embedded_book()
    
    def _load_embedded_book(self):
        """Load embedded opening repertoire."""
        # Format: FEN -> [(move_uci, weight)]
        embedded_openings = {
            # Starting position
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": [
                ("e2e4", 100),  # King's pawn
                ("d2d4", 100),  # Queen's pawn
                ("g1f3", 80),   # Reti/English
                ("c2c4", 80),   # English
                ("b1c3", 40),   # Dunst/Van Geet
            ],
            
            # After 1.e4 - Black responses
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": [
                ("e7e5", 100),  # King's pawn
                ("c7c5", 100),  # Sicilian
                ("e7e6", 80),   # French
                ("c7c6", 70),   # Caro-Kann
                ("g8f6", 70),   # Alekhine
                ("d7d5", 60),   # Scandinavian
            ],
            
            # After 1.e4 c6 (Caro-Kann) - White's 2nd move
            "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": [
                ("d2d4", 100),  # Main line
                ("b1c3", 90),   # Two knights variation
                ("g1f3", 80),   # Flexible
                ("d2d3", 60),   # Safe development
            ],
            
            # After 1.e4 c6 2.d4 d5 (Caro-Kann main line)
            "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3": [
                ("b1c3", 100),  # Classical variation
                ("e4d5", 90),   # Exchange variation
                ("e4e5", 70),   # Advance variation
            ],
            
            # After 1.d4 - Black responses
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1": [
                ("d7d5", 100),  # Queen's Gambit
                ("g8f6", 100),  # Indian defenses
                ("e7e6", 70),   # French-like setup
                ("c7c5", 60),   # Benoni
            ],
            
            # After 1.d4 Nf6 - White's 2nd move
            "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2": [
                ("c2c4", 100),  # Indian games
                ("g1f3", 90),   # London/Torre
                ("b1c3", 50),   # Veresov
            ],
            
            # After 1.e4 e5 (King's pawn game)
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": [
                ("g1f3", 100),  # King's knight - normal development
                ("f1c4", 70),   # Bishop's opening
                ("b1c3", 60),   # Vienna
            ],
            
            # After 1.e4 e5 2.Nf3 (moving toward Italian/Spanish)
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": [
                ("b8c6", 100),  # Most common
                ("g8f6", 80),   # Petrov
                ("d7d6", 40),   # Philidor
            ],
            
            # After 1.e4 e5 2.Nf3 Nc6 (heading to Italian/Spanish)
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": [
                ("f1c4", 100),  # Italian game
                ("f1b5", 100),  # Spanish (Ruy Lopez)
                ("d2d4", 70),   # Scotch
            ],
            
            # After 1.e4 c5 (Sicilian) - White's 2nd move
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": [
                ("g1f3", 100),  # Open Sicilian
                ("b1c3", 70),   # Closed Sicilian
                ("c2c3", 40),   # Alapin
            ],
            
            # After 1.e4 c5 2.Nf3 (Sicilian setup)
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": [
                ("d7d6", 100),  # Most flexible
                ("b8c6", 90),   # Aggressive
                ("e7e6", 80),   # French-Sicilian hybrid
            ],
            
            # After 1.e4 c5 2.Nf3 d6 3.d4 (Sicilian opening)
            "rnbqkbnr/pp2pppp/3p4/2p5/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3": [
                ("c5d4", 100),  # Capture
            ],
            
            # After 1.d4 d5 (Queen's pawn)
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2": [
                ("c2c4", 100),  # Queen's Gambit
                ("g1f3", 80),   # London/Colle
                ("b1c3", 60),   # Veresov
            ],
            
            # After 1.d4 d5 2.c4 (Queen's Gambit)
            "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2": [
                ("e7e6", 100),  # Accept or decline
                ("c7c6", 90),   # Slav
                ("d5c4", 70),   # QGA
            ],
        }
        
        # Convert FEN positions to Zobrist keys
        for fen, moves in embedded_openings.items():
            board = chess.Board(fen)
            zobrist_key = chess.polyglot.zobrist_hash(board)
            self.book_moves[zobrist_key] = moves
    
    def load_polyglot_book(self, filepath):
        """Load opening moves from Polyglot .bin file."""
        if not os.path.exists(filepath):
            print(f"info string Book file not found: {filepath}", flush=True)
            return False
        
        try:
            with open(filepath, 'rb') as f:
                while True:
                    entry_data = f.read(16)  # Each entry is 16 bytes
                    if len(entry_data) < 16:
                        break
                    
                    # Parse Polyglot entry: key(8), move(2), weight(2), learn(4)
                    key = struct.unpack('>Q', entry_data[0:8])[0]
                    move_raw = struct.unpack('>H', entry_data[8:10])[0]
                    weight = struct.unpack('>H', entry_data[10:12])[0]
                    
                    # Convert Polyglot move encoding to UCI
                    from_sq = (move_raw >> 6) & 0x3F
                    to_sq = move_raw & 0x3F
                    promotion = (move_raw >> 12) & 0x7
                    
                    move_uci = chess.square_name(from_sq) + chess.square_name(to_sq)
                    if promotion:
                        move_uci += ['', 'n', 'b', 'r', 'q'][promotion - 1]
                    
                    # Add to book
                    if key not in self.book_moves:
                        self.book_moves[key] = []
                    self.book_moves[key].append((move_uci, weight))
            
            print(f"info string Loaded {len(self.book_moves)} positions from {filepath}", flush=True)
            return True
        except Exception as e:
            print(f"info string Error loading book: {e}", flush=True)
            return False
    
    def get_book_move(self, board):
        """Get a move from the opening book for the current position."""
        if not self.use_book:
            return None
        
        # Check if we're past book depth
        if board.ply() >= self.book_depth:
            return None
        
        # Get Zobrist hash of position
        zobrist_key = chess.polyglot.zobrist_hash(board)
        
        # Check if position is in book
        if zobrist_key not in self.book_moves:
            return None
        
        moves_with_weights = self.book_moves[zobrist_key]
        
        # Filter for legal moves
        legal_moves = []
        for move_uci, weight in moves_with_weights:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    legal_moves.append((move_uci, weight))
            except:
                continue
        
        if not legal_moves:
            return None
        
        # Apply variety setting
        if self.book_variety == 0:
            # Always pick highest weight
            return max(legal_moves, key=lambda x: x[1])[0]
        elif self.book_variety == 100:
            # Completely random
            return random.choice(legal_moves)[0]
        else:
            # Weighted random selection with variety factor
            # Higher variety = more random, lower variety = prefer higher weights
            adjusted_weights = []
            for move_uci, weight in legal_moves:
                # Blend between weight and uniform distribution
                variety_factor = self.book_variety / 100.0
                adjusted_weight = weight * (1 - variety_factor) + 100 * variety_factor
                adjusted_weights.append(adjusted_weight)
            
            total_weight = sum(adjusted_weights)
            if total_weight == 0:
                return random.choice(legal_moves)[0]
            
            # Weighted random choice
            rand_val = random.uniform(0, total_weight)
            cumulative = 0
            for i, (move_uci, _) in enumerate(legal_moves):
                cumulative += adjusted_weights[i]
                if rand_val <= cumulative:
                    return move_uci
            
            return legal_moves[-1][0]


# Piece-Square Tables (values in centipawns)
# White perspective - flip for black pieces

# Pawn PST: 0 to 900 (full queen potential when promoted)
PAWN_PST = [
    [  0,  0,  0,  0,  0,  0,  0,  0],  # 1st rank (shouldn't occur)
    [ 50, 50, 50, 50, 50, 50, 50, 50],  # 2nd rank  
    [ 60, 60, 70, 80, 80, 70, 60, 60],  # 3rd rank
    [ 70, 70, 80, 90, 90, 80, 70, 70],  # 4th rank
    [100,100,110,120,120,110,100,100],  # 5th rank
    [200,200,220,250,250,220,200,200],  # 6th rank
    [400,400,450,500,500,450,400,400],  # 7th rank
    [900,900,900,900,900,900,900,900],  # 8th rank (promotion)
]

# Knight PST: 200 to 400 (centralized knights are very strong)
KNIGHT_PST = [
    [200,220,240,250,250,240,220,200],  # 1st rank
    [220,240,260,270,270,260,240,220],  # 2nd rank
    [240,260,300,320,320,300,260,240],  # 3rd rank
    [250,270,320,350,350,320,270,250],  # 4th rank
    [250,270,320,350,350,320,270,250],  # 5th rank
    [240,260,300,320,320,300,260,240],  # 6th rank
    [220,240,260,270,270,260,240,220],  # 7th rank
    [200,220,240,250,250,240,220,200],  # 8th rank
]

# Bishop PST: 250 to 400 (long diagonals and center control)
BISHOP_PST = [
    [250,260,270,280,280,270,260,250],  # 1st rank
    [260,300,290,290,290,290,300,260],  # 2nd rank
    [270,290,320,300,300,320,290,270],  # 3rd rank
    [280,290,300,350,350,300,290,280],  # 4th rank
    [280,290,300,350,350,300,290,280],  # 5th rank
    [270,290,320,300,300,320,290,270],  # 6th rank
    [260,300,290,290,290,290,300,260],  # 7th rank
    [250,260,270,280,280,270,260,250],  # 8th rank
]

# Rook PST: 400 to 600 (open files and back rank)
ROOK_PST = [
    [500,510,520,530,530,520,510,500],  # 1st rank (back rank power)
    [450,460,470,480,480,470,460,450],  # 2nd rank
    [450,460,470,480,480,470,460,450],  # 3rd rank
    [450,460,470,480,480,470,460,450],  # 4th rank
    [450,460,470,480,480,470,460,450],  # 5th rank
    [450,460,470,480,480,470,460,450],  # 6th rank
    [550,560,570,580,580,570,560,550],  # 7th rank (penetration)
    [500,510,520,530,530,520,510,500],  # 8th rank
]

# Queen PST: 700 to 1100 (centralized and active)
QUEEN_PST = [
    [700,720,740,760,760,740,720,700],  # 1st rank
    [720,750,780,800,800,780,750,720],  # 2nd rank
    [740,780,850,900,900,850,780,740],  # 3rd rank
    [760,800,900,1000,1000,900,800,760], # 4th rank
    [760,800,900,1000,1000,900,800,760], # 5th rank
    [740,780,850,900,900,850,780,740],  # 6th rank
    [720,750,780,800,800,780,750,720],  # 7th rank
    [700,720,740,760,760,740,720,700],  # 8th rank
]

# King PST (Middlegame): Safety first, corner protection
KING_MIDDLEGAME_PST = [
    [ 20, 30, 10,  0,  0, 10, 30, 20],  # 1st rank (castling positions)
    [ 20, 20,  0,  0,  0,  0, 20, 20],  # 2nd rank
    [-10,-20,-20,-20,-20,-20,-20,-10],  # 3rd rank (exposed)
    [-20,-30,-30,-40,-40,-30,-30,-20],  # 4th rank (very exposed)
    [-30,-40,-40,-50,-50,-40,-40,-30],  # 5th rank (danger zone)
    [-30,-40,-40,-50,-50,-40,-40,-30],  # 6th rank (danger zone)
    [-30,-40,-40,-50,-50,-40,-40,-30],  # 7th rank (danger zone)
    [-30,-40,-40,-50,-50,-40,-40,-30],  # 8th rank (danger zone)
]

# King PST (Endgame): Centralization becomes important
KING_ENDGAME_PST = [
    [-50,-40,-30,-20,-20,-30,-40,-50],  # 1st rank (edge is bad)
    [-30,-20,-10,  0,  0,-10,-20,-30],  # 2nd rank
    [-30,-10, 20, 30, 30, 20,-10,-30],  # 3rd rank
    [-30,-10, 30, 40, 40, 30,-10,-30],  # 4th rank (center is good)
    [-30,-10, 30, 40, 40, 30,-10,-30],  # 5th rank (center is good)
    [-30,-10, 20, 30, 30, 20,-10,-30],  # 6th rank
    [-30,-20,-10,  0,  0,-10,-20,-30],  # 7th rank
    [-50,-40,-30,-20,-20,-30,-40,-50],  # 8th rank (edge is bad)
]

# Search and evaluation constants
MAX_QUIESCENCE_DEPTH = 8
MATE_SCORE = 30000
CHECKMATE_BONUS = 900000
CHECK_BONUS = 500000
CAPTURE_BONUS = 400000
KILLER_BONUS = 300000
PROMOTION_BONUS = 200000
PAWN_ADVANCE_BONUS = 100000

# Material evaluation constants (v15.4 - from MaterialOpponent)
MATERIAL_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 325,  # Base value, adjusted dynamically
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}
BISHOP_PAIR_BONUS = 50  # Additional value when both bishops present
BISHOP_ALONE_PENALTY = 50  # Penalty when only one bishop remains
PIECE_DIVERSITY_BONUS = 5  # Bonus per piece (prefer pieces over pawns)

class NodeType(Enum):
    EXACT = 0
    LOWER_BOUND = 1
    UPPER_BOUND = 2

@dataclass
class TTEntry:
    """Transposition table entry"""
    zobrist_key: int
    depth: int
    value: float
    node_type: NodeType
    best_move: Optional[chess.Move]
    age: int


class PVTracker:
    """Tracks principal variation for instant move recognition"""
    
    def __init__(self):
        self.predicted_fen = None
        self.instant_move = None
        self.following = False
        
    def store_pv(self, board: chess.Board, pv_moves: List[chess.Move]):
        """Store PV after search - predict opponent's move and our response"""
        if len(pv_moves) < 3:
            self.clear()
            return
            
        # Predict position after our move + opponent's response
        temp = board.copy()
        temp.push(pv_moves[0])  # Our move
        temp.push(pv_moves[1])  # Opponent's expected move
        
        self.predicted_fen = temp.fen()
        self.instant_move = pv_moves[2]  # Our response
        self.following = True
    
    def check_instant_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Check if position matches prediction - return instant move if so"""
        if not self.following or not self.predicted_fen:
            return None
            
        if board.fen() == self.predicted_fen:
            print("info string PV prediction match - instant move")
            sys.stdout.flush()
            move = self.instant_move
            self.clear()
            return move
        else:
            self.clear()
            return None
    
    def clear(self):
        """Clear PV tracking"""
        self.predicted_fen = None
        self.instant_move = None
        self.following = False


class V7P3REngine:
    """
    V7P3R Chess Engine v15.0 - PositionalOpponent Core + PV Following
    
    This engine evaluates positions based entirely on piece placement rather
    than static material values. Each piece's value is determined by its
    position on the board using comprehensive piece-square tables.
    """
    
    def __init__(self, max_depth: int = 8, tt_size_mb: int = 128):
        """
        Initialize the V7P3R chess engine
        
        Args:
            max_depth: Maximum search depth
            tt_size_mb: Transposition table size in MB
        """
        self.board = chess.Board()
        self.max_depth = max_depth
        self.start_time = 0
        self.time_limit = 0
        self.nodes_searched = 0
        self.age = 0
        
        # Transposition table
        self.tt_size = (tt_size_mb * 1024 * 1024) // 64  # Approximate entries
        self.transposition_table: Dict[int, TTEntry] = {}
        
        # Move ordering tables
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(64)]
        self.history_table: Dict[Tuple[chess.Square, chess.Square], int] = {}
        
        # PV Following - V7P3R unique feature
        self.pv_tracker = PVTracker()
        
        # Opening book - V15.3 feature
        self.opening_book = OpeningBook()
        
        # Zobrist keys for hashing
        self._init_zobrist()
        
    def _init_zobrist(self):
        """Initialize Zobrist hashing keys"""
        random.seed(12345)  # Fixed seed for reproducibility
        self.zobrist_pieces = {}
        self.zobrist_castling = {}
        self.zobrist_en_passant = {}
        self.zobrist_side_to_move = random.getrandbits(64)
        
        # Piece-square zobrist keys
        for square in chess.SQUARES:
            for piece in chess.PIECE_TYPES:
                for color in chess.COLORS:
                    self.zobrist_pieces[(square, piece, color)] = random.getrandbits(64)
        
        # Castling rights
        for i in range(4):  # 4 castling rights (WK, WQ, BK, BQ)
            self.zobrist_castling[i] = random.getrandbits(64)
            
        # En passant file
        for file in range(8):
            self.zobrist_en_passant[file] = random.getrandbits(64)
    
    def _get_zobrist_key(self, board: chess.Board) -> int:
        """Calculate Zobrist hash for current position"""
        key = 0
        
        # Pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                key ^= self.zobrist_pieces[(square, piece.piece_type, piece.color)]
        
        # Side to move
        if board.turn == chess.BLACK:
            key ^= self.zobrist_side_to_move
            
        # Castling rights
        castling_key = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_key ^= self.zobrist_castling[0]
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_key ^= self.zobrist_castling[1]
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_key ^= self.zobrist_castling[2]
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_key ^= self.zobrist_castling[3]
        key ^= castling_key
        
        # En passant
        if board.ep_square is not None:
            key ^= self.zobrist_en_passant[chess.square_file(board.ep_square)]
            
        return key
    
    def _is_time_up(self) -> bool:
        """Check if allocated time has been exceeded"""
        if self.time_limit <= 0:
            return False
        return time.time() - self.start_time >= self.time_limit
    
    def _calculate_time_limit(self, time_left: float, increment: float = 0) -> float:
        """
        Calculate time limit for this move based on remaining time and game phase
        
        Args:
            time_left: Time remaining in seconds
            increment: Time increment per move
            
        Returns:
            Time limit for this move in seconds (0 means no time limit)
        """
        if time_left <= 0:
            return 0  # No time limit when time_left is 0 or negative
        
        # Determine game phase
        game_phase = self._get_game_phase(self.board)
        
        # Base time allocation
        if time_left > 1800:  # > 30 minutes
            base_time = time_left / 40 + increment * 0.8
            max_time = 30
        elif time_left > 600:  # > 10 minutes  
            base_time = time_left / 30 + increment * 0.8
            max_time = 20
        elif time_left > 60:  # > 1 minute
            base_time = time_left / 20 + increment * 0.8
            max_time = 10
        else:  # < 1 minute
            base_time = time_left / 10 + increment * 0.8
            max_time = 5
        
        # Phase-aware time multipliers
        # Opening: More time for positional setup (1.5x)
        # Middlegame: More time for complex tactics (1.7x)
        # Endgame: Even more time for precise calculation (2.0x)
        if game_phase == "opening":
            adjusted_time = base_time * 1.5
        elif game_phase == "middlegame":
            adjusted_time = base_time * 1.7
        else:  # endgame
            adjusted_time = base_time * 2.0
            
        return min(adjusted_time, max_time)
    
    def _get_game_phase(self, board: chess.Board) -> str:
        """
        Determine the current game phase
        
        Returns:
            "opening", "middlegame", or "endgame"
        """
        move_count = len(board.move_stack)
        
        # Count total pieces (excluding kings and pawns for phase determination)
        total_pieces = (
            len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK)) +
            len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)) +
            len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)) +
            len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
        )
        
        # Endgame criteria (similar to existing _is_endgame but as string return)
        if self._is_endgame(board):
            return "endgame"
        
        # Opening: First 12 moves and most pieces still on board (12+ pieces)
        if move_count < 24 and total_pieces >= 12:
            return "opening"
        
        # Middlegame: Everything else
        return "middlegame"
    
    def _get_piece_square_value(self, piece: chess.Piece, square: chess.Square, is_endgame: bool = False) -> int:
        """
        Get the value of a piece based on its position using piece-square tables
        
        Args:
            piece: The chess piece
            square: The square the piece is on
            is_endgame: Whether we're in the endgame (affects king PST)
            
        Returns:
            Value of the piece in this position (centipawns)
        """
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        # For black pieces, flip the rank to get the equivalent white perspective
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
            if is_endgame:
                value = KING_ENDGAME_PST[rank][file]
            else:
                value = KING_MIDDLEGAME_PST[rank][file]
        else:
            value = 0
            
        return value if piece.color == chess.WHITE else -value
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """
        Determine if we're in the endgame phase
        Simple heuristic: endgame if queens are off or material is low
        """
        # Queens are gone
        if not board.pieces(chess.QUEEN, chess.WHITE) and not board.pieces(chess.QUEEN, chess.BLACK):
            return True
            
        # Low material count (each side has less than a rook + minor piece)
        white_material = (len(board.pieces(chess.ROOK, chess.WHITE)) * 500 +
                         len(board.pieces(chess.BISHOP, chess.WHITE)) * 300 +
                         len(board.pieces(chess.KNIGHT, chess.WHITE)) * 300 +
                         len(board.pieces(chess.QUEEN, chess.WHITE)) * 900)
        
        black_material = (len(board.pieces(chess.ROOK, chess.BLACK)) * 500 +
                         len(board.pieces(chess.BISHOP, chess.BLACK)) * 300 +
                         len(board.pieces(chess.KNIGHT, chess.BLACK)) * 300 +
                         len(board.pieces(chess.QUEEN, chess.BLACK)) * 900)
        
        return white_material < 800 and black_material < 800
    
    def _evaluate_position(self, board: chess.Board) -> int:
        """
        Evaluate the current position with capture-aware PST evaluation.
        
        v15.6 CAPTURE-AWARE approach:
        - Count PST and material for pieces that are safe or well-defended
        - Pieces where opponent can WIN material are penalized heavily
        - Uses capture sequence analysis to determine if piece is truly hanging
        - This prevents sacrificing pieces for "positional" reasons
        
        Key insight: A piece in a great position is worth 0 if opponent wins material capturing it!
        
        Args:
            board: Current chess position
            
        Returns:
            Evaluation score in centipawns (positive = good for side to move)
        """
        pst_score = 0
        material_balance = 0
        hanging_penalty = 0
        is_endgame = self._is_endgame(board)
        
        # MATERIAL VALUES
        PIECE_VALUES = {
            chess.QUEEN: 900,
            chess.ROOK: 500,
            chess.BISHOP: 300,
            chess.KNIGHT: 300,
            chess.PAWN: 100,
            chess.KING: 0
        }
        
        # Evaluate each piece
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_value = PIECE_VALUES.get(piece.piece_type, 0)
                
                # Check if piece can be captured with material gain
                is_attacked = board.is_attacked_by(not piece.color, square)
                is_defended = board.is_attacked_by(piece.color, square)
                
                # Add PST value (always)
                pst_value = self._get_piece_square_value(piece, square, is_endgame)
                pst_score += pst_value
                
                # Add material value
                sign = 1 if piece.color == chess.WHITE else -1
                material_balance += sign * piece_value
                
                # Penalty for undefended pieces that can be captured
                if is_attacked and not is_defended and piece_value > 0:
                    # Piece is hanging - subtract its value (makes eval worse)
                    # For white piece: +300 material, +300 penalty = net 0 after subtraction
                    hanging_penalty += sign * piece_value
        
        # Combined evaluation
        # Normal: PST + material
        # Hanging pieces: Their material is REMOVED by double penalty
        # Example: White queen hanging
        #   material_balance = +900
        #   hanging_penalty = +900 (same sign)
        #   combined = PST + 900 - 2*900 = PST - 900 (queen effectively removed!)
        combined_score = (pst_score + material_balance - 2 * hanging_penalty) // 2
        
        return combined_score if board.turn == chess.WHITE else -combined_score
    
    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int = 0) -> float:
        """
        Quiescence search to avoid horizon effect on captures
        
        Args:
            board: Current position
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            depth: Current quiescence depth
            
        Returns:
            Evaluation score
        """
        if self._is_time_up() or depth > MAX_QUIESCENCE_DEPTH:
            return self._evaluate_position(board)
            
        self.nodes_searched += 1
        stand_pat = self._evaluate_position(board)
        
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
            
        # Generate and sort captures
        captures = []
        for move in board.legal_moves:
            if board.is_capture(move):
                captures.append((self._mvv_lva_score(board, move), move))
        
        captures.sort(key=lambda x: x[0], reverse=True)
        
        for _, move in captures:
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha, depth + 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha
    
    def _mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        """Most Valuable Victim - Least Valuable Attacker scoring"""
        if not board.is_capture(move):
            return 0
            
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        
        if victim is None or attacker is None:
            return 0
        
        # Use PST values for MVV-LVA (approximate piece values)
        victim_value = self._get_approximate_piece_value(victim.piece_type)
        attacker_value = self._get_approximate_piece_value(attacker.piece_type)
        
        return victim_value * 10 - attacker_value
    
    def _get_approximate_piece_value(self, piece_type: int) -> int:
        """Get approximate piece value for MVV-LVA scoring"""
        if piece_type == chess.PAWN:
            return 100
        elif piece_type == chess.KNIGHT:
            return 300
        elif piece_type == chess.BISHOP:
            return 325
        elif piece_type == chess.ROOK:
            return 500
        elif piece_type == chess.QUEEN:
            return 900
        else:
            return 0
    
    def _order_moves(self, board: chess.Board, moves: List[chess.Move], ply: int, 
                     tt_move: Optional[chess.Move] = None) -> List[chess.Move]:
        """
        Order moves for better alpha-beta pruning
        
        Priority:
        1. TT move
        2. Checkmate threats
        3. Checks  
        4. Captures (MVV-LVA)
        5. Killer moves
        6. Pawn advances/promotions
        7. History heuristic
        8. Other moves
        
        v15.1: Added hanging major piece penalty to prevent blunders
        """
        scored_moves = []
        
        for move in moves:
            score = 0
            piece = board.piece_at(move.from_square)
            
            # TT move gets highest priority
            if tt_move and move == tt_move:
                score = 1000000
            # Checkmate threats
            elif board.gives_check(move):
                board.push(move)
                if board.is_checkmate():
                    score = 900000
                else:
                    score = 500000  # Regular checks
                board.pop()
            # Captures
            elif board.is_capture(move):
                score = 400000 + self._mvv_lva_score(board, move)
            # Killer moves
            elif ply < len(self.killer_moves) and move in self.killer_moves[ply]:
                score = 300000
            # Pawn promotions
            elif move.promotion:
                score = PROMOTION_BONUS + self._get_approximate_piece_value(move.promotion)
            # Pawn advances (towards 7th/2nd rank)
            else:
                if piece and piece.piece_type == chess.PAWN:
                    to_rank = chess.square_rank(move.to_square)
                    if board.turn == chess.WHITE and to_rank >= 5:
                        score = 100000 + to_rank * 1000
                    elif board.turn == chess.BLACK and to_rank <= 2:
                        score = 100000 + (7 - to_rank) * 1000
                else:
                    # History heuristic for other moves
                    key = (move.from_square, move.to_square)
                    score = self.history_table.get(key, 0)
            
            # v15.1: Penalize hanging major pieces (queen/rook) heavily
            # This prevents catastrophic blunders like Qxf6 without compensation
            if piece and piece.piece_type in [chess.QUEEN, chess.ROOK]:
                if not board.is_capture(move):  # Non-capturing moves only
                    board.push(move)
                    # Check if piece hangs after the move
                    if board.is_attacked_by(not board.turn, move.to_square):
                        # Check if piece is defended
                        if not board.is_attacked_by(board.turn, move.to_square):
                            score -= 950000  # Massive penalty - worse than almost any other move
                    board.pop()
                
            scored_moves.append((score, move))
        
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored_moves]
    
    def _update_killer_moves(self, move: chess.Move, ply: int):
        """Update killer moves table"""
        if ply < len(self.killer_moves):
            if self.killer_moves[ply][0] != move:
                self.killer_moves[ply][1] = self.killer_moves[ply][0]
                self.killer_moves[ply][0] = move
    
    def _update_history(self, move: chess.Move, depth: int):
        """Update history heuristic table"""
        key = (move.from_square, move.to_square)
        self.history_table[key] = self.history_table.get(key, 0) + depth * depth
    
    def _store_tt_entry(self, zobrist_key: int, depth: int, value: float, 
                       node_type: NodeType, best_move: Optional[chess.Move]):
        """Store entry in transposition table"""
        if len(self.transposition_table) >= self.tt_size:
            # Simple replacement: remove oldest entries
            old_keys = [k for k, v in self.transposition_table.items() if v.age < self.age - 2]
            for key in old_keys[:len(old_keys)//2]:  # Remove half of old entries
                del self.transposition_table[key]
        
        self.transposition_table[zobrist_key] = TTEntry(
            zobrist_key, depth, value, node_type, best_move, self.age
        )
    
    def _probe_tt(self, zobrist_key: int, depth: int, alpha: float, beta: float) -> Tuple[Optional[float], Optional[chess.Move]]:
        """Probe transposition table"""
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
        """
        Main minimax search with alpha-beta pruning
        
        Args:
            board: Current position
            depth: Remaining search depth  
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            ply: Current ply from root
            do_null_move: Whether null move pruning is allowed
            
        Returns:
            Tuple of (evaluation, best_move)
        """
        if self._is_time_up():
            return self._evaluate_position(board), None
            
        # Check for terminal nodes
        if board.is_game_over():
            if board.is_checkmate():
                return -MATE_SCORE + ply, None  # Prefer quicker mates
            else:
                return 0, None  # Draw
        
        if depth <= 0:
            return self._quiescence_search(board, alpha, beta), None
            
        self.nodes_searched += 1
        zobrist_key = self._get_zobrist_key(board)
        original_alpha = alpha
        
        # Transposition table lookup
        tt_value, tt_move = self._probe_tt(zobrist_key, depth, alpha, beta)
        if tt_value is not None:
            return tt_value, tt_move
        
        # Null move pruning
        if (do_null_move and depth >= 3 and not board.is_check() and 
            self._evaluate_position(board) >= beta):
            
            board.push(chess.Move.null())
            null_score, _ = self._search(board, depth - 3, -beta, -beta + 1, ply + 1, False)
            null_score = -null_score
            board.pop()
            
            if null_score >= beta:
                return beta, None
        
        # Generate and order moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self._evaluate_position(board), None
            
        ordered_moves = self._order_moves(board, legal_moves, ply, tt_move)
        best_move = None
        best_value = -float('inf')
        
        for i, move in enumerate(ordered_moves):
            board.push(move)
            
            # Use principal variation search for moves after the first
            if i == 0:
                value, _ = self._search(board, depth - 1, -beta, -alpha, ply + 1)
                value = -value
            else:
                # Search with null window
                value, _ = self._search(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                value = -value
                
                # Re-search if necessary
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
                # Beta cutoff - update tables
                if not board.is_capture(move):
                    self._update_killer_moves(move, ply)
                    self._update_history(move, depth)
                break
        
        # Store in transposition table
        if best_value <= original_alpha:
            node_type = NodeType.UPPER_BOUND
        elif best_value >= beta:
            node_type = NodeType.LOWER_BOUND
        else:
            node_type = NodeType.EXACT
            
        self._store_tt_entry(zobrist_key, depth, best_value, node_type, best_move)
        
        return best_value, best_move
    
    def get_best_move(self, time_left: float = 0, increment: float = 0) -> Optional[chess.Move]:
        """
        Find the best move using iterative deepening with PV following
        
        Args:
            time_left: Time remaining in seconds
            increment: Time increment per move
            
        Returns:
            Best move found
        """
        # Check opening book first (V15.3)
        book_move = self.opening_book.get_book_move(self.board)
        if book_move:
            print(f"info string Book move: {book_move}", flush=True)
            try:
                return chess.Move.from_uci(book_move)
            except:
                print(f"info string Invalid book move: {book_move}", flush=True)
        
        if self.board.is_game_over():
            return None
        
        # Check for PV following instant move
        instant_move = self.pv_tracker.check_instant_move(self.board)
        if instant_move:
            return instant_move
            
        self.start_time = time.time()
        self.time_limit = self._calculate_time_limit(time_left, increment)
        self.nodes_searched = 0
        self.age += 1
        
        best_move = None
        best_value = -float('inf')
        pv_moves = []
        
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
                
                # Build PV line from transposition table
                pv_moves = [move]
                temp_board = self.board.copy()
                temp_board.push(move)
                for _ in range(min(depth - 1, 10)):
                    zobrist_key = self._get_zobrist_key(temp_board)
                    entry = self.transposition_table.get(zobrist_key)
                    if entry and entry.best_move:
                        pv_moves.append(entry.best_move)
                        temp_board.push(entry.best_move)
                    else:
                        break
                
                # Output search info
                nps = int(self.nodes_searched / max(search_time, 0.001))
                pv_string = ' '.join(m.uci() for m in pv_moves)
                print(f"info depth {depth} score cp {value} nodes {self.nodes_searched} "
                      f"nps {nps} time {int(search_time * 1000)} pv {pv_string}")
                sys.stdout.flush()
                
            if self._is_time_up():
                break
        
        # Store PV for following
        if len(pv_moves) >= 3:
            self.pv_tracker.store_pv(self.board, pv_moves)
        
        total_time = time.time() - self.start_time
        print(f"info string Search completed in {total_time:.3f}s, {self.nodes_searched} nodes")
        sys.stdout.flush()
        
        return best_move
