#!/usr/bin/env python3
"""
V7P3R Chess Engine v15.0 - Clean Rebuild

Rebuilt from V12.6 foundation with Material Opponent baseline performance.
Focus: Deep tactical search (depth 8-10) with pure material evaluation.

REBUILD PHILOSOPHY:
- Core search infrastructure from proven V12.6 codebase
- Move ordering aligned with Material Opponent's effectiveness
- Pure material counting for baseline (no heuristics initially)
- Aggressive depth targets (8-10 ply) for tactical awareness
- Clean, maintainable code structure

CORE COMPONENTS:
- Alpha-beta search with negamax
- Iterative deepening with time management
- Transposition table (Zobrist hashing)
- Killer moves and history heuristic
- Quiescence search (captures + checks)
- Null move pruning
- Late move reduction (LMR)

Author: Pat Snyder
Version: 15.0 - Clean Material Baseline
"""

import time
import chess
import sys
import random
from typing import Optional, Tuple, List, Dict
from collections import defaultdict


class TranspositionEntry:
    """Entry in the transposition table"""
    def __init__(self, depth: int, score: int, best_move: Optional[chess.Move], 
                 node_type: str, zobrist_hash: int):
        self.depth = depth
        self.score = score
        self.best_move = best_move
        self.node_type = node_type  # 'exact', 'lowerbound', 'upperbound'
        self.zobrist_hash = zobrist_hash


class KillerMoves:
    """Killer move storage - 2 killer moves per depth"""
    def __init__(self):
        self.killers: Dict[int, List[chess.Move]] = defaultdict(list)
    
    def store_killer(self, move: chess.Move, depth: int):
        """Store a killer move at the given depth"""
        if move not in self.killers[depth]:
            self.killers[depth].insert(0, move)
            if len(self.killers[depth]) > 2:
                self.killers[depth].pop()
    
    def get_killers(self, depth: int) -> List[chess.Move]:
        """Get killer moves for the given depth"""
        return self.killers[depth]
    
    def is_killer(self, move: chess.Move, depth: int) -> bool:
        """Check if a move is a killer at the given depth"""
        return move in self.killers[depth]


class HistoryHeuristic:
    """History heuristic for move ordering"""
    def __init__(self):
        self.history: Dict[str, int] = defaultdict(int)
    
    def update_history(self, move: chess.Move, depth: int):
        """Update history score for a move"""
        move_key = f"{move.from_square}-{move.to_square}"
        self.history[move_key] += depth * depth
    
    def get_history_score(self, move: chess.Move) -> int:
        """Get history score for a move"""
        move_key = f"{move.from_square}-{move.to_square}"
        return self.history[move_key]


class ZobristHashing:
    """Zobrist hashing for transposition table"""
    def __init__(self):
        random.seed(12345)  # Deterministic for reproducibility
        self.piece_square_table = {}
        self.side_to_move = random.getrandbits(64)
        self.castling_rights = {}
        self.en_passant = {}
        
        # Generate random numbers for each piece on each square
        for square in range(64):
            for piece_type in range(1, 7):  # PAWN to KING
                for color in [chess.WHITE, chess.BLACK]:
                    key = (square, piece_type, color)
                    self.piece_square_table[key] = random.getrandbits(64)
        
        # Castling rights
        for i in range(4):  # WK, WQ, BK, BQ
            self.castling_rights[i] = random.getrandbits(64)
        
        # En passant files
        for file in range(8):
            self.en_passant[file] = random.getrandbits(64)
    
    def hash_position(self, board: chess.Board) -> int:
        """Generate Zobrist hash for the position"""
        hash_value = 0
        
        # Hash pieces
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                key = (square, piece.piece_type, piece.color)
                hash_value ^= self.piece_square_table[key]
        
        # Hash side to move
        if board.turn == chess.BLACK:
            hash_value ^= self.side_to_move
        
        # Hash castling rights
        castling_key = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_key ^= self.castling_rights[0]
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_key ^= self.castling_rights[1]
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_key ^= self.castling_rights[2]
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_key ^= self.castling_rights[3]
        hash_value ^= castling_key
        
        # Hash en passant
        if board.ep_square is not None:
            hash_value ^= self.en_passant[chess.square_file(board.ep_square)]
            
        return hash_value


class V7P3REngine:
    """V7P3R Chess Engine v15.0 - Clean Material Baseline"""
    
    def __init__(self):
        # Pure material values - no complexity
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 325,  # Slight bishop preference like Material Opponent
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        # Search configuration - AGGRESSIVE DEPTH TARGET
        self.default_depth = 8  # Start at depth 8, goal is depth 10
        self.nodes_searched = 0
        
        # Search infrastructure
        self.transposition_table: Dict[int, TranspositionEntry] = {}
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        self.zobrist = ZobristHashing()
        
        # Configuration
        self.max_tt_entries = 100000  # Generous TT size for deep search
        
        # Performance monitoring
        self.search_stats = {
            'nodes_per_second': 0,
            'tt_hits': 0,
            'tt_stores': 0,
            'killer_hits': 0,
            'beta_cutoffs': 0,
        }
        
        # Bishop pair bonus (like Material Opponent)
        self.bishop_pair_bonus = 50
        self.bishop_alone_penalty = 50
    
    def search(self, board: chess.Board, time_limit: float = 3.0, depth: Optional[int] = None) -> chess.Move:
        """
        Main search entry point with iterative deepening
        """
        self.nodes_searched = 0
        self.search_start_time = time.time()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        # Calculate time allocation
        target_time, max_time = self._calculate_time_allocation(board, time_limit)
        
        # Iterative deepening
        best_move = legal_moves[0]
        best_score = -99999
        search_depth = depth if depth else self.default_depth
        
        for current_depth in range(1, search_depth + 1):
            iteration_start = time.time()
            
            # Time management checks
            elapsed = time.time() - self.search_start_time
            if elapsed > target_time:
                break
            
            # Predict if next iteration will exceed max time
            if current_depth > 1:
                last_iteration_time = time.time() - iteration_start
                if elapsed + (last_iteration_time * 2.5) > max_time:
                    break
            
            try:
                # Call recursive search for this depth
                score, move = self._recursive_search(board, current_depth, -99999, 99999, True)
                
                # Update best move if valid result
                if move and move != chess.Move.null():
                    best_move = move
                    best_score = score
                    
                    # Print UCI info
                    elapsed_ms = int((time.time() - self.search_start_time) * 1000)
                    nps = int(self.nodes_searched / max(elapsed_ms / 1000, 0.001))
                    
                    # Extract PV
                    pv_line = self._extract_pv(board, current_depth)
                    pv_string = " ".join([str(m) for m in pv_line])
                    
                    print(f"info depth {current_depth} score cp {int(score)} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {pv_string}")
                    sys.stdout.flush()
                
                # Time check after iteration
                elapsed = time.time() - self.search_start_time
                if elapsed > target_time:
                    break
                    
            except Exception as e:
                print(f"info string Search interrupted at depth {current_depth}: {e}")
                break
        
        return best_move
    
    def _recursive_search(self, board: chess.Board, depth: int, alpha: float, beta: float, 
                         do_null_move: bool = True) -> Tuple[float, Optional[chess.Move]]:
        """
        Recursive alpha-beta search with all optimizations
        Returns (score, best_move) tuple
        """
        self.nodes_searched += 1
        
        # Periodic time checking (every 1000 nodes)
        if hasattr(self, 'search_start_time') and self.nodes_searched % 1000 == 0:
            elapsed = time.time() - self.search_start_time
            # Simple emergency check - just return current eval if time exceeded
            if hasattr(self, 'max_allocated_time') and elapsed > self.max_allocated_time:
                return self._evaluate_material(board), None
        
        # Check for game over
        if board.is_game_over():
            if board.is_checkmate():
                return -29000.0 + (self.default_depth - depth), None
            else:
                return 0.0, None  # Draw
        
        # Transposition table probe
        tt_hit, tt_score, tt_move = self._probe_tt(board, depth, int(alpha), int(beta))
        if tt_hit:
            self.search_stats['tt_hits'] += 1
            return float(tt_score), tt_move
        
        # Drop into quiescence at leaf nodes
        if depth <= 0:
            score = self._quiescence_search(board, alpha, beta, 8)
            return score, None
        
        # Null move pruning
        if (do_null_move and depth >= 3 and not board.is_check() and 
            self._has_non_pawn_pieces(board)):
            
            # Make null move
            board.push(chess.Move.null())
            null_score, _ = self._recursive_search(board, depth - 3, -beta, -beta + 1, False)
            null_score = -null_score
            board.pop()
            
            if null_score >= beta:
                return null_score, None
        
        # Move generation and ordering
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0, None
        
        ordered_moves = self._order_moves(board, legal_moves, depth, tt_move)
        
        # Main search loop
        best_score = -99999.0
        best_move = None
        original_alpha = alpha
        moves_searched = 0
        
        for move in ordered_moves:
            board.push(move)
            
            # Late Move Reduction (LMR)
            reduction = self._calculate_lmr_reduction(move, moves_searched, depth, board)
            
            if reduction > 0:
                # Search with reduction
                score, _ = self._recursive_search(board, depth - 1 - reduction, -beta, -alpha, True)
                score = -score
                
                # Re-search at full depth if it failed high
                if score > alpha:
                    score, _ = self._recursive_search(board, depth - 1, -beta, -alpha, True)
                    score = -score
            else:
                # Full depth search
                score, _ = self._recursive_search(board, depth - 1, -beta, -alpha, True)
                score = -score
            
            board.pop()
            moves_searched += 1
            
            # Update best move
            if score > best_score:
                best_score = score
                best_move = move
            
            # Alpha-beta updates
            alpha = max(alpha, score)
            if alpha >= beta:
                # Beta cutoff
                self.search_stats['beta_cutoffs'] += 1
                if not board.is_capture(move):
                    self.killer_moves.store_killer(move, depth)
                    self.history_heuristic.update_history(move, depth)
                break
        
        # Store in transposition table
        self._store_tt(board, depth, int(best_score), best_move, int(original_alpha), int(beta))
        
        return best_score, best_move
    
    def _order_moves(self, board: chess.Board, moves: List[chess.Move], depth: int,
                     tt_move: Optional[chess.Move] = None) -> List[chess.Move]:
        """
        Move ordering aligned with Material Opponent's proven structure:
        1. TT move
        2. Checks (checkmate threats highest)
        3. Captures (MVV-LVA)
        4. Killer moves
        5. Pawn advances/promotions
        6. History heuristic
        7. Other quiet moves
        """
        if len(moves) <= 2:
            return moves
        
        scored_moves = []
        killer_set = set(self.killer_moves.get_killers(depth))
        
        for move in moves:
            score = 0
            
            # 1. TT move (highest priority)
            if tt_move and move == tt_move:
                score = 1000000
            
            # 2. Checkmate threats
            elif self._gives_checkmate(board, move):
                score = 900000
            
            # 3. Checks
            elif board.gives_check(move):
                score = 800000
            
            # 4. Captures (MVV-LVA)
            elif board.is_capture(move):
                victim = board.piece_at(move.to_square)
                victim_value = self.piece_values.get(victim.piece_type, 0) if victim else 0
                attacker = board.piece_at(move.from_square)
                attacker_value = self.piece_values.get(attacker.piece_type, 0) if attacker else 0
                score = 700000 + (victim_value * 10 - attacker_value)
            
            # 5. Killer moves
            elif move in killer_set:
                score = 600000
                self.search_stats['killer_hits'] += 1
            
            # 6. Pawn advances and promotions
            elif move.promotion:
                score = 500000 + self.piece_values.get(move.promotion, 0)
            
            # 7. History heuristic and other quiet moves
            else:
                history_score = self.history_heuristic.get_history_score(move)
                score = history_score
            
            scored_moves.append((score, move))
        
        # Sort by score (descending)
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored_moves]
    
    def _evaluate_material(self, board: chess.Board) -> float:
        """
        Pure material evaluation with dynamic bishop pair bonus
        Matches Material Opponent's proven evaluation
        """
        score = 0
        
        # Count bishops for dynamic evaluation
        white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
        
        # Evaluate each piece type
        for piece_type in chess.PIECE_TYPES:
            if piece_type == chess.KING:
                continue
            
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            
            if piece_type == chess.BISHOP:
                # Dynamic bishop evaluation
                white_bishop_value = self.piece_values[chess.BISHOP]
                black_bishop_value = self.piece_values[chess.BISHOP]
                
                # Bishop pair bonus
                if white_bishops == 2:
                    white_bishop_value += self.bishop_pair_bonus // 2
                elif white_bishops == 1:
                    white_bishop_value -= self.bishop_alone_penalty // 2
                
                if black_bishops == 2:
                    black_bishop_value += self.bishop_pair_bonus // 2
                elif black_bishops == 1:
                    black_bishop_value -= self.bishop_alone_penalty // 2
                
                score += white_count * white_bishop_value
                score -= black_count * black_bishop_value
            else:
                # Standard piece values
                score += white_count * self.piece_values[piece_type]
                score -= black_count * self.piece_values[piece_type]
        
        # Small bonus for piece diversity (prefer pieces over pawns)
        white_pieces = sum(len(board.pieces(pt, chess.WHITE)) for pt in chess.PIECE_TYPES if pt != chess.KING)
        black_pieces = sum(len(board.pieces(pt, chess.BLACK)) for pt in chess.PIECE_TYPES if pt != chess.KING)
        score += (white_pieces - black_pieces) * 5
        
        # Return from current player's perspective
        return score if board.turn == chess.WHITE else -score
    
    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int) -> float:
        """
        Quiescence search for tactical stability
        Only searches captures and checks to avoid horizon effect
        """
        self.nodes_searched += 1
        
        # Depth limit
        if depth <= 0:
            return self._evaluate_material(board)
        
        # Stand pat evaluation
        stand_pat = self._evaluate_material(board)
        
        if stand_pat >= beta:
            return beta
        
        if stand_pat > alpha:
            alpha = stand_pat
        
        # Generate tactical moves (captures and checks)
        tactical_moves = []
        for move in board.legal_moves:
            if board.is_capture(move) or board.gives_check(move):
                tactical_moves.append(move)
        
        if not tactical_moves:
            return stand_pat
        
        # Sort tactical moves by MVV-LVA
        scored_moves = []
        for move in tactical_moves:
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                victim_value = self.piece_values.get(victim.piece_type, 0) if victim else 0
                attacker = board.piece_at(move.from_square)
                attacker_value = self.piece_values.get(attacker.piece_type, 0) if attacker else 0
                score = victim_value * 10 - attacker_value
                scored_moves.append((score, move))
            else:
                scored_moves.append((0, move))
        
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        
        # Search tactical moves
        for _, move in scored_moves:
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha, depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            
            if score > alpha:
                alpha = score
        
        return alpha
    
    def _probe_tt(self, board: chess.Board, depth: int, alpha: int, beta: int) -> Tuple[bool, int, Optional[chess.Move]]:
        """Probe transposition table"""
        zobrist_hash = self.zobrist.hash_position(board)
        
        if zobrist_hash in self.transposition_table:
            entry = self.transposition_table[zobrist_hash]
            
            if entry.depth >= depth:
                if entry.node_type == 'exact':
                    return True, entry.score, entry.best_move
                elif entry.node_type == 'lowerbound' and entry.score >= beta:
                    return True, entry.score, entry.best_move
                elif entry.node_type == 'upperbound' and entry.score <= alpha:
                    return True, entry.score, entry.best_move
            
            # Return TT move even if depth insufficient
            return False, 0, entry.best_move
        
        return False, 0, None
    
    def _store_tt(self, board: chess.Board, depth: int, score: int,
                  best_move: Optional[chess.Move], alpha: int, beta: int):
        """Store entry in transposition table"""
        zobrist_hash = self.zobrist.hash_position(board)
        
        # Determine node type
        if score <= alpha:
            node_type = 'upperbound'
        elif score >= beta:
            node_type = 'lowerbound'
        else:
            node_type = 'exact'
        
        # Simple replacement: clear oldest entries when full
        if len(self.transposition_table) >= self.max_tt_entries:
            entries = list(self.transposition_table.items())
            entries.sort(key=lambda x: x[1].depth, reverse=True)
            self.transposition_table = dict(entries[:int(self.max_tt_entries * 0.75)])
        
        entry = TranspositionEntry(depth, score, best_move, node_type, zobrist_hash)
        self.transposition_table[zobrist_hash] = entry
        self.search_stats['tt_stores'] += 1
    
    def _extract_pv(self, board: chess.Board, max_depth: int) -> List[chess.Move]:
        """Extract principal variation from TT"""
        pv = []
        temp_board = board.copy()
        
        for _ in range(min(max_depth, 10)):  # Limit PV length
            zobrist_hash = self.zobrist.hash_position(temp_board)
            
            if zobrist_hash in self.transposition_table:
                entry = self.transposition_table[zobrist_hash]
                if entry.best_move and entry.best_move in temp_board.legal_moves:
                    pv.append(entry.best_move)
                    temp_board.push(entry.best_move)
                else:
                    break
            else:
                break
        
        return pv
    
    def _has_non_pawn_pieces(self, board: chess.Board) -> bool:
        """Check if current side has non-pawn pieces (for null move pruning)"""
        for square in range(64):
            piece = board.piece_at(square)
            if piece and piece.color == board.turn and piece.piece_type != chess.PAWN:
                return True
        return False
    
    def _gives_checkmate(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if a move gives checkmate"""
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        return is_mate
    
    def _calculate_lmr_reduction(self, move: chess.Move, moves_searched: int, 
                                 depth: int, board: chess.Board) -> int:
        """
        Calculate Late Move Reduction
        Don't reduce: first 3 moves, tactical moves, low depths
        """
        # No reduction for first few moves
        if moves_searched < 3:
            return 0
        
        # No reduction at low depths
        if depth < 3:
            return 0
        
        # No reduction for tactical moves
        if board.is_capture(move) or board.gives_check(move):
            return 0
        
        # Calculate reduction
        reduction = 1
        
        if moves_searched >= 8:
            reduction += 1
        if depth >= 6:
            reduction += 1
        
        # Cap reduction
        return min(reduction, depth - 1)
    
    def _calculate_time_allocation(self, board: chess.Board, base_time: float) -> Tuple[float, float]:
        """
        Calculate time allocation based on position
        Returns (target_time, max_time)
        Uses V12.6's proven time management
        """
        moves_played = len(board.move_stack)
        legal_moves = list(board.legal_moves)
        num_moves = len(legal_moves)
        
        time_factor = 1.0
        
        # Game phase
        if moves_played < 15:  # Opening
            time_factor *= 0.8
        elif moves_played < 40:  # Middlegame
            time_factor *= 1.2
        else:  # Endgame
            time_factor *= 0.9
        
        # Position complexity
        if board.is_check():
            time_factor *= 1.3
        
        if num_moves <= 5:
            time_factor *= 0.7
        elif num_moves >= 35:
            time_factor *= 1.4
        
        # Calculate times
        target_time = min(base_time * time_factor * 0.8, base_time * 0.9)
        max_time = min(base_time * time_factor, base_time)
        
        # Store max time for emergency checks
        self.max_allocated_time = max_time
        
        return target_time, max_time
    
    def new_game(self):
        """Reset for new game"""
        self.transposition_table.clear()
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        self.nodes_searched = 0
        
        for key in self.search_stats:
            self.search_stats[key] = 0
