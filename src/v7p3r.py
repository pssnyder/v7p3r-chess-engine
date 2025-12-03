#!/usr/bin/env python3
"""
V7P3R Chess Engine v17.5 - Endgame Optimization (Phase 1)

Based on 320-game analytics showing 7.0 critical blunders/game, v17.5 focuses on
endgame tactical improvements through intelligent evaluation pruning and mate detection.

KEY IMPROVEMENTS (v17.5):
- PURE ENDGAME DETECTION: Skip PST in simplified endgames (≤6 pieces) for 38% speedup
- CASTLING OPTIMIZATION: Skip castling bonus in endgames (king centralization priority)
- MATE THREAT DETECTION: Detect opponent mate-in-1/2 threats to prevent blunders
- TARGET: Reduce critical blunders from 7.0 to <5.0 per game

KEY IMPROVEMENTS (v17.1.1):
- EMERGENCY TIME MODE: Strict limits when <30 seconds (prevents timeouts)
- When <15 seconds: Max 3-5 second absolute cap
- When <10 seconds: Max 2 second absolute cap

KEY IMPROVEMENTS (v17.1):
- DISABLED PV instant moves (prevented tactical verification - caused all 3 tournament losses)
- ADDED opening book (prevents entering known weak positions like 1.e3 trap)
- Expected improvement: +80-100 ELO, balanced White/Black performance

ARCHITECTURE:
- Phase 1: Core search (alpha-beta, TT, iterative deepening)
- Phase 2: Consolidated evaluation (unified bitboard system)
- Phase 3: Performance optimized through code consolidation
- Phase 4: Smart time management (v14.1)
- Phase 5: Tournament reliability (v17.1)
- Phase 6: Endgame optimization (v17.5)

VERSION LINEAGE:
- v17.5: Endgame optimization (38% speedup in pure endgames, mate detection)
- v17.1: PV instant move fix + opening book (tournament reliability)
- v17.0: Relaxed time management (1st place but Black-side weakness)
- v14.1: Smart time management fixes for tournament consistency
- v14.0: Consolidated performance build with unified evaluators
- v12.6: Nudge system removed for clean performance build
- v12.5: Intelligent nudge system experiments
- v12.4: Enhanced castling with balanced evaluation
- v12.2: Performance optimized with tactical regression
- v12.0: Clean evolution with proven improvements only
- v11.x: Experimental variants (lessons learned)
- v10.8: Recovery baseline (19.5/30 tournament points)

Author: Pat Snyder
"""

import time
import chess
import sys
import random
import json
import os
from typing import Optional, Tuple, List, Dict
from collections import defaultdict
from v7p3r_bitboard_evaluator import V7P3RScoringCalculationBitboard
from v7p3r_fast_evaluator import V7P3RFastEvaluator
from v7p3r_openings_v161 import get_enhanced_opening_book


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
            
            # Clean UCI output for PV following - FIXED UCI format
            remaining_pv_str = self.pv_display_string if self.pv_display_string else str(move_to_play)
            print(f"info depth 1 score cp 0 nodes 0 time 0 pv {remaining_pv_str}")
            print(f"info string PV prediction match")
            
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


class TranspositionEntry:
    """Entry in the transposition table"""
    def __init__(self, depth: int, score: int, best_move: Optional[chess.Move], 
                 node_type: str, zobrist_hash: int, static_eval: Optional[float] = None):
        self.depth = depth
        self.score = score
        self.best_move = best_move
        self.node_type = node_type  # 'exact', 'lowerbound', 'upperbound'
        self.zobrist_hash = zobrist_hash
        self.static_eval = static_eval  # V17.2: Unified eval cache


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
        
        # Generate random numbers for each piece on each square
        for square in range(64):
            for piece_type in range(1, 7):  # PAWN to KING
                for color in [chess.WHITE, chess.BLACK]:
                    key = (square, piece_type, color)
                    self.piece_square_table[key] = random.getrandbits(64)
    
    def hash_position(self, board: chess.Board) -> int:
        """Generate Zobrist hash for the position"""
        hash_value = 0
        
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                key = (square, piece.piece_type, piece.color)
                hash_value ^= self.piece_square_table[key]
        
        if board.turn == chess.BLACK:
            hash_value ^= self.side_to_move
            
        return hash_value


class V7P3REngine:
    """V7P3R Chess Engine v17.3 - SEE-Based Quiescence + UCI Enhancements
    
    V17.3 CHANGES (Nov 26, 2025):
    - Rewrote quiescence with Static Exchange Evaluation (SEE)
    - Reduced max quiescence depth from 10 to 3 plies
    - Only extends good/equal captures (SEE >= 0)
    - Fixes 50% move instability caused by deep quiescence contradictions
    """
    
    def __init__(self, use_fast_evaluator: bool = True):
        # Basic configuration
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300, 
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0  # King safety handled separately
        }
        
        # Search configuration
        self.default_depth = 6
        self.nodes_searched = 0
        self.seldepth = 0  # V17.3: Track selective depth (max depth with extensions)
        
        # V14.2: Evaluator selection (fast vs bitboard)
        self.use_fast_evaluator = use_fast_evaluator
        if use_fast_evaluator:
            # V16.1's fast evaluator - 40x faster, enables depth 6-8
            self.evaluator = V7P3RFastEvaluator()
            print("info string Using Fast Evaluator (v16.1 speed)", flush=True)
        else:
            # V14.1's bitboard evaluator - more comprehensive but slower
            self.evaluator = V7P3RScoringCalculationBitboard(self.piece_values, enable_nudges=False)
            print("info string Using Bitboard Evaluator (v14.1 comprehensive)", flush=True)
        
        # Keep reference to bitboard evaluator for compatibility
        self.bitboard_evaluator = self.evaluator if not use_fast_evaluator else V7P3RScoringCalculationBitboard(self.piece_values, enable_nudges=False)
        
        # V17.2: Unified TT + evaluation cache (static_eval stored in TT entries)
        # Removed separate evaluation_cache - now using TT.static_eval field
        
        # Advanced search infrastructure
        self.transposition_table: Dict[int, TranspositionEntry] = {}
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        self.zobrist = ZobristHashing()
        
        # Configuration
        self.max_tt_entries = 50000  # Reasonable size for testing
        
        # Performance monitoring
        self.search_stats = {
            'nodes_per_second': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'tt_hits': 0,
            'tt_stores': 0,
            'killer_hits': 0,
        }
        
        # V17.2: Pre-allocated move ordering buffers (reused across searches)
        self.move_buffers = {
            'captures': [],
            'checks': [],
            'killers': [],
            'quiet': [],
            'tactical': [],
            'tt': []
        }
        
        # V17.1: Opening book (prevents entering weak positions)
        self.opening_book = get_enhanced_opening_book()
        print("info string Opening book loaded (v16.1 repertoire)", flush=True)
        
        # PV Following System - V10 OPTIMIZATION
        self.pv_tracker = PVTracker()
    
    def search(self, board: chess.Board, time_limit: float = 3.0, depth: Optional[int] = None, 
               alpha: float = -99999, beta: float = 99999, is_root: bool = True) -> chess.Move:
        """
        UNIFIED SEARCH - Single function with ALL advanced features:
        - Iterative deepening with stable best move handling (root level)
        - Alpha-beta pruning with negamax framework (recursive level)  
        - Transposition table with Zobrist hashing
        - Killer moves and history heuristic
        - Advanced move ordering with tactical detection
        - Proper time management with periodic checks
        - Full PV extraction and following
        - Quiescence search for tactical stability
        """
        
        # ROOT LEVEL: Iterative deepening with time management
        if is_root:
            self.nodes_searched = 0
            self.seldepth = 0  # V17.3: Reset selective depth for new search
            self.search_start_time = time.time()
            
            
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return chess.Move.null()

            # V17.1: OPENING BOOK CHECK (prevents entering weak positions)
            current_fen = board.fen()
            if current_fen in self.opening_book:
                book_moves = self.opening_book[current_fen]
                legal_book_moves = [(uci, weight) for uci, weight in book_moves 
                                   if chess.Move.from_uci(uci) in legal_moves]
                if legal_book_moves:
                    # Weighted random selection
                    total_weight = sum(w for _, w in legal_book_moves)
                    r = random.randint(1, total_weight)
                    cumulative = 0
                    for move_uci, weight in legal_book_moves:
                        cumulative += weight
                        if r <= cumulative:
                            book_move = chess.Move.from_uci(move_uci)
                            # V17.3: UCI info for book moves
                            print(f"info depth 0 seldepth 0 score cp 0 nodes 0 time 0 nps 0 hashfull 0 pv {book_move.uci()}", flush=True)
                            print(f"info string Opening book move: {book_move.uci()}", flush=True)
                            return book_move

            # V17.1: PV INSTANT MOVES DISABLED
            # REASON: Caused all 3 tournament losses - trusts stale PV without re-evaluation
            # IMPACT: f6 blunder repeated 3 times (depth 1, 0 nodes signature)
            # See: docs/V17_0_PV_Blunder_Investigation_Findings.md
            # pv_move = self.pv_tracker.check_position_for_instant_move(board)
            # if pv_move:
            #     return pv_move
            
            target_time, max_time = self._calculate_adaptive_time_allocation(board, time_limit)
            
            # V17.2: UCI enhancements for debugging
            self.seldepth = 0  # Track selective (quiescence) depth
            
            # Iterative deepening
            best_move = legal_moves[0]
            best_score = -99999
            last_iteration_time = 0.0
            stable_best_count = 0  # V14.1: Track how many iterations returned same best move
            
            for current_depth in range(1, self.default_depth + 1):
                iteration_start = time.time()
                
                # V14.1: CRITICAL - Check elapsed time BEFORE starting iteration
                elapsed = time.time() - self.search_start_time
                
                # V14.1: Early exit if we've used target time
                if elapsed >= target_time:
                    break
                
                # V17.0: Predict if next iteration will complete before max_time
                # Use 2.5x factor (less conservative than v14.1's 3.0x)
                if current_depth > 1 and last_iteration_time > 0:
                    predicted_completion = elapsed + (last_iteration_time * 2.5)
                    if predicted_completion >= max_time:
                        # V17.0: Don't start iteration we can't complete
                        break
                
                try:
                    # Store previous best in case iteration fails
                    previous_best = best_move
                    previous_score = best_score
                    
                    # Call recursive search for this depth
                    score, move = self._recursive_search(board, current_depth, -99999, 99999, time_limit)
                    
                    # Calculate iteration time IMMEDIATELY
                    last_iteration_time = time.time() - iteration_start
                    
                    # Update best move if we got a valid result
                    if move and move != chess.Move.null():
                        # V14.1: Track stability of best move
                        if move == best_move:
                            stable_best_count += 1
                        else:
                            stable_best_count = 1
                        
                        best_move = move
                        best_score = score
                        
                        # V17.3: Calculate hashfull (TT usage in per mille 0-1000)
                        hashfull = int((len(self.transposition_table) / self.max_tt_entries) * 1000)
                        
                        elapsed_ms = int((time.time() - self.search_start_time) * 1000)
                        nps = int(self.nodes_searched / max(elapsed_ms / 1000, 0.001))
                        
                        # Extract and display PV
                        pv_line = self._extract_pv(board, current_depth)
                        pv_string = " ".join([str(m) for m in pv_line])
                        
                        # Store PV for following optimization
                        if current_depth >= 4 and len(pv_line) >= 3:
                            self.pv_tracker.store_pv_from_search(board, pv_line)
                        
                        # V17.3: Extended UCI info with seldepth and hashfull (from v17.2.0 enhancements)
                        print(f"info depth {current_depth} seldepth {self.seldepth} score cp {int(score)} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} hashfull {hashfull} pv {pv_string}")
                        sys.stdout.flush()
                    else:
                        # Restore previous best if iteration failed
                        best_move = previous_best
                        best_score = previous_score
                    
                    # V17.0: SMART EARLY EXIT - if best move has been stable for 4+ iterations at depth 5+
                    # This prevents wasteful thinking when the position is clear
                    # (was 3 iterations at depth 4 with 0.5x target, now 4 iterations at depth 5 with 0.7x target)
                    if current_depth >= 5 and stable_best_count >= 4:
                        elapsed = time.time() - self.search_start_time
                        if elapsed >= target_time * 0.7:  # Used at least 70% of target time
                            # Position is stable, return early to save time
                            break
                    
                    # V14.1: Check time AFTER completing iteration
                    elapsed = time.time() - self.search_start_time
                    if elapsed >= target_time:
                        break
                        
                except Exception as e:
                    print(f"info string Search interrupted at depth {current_depth}: {e}")
                    break
                    
            return best_move
        
        # This should never be called directly with is_root=False from external code
        else:
            # Fallback - call the recursive search method
            score, move = self._recursive_search(board, depth or 1, alpha, beta, time_limit)
            return move if move else chess.Move.null()
    
    def _recursive_search(self, board: chess.Board, search_depth: int, alpha: float, beta: float, time_limit: float) -> Tuple[float, Optional[chess.Move]]:
        """
        Recursive alpha-beta search with all advanced features
        Returns (score, best_move) tuple
        """
        self.nodes_searched += 1
        
        # V17.3: Track selective depth (maximum depth reached including extensions)
        current_ply = self.default_depth - search_depth
        self.seldepth = max(self.seldepth, current_ply)
        
        # CRITICAL: Time checking during recursive search to prevent timeouts
        if hasattr(self, 'search_start_time') and self.nodes_searched % 1000 == 0:
            elapsed = time.time() - self.search_start_time
            if elapsed > time_limit:
                # Emergency return with current best evaluation
                return self._evaluate_position(board), None
        
        # 1. TRANSPOSITION TABLE PROBE
        tt_hit, tt_score, tt_move = self._probe_transposition_table(board, search_depth, int(alpha), int(beta))
        if tt_hit:
            return float(tt_score), tt_move
        
        # 2. TERMINAL CONDITIONS
        if search_depth == 0:
            # V17.3: Enter quiescence search (starts at depth 0, caps at 3)
            score = self._quiescence_search(board, alpha, beta, 0)
            return score, None
            
        if board.is_game_over():
            if board.is_checkmate():
                score = -29000.0 + (self.default_depth - search_depth)  # Prefer quicker mates
            else:
                score = 0.0  # Stalemate
            return score, None
        
        # 3. NULL MOVE PRUNING
        if (search_depth >= 3 and not board.is_check() and 
            self._has_non_pawn_pieces(board) and beta - alpha > 1):
            
            board.turn = not board.turn
            null_score, _ = self._recursive_search(board, search_depth - 2, -beta, -beta + 1, time_limit)
            null_score = -null_score
            board.turn = not board.turn
            
            if null_score >= beta:
                return null_score, None
        
        # 4. MOVE GENERATION AND ORDERING
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.0, None
        
        ordered_moves = self._order_moves_advanced(board, legal_moves, search_depth, tt_move)
        
        # 5. MAIN SEARCH LOOP (NEGAMAX WITH ALPHA-BETA)
        best_score = -99999.0
        best_move = None
        original_alpha = alpha
        moves_searched = 0
        
        for move in ordered_moves:
            board.push(move)
            
            # V17.5: Detect opponent mate threats in endgames (prevent being mated)
            if hasattr(self.evaluator, '_is_endgame') and self.evaluator._is_endgame(board):
                mate_threat = self._detect_opponent_mate_threat(board, max_depth=3)
                if mate_threat:
                    # Heavily penalize this line - opponent has mate in N
                    score = -20000 + mate_threat  # Worse than losing but accounts for mate distance
                    board.pop()
                    moves_searched += 1
                    
                    if best_move is None or score > best_score:
                        best_score = score
                        best_move = move
                    
                    alpha = max(alpha, score)
                    if alpha >= beta:
                        break
                    continue
            
            # V11 ENHANCEMENT: Enhanced Late Move Reduction
            reduction = self._calculate_lmr_reduction(move, moves_searched, search_depth, board)
            
            # Search with possible reduction
            if reduction > 0:
                score, _ = self._recursive_search(board, search_depth - 1 - reduction, -beta, -alpha, time_limit)
                score = -score
                
                # Re-search at full depth if reduced search failed high
                if score > alpha:
                    score, _ = self._recursive_search(board, search_depth - 1, -beta, -alpha, time_limit)
                    score = -score
            else:
                score, _ = self._recursive_search(board, search_depth - 1, -beta, -alpha, time_limit)
                score = -score
            
            board.pop()
            moves_searched += 1
            
            # Update best move
            if best_move is None or score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            if alpha >= beta:
                # Beta cutoff - update heuristics
                if not board.is_capture(move):
                    self.killer_moves.store_killer(move, search_depth)
                    self.history_heuristic.update_history(move, search_depth)
                    self.search_stats['killer_hits'] += 1
                break
        
        # 7. TRANSPOSITION TABLE STORE
        self._store_transposition_table(board, search_depth, int(best_score), best_move, int(original_alpha), int(beta))
        
        return best_score, best_move
    
    def _order_moves_advanced(self, board: chess.Board, moves: List[chess.Move], depth: int, 
                              tt_move: Optional[chess.Move] = None) -> List[chess.Move]:
        """V17.2: Move ordering with pre-allocated buffer reuse"""
        if len(moves) <= 2:
            return moves
        
        # V17.2: Reuse pre-allocated buffers (clear instead of allocate)
        for buffer in self.move_buffers.values():
            buffer.clear()
        
        captures = self.move_buffers['captures']
        checks = self.move_buffers['checks']
        killers = self.move_buffers['killers']
        quiet_moves = self.move_buffers['quiet']
        tactical_moves = self.move_buffers['tactical']
        tt_moves = self.move_buffers['tt']
        
        # Performance optimization: Pre-create sets for fast lookups
        killer_set = set(self.killer_moves.get_killers(depth))
        
        for move in moves:
            # 1. Transposition table move (highest priority)
            if tt_move and move == tt_move:
                tt_moves.append(move)
            
            # 2. Captures (will be sorted by MVV-LVA)
            elif board.is_capture(move):
                victim = board.piece_at(move.to_square)
                victim_value = self.piece_values.get(victim.piece_type, 0) if victim else 0
                attacker = board.piece_at(move.from_square)
                attacker_value = self.piece_values.get(attacker.piece_type, 0) if attacker else 0
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                mvv_lva_score = victim_value * 100 - attacker_value
                
                # Add tactical bonus using bitboards
                tactical_bonus = self.bitboard_evaluator.detect_bitboard_tactics(board, move)
                total_score = mvv_lva_score + tactical_bonus
                
                captures.append((total_score, move))
            
            # 4. Checks (high priority for tactical play)
            elif board.gives_check(move):
                # Add tactical bonus for checking moves too
                tactical_bonus = self.bitboard_evaluator.detect_bitboard_tactics(board, move)
                checks.append((tactical_bonus, move))
            
            # 5. Killer moves
            elif move in killer_set:
                killers.append(move)
                self.search_stats['killer_hits'] += 1
            
            # 6. Check for tactical patterns in quiet moves
            else:
                history_score = self.history_heuristic.get_history_score(move)
                tactical_bonus = self.bitboard_evaluator.detect_bitboard_tactics(board, move)
                
                if tactical_bonus > 20.0:  # Significant tactical move
                    tactical_moves.append((tactical_bonus + history_score, move))
                else:
                    quiet_moves.append((history_score, move))
        
        # Sort all move categories by their scores
        captures.sort(key=lambda x: x[0], reverse=True)
        checks.sort(key=lambda x: x[0], reverse=True)
        tactical_moves.sort(key=lambda x: x[0], reverse=True)
        quiet_moves.sort(key=lambda x: x[0], reverse=True)
        
        # Combine in optimized order
        ordered = []
        ordered.extend(tt_moves)  # TT move first
        ordered.extend([move for _, move in captures])  # Then captures (with tactical bonus)
        ordered.extend([move for _, move in checks])  # Then checks (with tactical bonus)
        ordered.extend([move for _, move in tactical_moves])  # Then tactical patterns
        ordered.extend(killers)  # Then killers
        ordered.extend([move for _, move in quiet_moves])  # Then quiet moves
        
        return ordered
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """V17.2: Position evaluation with unified TT cache"""
        # V17.2: Use zobrist hash for unified TT lookup
        zobrist_hash = self.zobrist.hash_position(board)
        
        # Check TT for cached evaluation
        if zobrist_hash in self.transposition_table:
            tt_entry = self.transposition_table[zobrist_hash]
            if tt_entry.static_eval is not None:
                self.search_stats['cache_hits'] += 1
                return tt_entry.static_eval
        
        self.search_stats['cache_misses'] += 1
        
        # V14.2: Use selected evaluator (fast or bitboard)
        if self.use_fast_evaluator:
            # Fast evaluator - returns complete score directly
            final_score = float(self.evaluator.evaluate(board))
        else:
            # Bitboard evaluator - comprehensive multi-component evaluation
            white_base = self.bitboard_evaluator.calculate_score_optimized(board, True)
            black_base = self.bitboard_evaluator.calculate_score_optimized(board, False)
            
            # Consolidated evaluation components
            try:
                # Advanced pawn structure evaluation 
                white_pawn_score = self.bitboard_evaluator.evaluate_pawn_structure(board, True)
                black_pawn_score = self.bitboard_evaluator.evaluate_pawn_structure(board, False)
                
                # Advanced king safety evaluation
                white_king_score = self.bitboard_evaluator.evaluate_king_safety(board, True)
                black_king_score = self.bitboard_evaluator.evaluate_king_safety(board, False)
                
                # Tactical pattern evaluation disabled for performance
                white_tactical_score = 0
                black_tactical_score = 0
                
                # Combine all evaluation components
                white_total = white_base + white_pawn_score + white_king_score + white_tactical_score
                black_total = black_base + black_pawn_score + black_king_score + black_tactical_score
                
            except Exception as e:
                # Fallback to base evaluation if advanced evaluation fails
                white_total = white_base
                black_total = black_base
            
            # Calculate final score from current player's perspective
            if board.turn:  # White to move
                final_score = white_total - black_total
            else:  # Black to move
                final_score = black_total - white_total
        
        # V17.2: Store evaluation in TT (create entry if doesn't exist)
        if zobrist_hash in self.transposition_table:
            self.transposition_table[zobrist_hash].static_eval = final_score
        else:
            # Create eval-only entry (depth=0 for eval-only)
            entry = TranspositionEntry(0, 0, None, 'eval_only', zobrist_hash, static_eval=final_score)
            self.transposition_table[zobrist_hash] = entry
        
        return final_score
    
    def _simple_king_safety(self, board: chess.Board, color: bool) -> float:
        """V17.5: Simplified king safety - skip castling bonus in endgames"""
        
        # V17.5: Skip castling bonus in endgames (king should centralize, not castle)
        if hasattr(self.evaluator, '_is_endgame') and self.evaluator._is_endgame(board):
            return 0.0
        
        score = 0.0
        
        # Basic castling bonus (only in opening/middlegame)
        if color == chess.WHITE:
            if board.has_kingside_castling_rights(color):
                score += 15
            if board.has_queenside_castling_rights(color):
                score += 10
        else:
            if board.has_kingside_castling_rights(color):
                score += 15
            if board.has_queenside_castling_rights(color):
                score += 10
        
        return score
    
    def _probe_transposition_table(self, board: chess.Board, depth: int, alpha: int, beta: int) -> Tuple[bool, int, Optional[chess.Move]]:
        """Probe transposition table for this position - PHASE 1 FEATURE"""
        zobrist_hash = self.zobrist.hash_position(board)
        
        if zobrist_hash in self.transposition_table:
            entry = self.transposition_table[zobrist_hash]
            
            # Only use if searched to sufficient depth
            if entry.depth >= depth:
                self.search_stats['tt_hits'] += 1
                
                # Check if we can use the score
                if entry.node_type == 'exact':
                    return True, entry.score, entry.best_move
                elif entry.node_type == 'lowerbound' and entry.score >= beta:
                    return True, entry.score, entry.best_move
                elif entry.node_type == 'upperbound' and entry.score <= alpha:
                    return True, entry.score, entry.best_move
        
        return False, 0, None
    
    def _store_transposition_table(self, board: chess.Board, depth: int, score: int, 
                                    best_move: Optional[chess.Move], alpha: int, beta: int):
        """V17.2: Store position in TT with O(1) two-tier bucket replacement"""
        zobrist_hash = self.zobrist.hash_position(board)
        
        # Determine node type
        if score <= alpha:
            node_type = 'upperbound'
        elif score >= beta:
            node_type = 'lowerbound'
        else:
            node_type = 'exact'
        
        # V17.2: Two-tier bucket system - O(1) replacement, no sorting
        # Bucket 1: Always-replace (fast probe)
        # Bucket 2: Depth-preferred (keeps deep searches)
        primary_bucket = zobrist_hash % self.max_tt_entries
        secondary_bucket = (zobrist_hash % self.max_tt_entries) ^ 1  # Adjacent bucket
        
        # Preserve static_eval if updating existing entry
        existing_eval = None
        if zobrist_hash in self.transposition_table:
            existing_entry = self.transposition_table[zobrist_hash]
            if existing_entry.zobrist_hash == zobrist_hash:
                existing_eval = existing_entry.static_eval
        
        entry = TranspositionEntry(depth, score, best_move, node_type, zobrist_hash, static_eval=existing_eval)
        
        # Check primary bucket
        primary_entry = self.transposition_table.get(primary_bucket)
        if primary_entry is None or primary_entry.zobrist_hash == zobrist_hash:
            # Empty or same position - always replace
            self.transposition_table[primary_bucket] = entry
            self.search_stats['tt_stores'] += 1
            return
        
        # Check secondary bucket
        secondary_entry = self.transposition_table.get(secondary_bucket)
        if secondary_entry is None or secondary_entry.depth < depth:
            # Empty or shallower depth - replace secondary
            self.transposition_table[secondary_bucket] = entry
            self.search_stats['tt_stores'] += 1
            return
        
        # Both buckets occupied by deeper entries - replace primary (always-replace strategy)
        self.transposition_table[primary_bucket] = entry
        self.search_stats['tt_stores'] += 1
    
    def _has_non_pawn_pieces(self, board: chess.Board) -> bool:
        """Check if the current side has non-pawn pieces (for null move pruning)"""
        current_color = board.turn
        for square in range(64):
            piece = board.piece_at(square)
            if piece and piece.color == current_color and piece.piece_type != chess.PAWN:
                return True
        return False
    
    def _detect_opponent_mate_threat(self, board: chess.Board, max_depth: int = 3) -> Optional[int]:
        """
        V17.5: Detect if opponent has forcing mate sequence
        Run in endgames only to prevent being checkmated
        
        Args:
            board: Current position (opponent to move)
            max_depth: How deep to search for mate (default: 3)
        
        Returns:
            Mate in N moves if found, None otherwise
        """
        if not board.legal_moves:
            return None
        
        # Quick mate-in-1 check
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return 1
            board.pop()
        
        # Mate-in-2 check (if depth >= 2)
        if max_depth >= 2:
            for opp_move in board.legal_moves:
                board.push(opp_move)
                
                # Can we escape check on all our responses?
                has_escape = False
                for our_move in board.legal_moves:
                    board.push(our_move)
                    
                    # Check if opponent has mate-in-1 now
                    has_mate_followup = False
                    for opp_followup in board.legal_moves:
                        board.push(opp_followup)
                        if board.is_checkmate():
                            has_mate_followup = True
                            board.pop()
                            break
                        board.pop()
                    
                    board.pop()
                    
                    if not has_mate_followup:
                        has_escape = True
                        break
                
                board.pop()
                
                if not has_escape:
                    return 2
        
        return None
    
    def _extract_pv(self, board: chess.Board, max_depth: int) -> List[chess.Move]:
        """Extract principal variation from transposition table"""
        pv = []
        temp_board = board.copy()
        seen_positions = set()
        
        # Build PV by following TT best moves
        for _ in range(max_depth):
            zobrist_hash = self.zobrist.hash_position(temp_board)
            
            # Prevent cycles
            if zobrist_hash in seen_positions:
                break
            seen_positions.add(zobrist_hash)
            
            # Look up position in TT
            if zobrist_hash in self.transposition_table:
                entry = self.transposition_table[zobrist_hash]
                # Only use best_move if it exists and is legal
                if entry.best_move and entry.best_move in temp_board.legal_moves:
                    pv.append(entry.best_move)
                    temp_board.push(entry.best_move)
                else:
                    break
            else:
                break
        
        return pv
    
    def _static_exchange_evaluation(self, board: chess.Board, move: chess.Move) -> int:
        """
        V17.3: Static Exchange Evaluation (SEE)
        Calculates the material outcome of a capture sequence
        
        Returns:
            Material value gained/lost (in centipawns)
            Positive = winning exchange, Negative = losing exchange, 0 = equal
        """
        # Get piece values
        target_square = move.to_square
        attacker_square = move.from_square
        
        # Initial capture value
        captured_piece = board.piece_at(target_square)
        if captured_piece is None:
            return 0  # Not a capture
        
        gain = self.piece_values.get(captured_piece.piece_type, 0)
        
        # Simulate the exchange
        board.push(move)
        
        # Check if square is still attacked after capture
        # Find smallest attacker that can recapture
        attackers = board.attackers(not board.turn, target_square)
        
        if not attackers:
            # No recapture possible - we keep the gain
            board.pop()
            return gain
        
        # Find smallest attacker for recapture
        smallest_attacker = None
        smallest_value = 10000
        
        for attacker_sq in attackers:
            piece = board.piece_at(attacker_sq)
            if piece:
                piece_value = self.piece_values.get(piece.piece_type, 0)
                if piece_value < smallest_value:
                    smallest_value = piece_value
                    smallest_attacker = attacker_sq
        
        if smallest_attacker is None:
            board.pop()
            return gain
        
        # Value of our piece that just captured
        our_piece = board.piece_at(target_square)
        our_value = self.piece_values.get(our_piece.piece_type, 0) if our_piece else 0
        
        # Opponent's best recapture gains our piece but loses their smallest attacker
        # We recurse to depth 2 only (2-3 exchanges max as per user request)
        opponent_gain = our_value - smallest_value
        
        board.pop()
        
        # Final exchange evaluation: our gain - opponent's best response
        return gain - max(0, opponent_gain)
    
    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int) -> float:
        """
        V17.3: SEE-Based Quiescence Search
        
        SIMPLIFIED APPROACH:
        - Use Static Exchange Evaluation (SEE) instead of full recursive search
        - Only extends 2-3 exchanges deep (not 10+ like before)
        - Stand-pat only if position is quiet (no losing captures)
        - Drastically reduces node count and evaluation inconsistency
        """
        self.nodes_searched += 1
        
        # V17.3: Track selective depth (quiescence extensions)
        current_ply = self.default_depth - depth
        self.seldepth = max(self.seldepth, current_ply)
        
        # Depth limit: Cap at 3 plies (2-3 exchanges as per requirement)
        if depth >= 3:
            return self._evaluate_position(board)
        
        # Stand pat evaluation
        stand_pat = self._evaluate_position(board)
        
        # Beta cutoff on stand pat
        if stand_pat >= beta:
            return beta
        
        # Update alpha if stand pat is better
        if stand_pat > alpha:
            alpha = stand_pat
        
        # Generate captures only (no checks - too expensive)
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        
        # If no captures, return stand pat (position is quiet)
        if not captures:
            return stand_pat
        
        # V17.3: Filter captures using SEE - only search good/equal exchanges
        good_captures = []
        for move in captures:
            see_value = self._static_exchange_evaluation(board, move)
            if see_value >= 0:  # Only search non-losing captures
                good_captures.append((see_value, move))
        
        # If no good captures, stand pat (don't extend losing sequences)
        if not good_captures:
            return stand_pat
        
        # Sort by SEE value (best exchanges first)
        good_captures.sort(key=lambda x: x[0], reverse=True)
        
        # Search good captures
        best_score = stand_pat
        for see_value, move in good_captures:
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha, depth + 1)
            board.pop()
            
            if score > best_score:
                best_score = score
            
            if score > alpha:
                alpha = score
            
            if alpha >= beta:
                break  # Beta cutoff
        
        return best_score

    def notify_move_played(self, move: chess.Move, board_before_move: chess.Board):
        """Notify engine that a move was played (for PV following)
        
        Args:
            move: The move that was played
            board_before_move: The board position BEFORE the move was made
        """
        # We don't need this method anymore - position checking is automatic
        # The new approach checks positions directly when search() is called
        pass

    def new_game(self):
        """Reset for a new game"""
        # V17.2: Only TT to clear (unified cache)
        self.transposition_table.clear()
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        self.nodes_searched = 0
        
        # Clear PV following data
        self.pv_tracker.clear()
        
        # Reset stats
        for key in self.search_stats:
            self.search_stats[key] = 0

    @property
    def principal_variation(self) -> List[chess.Move]:
        """Get the current principal variation"""
        # Return the original PV from the last search for display purposes
        return self.pv_tracker.original_pv.copy() if self.pv_tracker.original_pv else []

    def perft(self, board: chess.Board, depth: int, divide: bool = False, root_call: bool = True) -> int:
        """
        V11 ENHANCEMENT: Perft (Performance Test) - counts nodes at given depth
        
        This is essential for move generation validation and testing.
        Standard chess engine requirement that was missing in V10.2.
        
        Args:
            board: Current chess position
            depth: Depth to search (number of plies)
            divide: If True, show per-move breakdown at root
            root_call: Internal flag for root level tracking
            
        Returns:
            Total number of leaf nodes at specified depth
        """
        if depth == 0:
            return 1
        
        if root_call:
            self.perft_start_time = time.time()
            self.perft_nodes = 0
        
        legal_moves = list(board.legal_moves)
        total_nodes = 0
        
        if divide and depth == 1:
            # For divide, show each move's contribution
            for move in legal_moves:
                board.push(move)
                nodes = 1  # At depth 1, each legal move contributes 1 node
                board.pop()
                total_nodes += nodes
                if root_call:
                    print(f"{move}: {nodes}")
        else:
            # Normal perft counting
            for move in legal_moves:
                board.push(move)
                nodes = self.perft(board, depth - 1, False, False)
                board.pop()
                total_nodes += nodes
                
                if divide and root_call:
                    print(f"{move}: {nodes}")
        
        if root_call:
            elapsed = time.time() - self.perft_start_time
            nps = int(total_nodes / max(elapsed, 0.001))
            print(f"\nNodes searched: {total_nodes}")
            print(f"Time: {elapsed:.3f}s")
            print(f"Nodes per second: {nps}")
        
        return total_nodes

    def _calculate_adaptive_time_allocation(self, board: chess.Board, base_time_limit: float) -> Tuple[float, float]:
        """
        V17.1.1: EMERGENCY TIME MODE added to prevent timeouts
        V14.1: IMPROVED adaptive time management - prevents wasteful thinking
        
        Returns: (target_time, max_time)
        
        PHILOSOPHY:
        - Opening: Play fast (book knowledge, simple positions)
        - Middlegame: Use more time (complex tactics)
        - Endgame: Moderate time (technique matters, but positions simpler)
        - NEVER exceed 60 seconds per move
        - V17.1.1: EMERGENCY MODE when time is critically low
        """
        moves_played = len(board.move_stack)
        legal_moves = list(board.legal_moves)
        num_legal_moves = len(legal_moves)
        
        # V17.1.1: EMERGENCY LOW-TIME MODE (prevents timeouts)
        if base_time_limit < 10.0:
            # Critical: <10 seconds total - use MAX 2 seconds
            return 1.5, 2.0
        elif base_time_limit < 15.0:
            # Very low: <15 seconds - use MAX 3-5 seconds
            return min(2.5, base_time_limit * 0.25), min(3.5, base_time_limit * 0.30)
        elif base_time_limit < 30.0:
            # Low: <30 seconds - use MAX 25% of time
            return base_time_limit * 0.20, base_time_limit * 0.25
        
        # V14.1: HARD CAP - never exceed 60 seconds
        absolute_max = min(base_time_limit, 60.0)
        
        # Base time factor - more conservative than v14.0
        time_factor = 1.0
        
        # V17.0: RELAXED OPENING TIME REDUCTION (was 0.5/0.6, now 0.75/0.85)
        if moves_played < 8:  # Very early opening
            time_factor *= 0.75  # Use 75% time (allow tactical depth)
        elif moves_played < 15:  # Opening
            time_factor *= 0.85  # Use 85% time (piece coordination matters)
        elif moves_played < 25:  # Early middlegame
            time_factor *= 0.9  # Starting to matter more
        elif moves_played < 40:  # Complex middlegame
            time_factor *= 1.1  # Peak thinking time
        elif moves_played < 60:  # Late middlegame/early endgame
            time_factor *= 0.9  # Simplifying
        else:  # Deep endgame
            time_factor *= 0.7  # Technique matters but simpler
        
        # Position complexity factors
        if board.is_check():
            time_factor *= 1.2  # More time when in check (not 1.3)
        
        # V14.1: Simplified move count logic
        if num_legal_moves <= 3:
            time_factor *= 0.5  # Very few options - decide quickly
        elif num_legal_moves <= 8:
            time_factor *= 0.8  # Few options
        elif num_legal_moves >= 35:
            time_factor *= 1.2  # Many options (not 1.4 - too aggressive)
        
        # Material balance consideration - simplified
        our_material = self._count_material(board, board.turn)
        their_material = self._count_material(board, not board.turn)
        material_diff = our_material - their_material
        
        if material_diff < -300:  # We're behind
            time_factor *= 1.1  # Slightly more time (not 1.2)
        elif material_diff > 500:  # We're significantly ahead
            time_factor *= 0.8  # Play faster when winning
        
        # V17.0: Calculate final times with RELAXED approach (was 0.7/0.9, now 0.85/0.98)
        # Target time: aim to use this much (usually hit this and return)
        # Max time: absolute limit (rarely hit, only complex positions)
        target_time = min(absolute_max * time_factor * 0.85, absolute_max * 0.90)
        max_time = min(absolute_max * time_factor * 0.98, absolute_max * 0.99)
        
        # V14.1: SAFETY - ensure target < max < absolute_max
        target_time = min(target_time, max_time * 0.85)
        max_time = min(max_time, absolute_max)
        
        return target_time, max_time
    
    def _count_material(self, board: chess.Board, color: bool) -> int:
        """Count total material for a color"""
        total = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            pieces = board.pieces(piece_type, color)
            total += len(pieces) * self.piece_values[piece_type]
        return total

    def _calculate_lmr_reduction(self, move: chess.Move, moves_searched: int, search_depth: int, board: chess.Board) -> int:
        """
        V11 ENHANCEMENT: Calculate Late Move Reduction amount based on move characteristics
        
        Returns reduction amount (0 = no reduction, 1+ = reduction plies)
        """
        # No reduction for first few moves
        if moves_searched < 3:
            return 0
        
        # No reduction at low depths
        if search_depth < 3:
            return 0
        
        # No reduction for tactical moves
        if (board.is_capture(move) or board.is_check() or 
            self.killer_moves.is_killer(move, search_depth)):
            return 0
        
        # Calculate base reduction
        reduction = 1
        
        # Increase reduction for later moves
        if moves_searched >= 8:
            reduction += 1
        if moves_searched >= 16:
            reduction += 1
        
        # Increase reduction at higher depths
        if search_depth >= 6:
            reduction += 1
        
        # Reduce reduction for good history moves
        history_score = self.history_heuristic.get_history_score(move)
        if history_score > 50:  # High history score
            reduction = max(0, reduction - 1)
        
        # Maximum reduction cap
        reduction = min(reduction, search_depth - 1)
        
        return reduction
