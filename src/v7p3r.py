#!/usr/bin/env python3
"""
V7P3R Chess Engine v19.5.1 - Timeout Fix (Endgame Responsiveness)

CRITICAL FIX (v19.5.1):
Fixed timeout responsiveness for endgames - Tournament showed 782 NPS in endgames
causing 26.3s moves when only 16s available. Timeout checks every 1000 nodes = 1.3s
between checks. Now checks every 100 nodes = 0.13s responsiveness + quiescence entry check.

PREVIOUS FIX (v19.5):
Fixed time management bottleneck - v19.4.1 was stopping at 4.3s when given 10s!
Legacy game-time-remaining heuristics were cutting search short. Now uses full
allocated time, targeting 90% with hard limit at 100%.

PHASE 1: ULTRA-FAST CHARACTER TRAITS
Building on Phase 0 (26K NPS baseline), adding v7p3r's playing style:

MaterialOpponent Core (Phase 0):
- Material counting (P=100, N=300, B=325, R=500, Q=900)
- Bishop pair bonus (+50cp)

Phase 1 Additions (Ultra-Fast Character Traits):
1. Castling bonus (+30cp) - King safety is CORE v7p3r defensive trait
2. Pawn advancement (+10cp/rank) - Aggressive pawn play is SIGNATURE v7p3r
3. Passed pawn detection (+20cp/rank) - Endgame conversion is v7p3r STRENGTH

PERFORMANCE (v19.5 with time fix):
- NPS: 24,000-32,000 (comparable to Phase 0)
- Depth in 10s: 7-9 expected (was 5 due to time management bug)
- Character: KING SAFETY + AGGRESSIVE PAWNS + ENDGAME CONVERSION

GOAL: Depth 10 in blitz time controls (5-10 seconds)

PHILOSOPHY: MaterialOpponent proved depth beats complexity.
Phase 1 adds minimal v7p3r character while maintaining that depth advantage.

PHASE 1 CHANGES - SPEED & CLEANUP:
- Removed modular evaluation system (4 files deleted)
- Removed context calculator (per-move overhead eliminated)
- Removed profile selector (6 profiles × decision tree overhead eliminated)
- Removed module registry (32+ modules × cost/criticality metadata eliminated)
- Simplified threefold threshold (constant 50cp vs dynamic 0-50cp calculation)
- NEW v19.1: Simplified move ordering (removed safety/tactical checks from hot path)
- NEW v19.5: Fixed time management (use full allocated time vs 25-30% cutoff)

EXPECTED IMPACT:
- Speed: 50-60% per-node time reduction vs v19.0 (move ordering was the bottleneck)
- NPS: 100,000+ (up from 6,800) - 15x improvement
- Depth: 8-10 in 5s (up from 3-6) - tactical awareness restored
- Timeout rate: 0% (down from 30%) - time stability achieved
- Code: Cleaner, faster, easier to profile further

ARCHITECTURE EVOLUTION:
- v19.1: Emergency performance fix - simplified move ordering (THIS VERSION)
- v19.0: Spring cleaning Phase 1 - modular eval removal (bottleneck discovered)
- v18.4: Mate-in-1 fast path + aspiration windows
- v18.3: PST optimization (+56 ELO vs v17.1)
- v18.0: Tactical safety system (safety checks now removed from move ordering)

VERSION LINEAGE:
- v19.1.0: Emergency performance fix (simplified move ordering, 15x speedup target)
- v19.0.0: Spring cleaning Phase 1 (modular eval removal, speed focus)
- v18.4.0: Mate-in-1 fast path + aspiration windows + memory stability
- v18.3.0: PST optimization (direct indexing, pre-computed tables)

Author: Pat Snyder
"""

import time
import chess
import sys
import random
import json
import os
from typing import Optional, Tuple, List, Dict
from collections import defaultdict, OrderedDict
from v7p3r_bitboard_evaluator import V7P3RScoringCalculationBitboard
from v7p3r_fast_evaluator import V7P3RFastEvaluator
from v7p3r_openings_v161 import get_enhanced_opening_book
from v7p3r_move_safety import MoveSafetyChecker


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
        self.max_history_entries = 10000  # V18.4 PHASE 1: Prevent unbounded growth
    
    def update_history(self, move: chess.Move, depth: int):
        """Update history score for a move"""
        move_key = f"{move.from_square}-{move.to_square}"
        self.history[move_key] += depth * depth
        
        # V18.4 PHASE 1: Clear old entries when limit exceeded
        if len(self.history) > self.max_history_entries:
            # Keep top 75% of entries by score (most successful moves)
            sorted_items = sorted(self.history.items(), key=lambda x: x[1], reverse=True)
            self.history = defaultdict(int, dict(sorted_items[:int(self.max_history_entries * 0.75)]))
    
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
    """V7P3R Chess Engine v18.2.0 - Combined Tactical + Positional"""
    
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
        
        # V18.0: Move safety checker for defensive tactics
        self.move_safety = MoveSafetyChecker(self.piece_values)
        
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
        
        # V19.0: Modular evaluation system REMOVED for simplicity and speed
        # Overhead removed: context calculation, profile selection, module registry
        # Expected speedup: 30-50% per-node time reduction
        
        # Keep reference to bitboard evaluator for compatibility
        self.bitboard_evaluator = self.evaluator if not use_fast_evaluator else V7P3RScoringCalculationBitboard(self.piece_values, enable_nudges=False)
        
        # V18.4 PHASE 1: Bounded evaluation cache (LRU, 20k entries)
        self.evaluation_cache = OrderedDict()  # position_hash -> evaluation (LRU cache)
        self.max_eval_cache_entries = 20000  # Prevent unbounded growth
        
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
            self.search_start_time = time.time()
            self.timeout_exceeded = False  # V19.5: Flag to propagate timeout through recursion
            
            # V19.0: Removed modular evaluation context calculation overhead
            # Previous: context_calculator + profile_selector calls before every search
            # Impact: 30-50% speedup from eliminating pre-search overhead
            
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
                            print(f"info string Opening book move: {book_move.uci()}", flush=True)
                            return book_move

            # V17.1: PV INSTANT MOVES DISABLED
            # REASON: Caused all 3 tournament losses - trusts stale PV without re-evaluation
            # IMPACT: f6 blunder repeated 3 times (depth 1, 0 nodes signature)
            # See: docs/V17_0_PV_Blunder_Investigation_Findings.md
            # pv_move = self.pv_tracker.check_position_for_instant_move(board)
            # if pv_move:
            #     return pv_move
            
            # V18.4: MATE-IN-1 FAST PATH
            # Check for immediate checkmate before starting expensive search
            # Expected impact: +10-20 ELO, 100% tactical accuracy, <1ms overhead
            # RATIONALE: 30% detection baseline (6/20 positions), avg 1.28s search time
            # With fast path: 100% detection, <10ms instant return, minimal overhead
            for move in legal_moves:
                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    print(f"info string Mate-in-1 found: {move.uci()}", flush=True)
                    return move
                board.pop()
            
            # V19.5: SIMPLIFIED TIME MANAGEMENT
            # Use the full allocated time - no premature cutoffs
            # This fixes the issue where v19.4.1 stopped at 4s when given 10s
            target_time = time_limit * 0.90  # Target 90% of allocated time
            max_time = time_limit  # Hard limit at allocated time
            
            # Iterative deepening
            best_move = legal_moves[0]
            best_score = -99999
            stable_best_count = 0  # V14.1: Track how many iterations returned same best move
            
            for current_depth in range(1, self.default_depth + 1):
                # V19.5: Check elapsed time BEFORE starting iteration
                elapsed = time.time() - self.search_start_time
                
                # V19.5: Early exit if we've used target time
                if elapsed >= target_time:
                    break
                
                # V19.5: REMOVED predictive completion check
                # Partial search data is still valuable - let engine search as deep as possible
                # Even if depth N doesn't complete, depths 1 to N-1 provide valid move ordering
                
                try:
                    # Store previous best in case iteration fails
                    previous_best = best_move
                    previous_score = best_score
                    
                    # V18.4: ASPIRATION WINDOWS
                    # Use narrow search window for depth 3+ to reduce nodes
                    # Expected impact: +20-40 ELO, 15-25% node reduction
                    # RATIONALE: 14,848 avg nodes baseline → target 11,878 nodes (20% reduction)
                    if current_depth >= 3 and best_score > -9000 and best_score < 9000:
                        # Start with narrow ±50cp window
                        window = 50
                        alpha = best_score - window
                        beta = best_score + window
                        
                        # Try narrow window first
                        score, move = self._recursive_search(board, current_depth, alpha, beta, time_limit)
                        
                        # Fail-low: Score dropped below window
                        if score <= alpha:
                            # Widen downward, re-search
                            window = 150
                            alpha = best_score - window
                            score, move = self._recursive_search(board, current_depth, alpha, beta, time_limit)
                            
                            # Still fail-low: Use full window downward
                            if score <= alpha:
                                alpha = -99999
                                score, move = self._recursive_search(board, current_depth, alpha, beta, time_limit)
                        
                        # Fail-high: Score exceeded window
                        elif score >= beta:
                            # Widen upward, re-search
                            window = 150
                            beta = best_score + window
                            score, move = self._recursive_search(board, current_depth, alpha, beta, time_limit)
                            
                            # Still fail-high: Use full window upward
                            if score >= beta:
                                beta = 99999
                                score, move = self._recursive_search(board, current_depth, alpha, beta, time_limit)
                    else:
                        # Depth 1-2 or extreme scores: Use full window
                        score, move = self._recursive_search(board, current_depth, -99999, 99999, time_limit)
                    
                    # V19.5: No longer tracking iteration time - removed predictive completion logic
                    
                    # Update best move if we got a valid result
                    if move and move != chess.Move.null():
                        # V14.1: Track stability of best move
                        if move == best_move:
                            stable_best_count += 1
                        else:
                            stable_best_count = 1
                        
                        best_move = move
                        best_score = score
                        
                        elapsed_ms = int((time.time() - self.search_start_time) * 1000)
                        nps = int(self.nodes_searched / max(elapsed_ms / 1000, 0.001))
                        
                        # Extract and display PV
                        pv_line = self._extract_pv(board, current_depth)
                        pv_string = " ".join([str(m) for m in pv_line])
                        
                        # Store PV for following optimization
                        if current_depth >= 4 and len(pv_line) >= 3:
                            self.pv_tracker.store_pv_from_search(board, pv_line)
                        
                        print(f"info depth {current_depth} score cp {int(score)} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {pv_string}")
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
            
            # V18.2: Dynamic threefold check based on material balance
            # Only avoid if move causes threefold AND we're in advantage
            if best_move and current_depth >= 3:
                if self._would_cause_threefold(board, best_move):
                    current_eval = self._evaluate_position(board)
                    
                    # V19.0: Simplified threefold threshold (was dynamic, now constant)
                    # Previous: get_threefold_threshold(context) returned 0-50cp based on material
                    # Now: Fixed 50cp threshold (was v17.8 value, proven stable)
                    threefold_threshold = 50
                    
                    # Only avoid threefold if eval > threshold (50cp = small pawn advantage)
                    if current_eval > threefold_threshold:
                        # Penalize the threefold move in transposition table
                        self._store_transposition_table(board, current_depth, -500, best_move, -99999, 99999)
                        # Re-run search at last completed depth to find alternative
                        score, alt_move = self._recursive_search(board, current_depth, -99999, 99999, time_limit)
                        if alt_move and alt_move != best_move:
                            # Found a different move - use it
                            best_move = alt_move
                            print(f"info string Avoided threefold repetition (eval {current_eval}cp > threshold {threefold_threshold}cp)", flush=True)
                        # If no alternative found (all moves cause threefold), keep original
                    
            return best_move
        
        # This should never be called directly with is_root=False from external code
        else:
            # Fallback - call the recursive search method
            score, move = self._recursive_search(board, depth or 1, alpha, beta, time_limit)
            return move if move else chess.Move.null()
    
    def _would_cause_threefold(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move would cause threefold repetition (O(1) hash lookup)"""
        board.push(move)
        is_threefold = board.is_repetition(2)  # Position occurred 2+ times before
        board.pop()
        return is_threefold
    
    def _recursive_search(self, board: chess.Board, search_depth: int, alpha: float, beta: float, time_limit: float) -> Tuple[float, Optional[chess.Move]]:
        """
        Recursive alpha-beta search with all advanced features
        Returns (score, best_move) tuple
        """
        self.nodes_searched += 1
        
        # CRITICAL: Time checking during recursive search to prevent timeouts
        # V19.5.1: Reduced from 1000 → 100 nodes for endgame responsiveness (782 NPS = 0.13s checks)
        if hasattr(self, 'search_start_time') and self.nodes_searched % 100 == 0:
            elapsed = time.time() - self.search_start_time
            if elapsed > time_limit:
                # Set timeout flag to propagate through all recursion levels
                self.timeout_exceeded = True
                return self._evaluate_position(board), None
        
        # V19.5: Check timeout flag to abort search immediately
        if hasattr(self, 'timeout_exceeded') and self.timeout_exceeded:
            return self._evaluate_position(board), None
    
    # Continue _recursive_search method (this got mis-indented in previous edit)
    # Note: The following code belongs to _recursive_search, NOT _would_cause_threefold
        
        # 1. TRANSPOSITION TABLE PROBE
        tt_hit, tt_score, tt_move = self._probe_transposition_table(board, search_depth, int(alpha), int(beta))
        if tt_hit:
            return float(tt_score), tt_move
        
        # 2. TERMINAL CONDITIONS
        if search_depth == 0:
            # Enter quiescence search for tactical stability
            # v19.3: Reduced from depth 4 → 1 (78% of search time was here!)
            score = self._quiescence_search(board, alpha, beta, 1)
            return score, None
        
        # 4. MOVE GENERATION (do this BEFORE expensive game-over check)
        legal_moves = list(board.legal_moves)
        
        # Game over detection: Only if no legal moves (much faster than is_game_over())
        if not legal_moves:
            # No legal moves - either checkmate or stalemate
            if board.is_check():
                # Checkmate - prefer quicker mates
                score = -29000.0 + (self.default_depth - search_depth)
            else:
                # Stalemate
                score = 0.0
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
        
        # 4. MOVE ORDERING (legal_moves already generated above)
        ordered_moves = self._order_moves_advanced(board, legal_moves, search_depth, tt_move)
        
        # 5. MAIN SEARCH LOOP (NEGAMAX WITH ALPHA-BETA + PVS)
        best_score = -99999.0
        best_move = None
        original_alpha = alpha
        moves_searched = 0
        
        for move in ordered_moves:
            board.push(move)
            
            # V19.5: PRINCIPAL VARIATION SEARCH (PVS)
            # This dramatically reduces nodes searched by using null windows
            
            if moves_searched == 0:
                # First move: Full window search
                reduction = self._calculate_lmr_reduction(move, moves_searched, search_depth, board)
                if reduction > 0:
                    score, _ = self._recursive_search(board, search_depth - 1 - reduction, -beta, -alpha, time_limit)
                    score = -score
                    if score > alpha:  # Re-search at full depth
                        score, _ = self._recursive_search(board, search_depth - 1, -beta, -alpha, time_limit)
                        score = -score
                else:
                    score, _ = self._recursive_search(board, search_depth - 1, -beta, -alpha, time_limit)
                    score = -score
            else:
                # Subsequent moves: Try null window search first (PVS)
                reduction = self._calculate_lmr_reduction(move, moves_searched, search_depth, board)
                if reduction > 0:
                    # LMR: reduced depth + null window
                    score, _ = self._recursive_search(board, search_depth - 1 - reduction, -alpha - 1, -alpha, time_limit)
                    score = -score
                else:
                    # No LMR: null window at full depth
                    score, _ = self._recursive_search(board, search_depth - 1, -alpha - 1, -alpha, time_limit)
                    score = -score
                
                # If null window failed high, re-search with full window
                if alpha < score < beta:
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
        """V19.1: Simplified high-performance move ordering
        
        REMOVED in v19.1 for 15x performance gain:
        - Move safety checks (0.083ms/move = 2.9ms/position overhead)
        - Bitboard tactical detection in ordering (expensive for all moves)
        - Complex multi-category sorting
        
        KEPT for proven effectiveness:
        - TT move priority (hash table lookup)
        - MVV-LVA for captures (simple arithmetic)
        - Check detection (built-in to chess library)
        - Killer moves (hash table lookup)
        - History heuristic (array lookup)
        
        Performance: Target 100,000+ NPS (was 6,800 NPS in v19.0)
        """
        if len(moves) <= 2:
            return moves
        
        # Fast path: TT move first if available
        if tt_move and tt_move in moves:
            remaining = [m for m in moves if m != tt_move]
            return [tt_move] + self._score_and_sort_moves(board, remaining, depth)
        
        return self._score_and_sort_moves(board, moves, depth)
    
    def _score_and_sort_moves(self, board: chess.Board, moves: List[chess.Move], depth: int) -> List[chess.Move]:
        """V19.1: Fast single-pass move scoring"""
        scored_moves = []
        killer_set = set(self.killer_moves.get_killers(depth))
        
        for move in moves:
            score = 0.0
            
            # 1. Captures: MVV-LVA only (no expensive tactical detection)
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                victim_value = self.piece_values.get(victim.piece_type, 0) if victim else 0
                attacker = board.piece_at(move.from_square)
                attacker_value = self.piece_values.get(attacker.piece_type, 0) if attacker else 0
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                score = (victim_value * 100 - attacker_value) + 10000  # Captures first
            
            # 2. Checks: Simple bonus (no tactical detection)
            elif board.gives_check(move):
                score = 5000
            
            # 3. Killer moves: Hash table lookup only
            elif move in killer_set:
                score = 3000
                self.search_stats['killer_hits'] += 1
            
            # 4. Quiet moves: History heuristic only (no tactical detection)
            else:
                score = self.history_heuristic.get_history_score(move)
            
            scored_moves.append((score, move))
        
        # Single sort by score descending
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored_moves]
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """
        V19.4.0 PHASE 1: Character-Defining Evaluation
        
        MaterialOpponent core + ultra-fast v7p3r character traits:
        1. Material counting + bishop pair (from MaterialOpponent)
        2. Castling bonus (king safety - core v7p3r defensive trait)
        3. Pawn advancement (aggressive pawn play - signature v7p3r)
        4. Passed pawn detection (endgame conversion - v7p3r strength)
        
        Expected: 23-25K NPS (slight slowdown for major character gain)
        Target: Depth 8-10 in 5 seconds
        """
        # Use chess library's fast _transposition_key() for caching
        cache_key = board._transposition_key()
        
        if cache_key in self.evaluation_cache:
            self.search_stats['cache_hits'] += 1
            # V18.4 PHASE 1: Move to end for LRU (most recently used)
            self.evaluation_cache.move_to_end(cache_key)
            return self.evaluation_cache[cache_key]
        
        self.search_stats['cache_misses'] += 1
        
        # === MATERIAL COUNTING (MaterialOpponent Core) ===
        score = 0.0
        
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 325,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        # Count material for both sides
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                           chess.ROOK, chess.QUEEN]:
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            piece_value = piece_values[piece_type]
            score += (white_count - black_count) * piece_value
        
        # Bishop pair bonus
        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
            score += 50
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
            score -= 50
        
        # === PHASE 1: CHARACTER-DEFINING EVALUATIONS ===
        
        # 1. CASTLING BONUS (King Safety - Core v7p3r Trait)
        # Cost: ~0.001ms (simple square checks)
        # Detect castling by checking if king is on castled square
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        # White castled if king on g1 (kingside) or c1 (queenside)
        if white_king_square in [chess.G1, chess.C1]:
            score += 30
        # Black castled if king on g8 (kingside) or c8 (queenside)
        if black_king_square in [chess.G8, chess.C8]:
            score -= 30
        
        # 2. PAWN ADVANCEMENT (Aggressive Character - Signature v7p3r)
        # Cost: ~0.002ms (iterate ~8 pawns, simple rank check)
        for square in board.pieces(chess.PAWN, chess.WHITE):
            rank = chess.square_rank(square)
            if rank >= 4:  # 5th rank or higher (0-indexed: rank 4 = 5th rank)
                score += (rank - 3) * 10  # +10/20/30/40/50 for ranks 5/6/7/8
        
        for square in board.pieces(chess.PAWN, chess.BLACK):
            rank = chess.square_rank(square)
            if rank <= 3:  # 4th rank or lower from Black's perspective
                score -= (4 - rank) * 10
        
        # 3. PASSED PAWN DETECTION (Endgame Strength - v7p3r Conversion Ability)
        # Cost: ~0.005-0.010ms (worst case: check 8 pawns × 3 files × up to 7 ranks)
        # Only check advanced pawns (rank 5+) to reduce overhead
        for square in board.pieces(chess.PAWN, chess.WHITE):
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            if rank >= 4:  # Only check pawns on 5th rank or higher
                is_passed = True
                # Check if any black pawns block this pawn's path
                for ahead_rank in range(rank + 1, 8):
                    # Check same file and adjacent files
                    for check_file in range(max(0, file - 1), min(8, file + 2)):
                        check_square = chess.square(check_file, ahead_rank)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                            is_passed = False
                            break
                    if not is_passed:
                        break
                
                if is_passed:
                    # Exponential bonus: rank 5=20cp, rank 6=40cp, rank 7=60cp, rank 8=80cp
                    score += (rank - 3) * 20
        
        # Black passed pawns (reverse logic)
        for square in board.pieces(chess.PAWN, chess.BLACK):
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            if rank <= 3:  # Only check pawns on 4th rank or lower (from black's view)
                is_passed = True
                # Check if any white pawns block this pawn's path
                for ahead_rank in range(rank - 1, -1, -1):  # Go down from black's perspective
                    for check_file in range(max(0, file - 1), min(8, file + 2)):
                        check_square = chess.square(check_file, ahead_rank)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                            is_passed = False
                            break
                    if not is_passed:
                        break
                
                if is_passed:
                    score -= (4 - rank) * 20
        
        # Return from current player's perspective
        final_score = score if board.turn == chess.WHITE else -score
        
        # V18.4 PHASE 1: Cache with LRU eviction (20k entry limit)
        if len(self.evaluation_cache) >= self.max_eval_cache_entries:
            # Remove oldest entry (FIFO/LRU)
            self.evaluation_cache.popitem(last=False)
        
        self.evaluation_cache[cache_key] = final_score
        return final_score
    
    def _simple_king_safety(self, board: chess.Board, color: bool) -> float:
        """V12.2: Simplified king safety evaluation for performance"""
        score = 0.0
        
        # Basic castling bonus
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
        """Store result in transposition table - PHASE 1 FEATURE"""
        zobrist_hash = self.zobrist.hash_position(board)
        
        # Determine node type
        if score <= alpha:
            node_type = 'upperbound'
        elif score >= beta:
            node_type = 'lowerbound'
        else:
            node_type = 'exact'
        
        # V18.4 PHASE 1: Depth-preferred replacement strategy
        if zobrist_hash in self.transposition_table:
            # Always replace existing entry for same position
            existing_depth = self.transposition_table[zobrist_hash].depth
            # Only replace if new search is deeper (preserve deep entries)
            if depth >= existing_depth:
                entry = TranspositionEntry(depth, score, best_move, node_type, zobrist_hash)
                self.transposition_table[zobrist_hash] = entry
                self.search_stats['tt_stores'] += 1
        elif len(self.transposition_table) < self.max_tt_entries:
            # Table not full, just add
            entry = TranspositionEntry(depth, score, best_move, node_type, zobrist_hash)
            self.transposition_table[zobrist_hash] = entry
            self.search_stats['tt_stores'] += 1
        else:
            # Table full, need to evict (depth-preferred replacement)
            # Find a shallow entry to replace (simple scan, replace first depth < new_depth)
            min_depth_hash = None
            min_depth = depth  # Only replace if victim is shallower
            for hash_key, entry in list(self.transposition_table.items())[:100]:  # Check first 100 for speed
                if entry.depth < min_depth:
                    min_depth = entry.depth
                    min_depth_hash = hash_key
            
            if min_depth_hash:
                # Replace shallow entry with deeper one
                del self.transposition_table[min_depth_hash]
                entry = TranspositionEntry(depth, score, best_move, node_type, zobrist_hash)
                self.transposition_table[zobrist_hash] = entry
                self.search_stats['tt_stores'] += 1
            # else: TT full with all deep entries, don't store shallow entry
    
    def _has_non_pawn_pieces(self, board: chess.Board) -> bool:
        """Check if the current side has non-pawn pieces (for null move pruning)"""
        current_color = board.turn
        for square in range(64):
            piece = board.piece_at(square)
            if piece and piece.color == current_color and piece.piece_type != chess.PAWN:
                return True
        return False
    
    def _extract_pv(self, board: chess.Board, max_depth: int) -> List[chess.Move]:
        """Extract principal variation from transposition table"""
        pv = []
        temp_board = board.copy()
        
        for depth in range(max_depth, 0, -1):
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
    
    def _quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int) -> float:
        """
        V19.3: Optimized quiescence search (reduced from depth 4 to 1)
        V19.5.1: Added timeout check at entry point
        
        Changes from v19.2:
        - Depth reduced 4 → 1 (was 78% of search time!)
        - Removed check generation (captures only)
        - Added delta pruning (skip hopeless captures)
        - Simplified move ordering (no sorting for shallow search)
        """
        self.nodes_searched += 1
        
        # V19.5.1: Check timeout flag to abort quiescence search immediately
        if hasattr(self, 'timeout_exceeded') and self.timeout_exceeded:
            return self._evaluate_position(board)
        
        # Stand pat evaluation
        stand_pat = self._evaluate_position(board)
        
        # Beta cutoff on stand pat
        if stand_pat >= beta:
            return beta
        
        # Delta pruning: If we're down by more than a queen, skip quiescence
        # (no single capture can save us)
        BIG_DELTA = 900  # Queen value
        if stand_pat < alpha - BIG_DELTA:
            return alpha
        
        # Update alpha if stand pat is better
        if stand_pat > alpha:
            alpha = stand_pat
        
        # Depth limit reached
        if depth <= 0:
            return stand_pat
        
        # Generate captures only (NOT checks - they add instability)
        # v19.3: Optimized - filter during generation instead of separate loop
        captures = []
        for move in board.legal_moves:
            if board.is_capture(move):
                # Delta pruning: Skip captures that can't improve position
                victim = board.piece_at(move.to_square)
                if victim:
                    victim_value = self.piece_values.get(victim.piece_type, 0)
                    # Only consider capture if it might improve alpha
                    if stand_pat + victim_value + 200 >= alpha:  # 200cp margin
                        captures.append(move)
        
        # If no good captures, return stand pat
        if not captures:
            return stand_pat
        
        # Simple MVV-LVA ordering (no full sort for depth 1)
        # Just put queen captures first, then rooks, etc.
        def capture_priority(move):
            victim = board.piece_at(move.to_square)
            if victim:
                return self.piece_values.get(victim.piece_type, 0)
            return 0
        
        captures.sort(key=capture_priority, reverse=True)
        
        # Search captures
        best_score = stand_pat
        for move in captures:
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha, depth - 1)
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
        self.evaluation_cache.clear()
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
