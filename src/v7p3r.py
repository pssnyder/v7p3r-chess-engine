#!/usr/bin/env python3
"""
V7P3R Chess Engine v14.5 - CRITICAL FIXES for UCI and Move Ordering

URGENT REGRESSION FIXES discovered in v14.4 testing showing massive blundering:
- V14.4 was losing to v10.8 due to constant piece blunders
- UCI output was completely broken (no depth/eval display)
- Safety prioritization was destroying tactical move ordering
- Over-aggressive emergency time limits causing depth-1 searches

V14.5 CRITICAL FIXES:
- FIXED: UCI output completely broken - replaced no-op logger with proper print statements
- FIXED: Safety prioritization reordering all moves - disabled destructive _apply_safety_prioritization
- FIXED: Emergency time limits too aggressive - relaxed from 70% to 85% threshold
- RESTORED: Tactical move ordering that was working in v14.0

ARCHITECTURE:
- Phase 1: Core search (alpha-beta, TT, iterative deepening)
- Phase 2: Reliable time management with proper depth achievement
- Phase 3: Tactical move ordering without safety interference

VERSION LINEAGE:
- v14.5: URGENT regression fixes for UCI output, move ordering, time management
- v14.4: REGRESSION - broken UCI, broken move ordering, excessive blundering (REVERTED)
- v14.3: Emergency fixes for time management and depth consistency
- v14.2: Performance optimizations
- v14.0: Consolidated performance build
- v12.6: Stable tournament baseline (71.6% score)

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

# Simple UCI Logger mock for clean architecture
class UCILogger:
    def __init__(self, debug_enabled=False):
        self.debug_enabled = debug_enabled
    def start_search(self): pass
    def debug_search_start(self, *args): pass
    def debug_time_allocation(self, *args): pass  
    def debug_game_phase(self, *args): pass
    def debug_emergency_stop(self, *args): pass
    def info_string(self, *args, **kwargs): pass
    def report_search_progress(self, *args): pass
    def debug_iteration_complete(self, *args): pass
    def final_search_summary(self, *args): pass


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
    """V7P3R Chess Engine v14.2 - Performance Optimizations & Game Phase Detection"""
    
    def __init__(self):
        # Basic configuration
        # Base piece values (V14.2: Cached dynamic bishop valuation)
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300, 
            chess.BISHOP: 300,  # Base value - cached dynamic calculation
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0  # King safety handled separately
        }
        
        # V14.2: Performance optimizations and caching
        self.bishop_value_cache = {}  # Cache for dynamic bishop values
        self.game_phase_cache = {}    # Cache for game phase detection
        self.search_depth_achieved = {}  # Track actual search depth per move
        
        # Search configuration
        self.default_depth = 6
        self.nodes_searched = 0
        
        # Evaluation components - V14.0: Consolidated bitboard evaluation
        self.bitboard_evaluator = V7P3RScoringCalculationBitboard(self.piece_values, enable_nudges=False)
        
        # Simple evaluation cache for speed
        self.evaluation_cache = {}  # position_hash -> evaluation
        
        # Advanced search infrastructure
        self.transposition_table: Dict[int, TranspositionEntry] = {}
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        self.zobrist = ZobristHashing()
        
        # V14.3: UCI Logging System for proper protocol compliance
        self.uci_logger = UCILogger(debug_enabled=False)  # Can be enabled via UCI option
        
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
            'average_depth_achieved': 0.0,
            'game_phase_switches': 0,
        }
        
        # PV Following System - V10 OPTIMIZATION
        self.pv_tracker = PVTracker()
    
    def search(self, board: chess.Board, time_limit: float = 3.0, depth: Optional[int] = None, 
               alpha: float = -99999, beta: float = 99999, is_root: bool = True) -> chess.Move:
        """
        V14.3 EMERGENCY SEARCH - Critical fixes for time management and depth consistency:
        - Emergency time controls to prevent flagging (CRITICAL)
        - Guaranteed minimum depth achievement 
        - Simplified time allocation removing complex calculations
        - Conservative game phase detection
        - Hard time limits with emergency bailouts
        """
        
        # ROOT LEVEL: Emergency time management with guaranteed minimum depth
        if is_root:
            self.nodes_searched = 0
            self.search_start_time = time.time()
            self.emergency_stop_flag = False  # V14.3: Ultra-conservative emergency flag
            self.current_time_limit = time_limit  # V14.3: Store for quiescence access
            
            # V14.3: Initialize UCI logging for search
            self.uci_logger.start_search()
            
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return chess.Move.null()

            # PV FOLLOWING OPTIMIZATION - check if current position triggers instant move
            pv_move = self.pv_tracker.check_position_for_instant_move(board)
            if pv_move:
                return pv_move
            
            # V14.3 CRITICAL FIX: Use emergency time allocation
            target_time, max_time = self._calculate_emergency_time_allocation(time_limit)
            
            # V14.3: SINGLE game phase detection for entire search (major optimization)
            game_phase = self._detect_game_phase_conservative(board)
            self.current_game_phase = game_phase  # Store for use throughout search
            
            # V14.3: Guaranteed minimum depth calculation
            minimum_depth = self._calculate_minimum_depth(time_limit)
            target_depth = max(minimum_depth, self._calculate_target_depth(game_phase, time_limit))
            
            # V14.3: Debug logging for search start
            self.uci_logger.debug_search_start(time_limit, target_depth)
            self.uci_logger.debug_time_allocation(target_time, max_time, time_limit)
            moves_played = len(board.move_stack)
            self.uci_logger.debug_game_phase(game_phase, 0, moves_played)  # Material calculation removed for simplicity
            
            # V14.5 DEBUG: Print actual time allocations
            print(f"info string V14.5 time allocations: time_limit={time_limit:.2f}s, target={target_time:.2f}s, max={max_time:.2f}s, min_depth={minimum_depth}")
            
            # Iterative deepening with EMERGENCY TIME CONTROLS
            best_move = legal_moves[0]
            best_score = -99999
            depths_completed = []
            
            for current_depth in range(1, target_depth + 1):
                iteration_start = time.time()
                
                # V14.3 CRITICAL: Emergency time checking with ultra-conservative hard limits
                elapsed = time.time() - self.search_start_time
                
                # V14.5 FIX: Relaxed emergency bailout - was too aggressive at 70%
                # Changed from 0.7 to 0.85 to allow proper depth achievement
                if elapsed > time_limit * 0.85:
                    self.uci_logger.debug_emergency_stop("Time limit", elapsed, time_limit)
                    self.emergency_stop_flag = True
                    break
                
                # V14.3: FORCE completion of minimum depth before time checks
                if current_depth <= minimum_depth:
                    # Don't break on time for minimum depth - this is critical for consistency
                    pass
                elif elapsed > target_time:
                    print(f"info string Target time ({target_time:.2f}s) reached at {elapsed:.2f}s, stopping before depth {current_depth}")
                    break
                
                # V14.5 FIX: REMOVED overly conservative iteration prediction
                # The prediction was preventing depth 4+ even with plenty of time remaining
                # We have other time checks (85% threshold, max_time checks) that are sufficient
                # if current_depth > minimum_depth and len(depths_completed) > 0:
                #     avg_recent_time = sum(depths_completed[-2:]) / len(depths_completed[-2:])
                #     predicted_time = elapsed + (avg_recent_time * 1.5)
                #     if predicted_time > max_time:
                #         print(f"info string Predicted time overrun, stopping at depth {current_depth}")
                #         break
                
                try:
                    # Store previous best in case iteration fails
                    previous_best = best_move
                    previous_score = best_score
                    
                    # V14.5 FIX: Relaxed emergency time check from 70% to 85%
                    elapsed = time.time() - self.search_start_time
                    if elapsed > max_time * 0.85 or self.emergency_stop_flag:
                        print(f"info string EMERGENCY STOP before depth {current_depth} at {elapsed:.2f}s")
                        self.emergency_stop_flag = True
                        break
                    
                    # Call recursive search for this depth with EMERGENCY TIME LIMIT
                    try:
                        score, move = self._recursive_search(board, current_depth, -99999, 99999, max_time)
                    except Exception as e:
                        print(f"info string Search exception at depth {current_depth}: {e}")
                        break
                    
                    # V14.5 FIX: Relaxed post-search time check from 70% to 85%
                    post_search_time = time.time() - self.search_start_time
                    if post_search_time > time_limit * 0.85:
                        print(f"info string EMERGENCY: Post-search time {post_search_time:.3f}s exceeds limit")
                        self.emergency_stop_flag = True
                        if move and move != chess.Move.null():
                            best_move = move
                            best_score = score
                        break
                    
                    # Track iteration completion time
                    iteration_time = time.time() - iteration_start
                    depths_completed.append(iteration_time)
                    
                    # V14.5 FIX: Relaxed emergency time check from 70% to 85%
                    total_elapsed = time.time() - self.search_start_time
                    if total_elapsed > time_limit * 0.85 or self.emergency_stop_flag:
                        print(f"info string Emergency time limit reached after depth {current_depth}")
                        self.emergency_stop_flag = True
                        # Use the result if we got one, otherwise keep previous
                        if move and move != chess.Move.null():
                            best_move = move
                            best_score = score
                        break
                    
                    # Update best move if we got a valid result
                    if move and move != chess.Move.null():
                        best_move = move
                        best_score = score
                        
                        elapsed_ms = int(total_elapsed * 1000)
                        nps = int(self.nodes_searched / max(total_elapsed, 0.001))
                        
                        # Extract and display PV
                        pv_line = self._extract_pv(board, current_depth)
                        pv_string = " ".join([str(m) for m in pv_line])
                        
                        # Store PV for following optimization
                        if current_depth >= 4 and len(pv_line) >= 3:
                            self.pv_tracker.store_pv_from_search(board, pv_line)
                        
                        # V14.3: Track search depth achieved
                        self.search_depth_achieved[best_move] = current_depth
                        
                        # V14.5 FIX: Restore proper UCI output that was broken in v14.4
                        print(f"info depth {current_depth} score cp {int(score)} nodes {self.nodes_searched} time {elapsed_ms} nps {nps} pv {pv_string}")
                        sys.stdout.flush()
                        
                        # V14.3: Debug info for iteration completion
                        self.uci_logger.debug_iteration_complete(current_depth, iteration_time, self.nodes_searched)
                    else:
                        # Restore previous best if iteration failed
                        best_move = previous_best
                        best_score = previous_score
                        
                except Exception as e:
                    self.uci_logger.info_string(f"Search interrupted at depth {current_depth}: {e}", debug_only=True)
                    # Keep previous best result
                    break
            
            # V14.3: Final search summary using UCI logger
            final_elapsed = time.time() - self.search_start_time
            final_depth = len(depths_completed)
            self.search_stats['average_depth_achieved'] = final_depth
            
            self.uci_logger.final_search_summary(str(best_move), final_depth, final_elapsed, 
                                               self.nodes_searched, time_limit)
            
            return best_move
        
        # Fallback for non-root calls
        else:
            score, move = self._recursive_search(board, depth or 1, alpha, beta, time_limit)
            return move if move else chess.Move.null()
    
    def _recursive_search(self, board: chess.Board, search_depth: int, alpha: float, beta: float, time_limit: float) -> Tuple[float, Optional[chess.Move]]:
        """
        Recursive alpha-beta search with all advanced features
        Returns (score, best_move) tuple
        """
        self.nodes_searched += 1
        
        # V14.3 ULTRA-AGGRESSIVE: Time checking during recursive search to prevent timeouts
        if hasattr(self, 'search_start_time') and self.nodes_searched % 50 == 0:  # Check 20x more frequently
            elapsed = time.time() - self.search_start_time
            # Check emergency stop flag first
            if hasattr(self, 'emergency_stop_flag') and self.emergency_stop_flag:
                return self._evaluate_position(board), None
            # Use ultra-conservative 60% limit during recursive search (even more aggressive)
            if elapsed > time_limit * 0.6:
                # Set emergency flag and return immediately
                if hasattr(self, 'emergency_stop_flag'):
                    self.emergency_stop_flag = True
                return self._evaluate_position(board), None
        
        # 1. TRANSPOSITION TABLE PROBE
        tt_hit, tt_score, tt_move = self._probe_transposition_table(board, search_depth, int(alpha), int(beta))
        if tt_hit:
            return float(tt_score), tt_move
        
        # 2. TERMINAL CONDITIONS
        if search_depth == 0:
            # Enter quiescence search for tactical stability
            score = self._quiescence_search(board, alpha, beta, 4)
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
            # V14.3: Emergency time check before each move in recursive search
            if hasattr(self, 'emergency_stop_flag') and self.emergency_stop_flag:
                break
            if hasattr(self, 'search_start_time') and moves_searched > 0 and moves_searched % 5 == 0:
                elapsed = time.time() - self.search_start_time
                if elapsed > time_limit * 0.6:
                    if hasattr(self, 'emergency_stop_flag'):
                        self.emergency_stop_flag = True
                    break
            
            board.push(move)
            
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
        """
        V14.4 ENHANCED TACTICAL move ordering - Phase 1 improvements for 1500+ puzzle performance
        
        Improvements based on diagnostic analysis:
        - Enhanced check prioritization (+200 bonus)
        - Valuable piece capture prioritization (+150 bonus) 
        - Multi-piece attack detection (+120 bonus)
        - Threat creation prioritization (+100 bonus)
        
        Target: +3-5% accuracy improvement on tactical puzzles
        """
        if len(moves) <= 2:
            return moves
        
        # Enhanced categories for tactical play
        tt_moves = []
        mate_threats = []      # NEW: Mate threat moves
        checks = []
        high_value_captures = [] # NEW: Separate high-value captures
        captures = []
        multi_attacks = []     # NEW: Moves attacking multiple pieces
        threats = []           # NEW: Moves creating threats
        killers = []
        development = []
        pawn_advances = []
        tactical_moves = []
        quiet_moves = []
        
        # Performance optimization: Pre-create sets for fast lookups
        killer_set = set(self.killer_moves.get_killers(depth))
        
        # V14.4: Pre-calculate tactical analysis using bitboard evaluator
        tactical_analysis = self.bitboard_evaluator.analyze_position_for_tactics_bitboard(board)
        
        # V14.4 Phase 2: Cache pin detection using bitboard evaluator
        cached_original_pins = self.bitboard_evaluator.detect_pins_bitboard(board)
        
        for move in moves:
            # 1. Transposition table move (highest priority)
            if tt_move and move == tt_move:
                tt_moves.append(move)
                continue
            
            # V14.4: Enhanced tactical move analysis
            tactical_score = self._calculate_tactical_move_score(board, move, tactical_analysis, cached_original_pins)
            
            # 2. Mate threats (NEW - highest tactical priority)
            if tactical_score.get('mate_threat', False):
                mate_threats.append((1000 + tactical_score['base_score'], move))
                continue
            
            # 3. Checks (ENHANCED - V14.4 prioritization)
            if board.gives_check(move):
                check_bonus = 200.0  # Increased from 60.0 based on diagnostic analysis
                discovered_check_bonus = 50.0 if tactical_score.get('discovered_check', False) else 0
                double_check_bonus = 100.0 if tactical_score.get('double_check', False) else 0
                
                total_check_score = check_bonus + discovered_check_bonus + double_check_bonus + tactical_score['base_score']
                checks.append((total_check_score, move))
                continue
            
            # 4. Multi-piece attacks (PRIORITY - before captures)
            if tactical_score.get('attacks_multiple', False):
                multi_attack_bonus = 350.0  # Much higher bonus for fork-like moves
                pin_bonus = 150.0 if tactical_score.get('creates_pin', False) else 0
                total_multi_score = multi_attack_bonus + pin_bonus + tactical_score['base_score']
                multi_attacks.append((total_multi_score, move))
                continue

            # 5. High-value captures (after multi-attacks to avoid double-categorization)
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                victim_value = self._get_dynamic_piece_value(board, victim.piece_type, not board.turn) if victim else 0
                attacker = board.piece_at(move.from_square)
                attacker_value = self._get_dynamic_piece_value(board, attacker.piece_type, board.turn) if attacker else 0
                
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                mvv_lva_score = victim_value * 100 - attacker_value
                
                # V14.4: Enhanced capture bonuses
                high_value_bonus = 300.0 if victim_value >= 500 else 150.0 if victim_value >= 300 else 0  # Higher bonuses
                safe_capture_bonus = 50.0 if tactical_score.get('safe_capture', False) else 0
                
                total_capture_score = mvv_lva_score + high_value_bonus + safe_capture_bonus + tactical_score['base_score']
                
                # Separate high-value captures for better ordering
                if victim_value >= 500:  # Queen (900) or Rook (500)
                    high_value_captures.append((total_capture_score, move))
                else:
                    captures.append((total_capture_score, move))
                continue
            
            # 6. Threat creation (NEW - moves that threaten valuable pieces)
            if tactical_score.get('creates_threat', False):
                threat_value = tactical_score.get('threat_value', 0)
                # Much higher bonuses, especially for queen threats
                if threat_value >= 900:  # Queen threat
                    threat_bonus = 500.0
                elif threat_value >= 500:  # Rook threat
                    threat_bonus = 400.0
                elif threat_value >= 300:  # Minor piece threat
                    threat_bonus = 300.0
                else:
                    threat_bonus = 200.0
                threats.append((threat_bonus + tactical_score['base_score'], move))
                continue
            
            # Get piece for subsequent checks
            piece = board.piece_at(move.from_square)
            
            # 7. Killer moves (after tactical categories)
            if move in killer_set:
                killers.append(move)
                self.search_stats['killer_hits'] += 1
                continue
            
            # 8. Opening central pawn moves (V14.3 improvement - maintained)
            if piece and piece.piece_type == chess.PAWN and self._is_opening_position(board):
                central_pawn_moves = ['e2e4', 'd2d4', 'e7e5', 'd7d5']
                if move.uci() in central_pawn_moves:
                    pawn_advances.append((80.0, move))  # High priority for central pawns
                    continue
            
            # 9. Development and patterns
            if piece:
                # Development moves (knights, bishops moving from starting squares)
                # V14.3: Penalize knight moves to edge in opening
                if (piece.piece_type == chess.KNIGHT and self._is_opening_position(board) and 
                    move.from_square in [chess.B1, chess.G1, chess.B8, chess.G8] and
                    move.to_square in [chess.A3, chess.H3]):
                    development.append((10.0, move))  # Low priority for edge knights
                    continue
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    starting_squares = {
                        chess.KNIGHT: [chess.B1, chess.G1, chess.B8, chess.G8],
                        chess.BISHOP: [chess.C1, chess.F1, chess.C8, chess.F8]
                    }
                    if move.from_square in starting_squares.get(piece.piece_type, []):
                        development.append((50.0, move))
                        continue
                
                # Pawn advances
                if piece.piece_type == chess.PAWN:
                    pawn_advances.append((10.0, move))
                    continue
            
            # 10. Remaining moves (with enhanced tactical scoring)
            history_score = self.history_heuristic.get_history_score(move)
            
            if tactical_score['base_score'] > 20.0:  # Significant tactical move
                tactical_moves.append((tactical_score['base_score'] + history_score, move))
            else:
                quiet_moves.append((history_score, move))
        
        # Sort all move categories by their scores (V14.4: Enhanced categories)
        mate_threats.sort(key=lambda x: x[0], reverse=True)
        checks.sort(key=lambda x: x[0], reverse=True)
        high_value_captures.sort(key=lambda x: x[0], reverse=True)
        captures.sort(key=lambda x: x[0], reverse=True)
        multi_attacks.sort(key=lambda x: x[0], reverse=True)
        threats.sort(key=lambda x: x[0], reverse=True)
        development.sort(key=lambda x: x[0], reverse=True)
        pawn_advances.sort(key=lambda x: x[0], reverse=True)
        tactical_moves.sort(key=lambda x: x[0], reverse=True)
        quiet_moves.sort(key=lambda x: x[0], reverse=True)
        
        # V14.4 ENHANCED TACTICAL ORDER: Multi-attacks prioritized before captures
        ordered = []
        ordered.extend(tt_moves)                                    # 1. TT move (highest priority)
        ordered.extend([move for _, move in mate_threats])         # 2. Mate threats
        ordered.extend([move for _, move in checks])               # 3. Checks (enhanced scoring)
        
        # 4. HIGH-VALUE THREATS (Queen threats treated like checks)
        high_value_threats = [move for score, move in threats if score >= 650]  # Queen threats (500 bonus + 165 base)
        ordered.extend(high_value_threats)
        
        ordered.extend([move for _, move in multi_attacks])        # 5. Multi-piece attacks (now BEFORE captures)
        ordered.extend([move for _, move in high_value_captures])  # 6. High-value captures (Q/R)
        ordered.extend([move for _, move in captures])             # 7. Other captures
        
        # 8. Other threats (after captures)
        other_threats = [move for score, move in threats if score < 650]
        ordered.extend(other_threats)
        
        ordered.extend(killers)                                    # 9. Killer moves
        ordered.extend([move for _, move in development])          # 10. Development
        ordered.extend([move for _, move in pawn_advances])        # 11. Pawn advances
        ordered.extend([move for _, move in tactical_moves])       # 12. Other tactical moves
        ordered.extend([move for _, move in quiet_moves])          # 13. Quiet moves
        
        # V14.5 FIX: DISABLE safety prioritization that was breaking tactical move ordering
        # The _apply_safety_prioritization was completely reordering moves, causing the engine
        # to avoid tactically correct moves because they appeared "unsafe"
        # TODO: Reimplement safety checks more carefully without destroying tactical order
        # ordered = self._apply_safety_prioritization(board, ordered)  # DISABLED IN V14.5
        
        return ordered

    def _apply_safety_prioritization(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """
        V14.4: Apply integrated blunder prevention to move ordering
        Converts BlunderProofFirewall concepts to bitboard-based safety analysis
        
        Prioritizes moves based on:
        1. Safety (no king/queen blunders)
        2. Control improvement
        3. Threat handling
        """
        if not moves:
            return moves
        
        # Evaluate safety for each move using bitboard operations
        move_safety_scores = []
        
        for move in moves:
            try:
                # Use bitboard-based safety evaluation
                safety_analysis = self.bitboard_evaluator.evaluate_move_safety_bitboard(board, move)
                
                safety_score = safety_analysis.get('safety_score', 0.0)
                is_safe = safety_analysis.get('is_safe', True)
                
                # Major penalty for unsafe moves (blunder prevention)
                if not is_safe:
                    safety_score -= 500.0  # Strong penalty for dangerous moves
                
                move_safety_scores.append((safety_score, move))
                
            except Exception as e:
                # Default to neutral safety score on error
                move_safety_scores.append((0.0, move))
        
        # Sort by safety score (descending - higher scores first)
        move_safety_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return moves in safety-prioritized order
        return [move for _, move in move_safety_scores]
    
    def _analyze_position_for_tactics(self, board: chess.Board) -> Dict:
        """
        V14.4: Analyze current position for tactical patterns
        Pre-calculates tactical information to improve move ordering efficiency
        """
        analysis = {
            'attacked_squares': set(),
            'defending_squares': set(),
            'valuable_pieces': [],  # Pieces worth >= 300cp
            'loose_pieces': [],     # Undefended pieces
            'pinned_pieces': [],
            'material_imbalance': 0,
            'piece_activity': 0
        }
        
        current_color = board.turn
        opponent_color = not current_color
        
        # Find valuable pieces for both sides
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_value = self._get_dynamic_piece_value(board, piece.piece_type, piece.color)
                if piece_value >= 300:  # Minor pieces and above
                    analysis['valuable_pieces'].append({
                        'square': square,
                        'piece': piece,
                        'value': piece_value,
                        'color': piece.color
                    })
        
        # Calculate attacked and defended squares
        for square in chess.SQUARES:
            if board.is_attacked_by(current_color, square):
                analysis['attacked_squares'].add(square)
            if board.is_attacked_by(opponent_color, square):
                analysis['defending_squares'].add(square)
        
        # Find loose pieces (not defended)
        for piece_info in analysis['valuable_pieces']:
            square = piece_info['square']
            if not board.is_attacked_by(piece_info['color'], square):
                analysis['loose_pieces'].append(piece_info)
        
        # Calculate material imbalance
        white_material = sum(self._get_dynamic_piece_value(board, p.piece_type, True) 
                           for p in board.piece_map().values() if p.color == chess.WHITE)
        black_material = sum(self._get_dynamic_piece_value(board, p.piece_type, False) 
                           for p in board.piece_map().values() if p.color == chess.BLACK)
        
        if current_color == chess.WHITE:
            analysis['material_imbalance'] = white_material - black_material
        else:
            analysis['material_imbalance'] = black_material - white_material
        
        return analysis
    
    def _calculate_tactical_move_score(self, board: chess.Board, move: chess.Move, analysis: Dict, cached_original_pins: Optional[Dict] = None) -> Dict:
        """
        V14.4: Calculate tactical score for a specific move
        Returns dictionary with tactical flags and base score
        """
        score_info = {
            'base_score': 0.0,
            'mate_threat': False,
            'discovered_check': False,
            'double_check': False,
            'safe_capture': False,
            'attacks_multiple': False,
            'creates_pin': False,
            'creates_threat': False,
            'threat_value': 0
        }
        
        # Create a copy to test the move
        test_board = board.copy()
        
        try:
            # Check if move is legal
            if move not in test_board.legal_moves:
                return score_info
            
            test_board.push(move)
            
            # 1. Mate threat detection
            if test_board.is_checkmate():
                score_info['mate_threat'] = True
                score_info['base_score'] += 1000
                test_board.pop()
                return score_info
            
            # 2. Check types
            if test_board.is_check():
                # Count attacking pieces to detect double check
                king_square = test_board.king(not board.turn)
                if king_square is not None:
                    attackers = list(test_board.attackers(board.turn, king_square))
                    
                    if len(attackers) > 1:
                        score_info['double_check'] = True
                        score_info['base_score'] += 100
                    
                    # Check for discovered check
                    piece = board.piece_at(move.from_square)
                    if piece and move.from_square not in attackers:
                        score_info['discovered_check'] = True
                        score_info['base_score'] += 50
            
            # 3. Safe capture analysis
            if board.is_capture(move):
                captured_square = move.to_square
                if not test_board.is_attacked_by(not board.turn, captured_square):
                    score_info['safe_capture'] = True
                    score_info['base_score'] += 25
            
            # 4. Multiple piece attacks (fork potential) - V14.4 Phase 2 ENHANCED
            piece = board.piece_at(move.from_square)
            if piece:
                # Count valuable enemy pieces attacked by THIS SPECIFIC PIECE after the move
                attacked_valuable = 0
                total_threat_value = 0
                
                # Get squares attacked by the moved piece in its new position
                piece_attacks = test_board.attacks(move.to_square)
                
                for piece_info in analysis['valuable_pieces']:
                    if piece_info['color'] != board.turn:  # Enemy piece
                        # Check if THIS moved piece attacks this valuable enemy piece
                        if piece_info['square'] in piece_attacks:
                            attacked_valuable += 1
                            total_threat_value += piece_info['value']
                
                if attacked_valuable >= 2:
                    score_info['attacks_multiple'] = True
                    # Higher bonus for forking multiple pieces with a single move
                    score_info['base_score'] += min(attacked_valuable * 60, 200)  # Increased bonus
                elif attacked_valuable == 1 and total_threat_value >= 300:
                    score_info['creates_threat'] = True
                    score_info['threat_value'] = total_threat_value
                    # Enhanced bonus for threatening high-value pieces (V14.4 Phase 2)
                    # Queen threat should be prioritized over multi-piece attacks
                    if total_threat_value >= 900:  # Queen threat
                        score_info['base_score'] += 150  # Higher than multi-attack bonus
                    elif total_threat_value >= 500:  # Rook threat
                        score_info['base_score'] += 80
                    else:  # Other high-value pieces
                        score_info['base_score'] += min(total_threat_value / 10, 50)
            
            # 5. Enhanced pin creation detection (V14.4 Phase 2)
            # Check if the move creates new pins using cached detection for performance
            try:
                # Use cached original pins to avoid expensive recalculation
                if cached_original_pins is not None:
                    original_pins = cached_original_pins
                else:
                    original_pins = self.bitboard_evaluator.detect_pins_bitboard(board)
                
                new_pins = self.bitboard_evaluator.detect_pins_bitboard(test_board)
                
                # Check if we created new pins for our color
                if board.turn == chess.WHITE:
                    original_pin_count = len(original_pins.get('white_pins', []))
                    new_pin_count = len(new_pins.get('white_pins', []))
                else:
                    original_pin_count = len(original_pins.get('black_pins', []))
                    new_pin_count = len(new_pins.get('black_pins', []))
                
                if new_pin_count > original_pin_count:
                    score_info['creates_pin'] = True
                    # Higher bonus for creating pins
                    pins_created = new_pin_count - original_pin_count
                    score_info['base_score'] += pins_created * 50  # Increased from 40
            except:
                # Fallback to simplified pin detection if enhanced fails
                for piece_info in analysis['valuable_pieces']:
                    if piece_info['color'] != board.turn:  # Enemy piece
                        enemy_square = piece_info['square']
                        enemy_king = test_board.king(not board.turn)
                        
                        if enemy_king and enemy_square != enemy_king:
                            if (abs(chess.square_file(enemy_square) - chess.square_file(enemy_king)) <= 1 or
                                abs(chess.square_rank(enemy_square) - chess.square_rank(enemy_king)) <= 1):
                                score_info['creates_pin'] = True
                                score_info['base_score'] += 40
                                break
            
            test_board.pop()
            
        except:
            # If any error occurs, return default score
            pass
        
        # Add baseline tactical bonus from bitboard evaluator
        try:
            bitboard_bonus = self.bitboard_evaluator.detect_bitboard_tactics(board, move)
            score_info['base_score'] += bitboard_bonus
        except:
            pass
        
        return score_info
    
    def _detect_pins(self, board: chess.Board) -> Dict:
        """
        V14.4 Phase 2: Enhanced pin detection using BITBOARD operations
        Much faster than square-by-square traversal - leverages chess library's efficient methods
        """
        pin_data = {
            'white_pins': [],
            'black_pins': [],
            'white_pinned': [],
            'black_pinned': [],
            'pin_score_white': 0.0,
            'pin_score_black': 0.0
        }
        
        try:
            # Use efficient pin detection for both colors
            for color in [chess.WHITE, chess.BLACK]:
                opponent_color = not color
                enemy_king_square = board.king(opponent_color)
                
                if enemy_king_square is None:
                    continue
                
                pin_score = 0.0
                
                # Get our sliding pieces (potential pinning pieces)
                our_queens = board.pieces(chess.QUEEN, color)
                our_rooks = board.pieces(chess.ROOK, color) 
                our_bishops = board.pieces(chess.BISHOP, color)
                sliding_pieces = our_queens | our_rooks | our_bishops
                
                for piece_square in sliding_pieces:
                    piece = board.piece_at(piece_square)
                    if not piece:
                        continue
                    
                    # Get squares between our piece and enemy king using bitboard
                    between_mask = chess.between(piece_square, enemy_king_square)
                    between_squares = list(chess.scan_forward(between_mask))
                    
                    # Skip if no squares between (adjacent) or not on same line
                    if not between_squares:
                        continue
                    
                    # Check if piece can potentially attack king (line type matching)
                    can_attack_king = self._can_piece_attack_line(piece.piece_type, piece_square, enemy_king_square)
                    if not can_attack_king:
                        continue
                    
                    # Count enemy pieces in between
                    enemy_pieces_between = []
                    for sq in between_squares:
                        piece_at_sq = board.piece_at(sq)
                        if piece_at_sq and piece_at_sq.color == opponent_color:
                            enemy_pieces_between.append((sq, piece_at_sq))
                    
                    # Pin exists if exactly one enemy piece between
                    if len(enemy_pieces_between) == 1:
                        pinned_square, pinned_piece = enemy_pieces_between[0]
                        
                        pin_value = self._calculate_pin_value(pinned_piece.piece_type)
                        pin_score += pin_value
                        
                        pin_info = {
                            'pinning_square': piece_square,
                            'pinning_piece': piece.piece_type,
                            'pinned_square': pinned_square,
                            'pinned_piece': pinned_piece.piece_type,
                            'king_square': enemy_king_square,
                            'value': pin_value
                        }
                        
                        if color == chess.WHITE:
                            pin_data['white_pins'].append(pin_info)
                            pin_data['black_pinned'].append(pinned_square)
                        else:
                            pin_data['black_pins'].append(pin_info)
                            pin_data['white_pinned'].append(pinned_square)
                
                # Store totals
                if color == chess.WHITE:
                    pin_data['pin_score_white'] = pin_score
                else:
                    pin_data['pin_score_black'] = pin_score
        
        except Exception:
            # Fallback to empty pin data if detection fails
            pass
        
        return pin_data
    
    def _can_piece_attack_line(self, piece_type: chess.PieceType, from_square: chess.Square, to_square: chess.Square) -> bool:
        """Check if piece type can potentially attack along the line between squares"""
        try:
            # Use chess library's efficient square calculations
            from_file, from_rank = chess.square_file(from_square), chess.square_rank(from_square)
            to_file, to_rank = chess.square_file(to_square), chess.square_rank(to_square)
            
            file_diff = to_file - from_file
            rank_diff = to_rank - from_rank
            
            # Check if it's a valid line for the piece type
            if piece_type == chess.QUEEN:
                # Queen moves in any straight line
                return file_diff == 0 or rank_diff == 0 or abs(file_diff) == abs(rank_diff)
            elif piece_type == chess.ROOK:
                # Rook moves horizontally or vertically
                return file_diff == 0 or rank_diff == 0
            elif piece_type == chess.BISHOP:
                # Bishop moves diagonally
                return abs(file_diff) == abs(rank_diff) and file_diff != 0
            
        except:
            pass
        
        return False
    
    def _calculate_pin_value(self, piece_type: chess.PieceType) -> float:
        """Calculate the tactical value of pinning a piece"""
        pin_values = {
            chess.PAWN: 15.0,
            chess.KNIGHT: 40.0,
            chess.BISHOP: 40.0,
            chess.ROOK: 60.0,
            chess.QUEEN: 80.0,
            chess.KING: 0.0  # King can't really be pinned effectively
        }
        return pin_values.get(piece_type, 0.0)

    def _evaluate_position(self, board: chess.Board) -> float:
        """V14.4 UNIFIED: Use bitboard evaluator for ALL evaluation logic"""
        # V14.4: All evaluation logic moved to bitboard evaluator for consistency and performance
        return self.bitboard_evaluator.evaluate_position_complete(board, self.evaluation_cache)
    
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
        
        # Simple replacement strategy for performance
        if len(self.transposition_table) >= self.max_tt_entries:
            # Clear 25% of entries when full (simple aging)
            entries = list(self.transposition_table.items())
            entries.sort(key=lambda x: x[1].depth, reverse=True)
            self.transposition_table = dict(entries[:int(self.max_tt_entries * 0.75)])
        
        entry = TranspositionEntry(depth, score, best_move, node_type, zobrist_hash)
        self.transposition_table[zobrist_hash] = entry
        self.search_stats['tt_stores'] += 1
    
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
        V14.3 EMERGENCY FIXED: Quiescence search with MANDATORY time checking
        Only search captures and checks to find quiet positions
        """
        self.nodes_searched += 1
        
        # V14.3 CRITICAL: Emergency time checking in quiescence to prevent infinite loops
        if hasattr(self, 'search_start_time') and hasattr(self, 'emergency_stop_flag'):
            if self.emergency_stop_flag:
                return self._evaluate_position(board)
            
            # Check time every node in quiescence (it's cheaper than getting stuck)
            elapsed = time.time() - self.search_start_time
            if elapsed > getattr(self, 'current_time_limit', 1.0) * 0.7:
                self.emergency_stop_flag = True
                return self._evaluate_position(board)
        
        # Stand pat evaluation
        stand_pat = self._evaluate_position(board)
        
        # Beta cutoff on stand pat
        if stand_pat >= beta:
            return beta
        
        # Update alpha if stand pat is better
        if stand_pat > alpha:
            alpha = stand_pat
        
        # V14.3: SIMPLIFIED depth limits - NO game phase detection per node
        max_quiescence_depth = 4  # Fixed depth limit for all phases
        
        # Depth limit reached
        if depth <= -max_quiescence_depth:
            return stand_pat
        
        # Generate tactical moves only (captures and checks)
        legal_moves = list(board.legal_moves)
        tactical_moves = []
        
        for move in legal_moves:
            # V14.3: SIMPLIFIED - only captures and checks, no complex endgame logic
            is_capture = board.is_capture(move)
            is_check = board.gives_check(move)
            
            # Always include captures and checks
            if is_capture or is_check:
                tactical_moves.append(move)
            # Also include promotions (always tactical)
            elif move.promotion:
                tactical_moves.append(move)
        
        # If no tactical moves, return stand pat (position is quiet)
        if not tactical_moves:
            return stand_pat
        
        # V14.3: SIMPLIFIED tactical move ordering - no caching, no complex scoring
        capture_scores = []
        for move in tactical_moves:
            score = 0
            
            if board.is_capture(move):
                # Simple MVV-LVA without dynamic caching
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    victim_value = self.piece_values.get(victim.piece_type, 0)
                    attacker_value = self.piece_values.get(attacker.piece_type, 0)
                    score = victim_value * 100 - attacker_value
                    
                    # V14.3: Enhanced trade evaluation - prefer equal trades and simplification
                    value_diff = abs(victim_value - attacker_value)
                    if value_diff <= 30:  # Equal trades (within 30 cp)
                        score += 80  # Strong bonus for equal trades
                    elif victim_value >= attacker_value:
                        score += 50  # Good trades get medium bonus
                    else:
                        score -= 20  # Penalty for bad trades (losing material)
            
            elif board.gives_check(move):
                # Fixed check priority
                score = 25
            
            elif move.promotion:
                # Promotions are valuable
                promoted_value = self.piece_values.get(move.promotion, 0)
                score = promoted_value + 100
            
            capture_scores.append((score, move))
        
        # Sort by score (higher is better)
        capture_scores.sort(key=lambda x: x[0], reverse=True)
        ordered_tactical = [move for _, move in capture_scores]
        
        # V14.3: SIMPLIFIED delta pruning
        delta_threshold = 200  # Fixed threshold for all phases
        
        # Search tactical moves
        best_score = stand_pat
        moves_searched = 0
        
        for move in ordered_tactical:
            # V14.3: EMERGENCY time check during quiescence search
            if moves_searched > 0 and moves_searched % 5 == 0:  # Check every 5 moves
                if hasattr(self, 'emergency_stop_flag') and self.emergency_stop_flag:
                    break
            
            # Simple delta pruning for captures
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                if victim:
                    victim_value = self.piece_values.get(victim.piece_type, 0)
                    # Skip captures that can't improve alpha enough
                    if stand_pat + victim_value + delta_threshold < alpha:
                        continue
            
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha, depth - 1)
            board.pop()
            
            moves_searched += 1
            
            if score > best_score:
                best_score = score
            
            if score > alpha:
                alpha = score
            
            if alpha >= beta:
                break  # Beta cutoff
            
            # V14.2: Limit search width in deep quiescence for performance
            if depth < -3 and moves_searched >= 8:
                break  # Stop after 8 moves in very deep quiescence
        
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
        """Reset for a new game - V14.2: Enhanced with cache clearing"""
        self.evaluation_cache.clear()
        self.transposition_table.clear()
        self.killer_moves = KillerMoves()
        self.history_heuristic = HistoryHeuristic()
        self.nodes_searched = 0
        
        # V14.2: Clear performance optimization caches
        self.bishop_value_cache.clear()
        self.game_phase_cache.clear()
        self.search_depth_achieved.clear()
        
        # Clear PV following data
        self.pv_tracker.clear()
        
        # Reset stats
        for key in self.search_stats:
            if key in ['nodes_per_second', 'cache_hits', 'cache_misses', 'tt_hits', 'tt_stores', 'killer_hits', 'game_phase_switches']:
                self.search_stats[key] = 0
            elif key == 'average_depth_achieved':
                self.search_stats[key] = 0.0

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
        V11 ENHANCEMENT: Adaptive time management based on position complexity
        
        Returns: (target_time, max_time)
        """
        moves_played = len(board.move_stack)
        legal_moves = list(board.legal_moves)
        num_legal_moves = len(legal_moves)
        
        # Base time factor
        time_factor = 1.0
        
        # Game phase adjustment
        if moves_played < 15:  # Opening
            time_factor *= 0.8  # Faster in opening
        elif moves_played < 40:  # Middle game
            time_factor *= 1.2  # More time in complex middle game
        else:  # Endgame
            time_factor *= 0.9  # Moderate time in endgame
        
        # Position complexity factors
        if board.is_check():
            time_factor *= 1.3  # More time when in check
        
        if num_legal_moves <= 5:
            time_factor *= 0.7  # Less time with few options
        elif num_legal_moves >= 35:
            time_factor *= 1.4  # More time with many options
        
        # Material balance consideration
        our_material = self._count_material(board, board.turn)
        their_material = self._count_material(board, not board.turn)
        material_diff = our_material - their_material
        
        if material_diff < -300:  # We're behind
            time_factor *= 1.2  # Take more time when behind
        elif material_diff > 300:  # We're ahead
            time_factor *= 0.9  # Play faster when ahead
        
        # Calculate final times
        target_time = min(base_time_limit * time_factor * 0.8, base_time_limit * 0.9)
        max_time = min(base_time_limit * time_factor, base_time_limit)
        
        return target_time, max_time
    
    def _calculate_emergency_time_allocation(self, base_time_limit: float, moves_played: int = 0) -> Tuple[float, float]:
        """
        V14.5: RELAXED time controls - v14.4 was too aggressive causing shallow searches
        
        Returns: (target_time, max_time) - Balanced allocations for proper depth achievement
        """
        # V14.5: Allow use of most of the time for better play
        if base_time_limit <= 1.0:
            # Time trouble - still be conservative
            return base_time_limit * 0.6, base_time_limit * 0.85
        elif base_time_limit <= 3.0:
            # Limited time - use most of it
            return base_time_limit * 0.7, base_time_limit * 0.9
        elif base_time_limit <= 10.0:
            # Standard time - allow good depth
            return base_time_limit * 0.75, base_time_limit * 0.92
        else:
            # Plenty of time - use it well
            return base_time_limit * 0.8, base_time_limit * 0.95
    
    def _calculate_minimum_depth(self, time_limit: float) -> int:
        """
        V14.3: Calculate guaranteed minimum depth based on available time
        
        CRITICAL: Always achieve this depth regardless of complexity
        """
        if time_limit >= 5.0:
            return 4  # Always get at least 4-ply with decent time
        elif time_limit >= 2.0:
            return 3  # Emergency minimum for limited time
        elif time_limit >= 0.5:
            return 2  # Last resort for time trouble
        else:
            return 1  # Absolute minimum to avoid timeout
    
    def _detect_game_phase_conservative(self, board: chess.Board) -> str:
        """
        V14.3: CONSERVATIVE game phase detection to prevent misclassification
        
        Uses stricter thresholds and defaults to middlegame when uncertain
        """
        # Use cached value if available
        material_hash = self._calculate_material_hash(board)
        if material_hash in self.game_phase_cache:
            return self.game_phase_cache[material_hash]
        
        # Conservative phase detection
        moves_played = len(board.move_stack)
        total_material = 0
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                piece_count = len(board.pieces(piece_type, color))
                total_material += piece_count * self.piece_values[piece_type]
        
        # CONSERVATIVE thresholds - stricter than V14.2
        if moves_played < 6 and total_material >= 5500:  # Very early opening with full material
            phase = 'opening'
        elif total_material <= 2000:  # Clear endgame threshold
            phase = 'endgame'
        else:
            phase = 'middlegame'  # DEFAULT to middlegame when uncertain
        
        # Cache for future use
        self.game_phase_cache[material_hash] = phase
        return phase
    
    def _calculate_advanced_time_allocation(self, board: chess.Board, base_time_limit: float) -> Tuple[float, float]:
        """
        V14.3: SIMPLIFIED time management - removed complex calculations that caused flagging
        
        Returns: (target_time, max_time) - Uses emergency allocation with minimal overhead
        """
        # V14.3 CRITICAL FIX: Use emergency time allocation instead of complex calculations
        return self._calculate_emergency_time_allocation(base_time_limit, len(board.move_stack))
    
    def _calculate_target_depth(self, game_phase: str, time_limit: float) -> int:
        """
        V14.2: Calculate target search depth based on game phase and available time
        
        Goal: Achieve consistent 10-ply depth when possible
        """
        base_depths = {
            'opening': 8,     # Standard opening depth
            'middlegame': 10, # Target 10-ply for complex positions
            'endgame': 12     # Can search deeper in endgame with fewer pieces
        }
        
        base_depth = base_depths.get(game_phase, 10)
        
        # Adjust based on available time
        if time_limit >= 10.0:  # Plenty of time
            return min(base_depth + 2, 15)
        elif time_limit >= 5.0:  # Standard time
            return base_depth
        elif time_limit >= 2.0:  # Limited time
            return max(base_depth - 1, 6)
        else:  # Very limited time
            return max(base_depth - 2, 5)
    
    def _count_material(self, board: chess.Board, color: bool) -> int:
        """Count total material for a color (V14.1: Dynamic piece values)"""
        total = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            pieces = board.pieces(piece_type, color)
            piece_value = self._get_dynamic_piece_value(board, piece_type, color)
            total += len(pieces) * piece_value
        return total

    def _get_dynamic_piece_value(self, board: chess.Board, piece_type: int, color: bool) -> int:
        """V14.2: Cached dynamic piece valuation for performance"""
        if piece_type != chess.BISHOP:
            return self.piece_values[piece_type]
        
        # Use cached value if available
        board_hash = hash(str(board.pieces(chess.BISHOP, chess.WHITE)) + str(board.pieces(chess.BISHOP, chess.BLACK)))
        cache_key = (board_hash, color)
        
        if cache_key in self.bishop_value_cache:
            return self.bishop_value_cache[cache_key]
        
        # Calculate dynamic bishop value
        bishops = board.pieces(chess.BISHOP, color)
        bishop_count = len(bishops)
        
        if bishop_count >= 2:
            # Bishop pair bonus: 325 each (two bishops > two knights)
            value = 325
        elif bishop_count == 1:
            # Single bishop penalty: 275 (one bishop < one knight)
            value = 275
        else:
            # No bishops remaining
            value = self.piece_values[piece_type]
        
        # Cache for future use
        self.bishop_value_cache[cache_key] = value
        return value

    def _detect_game_phase(self, board: chess.Board) -> str:
        """V14.2: NEW - Detect game phase for dynamic evaluation
        
        Returns: 'opening', 'middlegame', or 'endgame'
        """
        # Use cached value if available
        material_hash = self._calculate_material_hash(board)
        if material_hash in self.game_phase_cache:
            return self.game_phase_cache[material_hash]
        
        # Count total material (excluding pawns and kings)
        total_material = 0
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                piece_count = len(board.pieces(piece_type, color))
                total_material += piece_count * self.piece_values[piece_type]
        
        # Phase detection based on material and move count
        moves_played = len(board.move_stack)
        
        if moves_played < 8 and total_material >= 5000:  # Early opening with most pieces
            phase = 'opening'
        elif total_material <= 2500:  # Low material remaining
            phase = 'endgame'
        else:
            phase = 'middlegame'
        
        # Cache for future use
        self.game_phase_cache[material_hash] = phase
        return phase

    def _calculate_material_hash(self, board: chess.Board) -> int:
        """Calculate a hash based on material for caching game phases"""
        material_vector = []
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
                material_vector.append(len(board.pieces(piece_type, color)))
        return hash(tuple(material_vector))

    def _is_opening_position(self, board: chess.Board) -> bool:
        """V14.3: Detect if we're still in the opening phase"""
        # Simple heuristic: opening if most pieces are still on starting squares
        piece_count = len(board.piece_map())
        return piece_count >= 28  # Most pieces still on board
    
    def get_performance_report(self) -> str:
        """V14.2: Generate performance report for analysis"""
        avg_depth = self.search_stats.get('average_depth_achieved', 0)
        total_moves = len(self.search_depth_achieved)
        
        depth_distribution = {}
        for depth in self.search_depth_achieved.values():
            depth_distribution[depth] = depth_distribution.get(depth, 0) + 1
        
        cache_hits = self.search_stats.get('cache_hits', 0)
        cache_misses = self.search_stats.get('cache_misses', 0)
        cache_hit_rate = cache_hits / max(cache_hits + cache_misses, 1)
        
        tt_hits = self.search_stats.get('tt_hits', 0)
        tt_hit_rate = tt_hits / max(self.nodes_searched, 1)
        
        report = f"""
V14.2 Performance Report:
========================
Average Search Depth: {avg_depth:.1f} ply
Total Moves Analyzed: {total_moves}
Nodes Per Second: {self.search_stats.get('nodes_per_second', 0)}
Cache Hit Rate: {cache_hit_rate:.1%}
TT Hit Rate: {tt_hit_rate:.1%}

Depth Distribution:
{chr(10).join(f"  {depth} ply: {count} moves" for depth, count in sorted(depth_distribution.items()))}

Game Phase Switches: {self.search_stats.get('game_phase_switches', 0)}
Bishop Value Cache Size: {len(self.bishop_value_cache)}
Game Phase Cache Size: {len(self.game_phase_cache)}
"""
        return report

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
